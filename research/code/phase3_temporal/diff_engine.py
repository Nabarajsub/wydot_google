"""
Phase 3: Temporal Diff Engine (gamma dimension)
=================================================
Generates section-level diffs between document editions and stores
them as CHANGE nodes in Neo4j for temporal retrieval.

This is the TEMPORAL dimension (gamma) of SpecRAG.

Pipeline:
1. Load PageIndex trees for both editions
2. Align sections between old and new edition (exact + fuzzy matching)
3. Extract text from aligned section pairs
4. Generate structured diffs using Gemini
5. Store diffs as CHANGE nodes with temporal edges in Neo4j

Usage:
    python -m phase3_temporal.diff_engine                     # All temporal chains
    python -m phase3_temporal.diff_engine --chain standard_specs
    python -m phase3_temporal.diff_engine --pair "2010 Standard Specifications.pdf" "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"
"""
import json
import sys
import logging
import re
from pathlib import Path
from typing import Optional
from difflib import SequenceMatcher

import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_DIR, PAGEINDEX_DIR, TEMPORAL_DIR, CORPUS, TEMPORAL_CHAINS,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
)
from utils.gemini_client import gemini_generate_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Section Alignment ────────────────────────────────────

def load_tree(filename: str) -> dict:
    """Load a PageIndex tree by original PDF filename."""
    safe_name = filename.replace(" ", "_").replace("–", "-").replace(".pdf", "")
    tree_path = PAGEINDEX_DIR / f"{safe_name}_tree.json"
    if not tree_path.exists():
        # Try finding by glob
        candidates = list(PAGEINDEX_DIR.glob(f"*{safe_name[:30]}*_tree.json"))
        if candidates:
            tree_path = candidates[0]
        else:
            logger.error(f"Tree not found for: {filename}")
            return {}
    with open(tree_path) as f:
        return json.load(f)


def flatten_tree(tree: dict, path: str = "") -> list[dict]:
    """
    Flatten a PageIndex tree into a list of sections with paths.
    Returns [{title, page_start, page_end, path, level}, ...]
    """
    result = []
    title = tree.get("title", "")
    current_path = f"{path} > {title}" if path else title

    result.append({
        "title": title,
        "page_start": tree.get("page_start", 1),
        "page_end": tree.get("page_end", 1),
        "path": current_path,
        "level": current_path.count(">"),
    })

    for child in tree.get("children", []):
        result.extend(flatten_tree(child, current_path))

    return result


def normalize_section_title(title: str) -> str:
    """Normalize section title for matching."""
    title = title.lower().strip()
    title = re.sub(r'\s+', ' ', title)
    title = re.sub(r'[-–—]', '-', title)
    # Extract section numbers if present
    num_match = re.search(r'(\d{3}(?:\.\d{2})?)', title)
    if num_match:
        return num_match.group(1)
    return title


def align_sections(old_sections: list[dict], new_sections: list[dict],
                   similarity_threshold: float = 0.6) -> list[dict]:
    """
    Align sections between old and new editions.
    Uses exact match on section numbers first, then fuzzy title matching.

    Returns list of {old_section, new_section, match_type, similarity}
    """
    alignments = []
    matched_new = set()
    matched_old = set()

    # Pass 1: Exact match on section numbers
    old_nums = {}
    new_nums = {}

    for i, sec in enumerate(old_sections):
        num = normalize_section_title(sec["title"])
        if re.match(r'\d{3}', num):
            old_nums.setdefault(num, []).append(i)

    for i, sec in enumerate(new_sections):
        num = normalize_section_title(sec["title"])
        if re.match(r'\d{3}', num):
            new_nums.setdefault(num, []).append(i)

    for num in old_nums:
        if num in new_nums:
            old_idx = old_nums[num][0]
            new_idx = new_nums[num][0]
            alignments.append({
                "old_section": old_sections[old_idx],
                "new_section": new_sections[new_idx],
                "match_type": "exact_number",
                "similarity": 1.0,
            })
            matched_old.add(old_idx)
            matched_new.add(new_idx)

    # Pass 2: Fuzzy title matching for unmatched sections
    for i, old_sec in enumerate(old_sections):
        if i in matched_old:
            continue

        best_match = None
        best_score = 0

        for j, new_sec in enumerate(new_sections):
            if j in matched_new:
                continue

            score = SequenceMatcher(None,
                                   old_sec["title"].lower(),
                                   new_sec["title"].lower()).ratio()
            if score > best_score and score >= similarity_threshold:
                best_score = score
                best_match = j

        if best_match is not None:
            alignments.append({
                "old_section": old_sec,
                "new_section": new_sections[best_match],
                "match_type": "fuzzy_title",
                "similarity": best_score,
            })
            matched_old.add(i)
            matched_new.add(best_match)

    # Pass 3: Identify added sections (in new but not matched)
    for j, new_sec in enumerate(new_sections):
        if j not in matched_new and new_sec["level"] <= 2:
            alignments.append({
                "old_section": None,
                "new_section": new_sec,
                "match_type": "added",
                "similarity": 0.0,
            })

    # Pass 4: Identify removed sections (in old but not matched)
    for i, old_sec in enumerate(old_sections):
        if i not in matched_old and old_sec["level"] <= 2:
            alignments.append({
                "old_section": old_sec,
                "new_section": None,
                "match_type": "removed",
                "similarity": 0.0,
            })

    logger.info(f"  Alignment: {len(alignments)} pairs "
                f"(exact: {sum(1 for a in alignments if a['match_type']=='exact_number')}, "
                f"fuzzy: {sum(1 for a in alignments if a['match_type']=='fuzzy_title')}, "
                f"added: {sum(1 for a in alignments if a['match_type']=='added')}, "
                f"removed: {sum(1 for a in alignments if a['match_type']=='removed')})")

    return alignments


# ─── Text Extraction for Diff ─────────────────────────────

def extract_section_text(pdf_path: Path, page_start: int, page_end: int,
                         max_chars: int = 5000) -> str:
    """Extract text from a section's page range."""
    doc = fitz.open(str(pdf_path))
    texts = []

    start = max(0, page_start - 1)
    end = min(len(doc), page_end)

    for p in range(start, end):
        texts.append(doc[p].get_text("text"))

    doc.close()
    full = "\n".join(texts)
    return full[:max_chars]


# ─── LLM-Powered Diff Generation ─────────────────────────

def generate_diff(old_text: str, new_text: str, section_title: str,
                  old_year: int, new_year: int) -> dict:
    """
    Use Gemini to generate a structured diff between two section versions.

    Returns:
    {
        "change_type": "modified" | "added" | "removed" | "unchanged",
        "severity": "critical" | "major" | "minor" | "none",
        "summary": "Brief description of changes",
        "details": ["Specific change 1", "Specific change 2", ...],
        "added_content": ["New items added"],
        "removed_content": ["Items removed"],
        "modified_content": [{"before": ..., "after": ..., "significance": ...}],
        "keywords": ["relevant", "keywords"],
    }
    """
    if not old_text and new_text:
        return {
            "change_type": "added",
            "severity": "major",
            "summary": f"New section added in {new_year} edition",
            "details": [f"Section '{section_title}' is entirely new"],
            "added_content": [new_text[:500]],
            "removed_content": [],
            "modified_content": [],
            "keywords": [],
        }

    if old_text and not new_text:
        return {
            "change_type": "removed",
            "severity": "major",
            "summary": f"Section removed in {new_year} edition",
            "details": [f"Section '{section_title}' was removed"],
            "added_content": [],
            "removed_content": [old_text[:500]],
            "modified_content": [],
            "keywords": [],
        }

    prompt = f"""Compare these two versions of a WYDOT engineering specification section and generate a structured diff.

SECTION: {section_title}
OLD VERSION ({old_year}):
{old_text[:3000]}

NEW VERSION ({new_year}):
{new_text[:3000]}

Analyze the changes carefully. Focus on:
- Material specification changes (types, grades, strengths)
- Numerical value changes (tolerances, minimums, maximums)
- Added or removed requirements
- Reference/standard updates (e.g., AASHTO edition changes)
- Procedural changes

Return a JSON object with:
- "change_type": "modified" | "unchanged"
- "severity": "critical" (safety/structural changes) | "major" (requirement changes) | "minor" (editorial/format) | "none" (identical)
- "summary": Brief 1-2 sentence description
- "details": List of specific changes (strings)
- "added_content": List of new items/requirements added
- "removed_content": List of items/requirements removed
- "modified_content": List of {{"before": str, "after": str, "significance": str}}
- "keywords": List of relevant keywords for search

Return ONLY the JSON object.
"""
    try:
        return gemini_generate_json(prompt, temperature=0.0)
    except Exception as e:
        logger.warning(f"Diff generation failed: {e}")
        return {
            "change_type": "unknown",
            "severity": "unknown",
            "summary": f"Diff generation failed: {e}",
            "details": [],
            "added_content": [],
            "removed_content": [],
            "modified_content": [],
            "keywords": [],
        }


# ─── Neo4j Temporal Storage ──────────────────────────────

def store_diffs_in_neo4j(diffs: list[dict], old_file: str, new_file: str,
                         old_year: int, new_year: int):
    """
    Store temporal diffs as CHANGE nodes in Neo4j.

    Graph pattern:
    (old_doc:Document) -[:SUPERSEDES]-> (new_doc:Document)
    (change:Change) -[:CHANGED_FROM]-> (old_doc)
    (change:Change) -[:CHANGED_TO]-> (new_doc)
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        for i, diff in enumerate(diffs):
            section_title = diff.get("section_title", f"Section_{i}")
            change_type = diff.get("change_type", "unknown")

            if change_type == "unchanged":
                continue  # Skip unchanged sections

            change_id = f"change_{old_year}_{new_year}_{i}"

            session.run("""
                MERGE (ch:Change {id: $change_id})
                SET ch.section_title = $section_title,
                    ch.change_type = $change_type,
                    ch.severity = $severity,
                    ch.summary = $summary,
                    ch.details = $details,
                    ch.old_year = $old_year,
                    ch.new_year = $new_year,
                    ch.keywords = $keywords
                WITH ch
                OPTIONAL MATCH (d_old:Document {source: $old_file})
                OPTIONAL MATCH (d_new:Document {source: $new_file})
                FOREACH (_ IN CASE WHEN d_old IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (ch)-[:CHANGED_FROM]->(d_old)
                )
                FOREACH (_ IN CASE WHEN d_new IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (ch)-[:CHANGED_TO]->(d_new)
                )
            """,
                change_id=change_id,
                section_title=section_title,
                change_type=change_type,
                severity=diff.get("severity", "unknown"),
                summary=diff.get("summary", ""),
                details=json.dumps(diff.get("details", [])),
                old_year=old_year,
                new_year=new_year,
                keywords=diff.get("keywords", []),
                old_file=old_file,
                new_file=new_file,
            )

    driver.close()
    changed_count = sum(1 for d in diffs if d.get("change_type") != "unchanged")
    logger.info(f"  Stored {changed_count} CHANGE nodes in Neo4j")


# ─── Temporal Retriever ──────────────────────────────────

def temporal_retrieve(query: str, target_series: str = None) -> list[dict]:
    """
    Retrieve temporal/version-change information for a query.

    Steps:
    1. Search CHANGE nodes by keywords
    2. Also traverse SUPERSEDES chains for context
    3. Return change summaries with provenance
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    results = []

    with driver.session() as session:
        # Search CHANGE nodes by keyword match in summary/details
        change_results = session.run("""
            MATCH (ch:Change)
            WHERE ch.summary CONTAINS $query_term
               OR any(k IN ch.keywords WHERE toLower(k) CONTAINS toLower($query_term))
            OPTIONAL MATCH (ch)-[:CHANGED_FROM]->(d_old:Document)
            OPTIONAL MATCH (ch)-[:CHANGED_TO]->(d_new:Document)
            RETURN ch.section_title AS section,
                   ch.change_type AS change_type,
                   ch.severity AS severity,
                   ch.summary AS summary,
                   ch.details AS details,
                   ch.old_year AS old_year,
                   ch.new_year AS new_year,
                   d_old.display_title AS old_doc,
                   d_new.display_title AS new_doc
            ORDER BY ch.severity DESC
            LIMIT 10
        """, query_term=query.split()[0] if query.split() else "")

        for record in change_results:
            results.append({
                "text": f"CHANGE ({record['old_year']}→{record['new_year']}): "
                       f"{record['summary']}\n"
                       f"Section: {record['section']}\n"
                       f"Severity: {record['severity']}\n"
                       f"Details: {record['details']}",
                "section_title": record["section"],
                "change_type": record["change_type"],
                "severity": record["severity"],
                "old_year": record["old_year"],
                "new_year": record["new_year"],
                "old_doc": record["old_doc"],
                "new_doc": record["new_doc"],
                "dimension": "temporal",
                "retrieval_method": "change_node_search",
            })

        # Also get SUPERSEDES chain for context
        if target_series:
            chain_results = session.run("""
                MATCH path = (d1:Document)-[:SUPERSEDES*]->(d2:Document)
                WHERE d1.document_series = $series
                RETURN [n IN nodes(path) | {title: n.display_title, year: n.year}] AS chain
                ORDER BY length(path) DESC
                LIMIT 1
            """, series=target_series)

            for record in chain_results:
                chain = record["chain"]
                chain_text = " → ".join(
                    f"{n['title']} ({n['year']})" for n in chain
                )
                results.append({
                    "text": f"VERSION CHAIN: {chain_text}",
                    "dimension": "temporal",
                    "retrieval_method": "supersedes_chain",
                })

    driver.close()
    return results


# ─── Main Diff Pipeline ──────────────────────────────────

def run_diff_pipeline(chain_name: str = None,
                      old_file: str = None, new_file: str = None,
                      max_sections: int = 50):
    """
    Run the full temporal diff pipeline.
    Either specify a chain_name or explicit old/new files.
    """
    TEMPORAL_DIR.mkdir(parents=True, exist_ok=True)

    pairs_to_process = []

    if old_file and new_file:
        # Find year info
        old_year = new_year = 0
        for cat, info in CORPUS.items():
            for doc in info["documents"]:
                if doc["file"] == old_file:
                    old_year = doc["year"]
                if doc["file"] == new_file:
                    new_year = doc["year"]
        pairs_to_process.append((old_file, new_file, old_year, new_year))
    else:
        # Use temporal chains from config
        for cname, pairs in TEMPORAL_CHAINS.items():
            if chain_name and cname != chain_name:
                continue

            info = CORPUS[cname]
            docs_by_year = {d["year"]: d["file"] for d in info["documents"]}

            for old_year, new_year in pairs:
                if old_year in docs_by_year and new_year in docs_by_year:
                    pairs_to_process.append((
                        docs_by_year[old_year],
                        docs_by_year[new_year],
                        old_year, new_year
                    ))

    if not pairs_to_process:
        logger.error("No pairs to process!")
        return

    all_results = []

    for old_f, new_f, old_y, new_y in pairs_to_process:
        logger.info(f"\n{'='*60}")
        logger.info(f"DIFFING: {old_f} ({old_y}) vs {new_f} ({new_y})")
        logger.info(f"{'='*60}")

        # Load trees
        old_tree = load_tree(old_f)
        new_tree = load_tree(new_f)

        if not old_tree or not new_tree:
            logger.error(f"  Trees not found — run phase1 first!")
            continue

        # Flatten and align
        old_sections = flatten_tree(old_tree.get("tree", {}))
        new_sections = flatten_tree(new_tree.get("tree", {}))

        # Filter to meaningful sections (level 1-2 only)
        old_sections = [s for s in old_sections if 0 < s["level"] <= 2]
        new_sections = [s for s in new_sections if 0 < s["level"] <= 2]

        logger.info(f"  Old sections: {len(old_sections)}, New sections: {len(new_sections)}")

        alignments = align_sections(old_sections, new_sections)

        # Limit sections to process
        alignments = alignments[:max_sections]

        # Generate diffs
        diffs = []
        for i, alignment in enumerate(alignments):
            if i % 10 == 0:
                logger.info(f"  Diff progress: {i}/{len(alignments)}")

            old_sec = alignment.get("old_section")
            new_sec = alignment.get("new_section")

            old_text = ""
            new_text = ""
            section_title = ""

            if old_sec:
                old_text = extract_section_text(
                    DATA_DIR / old_f,
                    old_sec["page_start"], old_sec["page_end"]
                )
                section_title = old_sec["title"]

            if new_sec:
                new_text = extract_section_text(
                    DATA_DIR / new_f,
                    new_sec["page_start"], new_sec["page_end"]
                )
                if not section_title:
                    section_title = new_sec["title"]

            diff = generate_diff(old_text, new_text, section_title, old_y, new_y)
            diff["section_title"] = section_title
            diff["match_type"] = alignment["match_type"]
            diff["similarity"] = alignment["similarity"]
            diffs.append(diff)

        # Save diffs to disk
        output_file = TEMPORAL_DIR / f"diff_{old_y}_to_{new_y}.json"
        with open(output_file, "w") as f:
            json.dump({
                "old_file": old_f, "new_file": new_f,
                "old_year": old_y, "new_year": new_y,
                "total_alignments": len(alignments),
                "diffs": diffs,
            }, f, indent=2)
        logger.info(f"  Saved diffs to: {output_file.name}")

        # Store in Neo4j
        try:
            store_diffs_in_neo4j(diffs, old_f, new_f, old_y, new_y)
        except Exception as e:
            logger.warning(f"  Neo4j storage failed (will retry later): {e}")

        # Stats
        stats = {
            "pair": f"{old_y} → {new_y}",
            "total": len(diffs),
            "modified": sum(1 for d in diffs if d.get("change_type") == "modified"),
            "added": sum(1 for d in diffs if d.get("change_type") == "added"),
            "removed": sum(1 for d in diffs if d.get("change_type") == "removed"),
            "unchanged": sum(1 for d in diffs if d.get("change_type") == "unchanged"),
            "critical": sum(1 for d in diffs if d.get("severity") == "critical"),
            "major": sum(1 for d in diffs if d.get("severity") == "major"),
        }
        all_results.append(stats)

    # Print summary
    print(f"\n{'='*80}")
    print("TEMPORAL DIFF SUMMARY")
    print(f"{'='*80}")
    print(f"{'Pair':<20} {'Total':>6} {'Modified':>9} {'Added':>6} {'Removed':>8} {'Critical':>9} {'Major':>6}")
    print("-" * 80)
    for s in all_results:
        print(f"{s['pair']:<20} {s['total']:>6} {s['modified']:>9} {s['added']:>6} "
              f"{s['removed']:>8} {s['critical']:>9} {s['major']:>6}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate temporal diffs between editions")
    parser.add_argument("--chain", type=str, help="Process only this chain (e.g., standard_specs)")
    parser.add_argument("--pair", nargs=2, metavar=("OLD", "NEW"), help="Diff specific file pair")
    parser.add_argument("--max-sections", type=int, default=50, help="Max sections to diff per pair")
    args = parser.parse_args()

    if args.pair:
        run_diff_pipeline(old_file=args.pair[0], new_file=args.pair[1],
                         max_sections=args.max_sections)
    else:
        run_diff_pipeline(chain_name=args.chain, max_sections=args.max_sections)
