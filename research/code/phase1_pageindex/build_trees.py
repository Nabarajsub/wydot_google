"""
Phase 1: PageIndex Tree Builder
================================
Extracts hierarchical document structure (TOC) from all 19 PDFs
and saves as compact JSON trees for structural retrieval.

Based on the PageIndex paper concept:
- Uses PyMuPDF bookmark parsing for well-structured PDFs
- Falls back to regex TOC parsing for scanned/flat PDFs
- Outputs a JSON tree per document with page ranges

Usage:
    python -m phase1_pageindex.build_trees           # Build all 19 trees
    python -m phase1_pageindex.build_trees --file "2010 Standard Specifications.pdf"
"""
import json
import re
import sys
import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR, PAGEINDEX_DIR, CORPUS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Regex Patterns for Section Detection ─────────────────
PATTERNS = {
    "division": re.compile(
        r"^(?:DIVISION\s+(\d+)\s*[-–—]?\s*(.+)|DIVISION\s+(\d+))\s*$",
        re.IGNORECASE | re.MULTILINE
    ),
    "section": re.compile(
        r"^(?:SECTION\s+(\d+)\s*[-–—]?\s*(.+)|(\d{3})\s*[-–—]\s*(.+))\s*$",
        re.IGNORECASE | re.MULTILINE
    ),
    "chapter": re.compile(
        r"^(?:CHAPTER\s+(\d+)\s*[-–—]?\s*(.+))\s*$",
        re.IGNORECASE | re.MULTILINE
    ),
    "subsection": re.compile(
        r"^(\d{3}\.\d{2})\s+(.+?)$",
        re.MULTILINE
    ),
    "test_method": re.compile(
        r"^(?:(?:AASHTO|ASTM|WYDOT)\s+[A-Z][\s\-]?\d+)",
        re.IGNORECASE | re.MULTILINE
    ),
}


def extract_bookmarks(doc: fitz.Document) -> list[dict]:
    """
    Extract TOC from PyMuPDF bookmarks (if available).
    Returns list of {level, title, page} dicts.
    """
    toc = doc.get_toc(simple=True)  # [(level, title, page), ...]
    if not toc:
        return []

    bookmarks = []
    for level, title, page in toc:
        bookmarks.append({
            "level": level,
            "title": title.strip(),
            "page": page,
        })
    return bookmarks


def bookmarks_to_tree(bookmarks: list[dict], total_pages: int) -> dict:
    """
    Convert flat bookmark list to nested tree with page ranges.
    Each node: {title, page_start, page_end, children: [...]}
    """
    if not bookmarks:
        return {"title": "Document", "page_start": 1, "page_end": total_pages, "children": []}

    # Add page_end to each bookmark (= next bookmark's page - 1 or total_pages)
    for i, bm in enumerate(bookmarks):
        if i + 1 < len(bookmarks):
            bm["page_end"] = bookmarks[i + 1]["page"] - 1
        else:
            bm["page_end"] = total_pages

    # Build nested tree using level hierarchy
    root = {"title": "Document", "page_start": 1, "page_end": total_pages, "children": []}
    stack = [(root, 0)]  # (node, level)

    for bm in bookmarks:
        node = {
            "title": bm["title"],
            "page_start": bm["page"],
            "page_end": bm["page_end"],
            "children": [],
        }

        # Pop stack until we find a parent with lower level
        while len(stack) > 1 and stack[-1][1] >= bm["level"]:
            stack.pop()

        # Add as child of current top
        stack[-1][0]["children"].append(node)
        stack.append((node, bm["level"]))

    return root


def extract_structure_via_regex(doc: fitz.Document) -> list[dict]:
    """
    Fallback: Extract structure by scanning page text for Division/Section/Chapter patterns.
    Used when PDF has no bookmarks.
    """
    entries = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        # Check for Division headers
        for match in PATTERNS["division"].finditer(text):
            div_num = match.group(1) or match.group(3)
            div_title = match.group(2) or f"Division {div_num}"
            entries.append({
                "level": 1,
                "title": f"Division {div_num} - {div_title.strip()}",
                "page": page_num + 1,
            })

        # Check for Section headers
        for match in PATTERNS["section"].finditer(text):
            sec_num = match.group(1) or match.group(3)
            sec_title = match.group(2) or match.group(4) or f"Section {sec_num}"
            entries.append({
                "level": 2,
                "title": f"Section {sec_num} - {sec_title.strip()}",
                "page": page_num + 1,
            })

        # Check for Chapter headers
        for match in PATTERNS["chapter"].finditer(text):
            ch_num = match.group(1)
            ch_title = match.group(2) or f"Chapter {ch_num}"
            entries.append({
                "level": 1,
                "title": f"Chapter {ch_num} - {ch_title.strip()}",
                "page": page_num + 1,
            })

        # Check for subsections
        for match in PATTERNS["subsection"].finditer(text):
            sub_num = match.group(1)
            sub_title = match.group(2)
            entries.append({
                "level": 3,
                "title": f"{sub_num} {sub_title.strip()}",
                "page": page_num + 1,
            })

    # Deduplicate (same title + same page)
    seen = set()
    unique = []
    for e in entries:
        key = (e["title"], e["page"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    # Sort by page number, then level
    unique.sort(key=lambda x: (x["page"], x["level"]))
    return unique


def build_pageindex_tree(pdf_path: Path) -> dict:
    """
    Build a PageIndex tree for a single PDF.

    Returns:
        {
            "file": "filename.pdf",
            "total_pages": N,
            "node_count": M,
            "tree": { ... nested tree ... }
        }
    """
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    # Try bookmarks first
    bookmarks = extract_bookmarks(doc)
    source = "bookmarks"

    if not bookmarks or len(bookmarks) < 5:
        # Fallback to regex parsing
        logger.info(f"  Few/no bookmarks ({len(bookmarks)}), using regex fallback...")
        bookmarks = extract_structure_via_regex(doc)
        source = "regex"

    tree = bookmarks_to_tree(bookmarks, total_pages)
    node_count = count_nodes(tree)

    logger.info(f"  Built tree: {node_count} nodes from {source} ({total_pages} pages)")
    doc.close()

    return {
        "file": pdf_path.name,
        "total_pages": total_pages,
        "node_count": node_count,
        "extraction_method": source,
        "tree": tree,
    }


def count_nodes(tree: dict) -> int:
    """Count total nodes in tree."""
    count = 1
    for child in tree.get("children", []):
        count += count_nodes(child)
    return count


def build_all_trees(target_file: Optional[str] = None):
    """
    Build PageIndex trees for all documents in the corpus.
    Saves each tree as a JSON file in PAGEINDEX_DIR.
    """
    PAGEINDEX_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for category, info in CORPUS.items():
        for doc_entry in info["documents"]:
            filename = doc_entry["file"]

            if target_file and filename != target_file:
                continue

            pdf_path = DATA_DIR / filename
            if not pdf_path.exists():
                logger.warning(f"SKIP: {filename} not found at {pdf_path}")
                continue

            logger.info(f"Processing: {filename} ({doc_entry['year']} {info['type']})")

            tree_data = build_pageindex_tree(pdf_path)
            tree_data["category"] = category
            tree_data["doc_type"] = info["type"]
            tree_data["series"] = info["series"]
            tree_data["year"] = doc_entry["year"]

            # Save tree
            safe_name = filename.replace(" ", "_").replace("–", "-").replace(".pdf", "")
            output_path = PAGEINDEX_DIR / f"{safe_name}_tree.json"
            with open(output_path, "w") as f:
                json.dump(tree_data, f, indent=2)

            logger.info(f"  Saved to: {output_path.name}")
            results.append({
                "file": filename,
                "year": doc_entry["year"],
                "type": info["type"],
                "pages": tree_data["total_pages"],
                "nodes": tree_data["node_count"],
                "method": tree_data["extraction_method"],
            })

    # Print summary
    print("\n" + "=" * 80)
    print("PAGEINDEX TREE BUILD SUMMARY")
    print("=" * 80)
    print(f"{'File':<55} {'Year':>5} {'Pages':>6} {'Nodes':>6} {'Method':>10}")
    print("-" * 80)
    total_pages = 0
    total_nodes = 0
    for r in results:
        short_name = r["file"][:52] + "..." if len(r["file"]) > 55 else r["file"]
        print(f"{short_name:<55} {r['year']:>5} {r['pages']:>6} {r['nodes']:>6} {r['method']:>10}")
        total_pages += r["pages"]
        total_nodes += r["nodes"]
    print("-" * 80)
    print(f"{'TOTAL':<55} {'':>5} {total_pages:>6} {total_nodes:>6}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build PageIndex trees for WYDOT corpus")
    parser.add_argument("--file", type=str, help="Build tree for a single file only")
    args = parser.parse_args()

    build_all_trees(target_file=args.file)
