"""
Phase 2: Neo4j Ingestion Pipeline for Full Corpus
===================================================
Ingests all 19 WYDOT documents into Neo4j knowledge graph with:
- Hybrid chunking (structural regex + semantic)
- Entity extraction (9 types)
- SUPERSEDES version chains
- Cross-document REFERENCES edges
- Gemini embeddings (768d)

Extends the existing ingestneo4j_updated.py patterns for the full corpus.

Usage:
    python -m phase2_ingestion.ingest_corpus                    # Ingest all
    python -m phase2_ingestion.ingest_corpus --category construction_manuals
    python -m phase2_ingestion.ingest_corpus --file "2026 Construction Manual.pdf"
    python -m phase2_ingestion.ingest_corpus --dry-run          # Preview only
"""
import json
import re
import sys
import time
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field

import fitz  # PyMuPDF
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_DIR, CORPUS, TEMPORAL_CHAINS,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    GEMINI_API_KEY, GEMINI_FLASH_MODEL
)
from utils.gemini_client import gemini_generate_json, gemini_embed, retry_with_backoff

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Pydantic Schemas ─────────────────────────────────────

class DocumentMetadata(BaseModel):
    display_title: str = Field(description="Clean display title")
    document_series: str = Field(description="Series name for version chaining")
    year: int = Field(description="Publication year")
    primary_category: str = Field(description="Primary category")
    secondary_category: str = Field(default="", description="Secondary category")


class Entity(BaseModel):
    name: str = Field(description="Canonical entity name")
    label: str = Field(description="Entity type: Material|Standard|TestMethod|Specification|Form|Equipment|Concept|Project|Committee")


class Relationship(BaseModel):
    source_entity: str
    target_entity: str
    relation_type: str = Field(description="Relationship type in ALL_CAPS")


class GraphExtraction(BaseModel):
    entities: list[Entity] = []
    relationships: list[Relationship] = []


# ─── PDF Extraction ───────────────────────────────────────

def extract_pdf_text(pdf_path: Path) -> tuple[str, list[tuple[int, int, int]]]:
    """
    Extract text from PDF with page mapping.
    Returns (full_text, page_map) where page_map is [(start_char, end_char, page_num)].
    """
    doc = fitz.open(str(pdf_path))
    full_text = ""
    page_map = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        # Also extract tables as markdown
        tables = page.find_tables()
        for table in tables:
            try:
                df = table.to_pandas()
                text += "\n\n[TABLE]\n" + df.to_markdown(index=False) + "\n[/TABLE]\n"
            except Exception:
                pass

        start = len(full_text)
        full_text += text + "\n\n"
        end = len(full_text)
        page_map.append((start, end, page_num + 1))

    doc.close()
    return full_text, page_map


def get_page_number(char_pos: int, page_map: list[tuple[int, int, int]]) -> int:
    """Get page number for a character position."""
    for start, end, page in page_map:
        if start <= char_pos < end:
            return page
    return page_map[-1][2] if page_map else 1


# ─── Hybrid Chunking ─────────────────────────────────────

STRUCTURAL_PATTERNS = [
    re.compile(r"\n(?=DIVISION\s+\d+)", re.IGNORECASE),
    re.compile(r"\n(?=SECTION\s+\d+)", re.IGNORECASE),
    re.compile(r"\n(?=CHAPTER\s+\d+)", re.IGNORECASE),
    re.compile(r"\n(?=\d{3}\.\d{2}\s+[A-Z])", re.IGNORECASE),
]


def hybrid_chunk(text: str, page_map: list, source: str,
                 max_chunk_size: int = 2000, min_chunk_size: int = 200) -> list[dict]:
    """
    Two-layer chunking:
    Layer 1: Split on structural boundaries (DIVISION, SECTION, CHAPTER)
    Layer 2: Split large structural chunks by size with overlap
    """
    # Layer 1: Structural split
    split_points = set()
    for pattern in STRUCTURAL_PATTERNS:
        for match in pattern.finditer(text):
            split_points.add(match.start())

    split_points = sorted(split_points)

    if not split_points:
        # No structural markers — split by size
        split_points = list(range(0, len(text), max_chunk_size))

    # Create structural segments
    segments = []
    for i, start in enumerate(split_points):
        end = split_points[i + 1] if i + 1 < len(split_points) else len(text)
        segment = text[start:end].strip()
        if segment:
            # Detect section name from first line
            first_line = segment.split("\n")[0][:100]
            segments.append({
                "text": segment,
                "start_char": start,
                "section": first_line.strip(),
            })

    # Layer 2: Split large segments
    chunks = []
    seq = 0

    for seg in segments:
        seg_text = seg["text"]

        if len(seg_text) <= max_chunk_size:
            page = get_page_number(seg["start_char"], page_map)
            chunks.append({
                "id": f"{source}_chk_{seq}",
                "text": seg_text,
                "source": source,
                "section": seg["section"],
                "seq": seq,
                "page": page,
            })
            seq += 1
        else:
            # Split by paragraphs, then merge up to max_chunk_size
            paragraphs = seg_text.split("\n\n")
            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) > max_chunk_size and len(current_chunk) >= min_chunk_size:
                    page = get_page_number(seg["start_char"], page_map)
                    chunks.append({
                        "id": f"{source}_chk_{seq}",
                        "text": current_chunk.strip(),
                        "source": source,
                        "section": seg["section"],
                        "seq": seq,
                        "page": page,
                    })
                    seq += 1
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para

            if current_chunk.strip() and len(current_chunk.strip()) >= min_chunk_size:
                page = get_page_number(seg["start_char"], page_map)
                chunks.append({
                    "id": f"{source}_chk_{seq}",
                    "text": current_chunk.strip(),
                    "source": source,
                    "section": seg["section"],
                    "seq": seq,
                    "page": page,
                })
                seq += 1

    return chunks


# ─── Entity Extraction ────────────────────────────────────

# Global entity memory for inline resolution
_entity_memory = set()


def extract_entities(chunk_text: str, known_entities: set = None) -> dict:
    """
    Use Gemini to extract entities and relationships from a chunk.
    Uses inline entity resolution to avoid duplicates.
    """
    if known_entities is None:
        known_entities = _entity_memory

    known_list = list(known_entities)[:100]  # Last 100 for context
    known_str = ", ".join(known_list) if known_list else "None yet"

    prompt = f"""Extract entities and relationships from this WYDOT engineering specification text.

ENTITY TYPES (use exactly one):
- Material (e.g., "Type IL Portland Cement", "Class A Concrete")
- Standard (e.g., "AASHTO M-85", "ASTM C150")
- TestMethod (e.g., "AASHTO T-27", "WYDOT Test Method 412")
- Specification (e.g., "Section 501", "Division 800")
- Form (e.g., "Form 337", "Certification of Compliance")
- Equipment (e.g., "Nuclear Density Gauge", "Concrete Vibrator")
- Concept (e.g., "Compressive Strength", "Slump Test")
- Project (e.g., "I-80 Corridor", "Cheyenne Bypass")
- Committee (e.g., "WYDOT Materials Program", "FHWA")

ENTITY RESOLUTION:
These entities already exist: [{known_str}]
If you detect an entity that matches or closely matches one above, use the EXISTING name exactly.
Only create new entities for genuinely new concepts.

RELATIONSHIP TYPES (use ALL_CAPS):
REQUIRES, REFERENCES, TESTED_BY, SPECIFIED_IN, USED_FOR, SUPERSEDED_BY, PART_OF, APPROVED_BY

TEXT:
{chunk_text[:3000]}

Return a JSON object with "entities" (list of {{"name": str, "label": str}}) and "relationships" (list of {{"source_entity": str, "target_entity": str, "relation_type": str}}).
Return ONLY the JSON object.
"""
    try:
        result = gemini_generate_json(prompt, temperature=0.0)
        # Update entity memory
        for ent in result.get("entities", []):
            _entity_memory.add(ent["name"])
        return result
    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")
        return {"entities": [], "relationships": []}


# ─── Document Metadata Extraction ─────────────────────────

def get_document_metadata(text_preview: str, filename: str, year_hint: int,
                          type_hint: str, series_hint: str) -> dict:
    """Extract/construct document metadata."""
    prompt = f"""Analyze this document and provide metadata.

Filename: {filename}
Year hint: {year_hint}
Type hint: {type_hint}
Series hint: {series_hint}

First 2000 characters of text:
{text_preview[:2000]}

Return JSON with:
- "display_title": A clean, descriptive title
- "document_series": The series this belongs to (for version chaining with other editions)
- "year": Publication year (integer)
- "primary_category": Main category (e.g., "Standard Specifications", "Construction Manual", "Materials Testing")
- "secondary_category": Sub-category if applicable, else empty string

Return ONLY the JSON object.
"""
    try:
        meta = gemini_generate_json(prompt, temperature=0.0)
        # Use hints as fallback
        meta.setdefault("year", year_hint)
        meta.setdefault("document_series", series_hint)
        return meta
    except Exception:
        return {
            "display_title": filename.replace(".pdf", ""),
            "document_series": series_hint,
            "year": year_hint,
            "primary_category": type_hint,
            "secondary_category": "",
        }


# ─── Neo4j Writer ─────────────────────────────────────────

class Neo4jWriter:
    """Writes document graph to Neo4j."""

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_constraints()

    def _ensure_constraints(self):
        """Create constraints and indexes if they don't exist."""
        with self.driver.session() as session:
            # Constraints
            try:
                session.run("CREATE CONSTRAINT doc_source IF NOT EXISTS FOR (d:Document) REQUIRE d.source IS UNIQUE")
            except Exception:
                pass
            try:
                session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
            except Exception:
                pass

    def write_document(self, metadata: dict, chunks: list[dict],
                       entities_by_chunk: dict, embeddings: list[list[float]]):
        """Write a complete document to Neo4j."""
        source = metadata.get("source", metadata.get("file", "unknown"))

        with self.driver.session() as session:
            # 1. Create Document node
            session.run("""
                MERGE (d:Document {source: $source})
                SET d.display_title = $title,
                    d.document_series = $series,
                    d.year = $year,
                    d.primary_category = $primary_cat,
                    d.secondary_category = $secondary_cat
            """, source=source,
                title=metadata.get("display_title", ""),
                series=metadata.get("document_series", ""),
                year=metadata.get("year", 0),
                primary_cat=metadata.get("primary_category", ""),
                secondary_cat=metadata.get("secondary_category", ""))

            # 2. Create Section and Chunk nodes
            prev_chunk_id = None
            sections_created = set()

            for i, chunk in enumerate(chunks):
                section_name = chunk.get("section", "Unknown")
                embedding = embeddings[i] if i < len(embeddings) else None

                # Create Section if new
                if section_name not in sections_created:
                    session.run("""
                        MERGE (s:Section {name: $name, doc_source: $source})
                        WITH s
                        MATCH (d:Document {source: $source})
                        MERGE (d)-[:HAS_SECTION]->(s)
                    """, name=section_name, source=source)
                    sections_created.add(section_name)

                # Create Chunk node with embedding
                params = {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "source": source,
                    "section": section_name,
                    "seq": chunk["seq"],
                    "page": chunk["page"],
                }
                if embedding:
                    params["embedding"] = embedding

                session.run("""
                    MERGE (c:Chunk {id: $id})
                    SET c.text = $text, c.source = $source, c.section = $section,
                        c.seq = $seq, c.page = $page, c.embedding = $embedding
                    WITH c
                    MATCH (s:Section {name: $section, doc_source: $source})
                    MERGE (s)-[:HAS_CHUNK]->(c)
                """, **params)

                # NEXT_CHUNK chain
                if prev_chunk_id:
                    session.run("""
                        MATCH (c1:Chunk {id: $prev_id}), (c2:Chunk {id: $curr_id})
                        MERGE (c1)-[:NEXT_CHUNK]->(c2)
                    """, prev_id=prev_chunk_id, curr_id=chunk["id"])
                prev_chunk_id = chunk["id"]

                # 3. Create Entity nodes and MENTIONS edges
                chunk_entities = entities_by_chunk.get(chunk["id"], {})
                for ent in chunk_entities.get("entities", []):
                    label = ent.get("label", "Concept")
                    name = ent.get("name", "")
                    if not name:
                        continue

                    session.run(f"""
                        MERGE (e:{label} {{name: $name}})
                        WITH e
                        MATCH (c:Chunk {{id: $chunk_id}})
                        MERGE (c)-[:MENTIONS]->(e)
                    """, name=name, chunk_id=chunk["id"])

                # 4. Create inter-entity relationships
                for rel in chunk_entities.get("relationships", []):
                    rel_type = rel.get("relation_type", "RELATED_TO").upper().replace(" ", "_")
                    session.run(f"""
                        MATCH (e1 {{name: $source_name}}), (e2 {{name: $target_name}})
                        MERGE (e1)-[:{rel_type}]->(e2)
                    """, source_name=rel["source_entity"],
                        target_name=rel["target_entity"])

        logger.info(f"  Written to Neo4j: {len(chunks)} chunks, {len(sections_created)} sections")

    def build_supersedes_chains(self):
        """Create SUPERSEDES edges between documents of the same series."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Document)
                WHERE d.document_series IS NOT NULL AND d.year IS NOT NULL
                WITH d.document_series AS series, d ORDER BY d.year DESC
                WITH series, collect(d) AS docs
                WHERE size(docs) > 1
                UNWIND range(0, size(docs)-2) AS i
                WITH docs[i] AS newer, docs[i+1] AS older
                MERGE (newer)-[:SUPERSEDES]->(older)
                RETURN newer.display_title AS newer_title, newer.year AS newer_year,
                       older.display_title AS older_title, older.year AS older_year
            """)
            chains = list(result)
            for r in chains:
                logger.info(f"  SUPERSEDES: {r['newer_title']} ({r['newer_year']}) -> "
                          f"{r['older_title']} ({r['older_year']})")
            return chains

    def close(self):
        self.driver.close()


# ─── Main Ingestion Pipeline ─────────────────────────────

def ingest_document(pdf_path: Path, category: str, doc_entry: dict,
                    series_info: dict, writer: Neo4jWriter,
                    skip_entities: bool = False) -> dict:
    """
    Full ingestion pipeline for a single document.
    Returns stats dict.
    """
    filename = doc_entry["file"]
    year = doc_entry["year"]
    doc_type = series_info["type"]
    series = series_info["series"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Ingesting: {filename}")
    logger.info(f"  Type: {doc_type} | Year: {year} | Series: {series}")
    logger.info(f"{'='*60}")

    # Step 1: Extract text
    logger.info("  Step 1: Extracting PDF text...")
    full_text, page_map = extract_pdf_text(pdf_path)
    logger.info(f"  Extracted {len(full_text)} chars, {len(page_map)} pages")

    # Step 2: Get metadata
    logger.info("  Step 2: Extracting metadata...")
    metadata = get_document_metadata(full_text[:2000], filename, year, doc_type, series)
    metadata["source"] = filename
    metadata["file"] = filename
    logger.info(f"  Metadata: {metadata.get('display_title', filename)}")

    # Step 3: Hybrid chunking
    logger.info("  Step 3: Hybrid chunking...")
    chunks = hybrid_chunk(full_text, page_map, source=filename)
    logger.info(f"  Created {len(chunks)} chunks")

    # Step 4: Entity extraction (parallelized)
    entities_by_chunk = {}
    if not skip_entities:
        logger.info("  Step 4: Extracting entities (this takes a while)...")
        # Process in batches to respect rate limits
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                logger.info(f"    Entity extraction: {i}/{len(chunks)} chunks...")
            try:
                entities = extract_entities(chunk["text"])
                entities_by_chunk[chunk["id"]] = entities
            except Exception as e:
                logger.warning(f"    Entity extraction failed for chunk {i}: {e}")
                entities_by_chunk[chunk["id"]] = {"entities": [], "relationships": []}
    else:
        logger.info("  Step 4: SKIPPED entity extraction (--skip-entities)")

    # Step 5: Generate embeddings
    logger.info("  Step 5: Generating embeddings...")
    chunk_texts = [c["text"] for c in chunks]
    embeddings = gemini_embed(chunk_texts)
    logger.info(f"  Generated {len(embeddings)} embeddings")

    # Step 6: Write to Neo4j
    logger.info("  Step 6: Writing to Neo4j...")
    writer.write_document(metadata, chunks, entities_by_chunk, embeddings)

    stats = {
        "file": filename,
        "year": year,
        "type": doc_type,
        "pages": len(page_map),
        "chunks": len(chunks),
        "entities": sum(len(v.get("entities", [])) for v in entities_by_chunk.values()),
        "relationships": sum(len(v.get("relationships", [])) for v in entities_by_chunk.values()),
    }
    logger.info(f"  DONE: {stats}")
    return stats


def ingest_corpus(target_category: Optional[str] = None,
                  target_file: Optional[str] = None,
                  dry_run: bool = False,
                  skip_entities: bool = False):
    """
    Ingest all documents (or filtered subset) into Neo4j.
    """
    # Collect documents to ingest
    to_ingest = []
    for category, info in CORPUS.items():
        if target_category and category != target_category:
            continue
        for doc_entry in info["documents"]:
            if target_file and doc_entry["file"] != target_file:
                continue
            pdf_path = DATA_DIR / doc_entry["file"]
            if not pdf_path.exists():
                logger.warning(f"SKIP: {doc_entry['file']} not found")
                continue
            to_ingest.append((pdf_path, category, doc_entry, info))

    if not to_ingest:
        logger.error("No documents found to ingest!")
        return

    print(f"\n{'='*60}")
    print(f"SPECRAG CORPUS INGESTION")
    print(f"{'='*60}")
    print(f"Documents to ingest: {len(to_ingest)}")
    for pdf_path, cat, entry, info in to_ingest:
        print(f"  [{cat}] {entry['file']} ({entry['year']})")
    print(f"{'='*60}\n")

    if dry_run:
        print("DRY RUN - no changes made")
        return

    # Connect to Neo4j
    writer = Neo4jWriter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    all_stats = []
    try:
        for pdf_path, category, doc_entry, info in to_ingest:
            stats = ingest_document(pdf_path, category, doc_entry, info, writer,
                                   skip_entities=skip_entities)
            all_stats.append(stats)

        # Build SUPERSEDES chains
        logger.info("\nBuilding SUPERSEDES version chains...")
        writer.build_supersedes_chains()

    finally:
        writer.close()

    # Print summary
    print(f"\n{'='*80}")
    print("INGESTION SUMMARY")
    print(f"{'='*80}")
    print(f"{'File':<50} {'Year':>5} {'Pages':>6} {'Chunks':>7} {'Entities':>9}")
    print("-" * 80)
    for s in all_stats:
        name = s["file"][:47] + "..." if len(s["file"]) > 50 else s["file"]
        print(f"{name:<50} {s['year']:>5} {s['pages']:>6} {s['chunks']:>7} {s['entities']:>9}")
    print("-" * 80)
    print(f"{'TOTAL':<50} {'':>5} {sum(s['pages'] for s in all_stats):>6} "
          f"{sum(s['chunks'] for s in all_stats):>7} {sum(s['entities'] for s in all_stats):>9}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest WYDOT corpus into Neo4j")
    parser.add_argument("--category", type=str, help="Ingest only this category")
    parser.add_argument("--file", type=str, help="Ingest only this file")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--skip-entities", action="store_true",
                       help="Skip entity extraction (faster, for testing)")
    args = parser.parse_args()

    ingest_corpus(
        target_category=args.category,
        target_file=args.file,
        dry_run=args.dry_run,
        skip_entities=args.skip_entities,
    )
