"""
Phase 1: Structural Retriever (alpha dimension)
================================================
Given a query, navigates the PageIndex tree to find relevant pages.
Uses LLM reasoning to traverse the hierarchy.

This is the STRUCTURAL dimension (alpha) of SpecRAG.
"""
import json
import sys
import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR, PAGEINDEX_DIR, CORPUS
from utils.gemini_client import gemini_generate_json

logger = logging.getLogger(__name__)


def load_all_trees() -> dict:
    """Load all PageIndex trees from disk. Returns {filename: tree_data}."""
    trees = {}
    for tree_file in PAGEINDEX_DIR.glob("*_tree.json"):
        with open(tree_file) as f:
            data = json.load(f)
            trees[data["file"]] = data
    return trees


def tree_to_compact(tree: dict, depth: int = 0, max_depth: int = 3) -> str:
    """
    Convert tree to compact text for LLM navigation.
    Only shows up to max_depth levels to keep prompt small.
    """
    indent = "  " * depth
    title = tree.get("title", "Unknown")
    page_start = tree.get("page_start", "?")
    page_end = tree.get("page_end", "?")
    children = tree.get("children", [])

    line = f"{indent}[{page_start}-{page_end}] {title}"

    if depth >= max_depth and children:
        line += f" ({len(children)} subsections...)"
        return line

    lines = [line]
    for child in children:
        lines.append(tree_to_compact(child, depth + 1, max_depth))

    return "\n".join(lines)


def navigate_tree_for_query(query: str, trees: dict = None,
                            target_docs: list[str] = None) -> list[dict]:
    """
    Use LLM to navigate PageIndex trees and find relevant page ranges.

    Args:
        query: User's natural language query
        trees: Pre-loaded trees dict (optional, loads from disk if None)
        target_docs: List of specific filenames to search (optional, searches all)

    Returns:
        List of {file, page_start, page_end, section_path, relevance_reason}
    """
    if trees is None:
        trees = load_all_trees()

    if target_docs:
        trees = {k: v for k, v in trees.items() if k in target_docs}

    if not trees:
        logger.warning("No trees available for navigation")
        return []

    # Build compact tree representations for LLM
    tree_texts = []
    for filename, tree_data in trees.items():
        compact = tree_to_compact(tree_data["tree"], max_depth=3)
        tree_texts.append(
            f"=== DOCUMENT: {filename} ({tree_data['year']} {tree_data['doc_type']}) ===\n"
            f"Total Pages: {tree_data['total_pages']}\n"
            f"{compact}\n"
        )

    all_trees_text = "\n".join(tree_texts)

    prompt = f"""You are a document structure navigator for WYDOT (Wyoming Department of Transportation) engineering specifications.

Given a user query, analyze the document tree structures below and identify which sections are most likely to contain the answer.

DOCUMENT TREES:
{all_trees_text}

USER QUERY: {query}

Instructions:
1. Analyze the query to understand what information is being sought
2. Navigate the tree hierarchy to find the most relevant sections
3. Return 1-5 relevant page ranges, prioritizing the most specific match
4. Consider that related information may span multiple sections

Return a JSON array of objects with these fields:
- "file": the PDF filename
- "page_start": first relevant page (integer)
- "page_end": last relevant page (integer)
- "section_path": the hierarchy path like "Division 5 > Section 501 > 501.02"
- "relevance_reason": brief explanation of why this section is relevant

Return ONLY the JSON array, no other text.
Example: [{{"file": "2010 Standard Specifications.pdf", "page_start": 245, "page_end": 252, "section_path": "Division 5 > Section 501", "relevance_reason": "Contains concrete specifications"}}]
"""

    try:
        results = gemini_generate_json(prompt, temperature=0.0)
        if isinstance(results, list):
            return results
        return []
    except Exception as e:
        logger.error(f"Tree navigation failed: {e}")
        return []


def extract_pages(pdf_filename: str, page_start: int, page_end: int) -> str:
    """
    Extract text from specific page range of a PDF.

    Args:
        pdf_filename: Name of PDF file in DATA_DIR
        page_start: First page (1-indexed)
        page_end: Last page (1-indexed)

    Returns:
        Extracted text from the page range
    """
    pdf_path = DATA_DIR / pdf_filename
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return ""

    doc = fitz.open(str(pdf_path))
    texts = []

    # Convert to 0-indexed, clamp to valid range
    start = max(0, page_start - 1)
    end = min(len(doc), page_end)

    for page_num in range(start, end):
        page = doc[page_num]
        text = page.get_text("text")
        texts.append(f"--- Page {page_num + 1} ---\n{text}")

    doc.close()
    return "\n\n".join(texts)


def structural_retrieve(query: str, trees: dict = None,
                        target_docs: list[str] = None,
                        max_pages: int = 20) -> list[dict]:
    """
    Full structural retrieval pipeline:
    1. Navigate trees to find relevant sections
    2. Extract text from those pages
    3. Return structured context with provenance

    Args:
        query: User query
        trees: Pre-loaded trees (optional)
        target_docs: Specific documents to search (optional)
        max_pages: Maximum total pages to extract

    Returns:
        List of {text, file, page_start, page_end, section_path, relevance_reason, dimension}
    """
    # Step 1: Navigate trees
    nav_results = navigate_tree_for_query(query, trees, target_docs)

    if not nav_results:
        return []

    # Step 2: Extract text from identified pages (with budget)
    contexts = []
    pages_used = 0

    for result in nav_results:
        remaining = max_pages - pages_used
        if remaining <= 0:
            break

        page_start = result.get("page_start", 1)
        page_end = result.get("page_end", page_start + 5)

        # Clamp to budget
        actual_end = min(page_end, page_start + remaining - 1)

        text = extract_pages(result["file"], page_start, actual_end)
        pages_used += (actual_end - page_start + 1)

        contexts.append({
            "text": text,
            "file": result["file"],
            "page_start": page_start,
            "page_end": actual_end,
            "section_path": result.get("section_path", ""),
            "relevance_reason": result.get("relevance_reason", ""),
            "dimension": "structural",
            "retrieval_method": "pageindex_tree_navigation",
        })

    logger.info(f"Structural retrieval: {len(contexts)} sections, {pages_used} pages extracted")
    return contexts


if __name__ == "__main__":
    # Quick test
    trees = load_all_trees()
    print(f"Loaded {len(trees)} trees")

    test_query = "What are the requirements for concrete mix design?"
    results = structural_retrieve(test_query, trees)
    for r in results:
        print(f"\n[{r['file']} p{r['page_start']}-{r['page_end']}] {r['section_path']}")
        print(f"  Reason: {r['relevance_reason']}")
        print(f"  Text preview: {r['text'][:200]}...")
