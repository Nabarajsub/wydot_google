#!/usr/bin/env python3
"""
WYDOT GraphRAG Chatbot - Chainlit Version
Features:
- Native Login (SQLite-backed)
- Multimodal Support (Images, Audio, Video, PDF) via Gemini
- Multi-hop Reasoning
- Model Selection (Mistral / Gemini)
- Citation Linking [1], [2] -> Evidence Panel
- Audio Input Support
"""

import os
import glob
import re
import gc
import logging
from dotenv import load_dotenv

# --- FIX: SUPPRESS PYPDF WARNINGS ---
logging.getLogger("pypdf").setLevel(logging.ERROR)
# ------------------------------------

# Loaders
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector

# Internal metadata extraction
from pypdf import PdfReader
import docx

# Load environment variables
load_dotenv()

# Configuration
DATA_FOLDER = "data"
TRACKER_FILE = "local_ingestion_tracker.txt"
BATCH_SIZE = 50

# Setup Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def get_processed_files():
    if not os.path.exists(TRACKER_FILE):
        return set()
    with open(TRACKER_FILE, "r") as f:
        return set(line.strip() for line in f)


def log_processed_file(filename):
    with open(TRACKER_FILE, "a") as f:
        f.write(f"{filename}\n")


# --- ADVANCED METADATA EXTRACTORS ---

def get_pdf_internal_metadata(filepath):
    """Extract Author, Creation Date, and Title from PDF properties."""
    meta = {"author": "Unknown", "title": "Unknown", "created": "Unknown"}
    try:
        reader = PdfReader(filepath)
        info = reader.metadata
        if info:
            meta["author"] = info.get("/Author", "Unknown") or "Unknown"
            meta["title"] = info.get("/Title", "Unknown") or "Unknown"
            raw_date = info.get("/CreationDate", "")
            if "D:" in raw_date:
                meta["created"] = raw_date.split("D:")[1][:4]
    except Exception:
        pass
    return meta


def get_docx_internal_metadata(filepath):
    """Extract Author, Created, and Title from DOCX properties."""
    meta = {"author": "Unknown", "title": "Unknown", "created": "Unknown"}
    try:
        doc = docx.Document(filepath)
        props = doc.core_properties
        meta["author"] = props.author or "Unknown"
        meta["title"] = props.title or "Unknown"
        if props.created:
            meta["created"] = props.created.year
    except Exception:
        pass
    return meta


def get_filename_metadata(filename):
    """Infer metadata from the filename itself."""
    year_match = re.search(r"(20\d{2})", filename)
    year = int(year_match.group(1)) if year_match else 0

    filename_lower = filename.lower()
    if "specification" in filename_lower:
        doc_type = "Specification"
    elif "report" in filename_lower:
        doc_type = "Annual Report"
    elif "memo" in filename_lower:
        doc_type = "Memo"
    else:
        doc_type = "General Document"

    return year, doc_type


def find_section_header(text):
    """Detect 'SECTION 101' style headers."""
    match = re.search(
        r"(SECTION\s+\d+|DIVISION\s+\d+|CHAPTER\s+\d+)",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()
    return None


def process_file(filepath):
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()

    internal_meta = {"author": "Unknown", "title": "Unknown", "created": "Unknown"}

    if ext == ".pdf":
        loader = PyPDFLoader(filepath)
        internal_meta = get_pdf_internal_metadata(filepath)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(filepath)
        internal_meta = get_docx_internal_metadata(filepath)
    else:
        print(f"⚠️ Skipping unsupported file type: {filename}")
        return []

    try:
        docs = loader.load()
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")
        return []

    file_year, file_type = get_filename_metadata(filename)

    # Fallback logic for year
    final_year = (
        internal_meta["created"]
        if internal_meta["created"] != "Unknown"
        else file_year
    )
    if final_year == 0:
        final_year = "Unknown"

    processed_docs = []
    current_section = "General"

    for doc in docs:
        possible_header = find_section_header(doc.page_content)
        if possible_header:
            current_section = possible_header

        page_num = doc.metadata.get("page", 0) + 1

        doc.metadata.update(
            {
                "source": str(filename),
                "year": str(final_year),
                "doc_type": str(file_type),
                "section": str(current_section),
                "author": str(internal_meta["author"]),
                "title": str(internal_meta["title"]),
                "page": int(page_num),
            }
        )

        header = (
            f"SOURCE: {filename}\n"
            f"TITLE: {internal_meta['title']}\n"
            f"AUTHOR: {internal_meta['author']}\n"
            f"YEAR: {final_year}\n"
            f"TYPE: {file_type}\n"
            f"SECTION: {current_section}\n"
            f"PAGE: {page_num}\n"
            f"--- CONTENT ---\n"
        )

        doc.page_content = header + doc.page_content
        processed_docs.append(doc)

    return processed_docs


def main():
    if not os.path.exists(DATA_FOLDER):
        print(f"❌ Folder '{DATA_FOLDER}' not found.")
        return

    all_files = []
    for ext in ["*.pdf", "*.docx", "*.doc"]:
        all_files.extend(glob.glob(os.path.join(DATA_FOLDER, ext)))

    processed = get_processed_files()
    files_to_do = [f for f in all_files if os.path.basename(f) not in processed]

    total = len(files_to_do)
    print(f"🚀 Found {len(all_files)} files. Processing {total} new files in batches of {BATCH_SIZE}...")

    for i in range(0, total, BATCH_SIZE):
        batch_files = files_to_do[i: i + BATCH_SIZE]
        batch_docs = []

        print(f"\n📦 Batch {i // BATCH_SIZE + 1}: Loading {len(batch_files)} files...")

        for f in batch_files:
            batch_docs.extend(process_file(f))

        if not batch_docs:
            continue

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )

        chunks = text_splitter.split_documents(batch_docs)

        print(f"   > Vectorizing {len(chunks)} chunks...")

        try:
            Neo4jVector.from_documents(
                chunks,
                embeddings,
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                index_name="wydot_local_index",
                node_label="Chunk",
                text_node_property="text",
                embedding_node_property="embedding",
            )

            for f in batch_files:
                log_processed_file(os.path.basename(f))

            print("   ✅ Batch saved.")

        except Exception as e:
            print(f"   ❌ Batch failed: {e}")

        del batch_docs
        del chunks
        gc.collect()

    print("\n🎉 Full Ingestion Complete!")


if __name__ == "__main__":
    main()
