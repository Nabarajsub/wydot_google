#!/usr/bin/env python3
"""
WYDOT Document Ingestion Service - Cloud Run Version
Triggered by Cloud Storage events via Eventarc for automatic document ingestion.

Features:
- HTTP endpoint for Eventarc triggers
- Downloads files from GCS bucket
- Processes PDF/DOCX documents
- Ingests into Neo4j vector store
- Moves processed files to processed/ or failed/ folders
"""

import os
import re
import gc
import json
import logging
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any, List

from flask import Flask, request, jsonify
from google.cloud import storage
from dotenv import load_dotenv

# Document processing
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from pypdf import PdfReader
import docx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

# =========================================================
# CONFIGURATION
# =========================================================

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
NEO4J_INDEX_NAME = os.getenv("NEO4J_INDEX_NAME", "wydot_vector_index")

GCS_BUCKET = os.getenv("GCS_BUCKET", "wydot-documents")
GCS_INCOMING_PREFIX = os.getenv("GCS_INCOMING_PREFIX", "incoming/")
GCS_PROCESSED_PREFIX = os.getenv("GCS_PROCESSED_PREFIX", "processed/")
GCS_FAILED_PREFIX = os.getenv("GCS_FAILED_PREFIX", "failed/")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Initialize embeddings (lazy load for cold start optimization)
_embeddings = None

def get_embeddings():
    """Lazy load embeddings to optimize cold start time."""
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embeddings model...")
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        logger.info("Embeddings model loaded.")
    return _embeddings

# Initialize Flask app
app = Flask(__name__)

# =========================================================
# METADATA EXTRACTORS
# =========================================================

def get_pdf_internal_metadata(filepath: str) -> Dict[str, Any]:
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
    except Exception as e:
        logger.warning(f"Failed to extract PDF metadata: {e}")
    return meta


def get_docx_internal_metadata(filepath: str) -> Dict[str, Any]:
    """Extract Author, Created, and Title from DOCX properties."""
    meta = {"author": "Unknown", "title": "Unknown", "created": "Unknown"}
    try:
        doc = docx.Document(filepath)
        props = doc.core_properties
        meta["author"] = props.author or "Unknown"
        meta["title"] = props.title or "Unknown"
        if props.created:
            meta["created"] = props.created.year
    except Exception as e:
        logger.warning(f"Failed to extract DOCX metadata: {e}")
    return meta


def get_filename_metadata(filename: str) -> tuple[int, str]:
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


def find_section_header(text: str) -> Optional[str]:
    """Detect 'SECTION 101' style headers."""
    match = re.search(
        r"(SECTION\s+\d+|DIVISION\s+\d+|CHAPTER\s+\d+)",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()
    return None


# =========================================================
# DOCUMENT PROCESSING
# =========================================================

def process_file(filepath: str, filename: str) -> List[Any]:
    """Process a single file and return document chunks with metadata."""
    ext = os.path.splitext(filename)[1].lower()
    
    internal_meta = {"author": "Unknown", "title": "Unknown", "created": "Unknown"}

    if ext == ".pdf":
        loader = PyPDFLoader(filepath)
        internal_meta = get_pdf_internal_metadata(filepath)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(filepath)
        internal_meta = get_docx_internal_metadata(filepath)
    else:
        logger.warning(f"Skipping unsupported file type: {filename}")
        return []

    try:
        docs = loader.load()
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
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

        doc.metadata.update({
            "source": str(filename),
            "year": str(final_year),
            "doc_type": str(file_type),
            "section": str(current_section),
            "author": str(internal_meta["author"]),
            "title": str(internal_meta["title"]),
            "page": int(page_num),
            "ingested_at": datetime.utcnow().isoformat(),
        })

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


def ingest_to_neo4j(docs: List[Any], index_name: str) -> int:
    """Ingest documents into Neo4j vector store. Returns chunk count."""
    if not docs:
        return 0
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents")
    
    try:
        Neo4jVector.from_documents(
            chunks,
            get_embeddings(),
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            index_name=index_name,
            node_label="Chunk",
            text_node_property="text",
            embedding_node_property="embedding",
        )
        return len(chunks)
    except Exception as e:
        logger.error(f"Neo4j ingestion failed: {e}")
        raise


# =========================================================
# GCS FILE OPERATIONS
# =========================================================

def download_from_gcs(bucket_name: str, blob_name: str, local_path: str) -> None:
    """Download a file from GCS to local path."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    logger.info(f"Downloaded gs://{bucket_name}/{blob_name} to {local_path}")


def move_gcs_file(bucket_name: str, source_blob: str, dest_blob: str) -> None:
    """Move a file within GCS (copy + delete)."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    source = bucket.blob(source_blob)
    
    # Copy to destination
    bucket.copy_blob(source, bucket, dest_blob)
    logger.info(f"Copied gs://{bucket_name}/{source_blob} to gs://{bucket_name}/{dest_blob}")
    
    # Delete source
    source.delete()
    logger.info(f"Deleted gs://{bucket_name}/{source_blob}")


def get_destination_path(original_blob: str, success: bool) -> str:
    """Generate destination path for processed/failed files."""
    filename = os.path.basename(original_blob)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    prefix = GCS_PROCESSED_PREFIX if success else GCS_FAILED_PREFIX
    return f"{prefix}{timestamp}_{filename}"


# =========================================================
# HTTP ENDPOINTS
# =========================================================

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "wydot-ingestion",
        "neo4j_configured": bool(NEO4J_URI),
        "gcs_bucket": GCS_BUCKET,
    })


@app.route("/ingest", methods=["POST"])
def ingest_from_eventarc():
    """
    Handle Eventarc Cloud Storage trigger.
    
    Expected CloudEvents format with GCS object data:
    {
        "bucket": "bucket-name",
        "name": "incoming/document.pdf",
        "contentType": "application/pdf",
        ...
    }
    """
    # Parse CloudEvents data
    envelope = request.get_json()
    if not envelope:
        return jsonify({"error": "No JSON payload received"}), 400
    
    # Handle Eventarc wrapping
    if "message" in envelope:
        # Pub/Sub wrapped format
        message = envelope.get("message", {})
        if isinstance(message.get("data"), str):
            import base64
            data = json.loads(base64.b64decode(message["data"]).decode("utf-8"))
        else:
            data = message.get("data", {})
    else:
        # Direct CloudEvents format
        data = envelope
    
    bucket_name = data.get("bucket", GCS_BUCKET)
    blob_name = data.get("name")
    content_type = data.get("contentType", "")
    
    if not blob_name:
        return jsonify({"error": "No file name in event data"}), 400
    
    # Validate file is in incoming folder
    if not blob_name.startswith(GCS_INCOMING_PREFIX):
        logger.info(f"Ignoring file not in incoming folder: {blob_name}")
        return jsonify({"status": "ignored", "reason": "not in incoming folder"}), 200
    
    # Validate file type
    filename = os.path.basename(blob_name)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".docx", ".doc"]:
        logger.info(f"Ignoring unsupported file type: {filename}")
        return jsonify({"status": "ignored", "reason": f"unsupported file type: {ext}"}), 200
    
    logger.info(f"Processing file: gs://{bucket_name}/{blob_name}")
    
    # Process the file
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
        
        # Download from GCS
        download_from_gcs(bucket_name, blob_name, tmp_path)
        
        # Process document
        docs = process_file(tmp_path, filename)
        
        if not docs:
            raise ValueError(f"No content extracted from {filename}")
        
        # Ingest to Neo4j
        chunk_count = ingest_to_neo4j(docs, NEO4J_INDEX_NAME)
        
        # Move to processed folder
        dest_path = get_destination_path(blob_name, success=True)
        move_gcs_file(bucket_name, blob_name, dest_path)
        
        # Cleanup
        os.unlink(tmp_path)
        gc.collect()
        
        result = {
            "status": "success",
            "file": filename,
            "documents": len(docs),
            "chunks": chunk_count,
            "destination": dest_path,
        }
        logger.info(f"Successfully ingested: {result}")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Failed to process {blob_name}: {e}")
        
        # Try to move to failed folder
        try:
            dest_path = get_destination_path(blob_name, success=False)
            move_gcs_file(bucket_name, blob_name, dest_path)
        except Exception as move_error:
            logger.error(f"Failed to move file to failed folder: {move_error}")
            dest_path = "move_failed"
        
        # Cleanup temp file if it exists
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        return jsonify({
            "status": "error",
            "file": filename,
            "error": str(e),
            "destination": dest_path,
        }), 500


@app.route("/ingest/manual", methods=["POST"])
def ingest_manual():
    """
    Manual ingestion endpoint for testing or direct API calls.
    
    Expected JSON:
    {
        "bucket": "bucket-name",
        "blob": "incoming/document.pdf"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload"}), 400
    
    bucket_name = data.get("bucket", GCS_BUCKET)
    blob_name = data.get("blob")
    
    if not blob_name:
        return jsonify({"error": "Missing 'blob' field"}), 400
    
    # Reuse the eventarc handler logic
    fake_event = {
        "bucket": bucket_name,
        "name": blob_name,
    }
    
    # Temporarily set request.json to the fake event
    with app.test_request_context(json=fake_event):
        return ingest_from_eventarc()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting ingestion service on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
