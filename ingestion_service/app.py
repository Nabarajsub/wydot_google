import os
import logging
import tempfile
import json
import urllib.parse
import re
from flask import Flask, request, jsonify
from google.cloud import storage
import google.cloud.logging
from dotenv import load_dotenv

# Load local environment (for testing)
load_dotenv()

# Setup improved logging
client = google.cloud.logging.Client()
client.setup_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core processing libraries
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector

# === Configuration ===
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
GCS_BUCKET = os.environ.get("GCS_BUCKET") # Should be set in deployment
HF_HOME = os.environ.get("HF_HOME", "/app/model_cache")

# Initialize Flask
app = Flask(__name__)

# Initialize Embeddings (load from baked cache)
logger.info("loading embeddings model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    cache_folder=HF_HOME
)
logger.info("embeddings model loaded.")

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
    """Process a file and return documents with enhanced metadata."""
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()
    
    docs = []
    try:
        if ext == ".pdf":
            # Suppress pypdf warnings
            logging.getLogger("pypdf").setLevel(logging.ERROR)
            loader = PyPDFLoader(filepath)
            docs = loader.load()
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(filepath)
            docs = loader.load()
        else:
            logger.warning(f"Unsupported file type: {filename}")
            return []
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}")
        return []

    # Metadata enrichment
    processed_docs = []
    current_section = "General"
    
    # Infer basic metadata from filename
    year_match = re.search(r"(20\d{2})", filename)
    file_year = year_match.group(1) if year_match else "2024"
    
    filename_lower = filename.lower()
    if "specification" in filename_lower:
        doc_type = "Specification"
    elif "report" in filename_lower:
        doc_type = "Annual Report"
    elif "memo" in filename_lower:
        doc_type = "Memo"
    else:
        doc_type = "General Document"

    for doc in docs:
        # Detect section headers
        possible_header = find_section_header(doc.page_content)
        if possible_header:
            current_section = possible_header
            
        page_num = doc.metadata.get("page", 0) + 1
        
        doc.metadata.update({
            "source": filename,
            "year": file_year,
            "doc_type": doc_type,
            "section": current_section,
            "title": filename, # Simplify title logic for robustness
            "page": int(page_num)
        })
        
        # Prepend header for context (improves RAG retrieval)
        header = (
            f"SOURCE: {filename}\n"
            f"YEAR: {file_year}\n"
            f"TYPE: {doc_type}\n"
            f"SECTION: {current_section}\n"
            f"PAGE: {page_num}\n"
            f"--- CONTENT ---\n"
        )
        doc.page_content = header + doc.page_content
        processed_docs.append(doc)
        
    return processed_docs

@app.route("/", methods=["POST"])
@app.route("/ingest", methods=["POST"])
def ingest_file():
    # Parse CloudEvent
    envelope = request.get_json()
    if not envelope:
        logger.error("No JSON body received")
        return "Bad Request: No JSON", 400

    # Handle both raw GCS event and Eventarc wrapped event
    if "message" in envelope:
        # Pub/Sub wrapped
        try:
            import base64
            pubsub_message = envelope["message"]
            data = json.loads(base64.b64decode(pubsub_message["data"]).decode("utf-8"))
        except Exception:
            return "Bad Request: Invalid Pub/Sub message", 400
    else:
        # Direct Eventarc or raw
        data = envelope

    # Extract bucket and file info
    bucket_name = data.get("bucket")
    file_path = data.get("name") # e.g., "incoming/myfile.pdf"
    
    if not file_path or not bucket_name:
        logger.warning(f"Invalid event data: {data}")
        return "Ignored: Missing bucket/name", 200 # Acknowledge to stop retries

    if not file_path.startswith("incoming/"):
        logger.info(f"Skipping non-incoming file: {file_path}")
        return "Ignored", 200

    # DECODE FILE PATH (Critical fix for spaces)
    try:
        decoded_path = urllib.parse.unquote(file_path)
        if decoded_path != file_path:
            logger.info(f"Decoded path: {file_path} -> {decoded_path}")
    except Exception:
        decoded_path = file_path

    logger.info(f"Processing file: {file_path} from {bucket_name}")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Try both raw and decoded paths to find the blob
    blob = bucket.blob(file_path)
    if not blob.exists():
        blob = bucket.blob(decoded_path)
        if not blob.exists():
            logger.error(f"Blob not found: {file_path} (or {decoded_path})")
            # If it doesn't exist, we can't process it. Acknowledge to stop retries.
            return f"Error: Blob not found", 200 

    # Download to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_path)[1]) as tmp:
        blob.download_to_filename(tmp.name)
        tmp_path = tmp.name
    
    try:
        # Process
        logger.info(f"Downloaded to {tmp_path}, parsing...")
        docs = process_file(tmp_path)
        
        if not docs:
            logger.warning("No documents parsed.")
            # Move to failed
            new_blob_name = file_path.replace("incoming/", "failed/")
            bucket.rename_blob(blob, new_blob_name)
            logger.info(f"Moved to failed: {new_blob_name}")
            return "Processed (Empty)", 200

        # Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Generated {len(chunks)} chunks.")

        # Ingest to Neo4j
        if chunks:
            Neo4jVector.from_documents(
                chunks,
                embeddings,
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                index_name="wydot_vector_index", # Default index matching chat app
                node_label="Chunk",
                text_node_property="text",
                embedding_node_property="embedding"
            )
            logger.info("Ingested to Neo4j.")

        # Move to processed
        new_blob_name = file_path.replace("incoming/", "processed/")
        bucket.rename_blob(blob, new_blob_name)
        logger.info(f"Moved to processed: {new_blob_name}")

        return "Ingestion Complete", 200

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        # Move to failed?
        try:
           new_blob_name = file_path.replace("incoming/", "failed/")
           bucket.rename_blob(blob, new_blob_name)
        except:
           pass
        return f"Error: {e}", 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
