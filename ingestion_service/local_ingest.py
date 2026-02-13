#!/usr/bin/env python3
"""
WYDOT Local Ingestion Server
- Serves HTML upload page with WYDOT dashboard
- Processes PDF/DOCX via same logic as ingestneo4j.py
- Transcribes audio/video via Gemini 2.5 Flash
- Ingests everything into Neo4j (wydot_vector_index)
- Stores ingested files in ingested_data/
- Supports viewing and deleting ingested content
"""

import os
import re
import gc
import json
import time
import logging
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# --- Suppress noisy loggers ---
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Loaders
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_core.documents import Document

# Internal metadata extraction
from pypdf import PdfReader
import docx

# Gemini for transcription
import google.generativeai as genai

# Neo4j driver for delete operations
from neo4j import GraphDatabase

# Load environment
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB limit

# =========================================================
# CONFIGURATION
# =========================================================

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEO4J_INDEX = os.getenv("NEO4J_INDEX_DEFAULT", "wydot_vector_index")

# Ingested data storage
INGESTED_DIR = os.path.join(os.path.dirname(__file__), "ingested_data")
TRACKER_FILE = os.path.join(INGESTED_DIR, "ingestion_tracker.json")

os.makedirs(INGESTED_DIR, exist_ok=True)

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Setup Embeddings
print("🔄 Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ Embeddings model loaded.")

# Supported file types
TEXT_EXTENSIONS = {'.pdf', '.docx', '.doc'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.webm', '.ogg'}
VIDEO_EXTENSIONS = {'.mp4', '.webm', '.mov', '.avi'}


# =========================================================
# TRACKER (JSON file for ingested documents)
# =========================================================

def load_tracker():
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r") as f:
            return json.load(f)
    return {"files": []}


def save_tracker(tracker):
    with open(TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=2, default=str)


def add_to_tracker(filename, media_type, chunks, metadata=None):
    tracker = load_tracker()
    # Remove existing entry if present (re-ingestion)
    tracker["files"] = [f for f in tracker["files"] if f["filename"] != filename]
    tracker["files"].append({
        "filename": filename,
        "type": media_type,
        "chunks": chunks,
        "date": datetime.now().isoformat(),
        "metadata": metadata or {},
    })
    save_tracker(tracker)


def remove_from_tracker(filename):
    tracker = load_tracker()
    tracker["files"] = [f for f in tracker["files"] if f["filename"] != filename]
    save_tracker(tracker)


# =========================================================
# METADATA EXTRACTION (from ingestneo4j.py)
# =========================================================

def get_pdf_internal_metadata(filepath):
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
    meta = {"author": "Unknown", "title": "Unknown", "created": "Unknown"}
    try:
        doc_obj = docx.Document(filepath)
        props = doc_obj.core_properties
        meta["author"] = props.author or "Unknown"
        meta["title"] = props.title or "Unknown"
        if props.created:
            meta["created"] = props.created.year
    except Exception:
        pass
    return meta


def get_filename_metadata(filename):
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
    match = re.search(
        r"(SECTION\s+\d+|DIVISION\s+\d+|CHAPTER\s+\d+)",
        text,
        re.IGNORECASE,
    )
    return match.group(1).upper() if match else None


# =========================================================
# FILE PROCESSING
# =========================================================

def process_text_file(filepath, filename):
    """Process PDF/DOCX files — same logic as ingestneo4j.py"""
    ext = os.path.splitext(filename)[1].lower()
    internal_meta = {"author": "Unknown", "title": "Unknown", "created": "Unknown"}

    if ext == ".pdf":
        loader = PyPDFLoader(filepath)
        internal_meta = get_pdf_internal_metadata(filepath)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(filepath)
        internal_meta = get_docx_internal_metadata(filepath)
    else:
        return []

    try:
        docs = loader.load()
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")
        return []

    file_year, file_type = get_filename_metadata(filename)
    final_year = internal_meta["created"] if internal_meta["created"] != "Unknown" else file_year
    if final_year == 0:
        final_year = "Unknown"

    processed_docs = []
    current_section = "General"

    for doc_item in docs:
        possible_header = find_section_header(doc_item.page_content)
        if possible_header:
            current_section = possible_header

        page_num = doc_item.metadata.get("page", 0) + 1
        doc_item.metadata.update({
            "source": str(filename),
            "year": str(final_year),
            "doc_type": str(file_type),
            "section": str(current_section),
            "author": str(internal_meta["author"]),
            "title": str(internal_meta["title"]),
            "page": int(page_num),
            "media_type": "document",
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
        doc_item.page_content = header + doc_item.page_content
        processed_docs.append(doc_item)

    return processed_docs


def transcribe_media_with_gemini(filepath, filename, media_type):
    """Transcribe audio/video using Gemini 2.5 Flash"""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set — cannot transcribe media")

    print(f"🎙️ Transcribing {filename} with Gemini 2.5 Flash...")

    # Map extensions to MIME types — Gemini needs explicit types for audio
    mime_map = {
        '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.m4a': 'audio/mp4',
        '.ogg': 'audio/ogg', '.mp4': 'video/mp4', '.mov': 'video/quicktime',
        '.avi': 'video/x-msvideo',
    }
    ext = os.path.splitext(filename)[1].lower()
    # For .webm, decide based on media_type context
    if ext == '.webm':
        mime = 'video/webm' if media_type == 'video' else 'audio/webm'
    else:
        mime = mime_map.get(ext, f'{media_type}/{ext.lstrip(".")}')

    print(f"   📎 Uploading as MIME type: {mime}")
    uploaded_file = genai.upload_file(filepath, display_name=filename, mime_type=mime)

    # Wait for file to become ACTIVE (Gemini processes uploads asynchronously)
    print(f"   ⏳ Waiting for file to be processed by Gemini...")
    max_wait = 120  # seconds
    waited = 0
    while waited < max_wait:
        file_info = genai.get_file(uploaded_file.name)
        if file_info.state.name == "ACTIVE":
            break
        if file_info.state.name == "FAILED":
            raise ValueError(f"Gemini file processing failed for {filename}")
        time.sleep(3)
        waited += 3
        print(f"   ⏳ File state: {file_info.state.name} ({waited}s elapsed)...")

    if waited >= max_wait:
        raise ValueError(f"Timeout waiting for Gemini to process {filename}")

    print(f"   ✅ File is ACTIVE, generating transcription...")
    model = genai.GenerativeModel("gemini-2.5-flash")


    prompt = f"""Transcribe this {media_type} file completely and accurately.

After the transcription, provide a brief summary section with:
- TOPIC: Main topic discussed
- KEY_POINTS: 3-5 bullet points of key information
- SPEAKERS: Names or roles of speakers if identifiable
- DATE_MENTIONED: Any dates mentioned in the content

Format:
--- TRANSCRIPTION ---
[full transcription here]

--- SUMMARY ---
TOPIC: [topic]
KEY_POINTS:
- [point 1]
- [point 2]
...
SPEAKERS: [speakers]
DATE_MENTIONED: [dates]
"""

    response = model.generate_content([uploaded_file, prompt])
    transcript = response.text

    try:
        genai.delete_file(uploaded_file.name)
    except Exception:
        pass

    return transcript


def extract_transcript_metadata(transcript):
    """Extract structured metadata from Gemini transcript"""
    meta = {
        "topic": "Unknown",
        "speakers": "Unknown",
        "date_mentioned": "Unknown",
        "key_points": "",
    }

    topic_match = re.search(r"TOPIC:\s*(.+)", transcript)
    if topic_match:
        meta["topic"] = topic_match.group(1).strip()

    speakers_match = re.search(r"SPEAKERS:\s*(.+)", transcript)
    if speakers_match:
        meta["speakers"] = speakers_match.group(1).strip()

    date_match = re.search(r"DATE_MENTIONED:\s*(.+)", transcript)
    if date_match:
        meta["date_mentioned"] = date_match.group(1).strip()

    kp_match = re.search(r"KEY_POINTS:\s*((?:- .+\n?)+)", transcript)
    if kp_match:
        meta["key_points"] = kp_match.group(1).strip()

    return meta


def process_media_file(filepath, filename, media_type):
    """Process audio/video: transcribe → create documents"""
    transcript = transcribe_media_with_gemini(filepath, filename, media_type)
    meta = extract_transcript_metadata(transcript)

    file_year, _ = get_filename_metadata(filename)
    if file_year == 0:
        file_year = datetime.now().year

    header = (
        f"SOURCE: {filename}\n"
        f"TITLE: {meta['topic']}\n"
        f"MEDIA_TYPE: {media_type}\n"
        f"SPEAKERS: {meta['speakers']}\n"
        f"DATE_MENTIONED: {meta['date_mentioned']}\n"
        f"YEAR: {file_year}\n"
        f"TYPE: {media_type.capitalize()} Transcript\n"
        f"--- CONTENT ---\n"
    )

    doc = Document(
        page_content=header + transcript,
        metadata={
            "source": str(filename),
            "title": str(meta["topic"]),
            "year": str(file_year),
            "doc_type": f"{media_type.capitalize()} Transcript",
            "section": "Full Transcript",
            "author": str(meta["speakers"]),
            "page": 1,
            "media_type": media_type,
        }
    )

    return [doc], transcript, meta


def ingest_documents(docs):
    """Chunk and ingest documents into Neo4j"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    print(f"   > Vectorizing {len(chunks)} chunks...")

    Neo4jVector.from_documents(
        chunks,
        embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=NEO4J_INDEX,
        node_label="Chunk",
        text_node_property="text",
        embedding_node_property="embedding",
    )

    print(f"   ✅ {len(chunks)} chunks saved to Neo4j index '{NEO4J_INDEX}'.")
    return len(chunks)


def save_ingested_file(filepath, filename):
    """Copy ingested file to ingested_data/"""
    dest = os.path.join(INGESTED_DIR, filename)
    shutil.copy2(filepath, dest)


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No file selected"}), 400

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        if ext in TEXT_EXTENSIONS:
            print(f"\n📄 Processing document: {filename}")
            docs = process_text_file(tmp_path, filename)
            media_type = "document"
        elif ext in AUDIO_EXTENSIONS:
            print(f"\n🎵 Processing audio: {filename}")
            docs, transcript, meta = process_media_file(tmp_path, filename, "audio")
            media_type = "audio"
        elif ext in VIDEO_EXTENSIONS:
            print(f"\n🎬 Processing video: {filename}")
            docs, transcript, meta = process_media_file(tmp_path, filename, "video")
            media_type = "video"
        else:
            return jsonify({"status": "error", "message": f"Unsupported: {ext}"}), 400

        if not docs:
            return jsonify({"status": "error", "message": f"No content from {filename}"}), 400

        chunk_count = ingest_documents(docs)

        # Save file and track
        save_ingested_file(tmp_path, filename)
        add_to_tracker(filename, media_type, chunk_count,
                        meta if media_type in ("audio", "video") else None)

        del docs
        gc.collect()

        return jsonify({
            "status": "success",
            "message": f"Ingested {filename}",
            "chunks": chunk_count,
            "filename": filename,
            "index": NEO4J_INDEX,
        })

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Transcribe audio/video without ingesting — for preview"""
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400

    file = request.files["file"]
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext not in AUDIO_EXTENSIONS and ext not in VIDEO_EXTENSIONS:
        return jsonify({"status": "error", "message": "Not an audio/video file"}), 400

    media_type = "audio" if ext in AUDIO_EXTENSIONS else "video"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        transcript = transcribe_media_with_gemini(tmp_path, filename, media_type)
        meta = extract_transcript_metadata(transcript)

        return jsonify({
            "status": "success",
            "transcript": transcript,
            "metadata": meta,
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.route("/library")
def library():
    """List all ingested documents"""
    tracker = load_tracker()
    return jsonify(tracker)


@app.route("/delete", methods=["POST"])
def delete():
    """Delete all chunks for a file from Neo4j"""
    data = request.get_json()
    filename = data.get("filename")

    if not filename:
        return jsonify({"status": "error", "message": "No filename provided"}), 400

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run(
                "MATCH (c:Chunk) WHERE c.source = $source "
                "WITH c, count(c) AS cnt "
                "DETACH DELETE c "
                "RETURN cnt",
                source=filename
            )
            record = result.single()
            deleted = record["cnt"] if record else 0
        driver.close()

        # Remove from tracker
        remove_from_tracker(filename)

        # Remove physical file
        file_path = os.path.join(INGESTED_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

        print(f"🗑️ Deleted {deleted} chunks for '{filename}' from Neo4j.")
        return jsonify({"status": "success", "deleted": deleted, "filename": filename})

    except Exception as e:
        print(f"❌ Delete error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# =========================================================
# KNOWLEDGE GRAPH EXPLORER ENDPOINTS
# =========================================================

def _get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


@app.route("/kg/stats")
def kg_stats():
    """Get knowledge graph statistics"""
    try:
        driver = _get_driver()
        with driver.session() as session:
            total = session.run("MATCH (c:Chunk) RETURN count(c) AS cnt").single()["cnt"]
            sources = session.run("MATCH (c:Chunk) RETURN count(DISTINCT c.source) AS cnt").single()["cnt"]
            by_type = session.run(
                "MATCH (c:Chunk) "
                "RETURN coalesce(c.media_type, 'document') AS mtype, count(c) AS cnt "
                "ORDER BY cnt DESC"
            )
            type_counts = {r["mtype"]: r["cnt"] for r in by_type}
        driver.close()
        return jsonify({
            "total_chunks": total,
            "unique_sources": sources,
            "by_type": type_counts,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/kg/documents")
def kg_documents():
    """List all unique sources with chunk counts and metadata"""
    try:
        driver = _get_driver()
        with driver.session() as session:
            rows = session.run(
                "MATCH (c:Chunk) "
                "RETURN c.source AS source, "
                "       count(c) AS chunks, "
                "       head(collect(c.title)) AS title, "
                "       head(collect(c.doc_type)) AS doc_type, "
                "       head(collect(c.year)) AS year, "
                "       head(collect(c.author)) AS author, "
                "       head(collect(coalesce(c.media_type, 'document'))) AS media_type "
                "ORDER BY source"
            )
            docs = []
            for r in rows:
                docs.append({
                    "source": r["source"],
                    "chunks": r["chunks"],
                    "title": r["title"] or "Unknown",
                    "doc_type": r["doc_type"] or "Unknown",
                    "year": r["year"] or "Unknown",
                    "author": r["author"] or "Unknown",
                    "media_type": r["media_type"] or "document",
                })
        driver.close()
        return jsonify({"documents": docs})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/kg/chunks")
def kg_chunks():
    """Get all chunks for a specific source"""
    source = request.args.get("source")
    if not source:
        return jsonify({"status": "error", "message": "source parameter required"}), 400
    try:
        driver = _get_driver()
        with driver.session() as session:
            rows = session.run(
                "MATCH (c:Chunk) WHERE c.source = $source "
                "RETURN elementId(c) AS id, c.text AS text, c.source AS source, "
                "       c.title AS title, c.doc_type AS doc_type, "
                "       c.year AS year, c.author AS author, "
                "       c.section AS section, c.page AS page, "
                "       coalesce(c.media_type, 'document') AS media_type "
                "ORDER BY c.page, elementId(c)",
                source=source
            )
            chunks = []
            for r in rows:
                chunks.append({
                    "id": r["id"],
                    "text": r["text"][:500] if r["text"] else "",
                    "full_text": r["text"] or "",
                    "source": r["source"],
                    "title": r["title"] or "",
                    "doc_type": r["doc_type"] or "",
                    "year": r["year"] or "",
                    "author": r["author"] or "",
                    "section": r["section"] or "",
                    "page": r["page"] if r["page"] else 0,
                    "media_type": r["media_type"],
                })
        driver.close()
        return jsonify({"chunks": chunks, "total": len(chunks)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/kg/search", methods=["POST"])
def kg_search():
    """Semantic search across the knowledge graph"""
    data = request.get_json()
    query = data.get("query", "")
    top_k = data.get("top_k", 5)

    if not query:
        return jsonify({"status": "error", "message": "query is required"}), 400

    try:
        vector_store = Neo4jVector.from_existing_index(
            embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name=NEO4J_INDEX,
            node_label="Chunk",
            text_node_property="text",
            embedding_node_property="embedding",
        )
        results = vector_store.similarity_search_with_score(query, k=top_k)
        hits = []
        for doc, score in results:
            hits.append({
                "text": doc.page_content[:500],
                "full_text": doc.page_content,
                "score": round(float(score), 4),
                "source": doc.metadata.get("source", "Unknown"),
                "title": doc.metadata.get("title", "Unknown"),
                "doc_type": doc.metadata.get("doc_type", "Unknown"),
                "page": doc.metadata.get("page", 0),
            })
        return jsonify({"results": hits, "query": query})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/kg/update", methods=["POST"])
def kg_update():
    """Batch-update metadata fields for all chunks matching a source"""
    data = request.get_json()
    source_filter = data.get("source")
    updates = data.get("updates", {})

    if not source_filter:
        return jsonify({"status": "error", "message": "source is required"}), 400

    allowed_fields = {"source", "title", "doc_type", "year", "author", "section", "media_type"}
    update_fields = {k: v for k, v in updates.items() if k in allowed_fields and v}

    if not update_fields:
        return jsonify({"status": "error", "message": "No valid fields to update"}), 400

    # Map metadata field names → header labels embedded in chunk text
    text_label_map = {
        "source": "SOURCE",
        "title": "TITLE",
        "doc_type": "TYPE",
        "year": "YEAR",
        "author": "AUTHOR",
        "section": "SECTION",
        "media_type": "MEDIA_TYPE",
    }

    try:
        driver = _get_driver()
        with driver.session() as session:
            # 1. Update node properties
            set_clauses = ", ".join([f"c.{k} = ${k}" for k in update_fields])
            cypher = f"MATCH (c:Chunk) WHERE c.source = $source_filter SET {set_clauses} RETURN count(c) AS updated"
            params = {"source_filter": source_filter, **update_fields}
            result = session.run(cypher, params)
            updated = result.single()["updated"]

            # 2. Also update the embedded metadata lines inside c.text
            #    e.g. "TITLE: Cover" → "TITLE: Annual Report 2007"
            for field, new_value in update_fields.items():
                label = text_label_map.get(field)
                if not label:
                    continue
                # Use APOC-free regex replacement via Cypher string functions
                # Match lines like "TITLE: <anything>" and replace with new value
                text_cypher = (
                    "MATCH (c:Chunk) WHERE c.source = $new_source "
                    "AND c.text CONTAINS $old_prefix "
                    "WITH c, "
                    "  substring(c.text, 0, size(c.text)) AS full_text "
                    "WITH c, full_text "
                    "UNWIND split(full_text, '\\n') AS line "
                    "WITH c, collect(CASE "
                    "  WHEN line STARTS WITH $old_prefix THEN $new_line "
                    "  ELSE line "
                    "END) AS lines "
                    "SET c.text = reduce(s = '', l IN lines | s + l + '\\n')"
                )
                # The source may have just been updated, use the new value if present
                current_source = update_fields.get("source", source_filter)
                session.run(text_cypher, {
                    "new_source": current_source,
                    "old_prefix": f"{label}: ",
                    "new_line": f"{label}: {new_value}",
                })

            # 3. Re-embed only the changed chunks
            #    Fetch updated text, compute new embeddings, write back
            current_source = update_fields.get("source", source_filter)
            reembed_rows = session.run(
                "MATCH (c:Chunk) WHERE c.source = $src "
                "RETURN elementId(c) AS nid, c.text AS text",
                src=current_source
            )
            chunks_to_embed = [(r["nid"], r["text"]) for r in reembed_rows]

        driver.close()

        # Compute new embeddings outside the Neo4j session
        if chunks_to_embed:
            print(f"   🔄 Re-embedding {len(chunks_to_embed)} chunks...")
            texts = [t for _, t in chunks_to_embed]
            new_vectors = embeddings.embed_documents(texts)

            driver = _get_driver()
            with driver.session() as session:
                for (nid, _), vec in zip(chunks_to_embed, new_vectors):
                    session.run(
                        "MATCH (c:Chunk) WHERE elementId(c) = $nid "
                        "SET c.embedding = $vec",
                        nid=nid, vec=vec
                    )
            driver.close()
            print(f"   ✅ Re-embedded {len(chunks_to_embed)} chunks successfully.")

        print(f"📝 Updated {updated} chunks for source '{source_filter}': {update_fields} (properties + text + embeddings)")
        return jsonify({"status": "success", "updated": updated, "reembedded": len(chunks_to_embed), "fields": list(update_fields.keys())})
    except Exception as e:
        print(f"❌ Update error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 WYDOT Local Ingestion Server")
    print(f"  NEO4J_URI: {NEO4J_URI}")
    print(f"  NEO4J_INDEX: {NEO4J_INDEX}")
    print(f"  GEMINI_API_KEY: {'SET' if GEMINI_API_KEY else 'NOT SET'}")
    print(f"  INGESTED_DIR: {INGESTED_DIR}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5001, debug=True)
