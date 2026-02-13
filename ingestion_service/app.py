#!/usr/bin/env python3
"""
WYDOT Ingestion Service (unified):
- GET / : Serve upload UI (same interface as local dashboard).
- POST /upload, /transcribe, /library, /delete, /kg/* : Manual ingestion and KG explorer.
- POST / and POST /ingest : Eventarc trigger for GCS incoming/ (when GCS_BUCKET set).
Local: uses SQLite tracker (tracker_db.py). Cloud: same code + GCS for Eventarc.
"""
import os
import re
import gc
import json
import time
import logging
import tempfile
import shutil
import urllib.parse
from datetime import datetime
from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Shared utils
try:
    from utils import telemetry
    from utils import evaluation
    from utils.chat_history_store import get_chat_history_store
except ImportError:
    pass

logging.getLogger("pypdf").setLevel(logging.ERROR)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv()

# Optional GCP logging (only in cloud)
try:
    import google.cloud.logging
    google.cloud.logging.Client().setup_logging()
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core processing
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_core.documents import Document

try:
    from pypdf import PdfReader
    import docx
except ImportError:
    PdfReader = None
    docx = None

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

from neo4j import GraphDatabase

# Tracker: SQLite (local) – same API as JSON for compatibility
from tracker_db import load_tracker, add_to_tracker, remove_from_tracker

# === Config ===
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GCS_BUCKET = os.getenv("GCS_BUCKET")
# Use writable cache for embeddings: avoid /app on local (read-only or missing)
_raw_hf = os.getenv("HF_HOME", "").strip()
HF_HOME = None
if _raw_hf and os.path.isabs(_raw_hf):
    try:
        os.makedirs(_raw_hf, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=_raw_hf, delete=True):
            pass
        HF_HOME = _raw_hf
    except (OSError, PermissionError, FileNotFoundError):
        pass
if HF_HOME is None:
    HF_HOME = os.path.join(os.path.dirname(__file__), "model_cache")
    os.makedirs(HF_HOME, exist_ok=True)

NEO4J_INDEX = os.getenv("NEO4J_INDEX_DEFAULT", "wydot_vector_index")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INGESTED_DIR = os.path.join(os.path.dirname(__file__), "ingested_data")

os.makedirs(INGESTED_DIR, exist_ok=True)
if GEMINI_API_KEY and genai:
    genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

# Embeddings (use local model_cache when HF_HOME not set or not writable)
logger.info("Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    cache_folder=HF_HOME,
)
logger.info("Embeddings model loaded.")

TEXT_EXTENSIONS = {".pdf", ".docx", ".doc"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".webm", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi"}


# ---------- Helpers (Eventarc: simple) ----------
def find_section_header(text):
    m = re.search(r"(SECTION\s+\d+|DIVISION\s+\d+|CHAPTER\s+\d+)", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def process_file(filepath):
    """Used by Eventarc: simple metadata, PDF/DOCX only."""
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()
    docs = []
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(filepath)
            docs = loader.load()
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(filepath)
            docs = loader.load()
        else:
            return []
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
        return []
    year_match = re.search(r"(20\d{2})", filename)
    file_year = year_match.group(1) if year_match else "2024"
    filename_lower = filename.lower()
    doc_type = "Specification" if "specification" in filename_lower else "Annual Report" if "report" in filename_lower else "Memo" if "memo" in filename_lower else "General Document"
    processed = []
    current_section = "General"
    for doc in docs:
        h = find_section_header(doc.page_content)
        if h:
            current_section = h
        page_num = doc.metadata.get("page", 0) + 1
        doc.metadata.update({
            "source": filename,
            "year": file_year,
            "doc_type": doc_type,
            "section": current_section,
            "title": filename,
            "page": int(page_num),
        })
        doc.page_content = (
            f"SOURCE: {filename}\nYEAR: {file_year}\nTYPE: {doc_type}\nSECTION: {current_section}\nPAGE: {page_num}\n--- CONTENT ---\n"
            + doc.page_content
        )
        processed.append(doc)
    return processed


# ---------- Helpers (UI: rich metadata, audio/video) ----------
def get_pdf_internal_metadata(filepath):
    meta = {"author": "Unknown", "title": "Unknown", "created": "Unknown"}
    if not PdfReader:
        return meta
    try:
        reader = PdfReader(filepath)
        info = reader.metadata
        if info:
            meta["author"] = info.get("/Author", "Unknown") or "Unknown"
            meta["title"] = info.get("/Title", "Unknown") or "Unknown"
            raw = info.get("/CreationDate", "")
            if "D:" in raw:
                meta["created"] = raw.split("D:")[1][:4]
    except Exception:
        pass
    return meta


def get_docx_internal_metadata(filepath):
    meta = {"author": "Unknown", "title": "Unknown", "created": "Unknown"}
    if not docx:
        return meta
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
    fl = filename.lower()
    doc_type = "Specification" if "specification" in fl else "Annual Report" if "report" in fl else "Memo" if "memo" in fl else "General Document"
    return year, doc_type


def process_text_file(filepath, filename):
    """PDF/DOCX with rich metadata for UI."""
    ext = os.path.splitext(filename)[1].lower()
    internal_meta = get_pdf_internal_metadata(filepath) if ext == ".pdf" else get_docx_internal_metadata(filepath) if ext in [".docx", ".doc"] else {}
    if ext == ".pdf":
        loader = PyPDFLoader(filepath)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(filepath)
    else:
        return []
    try:
        docs = loader.load()
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
        return []
    file_year, file_type = get_filename_metadata(filename)
    final_year = internal_meta.get("created", "Unknown") if internal_meta.get("created") != "Unknown" else file_year
    if final_year == 0:
        final_year = "Unknown"
    processed = []
    current_section = "General"
    for doc_item in docs:
        h = find_section_header(doc_item.page_content)
        if h:
            current_section = h
        page_num = doc_item.metadata.get("page", 0) + 1
        doc_item.metadata.update({
            "source": str(filename),
            "year": str(final_year),
            "doc_type": str(file_type),
            "section": str(current_section),
            "author": str(internal_meta.get("author", "Unknown")),
            "title": str(internal_meta.get("title", "Unknown")),
            "page": int(page_num),
            "media_type": "document",
        })
        header = (
            f"SOURCE: {filename}\nTITLE: {internal_meta.get('title', 'Unknown')}\nAUTHOR: {internal_meta.get('author', 'Unknown')}\n"
            f"YEAR: {final_year}\nTYPE: {file_type}\nSECTION: {current_section}\nPAGE: {page_num}\n--- CONTENT ---\n"
        )
        doc_item.page_content = header + doc_item.page_content
        processed.append(doc_item)
    return processed


def transcribe_media_with_gemini(filepath, filename, media_type):
    if not GEMINI_API_KEY or not genai:
        raise ValueError("GEMINI_API_KEY is not set — cannot transcribe media")
    mime_map = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4", ".ogg": "audio/ogg",
                ".mp4": "video/mp4", ".mov": "video/quicktime", ".avi": "video/x-msvideo"}
    ext = os.path.splitext(filename)[1].lower()
    mime = "video/webm" if ext == ".webm" and media_type == "video" else "audio/webm" if ext == ".webm" else mime_map.get(ext, f"{media_type}/{ext.lstrip('.')}")
    uploaded_file = genai.upload_file(filepath, display_name=filename, mime_type=mime)
    for _ in range(40):
        st = genai.get_file(uploaded_file.name).state.name
        if st == "ACTIVE":
            break
        if st == "FAILED":
            raise ValueError(f"Gemini file processing failed for {filename}")
        time.sleep(3)
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = "Transcribe this " + media_type + " file completely. After the transcription add: TOPIC: ... KEY_POINTS: ... SPEAKERS: ... DATE_MENTIONED: ..."
    response = model.generate_content([uploaded_file, prompt])
    try:
        genai.delete_file(uploaded_file.name)
    except Exception:
        pass
    return response.text


def extract_transcript_metadata(transcript):
    meta = {"topic": "Unknown", "speakers": "Unknown", "date_mentioned": "Unknown", "key_points": ""}
    for name, pat in [("topic", r"TOPIC:\s*(.+)"), ("speakers", r"SPEAKERS:\s*(.+)"), ("date_mentioned", r"DATE_MENTIONED:\s*(.+)")]:
        m = re.search(pat, transcript)
        if m:
            meta[name] = m.group(1).strip()
    return meta


def process_media_file(filepath, filename, media_type):
    transcript = transcribe_media_with_gemini(filepath, filename, media_type)
    meta = extract_transcript_metadata(transcript)
    file_year, _ = get_filename_metadata(filename)
    if file_year == 0:
        file_year = datetime.now().year
    header = (
        f"SOURCE: {filename}\nTITLE: {meta['topic']}\nMEDIA_TYPE: {media_type}\nSPEAKERS: {meta['speakers']}\n"
        f"DATE_MENTIONED: {meta['date_mentioned']}\nYEAR: {file_year}\nTYPE: {media_type.capitalize()} Transcript\n--- CONTENT ---\n"
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
        },
    )
    return [doc], transcript, meta


def ingest_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
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
    return len(chunks)


def save_ingested_file(filepath, filename):
    dest = os.path.join(INGESTED_DIR, filename)
    shutil.copy2(filepath, dest)


def _get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


# ---------- Routes ----------
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
            docs = process_text_file(tmp_path, filename)
            media_type = "document"
            meta = None
        elif ext in AUDIO_EXTENSIONS:
            docs, _, meta = process_media_file(tmp_path, filename, "audio")
            media_type = "audio"
        elif ext in VIDEO_EXTENSIONS:
            docs, _, meta = process_media_file(tmp_path, filename, "video")
            media_type = "video"
        else:
            return jsonify({"status": "error", "message": f"Unsupported: {ext}"}), 400
        if not docs:
            return jsonify({"status": "error", "message": f"No content from {filename}"}), 400
        chunk_count = ingest_documents(docs)
        save_ingested_file(tmp_path, filename)
        add_to_tracker(filename, media_type, chunk_count, meta if media_type in ("audio", "video") else None)
        del docs
        gc.collect()
        return jsonify({"status": "success", "message": f"Ingested {filename}", "chunks": chunk_count, "filename": filename, "index": NEO4J_INDEX})
    except Exception as e:
        logger.exception("Upload failed")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.route("/transcribe", methods=["POST"])
def transcribe():
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
        return jsonify({"status": "success", "transcript": transcript, "metadata": meta})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.route("/library")
def library():
    return jsonify(load_tracker())


@app.route("/delete", methods=["POST"])
def delete():
    data = request.get_json() or {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"status": "error", "message": "No filename provided"}), 400
    try:
        driver = _get_driver()
        with driver.session() as session:
            r = session.run(
                "MATCH (c:Chunk) WHERE c.source = $source WITH c, count(c) AS cnt DETACH DELETE c RETURN cnt",
                source=filename,
            )
            rec = r.single()
            deleted = rec["cnt"] if rec else 0
        driver.close()
        remove_from_tracker(filename)
        fp = os.path.join(INGESTED_DIR, filename)
        if os.path.exists(fp):
            os.remove(fp)
        return jsonify({"status": "success", "deleted": deleted, "filename": filename})
    except Exception as e:
        logger.exception("Delete failed")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/kg/stats")
def kg_stats():
    try:
        driver = _get_driver()
        with driver.session() as session:
            total = session.run("MATCH (c:Chunk) RETURN count(c) AS cnt").single()["cnt"]
            sources = session.run("MATCH (c:Chunk) RETURN count(DISTINCT c.source) AS cnt").single()["cnt"]
            by_type = {r["mtype"]: r["cnt"] for r in session.run("MATCH (c:Chunk) RETURN coalesce(c.media_type, 'document') AS mtype, count(c) AS cnt")}
        driver.close()
        return jsonify({"total_chunks": total, "unique_sources": sources, "by_type": by_type})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/kg/documents")
def kg_documents():
    try:
        driver = _get_driver()
        with driver.session() as session:
            rows = session.run(
                "MATCH (c:Chunk) RETURN c.source AS source, count(c) AS chunks, "
                "head(collect(c.title)) AS title, head(collect(c.doc_type)) AS doc_type, "
                "head(collect(c.year)) AS year, head(collect(c.author)) AS author, "
                "head(collect(coalesce(c.media_type, 'document'))) AS media_type ORDER BY source"
            )
            docs = [{"source": r["source"], "chunks": r["chunks"], "title": r["title"] or "Unknown", "doc_type": r["doc_type"] or "Unknown", "year": r["year"] or "Unknown", "author": r["author"] or "Unknown", "media_type": r["media_type"] or "document"} for r in rows]
        driver.close()
        return jsonify({"documents": docs})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/kg/chunks")
def kg_chunks():
    source = request.args.get("source")
    if not source:
        return jsonify({"status": "error", "message": "source parameter required"}), 400
    try:
        driver = _get_driver()
        with driver.session() as session:
            # Neo4j 5: elementId(c); older: id(c)
            try:
                rows = list(session.run(
                    "MATCH (c:Chunk) WHERE c.source = $source RETURN elementId(c) AS id, c.text AS text, c.source AS source, c.title AS title, c.doc_type AS doc_type, c.year AS year, c.author AS author, c.section AS section, c.page AS page, coalesce(c.media_type, 'document') AS media_type ORDER BY c.page, elementId(c)",
                    source=source,
                ))
            except Exception:
                rows = list(session.run(
                    "MATCH (c:Chunk) WHERE c.source = $source RETURN toString(id(c)) AS id, c.text AS text, c.source AS source, c.title AS title, c.doc_type AS doc_type, c.year AS year, c.author AS author, c.section AS section, c.page AS page, coalesce(c.media_type, 'document') AS media_type ORDER BY c.page, id(c)",
                    source=source,
                ))
        driver.close()
        chunks = [{"id": r["id"], "text": (r["text"] or "")[:500], "source": r["source"], "title": r["title"] or "", "doc_type": r["doc_type"] or "", "year": r["year"] or "", "author": r["author"] or "", "section": r["section"] or "", "page": r["page"] if r["page"] is not None else 0, "media_type": r["media_type"]} for r in rows]
        return jsonify({"chunks": chunks, "total": len(chunks)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/kg/search", methods=["POST"])
def kg_search():
    data = request.get_json() or {}
    query = data.get("query", "")
    top_k = int(data.get("top_k", 5))
    if not query:
        return jsonify({"status": "error", "message": "query is required"}), 400
    try:
        vs = Neo4jVector.from_existing_index(
            embeddings, url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
            index_name=NEO4J_INDEX, node_label="Chunk", text_node_property="text", embedding_node_property="embedding",
        )
        results = vs.similarity_search_with_score(query, k=top_k)
        hits = [{"text": d.page_content[:500], "full_text": d.page_content, "score": round(float(sc), 4), "source": d.metadata.get("source", "Unknown"), "title": d.metadata.get("title", "Unknown"), "doc_type": d.metadata.get("doc_type", "Unknown"), "page": d.metadata.get("page", 0)} for d, sc in results]
        return jsonify({"results": hits, "query": query})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/kg/update", methods=["POST"])
def kg_update():
    data = request.get_json() or {}
    source_filter = data.get("source")
    updates = {k: v for k, v in (data.get("updates") or {}).items() if k in {"source", "title", "doc_type", "year", "author", "section", "media_type"} and v}
    if not source_filter or not updates:
        return jsonify({"status": "error", "message": "source and updates required"}), 400
    try:
        driver = _get_driver()
        with driver.session() as session:
            set_clauses = ", ".join([f"c.{k} = ${k}" for k in updates])
            session.run(f"MATCH (c:Chunk) WHERE c.source = $source_filter SET {set_clauses} RETURN count(c) AS updated", {"source_filter": source_filter, **updates})
            reembed_rows = list(session.run("MATCH (c:Chunk) WHERE c.source = $src RETURN elementId(c) AS nid, c.text AS text", src=updates.get("source", source_filter)))
        driver.close()
        if reembed_rows:
            texts = [r["text"] for r in reembed_rows]
            vecs = embeddings.embed_documents(texts)
            driver2 = _get_driver()
            with driver2.session() as session2:
                for (r, vec) in zip(reembed_rows, vecs):
                    try:
                        session2.run("MATCH (c:Chunk) WHERE elementId(c) = $nid SET c.embedding = $vec", nid=r["nid"], vec=vec)
                    except Exception:
                        session2.run("MATCH (c:Chunk) WHERE id(c) = $nid SET c.embedding = $vec", nid=int(r["nid"]), vec=vec)
            driver2.close()
        return jsonify({"status": "success", "updated": len(reembed_rows), "reembedded": len(reembed_rows), "fields": list(updates.keys())})
    except Exception as e:
        logger.exception("kg/update failed")
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------- Eventarc: GCS trigger (only when GCS_BUCKET set) ----------
@app.route("/ingest", methods=["POST"])
@app.route("/", methods=["POST"])
def ingest_event():
    if not GCS_BUCKET:
        return jsonify({"message": "Local mode: use /upload for file ingestion"}), 200
    envelope = request.get_json()
    if not envelope:
        return "Bad Request: No JSON", 400
    if "message" in envelope:
        try:
            import base64
            data = json.loads(base64.b64decode(envelope["message"]["data"]).decode("utf-8"))
        except Exception:
            return "Bad Request: Invalid Pub/Sub message", 400
    else:
        data = envelope
    bucket_name = data.get("bucket")
    file_path = data.get("name")
    if not file_path or not bucket_name or not file_path.startswith("incoming/"):
        return "Ignored", 200
    try:
        decoded_path = urllib.parse.unquote(file_path)
    except Exception:
        decoded_path = file_path
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        if not blob.exists():
            blob = bucket.blob(decoded_path)
        if not blob.exists():
            return "Error: Blob not found", 200
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_path)[1]) as tmp:
            blob.download_to_filename(tmp.name)
            tmp_path = tmp.name
        try:
            docs = process_file(tmp_path)
            if not docs:
                bucket.rename_blob(blob, file_path.replace("incoming/", "failed/"))
                return "Processed (Empty)", 200
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(docs)
            Neo4jVector.from_documents(
                chunks, embeddings, url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name=NEO4J_INDEX, node_label="Chunk", text_node_property="text", embedding_node_property="embedding",
            )
            bucket.rename_blob(blob, file_path.replace("incoming/", "processed/"))
            add_to_tracker(os.path.basename(file_path), "document", len(chunks))
            return "Ingestion Complete", 200
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except ImportError:
        return jsonify({"message": "google-cloud-storage not installed"}), 200
    except Exception as e:
        logger.exception("Ingestion failed")
        try:
            from google.cloud import storage
            bucket = storage.Client().bucket(bucket_name)
            blob = bucket.blob(file_path)
            bucket.rename_blob(blob, file_path.replace("incoming/", "failed/"))
        except Exception:
            pass
        return f"Error: {e}", 500



# ---------- Monitoring & Evaluation Routes ----------

@app.route("/api/monitoring/metrics")
def api_monitoring_metrics():
    try:
        if "telemetry" not in globals(): return jsonify({})
        return jsonify(telemetry.get_latency_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/monitoring/timeseries")
def api_monitoring_timeseries():
    try:
        if "telemetry" not in globals(): return jsonify([])
        return jsonify(telemetry.get_timeseries())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/monitoring/recent")
def api_monitoring_recent():
    try:
        if "telemetry" not in globals(): return jsonify([])
        return jsonify(telemetry.get_recent_metrics(limit=20))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/monitoring/models")
def api_monitoring_models():
    try:
        if "telemetry" not in globals(): return jsonify([])
        return jsonify(telemetry.get_model_comparison())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Evaluation Helpers (Re-implementing RAG for offline eval)
def eval_search_func(query, index_name="All Documents"):
    try:
        vs = Neo4jVector.from_existing_index(
            embeddings, url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
            index_name=NEO4J_INDEX, node_label="Chunk", text_node_property="text", embedding_node_property="embedding",
        )
        docs = vs.similarity_search(query, k=3)
        context = "\n\n".join([f"SOURCE: {d.metadata.get('source')}\n{d.page_content}" for d in docs])
        return context, [{"source": d.metadata.get("source")} for d in docs]
    except Exception as e:
        logger.error(f"Eval search failed: {e}")
        return "", []

def eval_generate_func(query, context, history):
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return "Gemini not available"
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""You are a helpful assistant.
CONTEXT:
{context}

QUESTION: {query}

Answer based on the context.
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Eval generate failed: {e}")
        return "Error generating answer"

@app.route("/api/evaluation/offline/run", methods=["POST"])
def api_eval_offline_run():
    try:
        if "evaluation" not in globals(): return jsonify({"error": "Evaluation module not loaded"}), 500
        # Check if we can run eval (needs params)
        if not GEMINI_AVAILABLE:
            return jsonify({"error": "LLM not available for evaluation"}), 400
        
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If there's a running loop (shouldn't be in Flask), use thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(asyncio.run, evaluation.run_offline_evaluation(eval_search_func, eval_generate_func)).result()
            else:
                result = loop.run_until_complete(evaluation.run_offline_evaluation(eval_search_func, eval_generate_func))
        except RuntimeError:
            result = asyncio.run(evaluation.run_offline_evaluation(eval_search_func, eval_generate_func))
        
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/evaluation/offline/results")
def api_eval_offline_results():
    try:
        if "evaluation" not in globals(): return jsonify({})
        res = evaluation.get_latest_offline_result()
        return jsonify(res or {})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/evaluation/online/scores")
def api_eval_online_scores():
    try:
        if "evaluation" not in globals(): return jsonify({})
        return jsonify(evaluation.get_online_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/evaluation/online/history")
def api_eval_online_history():
    try:
        if "evaluation" not in globals(): return jsonify([])
        return jsonify(evaluation.get_online_trends())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/evaluation/feedback")
def api_eval_feedback():
    try:
        store = get_chat_history_store()
        stats = store.get_feedback_stats()
        recent = store.get_recent_feedback(limit=10)
        return jsonify({"stats": stats, "recent": recent})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("WYDOT Ingestion (unified): GET / = UI, POST /upload = ingest, POST /ingest = Eventarc")
    print(f"  NEO4J_URI: {NEO4J_URI}")
    print(f"  GCS_BUCKET: {GCS_BUCKET or '(not set — local mode)'}")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
