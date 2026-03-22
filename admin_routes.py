#!/usr/bin/env python3
"""
Admin routes as a FastAPI sub-app.
Mounted at /admin/* in the unified Chainlit service.
Replaces the standalone Flask ingestion_service/app.py.

All original Flask routes are preserved as FastAPI equivalents:
  /admin/              → Upload UI
  /admin/health        → Health check
  /admin/upload        → File upload + ingest
  /admin/transcribe    → Audio/video transcription only
  /admin/library       → List ingested files
  /admin/delete        → Remove file from graph
  /admin/kg/stats      → KG statistics
  /admin/kg/documents  → List documents
  /admin/kg/chunks     → Get chunks for a document
  /admin/kg/search     → Vector similarity search
  /admin/kg/update     → Update chunk metadata
  /admin/ingest        → Eventarc GCS trigger
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
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from dotenv import load_dotenv

logger = logging.getLogger("wydot.admin")

# ── Config (shared with chatbot via same .env) ──
NEO4J_URI = os.getenv("NEO4J_URI") or os.getenv("NEO4J_URI_GEMINI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USERNAME_GEMINI")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD_GEMINI")
NEO4J_INDEX = os.getenv("NEO4J_INDEX_DEFAULT", "wydot_vector_index")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GCS_BUCKET = os.getenv("GCS_BUCKET")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "wydot-admin-2025")

if os.getenv("K_SERVICE"):
    INGESTED_DIR = "/tmp/ingested_data"
else:
    INGESTED_DIR = os.path.join(os.path.dirname(__file__), "ingestion_service", "ingested_data")
os.makedirs(INGESTED_DIR, exist_ok=True)

# HuggingFace model cache
_raw_hf = os.getenv("HF_HOME", "").strip()
HF_HOME = None
if _raw_hf and os.path.isabs(_raw_hf) and os.path.isdir(_raw_hf):
    HF_HOME = _raw_hf
if HF_HOME is None:
    HF_HOME = "/tmp/model_cache" if os.getenv("K_SERVICE") else os.path.join(os.path.dirname(__file__), "ingestion_service", "model_cache")
    os.makedirs(HF_HOME, exist_ok=True)

TEXT_EXTENSIONS = {".pdf", ".docx", ".doc"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".webm", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi"}

# ── Lazy-loaded heavy deps ──
_embeddings = None
_genai = None
_tracker_loaded = False
_tracker_funcs = {}


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        logger.info("Loading sentence-transformer model...")
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=HF_HOME,
        )
        logger.info("Embeddings model loaded.")
    return _embeddings


def _get_genai():
    global _genai
    if _genai is None:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _genai = genai
    return _genai


def _get_tracker():
    """Load tracker functions from ingestion_service/tracker_db.py if available, else use stubs."""
    global _tracker_loaded, _tracker_funcs
    if not _tracker_loaded:
        try:
            import sys
            svc_dir = os.path.join(os.path.dirname(__file__), "ingestion_service")
            if svc_dir not in sys.path:
                sys.path.insert(0, svc_dir)
            from tracker_db import load_tracker, add_to_tracker, remove_from_tracker
            _tracker_funcs = {
                "load": load_tracker,
                "add": add_to_tracker,
                "remove": remove_from_tracker,
            }
        except ImportError:
            logger.warning("tracker_db.py not found; tracker disabled")
            _tracker_funcs = {
                "load": lambda: [],
                "add": lambda *a, **kw: None,
                "remove": lambda *a, **kw: None,
            }
        _tracker_loaded = True
    return _tracker_funcs


def _get_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


# ── Document Processing (ported from ingestion_service/app.py) ──

def find_section_header(text):
    m = re.search(r"(SECTION\s+\d+|DIVISION\s+\d+|CHAPTER\s+\d+)", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def get_filename_metadata(filename):
    year_match = re.search(r"(20\d{2})", filename)
    year = int(year_match.group(1)) if year_match else 0
    fl = filename.lower()
    doc_type = (
        "Specification" if "specification" in fl
        else "Annual Report" if "report" in fl
        else "Memo" if "memo" in fl
        else "General Document"
    )
    return year, doc_type


def get_pdf_internal_metadata(filepath):
    meta = {"author": "Unknown", "title": "Unknown", "created": "Unknown"}
    try:
        from pypdf import PdfReader
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
    try:
        import docx
        doc_obj = docx.Document(filepath)
        props = doc_obj.core_properties
        meta["author"] = props.author or "Unknown"
        meta["title"] = props.title or "Unknown"
        if props.created:
            meta["created"] = props.created.year
    except Exception:
        pass
    return meta


def process_text_file(filepath, filename):
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
    from langchain_core.documents import Document
    ext = os.path.splitext(filename)[1].lower()
    internal_meta = (
        get_pdf_internal_metadata(filepath) if ext == ".pdf"
        else get_docx_internal_metadata(filepath) if ext in [".docx", ".doc"]
        else {}
    )
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
            f"SOURCE: {filename}\nTITLE: {internal_meta.get('title', 'Unknown')}\n"
            f"AUTHOR: {internal_meta.get('author', 'Unknown')}\n"
            f"YEAR: {final_year}\nTYPE: {file_type}\nSECTION: {current_section}\n"
            f"PAGE: {page_num}\n--- CONTENT ---\n"
        )
        doc_item.page_content = header + doc_item.page_content
        processed.append(doc_item)
    return processed


def transcribe_media_with_gemini(filepath, filename, media_type):
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")
    genai = _get_genai()
    mime_map = {
        ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
        ".ogg": "audio/ogg", ".mp4": "video/mp4", ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
    }
    ext = os.path.splitext(filename)[1].lower()
    mime = (
        "video/webm" if ext == ".webm" and media_type == "video"
        else "audio/webm" if ext == ".webm"
        else mime_map.get(ext, f"{media_type}/{ext.lstrip('.')}")
    )
    uploaded_file = genai.upload_file(filepath, display_name=filename, mime_type=mime)
    for _ in range(40):
        st = genai.get_file(uploaded_file.name).state.name
        if st == "ACTIVE":
            break
        if st == "FAILED":
            raise ValueError(f"Gemini file processing failed for {filename}")
        time.sleep(3)
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = (
        f"Transcribe this {media_type} file completely. After the transcription add: "
        "TOPIC: ... KEY_POINTS: ... SPEAKERS: ... DATE_MENTIONED: ..."
    )
    response = model.generate_content([uploaded_file, prompt])
    try:
        genai.delete_file(uploaded_file.name)
    except Exception:
        pass
    return response.text


def extract_transcript_metadata(transcript):
    meta = {"topic": "Unknown", "speakers": "Unknown", "date_mentioned": "Unknown", "key_points": ""}
    for name, pat in [
        ("topic", r"TOPIC:\s*(.+)"),
        ("speakers", r"SPEAKERS:\s*(.+)"),
        ("date_mentioned", r"DATE_MENTIONED:\s*(.+)"),
    ]:
        m = re.search(pat, transcript)
        if m:
            meta[name] = m.group(1).strip()
    return meta


def process_media_file(filepath, filename, media_type):
    from langchain_core.documents import Document
    transcript = transcribe_media_with_gemini(filepath, filename, media_type)
    meta = extract_transcript_metadata(transcript)
    file_year, _ = get_filename_metadata(filename)
    if file_year == 0:
        file_year = datetime.now().year
    header = (
        f"SOURCE: {filename}\nTITLE: {meta['topic']}\nMEDIA_TYPE: {media_type}\n"
        f"SPEAKERS: {meta['speakers']}\nDATE_MENTIONED: {meta['date_mentioned']}\n"
        f"YEAR: {file_year}\nTYPE: {media_type.capitalize()} Transcript\n--- CONTENT ---\n"
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
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_neo4j import Neo4jVector
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    Neo4jVector.from_documents(
        chunks,
        _get_embeddings(),
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=NEO4J_INDEX,
        node_label="Chunk",
        text_node_property="text",
        embedding_node_property="embedding",
    )
    return len(chunks)


# ── FastAPI Admin App ──

admin_app = FastAPI(title="WYDOT Admin", docs_url="/docs", redoc_url=None)


# Simple admin auth middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse


class AdminAuthMiddleware(BaseHTTPMiddleware):
    """
    Checks for admin auth on all routes except /health and /ingest (Eventarc).
    Auth via:
      - Cookie: admin_token=<ADMIN_PASSWORD>
      - Header: X-Admin-Token: <ADMIN_PASSWORD>
      - Query param: ?token=<ADMIN_PASSWORD>
    Login page served at /login if not authenticated.
    """
    OPEN_PATHS = {"/health", "/ingest", "/login", "/login/submit"}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Strip /admin prefix since this middleware runs inside the mounted app
        # The path here is relative to the mount point
        if path in self.OPEN_PATHS or path.startswith("/static"):
            return await call_next(request)

        token = (
            request.cookies.get("admin_token")
            or request.headers.get("x-admin-token")
            or request.query_params.get("token")
        )
        if token == ADMIN_PASSWORD:
            return await call_next(request)

        # Redirect to login
        from starlette.responses import RedirectResponse
        return RedirectResponse(url="/admin/login")


admin_app.add_middleware(AdminAuthMiddleware)


# ── Routes ──

@admin_app.get("/login", response_class=HTMLResponse)
async def admin_login_page():
    return """<!DOCTYPE html>
<html><head><title>WYDOT Admin Login</title>
<style>
  body { font-family: system-ui; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; background: #f0f2f5; }
  .card { background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); width: 360px; }
  h2 { margin-top: 0; color: #1a365d; }
  input { width: 100%; padding: 10px; margin: 8px 0; border: 1px solid #ddd; border-radius: 6px; box-sizing: border-box; font-size: 14px; }
  button { width: 100%; padding: 10px; background: #2563eb; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; }
  button:hover { background: #1d4ed8; }
  .error { color: #dc2626; font-size: 13px; display: none; }
</style></head>
<body>
<div class="card">
  <h2>WYDOT Admin</h2>
  <p style="color:#666;font-size:13px;">Document ingestion & knowledge graph management</p>
  <form method="POST" action="/admin/login/submit">
    <input type="password" name="password" placeholder="Admin password" required autofocus />
    <button type="submit">Sign In</button>
  </form>
</div>
</body></html>"""


@admin_app.post("/login/submit")
async def admin_login_submit(request: Request):
    form = await request.form()
    password = form.get("password", "")
    if password == ADMIN_PASSWORD:
        from starlette.responses import RedirectResponse
        response = RedirectResponse(url="/admin/", status_code=303)
        response.set_cookie("admin_token", ADMIN_PASSWORD, httponly=True, max_age=86400)
        return response
    return HTMLResponse(
        '<html><body><p style="color:red">Invalid password.</p>'
        '<a href="/admin/login">Try again</a></body></html>',
        status_code=401,
    )


@admin_app.get("/", response_class=HTMLResponse)
async def admin_index():
    """Serve the admin upload UI."""
    template_path = os.path.join(os.path.dirname(__file__), "ingestion_service", "templates", "upload.html")
    if os.path.exists(template_path):
        with open(template_path) as f:
            html = f.read()
        # Fix relative paths in the template to work under /admin/
        html = html.replace('action="/upload"', 'action="/admin/upload"')
        html = html.replace('action="/transcribe"', 'action="/admin/transcribe"')
        html = html.replace('action="/delete"', 'action="/admin/delete"')
        html = html.replace("'/upload'", "'/admin/upload'")
        html = html.replace("'/transcribe'", "'/admin/transcribe'")
        html = html.replace("'/delete'", "'/admin/delete'")
        html = html.replace("'/library'", "'/admin/library'")
        html = html.replace("'/kg/", "'/admin/kg/")
        html = html.replace('"/library"', '"/admin/library"')
        html = html.replace('"/kg/', '"/admin/kg/')
        html = html.replace('"/api/monitoring/', '"/admin/api/monitoring/')
        html = html.replace("'/api/monitoring/", "'/admin/api/monitoring/")
        html = html.replace('"/api/evaluation/', '"/admin/api/evaluation/')
        html = html.replace("'/api/evaluation/", "'/admin/api/evaluation/")
        return HTMLResponse(html)
    return HTMLResponse("<h1>WYDOT Admin</h1><p>Upload template not found. Place ingestion_service/templates/upload.html</p>")


@admin_app.get("/health")
async def admin_health():
    return {"status": "healthy", "service": "admin", "timestamp": time.time()}


@admin_app.post("/upload")
async def admin_upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file selected")

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
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
            raise HTTPException(400, f"Unsupported file type: {ext}")

        if not docs:
            raise HTTPException(400, f"No content extracted from {filename}")

        chunk_count = ingest_documents(docs)
        shutil.copy2(tmp_path, os.path.join(INGESTED_DIR, filename))

        tracker = _get_tracker()
        tracker["add"](filename, media_type, chunk_count, meta if media_type in ("audio", "video") else None)

        del docs
        gc.collect()

        return {"status": "success", "message": f"Ingested {filename}", "chunks": chunk_count, "filename": filename, "index": NEO4J_INDEX}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(500, str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@admin_app.post("/transcribe")
async def admin_transcribe(file: UploadFile = File(...)):
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in AUDIO_EXTENSIONS and ext not in VIDEO_EXTENSIONS:
        raise HTTPException(400, "Not an audio/video file")
    media_type = "audio" if ext in AUDIO_EXTENSIONS else "video"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        transcript = transcribe_media_with_gemini(tmp_path, filename, media_type)
        meta = extract_transcript_metadata(transcript)
        return {"status": "success", "transcript": transcript, "metadata": meta}
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@admin_app.get("/library")
async def admin_library():
    tracker = _get_tracker()
    return tracker["load"]()


@admin_app.post("/delete")
async def admin_delete(request: Request):
    data = await request.json()
    filename = data.get("filename")
    if not filename:
        raise HTTPException(400, "No filename provided")
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
        tracker = _get_tracker()
        tracker["remove"](filename)
        fp = os.path.join(INGESTED_DIR, filename)
        if os.path.exists(fp):
            os.remove(fp)
        return {"status": "success", "deleted": deleted, "filename": filename}
    except Exception as e:
        logger.exception("Delete failed")
        raise HTTPException(500, str(e))


@admin_app.get("/kg/stats")
async def admin_kg_stats():
    try:
        driver = _get_driver()
        with driver.session() as session:
            total = session.run("MATCH (c:Chunk) RETURN count(c) AS cnt").single()["cnt"]
            sources = session.run("MATCH (c:Chunk) RETURN count(DISTINCT c.source) AS cnt").single()["cnt"]
            by_type = {
                r["mtype"]: r["cnt"]
                for r in session.run(
                    "MATCH (c:Chunk) RETURN coalesce(c.media_type, 'document') AS mtype, count(c) AS cnt"
                )
            }
        driver.close()
        return {"total_chunks": total, "unique_sources": sources, "by_type": by_type}
    except Exception as e:
        raise HTTPException(500, str(e))


@admin_app.get("/kg/documents")
async def admin_kg_documents():
    try:
        driver = _get_driver()
        with driver.session() as session:
            rows = session.run(
                "MATCH (c:Chunk) RETURN c.source AS source, count(c) AS chunks, "
                "head(collect(c.title)) AS title, head(collect(c.doc_type)) AS doc_type, "
                "head(collect(c.year)) AS year, head(collect(c.author)) AS author, "
                "head(collect(coalesce(c.media_type, 'document'))) AS media_type ORDER BY source"
            )
            docs = [
                {
                    "source": r["source"], "chunks": r["chunks"],
                    "title": r["title"] or "Unknown", "doc_type": r["doc_type"] or "Unknown",
                    "year": r["year"] or "Unknown", "author": r["author"] or "Unknown",
                    "media_type": r["media_type"] or "document",
                }
                for r in rows
            ]
        driver.close()
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(500, str(e))


@admin_app.get("/kg/chunks")
async def admin_kg_chunks(source: str):
    try:
        driver = _get_driver()
        with driver.session() as session:
            try:
                rows = list(session.run(
                    "MATCH (c:Chunk) WHERE c.source = $source RETURN elementId(c) AS id, "
                    "c.text AS text, c.source AS source, c.title AS title, c.doc_type AS doc_type, "
                    "c.year AS year, c.author AS author, c.section AS section, c.page AS page, "
                    "coalesce(c.media_type, 'document') AS media_type ORDER BY c.page, elementId(c)",
                    source=source,
                ))
            except Exception:
                rows = list(session.run(
                    "MATCH (c:Chunk) WHERE c.source = $source RETURN toString(id(c)) AS id, "
                    "c.text AS text, c.source AS source, c.title AS title, c.doc_type AS doc_type, "
                    "c.year AS year, c.author AS author, c.section AS section, c.page AS page, "
                    "coalesce(c.media_type, 'document') AS media_type ORDER BY c.page, id(c)",
                    source=source,
                ))
        driver.close()
        chunks = [
            {
                "id": r["id"], "text": (r["text"] or "")[:500], "source": r["source"],
                "title": r["title"] or "", "doc_type": r["doc_type"] or "",
                "year": r["year"] or "", "author": r["author"] or "",
                "section": r["section"] or "", "page": r["page"] if r["page"] is not None else 0,
                "media_type": r["media_type"],
            }
            for r in rows
        ]
        return {"chunks": chunks, "total": len(chunks)}
    except Exception as e:
        raise HTTPException(500, str(e))


@admin_app.post("/kg/search")
async def admin_kg_search(request: Request):
    data = await request.json()
    query = data.get("query", "")
    top_k = int(data.get("top_k", 5))
    if not query:
        raise HTTPException(400, "query is required")
    try:
        from langchain_neo4j import Neo4jVector
        vs = Neo4jVector.from_existing_index(
            _get_embeddings(), url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
            index_name=NEO4J_INDEX, node_label="Chunk", text_node_property="text",
            embedding_node_property="embedding",
        )
        results = vs.similarity_search_with_score(query, k=top_k)
        hits = [
            {
                "text": d.page_content[:500], "full_text": d.page_content,
                "score": round(float(sc), 4),
                "source": d.metadata.get("source", "Unknown"),
                "title": d.metadata.get("title", "Unknown"),
                "doc_type": d.metadata.get("doc_type", "Unknown"),
                "page": d.metadata.get("page", 0),
            }
            for d, sc in results
        ]
        return {"results": hits, "query": query}
    except Exception as e:
        raise HTTPException(500, str(e))


@admin_app.post("/kg/update")
async def admin_kg_update(request: Request):
    data = await request.json()
    source_filter = data.get("source")
    updates = {
        k: v for k, v in (data.get("updates") or {}).items()
        if k in {"source", "title", "doc_type", "year", "author", "section", "media_type"} and v
    }
    if not source_filter or not updates:
        raise HTTPException(400, "source and updates required")
    try:
        driver = _get_driver()
        with driver.session() as session:
            set_clauses = ", ".join([f"c.{k} = ${k}" for k in updates])
            session.run(
                f"MATCH (c:Chunk) WHERE c.source = $source_filter SET {set_clauses} RETURN count(c) AS updated",
                {"source_filter": source_filter, **updates},
            )
            reembed_rows = list(session.run(
                "MATCH (c:Chunk) WHERE c.source = $src RETURN elementId(c) AS nid, c.text AS text",
                src=updates.get("source", source_filter),
            ))
        driver.close()
        if reembed_rows:
            texts = [r["text"] for r in reembed_rows]
            vecs = _get_embeddings().embed_documents(texts)
            driver2 = _get_driver()
            with driver2.session() as session2:
                for r, vec in zip(reembed_rows, vecs):
                    try:
                        session2.run(
                            "MATCH (c:Chunk) WHERE elementId(c) = $nid SET c.embedding = $vec",
                            nid=r["nid"], vec=vec,
                        )
                    except Exception:
                        session2.run(
                            "MATCH (c:Chunk) WHERE id(c) = $nid SET c.embedding = $vec",
                            nid=int(r["nid"]), vec=vec,
                        )
            driver2.close()
        return {"status": "success", "updated": len(reembed_rows), "reembedded": len(reembed_rows)}
    except Exception as e:
        logger.exception("kg/update failed")
        raise HTTPException(500, str(e))


@admin_app.post("/ingest")
async def admin_ingest_event(request: Request):
    """Eventarc GCS trigger (same logic as standalone service)."""
    if not GCS_BUCKET:
        return {"message": "Local mode: use /admin/upload for file ingestion"}

    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_neo4j import Neo4jVector
    from langchain_core.documents import Document
    import base64

    envelope = await request.json()
    if not envelope:
        raise HTTPException(400, "No JSON body")

    if "message" in envelope:
        try:
            data = json.loads(base64.b64decode(envelope["message"]["data"]).decode("utf-8"))
        except Exception:
            raise HTTPException(400, "Invalid Pub/Sub message")
    else:
        data = envelope

    bucket_name = data.get("bucket")
    file_path = data.get("name")
    if not file_path or not bucket_name or not file_path.startswith("incoming/"):
        return {"message": "Ignored"}

    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        decoded_path = urllib.parse.unquote(file_path)
        if not blob.exists():
            blob = bucket.blob(decoded_path)
        if not blob.exists():
            return {"message": "Blob not found"}

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_path)[1]) as tmp:
            blob.download_to_filename(tmp.name)
            tmp_path = tmp.name

        try:
            # Simple processing for Eventarc
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif ext in [".docx", ".doc"]:
                loader = Docx2txtLoader(tmp_path)
            else:
                bucket.rename_blob(blob, file_path.replace("incoming/", "failed/"))
                return {"message": "Unsupported format"}
            docs = loader.load()
            if not docs:
                bucket.rename_blob(blob, file_path.replace("incoming/", "failed/"))
                return {"message": "Empty"}

            filename = os.path.basename(file_path)
            year_match = re.search(r"(20\d{2})", filename)
            file_year = year_match.group(1) if year_match else "2024"
            current_section = "General"
            for doc in docs:
                h = find_section_header(doc.page_content)
                if h:
                    current_section = h
                page_num = doc.metadata.get("page", 0) + 1
                doc.metadata.update({
                    "source": filename, "year": file_year,
                    "section": current_section, "title": filename, "page": int(page_num),
                })
                doc.page_content = f"SOURCE: {filename}\nYEAR: {file_year}\nSECTION: {current_section}\nPAGE: {page_num}\n--- CONTENT ---\n" + doc.page_content

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(docs)
            Neo4jVector.from_documents(
                chunks, _get_embeddings(), url=NEO4J_URI, username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD, index_name=NEO4J_INDEX, node_label="Chunk",
                text_node_property="text", embedding_node_property="embedding",
            )
            bucket.rename_blob(blob, file_path.replace("incoming/", "processed/"))
            tracker = _get_tracker()
            tracker["add"](filename, "document", len(chunks))
            return {"message": "Ingestion Complete", "chunks": len(chunks)}
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except ImportError:
        return {"message": "google-cloud-storage not installed"}
    except Exception as e:
        logger.exception("Eventarc ingestion failed")
        try:
            from google.cloud import storage
            bucket = storage.Client().bucket(bucket_name)
            blob = bucket.blob(file_path)
            bucket.rename_blob(blob, file_path.replace("incoming/", "failed/"))
        except Exception:
            pass
        raise HTTPException(500, str(e))


# ── Monitoring & Evaluation routes ──

@admin_app.get("/api/monitoring/metrics")
async def admin_monitoring_metrics():
    try:
        from utils import telemetry
        return telemetry.get_latency_stats()
    except Exception:
        return {}

@admin_app.get("/api/monitoring/timeseries")
async def admin_monitoring_timeseries():
    try:
        from utils import telemetry
        return telemetry.get_timeseries()
    except Exception:
        return []

@admin_app.get("/api/monitoring/recent")
async def admin_monitoring_recent():
    try:
        from utils import telemetry
        return telemetry.get_recent_metrics(limit=20)
    except Exception:
        return []

@admin_app.get("/api/monitoring/models")
async def admin_monitoring_models():
    try:
        from utils import telemetry
        return telemetry.get_model_comparison()
    except Exception:
        return []

@admin_app.get("/api/evaluation/offline/results")
async def admin_eval_offline_results():
    try:
        from utils import evaluation
        res = evaluation.get_latest_offline_result()
        return res or {}
    except Exception:
        return {}

@admin_app.get("/api/evaluation/online/scores")
async def admin_eval_online_scores():
    try:
        from utils import evaluation
        return evaluation.get_online_stats()
    except Exception:
        return {}

@admin_app.get("/api/evaluation/feedback")
async def admin_eval_feedback(limit: int = 50):
    try:
        from utils.chat_history_store import get_chat_history_store
        store = get_chat_history_store()
        stats = store.get_feedback_stats()
        recent = store.get_recent_feedback(limit=min(limit, 200))
        return {"stats": stats, "recent": recent}
    except Exception as e:
        return {"error": str(e)}
