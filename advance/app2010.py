from app_base import *  # import your original app file unchanged
from wydot_features import *

from features_plus import (
    patch_app, render_keyboard_and_voice,
    render_assignments_ui, render_citations_bar,
    apply_citations_filter, render_section_aware_preview,
    use_multihop_if_enabled
)
# _ = patch_app(globals())
from wydot_features import (
    RBAC, ValidatorDB, VectorFusion, self_validate, solve_expression, convert_units,
    federated_search, KG, check_compliance, OfflineCache, render_map_html,
    make_report, route, workspace_summarize
) 

# -------------------------------------------------------------------

# ======= YOUR ORIGINAL APP (kept intact) =======
# (Pasted exactly as you provided; unchanged except for the import above.)
# ------------- BEGIN ORIGINAL CONTENT -------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, sqlite3, threading, json, base64, hmac, hashlib, io, csv, math, tempfile, re, pathlib
from typing import List, Dict, Any, Optional, Tuple, Generator
from urllib.parse import urlparse, quote
import base64 as _b64

# import streamlit as st
# import streamlit.components.v1 as components
# from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# ===== Kill stray "None" in the UI =====
# Some imported code is likely doing st.write(None) or similar.
# We monkey-patch Streamlit so those calls are ignored.

_orig_write = st.write
def _safe_write(*args, **kwargs):
    # Ignore bare st.write(None)
    if len(args) == 1 and args[0] is None and not kwargs:
        return
    return _orig_write(*args, **kwargs)
st.write = _safe_write

_orig_markdown = st.markdown
def _safe_markdown(*args, **kwargs):
    # Ignore st.markdown(None)
    if len(args) == 1 and args[0] is None and not kwargs:
        return
    return _orig_markdown(*args, **kwargs)
st.markdown = _safe_markdown
# ======================================

# ---------- Vertex GenAI (new SDK) ----------
from google import genai
from google.genai import types as gtypes
from google.oauth2 import service_account

# ---------- Milvus (direct) ----------
from pymilvus import connections, utility, Collection


# ---------- Optional community components ----------
try:
    from audio_recorder_streamlit import audio_recorder  # returns WAV bytes
except Exception:
    audio_recorder = None


# =========================================================
# Small helpers (compat, rerun)
# =========================================================
def _rerun():
    try:
        st.rerun()
    except Exception:
        pass

# =========================================================
# ENV & CONSTANTS
# =========================================================
load_dotenv()

# ---- Zilliz/Milvus Serverless (URI + TOKEN) ----
DEFAULT_MILVUS_URI = os.getenv(
    "MILVUS_URI",
    "https://in03-339fce16acb492f.serverless.aws-eu-central-1.cloud.zilliz.com",
)
DEFAULT_MILVUS_TOKEN = os.getenv(
    "MILVUS_TOKEN",
    "7f85f7a13313474b99a9d5225a2c5c590fbd749bb68c07719266c4889d9121c601dc948a8c9052c53dbd84dcfcbbcb6fbb36b17d",
)
DEFAULT_COLLECTION = os.getenv("MILVUS_COLLECTION", "wydotspec_llamaparse")

# Optional: where your PDFs are hosted (or directory with PDFs).
# Can be an HTTP(S) URL or a local path like:
#   C:\Users\nsubedi1\Desktop\WYDOT project\data
PDF_BASE_URL = os.getenv("PDF_BASE_URL", "").rstrip("/")

# Writable default for Streamlit Cloud
CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", os.path.join(tempfile.gettempdir(), "chat_history.sqlite3"))

# ---- Vertex tuned endpoint / project ----
PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("VERTEX_LOCATION") or "us-central1"

# Backward-compat single env var (kept if you only have one tuned endpoint)
LEGACY_TUNED_ENDPOINT = os.getenv("WYDOT_TUNED_ENDPOINT")  # projects/.../locations/.../endpoints/...

# New: allow two tuned endpoints (Pro & Flash)
ENV_TUNED_FLASH = os.getenv("WYDOT_TUNED_ENDPOINT_FLASH") or LEGACY_TUNED_ENDPOINT
ENV_TUNED_PRO   = os.getenv("WYDOT_TUNED_ENDPOINT_PRO")

# Base model fallbacks
BASE_FLASH = os.getenv("DEFAULT_BASE_FLASH", "gemini-2.5-flash")
BASE_PRO   = os.getenv("DEFAULT_BASE_PRO", "gemini-2.5-pro")

if not PROJECT:
    raise RuntimeError("Set GOOGLE_CLOUD_PROJECT (or PROJECT_ID) to your GCP project id.")

EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

VECTOR_FIELD = "vector"
METRIC_TYPE = "COSINE"

MAX_HISTORY_MSGS = 20
HISTORY_PAIRS_FOR_PROMPT = 6
SCHEMA_VERSION = 2  # bump to invalidate cached ChatHistoryStore when API/schema changes



if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"session_{int(time.time())}"

# --- Ensure offline cache exists ---
if "offline_cache" not in st.session_state:
    st.session_state["offline_cache"] = OfflineCache(embed_text_eval)

# --- Load conversation into offline cache ---
ocache = st.session_state["offline_cache"]

# Load chat history from your DB
messages = CHAT_DB.recent(
    effective_user_id(),
    st.session_state["session_id"],
    limit=MAX_HISTORY_MSGS
)

# Convert DB messages → offline RAG chunks
_ = ocache.load_from_session(messages)
# =========================================================
# Service Account auth loader (Streamlit secrets-first)
# =========================================================
CLOUD_SCOPE = ["https://www.googleapis.com/auth/cloud-platform"]

def _creds_from_secrets_dict() -> Optional[service_account.Credentials]:
    try:
        if "gcp_service_account" in st.secrets:
            info = dict(st.secrets["gcp_service_account"])
            return service_account.Credentials.from_service_account_info(info).with_scopes(CLOUD_SCOPE)
    except Exception as e:
        st.warning(f"[Auth] Failed loading st.secrets['gcp_service_account']: {e}")
    return None

def _creds_from_env_file() -> Optional[service_account.Credentials]:
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    try:
        if key_path and os.path.exists(key_path):
            return service_account.Credentials.from_service_account_file(key_path).with_scopes(CLOUD_SCOPE)
    except Exception as e:
        st.warning(f"[Auth] Failed loading GOOGLE_APPLICATION_CREDENTIALS file: {e}")
    return None

def _creds_from_env_json() -> Optional[service_account.Credentials]:
    try:
        raw = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
        if raw:
            info = json.loads(raw)
            return service_account.Credentials.from_service_account_info(info).with_scopes(CLOUD_SCOPE)
        b64 = os.getenv("GCP_SERVICE_ACCOUNT_BASE64")
        if b64:
            info = json.loads(base64.b64decode(b64).decode("utf-8"))
            return service_account.Credentials.from_service_account_info(info).with_scopes(CLOUD_SCOPE)
    except Exception as e:
        st.warning(f"[Auth] Failed loading env JSON creds: {e}")
    return None

def load_gcp_credentials() -> Tuple[Optional[service_account.Credentials], str]:
    for loader, label in [
        (_creds_from_secrets_dict, "st.secrets:gcp_service_account"),
        (_creds_from_env_file, "GOOGLE_APPLICATION_CREDENTIALS file"),
        (_creds_from_env_json, "env:GCP_SERVICE_ACCOUNT_JSON/BASE64"),
    ]:
        creds = loader()
        if creds:
            return creds, label
    return None, "ADC (no SA key found)"


# ===== Prompt =====
def format_prompt(context_text: str, extracted_text: str, question: str, history_text: str, has_media: bool = False) -> str:
    # Answer mode affects how we talk to the model
    answer_mode = st.session_state.get("answer_mode", "RAG (Specs)") if "answer_mode" in st.session_state else "RAG (Specs)"

    # Standard Gemini – ignore RAG context
    if answer_mode == "Standard Gemini (no specs)":
        base = (
            "You are a helpful and knowledgeable assistant.\n"
            "Use your own knowledge to answer the user's questions clearly and concisely.\n\n"
            "Conversation so far (most recent turns first):\n"
            "{history}\n\n"
        )
        if (question or "").strip():
            tail = (
                "User question: ```{question}```\n"
                "JUST PROVIDE THE ANSWER IN ENGLISH WITHOUT ``` AND NOTHING ELSE.\n"
            )
        else:
            tail = (
                "No explicit text question was provided. Provide a brief helpful response based on the conversation above.\n"
                "JUST PROVIDE THE ANSWER IN ENGLISH WITHOUT ``` AND NOTHING ELSE.\n"
            )
        return base.format(history=(history_text or "(no previous turns)")) + tail.format(question=question or "")

    # Web search (Google) – encourage use of the search tool, no RAG context
    if answer_mode == "Web search (Google)":
        base = (
            "You are a helpful assistant that can use the Google Search tool to fetch up-to-date information.\n"
            "For questions that depend on recent events, specific external facts, or URLs, you SHOULD call the search tool.\n\n"
            "Conversation so far (most recent turns first):\n"
            "{history}\n\n"
        )
        if (question or "").strip():
            tail = (
                "User question: ```{question}```\n"
                "Use the Google Search tool as needed. Summarize the result succinctly.\n"
                "JUST PROVIDE THE ANSWER IN ENGLISH WITHOUT ``` AND NOTHING ELSE.\n"
            )
        else:
            tail = (
                "No explicit text question was provided. Provide a brief helpful response based on the conversation above.\n"
                "JUST PROVIDE THE ANSWER IN ENGLISH WITHOUT ``` AND NOTHING ELSE.\n"
            )
        return base.format(history=(history_text or "(no previous turns)")) + tail.format(question=question or "")

    # Default: RAG / Workspace WYDOT chatbot
    ctx_for_prompt = context_text if context_text else extracted_text

    base = (
        "You are WYDOT chatbot, a polite and helpful Virtual Assistant of Wyoming Department of Transportation (WYDOT).\n"
        "Answer the question using the provided context and (if relevant) the prior conversation.\n\n"
        "Conversation so far (most recent turns first):\n"
        "{history}\n\n"
        "Context inside double backticks:``{context}``\n"
    )
    if (question or "").strip():
        tail = (
            "Question inside triple backticks:```{question}```\n"
            "If the question is out of scope, answer based on your role.\n"
            "JUST PROVIDE THE ANSWER IN ENGLISH WITHOUT ``` AND NOTHING ELSE.\n"
        )
    else:
        inferred = (
            "No explicit text question was provided. Attached media may include audio, video, images, or PDFs.\n"
            "Do the following in order:\n"
            "1) If audio/video is present: transcribe it (brief timestamps only if helpful) and extract the user's intent.\n"
            "2) If images/PDFs are present: describe/parse them (OCR key fields if relevant).\n"
            "3) Use any retrieved context and the transcription/parse to answer succinctly.\n"
            "4) If it's not a question, provide a concise WYDOT-relevant summary or actionable next steps.\n"
            "JUST PROVIDE THE ANSWER IN ENGLISH WITHOUT ``` AND NOTHING ELSE.\n"
        )
        tail = inferred if has_media else (
            "No explicit question was provided. Provide a brief helpful response based on the context above.\n"
            "JUST PROVIDE THE ANSWER IN ENGLISH WITHOUT ``` AND NOTHING ELSE.\n"
        )

    return base.format(
        history=(history_text or "(no previous turns)"),
        context=ctx_for_prompt,
    ) + tail.format(question=question or "")


# =========================================================
# google-genai Client (Vertex endpoint; uses SA credentials)
# =========================================================
@st.cache_resource(show_spinner=False)
def get_genai_client() -> genai.Client:
    creds, label = load_gcp_credentials()
    st.session_state["auth_label"] = label
    if creds:
        return genai.Client(vertexai=True, project=PROJECT, location=LOCATION, credentials=creds)
    return genai.Client(vertexai=True, project=PROJECT, location=LOCATION)  # ADC


# =========================================================
# Auth helpers (SQLite users)
# =========================================================
def _pbkdf2_hash(password: str, iterations: int = 200_000) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64.b64encode(salt).decode()}${_b64.b64encode(dk).decode()}"

def _pbkdf2_verify(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_b64, dk_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(iters)
        salt = _b64.b64decode(salt_b64)
        dk = _b64.b64decode(dk_b64)
        new_dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(new_dk, dk)
    except Exception:
        return False

def current_user_id() -> int:
    return int(st.session_state.get("user_id", 0))

def current_user_email() -> Optional[str]:
    return st.session_state.get("user_email")

def effective_user_id() -> int:
    """Logged-in user id if >0, else a per-browser-session anonymous id (never mixed)."""
    uid = current_user_id()
    if uid and uid > 0:
        return uid
    if "anon_uid" not in st.session_state:
        # 63-bit random positive integer (SQLite INTEGER fits 64-bit signed)
        st.session_state["anon_uid"] = int.from_bytes(os.urandom(8), "big") & ((1 << 63) - 1)
    return int(st.session_state["anon_uid"])


# =========================================================
# Chat history + Users (SQLite)
# =========================================================
class ChatHistoryStore:
    def __init__(self, db_path: str):
        self._lock = threading.Lock()

        # Ensure directory exists & connect with WAL + timeouts for cloud
        os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA busy_timeout=5000;")
        except Exception:
            pass

        # Users table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT,
                created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
            )
        """)

        # Messages table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT CHECK(role IN ('user','assistant')) NOT NULL,
                content TEXT NOT NULL,
                ts REAL NOT NULL DEFAULT (strftime('%s','now'))
            )
        """)

        # Ensure user_id column exists on messages; default 0 = (legacy) anonymous
        cols = [r[1] for r in self._conn.execute("PRAGMA table_info(messages)").fetchall()]
        if "user_id" not in cols:
            self._conn.execute("ALTER TABLE messages ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0")

        # Helpful indexes
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_uid_sid ON messages (user_id, session_id, id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)")
        self._conn.commit()

    # ---------- User ops ----------
    def create_user(self, email: str, password: str, display_name: Optional[str] = None) -> Tuple[Optional[int], Optional[str]]:
        email = (email or "").strip().lower()
        if not email or not password:
            return None, "Email and password are required."
        with self._lock:
            cur = self._conn.execute("SELECT id FROM users WHERE email=?", (email,))
            if cur.fetchone():
                return None, "This email is already registered."
            pw_hash = _pbkdf2_hash(password)
            self._conn.execute(
                "INSERT INTO users (email, password_hash, display_name) VALUES (?,?,?)",
                (email, pw_hash, display_name),
            )
            self._conn.commit()
            cur = self._conn.execute("SELECT id FROM users WHERE email=?", (email,))
            row = cur.fetchone()
            return (int(row[0]) if row else None), None

    def authenticate(self, email: str, password: str) -> Tuple[Optional[int], Optional[str]]:
        email = (email or "").strip().lower()
        with self._lock:
            cur = self._conn.execute("SELECT id, password_hash FROM users WHERE email=?", (email,))
            row = cur.fetchone()
            if not row:
                return None, "No account with that email."
            uid, pw_hash = int(row[0]), row[1]
            if not _pbkdf2_verify(password, pw_hash):
                return None, "Incorrect password."
            return uid, None

    def get_display_name(self, user_id: int) -> Optional[str]:
        with self._lock:
            cur = self._conn.execute("SELECT display_name, email FROM users WHERE id=?", (user_id,))
            row = cur.fetchone()
            return row[0] or row[1] if row else None

    # ---------- Message ops (scoped by user) ----------
    def add(self, user_id: int, session_id: str, role: str, content: str, ts: Optional[float] = None):
        if not session_id: session_id = "default"
        if ts is None: ts = time.time()
        user_id = int(user_id or 0)
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (user_id, session_id, role, content, ts) VALUES (?, ?, ?, ?, ?)",
                (user_id, session_id, role, content, ts)
            )
            self._conn.commit()

    def recent(self, user_id: int, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        if not session_id: session_id = "default"
        user_id = int(user_id or 0)
        with self._lock:
            cur = self._conn.execute(
                "SELECT role, content, ts FROM messages WHERE user_id=? AND session_id=? ORDER BY id DESC LIMIT ?",
                (user_id, session_id, limit)
            )
            rows = cur.fetchall()
        rows.reverse()
        return [{"role": r[0], "content": r[1], "ts": r[2]} for r in rows]

    def clear_session(self, user_id: int, session_id: str):
        user_id = int(user_id or 0)
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE user_id=? AND session_id=?", (user_id, session_id))
            self._conn.commit()

    # ---------- Conversations (per-user) ----------
    def list_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        user_id = int(user_id or 0)
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT session_id, COUNT(*) AS n, MAX(ts) AS last_ts
                FROM messages
                WHERE user_id=?
                GROUP BY session_id
                ORDER BY last_ts DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()
        return [
            {"session_id": r[0], "count": int(r[1]), "last_ts": float(r[2] or 0.0)}
            for r in rows
        ]

    def rename_session(self, user_id: int, old_sid: str, new_sid: str) -> bool:
        if not new_sid:
            return False
        user_id = int(user_id or 0)
        with self._lock:
            self._conn.execute(
                "UPDATE messages SET session_id=? WHERE user_id=? AND session_id=?",
                (new_sid, user_id, old_sid),
            )
            self._conn.commit()
        return True

    def all_messages(self, user_id: int, session_id: str) -> List[Dict[str, Any]]:
        user_id = int(user_id or 0)
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT id, role, content, ts
                FROM messages
                WHERE user_id=? AND session_id=?
                ORDER BY id ASC
                """,
                (user_id, session_id or "default"),
            )
            rows = cur.fetchall()
        return [
            {"id": int(r[0]), "role": r[1], "content": r[2], "ts": float(r[3])}
            for r in rows
        ]


@st.cache_resource(show_spinner=False)
def get_chat_store(path: str, schema_version: int = 1):
    # schema_version is used to invalidate cache on changes
    return ChatHistoryStore(path)

CHAT_DB = get_chat_store(CHAT_DB_PATH, schema_version=SCHEMA_VERSION)

def _ensure_conversation_api():
    global CHAT_DB
    missing = [m for m in ("list_sessions", "rename_session", "all_messages")
               if not hasattr(CHAT_DB, m)]
    if missing:
        try:
            get_chat_store.clear()
        except Exception:
            pass
        CHAT_DB = get_chat_store(CHAT_DB_PATH, schema_version=SCHEMA_VERSION)

_ = _ensure_conversation_api()


def add_to_history(session_id: str, role: str, content: str):
    CHAT_DB.add(effective_user_id(), session_id, role, content)


def get_history_text(session_id: str, max_pairs: int = HISTORY_PAIRS_FOR_PROMPT) -> str:
    limit = min(MAX_HISTORY_MSGS, 2 * max_pairs)
    msgs = CHAT_DB.recent(effective_user_id(), session_id, limit=limit)
    lines = []
    for m in msgs:
        prefix = "USER" if m["role"] == "user" else "ASSISTANT"
        lines.append(f"{prefix}: {m['content']}")
    return "\n".join(lines)


# =========================================================
# Milvus connection (URI+TOKEN) & schema probe
# =========================================================
@st.cache_resource(show_spinner=False)
def get_milvus_collection(uri: str, token: str, collection: str) -> Tuple[Optional[Collection], Optional[int], Optional[list]]:
    try:
        connections.connect(alias="default", uri=uri, token=token)
    except Exception as e:
        st.error(f"Milvus connect error: {e}")
        return None, None, None

    if not utility.has_collection(collection):
        st.error(f"Milvus collection not found: {collection}")
        return None, None, None

    col = Collection(collection)
    try:
        col.load()
    except Exception as e:
        st.warning(f"Milvus load() warning: {e}")

    dim = None
    fields_summary = []
    for f in col.schema.fields:
        params = getattr(f, "params", {}) or {}
        fields_summary.append({
            "name": f.name,
            "dtype": str(getattr(f, "dtype", "")),
            "is_primary": bool(getattr(f, "is_primary", False)),
            "auto_id": bool(getattr(f, "auto_id", False)),
            "params": dict(params),
        })
        if f.name == VECTOR_FIELD:
            dim = params.get("dim") or getattr(f, "dim", None)
            if dim is not None:
                dim = int(dim)
    if dim is None:
        st.error(f"Could not detect vector dimension for field '{VECTOR_FIELD}'.")
        return None, None, fields_summary

    return col, dim, fields_summary


# =========================================================
# Embeddings via google-genai (Vertex)
# =========================================================
def _extract_values_any(resp) -> List[float]:
    if hasattr(resp, "embedding") and getattr(resp.embedding, "values", None):
        return [float(x) for x in resp.embedding.values]
    if hasattr(resp, "embeddings") and getattr(resp, "embeddings", None):
        e0 = resp.embeddings[0]
        vals = getattr(e0, "values", None)
        if vals is not None:
            return [float(x) for x in vals]
    if isinstance(resp, dict):
        if "embedding" in resp and isinstance(resp["embedding"], dict) and "values" in resp["embedding"]:
            return [float(x) for x in resp["embedding"]["values"]]
        if "embeddings" in resp and isinstance(resp["embeddings"], list) and resp["embeddings"]:
            e0 = resp["embeddings"][0]
            if "values" in e0:
                return [float(x) for x in e0["values"]]
    try:
        js = json.loads(gtypes.to_json(resp))
        return _extract_values_any(js)
    except Exception:
        pass
    raise ValueError("Unexpected embedding response structure (no 'values' found).")


@st.cache_resource(show_spinner=False)
def get_embed_client() -> genai.Client:
    return get_genai_client()


def embed_query_vector(text: str, dim: int) -> List[float]:
    client = get_embed_client()
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=gtypes.EmbedContentConfig(
            output_dimensionality=dim,
            task_type="RETRIEVAL_QUERY",
        ),
    )
    vals = _extract_values_any(resp)
    n = math.sqrt(sum(v*v for v in vals))
    if n > 0:
        vals = [v / n for v in vals]
    return vals


# =========================================================
# Retrieval (Milvus direct)
# =========================================================
def milvus_similarity_search(query: str, k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    if not query or not query.strip():
        return "", []

    uri = st.session_state.get("milvus_uri", DEFAULT_MILVUS_URI)
    token = st.session_state.get("milvus_token", DEFAULT_MILVUS_TOKEN)
    collection = st.session_state.get("collection", DEFAULT_COLLECTION)

    col, dim, _ = get_milvus_collection(uri, token, collection)
    if not col or not dim:
        return "", []

    try:
        qv = embed_query_vector(query, dim)
    except Exception as e:
        st.warning(f"[Embed] {e}")
        return "", []

    try:
        res = col.search(
            data=[qv],
            anns_field=VECTOR_FIELD,
            param={"metric_type": METRIC_TYPE, "params": {"ef": 64}},
            limit=k,
            output_fields=["doc_id", "chunk_id", "page", "section", "source", "content"],
        )
    except Exception as e:
        st.warning(f"[Milvus search] {e}")
        return "", []

    chunks, sources = [], []
    if res and len(res) > 0:
        for hit in res[0]:
            md = hit.entity
            content = md.get("content") or ""
            chunks.append(content)
            sources.append({
                "doc_id": md.get("doc_id"),
                "page": md.get("page"),
                "source": md.get("source"),
                "preview": content[:300] if content else ""
            })
    return "\n\n".join(chunks), sources

# _ = patch_app(globals())


# =========================================================
# Build google-genai request (streaming)
# =========================================================
def build_contents_and_config(
    query: str,
    context_text: str,
    extracted_text: str,
    uploads: Optional[List[Dict[str, Any]]],
    history_text: str,
    tools: Optional[List[gtypes.Tool]] = None,
) -> Tuple[list, gtypes.GenerateContentConfig]:
    uploads = uploads or []
    has_media = len(uploads) > 0

    prompt_text = format_prompt(
        context_text=context_text,
        extracted_text=extracted_text,
        question=query or "",
        history_text=history_text,
        has_media=has_media,
    )

    parts = [gtypes.Part.from_text(text=prompt_text)]
    for item in uploads:
        b = item.get("bytes")
        if not b:
            continue
        mime = item.get("mime") or "application/octet-stream"
        parts.append(gtypes.Part.from_bytes(data=b, mime_type=mime))

    contents = [gtypes.Content(role="user", parts=parts)]

    config = gtypes.GenerateContentConfig(
        temperature=1,
        top_p=1,
        max_output_tokens=65535,
        safety_settings=[
            gtypes.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            gtypes.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            gtypes.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            gtypes.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ],
        tools=tools or None,
    )
    return contents, config


# =========================================================
# Model selection helpers
# =========================================================
def _init_model_state():
    st.session_state.setdefault("flash_endpoint", ENV_TUNED_FLASH)
    st.session_state.setdefault("pro_endpoint", ENV_TUNED_PRO)
    default_choice = "Tuned: Flash 2.5 (endpoint)" if ENV_TUNED_FLASH else "Base: gemini-2.5-flash"
    st.session_state.setdefault("model_choice", default_choice)

def get_selected_model_id() -> str:
    choice = st.session_state.get("model_choice", "Base: gemini-2.5-flash")
    flash_ep = st.session_state.get("flash_endpoint")
    pro_ep = st.session_state.get("pro_endpoint")

    if choice == "Tuned: Flash 2.5 (endpoint)":
        return flash_ep or BASE_FLASH
    if choice == "Tuned: Pro 2.5 (endpoint)":
        return pro_ep or BASE_PRO
    if choice == "Base: gemini-2.5-pro":
        return BASE_PRO
    return BASE_FLASH


# =========================================================
# Pipeline (streaming)
# =========================================================
def resultDocuments_streaming(
    query: str,
    extracted_text: str = "",
    uploads: Optional[List[Dict[str, Any]]] = None,
    session_id: str = "default",
) -> Generator[Tuple[str, List[Dict[str, Any]]], None, None]:
    answer_mode = st.session_state.get("answer_mode", "RAG (Specs)")

    # Reset web search state unless explicitly in web search mode
    if "last_web_sources" not in st.session_state:
        st.session_state["last_web_sources"] = []
    if "web_search_widget_html" not in st.session_state:
        st.session_state["web_search_widget_html"] = ""
    if answer_mode != "Web search (Google)":
        st.session_state["last_web_sources"] = []
        st.session_state["web_search_widget_html"] = ""

    # 1) Retrieval from Milvus in RAG-like modes
    # if answer_mode in ("RAG (Specs)", "Workspace (Gmail/Drive + Specs)"):
    #     context_text, sources = milvus_similarity_search(query if (query and query.strip()) else "", k=5)
    # else:
        
     # 1) Retrieval from Milvus in RAG-like modes (multi-hop aware)
    if answer_mode in ("RAG (Specs)", "Workspace (Gmail/Drive + Specs)"):
        context_text, sources = use_multihop_if_enabled(
            globals(),
            (query or "").strip(),
            default_k=5
        )
    else:
        context_text, sources = "", []
   

    # 2) Optional Workspace context
    workspace_context = ""
    if answer_mode == "Workspace (Gmail/Drive + Specs)":
        try:
            workspace_context = get_workspace_context(query or "")
        except Exception as e:
            st.info(f"Workspace context error: {e}")
            workspace_context = ""

    full_context_text_parts: List[str] = []
    if context_text:
        full_context_text_parts.append(context_text)
    if workspace_context:
        full_context_text_parts.append(workspace_context)
    full_context_text = "\n\n".join(full_context_text_parts)

    history_text = get_history_text(session_id, max_pairs=HISTORY_PAIRS_FOR_PROMPT)

    # 3) Add user turn to history
    if (query or "").strip():
        add_to_history(session_id, "user", query)
    elif uploads:
        kinds = []
        for u in uploads:
            mt = (u.get("mime") or "").lower()
            if mt.startswith("audio/"): kinds.append("audio")
            elif mt.startswith("video/"): kinds.append("video")
            elif mt.startswith("image/"): kinds.append("image")
            elif mt.endswith("pdf"):     kinds.append("pdf")
            else:                        kinds.append("file")
        add_to_history(session_id, "user", f"(attachments only: {', '.join(kinds)})")

    # 4) Choose tools config (Google Search tool)
    tools = None
    if answer_mode == "Web search (Google)":
        try:
            google_search_tool = gtypes.Tool(google_search=gtypes.GoogleSearch())
            tools = [google_search_tool]
        except Exception as e:
            st.warning(f"Failed to enable Google Search tool: {e}")

    # 5) In Standard Gemini mode, ignore any RAG / extra text context
    effective_extracted = extracted_text
    if answer_mode == "Standard Gemini (no specs)":
        full_context_text = ""
        effective_extracted = ""

    contents, config = build_contents_and_config(
        query=query or "",
        context_text=full_context_text,
        extracted_text=effective_extracted,
        uploads=uploads,
        history_text=history_text,
        tools=tools,
    )

    client = get_genai_client()
    model_id = get_selected_model_id()
    acc: List[str] = []

    # 6) Web search mode: synchronous call so we can capture citations + widget HTML
    if answer_mode == "Web search (Google)" and tools:
        try:
            resp = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=config,
            )
            text = (resp.text or "").strip()

            web_sources: List[Dict[str, str]] = []
            widget_html = ""
            try:
                if getattr(resp, "candidates", None):
                    cand0 = resp.candidates[0]
                    gm = getattr(cand0, "grounding_metadata", None)
                    if gm:
                        chunks = getattr(gm, "grounding_chunks", None) or []
                        for ch in chunks:
                            web = getattr(ch, "web", None)
                            if web and getattr(web, "uri", None):
                                web_sources.append({
                                    "url": getattr(web, "uri", ""),
                                    "title": getattr(web, "title", "") or getattr(web, "uri", ""),
                                })
                        se = getattr(gm, "search_entry_point", None)
                        if se and getattr(se, "rendered_content", None):
                            widget_html = getattr(se, "rendered_content")
            except Exception:
                pass

            st.session_state["last_web_sources"] = web_sources
            st.session_state["web_search_widget_html"] = widget_html

            if text:
                add_to_history(session_id, "assistant", text)
                yield text, sources
            else:
                err_text = "[Model returned an empty response.]"
                add_to_history(session_id, "assistant", err_text)
                yield err_text, sources
            return
        except Exception as e:
            err = f"[Model error] {e}"
            add_to_history(session_id, "assistant", err)
            yield err, sources
            return

    # 7) All other modes: keep existing streaming behaviour
    try:
        for chunk in client.models.generate_content_stream(
            model=model_id,
            contents=contents,
            config=config,
        ):
            if getattr(chunk, "text", None):
                acc.append(chunk.text)
                yield "".join(acc), sources
        final_text = "".join(acc).strip()
        if final_text:
            add_to_history(session_id, "assistant", final_text)
    except Exception as e:
        err = f"[Model error] {e}"
        add_to_history(session_id, "assistant", err)
        yield err, sources


# =========================================================
# PDF deep-link helpers
# =========================================================
def _looks_like_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https")
    except Exception:
        return False

def build_pdf_url(source: Optional[str], page: Optional[int]) -> Optional[str]:
    """
    Build a URL or file URI pointing to the PDF at the given source and page.

    - If `source` is already http/https or file://, use it directly.
    - If `source` is an absolute local path, convert to file:// URI.
    - Else, if PDF_BASE_URL is set, treat it as either a URL prefix or
      a local directory and join with the filename from `source`.
    """
    if not source:
        return None

    anchor = ""
    if isinstance(page, int) and page >= 1:
        anchor = f"#page={page}"

    # Already a URL or file URI
    if _looks_like_url(source) or str(source).startswith("file://"):
        return f"{source}{anchor}"

    # Absolute local path -> file URI
    try:
        p = pathlib.Path(source)
        if p.is_absolute():
            return p.as_uri() + anchor
    except Exception:
        pass

    # Fallback to PDF_BASE_URL
    base = PDF_BASE_URL
    if base:
        # If base is an absolute local path, convert to file URI
        if os.path.isabs(base) and not base.startswith("file://"):
            base_uri = pathlib.Path(base).as_uri()
        else:
            base_uri = base
        if not base_uri.endswith("/"):
            base_uri = base_uri + "/"
        filename = os.path.basename(str(source))
        return f"{base_uri}{quote(filename)}{anchor}"

    return None


# =========================================================
# Chat composer (attachments + camera + audio/video record)
# =========================================================
def render_chat_composer() -> Tuple[Optional[str], List[Dict[str, Any]]]:
    st.markdown(
        """
    <style>
      .composer {padding:10px 12px; border:1px solid #e6e6e6; border-radius:28px; background:#fff;}
      .composer .stTextInput>div>div>input {border:0 !important; outline:none !important; background:transparent !important;}
      .composer .stTextInput>label {display:none;}
      .composer .stButton>button {border-radius:24px; height:40px;}
      .icon-btn button {padding:0 10px; height:36px;}
      .chip {display:inline-block; padding:2px 8px; border-radius:12px; background:#f1f1f5; margin-right:6px; font-size:12px;}
      .right-sticky { position: sticky; top: 70px; max-height: calc(100vh - 90px); overflow-y: auto; }
      .left-scroll  { max-height: calc(100vh - 90px); overflow-y: auto; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    if "chat_attached_preview" not in st.session_state:
        st.session_state["chat_attached_preview"] = []
    if "camera_image" not in st.session_state:
        st.session_state["camera_image"] = None
    if "audio_bytes" not in st.session_state:
        st.session_state["audio_bytes"] = None
    if "video_file" not in st.session_state:
        st.session_state["video_file"] = None

    with st.form("chat_composer_form", clear_on_submit=True):
        c_plus, c_input, c_mic, c_send = st.columns([0.12, 1, 0.12, 0.15])

        with c_plus:
            with st.popover("➕", use_container_width=False):
                tab_up, tab_cam, tab_aud, tab_vid = st.tabs(["Upload", "Camera", "Record audio", "Record video"])

                with tab_up:
                    attachments = st.file_uploader(
                        "Attach files",
                        type=[
                            "png","jpg","jpeg","webp","gif",
                            "pdf",
                            "mp4","mov","mkv","webm","avi"
                        ],
                        accept_multiple_files=True,
                        help="Add images, PDFs, or videos"
                    )
                    st.caption("Tip: On mobile/Chrome you can pick 'Use camera' to record a video directly.")
                    if attachments is not None:
                        st.session_state["chat_attached_preview"] = attachments

                with tab_cam:
                    cam_img = st.camera_input("Click to take a photo")
                    if cam_img is not None:
                        st.session_state["camera_image"] = cam_img

                with tab_aud:
                    if audio_recorder is not None:
                        st.caption("Press to record / stop:")
                        wav_bytes = audio_recorder(text="", recording_color="#e74c3c", neutral_color="#2ecc71", icon_size="2x")
                        if wav_bytes:
                            st.audio(wav_bytes, format="audio/wav")
                            st.session_state["audio_bytes"] = wav_bytes
                    else:
                        st.info("Install `audio-recorder-streamlit` to record in-browser, or upload a file below.")
                        aud_up = st.file_uploader("Upload audio", type=["wav","mp3","m4a","webm"], accept_multiple_files=False, key="aud_up")
                        if aud_up is not None:
                            st.session_state["audio_bytes"] = aud_up.getvalue()

                with tab_vid:
                    vid_up = st.file_uploader("Record/Upload video", type=["mp4","mov","mkv","webm","avi"], accept_multiple_files=False, key="vid_up")
                    if vid_up is not None:
                        st.session_state["video_file"] = vid_up

            cnt = len(st.session_state.get("chat_attached_preview", []))
            cnt += 1 if st.session_state.get("camera_image") else 0
            cnt += 1 if st.session_state.get("audio_bytes") else 0
            cnt += 1 if st.session_state.get("video_file") else 0
            if cnt:
                st.markdown(f'<span class="chip">{cnt} attachment{"s" if cnt>1 else ""}</span>', unsafe_allow_html=True)

        with c_input:
            message = st.text_input("Ask anything", placeholder="Ask anything related to WYDOT", label_visibility="collapsed", key="composer_text")

        with c_mic:
            with st.popover("🎤", use_container_width=True):
                extra_aud = st.file_uploader("Upload audio (optional)", type=["wav","mp3","m4a","webm"], accept_multiple_files=False, key="extra_aud_up")
                if extra_aud is not None:
                    st.session_state["audio_bytes"] = extra_aud.getvalue()

        with c_send:
            submitted = st.form_submit_button("➤", type="primary", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        if submitted:
            payload: List[Dict[str, Any]] = []

            for f in list(st.session_state.get("chat_attached_preview", [])):
                try:
                    payload.append({
                        "bytes": f.getvalue(),
                        "mime": f.type or "application/octet-stream",
                        "name": f.name,
                    })
                except Exception as e:
                    st.warning(f"Failed to read {getattr(f, 'name', 'file')}: {e}")

            if st.session_state.get("camera_image") is not None:
                cam_file = st.session_state["camera_image"]
                try:
                    payload.append({
                        "bytes": cam_file.getvalue(),
                        "mime": "image/jpeg",
                        "name": "camera.jpg",
                    })
                except Exception as e:
                    st.warning(f"Camera image read failed: {e}")

            if st.session_state.get("audio_bytes"):
                payload.append({
                    "bytes": st.session_state["audio_bytes"],
                    "mime": "audio/wav",
                    "name": "recording.wav",
                })

            if st.session_state.get("video_file") is not None:
                v = st.session_state["video_file"]
                try:
                    payload.append({
                        "bytes": v.getvalue(),
                        "mime": v.type or "video/mp4",
                        "name": v.name or "recording.mp4",
                    })
                except Exception as e:
                    st.warning(f"Video read failed: {e}")

            st.session_state["chat_attached_preview"] = []
            st.session_state["camera_image"] = None
            st.session_state["audio_bytes"] = None
            st.session_state["video_file"] = None

            return (message.strip() if message else None), payload

    return None, []


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="WYDOT employee bot", page_icon="🛣️", layout="wide")
# _=render_keyboard_and_voice()
# Sidebar (now also handles account)
with st.sidebar:
    # ---------- Account ----------
    st.markdown("##  Account")
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = 0  # 0 = anonymous until signup/login
        st.session_state["user_email"] = None
        st.session_state["display_name"] = None

    if current_user_id() > 0:
        dn = st.session_state.get("display_name") or current_user_email() or f"User #{current_user_id()}"
        st.success(f"Logged in as: {dn}")
        colA, colB = st.columns(2)
        with colA:
            if st.button("🔓 Logout", use_container_width=True):
                st.session_state["user_id"] = 0
                st.session_state["user_email"] = None
                st.session_state["display_name"] = None
                # Keep anon_uid so anonymous conversations for this browser stay private
                st.session_state["session_id"] = "default"
                st.session_state["last_sources"] = []
                st.session_state["pdf_preview_url"] = None
                st.toast("Logged out.", icon="👋")
                _rerun()
        with colB:
            st.caption(" ")
    else:
        login_tab, signup_tab = st.tabs(["Log in", "Sign up"])
        with login_tab:
            with st.form("login_form_sidebar"):
                le = st.text_input("Email", placeholder="you@company.com")
                lp = st.text_input("Password", type="password")
                do_login = st.form_submit_button("Log in", use_container_width=True)
            if do_login:
                uid, err = CHAT_DB.authenticate(le, lp)
                if err:
                    st.error(err)
                else:
                    st.session_state["user_id"] = uid
                    st.session_state["user_email"] = (le or "").strip().lower()
                    st.session_state["display_name"] = CHAT_DB.get_display_name(uid)
                    # Reset to a neutral session for this (possibly first-time) user
                    st.session_state["session_id"] = "default"
                    st.success("Logged in ✅")
                    _rerun()

        with signup_tab:
            with st.form("signup_form_sidebar"):
                se = st.text_input("Email", placeholder="you@company.com", key="as_email")
                sd = st.text_input("Display name (optional)", key="ss_display")
                sp = st.text_input("Password", type="password", key="ss_pass")
                sc = st.text_input("Confirm password", type="password", key="ss_conf")
                do_signup = st.form_submit_button("Create account", use_container_width=True)
            if do_signup:
                if not se or not sp:
                    st.error("Email and password are required.")
                elif sp != sc:
                    st.error("Passwords do not match.")
                else:
                    uid, err = CHAT_DB.create_user(se, sp, sd)
                    if err:
                        st.error(err)
                    else:
                        st.session_state["user_id"] = uid
                        st.session_state["user_email"] = (se or "").strip().lower()
                        st.session_state["display_name"] = sd or se
                        st.session_state["session_id"] = "default"
                        st.success("Account created and logged in ✅")
                        _rerun()

    # st.markdown("---")

    # ---------- Existing sidebar features ----------
    _= _init_model_state()

    # st.markdown("Wydot employee bot 🛣️")
    # st.session_state.setdefault("milvus_uri", DEFAULT_MILVUS_URI)
    # st.session_state.setdefault("milvus_token", DEFAULT_MILVUS_TOKEN)
    # st.session_state.setdefault("collection", DEFAULT_COLLECTION)

    # uri = st.text_input("Milvus URI", st.session_state["milvus_uri"])
    # token = st.text_input("Milvus Token", st.session_state["milvus_token"], type="password")
    # collection = st.text_input("Milvus collection", st.session_state["collection"])

    # c1, c2 = st.columns(2)
    # with c1:
    #     if st.button("🔄 Reconnect Milvus"):
    #         st.session_state["milvus_uri"] = uri
    #         st.session_state["milvus_token"] = token
    #         st.session_state["collection"] = collection
    #         get_milvus_collection.clear()
    #         st.success("Reconnected.")
    # with c2:
    #     if st.button("🧹 Clear caches"):
    #         get_milvus_collection.clear()
    #         get_chat_store.clear()
    #         get_genai_client.clear()
    #         st.toast("Cleared Streamlit caches.", icon="🧹")

    st.markdown("---")
    st.markdown("## 🤖 Model selection")

    # Endpoint inputs
    # flash_in = st.text_input(
    #     "Flash tuned endpoint (projects/.../endpoints/...)",
    #     value=st.session_state.get("flash_endpoint") or "",
    #     help="Use the full endpoint resource name. If blank, the app will fall back to base gemini-2.5-flash.",
    # )
    # pro_in = st.text_input(
    #     "Pro tuned endpoint (projects/.../endpoints/...)",
    #     value=st.session_state.get("pro_endpoint") or "",
    #     help="Use the full endpoint resource name. If blank, the app will fall back to base gemini-2.5-pro.",
    # )

    # if st.button("💾 Save endpoints"):
    #     st.session_state["flash_endpoint"] = flash_in or None
    #     st.session_state["pro_endpoint"] = pro_in or None
    #     st.success("Saved.")

    # Choices
    choices = []
    if (st.session_state.get("flash_endpoint") or ENV_TUNED_FLASH):
        choices.append("Tuned: Flash 2.5 (endpoint)")
    if (st.session_state.get("pro_endpoint") or ENV_TUNED_PRO):
        choices.append("Tuned: Pro 2.5 (endpoint)")
    choices.extend(["Base: gemini-2.5-flash", "Base: gemini-2.5-pro"])

    prev_choice = st.session_state.get("model_choice")
    if prev_choice not in choices:
        st.session_state["model_choice"] = choices[0]

    st.session_state["model_choice"] = st.selectbox(
        "Active model",
        choices,
        index=choices.index(st.session_state["model_choice"]),
        help="Switch between your tuned Gemini 2.5 Pro / Flash endpoints or the base models.",
    )

    # Show effective model id
    # current_model_id = get_selected_model_id()
    # st.caption("**Effective model id** (sent to google-genai):")
    # st.code(current_model_id, language="text")

    # ---------- Answer mode selection ----------
    st.markdown("---")
    st.markdown("## 🧠 Answer mode")
    st.markdown("## 🧭 Retrieval options")
    st.session_state["enable_multihop"] = st.checkbox(
        "Enable Multi-hop RAG (decompose complex questions)", 
        value=st.session_state.get("enable_multihop", False),
        help="Break complex questions into 2–4 sub-queries and fuse results for better grounding."
    )

    st.session_state.setdefault("answer_mode", "RAG (Specs)")
    st.session_state["answer_mode"] = st.radio(
        "How should questions be answered?",
        options=[
            "RAG (Specs)",
            "Standard Gemini (no specs)",
            "Web search (Google)",
            "Workspace (Gmail/Drive + Specs)",
        ],
        help=(
            "RAG (Specs): default RAG over Milvus/specs.\n"
            "Standard Gemini (no specs): pure Gemini 2.5 Flash/Pro without spec documents.\n"
            "Web search (Google): enable Google Search tool for real-time answers.\n"
            "Workspace (Gmail/Drive + Specs): experimental, adds Gmail/Drive snippets when ENABLE_WORKSPACE_CONTEXT=1."
        ),
    )

    # st.markdown("---")
    # st.markdown("## 🔑 Auth / Project")
    # st.caption(f"Auth: Service Account ({st.session_state.get('auth_label','unknown')}). Project: {PROJECT}, Location: {LOCATION}")

    # if PDF_BASE_URL:
    #     st.markdown("---")
    #     st.markdown("## 📄 PDF base")
    #     st.caption("Using PDF_BASE_URL to build links (or directory for local PDFs):")
    #     st.code(PDF_BASE_URL, language="text")

    st.markdown("---")
    st.markdown("## 💬 Session")
    default_sid = st.session_state.get("session_id", "default")
    session_id = st.text_input("Session ID", value=default_sid, help="Use a stable ID per user/thread.")
    d1, d2 = st.columns(2)
    with d1:
        if st.button("Set session"):
            st.session_state["session_id"] = session_id
            st.success(f"Session set: {session_id}")
    with d2:
        if st.button("Clear session history"):
            CHAT_DB.clear_session(effective_user_id(), session_id)
            st.toast("Session history cleared for this account.", icon="🧹")
            _rerun()

    # ---------- My conversations ----------
    st.markdown("---")
    st.markdown("## 🗂️ My conversations")

    with st.expander("Browse & manage", expanded=False):
        sessions = CHAT_DB.list_sessions(effective_user_id())

        if not sessions:
            st.caption("No conversations yet. Start chatting to create one, or click **Create** below.")
        else:
            labels = [f"{s['session_id']}  ({s['count']} msgs)" for s in sessions]
            idx = st.selectbox(
                "Select a conversation",
                options=list(range(len(labels))),
                format_func=lambda i: labels[i],
                index=0,
            )
            selected_sid = sessions[idx]["session_id"]

            col_open, col_rename, col_delete = st.columns([0.8, 1.2, 0.8])
            with col_open:
                if st.button("Open", use_container_width=True, key="mc_open"):
                    st.session_state["session_id"] = selected_sid
                    st.success(f"Opened: {selected_sid}")
                    _rerun()

            with col_rename:
                new_name = st.text_input(
                    "Rename to",
                    value=selected_sid,
                    key="mc_rename_to",
                    help="Type the new name and click Rename",
                )
                if st.button("Rename", use_container_width=True, key="mc_rename"):
                    if new_name and new_name != selected_sid:
                        CHAT_DB.rename_session(effective_user_id(), selected_sid, new_name)
                        if st.session_state.get("session_id") == selected_sid:
                            st.session_state["session_id"] = new_name
                        st.toast("Renamed.", icon="✏️")
                        _rerun()

            with col_delete:
                if st.button("Delete", use_container_width=True, key="mc_delete"):
                    CHAT_DB.clear_session(effective_user_id(), selected_sid)
                    st.toast("Deleted conversation.", icon="🗑️")
                    _rerun()

            # Exports (JSONL / CSV)
            msgs = CHAT_DB.all_messages(effective_user_id(), selected_sid)

            jsonl_data = "\n".join(json.dumps(m, ensure_ascii=False) for m in msgs)
            st.download_button(
                "⬇️ Export JSONL",
                data=jsonl_data,
                file_name=f"{selected_sid}.jsonl",
                mime="application/json",
                use_container_width=True,
                key="mc_export_jsonl",
            )

            csv_buf = io.StringIO()
            writer = csv.DictWriter(csv_buf, fieldnames=["id", "role", "content", "ts"])
            writer.writeheader()
            writer.writerows(msgs)
            st.download_button(
                "⬇️ Export CSV",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name=f"{selected_sid}.csv",
                mime="text/csv",
                use_container_width=True,
                key="mc_export_csv",
            )

        st.markdown("---")
        new_sid = st.text_input(
            "New conversation name",
            placeholder="e.g., 2025-11-04 meeting",
            key="mc_new_sid",
        )
        if st.button("Create", use_container_width=True, key="mc_create"):
            name = (new_sid or "").strip() or f"session-{int(time.time())}"
            CHAT_DB.add(effective_user_id(), name, "assistant", "(created conversation)")
            st.session_state["session_id"] = name
            st.success(f"Created and switched to: {name}")
            _rerun()

    # st.markdown("---")
    # st.markdown("## 📝 Extra grounding text (optional)")
    # extracted_text = st.text_area(
    #     "Paste OCR/STT/parsed text if you want it included in the context",
    #     height=140,
    #     placeholder="Paste text extracted from a PDF, image, or audio transcript…"
    # )

# Layout: scrollable left chat, sticky right docs
col_chat, col_docs = st.columns([2, 0.8])

with col_chat:
    st.markdown("<div class='left-scroll'>", unsafe_allow_html=True)
    st.markdown("###  WYDOT Chatbot")
    st.markdown('<div id="chat_top"></div>', unsafe_allow_html=True)

    history_msgs = CHAT_DB.recent(effective_user_id(), st.session_state.get("session_id", "default"), limit=MAX_HISTORY_MSGS)
    for m in history_msgs:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.write(m["content"])

    # Process any submitted query first (from previous form submission)
    user_query = st.session_state.get("pending_query", "")
    uploads_payload = st.session_state.get("pending_uploads", [])
    
    # Clear pending values after reading
    if "pending_query" in st.session_state:
        del st.session_state["pending_query"]
    if "pending_uploads" in st.session_state:
        del st.session_state["pending_uploads"]

    if user_query or uploads_payload:
        answer_mode = st.session_state.get("answer_mode", "RAG (Specs)")

        with st.chat_message("user"):
            if user_query:
                st.write(user_query)
            if uploads_payload:
                st.caption("Attachments added.")

        # For RAG and Workspace modes, pre-populate last_sources for the right panel (multi-hop aware)
        if user_query and user_query.strip() and answer_mode in ("RAG (Specs)", "Workspace (Gmail/Drive + Specs)"):
            _, live_sources = use_multihop_if_enabled(globals(), user_query, default_k=5)
        else:
            live_sources = []
        st.session_state["last_sources"] = live_sources

        # Define extracted_text if not already defined (was commented out in sidebar)
        extracted_text = st.session_state.get("extracted_text", "")
        
        with st.chat_message("assistant"):
            placeholder = st.empty()
            current = ""
            for partial_text, sources_stream in resultDocuments_streaming(
                query=user_query or "",
                extracted_text=extracted_text,
                uploads=uploads_payload,
                session_id=st.session_state.get("session_id", "default"),
            ):
                current = partial_text
                placeholder.markdown(current)

        st.session_state["last_answer"] = current
        st.session_state["last_question"] = user_query or ""
        
        # Scroll to bottom after response completes
        st.markdown('<div id="chat_bottom"></div>', unsafe_allow_html=True)
        components.html(
            """
            <script>
            setTimeout(function() {
                window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
                const bottomEl = parent.document.getElementById('chat_bottom');
                if (bottomEl) {
                    bottomEl.scrollIntoView({behavior: 'smooth', block: 'end'});
                }
            }, 300);
            </script>
            """,
            height=0,
        )

    # Render composer at the bottom (AFTER all messages)
    user_query_new, uploads_payload_new = render_chat_composer()
    
    # If new query submitted, store it for next run
    if user_query_new or uploads_payload_new:
        st.session_state["pending_query"] = user_query_new
        st.session_state["pending_uploads"] = uploads_payload_new
        st.rerun()

    # Scroll to composer after rendering
    st.markdown('<div id="composer_anchor"></div>', unsafe_allow_html=True)
    components.html(
        """
        <script>
        setTimeout(function() {
            const anchor = parent.document.getElementById('composer_anchor');
            if (anchor) {
                anchor.scrollIntoView({behavior: 'instant', block: 'end'});
            }
            window.scrollTo({top: document.body.scrollHeight, behavior: 'instant'});
        }, 100);
        </script>
        """,
        height=0,
    )
    st.markdown("</div>", unsafe_allow_html=True)
if "history_2010" not in st.session_state:
    # store as list of dicts or tuples
    st.session_state["history_2010"] = []
def format_history_2010(turns) -> str:
    if not turns:
        return "(no previous turns)"
    lines = []
    # most recent first
    for t in reversed(turns):
        role = t.get("role", "user")
        if role == "user":
            prefix = "User"
        else:
            prefix = "WYDOT 2010 assistant"
        lines.append(f"{prefix}: {t.get('content','')}")
    return "\n".join(lines)

with col_docs:
    st.markdown("<div class='right-sticky'>", unsafe_allow_html=True)
    st.markdown("### Retrieved Documents")
    st.caption("Top-k chunks retrieved for the latest question (from Milvus/specs).")
    sources = st.session_state.get("last_sources", [])
    _ = render_citations_bar(sources)  # Suppress None return value
    if st.session_state.get("citations_filter"):
        sources = apply_citations_filter(sources)
    answer_mode = st.session_state.get("answer_mode", "RAG (Specs)")
    st.session_state.setdefault("last_web_sources", [])
    st.session_state.setdefault("web_search_widget_html", "")

    preview_url = st.session_state.get("pdf_preview_url")
    if preview_url:
        st.markdown("#### 🔎 Inline PDF preview")
        components.iframe(preview_url, height=600)
        st.markdown("---")

    if not sources:
        st.info("Ask a question in **RAG (Specs)** or **Workspace** mode to see retrieved documents here.")
    else:
        render_section_aware_preview(sources, build_pdf_url)
        
        
# =========================================================
# === 2010/2012 Helper & On-Demand Answer/Compare UI ===
# =========================================================

# st.markdown("---")
st.markdown("### 2010 & 2012 Specifications ")

COLL_2010 = os.getenv("MILVUS_COLLECTION_2010", "wydotspec_llamaparse_2010")
last_q = st.session_state.get("last_question", "")

if "history_2010" not in st.session_state:
    st.session_state["history_2010"] = []

def format_history_2010(turns) -> str:
    if not turns:
        return "(no previous turns)"
    lines = []
    for t in reversed(turns):
        prefix = "User" if t.get("role") == "user" else "WYDOT 2010 assistant"
        lines.append(f"{prefix}: {t.get('content','')}")
    return "\n".join(lines)

def _answer_from_2010_specs(question: str):
    """RAG + Multimodal retrieval strictly from 2010 Specs using gemini-2.5-flash base model."""
    if not question or not question.strip():
        return "", []

    old_coll = st.session_state.get("collection", DEFAULT_COLLECTION)
    st.session_state["collection"] = COLL_2010
    try:
        ctx_text, srcs = use_multihop_if_enabled(globals(), question.strip(), default_k=5)
    finally:
        st.session_state["collection"] = old_coll

    hist_txt = format_history_2010(st.session_state["history_2010"])

    prompt = (
        "You are a WYDOT assistant answering STRICTLY from the **2010 Specifications**.\n"
        "Use only the provided context (retrieved below) and conversation history.\n"
        "If the context does not contain the answer, respond with:\n"
        "'The 2010 specifications do not cover this information.'\n\n"
        f"Context (from 2010 Specs):\n{ctx_text}\n\n"
        f"Conversation so far:\n{hist_txt}\n\n"
        f"User question: {question}\n"
        "JUST PROVIDE THE ANSWER IN ENGLISH WITHOUT ``` AND NOTHING ELSE."
    )

    client = get_genai_client()
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[gtypes.Content(role="user", parts=[gtypes.Part.from_text(text=prompt)])],
        config=gtypes.GenerateContentConfig(
            temperature=0.2,
            top_p=1,
            max_output_tokens=4096,
            safety_settings=[
                gtypes.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                gtypes.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                gtypes.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                gtypes.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ],
        ),
    )

    text = (getattr(resp, "text", "") or "").strip()
    return text, srcs

def _compare_2010_2021(question: str):
    """Compare 2010 vs 2021 specs using gemini-2.5-flash on demand."""
    if not question or not question.strip():
        return "No question to compare.", [], []

    old_coll = st.session_state.get("collection", DEFAULT_COLLECTION)

    # Retrieve 2010 context
    st.session_state["collection"] = COLL_2010
    try:
        ctx2010, src2010 = use_multihop_if_enabled(globals(), question.strip(), default_k=5)
    finally:
        st.session_state["collection"] = old_coll

    # Retrieve 2021 (default) context
    st.session_state["collection"] = DEFAULT_COLLECTION
    try:
        ctx2021, src2021 = use_multihop_if_enabled(globals(), question.strip(), default_k=5)
    finally:
        st.session_state["collection"] = old_coll

    combined_ctx = (
        f"=== 2010 Specifications ===\n{ctx2010}\n\n"
        f"=== 2021 Specifications ===\n{ctx2021}"
    )

    prompt = (
        "You are a WYDOT assistant. Compare the 2010 vs 2021 specifications for the question below.\n"
        "Return only:\n"
        "- Similarities: brief bullets\n"
        "- Differences: brief bullets\n"
        "- Aggregate answer: concise recommendation/summary\n"
        "Do NOT quote long passages; synthesize.\n\n"
        f"Context:\n{combined_ctx}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    client = get_genai_client()
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[gtypes.Content(role="user", parts=[gtypes.Part.from_text(text=prompt)])],
        config=gtypes.GenerateContentConfig(
            temperature=0.2,
            top_p=0.95,
            max_output_tokens=1200,
            safety_settings=[
                gtypes.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                gtypes.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                gtypes.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                gtypes.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ],
        ),
    )

    answer = (getattr(resp, "text", "") or "").strip()
    return answer, src2010, src2021

# On-demand 2010 answer
with st.expander("2010 Specifications answer", expanded=False):
    if not last_q:
        st.info("Ask a question above, then click to answer from 2010 specs.")
    elif st.button("Run 2010 answer", key="run_2010_answer"):
        ans2010, src2010 = _answer_from_2010_specs(last_q)
        if ans2010:
            st.markdown("#### 🧾 2010 Specs Answer")
            st.markdown(ans2010)
            st.session_state["history_2010"].append({"role": "user", "content": last_q})
            st.session_state["history_2010"].append({"role": "assistant", "content": ans2010})
            with st.expander(" Retrieved 2010 Spec Chunks", expanded=False):
                render_section_aware_preview(src2010, build_pdf_url)
        else:
            st.warning("No answer found within the 2010 specifications.")

# On-demand compare 2010 vs 2021
with st.expander("Compare 2010 vs 2021", expanded=False):
    if not last_q:
        st.info("Ask a question above, then click to compare 2010 vs 2021 specs.")
    elif st.button("Compare now", key="compare_2010_2021"):
        comp_ans, src2010c, src2021c = _compare_2010_2021(last_q)
        st.markdown("#### Comparison (2010 vs 2021)")
        st.markdown(comp_ans)
        with st.expander("Sources: 2010", expanded=False):
            if src2010c:
                render_section_aware_preview(src2010c, build_pdf_url)
            else:
                st.caption("No 2010 chunks retrieved.")
        with st.expander("Sources: 2021", expanded=False):
            if src2021c:
                render_section_aware_preview(src2021c, build_pdf_url)
            else:
                st.caption("No 2021 chunks retrieved.")


# =========================================================
# EXTENSIONS (added below original UI without changing it)
# =========================================================

# --- Initialize feature helpers ---
if "rbac" not in st.session_state:
    st.session_state["rbac"] = RBAC(CHAT_DB_PATH)
if "validator_db" not in st.session_state:
    st.session_state["validator_db"] = ValidatorDB(CHAT_DB_PATH)
if "vfusion" not in st.session_state:
    st.session_state["vfusion"] = VectorFusion(CHAT_DB_PATH, embed_text_eval)
    _= st.session_state["vfusion"].seed_demo()
# if "offline_cache" not in st.session_state:
#     st.session_state["offline_cache"] = OfflineCache(os.path.join(tempfile.gettempdir(),"wydot_offline_cache.json"), embed_text_eval)
#     st.session_state["offline_cache"].seed_demo()
if "offline_cache" not in st.session_state:
    st.session_state["offline_cache"] = OfflineCache(embed_text_eval)

if "kg" not in st.session_state:
    kg = KG(); _= kg.build_demo(); st.session_state["kg"] = kg

rbac: RBAC = st.session_state["rbac"]
vdb: ValidatorDB = st.session_state["validator_db"]
vfusion: VectorFusion = st.session_state["vfusion"]
ocache: OfflineCache = st.session_state["offline_cache"]
kg: KG = st.session_state["kg"]

# --- Sidebar: Feature toggles & RBAC ---
with st.sidebar:
    st.markdown("---")
    st.markdown("## 🧩 Extensions")
    enable_fusion = st.checkbox("Enable Vector Fusion (Milvus + SQL)", value=True)
    enable_selfval = st.checkbox("Enable Self-Validation (Critique & Revise)", value=False)
    enable_offline = st.checkbox("Use Offline Cache (demo)", value=False)
    st.caption("These affect the right-panel helpers shown below.")

    st.markdown("---")
    st.markdown("## 🔐 Roles (RBAC)")
    uid = effective_user_id()
    colR1, colR2 = st.columns(2)
    with colR1:
        if st.button("Grant workspace_reader"):
            rbac.grant(uid, "workspace_reader"); rbac.log(uid,"grant_role",{"role":"workspace_reader"}); st.success("Granted.")
    with colR2:
        if st.button("Revoke workspace_reader"):
            rbac.revoke(uid, "workspace_reader"); rbac.log(uid,"revoke_role",{"role":"workspace_reader"}); st.success("Revoked.")

    st.caption("Workspace mode requires role `workspace_reader`.")

# --- Right column: add extensions UI ---

with col_docs:
    


    if enable_offline and last_q:
        off_hits = ocache.search(last_q, k=5)
    else:
        off_hits = []

    

    off_hits = st.session_state["offline_cache"].search(last_q, k=5) if last_q else []

# 2. Render UI
    with st.expander("Offline cache hits", expanded=False):
        if not last_q:
            st.info("Ask a question first.")
        else:
            if off_hits:
                for i, s in enumerate(off_hits,1):
                    st.write(
                        f"{i}. [{s.get('role')}] "
                        f"{s.get('preview','')[:200]}"
                    )
            else:
                st.caption("No offline matches.")


