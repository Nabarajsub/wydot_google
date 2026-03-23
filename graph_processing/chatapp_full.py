#!/usr/bin/env python3
"""
WYDOT GraphRAG Chatbot — Full Graph Edition
============================================
All features from chatapp_gemini.py PLUS graph-enhanced retrieval:
- Hybrid keyword + vector search (proper noun detection)
- Phase 4+5 document relationship context (PART_OF, SUPPLEMENTS, TEACHES, etc.)
- Contextual inference for person/entity identity queries
- Native Login (SQLite-backed)
- Multimodal Support (Images, Audio, Video, PDF) via Gemini
- Multi-hop Reasoning + Cypher Analytics
- Model Selection (Mistral / Gemini / OpenRouter)
- Citation Linking [SOURCE_X] -> Evidence Panel
- Audio Input Support
- FlashRank Reranking, HyDE Query Expansion

Run from graph_processing/:
  chainlit run chatapp_full.py -w
"""

# === CLOUD RUN DIAGNOSTIC LOGGING ===
import os
import sys

# Add parent directory to sys.path so modules from project root are importable
# when running from graph_processing/ subdirectory
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

print("🚀 App Startup Sequence Initiated", flush=True)
print(f"📍 Current Working Directory: {os.getcwd()}", flush=True)
print(f"🛠️ K_SERVICE Detection: {os.getenv('K_SERVICE', 'Not in Cloud Run')}", flush=True)
print(f"📦 PORT: {os.getenv('PORT', 'Not Set')}", flush=True)

# Preserve Cloud Run env vars before .env loading can overwrite them
_cloud_run_port = os.getenv("PORT") if os.getenv("K_SERVICE") else None
# DATABASE_URL="" in YAML forces SQLite; don't let .env overwrite it with PostgreSQL URL
_cloud_run_db_url = os.environ.get("DATABASE_URL") if os.getenv("K_SERVICE") else None

# Create temp directories at runtime (Cloud Run may reset /tmp between requests)
import os as _os
for _d in ["/tmp/.files", "/tmp/.chainlit"]:
    _os.makedirs(_d, exist_ok=True)

# Force Chainlit to use /tmp for everything (config.toml override)
# ONLY if running in Cloud Run (detected via env var or if needed)
# For local dev, we want persistence.
if _os.getenv("K_SERVICE"): # Cloud Run sets this
    _os.environ.setdefault("CHAINLIT_FILES_DIRECTORY", "/tmp/.files")
    _os.environ.setdefault("CHAINLIT_DB_FILE", "/tmp/chainlit_gemini.db")
    print("✅ Cloud Run environment detected. Files redirected to /tmp.")
else:
    # Local: use project dir for persistence
    _os.environ.setdefault("CHAINLIT_DB_FILE", "chainlit_gemini.db")
    print("🏠 Local environment detected.")
# === END CLOUD RUN SETUP ===

print("📦 Importing core libraries...")

# =========================================================
# IMPORTS & CONFIGURATION
# =========================================================

from typing import List, Dict, Any, Optional, Tuple
import os
import re
import datetime
import json
import time
import asyncio
import tempfile
import traceback
import urllib.parse
from dotenv import load_dotenv

import httpx
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from flashrank import Ranker, RerankRequest

# GraphRAG / LangChain imports
# Heavy imports moved to lazy loaders below to prevent Cloud Run startup timeouts
from langchain_core.documents import Document
from neo4j import GraphDatabase
from collections import OrderedDict

# ── Agentic multi-agent orchestrator (lazy-loaded) ──
_agentic_available = False
try:
    from agentic_solution.orchestrator import (
        run_orchestrator_async,
        TOOL_DISPLAY_NAMES,
        TOOL_AGENT_LABELS,
    )
    _agentic_available = True
    print("✅ Agentic orchestrator loaded", flush=True)
except ImportError as _agentic_err:
    print(f"⚠️ Agentic orchestrator not available: {_agentic_err}", flush=True)

# Module-level cache: maps Chainlit message ID -> {question, answer, user_email, sources}
# Used by upsert_feedback callback (which runs without Chainlit session context)
_feedback_context: OrderedDict = OrderedDict()
_FEEDBACK_CTX_MAX = 500  # keep last 500 entries to avoid memory bloat

# --- CUSTOM AUTH ENDPOINTS ---
from chainlit.server import app

class RegisterReq(BaseModel):
    email: str
    password: str

class VerifyReq(BaseModel):
    email: str
    code: str

@app.post("/auth/register")
async def register(req: RegisterReq):
    print(f"👤 [API] Registration attempt: {req.email}")
    if not CHAT_DB:
        return JSONResponse(status_code=500, content={"error": "Database not initialized"})
    
    uid, error = CHAT_DB.create_user(req.email, req.password, req.email.split('@')[0])
    if uid:
        print(f"✅ [API] Registration successful for {req.email}")
        return {"message": "Registration successful. Please verify.", "dev_code": "123456"}
    print(f"❌ [API] Registration failed for {req.email}: {error}")
    return JSONResponse(status_code=400, content={"error": error})

@app.post("/auth/verify")
async def verify_code(req: VerifyReq):
    print(f"🔑 [API] Verification attempt: {req.email}")
    if not CHAT_DB:
        return JSONResponse(status_code=500, content={"error": "Database not initialized"})
    
    try:
        user = CHAT_DB.get_user_by_email(req.email)
        if not user:
            return JSONResponse(status_code=404, content={"error": "User not found"})
        CHAT_DB.set_verified(user["id"])
        print(f"✅ [API] User {req.email} verified.")
        return {"status": "success", "message": "Account verified!"}
    except Exception as e:
        print(f"🔥 [API] Verification error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
# --- END CUSTOM AUTH ---

# --- SOURCE CONTENT ENDPOINT ---
# Chainlit's catch-all GET route (/{full_path:path}) is registered before our custom
# routes, so @app.get() routes get intercepted. Using app.mount() for a sub-app
# takes priority over router routes in Starlette/FastAPI.
from fastapi import FastAPI as _FastAPI
_api_app = _FastAPI()

@_api_app.get("/source/{thread_id}/{element_id}")
async def get_source_content(thread_id: str, element_id: str):
    """Serve source element content as plain text for Chainlit's frontend to render."""
    if not CHAT_DB:
        raise HTTPException(status_code=500, detail="Database not initialized")

    if not element_id.startswith("src_"):
        raise HTTPException(status_code=404, detail="Not a source element")

    try:
        parts = element_id.split("_")
        if len(parts) < 3:
            raise HTTPException(status_code=404, detail="Invalid element ID")
        ts_str = parts[1]
        src_idx = int(parts[2])

        # Find the thread
        thread_info = CHAT_DB.get_session_by_id(thread_id)
        if not thread_info:
            thread_info = CHAT_DB.get_session_by_id(f"thread_{thread_id}")
        if not thread_info:
            raise HTTPException(status_code=404, detail="Thread not found")

        uid = int(thread_info["userId"])
        actual_session_id = thread_id
        if not CHAT_DB.get_session_by_id(actual_session_id):
            actual_session_id = f"thread_{thread_id}"

        msgs = CHAT_DB.get_recent(uid, actual_session_id, MAX_HISTORY_MSGS)

        # Find matching message with sources
        target_msg = None
        for m in msgs:
            if not m.get("sources"):
                continue
            m_ts = m.get("ts")
            if m_ts is not None and str(int(float(m_ts))) == ts_str:
                target_msg = m
                break

        # Fallback: last assistant message with sources
        if not target_msg:
            msgs_with_sources = [m for m in msgs if m.get("sources") and m.get("role") == "assistant"]
            if msgs_with_sources:
                target_msg = msgs_with_sources[-1]

        if not target_msg:
            raise HTTPException(status_code=404, detail="Message not found")

        msg_sources = target_msg.get("sources")
        if not msg_sources or src_idx >= len(msg_sources):
            raise HTTPException(status_code=404, detail="Source index out of range")

        src = msg_sources[src_idx]
        content = (
            f"**Source {src.get('index', '?')}: {src.get('title', 'Unknown')}**\n\n"
            f"**File:** [{src.get('source', 'File')}]({src.get('url', '#')})\n"
            f"**Page:** {src.get('page', 'N/A')}\n"
            f"**Section:** {src.get('section', 'N/A')}\n"
            f"**Year:** {src.get('year', 'N/A')}\n\n"
            f"**Preview:**\n{src.get('preview', '')}"
        )
        return PlainTextResponse(content=content, media_type="text/plain")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[SOURCE ENDPOINT] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Insert the mount BEFORE Chainlit's catch-all route (/{full_path:path}).
# app.mount() appends to the end of routes, which comes AFTER the catch-all.
# We must insert it before the catch-all so it gets matched first.
from starlette.routing import Mount as _Mount
_catch_all_idx = next(
    (i for i, r in enumerate(app.routes)
     if hasattr(r, "path") and "{full_path" in getattr(r, "path", "")),
    len(app.routes)
)
@_api_app.get("/health/neo4j")
async def neo4j_keepalive():
    """Keep-alive endpoint: runs a cheap Cypher query to prevent AuraDB Free from pausing."""
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run("RETURN 1 AS heartbeat")
            result.single()
        return {"status": "ok", "neo4j": "alive"}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})

@_api_app.get("/health")
async def health_check():
    """Comprehensive health endpoint for monitoring."""
    import time as _time
    neo4j_status = "unknown"
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            session.run("RETURN 1 AS heartbeat").single()
        neo4j_status = "healthy"
    except Exception as e:
        neo4j_status = f"unhealthy: {str(e)[:100]}"

    uptime = _time.time() - _app_start_time
    return {
        "status": "healthy" if neo4j_status == "healthy" else "degraded",
        "neo4j": neo4j_status,
        "uptime_seconds": round(uptime, 1),
        "version": os.getenv("K_REVISION", "local"),
        "service": os.getenv("K_SERVICE", "local"),
    }

app.routes.insert(_catch_all_idx, _Mount("/api", app=_api_app))
# --- END SOURCE CONTENT ENDPOINT ---

# --- ADMIN ROUTES (unified service) ---
# Mount the admin FastAPI sub-app at /admin/* for document ingestion & KG management.
# Uses the same insert-before-catch-all pattern as /api above.
try:
    from admin_routes import admin_app as _admin_app
    _catch_all_idx_2 = next(
        (i for i, r in enumerate(app.routes)
         if hasattr(r, "path") and "{full_path" in getattr(r, "path", "")),
        len(app.routes)
    )
    app.routes.insert(_catch_all_idx_2, _Mount("/admin", app=_admin_app))
    print("✅ Admin routes mounted at /admin/*")
except ImportError as _admin_err:
    print(f"⚠️ Admin routes not available (admin_routes.py not found): {_admin_err}")
except Exception as _admin_err:
    print(f"⚠️ Admin routes failed to mount: {_admin_err}")
# --- END ADMIN ROUTES ---

# Google Vertex AI / Gemini (imports moved to lazy loaders)
GEMINI_AVAILABLE = True # Assume true, check imports lazily

# Google Cloud Storage - Public URLs
def generate_public_url(gcs_uri: str) -> str:
    """Generate a public URL for a GCS object."""
    try:
        if not gcs_uri or not gcs_uri.startswith("gs://"):
            return ""
        # Parse gs://bucket/blob_name
        parts = gcs_uri.replace("gs://", "").split("/", 1)
        if len(parts) != 2:
            return ""
        bucket_name, blob_name = parts
        # Decode first to avoid double-encoding (blob names in Neo4j may already be URL-encoded)
        decoded_blob = urllib.parse.unquote(blob_name)
        encoded_blob = urllib.parse.quote(decoded_blob, safe="/,")
        return f"https://storage.googleapis.com/{bucket_name}/{encoded_blob}"
    except Exception as e:
        print(f"Error generating public URL: {e}")
        return ""

# Load environment variables from DOTENV_PATH or .env
# When running from graph_processing/, also check parent dir
dotenv_path = os.getenv("DOTENV_PATH", ".env")
_parent_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if os.path.exists(dotenv_path):
    print(f"📂 Loading environment from: {dotenv_path}")
    load_dotenv(dotenv_path, override=True)
elif os.path.exists(_parent_env):
    print(f"📂 Loading environment from parent dir: {_parent_env}")
    load_dotenv(_parent_env, override=True)
else:
    print(f"⚠️ DOTENV_PATH not found: {dotenv_path}, trying default .env")
    load_dotenv(override=True)

# Prevent Chainlit from trying to connect to PostgreSQL in local mode
if not os.getenv("K_SERVICE"):
    _db_url = os.getenv("DATABASE_URL", "")
    if "cloudsql" in _db_url or "psycopg2" in _db_url:
        os.environ["DATABASE_URL"] = ""
        print("🏠 Local mode: cleared PostgreSQL DATABASE_URL")

# Restore Cloud Run env vars if .env overwrote them
if _cloud_run_port:
    os.environ["PORT"] = _cloud_run_port
    print(f"🔒 PORT restored to Cloud Run value: {_cloud_run_port}")
if _cloud_run_db_url is not None:
    os.environ["DATABASE_URL"] = _cloud_run_db_url
    print(f"🔒 DATABASE_URL restored to Cloud Run value: '{_cloud_run_db_url}' (empty=SQLite)")

print("🔑 Validating Authentication...")
# Enable login screen: Chainlit only shows auth when CHAINLIT_AUTH_SECRET is set
if not os.getenv("CHAINLIT_AUTH_SECRET"):
    os.environ["CHAINLIT_AUTH_SECRET"] = "wydot-dev-secret-change-in-production"

# =========================================================
# CONFIGURATION
# =========================================================

NEO4J_URI = os.getenv("NEO4J_URI_GEMINI", "neo4j+s://1c9edfe6.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME_GEMINI", "1c9edfe6")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD_GEMINI", "IlZpB7BG3sM34FQ5d_Juv5CidvCHvsMnoLkXHW18CSA")
print(f"🔗 Neo4j Config: {NEO4J_URI} (User: {NEO4J_USERNAME})")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE_GEMINI", "1c9edfe6")

# --- Neo4j Connection Pooling Singleton ---
_neo4j_driver = None
_app_start_time = time.time()

def get_neo4j_driver():
    """Return a singleton Neo4j driver instance for connection pooling."""
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        print("🔗 Neo4j driver singleton created (connection pooling enabled)")
    return _neo4j_driver
NEO4J_INDEX_DEFAULT = os.getenv("NEO4J_INDEX_DEFAULT_GEMINI", "wydot_gemini_index")
NEO4J_INDEX_2021 = os.getenv("NEO4J_INDEX_2021", "wydot_vector_index_2021")

print("💬 Initializing API Keys...")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Mapping of model names (UI) to OpenRouter IDs
OPENROUTER_MODELS = {
    "GPT-5.2 Pro": "openai/gpt-5.2-pro",
    "GPT-5.2 Chat": "openai/gpt-5.2-chat",
    "Claude Opus 4.6": "anthropic/claude-opus-4.6",
    "Claude Sonnet 4.6": "anthropic/claude-sonnet-4.6",
    "Gemini 3 Flash": "google/gemini-3-flash",
    "MiniMax M2.5": "minimax/minimax-m2.5",
    "Step 3.5 Flash": "stepfun/step-3.5-flash",
    "Llama 3.1 405B": "meta-llama/llama-3.1-405b-instruct",
    "Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct",
    "DeepSeek V3": "deepseek/deepseek-chat-v3",
    "DeepSeek V3.2 Speciale": "deepseek/deepseek-v3.2-speciale",
    "DeepSeek Coder V2": "deepseek/deepseek-coder-v2",
    "Qwen 2.5 72B": "qwen/qwen-2.5-72b-instruct",
    "Mistral Nemo": "mistralai/mistral-nemo"
}

MAX_HISTORY_MSGS = 20
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "10"))
FETCH_K = int(os.getenv("FETCH_K", "15"))

# Global Ranker Instance (lazy initialization recommended if memory is tight, 
# but for local we'll go direct)
_RANKER_INSTANCE = None

def get_reranker():
    global _RANKER_INSTANCE
    if _RANKER_INSTANCE is None:
        try:
            print("🚀 Initializing FlashRank Ranker...")
            _RANKER_INSTANCE = Ranker()
            print("✅ FlashRank Ranker ready.")
        except Exception as e:
            print(f"❌ Failed to initialize FlashRank: {e}")
    return _RANKER_INSTANCE

# Debug: Print configuration (hide password)
print("=" * 60)
print("🔧 NEO4J CONFIGURATION:")
print(f"  URI: {NEO4J_URI}")
print(f"  Username: {NEO4J_USERNAME}")
print(f"  Password: {'*' * len(NEO4J_PASSWORD) if NEO4J_PASSWORD else 'NOT SET'}")
print(f"  Database: {NEO4J_DATABASE}")
print(f"  Index: {NEO4J_INDEX_DEFAULT}")
print(f"🔧 API KEYS:")
print(f"  MISTRAL_API_KEY: {'SET' if MISTRAL_API_KEY else 'NOT SET'}")
print(f"  GEMINI_API_KEY: {'SET' if GEMINI_API_KEY else 'NOT SET'}")
print(f"  OPENROUTER_API_KEY: {'SET' if OPENROUTER_API_KEY else 'NOT SET'}")
if OPENROUTER_API_KEY:
    print(f"  OPENROUTER_API_KEY (debug): {str(OPENROUTER_API_KEY)[:8]}...")
print("=" * 60)

# =========================================================
# DATABASE / AUTH STORE (abstracted for local SQLite → Cloud SQL)
# =========================================================

# Local Mode detection: If running locally and DATABASE_URL points to Cloud SQL,
# ignore it so we use local SQLite instead of trying (and failing) to connect to Cloud SQL.
if not os.getenv("K_SERVICE") and "cloudsql" in os.getenv("DATABASE_URL", ""):
    print("🏠 Local mode detected: ignoring Cloud SQL DATABASE_URL. Using SQLite.")
    os.environ["DATABASE_URL"] = ""

print(f"🏗️ Initializing CHAT_DB...", flush=True)
print(f"   DATABASE_URL at init time: '{os.getenv('DATABASE_URL', '(not set)')[:60]}...'", flush=True)
try:
    from utils.chat_history_store import get_chat_history_store
    CHAT_DB = get_chat_history_store()
    print(f"✅ CHAT_DB initialized: {type(CHAT_DB).__name__}", flush=True)
except ImportError:
    try:
        from chat_history_store import get_chat_history_store  # when run from repo root
        CHAT_DB = get_chat_history_store()
        print(f"✅ CHAT_DB initialized (alt import): {type(CHAT_DB).__name__}", flush=True)
    except Exception as _e:
        print(f"⚠️ Failure in chat history store import: {_e}", flush=True)
        import traceback; traceback.print_exc()
        CHAT_DB = None
except Exception as _e:
    print(f"⚠️ Chat history store init failed: {_e}", flush=True)
    import traceback; traceback.print_exc()
    CHAT_DB = None

print("📈 Setting up Telemetry and Evaluation...")
# Online RAG telemetry (local SQLite; Cloud: BigQuery later)
try:
    from utils import telemetry
    from utils import evaluation as online_eval
except ImportError:
    try:
        import telemetry
    except ImportError:
        # Create a stub telemetry module so the app can still start
        class _TelemetryStub:
            def record_request(self, **kwargs): pass
        telemetry = _TelemetryStub()
        print("⚠️ Telemetry module not available, using stub")

# Conversation memory: Redis (fast) with DB fallback
try:
    from utils import conversation_memory as conv_mem
except ImportError:
    import conversation_memory as conv_mem

# =========================================================
# CHAINLIT DATA LAYER (for Sidebar History)
# =========================================================

from chainlit.data.base import BaseDataLayer
from chainlit.types import Pagination, ThreadFilter, PaginatedResponse, ThreadDict, ThreadDict
from typing import Optional, List, Dict, Any

class WydotDataLayer(BaseDataLayer):
    """Custom Data Layer to adapt SQLiteChatHistoryStore for Chainlit UI."""
    
    async def get_user(self, identifier: str):
        try:
            user_dict = CHAT_DB.get_user_by_email(identifier)
            if not user_dict:
                return None
                
            # Format timestamp as ISO 8601 for Chainlit stability
            ts = user_dict.get("created_at", time.time())
            if isinstance(ts, (int, float)):
                created_at = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            else:
                created_at = str(ts)

            is_guest = (identifier == "guest@app.local")
            return cl.PersistedUser(
                id=str(user_dict["id"]),
                createdAt=created_at,
                identifier=user_dict["email"],
                display_name=user_dict["name"],
                metadata={"db_id": user_dict["id"], "name": user_dict["name"], "verified": True, "is_guest": is_guest}
            )
        except Exception:
            return None

    async def create_user(self, user: cl.User):
        """Called during registration or first login."""
        try:
            import time
            db_id = user.metadata.get("db_id")
            
            if not db_id:
                password = getattr(user, "password", None)
                if password:
                    uid, error = CHAT_DB.create_user(user.identifier, password, user.display_name)
                    if uid:
                        db_id = uid
                    else:
                        return None
                else:
                    db_id = str(int(time.time()*1000)) # Fallback
            
            created_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            is_guest = (user.identifier == "guest@app.local")
            
            return cl.PersistedUser(
                id=str(db_id),
                createdAt=created_at,
                identifier=user.identifier,
                display_name=user.display_name or user.identifier,
                metadata={"db_id": db_id, "name": user.display_name or user.identifier, "verified": True, "is_guest": is_guest}
            )
        except Exception:
            return None

    async def list_threads(self, pagination: Pagination, filters: ThreadFilter) -> PaginatedResponse[ThreadDict]:
        user_id = filters.userId
        search_keyword = filters.search  # Get search term from filters
        
        if not user_id:
            return PaginatedResponse(data=[], pageInfo=cl.types.PageInfo(hasNextPage=False, endCursor=None, startCursor=None))
            
        try:
            uid = int(user_id)
        except ValueError:
             return PaginatedResponse(data=[], pageInfo=cl.types.PageInfo(hasNextPage=False, endCursor=None, startCursor=None))

        # Recover user identifier to check if guest
        user_dict = CHAT_DB.get_user_by_id(uid)
        user_email = user_dict.get("email", "") if user_dict else ""
        
        # If guest, return empty to maintain stateless UI as requested
        if user_email == "guest@app.local":
            return PaginatedResponse(
                data=[],
                pageInfo=cl.types.PageInfo(hasNextPage=False, endCursor=None, startCursor=None)
            )

        # For registered users, fetch and return real threads
        sessions = CHAT_DB.get_user_sessions(uid, search_term=search_keyword)
        threads: List[ThreadDict] = []
        for s in sessions:
            try:
                ts = s["createdAt"]
                if isinstance(ts, (int, float)):
                    # ISO 8601 string is the most robust format for Chainlit's React frontend
                    created_at = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
                else:
                    created_at = str(ts)
            except Exception:
                traceback.print_exc()
                created_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

            threads.append({
                "id": str(s["id"]),
                "createdAt": created_at, 
                "name": str(s["name"]),
                "userId": str(uid),
                "userIdentifier": str(user_email),
                "tags": [],
                "metadata": {},
                "steps": [],
                "elements": [],
            })
            
        print(f"[DEBUG list_threads] Result count: {len(threads)}")
        return PaginatedResponse(
            data=threads,
            pageInfo=cl.types.PageInfo(hasNextPage=False, endCursor=None, startCursor=None)
        )

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        import datetime
        import uuid as _uuid
        # Try to find thread by ID directly
        thread_info = CHAT_DB.get_session_by_id(thread_id)
        
        # Backward compat: try with thread_ prefix if not found
        actual_session_id = thread_id
        if not thread_info:
            actual_session_id = f"thread_{thread_id}"
            thread_info = CHAT_DB.get_session_by_id(actual_session_id)
        
        if not thread_info:
            return None
            
        ts = thread_info["createdAt"]
        created_at = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z") if isinstance(ts, (int, float)) else str(ts)

        # Resolve user email for userIdentifier (Chainlit checks this for authorization)
        user_email = ""
        uid = None
        try:
            uid = int(thread_info["userId"])
            user_dict = CHAT_DB.get_user_by_id(uid)
            if user_dict:
                user_email = user_dict.get("email", "")
        except (ValueError, TypeError):
            pass

        # Build steps from stored messages so the frontend can render them
        steps = []
        elements = []
        if uid is not None:
            raw_msgs = CHAT_DB.get_recent(uid, actual_session_id, MAX_HISTORY_MSGS)
            # raw_msgs is ordered newest first (DESC) by get_recent, but steps should be chronological?
            # get_recent returns reversed rows (oldest first) at the end: rows.reverse().
            # So raw_msgs are oldest first. Correct.
            
            for i, msg in enumerate(raw_msgs):
                role = msg.get("role", "assistant")
                content = msg.get("content", "")
                step_name = "user" if role == "user" else "Assistant"
                msg_ts = msg.get("ts")
                
                # Format timestamp as ISO 8601 string
                if isinstance(msg_ts, (int, float)):
                    step_created = datetime.datetime.fromtimestamp(msg_ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
                else:
                    step_created = created_at
                
                step_id = msg.get("id") or f"step_{actual_session_id}_{i}"
                step_elements = [] 
                
                # Check for sources to restore elements
                msg_sources = msg.get("sources")
                if msg_sources and isinstance(msg_sources, list):
                    for src_idx, src in enumerate(msg_sources):
                        safe_ts = int(msg_ts) if isinstance(msg_ts, (int, float)) else 0
                        src_id = f"src_{safe_ts}_{src_idx}"
                        elem = {
                            "id": src_id,
                            "threadId": thread_id,
                            "type": "text",
                            "name": f"Source {src.get('index', '?')}",
                            "display": "side",
                            "forId": step_id,
                            "mime": "text/plain",
                            "url": f"/api/source/{thread_id}/{src_id}",
                        }
                        elements.append(elem)
                        step_elements.append(elem)

                steps.append({
                    "id": step_id,
                    "threadId": thread_id,
                    "name": step_name,
                    "type": "assistant_message" if role == "assistant" else "user_message",
                    "output": content,
                    "createdAt": step_created,
                    "start": step_created,
                    "end": step_created,
                    "elements": step_elements,
                    "metadata": {},
                    "streaming": False,
                    "isError": False
                })

        result = {
            "id": thread_id,
            "createdAt": created_at,
            "name": thread_info["name"],
            "userId": str(thread_info["userId"]),
            "userIdentifier": user_email,
            "tags": [],
            "metadata": {},
            "steps": steps,
            "elements": elements 
        }
        print(f"[DEBUG get_thread] Reconstructed thread {thread_id} with {len(steps)} steps and {len(elements)} elements")
        return result

    async def update_thread(self, thread_id: str, name: str = None, user_id: str = None, metadata: Dict = None, tags: List[str] = None):
        pass

    async def delete_thread(self, thread_id: str):
        pass
    
    async def create_step(self, step_dict):
        # Chainlit creates "steps" for messages. feedback.forId uses the STEP id,
        # not msg.id. We must copy feedback context from parent (msg.id) to step id.
        try:
            step_id = step_dict.get("id") if isinstance(step_dict, dict) else getattr(step_dict, "id", None)
            parent_id = step_dict.get("parentId") if isinstance(step_dict, dict) else getattr(step_dict, "parentId", None)
            print(f"[CREATE_STEP] step_id={step_id}, parentId={parent_id}", flush=True)

            if step_id and parent_id and parent_id in _feedback_context:
                _feedback_context[step_id] = _feedback_context[parent_id]
                print(f"[CREATE_STEP] Mapped step {step_id} -> parent {parent_id} context", flush=True)
                try:
                    ctx = _feedback_context[parent_id]
                    CHAT_DB.save_feedback_context(
                        cl_msg_id=step_id,
                        question=ctx.get("question"),
                        answer=ctx.get("answer"),
                        user_email=ctx.get("user_email"),
                        sources=ctx.get("sources"),
                    )
                except Exception:
                    pass
            elif step_id and not parent_id:
                output = step_dict.get("output") if isinstance(step_dict, dict) else getattr(step_dict, "output", None)
                if output and _feedback_context:
                    for mid, ctx in reversed(list(_feedback_context.items())):
                        if ctx.get("answer") and output[:100] in ctx["answer"][:200]:
                            _feedback_context[step_id] = ctx
                            print(f"[CREATE_STEP] Matched step {step_id} to msg {mid} via output", flush=True)
                            try:
                                CHAT_DB.save_feedback_context(
                                    cl_msg_id=step_id,
                                    question=ctx.get("question"),
                                    answer=ctx.get("answer"),
                                    user_email=ctx.get("user_email"),
                                    sources=ctx.get("sources"),
                                )
                            except Exception:
                                pass
                            break
        except Exception as e:
            print(f"[CREATE_STEP] Error: {e}", flush=True)

    async def update_step(self, step_dict):
        # Also try to map on update (Chainlit may update step with final output)
        try:
            step_id = step_dict.get("id") if isinstance(step_dict, dict) else getattr(step_dict, "id", None)
            parent_id = step_dict.get("parentId") if isinstance(step_dict, dict) else getattr(step_dict, "parentId", None)

            if step_id and step_id not in _feedback_context:
                if parent_id and parent_id in _feedback_context:
                    _feedback_context[step_id] = _feedback_context[parent_id]
                    print(f"[UPDATE_STEP] Mapped step {step_id} -> parent {parent_id}", flush=True)
                    try:
                        ctx = _feedback_context[parent_id]
                        CHAT_DB.save_feedback_context(
                            cl_msg_id=step_id,
                            question=ctx.get("question"),
                            answer=ctx.get("answer"),
                            user_email=ctx.get("user_email"),
                            sources=ctx.get("sources"),
                        )
                    except Exception:
                        pass
                else:
                    output = step_dict.get("output") if isinstance(step_dict, dict) else getattr(step_dict, "output", None)
                    if output and _feedback_context:
                        for mid, ctx in reversed(list(_feedback_context.items())):
                            if ctx.get("answer") and output[:100] in ctx["answer"][:200]:
                                _feedback_context[step_id] = ctx
                                print(f"[UPDATE_STEP] Matched step {step_id} to msg {mid} via output", flush=True)
                                try:
                                    CHAT_DB.save_feedback_context(
                                        cl_msg_id=step_id,
                                        question=ctx.get("question"),
                                        answer=ctx.get("answer"),
                                        user_email=ctx.get("user_email"),
                                        sources=ctx.get("sources"),
                                    )
                                except Exception:
                                    pass
                                break
        except Exception as e:
            print(f"[UPDATE_STEP] Error: {e}", flush=True)
        
    async def delete_step(self, step_id):
        pass
        
    async def create_element(self, element):
        pass

    async def get_element(self, thread_id, element_id):
        print(f"[DEBUG get_element] CALLED thread_id={thread_id}, element_id={element_id}")
        if not element_id.startswith("src_"):
            return None
        
        try:
            # element_id format: src_{ts}_{idx}
            parts = element_id.split("_")
            if len(parts) < 3:
                return None
            ts_str = parts[1]
            src_idx = int(parts[2])
            
            # Fetch thread info to get userId
            thread_info = CHAT_DB.get_session_by_id(thread_id)
            if not thread_info:
                # Try with thread_ prefix
                thread_info = CHAT_DB.get_session_by_id(f"thread_{thread_id}")
            
            if not thread_info:
                return None
            
            uid = int(thread_info["userId"])
            # Fetch messages for this thread
            actual_session_id = thread_id
            if not CHAT_DB.get_session_by_id(actual_session_id):
                 actual_session_id = f"thread_{thread_id}"

            msgs = CHAT_DB.get_recent(uid, actual_session_id, MAX_HISTORY_MSGS)
            
            # Find the message with matching timestamp that has sources
            target_msg = None
            for m in msgs:
                if not m.get("sources"):
                    continue
                m_ts = m.get("ts")
                if m_ts is not None:
                    if str(int(float(m_ts))) == ts_str:
                         target_msg = m
                         print(f"[DEBUG get_element] Found matching message with sources (ts match)")
                         break

            # Fallback for ts_str=="0" or no exact match: find the nth assistant message with sources
            if not target_msg:
                msgs_with_sources = [m for m in msgs if m.get("sources") and m.get("role") == "assistant"]
                if msgs_with_sources:
                    # Use the source index to guess which message it belongs to
                    # For now take the last one (most recent) as a reasonable default
                    target_msg = msgs_with_sources[-1]
                    print(f"[DEBUG get_element] Using fallback: last assistant message with sources")

            if not target_msg:
                return None
            
            msg_sources = target_msg.get("sources")
            if not msg_sources or src_idx >= len(msg_sources):
                return None
            
            src = msg_sources[src_idx]

            # Return an ElementDict (plain dict) with url pointing to content endpoint.
            # Chainlit's frontend fetches element content from 'url', not 'content'.
            return {
                "id": element_id,
                "threadId": thread_id,
                "type": "text",
                "name": f"Source {src.get('index', '?')}",
                "display": "side",
                "mime": "text/plain",
                "url": f"/api/source/{thread_id}/{element_id}",
            }
        except Exception as e:
            print(f"[DEBUG get_element] Error: {e}")
            return None

    async def delete_element(self, element_id, thread_id=None):
        pass
        
    async def upsert_feedback(self, feedback):
        import logging
        logger = logging.getLogger("chainlit")
        try:
            # Get the message ID this feedback is for
            for_id = None
            if hasattr(feedback, 'forId'):
                for_id = feedback.forId
            elif isinstance(feedback, dict):
                for_id = feedback.get("forId")

            question = None
            answer = None
            user_email = None

            print(f"[FEEDBACK_UPSERT] forId={for_id}, cache_keys={list(_feedback_context.keys())[:5]}, cache_size={len(_feedback_context)}", flush=True)

            # Strategy 1: In-memory cache (fast, works within same process)
            ctx = _feedback_context.get(for_id, {})
            if ctx:
                question = ctx.get("question")
                answer = ctx.get("answer")
                user_email = ctx.get("user_email")
                logger.info(f"[FEEDBACK] forId={for_id}, source=memory_cache, question={question[:50] if question else None}")

            # Strategy 2: Database lookup via feedback_context table (no FK constraints)
            if not question and for_id:
                try:
                    db_ctx = CHAT_DB.get_feedback_context(for_id)
                    if db_ctx:
                        question = question or db_ctx.get("question")
                        answer = answer or db_ctx.get("answer")
                        user_email = user_email or db_ctx.get("user_email")
                        logger.info(f"[FEEDBACK] forId={for_id}, source=db_lookup, question={question[:50] if question else None}")
                except Exception as db_err:
                    logger.warning(f"[FEEDBACK] DB lookup failed: {db_err}")

            # Strategy 3: Fallback — use most recent feedback context entry
            if not question and _feedback_context:
                last_key = list(_feedback_context.keys())[-1]
                ctx = _feedback_context[last_key]
                question = ctx.get("question")
                answer = ctx.get("answer")
                user_email = ctx.get("user_email")
                print(f"[FEEDBACK_UPSERT] Strategy 3: used most recent context (key={last_key})", flush=True)

            print(f"[FEEDBACK_UPSERT] FINAL: question={'YES' if question else 'NO'}, answer={'YES' if answer else 'NO'}, user={user_email}", flush=True)
            CHAT_DB.upsert_feedback(feedback, question=question, answer=answer, user_email=user_email)
            return True
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            import traceback; traceback.print_exc()
            return False
        
    async def delete_feedback(self, feedback_id):
        return False
        
    async def build_debug_url(self):
        return ""
        
    async def get_thread_author(self, thread_id: str) -> str:
        """Return the identifier (email) of the thread's author. 
        Chainlit's ACL compares this against the logged-in user's identifier."""
        print(f"[DEBUG get_thread_author] thread_id={thread_id}")
        thread_info = CHAT_DB.get_session_by_id(thread_id)
        print(f"[DEBUG get_thread_author] thread_info direct={thread_info}")
        if not thread_info:
            # Try with thread_ prefix for backward compat
            thread_info = CHAT_DB.get_session_by_id(f"thread_{thread_id}")
            print(f"[DEBUG get_thread_author] thread_info with prefix={thread_info}")
        
        if not thread_info:
            print(f"[DEBUG get_thread_author] NO thread_info found, returning empty")
            return ""
        
        # Look up user email from user ID
        try:
            user_id = int(thread_info["userId"])
            print(f"[DEBUG get_thread_author] looking up user_id={user_id}")
            user_dict = CHAT_DB.get_user_by_id(user_id)
            print(f"[DEBUG get_thread_author] user_dict={user_dict}")
            if user_dict:
                email = user_dict.get("email", "")
                print(f"[DEBUG get_thread_author] returning email={email}")
                return email
        except (ValueError, TypeError) as e:
            print(f"[DEBUG get_thread_author] exception: {e}")
            pass
        
        print(f"[DEBUG get_thread_author] fallthrough, returning empty")
        return ""
        
    async def get_favorite_steps(self, user_id):
        return []

    async def close(self):
        pass

@cl.data_layer
def get_data_layer():
    return WydotDataLayer()



# =========================================================
# CHAINLIT CALLBACKS
# =========================================================

@cl.password_auth_callback
def auth_callback(username, password):
    """Login: authenticate ONLY."""
    if not CHAT_DB:
        return None
    try:
        result = CHAT_DB.authenticate(username, password)
        uid, name = result
        if uid:
            verified = CHAT_DB.is_verified(uid)
            if not verified:
                return None

            is_guest = (username == "guest@app.local")
            user = cl.User(
                identifier=username,
                display_name=name or username,
                metadata={"db_id": uid, "is_guest": is_guest},
            )
            return user
    except Exception:
        pass
    return None

@cl.on_chat_resume
async def on_chat_resume(thread: Dict):
    """
    Called when user resumes a past conversation from the history panel.
    Messages are already rendered via get_thread() steps. Here we only
    restore session state so the user can continue chatting.
    """
    try:
        thread_id = thread.get("id")
        print(f"[RESUME] on_chat_resume called, thread_id={thread_id}")
        
        # Set session state
        cl.user_session.set("thread_id", thread_id)
        cl.user_session.set("session_id", thread_id)
        
        # Set up ChatSettings (same as on_chat_start)
        # Small delay to ensure frontend is ready during resume/refresh
        await cl.sleep(0.5) 
        settings = await cl.ChatSettings(
            [
                Select(
                    id="model",
                    label="Model",
                    values=["Mistral Large", "Gemini 2.5 Flash"] + list(OPENROUTER_MODELS.keys()),
                    initial_index=1,
                ),
                Select(
                    id="index",
                    label="Document Index",
                    values=["All Documents", "2021 Specs", "2010 Specs"],
                    initial_index=0,
                ),
                Switch(id="agentic_mode", label="Multi-Agent Mode (Domain Routing)", initial=False),
                Switch(id="thinking_mode", label="Thinking Mode (Slower, Detailed)", initial=False),
                Switch(id="multihop", label="Multi-hop Reasoning", initial=False),
                Switch(id="reranking", label="Reranking (FlashRank)", initial=False),
                Switch(id="hyde", label="HyDE (Query Expansion)", initial=False),
                Slider(id="fetch_k", label="Initial Candidates (FETCH_K)", min=10, max=100, step=5, initial=15),
            ]
        ).send()
        cl.user_session.set("settings", settings)
        cl.user_session.set("memory", [])

        # Handle backward-compat session ID for old threads saved with "thread_" prefix
        user = cl.user_session.get("user")
        if user and user.metadata.get("db_id"):
            uid = user.metadata["db_id"]
            # Check if messages exist with the thread_id directly
            history_msgs = CHAT_DB.get_recent(uid, thread_id, 1)
            if not history_msgs:
                # Try with thread_ prefix (old sessions)
                prefixed_id = f"thread_{thread_id}"
                if CHAT_DB.get_recent(uid, prefixed_id, 1):
                    cl.user_session.set("session_id", prefixed_id)
                    print(f"[RESUME] Using prefixed session_id: {prefixed_id}")
        
        print(f"[RESUME] Resume complete! session_id={cl.user_session.get('session_id')}")
    except Exception as e:
        print(f"[RESUME] ERROR: {e}")
        import traceback
        traceback.print_exc()
        await cl.Message(content="⚠️ Could not fully restore conversation. You can continue chatting normally.").send()

# =========================================================
# AUDIO CALLBACKS
# =========================================================

@cl.on_audio_start
async def on_audio_start():
    """Called when audio recording starts."""
    # Initialize audio buffer in session
    cl.user_session.set("audio_chunks", [])
    return True

@cl.on_audio_chunk  
async def on_audio_chunk(chunk):
    """Called for each audio chunk during recording."""
    audio_chunks = cl.user_session.get("audio_chunks", [])
    # chunk is an InputAudioChunk object with 'data' attribute
    audio_chunks.append(chunk.data if hasattr(chunk, 'data') else chunk)
    cl.user_session.set("audio_chunks", audio_chunks)

@cl.on_audio_end
async def on_audio_end():
    """Called when audio recording ends. Process the audio with Gemini."""
    audio_chunks = cl.user_session.get("audio_chunks", [])
    
    if not audio_chunks:
        await cl.Message(content="No audio recorded.").send()
        return
    
    # Combine audio chunks into a single buffer
    audio_data = b"".join([chunk if isinstance(chunk, bytes) else b"" for chunk in audio_chunks])
    
    if not audio_data:
        await cl.Message(content="No audio data captured.").send()
        return
    
    # Create proper WAV file with headers
    import io
    import wave
    import tempfile
    
    # Chainlit records at 44100Hz, 16-bit, mono by default
    sample_rate = 44100
    channels = 1
    sample_width = 2  # 16-bit = 2 bytes
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    
    # Get info about the audio chunks for debugging
    chunk_count = len(audio_chunks)
    total_bytes = len(audio_data)
    wav_bytes = wav_buffer.getvalue()
    
    # Check if the first chunk has mime_type info (Chainlit InputAudioChunk)
    # Chainlit typically sends audio as webm or opus format
    
    # Step 1: Transcribe audio with Gemini
    msg = cl.Message(content=f"🎤 Transcribing... ({chunk_count} chunks, {total_bytes} bytes)")
    await msg.send()
    
    try:
        import google.generativeai as genai
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Transcribe only - get the question text in English
        transcribe_response = model.generate_content([
            """You are transcribing audio from a user asking questions about WYDOT (Wyoming Department of Transportation).

IMPORTANT:
- The user is speaking ENGLISH
- They are asking about road specifications, construction, speed limits, or other transportation topics
- Transcribe EXACTLY what you hear in English
- Return ONLY the transcribed text, nothing else
- If the audio is unclear or silent, return exactly: UNCLEAR

Transcribe the following audio:""",
            {"mime_type": "audio/wav", "data": wav_bytes}
        ])
        
        transcribed_text = transcribe_response.text.strip()
        
        if not transcribed_text or transcribed_text == "UNCLEAR":
            msg.content = "⚠️ Could not understand the audio. Please try again."
            await msg.update()
            cl.user_session.set("audio_chunks", [])
            return
        
        # Show transcription
        msg.content = f"🎤 **Your question:** {transcribed_text}\n\n🔍 Searching documents..."
        await msg.update()
        
        # Step 2: Search knowledge graph (same as text query)
        settings = cl.user_session.get("settings", {})
        index_name = settings.get("index", "All Documents")
        model_type = "gemini" if settings.get("model", "Mistral Large") == "Gemini 2.5 Flash" else "mistral"
        
        context, sources = search_graph(transcribed_text, index_name)
        
        if not context:
            msg.content = f"🎤 **Your question:** {transcribed_text}\n\nNo relevant documents found."
            await msg.update()
            cl.user_session.set("audio_chunks", [])
            return
        
        # Step 3: Generate answer with context
        user = cl.user_session.get("user")
        history_msgs = []
        if user and user.metadata.get("db_id"):
            session_id = cl.user_session.get("session_id", "cl_session")
            history_msgs = conv_mem.get_recent(
                user.metadata["db_id"], session_id, limit=MAX_HISTORY_MSGS,
                fallback=CHAT_DB.get_recent,
            )
        
        if model_type == "gemini":
            answer = generate_answer_gemini(transcribed_text, context, history_msgs, enhanced_citations=True)
        else:
            answer = generate_answer_mistral(transcribed_text, context, history_msgs, enhanced_citations=True)
        
        # Enhance citations
        enhanced_answer = enhance_citations_in_response(answer, sources)
        
        # Create source elements for side panel
        clean_elements = []
        for src in sources:
            clean_elements.append(
                cl.Text(
                    name=f"Source {src['index']}", 
                    content=f"**Source {src['index']}: {src['title']}**\n\n**File:** {src['source']}\n**Section:** {src.get('section', 'N/A')}\n**Year:** {src.get('year', 'N/A')}\n\n**Preview:**\n{src['preview']}", 
                    display="side"
                )
            )
        
        msg.content = f"🎤 **Your question:** {transcribed_text}\n\n{enhanced_answer}"
        msg.elements = clean_elements
        await msg.update()
        
        # Persist: Redis (fast) + DB (durable) if NOT guest
        if user and user.metadata.get("db_id") and not user.metadata.get("is_guest"):
            session_id = cl.user_session.get("session_id", "cl_session")
            uid = user.metadata["db_id"]
            conv_mem.append(uid, session_id, "user", f"[Voice] {transcribed_text}")
            conv_mem.append(uid, session_id, "assistant", enhanced_answer)
            CHAT_DB.add_message(uid, session_id, "user", f"[Voice] {transcribed_text}")
            CHAT_DB.add_message(uid, session_id, "assistant", enhanced_answer)

    except Exception as e:
        msg.content = f"⚠️ Error processing audio: {str(e)}"
        await msg.update()
    
    cl.user_session.set("audio_chunks", [])

# =========================================================
# MODELS & RETRIEVAL (cached so we don't reload on every message)
# =========================================================

_EMBEDDINGS_CACHE = {}
_VECTOR_STORE_CACHE = {}
_DOC_TITLES_CACHE = {"titles": [], "last_updated": 0}

# --- Embedding Query LRU Cache ---
# Avoids re-calling Gemini API for repeated/identical queries
from collections import OrderedDict

_EMBEDDING_QUERY_CACHE = OrderedDict()
_EMBEDDING_CACHE_MAX = int(os.getenv("EMBEDDING_CACHE_MAX", "500"))

def cached_embed_query(query: str, embeddings_model=None) -> list:
    """Cache embedding vectors for query strings to avoid repeat API calls."""
    normalized = query.strip().lower()
    if normalized in _EMBEDDING_QUERY_CACHE:
        _EMBEDDING_QUERY_CACHE.move_to_end(normalized)
        return _EMBEDDING_QUERY_CACHE[normalized]
    if embeddings_model is None:
        embeddings_model = get_embeddings_model()
    vector = embeddings_model.embed_query(query)
    _EMBEDDING_QUERY_CACHE[normalized] = vector
    if len(_EMBEDDING_QUERY_CACHE) > _EMBEDDING_CACHE_MAX:
        _EMBEDDING_QUERY_CACHE.popitem(last=False)
    return vector

# --- Lazy Loaders for Models ---

def get_embeddings_model(use_gemini: bool = False):
    """Load embeddings once and reuse."""
    key = "gemini"
    if key in _EMBEDDINGS_CACHE:
        return _EMBEDDINGS_CACHE[key]

    print("🔄 Loading Gemini Embeddings model (models/gemini-embedding-001)...")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    _EMBEDDINGS_CACHE[key] = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY,
        task_type="retrieval_query"
    )
    print("✅ Embeddings model cached.")
    return _EMBEDDINGS_CACHE[key]

def get_retriever(index_name: str, use_gemini: bool = False):
    cache_key = (index_name, use_gemini)
    if cache_key in _VECTOR_STORE_CACHE:
        return _VECTOR_STORE_CACHE[cache_key]
    try:
        from langchain_neo4j import Neo4jVector
        embeds = get_embeddings_model(use_gemini)
        
        # Hybrid Search: Neo4j allows combining vector with keyword search
        # We can implement this by providing a search_type="hybrid" if the provider supports it,
        # or by customizing the retrieval query.
        
        retriever = Neo4jVector.from_existing_index(
            embeds,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            index_name=index_name,
            node_label="Chunk",
            text_node_property="text",
            embedding_node_property="embedding"
        ).as_retriever(search_kwargs={"k": FETCH_K})
        
        _VECTOR_STORE_CACHE[cache_key] = retriever
        return retriever
    except Exception as e:
        print(f"Retriever error: {e}. Falling back to standard vector search.")
        try:
            from langchain_neo4j import Neo4jVector
            embeds = get_embeddings_model(use_gemini)
            retriever = Neo4jVector.from_existing_index(
                embeds,
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                database=NEO4J_DATABASE,
                index_name=index_name,
                node_label="Chunk",
                text_node_property="text",
                embedding_node_property="embedding"
            ).as_retriever(search_kwargs={"k": FETCH_K})
            _VECTOR_STORE_CACHE[cache_key] = retriever
            return retriever
        except:
            return None

def get_batch_neighbors(chunk_ids: List[str], count: int = 2) -> Dict[str, str]:
    """Fetch adjacent chunks via NEXT_CHUNK graph edges for context expansion.
    
    Uses the 87K+ NEXT_CHUNK relationships in the graph for proper
    sequential context window expansion.
    """
    if not chunk_ids:
        return {}

    neighbor_map = {}
    try:
        from neo4j import GraphDatabase
        driver = get_neo4j_driver()
        with driver.session(database=NEO4J_DATABASE) as session:
            query = """
            UNWIND $cids AS cid
            MATCH (c:Chunk {id: cid})
            OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(next1:Chunk)
            OPTIONAL MATCH (next1)-[:NEXT_CHUNK]->(next2:Chunk)
            OPTIONAL MATCH (prev1:Chunk)-[:NEXT_CHUNK]->(c)
            WITH cid, 
                 prev1.text AS prev_text,
                 next1.text AS next1_text,
                 next2.text AS next2_text
            RETURN cid, prev_text, next1_text, next2_text
            """
            result = session.run(query, cids=chunk_ids)
            for record in result:
                cid = record["cid"]
                neighbor_texts = []
                if record["prev_text"]:
                    neighbor_texts.append(record["prev_text"][:500])
                if record["next1_text"]:
                    neighbor_texts.append(record["next1_text"][:500])
                if record["next2_text"] and count > 1:
                    neighbor_texts.append(record["next2_text"][:500])
                if neighbor_texts:
                    neighbor_map[cid] = "\n".join(neighbor_texts)
        pass  # Driver managed by connection pool singleton
    except Exception as e:
        print(f"Error fetching NEXT_CHUNK neighbors: {e}")
        
    return neighbor_map


async def generate_hyde_snippet(query: str) -> str:
    """Generate a hypothetical technical snippet for the query (HyDE)."""
    hyde_prompt = f"""
    You are a technical document writer. Given the user query, write a single paragraph (maximum 150 words) 
    that looks like it was extracted directly from a technical specification or engineering manual. 
    Use technical language, refer to sections or standards if appropriate.
    
    User Query: {query}
    
    Hypothetical Snippet:
    """
    try:
        model = get_gemini_llm()
        response = await model.generate_content_async(hyde_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"HyDE generation failed: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# HYBRID KEYWORD + VECTOR SEARCH (Graph-Enhanced)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_keywords(query: str) -> List[str]:
    """Extract likely proper nouns or quoted terms from a query for keyword search."""
    keywords = []
    # Quoted terms
    quoted = re.findall(r'"([^"]+)"', query)
    keywords.extend(quoted)
    # Proper noun detection: capitalized multi-word phrases (2+ words)
    proper_nouns = re.findall(r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', query)
    keywords.extend(proper_nouns)
    # If nothing found, try extracting keywords after common question words
    if not keywords:
        q_lower = query.lower().strip()
        for prefix in ("who is ", "what is ", "tell me about ", "find ", "search for "):
            if q_lower.startswith(prefix):
                remainder = query[len(prefix):].strip().rstrip("?. ")
                if remainder and len(remainder) > 2:
                    keywords.append(remainder)
                break
    return list(set(keywords))


def _keyword_search(keywords: List[str], limit: int = 8) -> List:
    """Search chunks by keyword via Cypher CONTAINS. Returns LangChain Document objects."""
    if not keywords:
        return []
    from neo4j import GraphDatabase
    from langchain_core.documents import Document
    driver = get_neo4j_driver()
    results = []
    seen_ids = set()
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            for kw in keywords:
                cypher = """
                MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
                WHERE toLower(c.text) CONTAINS toLower($keyword)
                RETURN c.id AS id, c.text AS text, c.source AS chunk_source,
                       d.source AS source, d.display_title AS title,
                       d.year AS year, s.name AS section, c.page AS page
                LIMIT $limit
                """
                for rec in session.run(cypher, keyword=kw, limit=limit):
                    cid = rec["id"]
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        doc = Document(
                            page_content=rec["text"],
                            metadata={
                                "id": cid,
                                "source": rec["source"] or rec["chunk_source"],
                                "title": rec["title"],
                                "year": rec["year"],
                                "section": rec["section"],
                                "page": rec["page"],
                            }
                        )
                        results.append(doc)
    except Exception as e:
        print(f"⚠️ Keyword search failed: {e}")
    finally:
        pass  # Driver managed by connection pool singleton
    print(f"🔑 Keyword search for {keywords}: found {len(results)} chunks")
    return results


def _fetch_doc_relationships(doc_sources: List[str]) -> Dict[str, List]:
    """Fetch Phase 4+5 document-level relationships for a set of source documents."""
    if not doc_sources:
        return {}
    from neo4j import GraphDatabase
    driver = get_neo4j_driver()
    rel_map = {}
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            cypher = """
            UNWIND $sources AS src
            MATCH (d:Document {source: src})
            OPTIONAL MATCH (d)-[r]->(related:Document)
            WHERE type(r) IN ['PART_OF', 'SUPPLEMENTS', 'INSTRUCTIONS_FOR',
                               'NEXT_IN_SERIES', 'COVERS_CORRIDOR',
                               'TEACHES', 'IMPLEMENTS', 'AMENDS',
                               'COMPANION_TO', 'GOVERNED_BY', 'SUPERSEDES']
            RETURN src, collect(DISTINCT {
                rel_type: type(r),
                target_title: related.display_title,
                target_source: related.source,
                target_year: related.year
            }) AS doc_relationships
            """
            for rec in session.run(cypher, sources=doc_sources):
                rels = [r for r in rec["doc_relationships"] if r.get("target_source")]
                if rels:
                    rel_map[rec["src"]] = rels
    except Exception as e:
        print(f"⚠️ Doc relationship fetch failed: {e}")
    finally:
        pass  # Driver managed by connection pool singleton
    return rel_map


def _llm_route_query(query: str) -> dict:
    """Use a fast LLM call to classify which document category a query targets.
    Returns dict with 'category' and optional 'year', 'series_filter'."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = f"""You are a query router for the Wyoming Department of Transportation (WYDOT) knowledge base.
Given a user query, classify it into ONE document category and extract any year mentioned.

CATEGORIES:
- STANDARD_SPECS: Construction specifications, material requirements, pay items, tolerances, testing thresholds, grinding, concrete, asphalt, bridge decks, reinforcing steel, guardrails, pavement smoothness, erosion control, traffic control devices, welding, pile driving, culverts, any "Section XXX" reference, warranties, penalties, liquidated damages, contractor requirements, insurance, change orders, disputes, certifications, environmental requirements, hazardous materials, temporary structures, work zone safety, surveying, progress measurement, paint, coatings — basically ANY question about how construction work should be done or what rules/specs apply.
- CONSTRUCTION_MANUAL: Field inspection procedures, project administration, how-to for engineers/inspectors, construction management processes.
- MATERIALS_TESTING: Lab test procedures, sampling methods, testing frequencies, material acceptance.
- DESIGN_MANUAL: Road/bridge design standards, geometric design, horizontal/vertical alignment.
- TRAFFIC_CRASHES: Crash statistics, crash data, fatalities by county/year, accident trends.
- STIP: Transportation improvement program, project funding, planned projects.
- ANNUAL_REPORT: WYDOT annual reports, department accomplishments, organizational info.
- BRIDGE_PROGRAM: Bridge plans, bridge design manual, bridge ratings, load postings.
- HIGHWAY_SAFETY: Safety plans, safety improvement programs, vulnerable road users.
- GENERAL: Doesn't fit above or spans multiple categories — use global search.

Respond in EXACTLY this format (nothing else):
CATEGORY: <category>
YEAR: <year or NONE>
SERIES: <specific document_series name if obvious, or NONE>

Query: {query}"""

        response = model.generate_content(prompt)
        text = response.text.strip()

        category = "GENERAL"
        year = None
        series = None
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('CATEGORY:'):
                category = line.split(':', 1)[1].strip()
            elif line.startswith('YEAR:'):
                y = line.split(':', 1)[1].strip()
                if y != 'NONE' and y.isdigit():
                    year = int(y)
            elif line.startswith('SERIES:'):
                s = line.split(':', 1)[1].strip()
                if s != 'NONE':
                    series = s

        print(f"🤖 LLM Router → {category}" + (f" (year={year})" if year else "") + (f" (series={series})" if series else ""))
        return {"category": category, "year": year, "series": series}
    except Exception as e:
        print(f"⚠️ LLM routing failed: {e}, falling back to regex")
        return {"category": "FALLBACK", "year": None, "series": None}


# Mapping from LLM category to Neo4j document_series filters
_CATEGORY_TO_SERIES_FILTERS = {
    "STANDARD_SPECS": [
        "d.document_series CONTAINS 'Standard Spec' OR d.display_title CONTAINS 'Standard Spec'",
    ],
    "CONSTRUCTION_MANUAL": [
        "d.document_series = 'Construction Manual' OR d.display_title CONTAINS 'Construction Manual'",
    ],
    "MATERIALS_TESTING": [
        "d.document_series = 'Materials Testing Manual' OR d.display_title CONTAINS 'Materials Testing Manual'",
    ],
    "DESIGN_MANUAL": [
        "d.document_series = 'Design Manual' OR d.document_series = 'Road Design Manual' OR d.display_title CONTAINS 'Design Manual'",
    ],
    "TRAFFIC_CRASHES": [
        "d.document_series = 'Report on Traffic Crashes' OR d.display_title CONTAINS 'Traffic Crash'",
    ],
    "STIP": [
        "d.document_series CONTAINS 'State Transportation Improvement' OR d.display_title CONTAINS 'STIP'",
    ],
    "ANNUAL_REPORT": [
        "d.document_series CONTAINS 'Annual Report' OR d.display_title CONTAINS 'Annual Report'",
    ],
    "BRIDGE_PROGRAM": [
        "d.document_series CONTAINS 'Bridge' OR d.display_title CONTAINS 'Bridge'",
    ],
    "HIGHWAY_SAFETY": [
        "d.document_series CONTAINS 'Highway Safety' OR d.document_series CONTAINS 'Strategic Highway Safety' OR d.display_title CONTAINS 'Safety Plan'",
    ],
}


def _detect_target_documents(query: str) -> List[str]:
    """Detect specific documents referenced in the query using LLM routing.
    Falls back to regex for explicit mentions. Returns list of Document.source values."""
    q_lower = query.lower()
    years = re.findall(r'\b(20\d{2})\b', query)
    target_sources = []

    try:
        from neo4j import GraphDatabase
        driver = get_neo4j_driver()
        with driver.session(database=NEO4J_DATABASE) as session:

            # ── Fast regex for explicit document mentions (no LLM needed) ──
            is_specs_query = any(x in q_lower for x in ['standard spec', 'standard specification', ' specs ', ' specs.', ' specs,', ' spec ', ' spec.', ' spec,'])
            if not is_specs_query and years:
                is_specs_query = bool(re.search(r'(?i)\b(?:specs?|specifications?)\b', query))

            section_refs = re.findall(r'(?i)(?:section|subsection)\s*(\d+(?:\.\d+)*)', query)

            if is_specs_query or section_refs:
                # Explicit spec/section reference — no need for LLM
                target_years = years if years else ['2021']
                for year in target_years:
                    r = session.run('''
                        MATCH (d:Document)
                        WHERE (d.document_series CONTAINS 'Standard Spec' OR d.display_title CONTAINS 'Standard Spec')
                              AND d.year = $year
                        RETURN d.source AS source
                    ''', year=int(year))
                    target_sources.extend([rec['source'] for rec in r])
                if not target_sources:
                    r = session.run('''
                        MATCH (d:Document)
                        WHERE d.document_series CONTAINS 'Standard Spec' OR d.display_title CONTAINS 'Standard Spec'
                        RETURN d.source AS source
                    ''')
                    target_sources.extend([rec['source'] for rec in r])

            elif 'construction manual' in q_lower:
                for year in (years or []):
                    r = session.run('''
                        MATCH (d:Document)
                        WHERE d.display_title CONTAINS 'Construction Manual' AND d.year = $year
                        RETURN d.source AS source
                    ''', year=int(year))
                    target_sources.extend([rec['source'] for rec in r])
                if not target_sources:
                    r = session.run('''
                        MATCH (d:Document)
                        WHERE d.display_title CONTAINS 'Construction Manual'
                        RETURN d.source AS source ORDER BY d.year DESC LIMIT 3
                    ''')
                    target_sources.extend([rec['source'] for rec in r])

            elif 'annual report' in q_lower:
                for year in (years or []):
                    r = session.run('''
                        MATCH (d:Document)
                        WHERE d.display_title CONTAINS 'Annual Report' AND d.year = $year
                        RETURN d.source AS source
                    ''', year=int(year))
                    target_sources.extend([rec['source'] for rec in r])

            else:
                # ── LLM-based routing for all other queries ──
                route = _llm_route_query(query)
                category = route["category"]
                llm_year = route.get("year")

                if category in _CATEGORY_TO_SERIES_FILTERS and category != "GENERAL":
                    filters = _CATEGORY_TO_SERIES_FILTERS[category]
                    where_clause = " OR ".join(f"({f})" for f in filters)

                    use_year = llm_year or (int(years[0]) if years else None)
                    if use_year:
                        cypher = f'''
                            MATCH (d:Document)
                            WHERE ({where_clause}) AND d.year = $year
                            RETURN d.source AS source
                        '''
                        r = session.run(cypher, year=use_year)
                        target_sources.extend([rec['source'] for rec in r])

                    if not target_sources:
                        cypher = f'''
                            MATCH (d:Document)
                            WHERE {where_clause}
                            RETURN d.source AS source
                        '''
                        r = session.run(cypher)
                        target_sources.extend([rec['source'] for rec in r])

                    # For STANDARD_SPECS, also include latest Construction Manual
                    if category == "STANDARD_SPECS":
                        r = session.run('''
                            MATCH (d:Document)
                            WHERE d.display_title CONTAINS 'Construction Manual'
                            RETURN d.source AS source
                            ORDER BY d.year DESC LIMIT 2
                        ''')
                        target_sources.extend([rec['source'] for rec in r])

                elif category == "FALLBACK":
                    if years:
                        for year in years:
                            terms = [t for t in re.sub(r'[^\w\s]', ' ', query).split()
                                     if len(t) > 3 and t.lower() not in {'what', 'show', 'tell', 'compare', 'between', 'difference', 'from', 'with', 'that', 'this', 'have', 'does', 'standard', 'spec', 'specifications'}]
                            for term in terms[:3]:
                                r = session.run('''
                                    MATCH (d:Document)
                                    WHERE d.year = $year AND (d.display_title CONTAINS $term OR d.source CONTAINS $term)
                                    RETURN d.source AS source LIMIT 3
                                ''', year=int(year), term=term)
                                target_sources.extend([rec['source'] for rec in r])

        pass  # Driver managed by connection pool singleton
    except Exception as e:
        print(f"⚠️ Document detection failed: {e}")

    result = list(set(target_sources))
    if result:
        print(f"🎯 Detected target documents ({len(result)}): {[s[:40] for s in result[:5]]}")
    return result


def _document_scoped_search(query: str, doc_sources: List[str], limit: int = 20) -> List[Document]:
    """Search only within specific documents using vector similarity + fulltext."""
    if not doc_sources:
        return []
    results = []
    seen_ids = set()

    try:
        from neo4j import GraphDatabase
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GEMINI_API_KEY,
        )
        query_embedding = embeddings.embed_query(query)

        driver = get_neo4j_driver()
        with driver.session(database=NEO4J_DATABASE) as session:
            # 1. Vector search scoped to specific documents
            k_candidates = min(limit * 5, 250)
            cypher_vec = """
            CALL db.index.vector.queryNodes('wydot_gemini_index', $k, $emb)
            YIELD node, score
            MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(node)
            WHERE d.source IN $sources
            RETURN node.id AS id, node.text AS text, d.source AS source,
                   d.display_title AS title, d.year AS year,
                   s.name AS section, node.page AS page, score
            ORDER BY score DESC LIMIT $limit
            """
            r = session.run(cypher_vec, k=k_candidates, emb=query_embedding, sources=doc_sources, limit=limit)
            for rec in r:
                cid = rec["id"]
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    results.append(Document(
                        page_content=rec["text"],
                        metadata={
                            "id": cid, "source": rec["source"], "title": rec["title"],
                            "year": rec["year"], "section": rec["section"], "page": rec["page"],
                            "search_type": "scoped_vector", "score": rec["score"]
                        }
                    ))

            # 2. Also fulltext search within same documents
            keywords = _extract_keywords(query)
            if keywords:
                ft_query = " ".join(keywords)
                cypher_ft = """
                CALL db.index.fulltext.queryNodes('chunk_fulltext', $ftq)
                YIELD node, score
                MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(node)
                WHERE d.source IN $sources
                RETURN node.id AS id, node.text AS text, d.source AS source,
                       d.display_title AS title, d.year AS year,
                       s.name AS section, node.page AS page, score
                ORDER BY score DESC LIMIT $limit
                """
                r = session.run(cypher_ft, ftq=ft_query, sources=doc_sources, limit=limit)
                for rec in r:
                    cid = rec["id"]
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        results.append(Document(
                            page_content=rec["text"],
                            metadata={
                                "id": cid, "source": rec["source"], "title": rec["title"],
                                "year": rec["year"], "section": rec["section"], "page": rec["page"],
                                "search_type": "scoped_fulltext"
                            }
                        ))

        pass  # Driver managed by connection pool singleton
    except Exception as e:
        print(f"⚠️ Document-scoped search failed: {e}")

    print(f"🎯 Scoped search found {len(results)} chunks from {len(doc_sources)} documents")
    return results


def _merge_keyword_and_vector_docs(keyword_docs: List, vector_docs: List, limit: int) -> List:
    """Merge keyword docs (first) with vector docs, deduplicating by chunk ID."""
    seen_ids = set()
    merged = []
    for d in keyword_docs:
        cid = d.metadata.get("id")
        if cid not in seen_ids:
            seen_ids.add(cid)
            merged.append(d)
    for d in vector_docs:
        cid = d.metadata.get("id")
        if cid not in seen_ids:
            seen_ids.add(cid)
            merged.append(d)
    return merged[:limit]


def _inject_doc_relationships(doc, doc_rel_map: Dict):
    """Inject Phase 4+5 document relationships into a doc's page_content."""
    source = doc.metadata.get("source", "")
    rels = doc_rel_map.get(source, [])
    if rels and "--- RELATED DOCUMENTS ---" not in doc.page_content:
        rels_str = "\n".join(
            f"* {r['rel_type']}: {r.get('target_title', r['target_source'])} ({r.get('target_year', '')})"
            for r in rels[:8]
        )
        doc.page_content += f"\n\n--- RELATED DOCUMENTS ---\n{rels_str}\n"


async def search_graph_async(query: str, index_name: str, use_gemini: bool = False) -> Tuple[str, List[Dict]]:
    ret = get_retriever(index_name, use_gemini)
    if not ret: return "", []
    
    # Use session setting for FETCH_K if available
    settings = cl.user_session.get("settings") or {}
    k_val = int(settings.get("fetch_k", FETCH_K))
    
    try:
        # HyDE expansion if enabled
        hyde_enabled = settings.get("hyde", False)
        search_query = query
        if hyde_enabled:
            print("🧪 HyDE enabled: generating hypothetical snippet...")
            hyde_snippet = await generate_hyde_snippet(query)
            if hyde_snippet:
                search_query = f"{query} {hyde_snippet}"
                print(f"   HyDE expanded query length: {len(search_query)} chars")

        # Use ainvoke if available, otherwise run in thread
        if hasattr(ret, "ainvoke"):
            ret.search_kwargs["k"] = k_val
            vector_docs = await ret.ainvoke(search_query)
        else:
            import asyncio
            ret.search_kwargs["k"] = k_val
            vector_docs = await asyncio.to_thread(ret.invoke, search_query)

        # ── Document-Scoped Search (LLM routing for large graphs) ──
        target_doc_sources = _detect_target_documents(query)
        scoped_docs = _document_scoped_search(query, target_doc_sources, limit=20) if target_doc_sources else []

        # ── Hybrid: merge keyword hits with vector results ──
        keywords = _extract_keywords(query)
        keyword_docs = _keyword_search(keywords, limit=12) if keywords else []

        # Priority: scoped docs first, then keyword hits, then vector results
        if scoped_docs:
            docs = _merge_keyword_and_vector_docs(scoped_docs, keyword_docs + vector_docs, k_val + 10)
        else:
            docs = _merge_keyword_and_vector_docs(keyword_docs, vector_docs, k_val)

        # --- Metadata Enrichment ---
        # Since our index might not have source/title on the Chunk node directly,
        # we try to fetch them from the linked Document nodes if missing.
        chunks, sources = [], []

        from neo4j import GraphDatabase
        driver = get_neo4j_driver()

        # 1. Collect all chunk IDs to batch request metadata
        chunk_ids = [doc.metadata.get("id") for doc in docs if doc.metadata.get("id")]
        
        # 2. Run a single batched query to get metadata and Graph facts
        enrichment_map = {}
        if chunk_ids:
            try:
                with driver.session(database=NEO4J_DATABASE) as session:
                    cypher = """
                    MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
                    WHERE c.id IN $cids
                    OPTIONAL MATCH (newer_d:Document)-[:SUPERSEDES]->(d)
                    OPTIONAL MATCH (newer_d)-[:HAS_SECTION]->(newer_s:Section {name: s.name})-[:HAS_CHUNK]->(newer_c:Chunk)
                    OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)-[r]-(neighbor:Entity)
                    RETURN c.id as cid, d.source as source, d.display_title as title, d.year as year, 
                           d.document_series as series, s.name as section,
                           newer_d.year as newer_year, newer_c.text as newer_text, newer_d.source as newer_source,
                           collect(DISTINCT e.name + " " + type(r) + " " + neighbor.name) as graph_facts
                    """
                    result = session.run(cypher, cids=chunk_ids)
                    for record in result:
                        enrichment_map[record["cid"]] = record
            except Exception as e:
                print(f"⚠️ Batch Metadata enrichment failed: {e}")
        
        # Phase 4+5: Document-level relationships
        doc_sources_list = list({enrichment_map[cid].get("source") or "" for cid in enrichment_map} - {""})
        doc_rel_map = _fetch_doc_relationships(doc_sources_list)
        
        for i, doc in enumerate(docs):
            meta = doc.metadata
            chunk_id = meta.get("id")
            
            source_file = meta.get("source")
            title = meta.get("title")
            section_name = meta.get("section", "")
            
            if chunk_id in enrichment_map:
                link_res = enrichment_map[chunk_id]
                source_file = link_res["source"] or source_file
                title = link_res["title"] or title
                meta["source"] = source_file
                meta["title"] = title
                meta["year"] = link_res["year"]
                meta["series"] = link_res["series"]
                if link_res["section"]:
                    section_name = link_res["section"]
                    meta["section"] = section_name
                    
                # INJECT SUPERSEDED CONTENT
                if link_res["newer_text"] and link_res["newer_year"]:
                    append_str = (
                        f"\n\n[WARNING: This section from {meta['year']} is SUPERSEDED by {link_res['newer_year']} "
                        f"document: {link_res['newer_source']}]\n"
                        f"--- NEWER VERSION CONTENT ---\n{link_res['newer_text']}\n"
                    )
                    # Only append if not already appended (in case of chunk overlap)
                    if "[WARNING: This section" not in doc.page_content:
                        doc.page_content += append_str
                        
                # INJECT GRAPH FACTS
                facts = link_res.get("graph_facts", [])
                valid_facts = [f for f in facts if f and not f.isspace() and "None" not in f]
                if valid_facts:
                    facts_str = "\n".join([f"* {f}" for f in valid_facts])
                    if "--- KNOWLEDGE GRAPH FACTS ---" not in doc.page_content:
                        doc.page_content += f"\n\n--- KNOWLEDGE GRAPH FACTS ---\n{facts_str}\n"

            else:
                # Fallback: Find matching chunk by text snippet if ID wasn't in Graph
                try:
                    with driver.session(database=NEO4J_DATABASE) as session:
                        cypher = """
                        MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
                        WHERE c.text CONTAINS $text_snippet
                        OPTIONAL MATCH (newer_d:Document)-[:SUPERSEDES]->(d)
                        OPTIONAL MATCH (newer_d)-[:HAS_SECTION]->(newer_s:Section {name: s.name})-[:HAS_CHUNK]->(newer_c:Chunk)
                        RETURN d.source as source, d.display_title as title, d.year as year,
                               d.document_series as series, s.name as section,
                               newer_d.year as newer_year, newer_c.text as newer_text, newer_d.source as newer_source
                        LIMIT 1
                        """
                        link_res = session.run(cypher, text_snippet=doc.page_content[:100]).single()
                        if link_res:
                            source_file = link_res["source"] or source_file
                            title = link_res["title"] or title
                            meta["source"] = source_file
                            meta["title"] = title
                            meta["year"] = link_res["year"]
                            meta["series"] = link_res["series"]
                            if link_res["section"]:
                                section_name = link_res["section"]
                                meta["section"] = section_name
                                
                            if link_res["newer_text"] and link_res["newer_year"]:
                                append_str = (
                                    f"\n\n[WARNING: This section from {meta['year']} is SUPERSEDED by {link_res['newer_year']} "
                                    f"document: {link_res['newer_source']}]\n"
                                    f"--- NEWER VERSION CONTENT ---\n{link_res['newer_text']}\n"
                                )
                                if "[WARNING: This section" not in doc.page_content:
                                    doc.page_content += append_str
                except Exception as e:
                    pass

            # Inject Phase 4+5 document relationships into context
            _inject_doc_relationships(doc, doc_rel_map)

            if not title or title in ["Unknown", "Untitled", "None", ""]:
                title = source_file or "Untitled"

            # Extract just the filename for display (strip path if present)
            display_source = source_file or "File"
            if "/" in display_source:
                display_source = display_source.split("/")[-1]

            chunks.append(f"[SOURCE_{i+1}]\n{doc.page_content}")
            sources.append({
                "id": f"source_{i+1}",
                "index": i + 1,
                "title": title,
                "source": display_source,
                "year": meta.get("year", ""),
                "section": section_name or meta.get("section", ""),
                "page": "" if meta.get("page", 0) == 0 else meta.get("page", ""),
                "preview": doc.page_content[:300]
            })
            
            # --- URL Generation ---
            gcs_path = meta.get("gcs_path", "")
            if not gcs_path:
                project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
                if project_id:
                    bucket_name = f"wydot-documents-{project_id}"
                    if source_file:
                        gcs_path = f"gs://{bucket_name}/wydot documents/{source_file}"
            public_url = generate_public_url(gcs_path)
            page = meta.get("page", "")
            if public_url and page: public_url += f"#page={page}"
            sources[-1]["url"] = public_url

        pass  # Driver managed by connection pool singleton

        # --- FlashRank Reranking (only if enabled in settings) ---
        reranking_enabled = settings.get("reranking", False)
        if chunks and reranking_enabled:
            ranker = get_reranker()
            if ranker:
                try:
                    # Prepare format for FlashRank
                    passages = [
                        {"id": i, "text": doc.page_content, "meta": doc.metadata} 
                        for i, doc in enumerate(docs)
                    ]
                    
                    rerank_request = RerankRequest(query=query, passages=passages)
                    results = ranker.rerank(rerank_request)
                    
                    # Select Top K
                    top_results = results[:RETRIEVAL_K]
                    
                    # Reconstruct chunks and sources from top results
                    reranked_chunks = []
                    reranked_sources = []
                    
                    # 1. Collect chunk IDs for NEXT_CHUNK neighbor fetch
                    cids_to_fetch = []
                    for res in top_results:
                        idx = res['id']
                        cid = passages[idx]['meta'].get('id', '')
                        if cid:
                            cids_to_fetch.append(cid)
                    
                    # 2. Batch fetch neighbors via NEXT_CHUNK
                    neighbor_map = get_batch_neighbors(list(set(cids_to_fetch)), count=2)
                    
                    for i, res in enumerate(top_results):
                        idx = res['id']
                        s_info = sources[idx]
                        s_info.update({"index": i + 1, "id": f"source_{i+1}", "score": f"{res['score']:.2f}"})
                        reranked_sources.append(s_info)
                        
                        # Re-read page_content from 'passages' since we have ID
                        content = str(passages[idx]['text'])
                        
                        # Enhancement: Graph-aware context expansion via NEXT_CHUNK
                        cid = passages[idx]['meta'].get('id', '')
                        neighbors = neighbor_map.get(cid, "")
                        
                        if neighbors and neighbors.strip() not in content:
                             content += "\n--- Adjacent Chunk Context ---\n" + neighbors
                        
                        reranked_chunks.append(f"[SOURCE_{i+1}]\n{content}")
                    
                    return "\n\n".join(reranked_chunks), reranked_sources
                except Exception as e:
                    print(f"Reranking failed, falling back to vector results: {e}")

        return "\n\n".join(chunks), sources
    except Exception as e:
        print(f"Search error: {e}")
        return "", []

def search_graph(query: str, index_name: str, use_gemini: bool = False) -> Tuple[str, List[Dict]]:
    # Sync version
    ret = get_retriever(index_name, use_gemini)
    if not ret: return "", []
    try:
        ret.search_kwargs["k"] = FETCH_K
        vector_docs = ret.invoke(query)

        # ── Document-Scoped Search (LLM routing for large graphs) ──
        target_doc_sources = _detect_target_documents(query)
        scoped_docs = _document_scoped_search(query, target_doc_sources, limit=20) if target_doc_sources else []

        # ── Hybrid: merge keyword hits with vector results ──
        keywords = _extract_keywords(query)
        keyword_docs = _keyword_search(keywords, limit=12) if keywords else []

        # Priority: scoped docs first, then keyword hits, then vector results
        if scoped_docs:
            docs = _merge_keyword_and_vector_docs(scoped_docs, keyword_docs + vector_docs, FETCH_K + 10)
        else:
            docs = _merge_keyword_and_vector_docs(keyword_docs, vector_docs, FETCH_K)

        chunks, sources = [], []

        from neo4j import GraphDatabase
        driver = get_neo4j_driver()

        # 1. Collect all chunk IDs to batch request metadata
        chunk_ids = [doc.metadata.get("id") for doc in docs if doc.metadata.get("id")]
        
        # 2. Run a single batched query to get metadata and Graph facts
        enrichment_map = {}
        if chunk_ids:
            try:
                with driver.session(database=NEO4J_DATABASE) as session:
                    cypher = """
                    MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
                    WHERE c.id IN $cids
                    OPTIONAL MATCH (newer_d:Document)-[:SUPERSEDES]->(d)
                    OPTIONAL MATCH (newer_d)-[:HAS_SECTION]->(newer_s:Section {name: s.name})-[:HAS_CHUNK]->(newer_c:Chunk)
                    OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)-[r]-(neighbor:Entity)
                    RETURN c.id as cid, d.source as source, d.display_title as title, d.year as year,
                           d.document_series as series, s.name as section,
                           newer_d.year as newer_year, newer_c.text as newer_text, newer_d.source as newer_source,
                           collect(DISTINCT e.name + " " + type(r) + " " + neighbor.name) as graph_facts
                    """
                    result = session.run(cypher, cids=chunk_ids)
                    for record in result:
                        enrichment_map[record["cid"]] = record
            except Exception as e:
                print(f"⚠️ Batch Metadata enrichment failed: {e}")
        
        # Phase 4+5: Document-level relationships
        doc_sources_list = list({enrichment_map[cid].get("source") or "" for cid in enrichment_map} - {""})
        doc_rel_map = _fetch_doc_relationships(doc_sources_list)
        
        for i, doc in enumerate(docs):
            meta = doc.metadata
            chunk_id = meta.get("id")
            source_file = meta.get("source")
            title = meta.get("title")
            section_name = meta.get("section", "")
            
            if chunk_id in enrichment_map:
                link_res = enrichment_map[chunk_id]
                source_file = link_res["source"] or source_file
                title = link_res["title"] or title
                meta["source"] = source_file
                meta["title"] = title
                meta["year"] = link_res["year"]
                meta["series"] = link_res["series"]
                if link_res["section"]:
                    section_name = link_res["section"]
                    meta["section"] = section_name
                    
                # INJECT SUPERSEDED CONTENT
                if link_res["newer_text"] and link_res["newer_year"]:
                    append_str = (
                        f"\n\n[WARNING: This section from {meta['year']} is SUPERSEDED by {link_res['newer_year']} "
                        f"document: {link_res['newer_source']}]\n"
                        f"--- NEWER VERSION CONTENT ---\n{link_res['newer_text']}\n"
                    )
                    # Only append if not already appended
                    if "[WARNING: This section" not in doc.page_content:
                        doc.page_content += append_str
                        
                # INJECT GRAPH FACTS
                facts = link_res.get("graph_facts", [])
                valid_facts = [f for f in facts if f and not f.isspace() and "None" not in f]
                if valid_facts:
                    facts_str = "\n".join([f"* {f}" for f in valid_facts])
                    if "--- KNOWLEDGE GRAPH FACTS ---" not in doc.page_content:
                        doc.page_content += f"\n\n--- KNOWLEDGE GRAPH FACTS ---\n{facts_str}\n"

            else:
                # Fallback: Find matching chunk by text snippet if ID wasn't in Graph
                try:
                    with driver.session(database=NEO4J_DATABASE) as session:
                        cypher = """
                        MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
                        WHERE c.text CONTAINS $text_snippet
                        OPTIONAL MATCH (newer_d:Document)-[:SUPERSEDES]->(d)
                        OPTIONAL MATCH (newer_d)-[:HAS_SECTION]->(newer_s:Section {name: s.name})-[:HAS_CHUNK]->(newer_c:Chunk)
                        RETURN d.source as source, d.display_title as title, d.year as year,
                               d.document_series as series, s.name as section,
                               newer_d.year as newer_year, newer_c.text as newer_text, newer_d.source as newer_source
                        LIMIT 1
                        """
                        # Get a small snippet that Neo4j can QUICKLY find via CONTAINS
                        snippet = doc.page_content[:100]
                        link_res = session.run(cypher, text_snippet=snippet).single()
                        if link_res:
                            source_file = link_res["source"] or source_file
                            title = link_res["title"] or title
                            meta["source"] = source_file
                            meta["title"] = title
                            meta["year"] = link_res["year"]
                            meta["series"] = link_res["series"]
                            if link_res["section"]:
                                section_name = link_res["section"]
                                meta["section"] = section_name
                            
                            if link_res["newer_text"] and link_res["newer_year"]:
                                append_str = (
                                    f"\n\n[WARNING: This section from {meta['year']} is SUPERSEDED by {link_res['newer_year']} "
                                    f"document: {link_res['newer_source']}]\n"
                                    f"--- NEWER VERSION CONTENT ---\n{link_res['newer_text']}\n"
                                )
                                if "[WARNING: This section" not in doc.page_content:
                                    doc.page_content += append_str
                except Exception as e:
                    pass

            # Inject Phase 4+5 document relationships into context
            _inject_doc_relationships(doc, doc_rel_map)

            if title in ["Unknown", "Untitled", "None", "", None]:
                title = source_file or "Untitled"
            
            display_source = source_file or "File"
            if "/" in display_source:
                display_source = display_source.split("/")[-1]
            
            chunks.append(f"[SOURCE_{i+1}]\n{doc.page_content}")
            sources.append({
                "id": f"source_{i+1}",
                "index": i + 1,
                "title": title,
                "source": display_source,
                "year": meta.get("year", ""),
                "section": section_name or meta.get("section", ""),
                "page": "" if meta.get("page", 0) == 0 else meta.get("page", ""),
                "preview": doc.page_content[:300]
            })
            
            # URL Generation
            gcs_path = meta.get("gcs_path", "")
            if not gcs_path:
                project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
                if project_id:
                    bucket_name = f"wydot-documents-{project_id}"
                    if source_file:
                        gcs_path = f"gs://{bucket_name}/wydot documents/{source_file}"
            public_url = generate_public_url(gcs_path)
            page = meta.get("page", "")
            if public_url and page: public_url += f"#page={page}"
            sources[-1]["url"] = public_url
        
        pass  # Driver managed by connection pool singleton

        # --- FlashRank Reranking (only if enabled in settings) ---
        reranking_enabled = settings.get("reranking", False) if 'settings' in dir() else False
        if chunks and reranking_enabled:
            ranker = get_reranker()
            if ranker:
                try:
                    passages = [{"id": i, "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(docs)]
                    rerank_request = RerankRequest(query=query, passages=passages)
                    results = ranker.rerank(rerank_request)
                    top_results = results[:RETRIEVAL_K]
                    
                    reranked_chunks, reranked_sources = [], []
                    
                    # 1. Collect chunk IDs for NEXT_CHUNK neighbor fetch
                    cids_to_fetch = []
                    for res in top_results:
                        idx = res['id']
                        cid = passages[idx]['meta'].get('id', '')
                        if cid:
                            cids_to_fetch.append(cid)
                    
                    # 2. Batch fetch neighbors via NEXT_CHUNK
                    neighbor_map = get_batch_neighbors(list(set(cids_to_fetch)), count=2)
                    
                    for i, res in enumerate(top_results):
                        idx = res['id']
                        s_info = sources[idx]
                        s_info.update({"index": i + 1, "id": f"source_{i+1}", "score": f"{res['score']:.2f}"})
                        reranked_sources.append(s_info)
                        
                        content = str(passages[idx]['text'])
                        
                        # Enhancement: Graph-aware context expansion via NEXT_CHUNK
                        cid = passages[idx]['meta'].get('id', '')
                        neighbors = neighbor_map.get(cid, "")
                        
                        if neighbors and neighbors.strip() not in content:
                             content += "\n--- Adjacent Chunk Context ---\n" + neighbors
                             
                        reranked_chunks.append(f"[SOURCE_{i+1}]\n{content}")
                    return "\n\n".join(reranked_chunks), reranked_sources
                except Exception as e:
                    print(f"Sync Reranking failed: {e}")

        return "\n\n".join(chunks), sources
    except Exception as e:
        print(f"Search error: {e}")
        return "", []

def get_all_document_titles() -> List[str]:
    """Fetch unique document sources from Neo4j for query enhancement."""
    global _DOC_TITLES_CACHE
    now = time.time()
    # Cache for 1 hour
    if _DOC_TITLES_CACHE["titles"] and (now - _DOC_TITLES_CACHE["last_updated"] < 3600):
        return _DOC_TITLES_CACHE["titles"]
        
    try:
        from neo4j import GraphDatabase
        driver = get_neo4j_driver()
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run("MATCH (c:Chunk) RETURN DISTINCT c.source as source")
            titles = [record["source"] for record in result if record["source"]]
            _DOC_TITLES_CACHE["titles"] = sorted(titles)
            _DOC_TITLES_CACHE["last_updated"] = now
            return _DOC_TITLES_CACHE["titles"]
    except Exception as e:
        print(f"Error fetching document titles: {e}")
        return []

async def route_query_intent(query: str, use_gemini: bool = False) -> str:
    """Classifies the user query to determine if it needs Graph Analytics or Standard Search."""
    prompt = f"""You are a query intent classifier for a civil engineering Knowledge Graph.
Classify the following user query into exactly one of these categories:
1. STANDARD_SEARCH: General questions about specifications, requirements, facts, or instructions.
2. GLOBAL_ANALYTICS: Questions asking for counts, top N lists, or aggregations (e.g. "most cited testing standards", "how many materials").
3. IMPACT_ANALYSIS: Questions asking about dependencies or what is affected by a change (e.g. "what relies on Portland cement").
4. GRAPH_TRAVERSAL: Questions requiring multi-hop reasoning across explicitly different entities.

Query: {query}

Respond ONLY with the category name (e.g. STANDARD_SEARCH). No other text.
"""
    try:
        if use_gemini:
            model = get_gemini_llm()
            res = await model.generate_content_async(prompt)
            intent = res.text.strip()
        else:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            llm = get_mistral_llm()
            template = ChatPromptTemplate.from_template("{input_prompt}")
            chain = template | llm | StrOutputParser()
            intent = await chain.ainvoke({"input_prompt": prompt})
            intent = intent.strip()
            
        for valid in ["STANDARD_SEARCH", "GLOBAL_ANALYTICS", "IMPACT_ANALYSIS", "GRAPH_TRAVERSAL"]:
            if valid in intent:
                return valid
    except Exception as e:
        print(f"Intent routing failed: {e}")
        
    return "STANDARD_SEARCH"

async def search_graph_cypher_async(query: str, intent: str, use_gemini: bool = False) -> Tuple[str, List[Dict]]:
    """Generates and executes a Cypher query for analytical/traversal intents. Falls back to standard search."""
    schema_info = """
    Node Labels:
    - Document (source, display_title, year, document_series)
    - Section (name)
    - Chunk (id, text, chunk_seq, page)
    - Entity (name) - Canonical engineering entities (materials, standards, concepts).
    
    Relationships:
    - (Document)-[:HAS_SECTION]->(Section)
    - (Section)-[:HAS_CHUNK]->(Chunk)
    - (Document)-[:SUPERSEDES]->(Document)
    - (Chunk)-[:MENTIONS]->(Entity)
    - (Entity)-[r]->(Entity) - Various extracted relationships (e.g., REQUIRES, USES_MATERIAL, TESTED_BY).
    """
    
    prompt = f"""You are a Neo4j Cypher expert.
Given the schema and the user query (Intent: {intent}), generate a read-only Cypher query to answer the question.
SCHEMA:
{schema_info}

RULES:
1. Never use write clauses (CREATE, MERGE, SET, DELETE, REMOVE).
2. Limit the results to a reasonable number (LIMIT 25).
3. Return elements that are descriptive (e.g., node properties rather than the node objects themselves).
4. Do not output markdown code blocks. Output ONLY the raw Cypher string.
5. Entity names should be matched case text CONTAINS or similar if exact name not known.

Query: {query}
"""
    try:
        if use_gemini:
            model = get_gemini_llm()
            res = await model.generate_content_async(prompt)
            cypher = res.text.strip()
        else:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            llm = get_mistral_llm()
            template = ChatPromptTemplate.from_template("{input_prompt}")
            chain = template | llm | StrOutputParser()
            cypher = await chain.ainvoke({"input_prompt": prompt})
            cypher = cypher.strip()
            
        cypher = cypher.replace("```cypher", "").replace("```", "").strip()
        print(f"🧠 Generated Cypher:\n{cypher}")
        
        disallowed = ["CREATE", "MERGE", "SET", "DELETE", "REMOVE", "DROP"]
        if any(bad in cypher.upper() for bad in disallowed):
            print("⚠️ Cypher generation rejected for safety.")
            return "", []

        from neo4j import GraphDatabase
        driver = get_neo4j_driver()
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher)
            records = [record.data() for record in result]
            
        if not records:
            print("🧠 Cypher returned no records. Falling back to structured vector search.")
            return "", []
            
        import json
        context = "### Neo4j Graph Analytics Results\n"
        for i, rec in enumerate(records):
            context += f"{i+1}. {json.dumps(rec)}\n"
            
        dummy_source = [{
            "id": "source_1",
            "index": 1,
            "title": "Knowledge Graph Analytics",
            "source": "Graph Database Query",
            "year": "N/A",
            "section": "Analytics",
            "page": "",
            "preview": "Results generated directly via Cypher Traversal.",
            "url": ""
        }]
        return context, dummy_source
        
    except Exception as e:
        print(f"Cypher pipeline failed: {e}. Falling back...")
        return "", []


async def decompose_query(query: str, model_type: str = "mistral") -> List[str]:
    titles = get_all_document_titles()
    titles_context = "\n".join([f"- {t}" for t in titles]) if titles else "No specific documents listed."
    
    decomposition_prompt = f"""You are an agentic query analyzer for the WYDOT Knowledge Graph.
Your goal is to break down the user's question into 2-3 specific sub-queries.

KNOWLEDGE GRAPH DOCUMENTS:
{titles_context}

INSTRUCTIONS:
1. Identify if the user is referring to a specific document (e.g., "2021 specs", "construction manual").
2. Map their short-hand to the FULL document title from the list above.
3. If multiple years are mentioned, create separate sub-queries for each year and its specific document.
4. If no specific document is mentioned, keep the query general.

Main Question: {query}

Provide the sub-queries as a numbered list, one per line. Be concise. Do NOT add any preamble.
"""
    try:
        if model_type == "gemini":
            model = get_gemini_llm()
            response = await model.generate_content_async(decomposition_prompt)
            result = response.text
        else:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            llm = get_mistral_llm()
            template = ChatPromptTemplate.from_template("{input_prompt}")
            chain = template | llm | StrOutputParser()
            result = await chain.ainvoke({"input_prompt": decomposition_prompt})
        
        lines = result.strip().split('\n')
        sub_queries = []
        for line in lines:
            line = line.strip()
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            if line and len(line) > 10:
                sub_queries.append(line)
        
        return sub_queries[:3] if sub_queries else [query]
    except Exception as e:
        print(f"Query decomposition failed: {e}")
        return [query]

async def search_graph_multihop_async(query: str, index_name: str, use_gemini: bool = False) -> Tuple[str, List[Dict]]:
    model_type = "gemini" if use_gemini else "mistral"
    sub_queries = await decompose_query(query, model_type)
    
    if len(sub_queries) == 1 and sub_queries[0] == query:
        return await search_graph_async(query, index_name, use_gemini)
    
    print(f"🧠 Multi-hop: Decomposed into {sub_queries}")
    
    import asyncio
    tasks = [search_graph_async(sq, index_name, use_gemini) for sq in sub_queries]
    results = await asyncio.gather(*tasks)
    
    all_chunks = []
    all_sources = []
    seen_content = set()
    
    # Merge results
    for ctx, sources in results:
        # ctx is a string of [SOURCE_X] blocs
        # We need to parse them back or just append them and let ranker handle it
        # However, search_graph_async already reranks. 
        # For multihop, we should ideally retrieve from ALL subqueries and then do one FINAL rerank.
        # But to keep it simple and reuse existing logic, we'll collect the reranked results from each.
        if ctx:
            all_chunks.append(ctx)
        all_sources.extend(sources)

    # De-duplicate sources by title/source/preview
    unique_sources = []
    seen_sources = set()
    for s in all_sources:
        s_key = (s.get("title"), s.get("source"), s.get("preview", "")[:500])
        if s_key not in seen_sources:
            seen_sources.add(s_key)
            unique_sources.append(s)
    
    # Re-index sources for the LLM
    final_chunks = []
    for i, s in enumerate(unique_sources[:RETRIEVAL_K]):
        s["index"] = i + 1
        s["id"] = f"source_{i+1}"
        # We don't have the clean page_content easily here without repeating search_graph_async's work
        # but the search_graph_async returns the top chunks.
        # Let's just combine the contexts and inform the LLM
        pass

    # Simplified merge: join all contexts and deduplicate sources
    combined_context = "\n\n".join(all_chunks)
    
    return combined_context, unique_sources[:RETRIEVAL_K]


def get_gemini_llm():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set")
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel('gemini-2.5-flash')

def get_mistral_llm():
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY is not set")
    from langchain_mistralai import ChatMistralAI
    return ChatMistralAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        mistral_api_key=MISTRAL_API_KEY
    )

def build_prompt_with_history(question: str, context: str, history_msgs: List[Dict[str, Any]], enhanced_citations: bool = True) -> str:
    history_text = ""
    if history_msgs:
        recent = history_msgs[-10:] 
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
   
    history_section = f"CONVERSATION HISTORY:\n{history_text}" if history_text else ""
   
    citation_instruction = ""
    if enhanced_citations:
        citation_instruction = """
CITATION FORMAT & RELEVANCE RULES:
1. When referencing information from sources, use this format: "[SOURCE_X]" at the end of the sentence or clause.
2. ONLY cite a source if the information you are providing comes DIRECTLY from that specific source document.
3. DO NOT add unnecessary or generic citations. If a source is not relevant to the user's specific question, IGNORE it completely and do not cite it.
4. Example: "The concrete strength must be 4000 psi [SOURCE_1]. However, other specifications mention 5000 psi [SOURCE_2]."
5. Do NOT include specific source titles or descriptions in the text. Just use the bracketed identifier.
"""
   
    prompt = f"""You are a high-level WYDOT expert assistant. 
    
Your goal is to provide **comprehensive, accurate, and high-quality answers** based on the provided context.
Synthesize information from multiple sources to give a complete picture.

PRECISION & FORMATTING:
1. **Tables**: If the answer involves values from multiple sections, years, or categories, use a Markdown table for comparison.
2. **Exhaustive**: If a question asks for requirements, list ALL relevant requirements found in the context.
3. **Accuracy**: If information is missing or contradictory, state it clearly citing the specific sources.
4. **Version Comparison**: ONLY IF the context contains multiple versions of the EXACT SAME document marked with `[WARNING: ... SUPERSEDED]`, you must explicitly compare them. Highlight what has remained the same and provide a detailed breakdown of changes.
5. **Chronological Order**: ONLY IF you are performing a Version Comparison across superseded documents, present the most recent year first. DO NOT artificially group or sort standard documents by year (e.g., driver licenses, biographies) unless explicitly comparing versions of the same specification.

CONTEXTUAL INFERENCE:
- If asked "who is X?" and the context doesn't have a biography but mentions X in a role
  (e.g. "Mark Gordon, Governor"), infer and state the role/title from those contextual mentions.
- Use ALL available clues: document headers, meeting attendees, signatures, titles.

METADATA GUIDE:
- SOURCE: Filename
- YEAR: Document Year
- AUTHOR: Document Author (if available)
- SECTION: Specific section (e.g., Section 101)

{history_section}

CONTEXT:
{context}

{citation_instruction}

QUESTION: {question}

Instructions:
- Provide accurate, helpful answers based **strictly** on the relevant context.
- If the user is just saying hello or asking a conversational question (e.g., "hi", "how are you"), reply normally and politely without citing sources.
- For technical questions, if none of the provided documents contain the answer, state "I do not have enough information to answer this based on the provided documents" and DO NOT cite any sources.
- Structure your answer with clear headings and bullet points where appropriate.
"""
    return prompt


async def generate_answer_mistral_stream(question: str, context: str, history: List[Dict[str, Any]], enhanced_citations: bool = True):
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        llm = get_mistral_llm()
        full_prompt_str = build_prompt_with_history(question, context, history, enhanced_citations)
       
        template = ChatPromptTemplate.from_template("{input_prompt}")
        chain = template | llm | StrOutputParser()
        
        async for chunk in chain.astream({"input_prompt": full_prompt_str}):
            yield chunk
            
    except Exception as e:
        yield f"Error generating response: {e}"

async def generate_answer_gemini_stream(question: str, context: str, history: List[Dict[str, Any]], enhanced_citations: bool = True):
    try:
        model = get_gemini_llm()
        full_prompt_str = build_prompt_with_history(question, context, history, enhanced_citations)
       
        response_stream = await model.generate_content_async(full_prompt_str, stream=True)
        async for chunk in response_stream:
            try:
                if chunk.text:
                    yield chunk.text
            except ValueError:
                # Handle finish_reason=1 or cases where text part is missing
                continue
       
    except Exception as e:
        yield f"\n\n⚠️ Error generating response: {e}"

def generate_answer_mistral(question: str, context: str, history: List[Dict[str, Any]], enhanced_citations: bool = True) -> str:
    # Keep sync version for compatibility/fallback
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        llm = get_mistral_llm()
        full_prompt_str = build_prompt_with_history(question, context, history, enhanced_citations)
        template = ChatPromptTemplate.from_template("{input_prompt}")
        chain = template | llm | StrOutputParser()
        return chain.invoke({"input_prompt": full_prompt_str})
    except Exception as e:
        return f"Error: {e}"

def generate_answer_gemini(question: str, context: str, history: List[Dict[str, Any]], enhanced_citations: bool = True) -> str:
    try:
        model = get_gemini_llm()
        full_prompt_str = build_prompt_with_history(question, context, history, enhanced_citations)
        response = model.generate_content(full_prompt_str)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# 1. Shared HTTP Client for Anyio stability
_HTTP_CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0), verify=True)

async def generate_answer_openrouter_stream(model_id: str, question: str, context: str, history: List[Dict[str, Any]], enhanced_citations: bool = True):
    if not OPENROUTER_API_KEY:
        yield "Error: OPENROUTER_API_KEY is not set."
        return
    full_prompt = build_prompt_with_history(question, context, history, enhanced_citations)
    messages = [{"role": "system", "content": "You are a high-level WYDOT expert assistant."}]
    for msg in history[-MAX_HISTORY_MSGS:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": full_prompt})
    import json
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "HTTP-Referer": "https://wydot.gov", "X-Title": "WYDOT Assistant", "Content-Type": "application/json"}
    payload = {"model": model_id, "messages": messages, "stream": True, "temperature": LLM_TEMPERATURE}
    try:
        async with _HTTP_CLIENT.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                err_body = await response.aread()
                yield f"OpenRouter Error ({response.status_code}): {err_body.decode()}"
                if response.status_code == 401:
                    print(f"❌ OpenRouter 401 Details: {err_body.decode()}")
                return
            async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]": break
                        try:
                            data = json.loads(data_str)
                            chunk = data["choices"][0]["delta"].get("content", "")
                            if chunk: yield chunk
                        except: continue
    except Exception as e:
        yield f"OpenRouter Connection Error: {e}"

async def generate_answer_openrouter(model_id: str, question: str, context: str, history: List[Dict[str, Any]], enhanced_citations: bool = True) -> str:
    if not OPENROUTER_API_KEY: return "Error: OPENROUTER_API_KEY is not set."
    full_prompt = build_prompt_with_history(question, context, history, enhanced_citations)
    messages = [{"role": "system", "content": "You are a high-level WYDOT expert assistant."}]
    for msg in history[-MAX_HISTORY_MSGS:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": full_prompt})
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model_id, "messages": messages, "temperature": LLM_TEMPERATURE}
    try:
        response = await _HTTP_CLIENT.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            err_body = response.text
            if response.status_code == 401:
                print(f"❌ OpenRouter 401 Details: {err_body}")
            return f"OpenRouter Error ({response.status_code}): {err_body}"
    except Exception as e:
        return f"OpenRouter Connection Error: {e}"


# =========================================================
# ENHANCED CITATION PROCESSING
# =========================================================

def enhance_citations_in_response(response: str, sources: List[Dict[str, Any]]) -> str:
    """Transform [SOURCE_X] citations into clickable element references.

    Chainlit's frontend (a4e function) scans message text for element names
    and converts them to clickable links. We output just 'SOURCE X' (no brackets)
    so a4e can wrap it as [Source X](Source_X) for the side panel link.
    """

    def replace_citation(match):
        citation_text = match.group(0)
        source_num_match = re.search(r'SOURCE_(\d+)', citation_text)

        if not source_num_match:
            return citation_text

        source_num = source_num_match.group(1)
        source_id = f"source_{source_num}"

        # Check if source exists
        source_exists = any(src['id'] == source_id for src in sources)

        if source_exists:
            # Return element name without brackets so Chainlit's a4e function
            # can find it and wrap it as a clickable [Source X](Source_X) link.
            # Using brackets like [Source X] would cause a4e to produce
            # [[Source X](Source_X)] which is broken markdown.
            return f"Source {source_num}"

        return citation_text

    # Replace [SOURCE_X] pattern (including surrounding brackets)
    pattern = r'\[?SOURCE_(\d+)\]?'
    
    return re.sub(pattern, replace_citation, response)

# =========================================================
# CHAINLIT APP
# =========================================================

@cl.on_chat_start
async def start():
    # User is already verified if they passed auth_callback


    # Get or create thread ID for this conversation
    # Get or create thread ID for this conversation
    thread_id = cl.context.session.thread_id
    session_id = thread_id if thread_id else f"session_{int(time.time())}"
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("thread_id", thread_id)

    # Chat Settings
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Model",
                values=["Mistral Large", "Gemini 2.5 Flash"] + list(OPENROUTER_MODELS.keys()),
                initial_index=1,
            ),
            Select(
                id="index",
                label="Document Index",
                values=["All Documents", "2021 Specs", "2010 Specs"],
                initial_index=0,
            ),
            Switch(id="agentic_mode", label="Multi-Agent Mode (Domain Routing)", initial=False),
            Switch(id="thinking_mode", label="Thinking Mode (Slower, Detailed)", initial=False),
            Switch(id="multihop", label="Multi-hop Reasoning", initial=False),
            Switch(id="reranking", label="Reranking (FlashRank)", initial=False),
            Switch(id="hyde", label="HyDE (Query Expansion)", initial=False),
            Slider(id="fetch_k", label="Initial Candidates (FETCH_K)", min=10, max=100, step=5, initial=15),
        ]
    ).send()
    
    cl.user_session.set("settings", settings)
    
    # Send welcome message
    # await cl.Message(
    #     content="**Welcome to WYDOT Assistant!**\n\nAsk about specifications, upload plans/images for analysis, or use the audio button to speak."
    # ).send()
    
    # Initialize conversational memory for this session
    cl.user_session.set("memory", [])

@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("settings", settings)

@cl.on_message
async def main(message: cl.Message):
    settings = cl.user_session.get("settings") or {}
    model_choice = settings.get("model", "Mistral Large")
    index_choice = settings.get("index", "All Documents")
    thinking_mode = settings.get("thinking_mode", False)
    multihop = settings.get("multihop", False)
    
    index_map = {"All Documents": NEO4J_INDEX_DEFAULT, "2021 Specs": NEO4J_INDEX_2021}
    index_name = index_map.get(index_choice, NEO4J_INDEX_DEFAULT)
    
    # MULTIMODAL HANDLING
    files = message.elements
    has_media = any(f.mime and ("image" in f.mime or "audio" in f.mime or "pdf" in f.mime) for f in files)
    
    msg = cl.Message(content="")
    
    # 1. Handle Multimodal (Gemini required)
    if has_media:
        if model_choice != "Gemini 2.5 Flash":
            await cl.Message(content="⚠️ **Please switch to 'Gemini 2.5 Flash' in settings to analyze files.**").send()
            return
            
        await msg.stream_token("🧠 **Analyzing media with Gemini...**\n")
        
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        parts = [message.content]
        for f in files:
            with open(f.path, "rb") as fd:
                data = fd.read()
            parts.append({"mime_type": f.mime, "data": data})
            
        response = model.generate_content(parts).text
        await msg.stream_token(response)
        await msg.send()

        # Cache context for feedback callback (which has no session access)
        user = cl.user_session.get("user")
        _user_email = None
        if user:
            _user_email = getattr(user, 'identifier', None) or (user.metadata or {}).get("email")
        _feedback_context[msg.id] = {
            "question": message.content[:500],
            "answer": response[:1000],
            "user_email": _user_email,
            "sources": None,
        }
        while len(_feedback_context) > _FEEDBACK_CTX_MAX:
            _feedback_context.popitem(last=False)

        # Persist feedback context to DB (no FK — works for all users)
        try:
            CHAT_DB.save_feedback_context(
                cl_msg_id=msg.id,
                question=message.content[:500],
                answer=response[:1000],
                user_email=_user_email,
            )
        except Exception as e:
            print(f"[WARN] Failed to save feedback context: {e}")

        # Persist multimodal interaction if NOT guest
        if user and user.metadata.get("db_id") and not user.metadata.get("is_guest"):
            session_id = cl.user_session.get("session_id", "cl_session")
            uid = user.metadata["db_id"]
            file_note = f" [Attached {len(files)} file(s)]"
            CHAT_DB.add_message(uid, session_id, "user", message.content + file_note, cl_msg_id=message.id)
            CHAT_DB.add_message(uid, session_id, "assistant", response, cl_msg_id=msg.id)

        return

    # ── 1b. AGENTIC MULTI-AGENT MODE ──
    agentic_mode = settings.get("agentic_mode", False)
    if agentic_mode and _agentic_available:
        t0 = time.perf_counter()

        # Live loading message that updates as agents work
        loading_lines = ["**Multi-Agent Mode** — Routing to specialized agents...", ""]
        await msg.stream_token("\n".join(loading_lines))
        await msg.send()

        loop = asyncio.get_event_loop()

        def on_tool_call(event_type, tool_name, args, chunk_count):
            """Callback from orchestrator — updates loading message live."""
            if event_type == "start":
                display = TOOL_DISPLAY_NAMES.get(tool_name, f"Calling {tool_name}...")
                agent_label = TOOL_AGENT_LABELS.get(tool_name, "Agent")
                query_arg = args.get("query", args.get("topic", args.get("section_number", "")))
                loading_lines.append(f"{display}")
                loading_lines.append(f"   *{agent_label} — \"{query_arg[:60]}\"*")
                msg.content = "\n".join(loading_lines)
                asyncio.run_coroutine_threadsafe(msg.update(), loop)
            elif event_type == "done":
                agent_label = TOOL_AGENT_LABELS.get(tool_name, "Agent")
                loading_lines.append(f"   Returned **{chunk_count}** chunks")
                loading_lines.append("")
                msg.content = "\n".join(loading_lines)
                asyncio.run_coroutine_threadsafe(msg.update(), loop)

        try:
            chat_history = cl.user_session.get("memory", [])
            answer, sources_raw, tool_events = await run_orchestrator_async(
                message.content, chat_history, on_tool_call=on_tool_call
            )
            elapsed = time.time() - t0

            # Build agent routing header
            agents_used = list(dict.fromkeys(e["agent_label"] for e in tool_events))
            total_chunks = sum(e["chunk_count"] for e in tool_events)
            agents_str = ", ".join(f"**{a}**" for a in agents_used)
            header = f"*Agents: {agents_str} | {total_chunks} chunks | {elapsed:.1f}s*\n\n---\n\n"

            # Normalize citations [SOURCE X] -> [Source X]
            answer = re.sub(r'\[SOURCE\s+(\d+)', lambda m: f'[Source {m.group(1)}', answer, flags=re.IGNORECASE)

            # Build source elements for side panel
            clean_elements = []
            formatted_sources = []
            if sources_raw:
                for i, s in enumerate(sources_raw[:20], 1):
                    element_name = f"Source {i}"
                    title = s.get("title", "Unknown Document")
                    doc_source = s.get("source", "N/A")
                    page = s.get("page", "N/A")
                    section = s.get("section", "N/A")
                    year = s.get("year", "N/A")
                    text = s.get("text", "No content available.")

                    formatted_sources.append({
                        "index": i, "title": title, "source": doc_source,
                        "page": page, "section": section, "year": year,
                        "preview": text[:500],
                    })

                    # Only add if cited in answer
                    answer_upper = answer.upper()
                    if any(f"SOURCE {i}{c}" in answer_upper for c in ["]", ",", " ", "."]):
                        clean_elements.append(
                            cl.Text(
                                name=element_name,
                                content=f"**Source {i}: {title}**\n\n**File:** {doc_source}\n**Page:** {page}\n**Section:** {section}\n**Year:** {year}\n\n**Preview:**\n{text[:1500]}",
                                display="side"
                            )
                        )

            # Final message with header + answer + sources
            msg.content = header + answer
            msg.elements = clean_elements
            await msg.update()

            # ── Feedback context (same as regular flow) ──
            user = cl.user_session.get("user")
            _user_email = None
            if user:
                _user_email = getattr(user, 'identifier', None) or (user.metadata or {}).get("email")
            _feedback_context[msg.id] = {
                "question": message.content[:500],
                "answer": answer[:1000],
                "user_email": _user_email,
                "sources": formatted_sources,
            }
            while len(_feedback_context) > _FEEDBACK_CTX_MAX:
                _feedback_context.popitem(last=False)
            try:
                CHAT_DB.save_feedback_context(
                    cl_msg_id=msg.id,
                    question=message.content[:500],
                    answer=answer[:1000],
                    user_email=_user_email,
                    sources=formatted_sources,
                )
            except Exception as e:
                print(f"[WARN] Failed to save agentic feedback context: {e}")

            # ── Chat history persistence (same as regular flow) ──
            session_id = cl.user_session.get("session_id", "cl_session")
            if user and user.metadata.get("db_id") and not user.metadata.get("is_guest"):
                uid = user.metadata["db_id"]
                conv_mem.append(uid, session_id, "user", message.content)
                conv_mem.append(uid, session_id, "assistant", answer)
                CHAT_DB.add_message(uid, session_id, "user", message.content, cl_msg_id=message.id)
                CHAT_DB.add_message(uid, session_id, "assistant", answer, cl_msg_id=msg.id)
            else:
                memory = cl.user_session.get("memory", [])
                memory.append({"role": "user", "content": message.content})
                memory.append({"role": "assistant", "content": answer})
                if len(memory) > 20:
                    memory = memory[-20:]
                cl.user_session.set("memory", memory)

            # ── Telemetry ──
            total_ms = (time.perf_counter() - t0) * 1000
            telemetry.record_request(
                retrieval_latency_ms=total_ms * 0.6,
                generation_latency_ms=total_ms * 0.4,
                total_latency_ms=total_ms,
                num_sources=len(formatted_sources),
                model_used="Gemini (Agentic)",
                index_used="multi-agent",
                has_error=False,
            )

        except Exception as e:
            msg.content = f"**Error in Multi-Agent Mode:** {str(e)}"
            await msg.update()
            print(f"[AGENTIC ERROR] {e}", flush=True)
            traceback.print_exc()

        return

    # 2. Text Retrieval Route (with timing for telemetry)
    t0 = time.perf_counter()
    await msg.stream_token(" **Thinking...**\n")
    
    use_gemini = "Gemini" in model_choice
    if thinking_mode:
        # SYNC (Blocking) Search
        context, sources = search_graph(message.content, index_name)
    elif multihop:
        # INTENT ROUTING & ADVANCED GRAPH SEARCH
        await msg.stream_token(" **Analyzing query intent...**\n")
        intent = await route_query_intent(message.content, use_gemini)
        
        if intent in ["GLOBAL_ANALYTICS", "IMPACT_ANALYSIS", "GRAPH_TRAVERSAL"]:
            await msg.stream_token(f" **Graph Intent Detected ({intent}). Generating Cypher...**\n")
            context, sources = await search_graph_cypher_async(message.content, intent, use_gemini)
            if not context:
                await msg.stream_token(" **Cypher returned no results. Falling back to multi-hop vector RAG...**\n")
                context, sources = await search_graph_multihop_async(message.content, index_name, use_gemini)
        else:
            await msg.stream_token(" **Standard Search Intent. Retrieving documents...**\n")
            context, sources = await search_graph_multihop_async(message.content, index_name, use_gemini)
    else:
        # ASYNC Search
        context, sources = await search_graph_async(message.content, index_name, use_gemini)
        
    retrieval_ms = (time.perf_counter() - t0) * 1000

    if not context:
        await msg.stream_token("No relevant documents found.")
        await msg.send()
        telemetry.record_request(
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=0,
            total_latency_ms=retrieval_ms,
            num_sources=0,
            model_used=model_choice,
            index_used=index_name,
            has_error=False,
        )
        return

    # 3. Generate Answer (Streaming)
    if model_choice == "Gemini 2.5 Flash":
        model_type = "gemini"
    elif model_choice == "Mistral Large":
        model_type = "mistral"
    else:
        model_type = "openrouter"
        model_id = OPENROUTER_MODELS.get(model_choice, "openai/gpt-5.2-pro")
    user = cl.user_session.get("user")
    session_id = cl.user_session.get("session_id", "cl_session")

    if user and user.metadata.get("db_id") and not user.metadata.get("is_guest"):
        history_msgs = conv_mem.get_recent(
            user.metadata["db_id"], session_id, limit=MAX_HISTORY_MSGS,
            fallback=CHAT_DB.get_recent,
        )
    else:
        history_msgs = cl.user_session.get("memory", [])

    t_gen_start = time.perf_counter()
    
    # Streaming vs Sync Generation
    full_answer = ""
    try:
        if thinking_mode:
            # SYNC Generation (Wait for full answer)
            if model_type == "gemini":
                full_answer = generate_answer_gemini(message.content, context, history_msgs, enhanced_citations=True)
            elif model_type == "mistral":
                full_answer = generate_answer_mistral(message.content, context, history_msgs, enhanced_citations=True)
            else:
                full_answer = await generate_answer_openrouter(model_id, message.content, context, history_msgs, enhanced_citations=True)
            await msg.stream_token(full_answer) # Send all at once
        else:
            # ASYNC / Streaming Generation
            if model_type == "gemini":
                async for chunk in generate_answer_gemini_stream(message.content, context, history_msgs, enhanced_citations=True):
                    await msg.stream_token(chunk)
                    full_answer += chunk
            elif model_type == "mistral":
                async for chunk in generate_answer_mistral_stream(message.content, context, history_msgs, enhanced_citations=True):
                    await msg.stream_token(chunk)
                    full_answer += chunk
            else:
                async for chunk in generate_answer_openrouter_stream(model_id, message.content, context, history_msgs, enhanced_citations=True):
                    await msg.stream_token(chunk)
                    full_answer += chunk
                    
    except Exception as e:
        error_text = f"\n\n⚠️ Error during generation: {e}"
        await msg.stream_token(error_text)
        full_answer += error_text

    generation_ms = (time.perf_counter() - t_gen_start) * 1000
    total_ms = (time.perf_counter() - t0) * 1000

    # 4. Enhance Citations: Transform [SOURCE_X] -> [X]
    # Since we streamed [SOURCE_X], we might want to do a final replacement update
    enhanced_answer = enhance_citations_in_response(full_answer, sources)
    
    # Check if enhanced answer is different (it will be if citations were found)
    # We update the final message content to ensure clean citations
    if enhanced_answer != full_answer:
        msg.content = enhanced_answer
        await msg.update()

    # 5. Create source elements for side panel
    clean_elements = []
    for src in sources:
        element_name = f"Source {src['index']}"
        # ONLY add the source to the UI if the LLM actually found it relevant and cited it
        if element_name in enhanced_answer:
            clean_elements.append(
                cl.Text(
                    name=element_name,
                    content=f"**Source {src['index']}: {src['title']}**\n\n**File:** [{src['source']}]({src.get('url', '#')})\n**Page:** {src.get('page', 'N/A')}\n**Section:** {src.get('section', 'N/A')}\n**Year:** {src.get('year', 'N/A')}\n\n**Preview:**\n{src['preview']}",
                    display="side"
                )
            )
    # Final update with sources
    msg.elements = clean_elements
    await msg.update()

    # Cache context for feedback callback (which has no session access)
    _user_email = None
    if user:
        _user_email = getattr(user, 'identifier', None) or (user.metadata or {}).get("email")
    print(f"[FEEDBACK_CTX] Saving context for msg.id={msg.id}, question={message.content[:50]}, user={_user_email}", flush=True)
    _feedback_context[msg.id] = {
        "question": message.content[:500],
        "answer": enhanced_answer[:1000],
        "user_email": _user_email,
        "sources": sources,
    }
    while len(_feedback_context) > _FEEDBACK_CTX_MAX:
        _feedback_context.popitem(last=False)

    # Persist feedback context to DB (no FK constraints — works for all users incl guests)
    try:
        CHAT_DB.save_feedback_context(
            cl_msg_id=msg.id,
            question=message.content[:500],
            answer=enhanced_answer[:1000],
            user_email=_user_email,
            sources=sources,
        )
    except Exception as e:
        print(f"[WARN] Failed to save feedback context: {e}")

    # Update conversation: Redis + in-session; persist to DB if NOT guest
    if user and user.metadata.get("db_id") and not user.metadata.get("is_guest"):
        uid = user.metadata["db_id"]
        conv_mem.append(uid, session_id, "user", message.content)
        conv_mem.append(uid, session_id, "assistant", enhanced_answer)
        CHAT_DB.add_message(uid, session_id, "user", message.content, cl_msg_id=message.id)
        CHAT_DB.add_message(uid, session_id, "assistant", enhanced_answer, sources=sources, cl_msg_id=msg.id)
    else:
        history_msgs.append({"role": "user", "content": message.content})
        history_msgs.append({"role": "assistant", "content": enhanced_answer})
        cl.user_session.set("memory", history_msgs)

    # Online telemetry (local SQLite; Cloud: BigQuery)
    telemetry.record_request(
        retrieval_latency_ms=retrieval_ms,
        generation_latency_ms=generation_ms,
        total_latency_ms=total_ms,
        num_sources=len(sources),
        response_text=enhanced_answer,
        model_used=model_choice,
        index_used=index_name,
        user_id=user.metadata.get("db_id") if user else None,
        thread_id=cl.context.session.thread_id if cl.context.session else None,
        session_id=session_id,
        has_error=False,
    )

    # Online Evaluation (Background)
    online_eval.record_online_eval(
        session_id=session_id,
        question=message.content,
        answer=enhanced_answer,
        context=context,
        num_sources=len(sources),
        latency_ms=total_ms,
        has_error=False
    )
