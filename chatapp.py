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

# === CLOUD RUN DIAGNOSTIC LOGGING ===
import os
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
    _os.environ.setdefault("CHAINLIT_DB_FILE", "/tmp/chainlit.db")
    print("✅ Cloud Run environment detected. Files redirected to /tmp.")
else:
    # Local: use project dir for persistence
    _os.environ.setdefault("CHAINLIT_DB_FILE", "chainlit.db")
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
app.routes.insert(_catch_all_idx, _Mount("/api", app=_api_app))
# --- END SOURCE CONTENT ENDPOINT ---

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
        # Encode the blob path (e.g. spaces to %20)
        encoded_blob = urllib.parse.quote(blob_name)
        return f"https://storage.googleapis.com/{bucket_name}/{encoded_blob}"
    except Exception as e:
        print(f"Error generating public URL: {e}")
        return ""

# Load environment variables from DOTENV_PATH or .env
dotenv_path = os.getenv("DOTENV_PATH", ".env")
if os.path.exists(dotenv_path):
    print(f"📂 Loading environment from: {dotenv_path}")
    load_dotenv(dotenv_path, override=True)  # Force override for local development
else:
    print(f"⚠️ DOTENV_PATH not found: {dotenv_path}, trying default .env")
    load_dotenv(override=True)

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

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
print(f"🔗 Neo4j Config: {NEO4J_URI} (User: {NEO4J_USERNAME})")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
NEO4J_INDEX_DEFAULT = os.getenv("NEO4J_INDEX_DEFAULT", "wydot_vector_index")
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
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-large-latest")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "10"))
FETCH_K = int(os.getenv("FETCH_K", "25"))

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
        # Lookup user by email/identifier
        user_dict = CHAT_DB.get_user_by_email(identifier)
        if not user_dict:
            return None
            
        return cl.PersistedUser(
            id=str(user_dict["id"]),
            createdAt=str(user_dict["created_at"]),
            identifier=user_dict["email"],
            display_name=user_dict["name"],
            metadata={"db_id": user_dict["id"], "name": user_dict["name"], "verified": True, "is_guest": False}
        )

    async def create_user(self, user: cl.User):
        """Called during registration or first login."""
        print(f"👤 [DATALAYER] create_user: {user.identifier}")
        import time
        
        # Check if user already has db_id in metadata (from auth_callback)
        db_id = user.metadata.get("db_id")
        
        if not db_id:
            # This is a REGISTRATION flow (user doesn't exist yet)
            password = getattr(user, "password", None)
            if password:
                print(f"📝 [DATALAYER] Registering new user: {user.identifier}")
                uid, error = CHAT_DB.create_user(user.identifier, password, user.display_name)
                if uid:
                    db_id = uid
                    print(f"✅ [DATALAYER] Registration successful (ID: {uid})")
                else:
                    print(f"❌ [DATALAYER] Registration failed: {error}")
                    return None
            else:
                # No password, and no db_id: something is wrong or it's an unsupported flow
                print(f"⚠️ [DATALAYER] No password or db_id for {user.identifier}")
                db_id = str(int(time.time()*1000)) # Fallback
        
        return cl.PersistedUser(
            id=str(db_id),
            createdAt=str(time.time()),
            identifier=user.identifier,
            display_name=user.display_name or user.identifier,
            metadata={"db_id": db_id, "name": user.display_name or user.identifier, "verified": True}
        )

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
                
                step_id = f"step_{actual_session_id}_{i}"
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
        pass
        
    async def update_step(self, step_dict):
        pass
        
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
            CHAT_DB.upsert_feedback(feedback)
            return True
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
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
    print(f"🔐 [AUTH] ====== LOGIN ATTEMPT ======", flush=True)
    print(f"🔐 [AUTH] Username: {username}", flush=True)
    print(f"🔐 [AUTH] CHAT_DB type: {type(CHAT_DB).__name__ if CHAT_DB else 'None'}", flush=True)
    if not CHAT_DB:
        print(f"❌ [AUTH] CHAT_DB is None! Cannot authenticate.", flush=True)
        return None
    try:
        result = CHAT_DB.authenticate(username, password)
        print(f"🔐 [AUTH] authenticate() returned: {result}", flush=True)
        uid, name = result
        if uid:
            print(f"✅ [AUTH] Success: {username} (ID: {uid})", flush=True)
            verified = CHAT_DB.is_verified(uid)
            print(f"🔐 [AUTH] is_verified({uid}) = {verified}", flush=True)
            if not verified:
                print(f"⚠️ [AUTH] User {username} not verified. Returning None.", flush=True)
                return None

            is_guest = (username == "guest@app.local")
            user = cl.User(
                identifier=username,
                display_name=name or username,
                metadata={"db_id": uid, "is_guest": is_guest},
            )
            print(f"✅ [AUTH] Returning cl.User: identifier={user.identifier}, is_guest={is_guest}", flush=True)
            return user
        print(f"❌ [AUTH] Denied: {username} (name={name})", flush=True)
    except Exception as e:
        print(f"🔥 [AUTH] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    print(f"❌ [AUTH] Returning None for {username}", flush=True)
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
                    initial_index=0,
                ),
                Select(
                    id="index",
                    label="Document Index",
                    values=["All Documents", "2021 Specs", "2010 Specs"],
                    initial_index=0,
                ),
                Switch(id="thinking_mode", label="Thinking Mode (Slower, Detailed)", initial=False),
                Switch(id="multihop", label="Multi-hop Reasoning", initial=False),
                Slider(id="fetch_k", label="Initial Candidates (FETCH_K)", min=10, max=100, step=5, initial=25),
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

# --- Lazy Loaders for Models ---

def get_embeddings_model(use_gemini: bool = False):
    """Load embeddings once and reuse."""
    from langchain_core.embeddings import Embeddings

    class VertexCustomEmbeddings(Embeddings):
        """Custom Embeddings class to call Vertex AI Prediction Endpoint."""
        def __init__(self, endpoint_url: str):
            self.endpoint_url = endpoint_url
            self.session = None

        def _get_session(self):
            import requests
            if not self.session:
                self.session = requests.Session()
            return self.session

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            try:
                session = self._get_session()
                response = session.post(self.endpoint_url, json={"instances": texts})
                response.raise_for_status()
                return response.json().get("predictions", [])
            except Exception as e:
                print(f"Vertex Embedding Error: {e}")
                return []

        def embed_query(self, text: str) -> List[float]:
            res = self.embed_documents([text])
            return res[0] if res else []

    class GeminiEmbeddings(Embeddings):
        def __init__(self, api_key: str):
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = "models/text-embedding-004"
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            import google.generativeai as genai
            return [genai.embed_content(model=self.model, content=t, task_type="retrieval_document")['embedding'] for t in texts]
        def embed_query(self, text: str) -> List[float]:
            import google.generativeai as genai
            return genai.embed_content(model=self.model, content=text, task_type="retrieval_query")['embedding']
    """Load embeddings once and reuse."""
    # Priority: Force local sentence-transformers as requested
    # We ignore use_gemini and vertex_ep if we want to be strict, 
    # but let's just make 'huggingface' the default and primary choice.
    
    key = "huggingface"
    if key in _EMBEDDINGS_CACHE:
        return _EMBEDDINGS_CACHE[key]

    print("🔄 Loading sentence-transformer model (all-MiniLM-L6-v2)...")
    from langchain_huggingface import HuggingFaceEmbeddings
    _EMBEDDINGS_CACHE[key] = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("✅ Embeddings model cached.")
    return _EMBEDDINGS_CACHE[key]
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

def get_neighbors(source_file: str, section: str, count: int = 1) -> str:
    """Fetch adjacent chunks from the same document section for more context."""
    if not source_file or not section:
        return ""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        with driver.session(database=NEO4J_DATABASE) as session:
            # Simple query to get chunks from same source and section
            # assuming 'source' and 'section' are properties on Chunk nodes
            query = """
            MATCH (c:Chunk {source: $source, section: $section})
            RETURN c.text as text
            LIMIT $limit
            """
            result = session.run(query, source=source_file, section=section, limit=count)
            texts = [record["text"] for record in result]
            return "\n".join(texts)
    except Exception as e:
        print(f"Error fetching neighbors: {e}")
        return ""


async def search_graph_async(query: str, index_name: str, use_gemini: bool = False) -> Tuple[str, List[Dict]]:
    ret = get_retriever(index_name, use_gemini)
    if not ret: return "", []
    # Use session setting for FETCH_K if available
    settings = cl.user_session.get("settings") or {}
    k_val = int(settings.get("fetch_k", FETCH_K))
    
    try:
        # Use ainvoke if available, otherwise run in thread
        if hasattr(ret, "ainvoke"):
            # Set k to k_val for initial retrieval
            ret.search_kwargs["k"] = k_val
            docs = await ret.ainvoke(query)
        else:
            # Fallback for older LangChain versions or synchronous retrievers
            import asyncio
            ret.search_kwargs["k"] = k_val
            docs = await asyncio.to_thread(ret.invoke, query)
            
        chunks, sources = [], []
        for i, doc in enumerate(docs):
            chunks.append(f"[SOURCE_{i+1}]\n{doc.page_content}")
            meta = doc.metadata
            title = meta.get("title", "Untitled")
            source_file = meta.get("source", "File")
            if title in ["Unknown", "Untitled", "None", ""]:
                title = source_file

            sources.append({
                "id": f"source_{i+1}",
                "index": i + 1, # citation index [1], [2]
                "title": title,
                "source": meta.get("source", "File"),
                "year": meta.get("year", ""),
                "section": meta.get("section", ""),
                "page": meta.get("page", ""),
                "preview": doc.page_content[:300]
            })
            
            # Generate signed URL if GCS path is available
            gcs_path = meta.get("gcs_path", "")
            
            # Fallback for documents in 'wydot documents' folder
            if not gcs_path:
                project_id = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
                if project_id:
                    bucket_name = f"wydot-documents-{project_id}"
                    source_file = meta.get("source", "")
                    if source_file:
                        gcs_path = f"gs://{bucket_name}/wydot documents/{source_file}"

            public_url = generate_public_url(gcs_path)
            
            # Append page number hash if available
            page = meta.get("page", "")
            if public_url and page:
                public_url += f"#page={page}"
                
            sources[-1]["url"] = public_url

        # --- FlashRank Reranking ---
        if chunks:
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
                    
                    for i, res in enumerate(top_results):
                        idx = res['id']
                        s_info = sources[idx]
                        s_info.update({"index": i + 1, "id": f"source_{i+1}", "score": f"{res['score']:.2f}"})
                        reranked_sources.append(s_info)
                        
                        # Re-read page_content from 'passages' since we have ID
                        content = str(passages[idx]['text'])
                        
                        # Enhancement: Graph-aware context expansion (Pulling in neighbors)
                        neighbors = get_neighbors(s_info.get("source"), s_info.get("section"), count=1)
                        if neighbors and neighbors.strip() not in content:
                             content += "\n--- Additional Context from same Section ---\n" + neighbors
                        
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
        docs = ret.invoke(query)
        chunks, sources = [], []
        for i, doc in enumerate(docs):
            chunks.append(f"[SOURCE_{i+1}]\n{doc.page_content}")
            meta = doc.metadata
            title = meta.get("title", "Untitled")
            source_file = meta.get("source", "File")
            if title in ["Unknown", "Untitled", "None", ""]:
                title = source_file
            sources.append({
                "id": f"source_{i+1}",
                "index": i + 1,
                "title": title,
                "source": meta.get("source", "File"),
                "year": meta.get("year", ""),
                "section": meta.get("section", ""),
                "page": meta.get("page", ""),
                "preview": doc.page_content[:300]
            })
            sources[-1]["url"] = "#"

        # --- FlashRank Reranking ---
        if chunks:
            ranker = get_reranker()
            if ranker:
                try:
                    passages = [{"id": i, "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(docs)]
                    rerank_request = RerankRequest(query=query, passages=passages)
                    results = ranker.rerank(rerank_request)
                    top_results = results[:RETRIEVAL_K]
                    
                    reranked_chunks, reranked_sources = [], []
                    for i, res in enumerate(top_results):
                        idx = res['id']
                        s_info = sources[idx]
                        s_info.update({"index": i + 1, "id": f"source_{i+1}", "score": f"{res['score']:.2f}"})
                        reranked_sources.append(s_info)
                        
                        content = str(passages[idx]['text'])
                        
                        # Enhancement: Graph-aware context expansion
                        neighbors = get_neighbors(str(s_info.get("source")), str(s_info.get("section")), count=1)
                        if neighbors and neighbors.strip() not in content:
                             content += "\n--- Additional Context from same Section ---\n" + neighbors
                             
                        reranked_chunks.append(f"[SOURCE_{i+1}]\n{content}")
                    return "\n\n".join(reranked_chunks), reranked_sources
                except Exception as e:
                    print(f"Sync Reranking failed: {e}")

        return "\n\n".join(chunks), sources
    except Exception as e:
        print(f"Search error: {e}")
        return "", []

async def decompose_query(query: str, model_type: str = "mistral") -> List[str]:
    decomposition_prompt = f"""Break down this complex question into 2-3 simpler sub-questions that need to be answered to fully address the main question.

Main Question: {query}

Provide the sub-questions as a numbered list, one per line. Be concise. Do NOT add any preamble.
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
        s_key = (s.get("title"), s.get("source"), s.get("preview", "")[:100])
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
CITATION FORMAT:
When referencing information from sources, use this format:
"[SOURCE_X]" at the end of the sentence or clause.

Example: "The concrete strength must be 4000 psi [SOURCE_1]. However, other specifications mention 5000 psi [SOURCE_2]."

Do NOT include specific source titles or descriptions in the text. Just use the bracketed identifier.
"""
   
    prompt = f"""You are a high-level WYDOT expert assistant. 
    
Your goal is to provide **comprehensive, accurate, and high-quality answers** based on the provided context.
Synthesize information from multiple sources to give a complete picture.

PRECISION & FORMATTING:
1. **Tables**: If the answer involves values from multiple sections, years, or categories, use a Markdown table for comparison.
2. **Exhaustive**: If a question asks for requirements, list ALL relevant requirements found in the context.
3. **Accuracy**: If information is missing or contradictory, state it clearly citing the specific sources.

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
- Provide accurate, helpful answers based on the context.
- Cite specific sections when relevant using the [SOURCE_X] format.
- If the answer isn't in the context, say so clearly.
- Keep answers concise but comprehensive.
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
             yield chunk.text
       
    except Exception as e:
        yield f"Error generating response: {e}"

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
                initial_index=0,
            ),
            Select(
                id="index",
                label="Document Index",
                values=["All Documents", "2021 Specs", "2010 Specs"],
                initial_index=0,
            ),
            Switch(id="thinking_mode", label="Thinking Mode (Slower, Detailed)", initial=False),
            Switch(id="multihop", label="Multi-hop Reasoning", initial=False),
            Slider(id="fetch_k", label="Initial Candidates (FETCH_K)", min=10, max=100, step=5, initial=25),
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
        
        # Persist multimodal interaction if NOT guest
        user = cl.user_session.get("user")
        if user and user.metadata.get("db_id") and not user.metadata.get("is_guest"):
            session_id = cl.user_session.get("session_id", "cl_session")
            uid = user.metadata["db_id"]
            # Save user message with note about files
            file_note = f" [Attached {len(files)} file(s)]"
            CHAT_DB.add_message(uid, session_id, "user", message.content + file_note)
            CHAT_DB.add_message(uid, session_id, "assistant", response)
            
        return

    # 2. Text Retrieval Route (with timing for telemetry)
    t0 = time.perf_counter()
    await msg.stream_token("🔍 **Searching documents...**\n")
    
    use_gemini = "Gemini" in model_choice
    if thinking_mode:
        # SYNC (Blocking) Search
        context, sources = search_graph(message.content, index_name)
    elif multihop:
        # MULTI-HOP Search
        await msg.stream_token("🧠 **Analyzing multi-part question...**\n")
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
        # Reconstruct the source ID for the element name to match the citation format
        # This ensures that [Source X] in the response links to the correct sidebar element
        element_name = f"Source {src['index']}"
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

    # Update conversation: Redis + in-session; persist to DB if NOT guest
    if user and user.metadata.get("db_id") and not user.metadata.get("is_guest"):
        uid = user.metadata["db_id"]
        conv_mem.append(uid, session_id, "user", message.content)
        conv_mem.append(uid, session_id, "assistant", enhanced_answer)
        CHAT_DB.add_message(uid, session_id, "user", message.content)
        CHAT_DB.add_message(uid, session_id, "assistant", enhanced_answer, sources=sources)
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
