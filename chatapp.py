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
print("🚀 App Startup Sequence Initiated")
print(f"📍 Current Working Directory: {os.getcwd()}")
print(f"🛠️ K_SERVICE Detection: {os.getenv('K_SERVICE', 'Not in Cloud Run')}")
print(f"📦 PORT: {os.getenv('PORT', 'Not Set')}")

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
import time
import tempfile
import urllib.parse
from dotenv import load_dotenv

import chainlit as cl
from chainlit.input_widget import Select, Switch
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# GraphRAG / LangChain imports
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings

# Google Vertex AI / Gemini
try:
    import google.generativeai as genai
    from google.generativeai import GenerativeModel
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

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
    load_dotenv(dotenv_path)
else:
    print(f"⚠️ DOTENV_PATH not found: {dotenv_path}, trying default .env")
    load_dotenv()

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
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-large-latest")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "8"))
MAX_HISTORY_MSGS = 20

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
print("=" * 60)

# =========================================================
# DATABASE / AUTH STORE (abstracted for local SQLite → Cloud SQL)
# =========================================================

try:
    from utils.chat_history_store import get_chat_history_store
    CHAT_DB = get_chat_history_store()
except ImportError:
    from chat_history_store import get_chat_history_store  # when run from repo root
    print("⚠️ Failure in chat history store import.")
    CHAT_DB = None

print("📈 Setting up Telemetry and Evaluation...")
# Online RAG telemetry (local SQLite; Cloud: BigQuery later)
try:
    from utils import telemetry
    from utils import evaluation as online_eval
except ImportError:
    import telemetry

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
        # Chainlit calls this when a user logs in.
        # We need to return a PersistedUser.
        # User class only has: identifier, display_name, metadata
        import time
        db_id = user.metadata.get("db_id", str(int(time.time()*1000)))
        return cl.PersistedUser(
            id=str(db_id),
            createdAt=str(time.time()),
            identifier=user.identifier,
            display_name=user.metadata.get("name"),
            metadata=user.metadata
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

        # Pass search_keyword to DB
        sessions = CHAT_DB.get_user_sessions(uid, search_term=search_keyword)
        
        # Resolve user email for userIdentifier (Chainlit compares this against user.identifier)
        user_dict = CHAT_DB.get_user_by_id(uid)
        user_email = user_dict.get("email", "") if user_dict else ""
        
        # Ensure we return ThreadDict objects
        threads: List[ThreadDict] = []
        import datetime
        for s in sessions:
            try:
                ts = s["createdAt"]
                if isinstance(ts, (int, float)):
                    created_at = datetime.datetime.fromtimestamp(ts).isoformat()
                else:
                     created_at = str(ts)
            except Exception:
                created_at = datetime.datetime.now().isoformat()

            # Use the DB session_id as the thread ID directly
            thread_id = s["id"]

            threads.append({
                "id": thread_id,
                "createdAt": created_at, 
                "name": s["name"],
                "userId": user_id,
                "userIdentifier": user_email,  # MUST be email, not numeric ID
                "tags": [],
                "metadata": {},
                "steps": [],
                "elements": [],
            })
            
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
        created_at = datetime.datetime.fromtimestamp(ts).isoformat() if isinstance(ts, (int, float)) else str(ts)

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
                step_type = "user_message" if role == "user" else "assistant_message"
                step_name = "user" if role == "user" else "WYDOT Assistant"
                msg_ts = msg.get("ts")
                if isinstance(msg_ts, (int, float)):
                    step_created = datetime.datetime.fromtimestamp(msg_ts).isoformat()
                else:
                    step_created = created_at  # fallback to thread creation time

                step_id = str(_uuid.uuid4())
                
                # Check for sources to restore elements
                msg_sources = msg.get("sources")
                if msg_sources and isinstance(msg_sources, list):
                    print(f"[DEBUG get_thread] Found {len(msg_sources)} sources for message {i}")
                    for src_idx, src in enumerate(msg_sources):
                        # Create Element dict manually as we are in Data Layer
                        # Using deterministic ID: src_{timestamp}_{index}
                        # Safe fallback if msg_ts is missing
                        safe_ts = int(msg_ts) if isinstance(msg_ts, (int, float)) else 0
                        src_id = f"src_{safe_ts}_{src_idx}"
                        elem = {
                            "id": src_id,
                            "threadId": thread_id,
                            "type": "text",
                            "url": None,
                            "chainlitKey": None,
                            "name": f"Source {src.get('index', '?')}",
                            "display": "side",
                            "objectKey": None,
                            "forId": step_id,
                            "mime": "text/markdown",
                            # Content of the source text element
                            "content": f"**Source {src.get('index', '?')}: {src.get('title', 'Unknown')}**\n\n**File:** [{src.get('source', 'File')}]({src.get('url', '#')})\n**Page:** {src.get('page', 'N/A')}\n**Section:** {src.get('section', 'N/A')}\n**Year:** {src.get('year', 'N/A')}\n\n**Preview:**\n{src.get('preview', '')}"
                        }
                        elements.append(elem)

                steps.append({
                    "id": step_id,
                    "threadId": thread_id,
                    "parentId": None,
                    "name": step_name,
                    "type": step_type,
                    "output": content,
                    "input": "",
                    "createdAt": step_created,
                    "start": step_created,
                    "end": step_created,
                    "metadata": {},
                    "streaming": False,
                    "isError": False,
                    "showInput": False,
                })

        result = {
            "id": thread_id,
            "createdAt": created_at,
            "name": thread_info["name"],
            "userId": thread_info["userId"],
            "userIdentifier": user_email,
            "tags": [],
            "metadata": {},
            "steps": steps,
            "elements": elements 
        }
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
        print(f"[DEBUG get_element] thread_id={thread_id}, element_id={element_id}")
        if not element_id.startswith("src_"):
            return None
        
        try:
            # element_id format: src_{ts}_{idx}
            parts = element_id.split("_")
            if len(parts) < 3:
                return None
            ts = int(parts[1])
            src_idx = int(parts[2])
            
            # Fetch thread info to get userId
            thread_info = CHAT_DB.get_session_by_id(thread_id)
            if not thread_info:
                return None
            
            uid = int(thread_info["userId"])
            # Fetch messages for this thread
            msgs = CHAT_DB.get_recent(uid, thread_id, MAX_HISTORY_MSGS)
            
            # Find the message with matching timestamp
            target_msg = None
            for m in msgs:
                m_ts = m.get("ts")
                if isinstance(m_ts, (int, float)) and int(m_ts) == ts:
                    target_msg = m
                    break
                elif ts == 0 and m_ts is None:
                    target_msg = m
                    break
            
            if not target_msg:
                return None
            
            msg_sources = target_msg.get("sources")
            if not msg_sources or src_idx >= len(msg_sources):
                return None
            
            src = msg_sources[src_idx]
            
            # Reconstruct the element (same content as in get_thread)
            return {
                "id": element_id,
                "threadId": thread_id,
                "type": "text",
                "url": None,
                "chainlitKey": None,
                "name": f"Source {src.get('index', '?')}",
                "display": "side",
                "objectKey": None,
                "mime": "text/markdown",
                "content": f"**Source {src.get('index', '?')}: {src.get('title', 'Unknown')}**\n\n**File:** [{src.get('source', 'File')}]({src.get('url', '#')})\n**Page:** {src.get('page', 'N/A')}\n**Section:** {src.get('section', 'N/A')}\n**Year:** {src.get('year', 'N/A')}\n\n**Preview:**\n{src.get('preview', '')}"
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
    """Login: authenticate ONLY. No auto-registration."""
    uid, name = CHAT_DB.authenticate(username, password)
    if uid:
        if not CHAT_DB.is_verified(uid):
            # In a real app, we might want to return a specific error or redirect,
            # but Chainlit auth callback just returns User or None.
            return None 
            
        is_guest = (username == "guest@app.local")
        return cl.User(
            # User class has: identifier, display_name, metadata (NO id field)
            identifier=username,
            metadata={"db_id": uid, "name": name or username, "verified": True, "is_guest": is_guest},
        )
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
                    values=["Mistral Large", "Gemini 2.5 Flash"],
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
        
        # Persist: Redis (fast) + DB (durable)
        if user and user.metadata.get("db_id"):
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
        genai.configure(api_key=api_key)
        self.model = "models/text-embedding-004"
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [genai.embed_content(model=self.model, content=t, task_type="retrieval_document")['embedding'] for t in texts]
    def embed_query(self, text: str) -> List[float]:
        return genai.embed_content(model=self.model, content=text, task_type="retrieval_query")['embedding']

def get_embeddings_model(use_gemini: bool = False):
    """Load embeddings once and reuse."""
    # check for vertex endpoint override first
    vertex_ep = os.getenv("VERTEX_EMBEDDING_ENDPOINT")
    
    key = "gemini" if use_gemini else ("vertex" if vertex_ep else "huggingface")
    
    if key in _EMBEDDINGS_CACHE:
        return _EMBEDDINGS_CACHE[key]

    if use_gemini and GEMINI_AVAILABLE and GEMINI_API_KEY:
        _EMBEDDINGS_CACHE[key] = GeminiEmbeddings(GEMINI_API_KEY)
    elif vertex_ep:
        print(f"🔄 Using Vertex AI Embeddings Endpoint: {vertex_ep}")
        _EMBEDDINGS_CACHE[key] = VertexCustomEmbeddings(vertex_ep)
    else:
        print("🔄 Loading embeddings model (local)...")
        _EMBEDDINGS_CACHE[key] = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("✅ Embeddings model cached.")
    return _EMBEDDINGS_CACHE[key]

def get_retriever(index_name: str, use_gemini: bool = False):
    cache_key = (index_name, use_gemini)
    if cache_key in _VECTOR_STORE_CACHE:
        return _VECTOR_STORE_CACHE[cache_key]
    try:
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
        ).as_retriever(search_kwargs={"k": RETRIEVAL_K})
        _VECTOR_STORE_CACHE[cache_key] = retriever
        return retriever
    except Exception as e:
        print(f"Retriever error: {e}")
        return None


async def search_graph_async(query: str, index_name: str, use_gemini: bool = False) -> Tuple[str, List[Dict]]:
    ret = get_retriever(index_name, use_gemini)
    if not ret: return "", []
    try:
        # Use ainvoke if available, otherwise run in thread
        if hasattr(ret, "ainvoke"):
            docs = await ret.ainvoke(query)
        else:
            # Fallback for older LangChain versions or synchronous retrievers
            import asyncio
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
        return "\n\n".join(chunks), sources
    except Exception as e:
        print(f"Search error: {e}")
        return "", []

def search_graph(query: str, index_name: str, use_gemini: bool = False) -> Tuple[str, List[Dict]]:
    # Sync version
    ret = get_retriever(index_name, use_gemini)
    if not ret: return "", []
    try:
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
            # Generate public URL logic omitted for brevity in sync fallback to avoid duplication, or strictly copy if needed.
            # For now, simplistic fallback:
            sources[-1]["url"] = "#"
        return "\n\n".join(chunks), sources
    except Exception as e:
        print(f"Search error: {e}")
        return "", []


def get_gemini_llm():
    if not GEMINI_AVAILABLE:
        raise ImportError("google-generativeai not installed")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set")
    genai.configure(api_key=GEMINI_API_KEY)
    return GenerativeModel('gemini-2.5-flash')

def get_mistral_llm():
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY is not set")
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
   
    prompt = f"""You are a WYDOT expert assistant. Answer based on the context below.

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
- Provide accurate, helpful answers based on the context
- Cite specific sections when relevant using the SOURCE_X format
- If the answer isn't in the context, say so clearly
- Keep answers concise but comprehensive
"""
    return prompt


async def generate_answer_mistral_stream(question: str, context: str, history: List[Dict[str, Any]], enhanced_citations: bool = True):
    try:
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


# =========================================================
# ENHANCED CITATION PROCESSING
# =========================================================

def enhance_citations_in_response(response: str, sources: List[Dict[str, Any]]) -> str:
    """Transform [SOURCE_X] citations into compact numbered citations [1], [2], etc."""
    
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
            # Return citation that matches element name: [Source 1]
            return f"[Source {source_num}]"
        
        return citation_text
    
    # Replace [SOURCE_X] pattern
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
                values=["Mistral Large", "Gemini 2.5 Flash"],
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
        
        genai.configure(api_key=GEMINI_API_KEY)
        model = GenerativeModel('gemini-2.5-flash')
        
        parts = [message.content]
        for f in files:
            with open(f.path, "rb") as fd:
                data = fd.read()
            parts.append({"mime_type": f.mime, "data": data})
            
        response = model.generate_content(parts).text
        await msg.stream_token(response)
        await msg.send()
        
        # Persist multimodal interaction
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
    
    if thinking_mode:
        # SYNC (Blocking) Search
        context, sources = search_graph(message.content, index_name)
    else:
        # ASYNC Search
        context, sources = await search_graph_async(message.content, index_name)
        
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
    model_type = "gemini" if "Gemini" in model_choice else "mistral"
    user = cl.user_session.get("user")
    session_id = cl.user_session.get("session_id", "cl_session")

    if user and user.metadata.get("db_id"):
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
            else:
                full_answer = generate_answer_mistral(message.content, context, history_msgs, enhanced_citations=True)
            await msg.stream_token(full_answer) # Send all at once
        else:
            # ASYNC / Streaming Generation
            if model_type == "gemini":
                async for chunk in generate_answer_gemini_stream(message.content, context, history_msgs, enhanced_citations=True):
                    await msg.stream_token(chunk)
                    full_answer += chunk
            else:
                async for chunk in generate_answer_mistral_stream(message.content, context, history_msgs, enhanced_citations=True):
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
        clean_elements.append(
            cl.Text(
                name=f"Source {src['index']}",
                content=f"**Source {src['index']}: {src['title']}**\n\n**File:** [{src['source']}]({src.get('url', '#')})\n**Page:** {src.get('page', 'N/A')}\n**Section:** {src.get('section', 'N/A')}\n**Year:** {src.get('year', 'N/A')}\n\n**Preview:**\n{src['preview']}",
                display="side"
            )
        )
    # Final update with sources
    msg.elements = clean_elements
    await msg.update()

    # Update conversation: Redis + in-session; persist to DB if authenticated AND NOT GUEST
    if user and user.metadata.get("db_id") and not user.metadata.get("is_guest"):
        uid = user.metadata["db_id"]
        conv_mem.append(uid, session_id, "user", message.content)
        conv_mem.append(uid, session_id, "assistant", enhanced_answer)
        # Pass sources to be saved in JSON column
        CHAT_DB.add_message(uid, session_id, "user", message.content)
        CHAT_DB.add_message(uid, session_id, "assistant", enhanced_answer, sources=sources)
    else:
        history_msgs = cl.user_session.get("memory", [])
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
