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
import re
import time
import sqlite3
import threading
import hashlib
import hmac
import tempfile
import base64 as _b64
from typing import List, Dict, Any, Optional, Tuple

import chainlit as cl
from chainlit.input_widget import Select, Switch
from dotenv import load_dotenv

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
import urllib.parse

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

# =========================================================
# CONFIGURATION
# =========================================================

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

NEO4J_INDEX_DEFAULT = os.getenv("NEO4J_INDEX_DEFAULT", "wydot_vector_index")
NEO4J_INDEX_2021 = os.getenv("NEO4J_INDEX_2021", "wydot_vector_index_2021")
NEO4J_INDEX_2010 = os.getenv("NEO4J_INDEX_2010", "wydot_vector_index_2010")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-large-latest")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# Cloud Run only allows writes to /tmp - force the DB there
CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", "/tmp/wydot_chat_history.sqlite3")
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
# DATABASE / AUTH STORE
# =========================================================

def _pbkdf2_hash(password: str, iterations: int = 200_000) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64.b64encode(salt).decode()}${_b64.b64encode(dk).decode()}"

def _pbkdf2_verify(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_b64, dk_b64 = stored.split("$")
        if algo != "pbkdf2_sha256": return False
        iterations = int(iters)
        salt = _b64.b64decode(salt_b64)
        dk = _b64.b64decode(dk_b64)
        new_dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(new_dk, dk)
    except Exception:
        return False

class ChatHistoryStore:
    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        db_dir = os.path.dirname(db_path)
        if db_dir: os.makedirs(db_dir, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL;")
        except: pass
        self._init_tables()

    def _init_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT,
                created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT CHECK(role IN ('user','assistant')) NOT NULL,
                content TEXT NOT NULL,
                ts REAL NOT NULL DEFAULT (strftime('%s','now')),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        self._conn.commit()

    def authenticate(self, email: str, password: str) -> Tuple[Optional[int], Optional[str]]:
        email = (email or "").strip().lower()
        with self._lock:
            cur = self._conn.execute("SELECT id, password_hash, display_name FROM users WHERE email=?", (email,))
            row = cur.fetchone()
            if not row: return None, "User not found"
            uid, pw_hash, name = row
            if not _pbkdf2_verify(password, pw_hash): return None, "Invalid password"
            return uid, name

    def create_user(self, email: str, password: str, display_name: str = None) -> Tuple[Optional[int], Optional[str]]:
        email = (email or "").strip().lower()
        with self._lock:
            try:
                pw_hash = _pbkdf2_hash(password)
                self._conn.execute("INSERT INTO users (email, password_hash, display_name) VALUES (?,?,?)", (email, pw_hash, display_name))
                self._conn.commit()
                cur = self._conn.execute("SELECT id FROM users WHERE email=?", (email,))
                return int(cur.fetchone()[0]), None
            except Exception as e:
                return None, str(e)

    def add_message(self, user_id: int, session_id: str, role: str, content: str):
        with self._lock:
            self._conn.execute("INSERT INTO messages (user_id, session_id, role, content) VALUES (?, ?, ?, ?)", (user_id, session_id, role, content))
            self._conn.commit()
            
    def get_recent(self, user_id: int, session_id: str, limit: int = 20) -> List[Dict]:
        with self._lock:
            cur = self._conn.execute("SELECT role, content FROM messages WHERE user_id=? AND session_id=? ORDER BY id DESC LIMIT ?", (user_id, session_id, limit))
            rows = cur.fetchall()
        rows.reverse()
        return [{"role": r[0], "content": r[1]} for r in rows]

CHAT_DB = ChatHistoryStore(CHAT_DB_PATH)

# =========================================================
# CHAINLIT CALLBACKS
# =========================================================

# NOTE: Authentication and data layer disabled for anonymous access
# Users can use the chatbot without login
# To enable login with chat history, uncomment the @cl.password_auth_callback below

# @cl.password_auth_callback
# def auth_callback(username, password):
#     uid, name = CHAT_DB.authenticate(username, password)
#     if uid:
#         return cl.User(identifier=username, metadata={"db_id": uid, "name": name or username})
#     new_uid, error = CHAT_DB.create_user(username, password, display_name=username.split('@')[0] if '@' in username else username)
#     if new_uid:
#         return cl.User(identifier=username, metadata={"db_id": new_uid, "name": username})
#     return None

@cl.on_chat_resume
async def on_chat_resume(thread: Dict):
    """
    Called when user resumes a past conversation from the history panel.
    Restore the conversation state and messages.
    """
    # Get thread metadata
    thread_id = thread.get("id")
    user_id = thread.get("userId")
    
    # Set session state
    cl.user_session.set("thread_id", thread_id)
    
    # Load messages from database if available
    user = cl.user_session.get("user")
    if user and user.metadata.get("db_id"):
        # Map thread_id to our session_id format
        session_id = f"thread_{thread_id}"
        cl.user_session.set("session_id", session_id)
        
        # Load recent messages
        history_msgs = CHAT_DB.get_recent(user.metadata["db_id"], session_id, limit=MAX_HISTORY_MSGS)
        
        # Display them in the chat
        for msg in history_msgs:
            await cl.Message(
                content=msg["content"],
                author=msg["role"]
            ).send()
    
    await cl.Message(content="💬 **Conversation resumed!** Continue where you left off.").send()

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
            history_msgs = CHAT_DB.get_recent(user.metadata["db_id"], session_id, limit=MAX_HISTORY_MSGS)
        
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
        
        # Log to DB
        if user and user.metadata.get("db_id"):
            session_id = cl.user_session.get("session_id", "cl_session")
            CHAT_DB.add_message(user.metadata["db_id"], session_id, "user", f"[Voice] {transcribed_text}")
            CHAT_DB.add_message(user.metadata["db_id"], session_id, "assistant", enhanced_answer)
        
    except Exception as e:
        msg.content = f"⚠️ Error processing audio: {str(e)}"
        await msg.update()
    
    cl.user_session.set("audio_chunks", [])

# =========================================================
# MODELS & RETRIEVAL
# =========================================================

class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = "models/text-embedding-004"
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [genai.embed_content(model=self.model, content=t, task_type="retrieval_document")['embedding'] for t in texts]
    def embed_query(self, text: str) -> List[float]:
        return genai.embed_content(model=self.model, content=text, task_type="retrieval_query")['embedding']

def get_retriever(index_name: str, use_gemini: bool = False):
    try:
        if use_gemini and GEMINI_AVAILABLE and GEMINI_API_KEY:
            embeds = GeminiEmbeddings(GEMINI_API_KEY)
        else:
            embeds = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
        return Neo4jVector.from_existing_index(
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
    except Exception as e:
        print(f"Retriever error: {e}")
        return None

def search_graph(query: str, index_name: str, use_gemini: bool = False) -> Tuple[str, List[Dict]]:
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

def generate_answer_mistral(question: str, context: str, history: List[Dict[str, Any]], enhanced_citations: bool = True) -> str:
    try:
        llm = get_mistral_llm()
        full_prompt_str = build_prompt_with_history(question, context, history, enhanced_citations)
       
        template = ChatPromptTemplate.from_template("{input_prompt}")
        chain = template | llm | StrOutputParser()
       
        response = chain.invoke({"input_prompt": full_prompt_str})
        return response
       
    except Exception as e:
        return f"Error generating response: {e}"

def generate_answer_gemini(question: str, context: str, history: List[Dict[str, Any]], enhanced_citations: bool = True) -> str:
    try:
        model = get_gemini_llm()
        full_prompt_str = build_prompt_with_history(question, context, history, enhanced_citations)
       
        response = model.generate_content(full_prompt_str)
        return response.text
       
    except Exception as e:
        return f"Error generating response: {e}"

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
    
    # Get or create thread ID for this conversation
    thread_id = cl.context.session.thread_id
    session_id = f"thread_{thread_id}" if thread_id else f"session_{int(time.time())}"
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
        return

    # 2. Text Retrieval Route
    await msg.stream_token("🔍 **Searching documents...**\n")
    context, sources = search_graph(message.content, index_name)
    
    if not context:
        await msg.stream_token("No relevant documents found.")
        await msg.send()
        return

    # 3. Generate Answer
    model_type = "gemini" if "Gemini" in model_choice else "mistral"
    
    # Get conversation history from session (in-memory)
    # This enables multi-turn context without requiring database login
    history_msgs = cl.user_session.get("memory", [])

    if model_type == "gemini":
        answer = generate_answer_gemini(message.content, context, history_msgs, enhanced_citations=True)
    else:
        answer = generate_answer_mistral(message.content, context, history_msgs, enhanced_citations=True)
    
    # 4. Enhance Citations: Transform [SOURCE_X] -> [X]
    enhanced_answer = enhance_citations_in_response(answer, sources)
    
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
    
    msg.content = enhanced_answer
    msg.elements = clean_elements
    await msg.update()
    
    # Update conversation history in session
    history_msgs.append({"role": "user", "content": message.content})
    history_msgs.append({"role": "assistant", "content": enhanced_answer})
    cl.user_session.set("memory", history_msgs)
    
    # Log to DB (if authenticated)
    user = cl.user_session.get("user")
    if user and user.metadata.get("db_id"):
        CHAT_DB.add_message(user.metadata["db_id"], "cl_session", "user", message.content)
        CHAT_DB.add_message(user.metadata["db_id"], "cl_session", "assistant", enhanced_answer)
