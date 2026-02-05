#!/usr/bin/env python3
"""
WYDOT GraphRAG Chatbot - Enhanced Production Version (Fixed)
Works with existing Neo4j database (1,500+ documents already ingested)
Features: 
- Truly scrollable chat with fixed evidence panel
- Chat composer stays at bottom
- Citations scroll within evidence panel only
- Model selection (Mistral + Gemini 2.5 Flash)
- Multi-hop reasoning toggle
"""

# =========================================================
# MACOS FIX - MUST BE BEFORE OTHER IMPORTS
# =========================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import sqlite3
import threading
import hashlib
import hmac
import tempfile
import base64 as _b64
import re
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
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

# Load environment variables
load_dotenv()

# =========================================================
# CONFIGURATION
# =========================================================

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Index names
NEO4J_INDEX_DEFAULT = os.getenv("NEO4J_INDEX_DEFAULT", "wydot_vector_index")
NEO4J_INDEX_2021 = os.getenv("NEO4J_INDEX_2021", "wydot_vector_index_2021")
NEO4J_INDEX_2010 = os.getenv("NEO4J_INDEX_2010", "wydot_vector_index_2010")

# LLM Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-large-latest")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# Chat history database
CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", os.path.join(tempfile.gettempdir(), "wydot_chat_history.sqlite3"))

# App settings
MAX_HISTORY_MSGS = 20
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "8"))

# =========================================================
# CUSTOM CSS FOR FIXED LAYOUT
# =========================================================

CUSTOM_CSS = """
<style>
    /* Hide default streamlit elements for cleaner layout */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container adjustments */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Chat messages container - scrollable */
    div[data-testid="stVerticalBlock"] > div:has(div.chat-messages-container) {
        height: calc(100vh - 300px);
        overflow-y: auto;
        overflow-x: hidden;
        padding-right: 10px;
        margin-bottom: 10px;
    }
    
    /* Evidence panel - fixed/sticky */
    .evidence-panel {
        position: sticky;
        top: 20px;
        height: calc(100vh - 200px);
        overflow-y: auto;
        overflow-x: hidden;
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    
    /* Smooth scrolling */
    .evidence-panel {
        scroll-behavior: smooth;
    }
    
    /* Source reference links */
    .source-ref {
        background-color: #e3f2fd;
        padding: 1px 5px;
        border-radius: 4px;
        cursor: pointer;
        text-decoration: none;
        color: #1976d2;
        font-weight: 600;
        font-size: 0.8em;
        vertical-align: super;
        border: 1px solid #90caf9;
        display: inline-block;
        margin: 0 1px;
        transition: all 0.2s ease;
    }
    
    .source-ref:hover {
        background-color: #bbdefb;
        border-color: #64b5f6;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Highlighted source in evidence panel */
    .source-highlight {
        background-color: #fff9c4 !important;
        border: 3px solid #fbc02d !important;
        padding: 15px !important;
        border-radius: 8px !important;
        animation: highlight-pulse 1s ease-in-out;
        scroll-margin-top: 20px;
    }
    
    @keyframes highlight-pulse {
        0%, 100% { 
            box-shadow: 0 0 0 0 rgba(251, 192, 45, 0.4);
            transform: scale(1);
        }
        50% { 
            box-shadow: 0 0 0 10px rgba(251, 192, 45, 0);
            transform: scale(1.02);
        }
    }
    
    /* Source cards in evidence panel */
    .source-card {
        background: white;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        border: 1px solid #ddd;
        transition: all 0.3s ease;
    }
    
    .source-card:hover {
        border-color: #90caf9;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Multi-hop reasoning indicator */
    .multihop-indicator {
        background-color: #e8f5e9;
        padding: 10px 15px;
        border-radius: 6px;
        border-left: 4px solid #4caf50;
        margin: 10px 0;
        font-weight: 500;
    }
    
    /* Reasoning step */
    .reasoning-step {
        background-color: #f3e5f5;
        padding: 12px;
        border-radius: 6px;
        margin: 10px 0;
        border-left: 3px solid #9c27b0;
    }
    
    /* Model indicator badge */
    .model-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-left: 8px;
    }
    
    .model-mistral {
        background-color: #ff6b35;
        color: white;
    }
    
    .model-gemini {
        background-color: #4285f4;
        color: white;
    }
    
    /* Source content preview */
    .source-preview {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 12px;
        max-height: 200px;
        overflow-y: auto;
        margin-top: 8px;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    /* Evidence panel header */
    .evidence-header {
        position: sticky;
        top: 0;
        background-color: #f8f9fa;
        padding: 10px 0;
        z-index: 10;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 15px;
    }
</style>
"""

# JavaScript for smooth scrolling to sources
SCROLL_SCRIPT = """
<script>
function scrollToSource(sourceId) {
    const evidencePanel = document.querySelector('.evidence-panel');
    const sourceElement = document.getElementById(sourceId);
    
    if (evidencePanel && sourceElement) {
        // Remove previous highlights
        document.querySelectorAll('.source-highlight').forEach(el => {
            el.classList.remove('source-highlight');
        });
        
        // Add highlight to clicked source
        sourceElement.classList.add('source-highlight');
        
        // Scroll within evidence panel
        const elementPosition = sourceElement.offsetTop;
        const panelScroll = evidencePanel.scrollTop;
        const panelHeight = evidencePanel.clientHeight;
        const elementHeight = sourceElement.clientHeight;
        
        // Calculate scroll position to center the element
        const scrollTo = elementPosition - (panelHeight / 2) + (elementHeight / 2);
        
        evidencePanel.scrollTo({
            top: scrollTo,
            behavior: 'smooth'
        });
    }
}

// Add click handlers to source references
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.source-ref').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const href = this.getAttribute('href');
            if (href && href.startsWith('#')) {
                scrollToSource(href.substring(1));
            }
        });
    });
});
</script>
"""

# =========================================================
# AUTHENTICATION & USER MANAGEMENT
# =========================================================

def _pbkdf2_hash(password: str, iterations: int = 200_000) -> str:
    """Hash password using PBKDF2"""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64.b64encode(salt).decode()}${_b64.b64encode(dk).decode()}"

def _pbkdf2_verify(password: str, stored: str) -> bool:
    """Verify password against stored hash"""
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

class ChatHistoryStore:
    """SQLite-based chat history with user authentication"""
   
    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
       
        try:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
       
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
       
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_session ON messages (user_id, session_id, id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)")
        self._conn.commit()
   
    def create_user(self, email: str, password: str, display_name: Optional[str] = None) -> Tuple[Optional[int], Optional[str]]:
        email = (email or "").strip().lower()
        if not email or not password:
            return None, "Email and password are required."
       
        with self._lock:
            try:
                cur = self._conn.execute("SELECT id FROM users WHERE email=?", (email,))
                if cur.fetchone():
                    return None, "This email is already registered."
               
                pw_hash = _pbkdf2_hash(password)
                self._conn.execute(
                    "INSERT INTO users (email, password_hash, display_name) VALUES (?,?,?)",
                    (email, pw_hash, display_name)
                )
                self._conn.commit()
               
                cur = self._conn.execute("SELECT id FROM users WHERE email=?", (email,))
                row = cur.fetchone()
                return (int(row[0]) if row else None), None
            except Exception as e:
                return None, f"Database error: {e}"
   
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
   
    def add_message(self, user_id: int, session_id: str, role: str, content: str):
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (user_id, session_id, role, content) VALUES (?, ?, ?, ?)",
                (user_id, session_id, role, content)
            )
            self._conn.commit()
   
    def get_recent(self, user_id: int, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT role, content, ts FROM messages WHERE user_id=? AND session_id=? ORDER BY id DESC LIMIT ?",
                (user_id, session_id, limit)
            )
            rows = cur.fetchall()
        rows.reverse()
        return [{"role": r[0], "content": r[1], "ts": r[2]} for r in rows]
   
    def list_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT session_id, COUNT(*) AS n, MAX(ts) AS last_ts
                FROM messages WHERE user_id=?
                GROUP BY session_id ORDER BY last_ts DESC
                """,
                (user_id,)
            )
            rows = cur.fetchall()
        return [{"session_id": r[0], "count": int(r[1]), "last_ts": float(r[2])} for r in rows]
   
    def clear_session(self, user_id: int, session_id: str):
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE user_id=? AND session_id=?", (user_id, session_id))
            self._conn.commit()

@st.cache_resource(show_spinner=False)
def get_chat_store(db_path: str):
    return ChatHistoryStore(db_path)

CHAT_DB = get_chat_store(CHAT_DB_PATH)

def current_user_id() -> int:
    return int(st.session_state.get("user_id", 0))

def effective_user_id() -> int:
    uid = current_user_id()
    if uid and uid > 0:
        return uid
    if "anon_uid" not in st.session_state:
        st.session_state["anon_uid"] = int.from_bytes(os.urandom(8), "big") & ((1 << 63) - 1)
    return int(st.session_state["anon_uid"])

# =========================================================
# GEMINI EMBEDDINGS WRAPPER
# =========================================================

class GeminiEmbeddings(Embeddings):
    """Custom Gemini Embeddings wrapper for LangChain"""
    
    def __init__(self, api_key: str, model_name: str = "models/text-embedding-004"):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed")
        genai.configure(api_key=api_key)
        self.model_name = model_name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']

# =========================================================
# NEO4J GRAPHRAG RETRIEVAL
# =========================================================

@st.cache_resource(show_spinner=False)
def get_embeddings(use_gemini: bool = False):
    if use_gemini and GEMINI_AVAILABLE and GEMINI_API_KEY:
        return GeminiEmbeddings(api_key=GEMINI_API_KEY)
    else:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_retriever(index_name: str = NEO4J_INDEX_DEFAULT, use_gemini: bool = False):
    try:
        embeddings = get_embeddings(use_gemini)
        return Neo4jVector.from_existing_index(
            embeddings,
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
        print(f"Neo4j connection failed: {e}")
        return None

def search_graph(query: str, index_name: str = NEO4J_INDEX_DEFAULT, k: int = None, use_gemini: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
    if not query or not query.strip():
        return "", []
   
    k = k or RETRIEVAL_K
    retriever = get_retriever(index_name, use_gemini)
   
    if not retriever:
        return "", []
   
    try:
        source_docs = retriever.invoke(query)[:k]
       
        chunks = []
        sources = []
       
        for idx, doc in enumerate(source_docs):
            content = doc.page_content
            chunks.append(f"[SOURCE_{idx+1}]\n{content}")
           
            meta = doc.metadata
            sources.append({
                "id": f"source_{idx+1}",
                "index": idx + 1,
                "title": meta.get("title", "Untitled"),
                "source": meta.get("source", "unknown"),
                "year": meta.get("year", ""),
                "section": meta.get("section", ""),
                "author": meta.get("author", ""),
                "page": meta.get("page", 1),
                "content": content,
                "preview": content[:400] if content else ""
            })
       
        return "\n\n".join(chunks), sources
       
    except Exception as e:
        st.warning(f"Search error: {e}")
        return "", []

# =========================================================
# LLM GENERATION
# =========================================================

@st.cache_resource(show_spinner=False)
def get_mistral_llm():
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY is not set")
    return ChatMistralAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        mistral_api_key=MISTRAL_API_KEY
    )

@st.cache_resource(show_spinner=False)
def get_gemini_llm():
    if not GEMINI_AVAILABLE:
        raise ImportError("google-generativeai not installed")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set")
    genai.configure(api_key=GEMINI_API_KEY)
    return GenerativeModel('gemini-2.5-flash')

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
# MULTI-HOP REASONING
# =========================================================

def decompose_query(query: str, model_type: str = "mistral") -> List[str]:
    decomposition_prompt = f"""Break down this complex question into 2-3 simpler sub-questions that need to be answered to fully address the main question.

Main Question: {query}

Provide the sub-questions as a numbered list, one per line. Be concise.
"""
    
    try:
        if model_type == "gemini" and GEMINI_AVAILABLE and GEMINI_API_KEY:
            model = get_gemini_llm()
            response = model.generate_content(decomposition_prompt)
            result = response.text
        else:
            llm = get_mistral_llm()
            template = ChatPromptTemplate.from_template("{input_prompt}")
            chain = template | llm | StrOutputParser()
            result = chain.invoke({"input_prompt": decomposition_prompt})
        
        lines = result.strip().split('\n')
        sub_queries = []
        for line in lines:
            line = line.strip()
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            if line and len(line) > 10:
                sub_queries.append(line)
        
        return sub_queries[:3]
    except Exception as e:
        st.warning(f"Query decomposition failed: {e}")
        return [query]

def multihop_reasoning(query: str, index_name: str, model_type: str = "mistral", use_gemini_embeddings: bool = False) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    sub_queries = decompose_query(query, model_type)
    
    reasoning_steps = []
    all_sources = []
    all_contexts = []
    
    for i, sub_q in enumerate(sub_queries):
        context, sources = search_graph(sub_q, index_name, use_gemini=use_gemini_embeddings)
        
        if context:
            if model_type == "gemini":
                sub_answer = generate_answer_gemini(sub_q, context, [], enhanced_citations=True)
            else:
                sub_answer = generate_answer_mistral(sub_q, context, [], enhanced_citations=True)
            
            reasoning_steps.append({
                "step": i + 1,
                "query": sub_q,
                "answer": sub_answer,
                "sources": len(sources)
            })
            
            all_contexts.append(f"Sub-question {i+1}: {sub_q}\nAnswer: {sub_answer}")
            all_sources.extend(sources)
    
    combined_context = "\n\n".join(all_contexts)
    synthesis_prompt = f"""Based on the following sub-question answers, provide a comprehensive answer to the main question.

Main Question: {query}

Sub-question Analysis:
{combined_context}

Provide a clear, comprehensive answer that synthesizes all the information above."""
    
    if model_type == "gemini":
        final_answer = generate_answer_gemini(query, synthesis_prompt, [], enhanced_citations=False)
    else:
        final_answer = generate_answer_mistral(query, synthesis_prompt, [], enhanced_citations=False)
    
    return final_answer, all_sources, reasoning_steps

# =========================================================
# ENHANCED CITATION PROCESSING
# =========================================================

def enhance_citations_in_response(response: str, sources: List[Dict[str, Any]]) -> str:
    """Enhance response with clickable citation links"""
    
    def replace_citation(match):
        # Match patterns like "[SOURCE_1]"
        citation_text = match.group(0)
        source_num_match = re.search(r'SOURCE_(\d+)', citation_text)
        
        if not source_num_match:
            return citation_text
            
        source_num = source_num_match.group(1)
        source_id = f"source_{source_num}"
        
        # Check if source exists
        source_exists = any(src['id'] == source_id for src in sources)
        
        if source_exists:
            # Create compact clickable link: [1]
            return f'<a href="#{source_id}" class="source-ref" onclick="scrollToSource(\'{source_id}\'); return false;" title="Click to view source">[{source_num}]</a>'
        
        return citation_text
    
    # Replace [SOURCE_X] pattern
    # We look for [SOURCE_X] or just SOURCE_X if the model forgets brackets
    pattern = r'\[?SOURCE_(\d+)\]?'
    
    return re.sub(pattern, replace_citation, response)

# =========================================================
# UI COMPONENTS
# =========================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔐 Account")
       
        if current_user_id() > 0:
            display_name = CHAT_DB.get_display_name(current_user_id())
            st.success(f"👤 {display_name}")
           
            if st.button("🔓 Logout", use_container_width=True):
                st.session_state["user_id"] = 0
                st.session_state["user_email"] = None
                st.session_state["session_id"] = f"session_{int(time.time())}"
                st.rerun()
        else:
            tab_login, tab_signup = st.tabs(["Login", "Sign Up"])
           
            with tab_login:
                with st.form("login_form"):
                    email = st.text_input("Email")
                    password = st.text_input("Password", type="password")
                    submit = st.form_submit_button("Login", use_container_width=True)
                   
                    if submit:
                        uid, err = CHAT_DB.authenticate(email, password)
                        if err:
                            st.error(err)
                        else:
                            st.session_state["user_id"] = uid
                            st.session_state["user_email"] = email
                            st.success("✅ Logged in!")
                            st.rerun()
           
            with tab_signup:
                with st.form("signup_form"):
                    email = st.text_input("Email", key="signup_email")
                    name = st.text_input("Name (optional)")
                    password = st.text_input("Password", type="password", key="signup_pass")
                    confirm = st.text_input("Confirm Password", type="password")
                    submit = st.form_submit_button("Create Account", use_container_width=True)
                   
                    if submit:
                        if password != confirm:
                            st.error("Passwords don't match")
                        else:
                            uid, err = CHAT_DB.create_user(email, password, name)
                            if err:
                                st.error(err)
                            else:
                                st.session_state["user_id"] = uid
                                st.session_state["user_email"] = email
                                st.success("✅ Account created!")
                                st.rerun()
       
        st.markdown("---")
       
        # Model Selection
        st.markdown("## 🤖 Model Selection")
        
        available_models = ["Mistral Large"]
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            available_models.append("Gemini 2.5 Flash")
        
        selected_model = st.selectbox(
            "LLM Model",
            options=available_models,
            key="selected_model_display",
            help="Select the language model to use"
        )
        
        if "Gemini" in selected_model:
            st.session_state["selected_model"] = "gemini"
            st.session_state["use_gemini_embeddings"] = False
            st.info("🔹 Using HuggingFace embeddings (Gemini for Answering)")
        else:
            st.session_state["selected_model"] = "mistral"
            st.session_state["use_gemini_embeddings"] = False
            st.info("🔹 Using HuggingFace embeddings")
        
        st.markdown("---")
        
        # Multi-hop reasoning toggle
        st.markdown("## 🧠 Reasoning Mode")
        multihop_enabled = st.checkbox(
            "Enable Multi-hop Reasoning",
            value=st.session_state.get("multihop_enabled", False),
            help="Break down complex queries into sub-questions for better answers"
        )
        st.session_state["multihop_enabled"] = multihop_enabled
        
        if multihop_enabled:
            st.success("✅ Multi-hop mode active")
        
        st.markdown("---")
       
        # Neo4j status
        st.markdown("## 📊 Database Status")
        try:
            if get_retriever():
                st.success("✅ Neo4j Connected")
                st.caption(f"📍 {NEO4J_URI}")
            else:
                st.error("❌ Connection failed")
        except Exception as e:
            st.error(f"❌ Error: {str(e)[:50]}")
       
        st.markdown("---")
       
        # Index selection
        st.markdown("## 📚 Document Index")
        available_indexes = {
            "All Documents": NEO4J_INDEX_DEFAULT,
            "2021 Specifications": NEO4J_INDEX_2021,
            "2010 Specifications": NEO4J_INDEX_2010,
        }
       
        selected_index = st.selectbox(
            "Select index",
            options=list(available_indexes.keys()),
            help="Choose which document set to search"
        )
        st.session_state["selected_index"] = available_indexes[selected_index]
       
        st.markdown("---")
       
        # Session management
        st.markdown("## 💬 Conversations")
       
        sessions = CHAT_DB.list_sessions(effective_user_id())
       
        if sessions:
            session_labels = [f"{s['session_id']} ({s['count']} msgs)" for s in sessions]
            selected_idx = st.selectbox(
                "Select conversation",
                options=range(len(sessions)),
                format_func=lambda i: session_labels[i]
            )
           
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📂 Open", use_container_width=True):
                    st.session_state["session_id"] = sessions[selected_idx]["session_id"]
                    st.rerun()
           
            with col2:
                if st.button("🗑️ Delete", use_container_width=True):
                    CHAT_DB.clear_session(effective_user_id(), sessions[selected_idx]["session_id"])
                    st.rerun()
       
        if st.button("➕ New Conversation", use_container_width=True):
            st.session_state["session_id"] = f"session_{int(time.time())}"
            st.session_state["last_sources"] = []
            st.rerun()
       
        if st.button("🧹 Clear Current", use_container_width=True):
            CHAT_DB.clear_session(effective_user_id(), st.session_state.get("session_id", "default"))
            st.session_state["last_sources"] = []
            st.rerun()

def render_source_preview(sources: List[Dict[str, Any]]):
    """Render source document preview in sticky panel"""
    if not sources:
        st.markdown("### 📚 Retrieved Evidence")
        st.info("No sources yet. Ask a question to see relevant documents.")
        return
   
    # Evidence header (sticky)
    st.markdown('<div class="evidence-header">', unsafe_allow_html=True)
    st.markdown("### 📚 Retrieved Evidence")
    st.caption(f"📊 {len(sources)} sources found")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Remove duplicates
    seen_ids = set()
    unique_sources = []
    for src in sources:
        if src['id'] not in seen_ids:
            seen_ids.add(src['id'])
            unique_sources.append(src)
    
    # Render each source
    for source in unique_sources:
        source_id = source['id']
        source_num = source['index']
        
        # Source card with anchor
        st.markdown(f'<div id="{source_id}" class="source-card">', unsafe_allow_html=True)
        
        st.markdown(f"#### 📄 Source {source_num}: {source['title']}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if source.get('section'):
                st.caption(f"📋 **Section:** {source['section']}")
            if source.get('source'):
                st.caption(f"🔗 **File:** {source['source']}")
        
        with col2:
            if source.get('year'):
                st.caption(f"📅 **Year:** {source['year']}")
            if source.get('page'):
                st.caption(f"📖 **Page:** {source['page']}")
        
        # Content preview
        content_display = source.get('content', source.get('preview', ''))[:600]
        st.markdown(f'<div class="source-preview">{content_display}...</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")

def render_reasoning_steps(steps: List[Dict[str, Any]]):
    """Render multi-hop reasoning steps"""
    st.markdown("### 🧠 Reasoning Process")
    
    for step in steps:
        st.markdown(f'<div class="reasoning-step">', unsafe_allow_html=True)
        st.markdown(f"**Step {step['step']}: {step['query']}**")
        st.markdown(f"{step['answer']}")
        st.caption(f"📚 Retrieved {step['sources']} sources")
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# MAIN APP
# =========================================================

def main():
    st.set_page_config(
        page_title="WYDOT  Chatbot",
        page_icon="🛣️",
        layout="wide"
    )
    
    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Inject scroll script
    st.markdown(SCROLL_SCRIPT, unsafe_allow_html=True)
   
    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = f"session_{int(time.time())}"
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "selected_index" not in st.session_state:
        st.session_state["selected_index"] = NEO4J_INDEX_DEFAULT
    if "last_sources" not in st.session_state:
        st.session_state["last_sources"] = []
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = "mistral"
    if "use_gemini_embeddings" not in st.session_state:
        st.session_state["use_gemini_embeddings"] = False
    if "multihop_enabled" not in st.session_state:
        st.session_state["multihop_enabled"] = False
    if "reasoning_steps" not in st.session_state:
        st.session_state["reasoning_steps"] = []
   
    # Render sidebar
    render_sidebar()
   
    # Main content
    model_name = st.session_state.get('selected_model', 'mistral').title()
    model_badge_class = "model-mistral" if model_name == "Mistral" else "model-gemini"
    multihop_status = "ON" if st.session_state.get('multihop_enabled') else "OFF"
    
    # Custom Header
    header_html = f"""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1px; padding-bottom: 1px; border-bottom: 2px solid #f0f2f6;">
        <h1 style="margin: 0; font-size: 2.2rem; padding: 0;">🛣️ WYDOT Assistant</h1>
        <div style="font-size: 0.9rem; color: #555; text-align: right;">
            <span style="margin-left: 10px;">📚 Docs: 1500+</span>
            <span class="model-badge {model_badge_class}">{model_name}</span>
            <span style="font-weight: 500; font-size: 0.85em; background: #eee; padding: 2px 6px; border-radius: 4px; margin-left: 8px;">🧠 Multi-hop: {multihop_status}</span>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
   
    # Two-column layout
    col_chat, col_sources = st.columns([2, 1])
   
    with col_chat:
        st.markdown("***💬 Chat")
        
        # Chat messages in scrollable container
        st.markdown('<div class="chat-messages-container">', unsafe_allow_html=True)
        
        # Load conversation history
        history_msgs = CHAT_DB.get_recent(
            effective_user_id(),
            st.session_state["session_id"],
            limit=MAX_HISTORY_MSGS
        )
       
        # Display chat history
        for msg in history_msgs:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant" and "SOURCE_" in msg["content"]:
                    enhanced_msg = enhance_citations_in_response(
                        msg["content"],
                        st.session_state.get("last_sources", [])
                    )
                    st.markdown(enhanced_msg, unsafe_allow_html=True)
                else:
                    st.markdown(msg["content"])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show reasoning steps if available
        if st.session_state.get("reasoning_steps"):
            with st.expander("🧠 View Reasoning Steps", expanded=False):
                render_reasoning_steps(st.session_state["reasoning_steps"])
       
    with col_sources:
        # Fixed evidence panel
        st.markdown('<div class="evidence-panel">', unsafe_allow_html=True)
        render_source_preview(st.session_state.get("last_sources", []))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input at bottom (outside columns for full width)
    if prompt := st.chat_input("Ask about WYDOT specifications, reports, or memos..."):
        # Add user message
        with col_chat:
            with st.chat_message("user"):
                st.markdown(prompt)
       
        CHAT_DB.add_message(effective_user_id(), st.session_state["session_id"], "user", prompt)
       
        # Generate response
        index_name = st.session_state.get("selected_index", NEO4J_INDEX_DEFAULT)
        model_type = st.session_state.get("selected_model", "mistral")
        use_gemini_emb = st.session_state.get("use_gemini_embeddings", False)
        multihop = st.session_state.get("multihop_enabled", False)
        
        with col_chat:
            with st.chat_message("assistant"):
                if multihop:
                    with st.spinner("🧠 Performing multi-hop reasoning..."):
                        response, sources, reasoning_steps = multihop_reasoning(
                            prompt,
                            index_name,
                            model_type,
                            use_gemini_emb
                        )
                        
                        st.session_state["last_sources"] = sources
                        st.session_state["reasoning_steps"] = reasoning_steps
                        
                        st.markdown(
                            '<div class="multihop-indicator">✅ Multi-hop reasoning applied</div>',
                            unsafe_allow_html=True
                        )
                else:
                    with st.spinner("🔍 Searching knowledge graph..."):
                        context, sources = search_graph(
                            prompt,
                            index_name=index_name,
                            use_gemini=use_gemini_emb
                        )
                       
                        st.session_state["last_sources"] = sources
                        st.session_state["reasoning_steps"] = []
                       
                        if not context:
                            response = "I couldn't find relevant information in the knowledge graph. Please try rephrasing your question."
                        else:
                            if model_type == "gemini":
                                response = generate_answer_gemini(prompt, context, history_msgs, enhanced_citations=True)
                            else:
                                response = generate_answer_mistral(prompt, context, history_msgs, enhanced_citations=True)
                
                # Enhance and display
                enhanced_response = enhance_citations_in_response(response, st.session_state["last_sources"])
                st.markdown(enhanced_response, unsafe_allow_html=True)
               
                # Save to history
                CHAT_DB.add_message(
                    effective_user_id(),
                    st.session_state["session_id"],
                    "assistant",
                    response
                )
        
        # Update evidence panel
        with col_sources:
            st.markdown('<div class="evidence-panel">', unsafe_allow_html=True)
            render_source_preview(st.session_state.get("last_sources", []))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Trigger rerun to show updated state
        st.rerun()

if __name__ == "__main__":
    main()