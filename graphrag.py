#!/usr/bin/env python3
"""
WYDOT GraphRAG Chatbot - Production Version
Works with existing Neo4j database (1,500+ documents already ingested)
Features: Chat history, user auth, sessions, file uploads, multi-index support
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
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

# GraphRAG / LangChain imports
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Google Vertex AI (Optional stub for future use)
try:
    from google import genai
    from google.genai import types as gtypes
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False

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
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-large-latest")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# Chat history database
CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", os.path.join(tempfile.gettempdir(), "wydot_chat_history.sqlite3"))

# App settings
MAX_HISTORY_MSGS = 20
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "8"))

# =========================================================
# AUTHENTICATION & USER MANAGEMENT
# =========================================================

def _pbkdf2_hash(password: str, iterations: int = 200_000) -> str:
    """Hash password using PBKDF2"""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    # Store salt and hash as base64 strings
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
        
        # Ensure directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
       
        # Enable WAL mode for better concurrency in SQLite
        try:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
       
        # Create tables
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
        """Create new user account"""
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
        """Authenticate user"""
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
        """Get user display name"""
        with self._lock:
            cur = self._conn.execute("SELECT display_name, email FROM users WHERE id=?", (user_id,))
            row = cur.fetchone()
            return row[0] or row[1] if row else None
   
    def add_message(self, user_id: int, session_id: str, role: str, content: str):
        """Add message to history"""
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (user_id, session_id, role, content) VALUES (?, ?, ?, ?)",
                (user_id, session_id, role, content)
            )
            self._conn.commit()
   
    def get_recent(self, user_id: int, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent messages"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT role, content, ts FROM messages WHERE user_id=? AND session_id=? ORDER BY id DESC LIMIT ?",
                (user_id, session_id, limit)
            )
            rows = cur.fetchall()
        rows.reverse()
        return [{"role": r[0], "content": r[1], "ts": r[2]} for r in rows]
   
    def list_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        """List all sessions for user"""
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
        """Clear session history"""
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE user_id=? AND session_id=?", (user_id, session_id))
            self._conn.commit()

@st.cache_resource(show_spinner=False)
def get_chat_store(db_path: str):
    """Get cached chat history store"""
    return ChatHistoryStore(db_path)

CHAT_DB = get_chat_store(CHAT_DB_PATH)

def current_user_id() -> int:
    """Get current logged-in user ID"""
    return int(st.session_state.get("user_id", 0))

def effective_user_id() -> int:
    """Get effective user ID (logged-in or anonymous)"""
    uid = current_user_id()
    if uid and uid > 0:
        return uid
    # Anonymous user - create stable ID per browser session
    if "anon_uid" not in st.session_state:
        st.session_state["anon_uid"] = int.from_bytes(os.urandom(8), "big") & ((1 << 63) - 1)
    return int(st.session_state["anon_uid"])

# =========================================================
# NEO4J GRAPHRAG RETRIEVAL
# =========================================================

@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Get HuggingFace embeddings model"""
    # Note: Ensure sentence-transformers is installed in your environment
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_retriever(index_name: str = NEO4J_INDEX_DEFAULT):
    """Get Neo4j vector retriever"""
    try:
        embeddings = get_embeddings()
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
        # In production, log this error instead of showing it to the user immediately
        print(f"Neo4j connection failed: {e}")
        return None

def search_graph(query: str, index_name: str = NEO4J_INDEX_DEFAULT, k: int = None) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Search Neo4j knowledge graph
    Returns: Tuple of (context_text, source_documents)
    """
    if not query or not query.strip():
        return "", []
   
    k = k or RETRIEVAL_K
    retriever = get_retriever(index_name)
   
    if not retriever:
        return "", []
   
    try:
        # Retrieve documents
        source_docs = retriever.invoke(query)[:k]
       
        # Extract content and metadata
        chunks = []
        sources = []
       
        for doc in source_docs:
            content = doc.page_content
            chunks.append(content)
           
            meta = doc.metadata
            sources.append({
                "title": meta.get("title", "Untitled"),
                "source": meta.get("source", "unknown"),
                "year": meta.get("year", ""),
                "section": meta.get("section", ""),
                "author": meta.get("author", ""),
                "page": meta.get("page", 1),
                "preview": content[:300] if content else ""
            })
       
        return "\n\n".join(chunks), sources
       
    except Exception as e:
        st.warning(f"Search error: {e}")
        return "", []

# =========================================================
# LLM GENERATION
# =========================================================

@st.cache_resource(show_spinner=False)
def get_llm():
    """Get Mistral LLM"""
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY is not set in environment variables.")
        
    return ChatMistralAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        mistral_api_key=MISTRAL_API_KEY
    )

def build_prompt_with_history(question: str, context: str, history_msgs: List[Dict[str, Any]]) -> str:
    """Build prompt with conversation history"""
   
    # Format history (last few exchanges)
    history_text = ""
    if history_msgs:
        recent = history_msgs[-10:] 
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
   
    history_section = f"CONVERSATION HISTORY:\n{history_text}" if history_text else ""
   
    # We construct the full prompt string here
    prompt = f"""You are a WYDOT expert assistant. Answer based on the context below.

METADATA GUIDE:
- SOURCE: Filename
- YEAR: Document Year
- AUTHOR: Document Author (if available)
- SECTION: Specific section (e.g., Section 101)

{history_section}

CONTEXT:
{context}

QUESTION: {question}

Instructions:
- Provide accurate, helpful answers based on the context
- Cite specific sections when relevant
- If the answer isn't in the context, say so clearly
- Keep answers concise but comprehensive
"""
    return prompt

def generate_answer(question: str, context: str, history: List[Dict[str, Any]]) -> str:
    """Generate answer using Mistral LLM"""
    try:
        llm = get_llm()
        full_prompt_str = build_prompt_with_history(question, context, history)
       
        # Pass the pre-formatted string as a single variable to avoid brace conflicts in context
        template = ChatPromptTemplate.from_template("{input_prompt}")
        chain = template | llm | StrOutputParser()
       
        response = chain.invoke({"input_prompt": full_prompt_str})
        return response
       
    except Exception as e:
        return f"Error generating response: {e}"

def generate_followup_questions(question: str, answer: str, context: str) -> List[str]:
    """Generate 3 follow-up questions based on the conversation"""
    try:
        llm = get_llm()
        
        followup_prompt = f"""Based on this Q&A about WYDOT documents, suggest 3 brief, natural follow-up questions a user might ask.

Question: {question}
Answer: {answer}

Generate 3 short follow-up questions (max 10 words each) that would help the user learn more. Return ONLY the questions, one per line, without numbering or bullets.

Examples of good follow-up questions:
- What are the specific requirements for this?
- How does this apply to residential projects?
- Are there exceptions to this rule?
"""
        
        template = ChatPromptTemplate.from_template("{input_prompt}")
        chain = template | llm | StrOutputParser()
        
        response = chain.invoke({"input_prompt": followup_prompt})
        
        # Parse questions
        questions = [q.strip().lstrip('-•*123456789.') for q in response.strip().split('\n') if q.strip()]
        return questions[:3]  # Return max 3 questions
        
    except Exception:
        # Return default questions if generation fails
        return [
            "Can you explain this in more detail?",
            "What are the requirements?",
            "Are there any exceptions?"
        ]

# =========================================================
# UI COMPONENTS
# =========================================================

def render_sidebar():
    """Render sidebar with auth and settings"""
    with st.sidebar:
        st.markdown("## 🔐 Account")
       
        if current_user_id() > 0:
            # Logged in
            display_name = CHAT_DB.get_display_name(current_user_id())
            st.success(f"👤 {display_name}")
           
            if st.button("🔓 Logout", use_container_width=True):
                st.session_state["user_id"] = 0
                st.session_state["user_email"] = None
                st.session_state["session_id"] = f"session_{int(time.time())}"
                st.rerun()
        else:
            # Not logged in
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
       
        # Neo4j status
        st.markdown("## 📊 Database Status")
        try:
            # Check connection by attempting to get retriever (lightweight check)
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
       
        # New conversation
        if st.button("➕ New Conversation", use_container_width=True):
            st.session_state["session_id"] = f"session_{int(time.time())}"
            st.rerun()
       
        # Clear current
        if st.button("🧹 Clear Current", use_container_width=True):
            CHAT_DB.clear_session(effective_user_id(), st.session_state.get("session_id", "default"))
            st.rerun()

def render_source_preview(sources: List[Dict[str, Any]]):
    """Render source document preview"""
    if not sources:
        st.info("No sources retrieved yet. Ask a question to see relevant documents.")
        return
   
    st.markdown("### 📚 Retrieved Sources")
   
    for i, source in enumerate(sources, 1):
        with st.expander(f"📄 {i}. {source['title']}", expanded=i<=3):
            col1, col2 = st.columns([2, 1])
           
            with col1:
                if source.get('section'):
                    st.caption(f"📋 Section: {source['section']}")
                if source.get('source'):
                    st.caption(f"🔗 Source: {source['source']}")
           
            with col2:
                if source.get('year'):
                    st.caption(f"📅 Year: {source['year']}")
                if source.get('page'):
                    st.caption(f"📖 Page: {source['page']}")
           
            st.markdown("**Preview:**")
            st.text(source.get('preview', '')[:400] + "...")

# =========================================================
# MAIN APP
# =========================================================

def main():
    st.set_page_config(
        page_title="WYDOT GraphRAG Chatbot",
        page_icon="🛣️",
        layout="wide"
    )
   
    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = f"session_{int(time.time())}"
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "selected_index" not in st.session_state:
        st.session_state["selected_index"] = NEO4J_INDEX_DEFAULT
    if "last_sources" not in st.session_state:
        st.session_state["last_sources"] = []
    if "followup_questions" not in st.session_state:
        st.session_state["followup_questions"] = []
   
    # Render sidebar
    render_sidebar()
   
    # Main content
    st.title("🛣️ WYDOT Knowledge Assistant")
    st.caption("Powered by Neo4j GraphRAG • 1,500+ Documents")
   
    # Two-column layout
    col_chat, col_sources = st.columns([2, 1])
   
    with col_chat:
        st.markdown("### 💬 Chat")
       
        # Load conversation history
        history_msgs = CHAT_DB.get_recent(
            effective_user_id(),
            st.session_state["session_id"],
            limit=MAX_HISTORY_MSGS
        )
       
        # Display chat history
        for idx, msg in enumerate(history_msgs):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # Show follow-up questions after assistant messages
                if msg["role"] == "assistant" and idx == len(history_msgs) - 1:
                    if st.session_state.get("followup_questions"):
                        st.markdown("---")
                        st.markdown("💡 **Want to know more? Try asking:**")
                        for fq in st.session_state["followup_questions"]:
                            st.markdown(f"- *{fq}*")
       
        # Chat input
        if prompt := st.chat_input("Ask about WYDOT specifications, reports, or memos..."):
            # Clear previous follow-up questions
            st.session_state["followup_questions"] = []
            
            # Add user message
            with st.chat_message("user"):
                st.markdown(prompt)
           
            CHAT_DB.add_message(effective_user_id(), st.session_state["session_id"], "user", prompt)
           
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge graph..."):
                    # Search graph
                    index_name = st.session_state.get("selected_index", NEO4J_INDEX_DEFAULT)
                    context, sources = search_graph(prompt, index_name=index_name)
                   
                    # Store sources for preview
                    st.session_state["last_sources"] = sources
                   
                    if not context:
                        response = "I couldn't find relevant information in the knowledge graph. Please try rephrasing your question."
                    else:
                        # Generate answer
                        response = generate_answer(prompt, context, history_msgs)
                   
                    st.markdown(response)
                   
                    # Add to history
                    CHAT_DB.add_message(effective_user_id(), st.session_state["session_id"], "assistant", response)
                    
                    # Generate follow-up questions
                    if context:
                        with st.spinner("Generating follow-up questions..."):
                            followups = generate_followup_questions(prompt, response, context)
                            st.session_state["followup_questions"] = followups
                            
                            if followups:
                                st.markdown("---")
                                st.markdown("💡 **Want to know more? Try asking:**")
                                for fq in followups:
                                    st.markdown(f"- *{fq}*")
            
            # Auto-scroll to bottom by rerunning
            st.rerun()
   
    with col_sources:
        render_source_preview(st.session_state.get("last_sources", []))
    
    # Custom CSS for auto-scroll
    st.markdown("""
        <style>
        /* Auto-scroll chat to bottom */
        .main .block-container {
            padding-bottom: 5rem;
        }
        
        /* Style follow-up questions */
        .stMarkdown em {
            color: #1f77b4;
            cursor: default;
        }
        
        /* Improve chat input visibility */
        .stChatFloatingInputContainer {
            bottom: 20px;
            background-color: white;
            border-top: 1px solid #e6e6e6;
        }
        </style>
        
        <script>
        // Auto-scroll to bottom when new message appears
        window.addEventListener('load', function() {
            const chatContainer = window.parent.document.querySelector('.main');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        });
        </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()