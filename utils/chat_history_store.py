"""
Chat history and auth store abstraction.
- Local: SQLite (default).
- Cloud: set DATABASE_URL for PostgreSQL/Cloud SQL; same interface.
"""
import os
import threading
import hashlib
import hmac
import base64 as _b64
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple

# -----------------------------------------------------------------------------
# Password hashing (PBKDF2)
# -----------------------------------------------------------------------------

def pbkdf2_hash(password: str, iterations: int = 200_000) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64.b64encode(salt).decode()}${_b64.b64encode(dk).decode()}"


def pbkdf2_verify(password: str, stored: str) -> bool:
    try:
        parts = stored.split("$")
        if len(parts) != 4: return False
        algo, iters, salt_b64, dk_b64 = parts
        if algo != "pbkdf2_sha256": return False
        iterations = int(iters)
        salt = _b64.b64decode(salt_b64)
        dk = _b64.b64decode(dk_b64)
        new_dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(new_dk, dk)
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Abstract store (for Cloud SQL later)
# -----------------------------------------------------------------------------

class BaseChatHistoryStore(ABC):
    @abstractmethod
    def authenticate(self, email: str, password: str) -> Tuple[Optional[int], Optional[str]]:
        """Return (user_id, display_name) or (None, error_message)."""
        pass

    @abstractmethod
    def create_user(self, email: str, password: str, display_name: Optional[str] = None) -> Tuple[Optional[int], Optional[str]]:
        """Return (user_id, None) or (None, error_message)."""
        pass

    @abstractmethod
    def add_message(self, user_id: int, session_id: str, role: str, content: str) -> None:
        pass

    @abstractmethod
    def get_recent(self, user_id: int, session_id: str, limit: int = 20) -> List[Dict[str, str]]:
        """Return list of {"role": "user"|"assistant", "content": "..."}."""
        pass


# -----------------------------------------------------------------------------
# SQLite implementation (local / default)
# -----------------------------------------------------------------------------

class SQLiteChatHistoryStore(BaseChatHistoryStore):
    def __init__(self, db_path: str):
        import sqlite3
        self._lock = threading.Lock()
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass
        self._init_tables()

    def _init_tables(self):
        import sqlite3
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
        # Create index; if messages table had old schema (no user_id), recreate it and retry
        try:
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_session ON messages(user_id, session_id);")
            self._conn.commit()
        except sqlite3.OperationalError as e:
            if "no such column: user_id" in str(e) or "user_id" in str(e):
                self._conn.execute("DROP TABLE IF EXISTS messages;")
                self._conn.execute("""
                    CREATE TABLE messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        session_id TEXT NOT NULL,
                        role TEXT CHECK(role IN ('user','assistant')) NOT NULL,
                        content TEXT NOT NULL,
                        ts REAL NOT NULL DEFAULT (strftime('%s','now')),
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    )
                """)
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_session ON messages(user_id, session_id);")
                self._conn.commit()
            else:
                raise

    def authenticate(self, email: str, password: str) -> Tuple[Optional[int], Optional[str]]:
        email = (email or "").strip().lower()
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, password_hash, display_name FROM users WHERE email=?", (email,)
            )
            row = cur.fetchone()
            if not row:
                return None, "User not found"
            uid, pw_hash, name = row
            if not pbkdf2_verify(password, pw_hash):
                return None, "Invalid password"
            return int(uid), (name or email)

    def create_user(self, email: str, password: str, display_name: Optional[str] = None) -> Tuple[Optional[int], Optional[str]]:
        email = (email or "").strip().lower()
        with self._lock:
            try:
                pw_hash = pbkdf2_hash(password)
                self._conn.execute(
                    "INSERT INTO users (email, password_hash, display_name) VALUES (?,?,?)",
                    (email, pw_hash, display_name),
                )
                self._conn.commit()
                cur = self._conn.execute("SELECT id FROM users WHERE email=?", (email,))
                return int(cur.fetchone()[0]), None
            except Exception as e:
                return None, str(e)

    def add_message(self, user_id: int, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (user_id, session_id, role, content) VALUES (?, ?, ?, ?)",
                (user_id, session_id, role, content),
            )
            self._conn.commit()

    def get_recent(self, user_id: int, session_id: str, limit: int = 20) -> List[Dict[str, str]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT role, content FROM messages WHERE user_id=? AND session_id=? ORDER BY id DESC LIMIT ?",
                (user_id, session_id, limit),
            )
            rows = cur.fetchall()
        rows.reverse()
        return [{"role": r[0], "content": r[1]} for r in rows]


# -----------------------------------------------------------------------------
# Factory: local SQLite or Cloud SQL (when DATABASE_URL is set)
# -----------------------------------------------------------------------------

def get_chat_history_store():
    """Return store: SQLite for local. Set DATABASE_URL (postgres) for Cloud SQL (add PGChatHistoryStore later)."""
    db_url = os.getenv("DATABASE_URL")
    if db_url and db_url.startswith("postgres"):
        try:
            from utils.chat_history_store_pg import PGChatHistoryStore  # noqa: F401
            return PGChatHistoryStore(db_url)
        except ImportError:
            pass
    db_path = os.getenv("CHAT_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "chat_history.sqlite3"))
    return SQLiteChatHistoryStore(db_path)
