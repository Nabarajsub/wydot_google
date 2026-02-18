"""
Chat history and auth store abstraction.
- Local: SQLite (default).
- Cloud: set DATABASE_URL for PostgreSQL/Cloud SQL; same interface.
"""
import os
import threading
import time
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


def _send_verification_email(email: str, code: str) -> None:
    """Send 6-digit verification code. Uses SMTP if configured, else logs to console and file."""
    # ALWAYS log to file for dev/testing ease
    try:
        log_path = "verification_codes.txt"
        if os.getenv("K_SERVICE"):
            log_path = "/tmp/verification_codes.txt"
        with open(log_path, "a") as f:
            import datetime
            f.write(f"[{datetime.datetime.now()}] To: {email} | Code: {code}\n")
    except Exception as e:
        print(f"Failed to write to verification_codes.txt: {e}")

    smtp_host = os.getenv("SMTP_HOST")
    if smtp_host:
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.utils import formataddr
            msg = MIMEText(f"Your WYDOT Assistant verification code is: {code}\n\nIt expires in 15 minutes.")
            msg["Subject"] = "WYDOT Assistant – verification code"
            msg["From"] = formataddr(("WYDOT Assistant", os.getenv("EMAIL_FROM", "noreply@wydot.local")))
            msg["To"] = email
            port = int(os.getenv("SMTP_PORT", "587"))
            user = os.getenv("SMTP_USER")
            password = os.getenv("SMTP_PASSWORD")
            with smtplib.SMTP(smtp_host, port) as s:
                if port == 587:
                    s.starttls()
                if user and password:
                    s.login(user, password)
                s.sendmail(msg["From"], [email], msg.as_string())
        except Exception as e:
            print(f"[WYDOT] Verification email send failed: {e}. Code for {email}: {code}")
    else:
        print(f"[WYDOT] Verification code for {email}: {code} (set SMTP_HOST to send real email)")


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
                created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
                verified INTEGER NOT NULL DEFAULT 1,
                verification_code TEXT,
                verification_expires_at REAL
            )
        """)
        for col in ("verified", "verification_code", "verification_expires_at"):
            try:
                self._conn.execute(f"ALTER TABLE users ADD COLUMN {col} " + (
                    "INTEGER NOT NULL DEFAULT 1" if col == "verified" else
                    "TEXT" if col == "verification_code" else "REAL"
                ))
            except sqlite3.OperationalError:
                pass
        self._conn.commit()
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
        # Create index; if messages table had old schema (no user_id or no sources), recreate it and retry
        try:
            # Check if sources column exists
            self._conn.execute("SELECT sources FROM messages LIMIT 1;")
        except sqlite3.OperationalError:
            # Add sources column if missing (migration)
            try:
                self._conn.execute("ALTER TABLE messages ADD COLUMN sources TEXT;")
                self._conn.commit()
            except sqlite3.OperationalError:
                # If alter fails or other issue, fallback to recreate (drastic, but consistent with previous logic)
                pass 

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
                        sources TEXT,
                        ts REAL NOT NULL DEFAULT (strftime('%s','now')),
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    )
                """)
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_session ON messages(user_id, session_id);")
                self._conn.commit()
            else:
                raise
        
        # Feedback table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                for_id TEXT NOT NULL,
                thread_id TEXT,
                user_id INTEGER,
                value INTEGER NOT NULL,
                comment TEXT,
                ts REAL NOT NULL DEFAULT (strftime('%s','now'))
            )
        """)
        self._conn.commit()

        # Seed guest user if not exists (same as CloudSQLChatHistoryStore)
        try:
            cur = self._conn.execute("SELECT id FROM users WHERE email='guest@app.local'")
            if not cur.fetchone():
                self._conn.execute(
                    "INSERT INTO users (email, password_hash, display_name, verified) VALUES (?, ?, ?, 1)",
                    ("guest@app.local", pbkdf2_hash("guest"), "Guest User"),
                )
                self._conn.commit()
                print("✅ Guest user seeded in SQLite.")
        except Exception as e:
            print(f"⚠️ Guest user seeding failed: {e}")

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

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Return user dict {id, email, name, created_at} or None."""
        email = (email or "").strip().lower()
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, email, display_name, created_at FROM users WHERE email=?", (email,)
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "id": int(row[0]),
                "email": row[1],
                "name": row[2] or row[1],
                "created_at": row[3]
            }

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Return user dict {id, email, name, created_at} by numeric ID."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, email, display_name, created_at FROM users WHERE id=?", (user_id,)
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "id": int(row[0]),
                "email": row[1],
                "name": row[2] or row[1],
                "created_at": row[3]
            }

    def is_verified(self, user_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("SELECT verified FROM users WHERE id=?", (user_id,))
            row = cur.fetchone()
            return bool(row and row[0]) if row else True

    def set_verified(self, user_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE users SET verified=1, verification_code=NULL, verification_expires_at=NULL WHERE id=?",
                (user_id,),
            )
            self._conn.commit()

    def set_verification_code(self, user_id: int, code: str, expires_minutes: int = 15) -> None:
        import time
        expires = time.time() + expires_minutes * 60
        with self._lock:
            self._conn.execute(
                "UPDATE users SET verification_code=?, verification_expires_at=? WHERE id=?",
                (code, expires, user_id),
            )
            self._conn.commit()

    def check_verification_code(self, email: str, code: str) -> Optional[int]:
        """Return user_id if code matches and not expired, else None."""
        import time
        email = (email or "").strip().lower()
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, verification_code, verification_expires_at FROM users WHERE email=?",
                (email,),
            )
            row = cur.fetchone()
            if not row:
                return None
            uid, stored_code, expires = row
            if not stored_code or stored_code != code.strip():
                return None
            if expires and time.time() > expires:
                return None
            return int(uid)

    def create_user(self, email: str, password: str, display_name: Optional[str] = None) -> Tuple[Optional[int], Optional[str]]:
        email = (email or "").strip().lower()
        with self._lock:
            try:
                pw_hash = pbkdf2_hash(password)
                # New users start unverified (verified=0); verification code set after insert
                self._conn.execute(
                    "INSERT INTO users (email, password_hash, display_name, verified) VALUES (?,?,?,0)",
                    (email, pw_hash, display_name),
                )
                self._conn.commit()
                cur = self._conn.execute("SELECT id FROM users WHERE email=?", (email,))
                uid = int(cur.fetchone()[0])
            except Exception as e:
                return None, str(e)
        # Generate and store 6-digit code, send email (outside lock)
        import random
        code = "".join(str(random.randint(0, 9)) for _ in range(6))
        self.set_verification_code(uid, code, expires_minutes=15)
        _send_verification_email(email, code)
        return uid, None

    def add_message(self, user_id: int, session_id: str, role: str, content: str, sources: List[Dict] = None) -> None:
        import json
        sources_json = json.dumps(sources) if sources else None
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (user_id, session_id, role, content, sources) VALUES (?, ?, ?, ?, ?)",
                (user_id, session_id, role, content, sources_json),
            )
            self._conn.commit()

    def get_recent(self, user_id: int, session_id: str, limit: int = 20) -> List[Dict]:
        import json
        with self._lock:
            cur = self._conn.execute(
                "SELECT role, content, sources, ts FROM messages WHERE user_id=? AND session_id=? ORDER BY id DESC LIMIT ?",
                (user_id, session_id, limit),
            )
            rows = cur.fetchall()
        rows.reverse()
        result = []
        for r in rows:
            msg = {"role": r[0], "content": r[1], "ts": r[3]}
            if r[2]:
                try:
                    msg["sources"] = json.loads(r[2])
                except:
                    pass
            result.append(msg)
        return result

    def get_user_sessions(self, user_id: int, search_term: str = None) -> List[Dict]:
        """Return list of {id, name, createdAt} for a user's history."""
        with self._lock:
            # Group by session_id to find unique threads
            if search_term:
                # Search within message content
                sql = """
                SELECT session_id, MAX(ts)
                FROM messages
                WHERE user_id = ? AND content LIKE ?
                GROUP BY session_id
                ORDER BY MAX(ts) DESC
                """
                params = (user_id, f"%{search_term}%")
            else:
                sql = """
                SELECT session_id, MAX(ts)
                FROM messages
                WHERE user_id = ?
                GROUP BY session_id
                ORDER BY MAX(ts) DESC
                """
                params = (user_id,)
            
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
            
            results = []
            for sid, ts in rows:
                # Get first user message as name
                cur2 = self._conn.execute("SELECT content FROM messages WHERE session_id=? AND role='user' ORDER BY id ASC LIMIT 1", (sid,))
                row2 = cur2.fetchone()
                # Truncate content for title
                name = (row2[0][:40] + "...") if row2 else f"Chat {sid}"
                # Ensure name is not too long or weird
                results.append({"id": sid, "name": name, "createdAt": ts})
            return results

    def get_thread_messages_all(self, user_id: int, session_id: str) -> List[Dict]:
        """Get all messages for a thread (for DataLayer)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT role, content, ts, id FROM messages WHERE user_id=? AND session_id=? ORDER BY id ASC",
                (user_id, session_id),
            )
            rows = cur.fetchall()
        
        # Convert to Chainlit Step dict format if possible, or just raw for now
        return [{"role": r[0], "content": r[1], "createdAt": r[2], "id": str(r[3])} for r in rows]

    def upsert_feedback(self, feedback) -> None:
        """Stored feedback from Chainlit (value 0 or 1).
        Accepts both Chainlit Feedback dataclass and plain dict."""
        import logging
        logger = logging.getLogger("chainlit")
        
        # Convert dataclass to dict if needed
        if hasattr(feedback, '__dataclass_fields__'):
            import dataclasses
            fb = dataclasses.asdict(feedback)
        elif hasattr(feedback, 'forId'):
            # Object with attributes
            fb = {
                "forId": getattr(feedback, 'forId', None),
                "threadId": getattr(feedback, 'threadId', None),
                "value": getattr(feedback, 'value', 1),
                "comment": getattr(feedback, 'comment', ""),
                "id": getattr(feedback, 'id', None),
            }
        else:
            fb = feedback  # Already a dict
        
        logger.info(f"[FEEDBACK] Saving feedback: forId={fb.get('forId')}, value={fb.get('value')}, comment={fb.get('comment')}, threadId={fb.get('threadId')}")
        
        with self._lock:
            # Check if exists
            cur = self._conn.execute("SELECT id FROM feedback WHERE for_id=?", (fb.get("forId"),))
            row = cur.fetchone()
            
            val = fb.get("value", 1)  # 1=up, 0=down
            comment = fb.get("comment", "") or ""
            
            if row:
                self._conn.execute(
                    "UPDATE feedback SET value=?, comment=?, ts=? WHERE id=?",
                    (val, comment, time.time(), row[0])
                )
            else:
                self._conn.execute(
                    "INSERT INTO feedback (for_id, thread_id, value, comment, ts) VALUES (?, ?, ?, ?, ?)",
                    (
                        fb.get("forId"),
                        fb.get("threadId"),
                        val,
                        comment,
                        time.time()
                    )
                )
            self._conn.commit()
            logger.info(f"[FEEDBACK] Feedback saved successfully")

    def get_feedback_stats(self) -> Dict:
        """Return basic stats: total, ups, downs."""
        with self._lock:
            cur = self._conn.execute("SELECT value FROM feedback")
            rows = cur.fetchall()
            total = len(rows)
            thumbs_up = sum(1 for r in rows if r[0] == 1)
            thumbs_down = total - thumbs_up
            return {
                "total": total,
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "up_rate": (thumbs_up / total * 100) if total > 0 else 0
            }

    def get_recent_feedback(self, limit: int = 50) -> List[Dict]:
        """Return detailed recent feedback."""
        with self._lock:
            # Look up user email via messages table subquery (since sessions table doesn't exist)
            cur = self._conn.execute("""
                SELECT f.value, f.comment, f.thread_id, f.ts,
                       COALESCE(u.email, 'Anonymous') as user_email,
                       (SELECT content FROM messages m2 
                        WHERE m2.session_id = f.thread_id AND m2.role = 'user' AND m2.ts <= m_asst.ts 
                        ORDER BY m2.ts DESC LIMIT 1) as question,
                       m_asst.sources as sources
                FROM feedback f
                LEFT JOIN messages m_asst ON (f.for_id = CAST(m_asst.id AS TEXT) OR f.for_id = m_asst.id)
                LEFT JOIN users u ON m_asst.user_id = u.id
                ORDER BY f.ts DESC LIMIT ?
            """, (limit,))
            rows = cur.fetchall()
            return [
                {
                    "value": r[0],
                    "comment": r[1],
                    "thread_id": r[2],
                    "ts": r[3],
                    "user": r[4],
                    "question": r[5],
                    "sources": r[6]
                }
                for r in rows
            ]

    def get_session_by_id(self, session_id: str) -> Optional[Dict]:
        """Get thread details by session_id (thread_id)."""
        with self._lock:
            # Check if session exists (by finding a message)
            cur = self._conn.execute(
                "SELECT user_id, ts FROM messages WHERE session_id=? ORDER BY id DESC LIMIT 1",
                (session_id,)
            )
            row = cur.fetchone()
            if not row:
                return None
            user_id, ts = row
            
            # Get name
            cur2 = self._conn.execute("SELECT content FROM messages WHERE session_id=? AND role='user' ORDER BY id ASC LIMIT 1", (session_id,))
            row2 = cur2.fetchone()
            name = (row2[0][:40] + "...") if row2 else f"Chat {session_id}"
            
            return {"id": session_id, "name": name, "createdAt": ts, "userId": str(user_id)}


# -----------------------------------------------------------------------------
# Factory: local SQLite or Cloud SQL (when DATABASE_URL is set)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Cloud SQL (PostgreSQL) implementation
# -----------------------------------------------------------------------------

class CloudSQLChatHistoryStore(BaseChatHistoryStore):
    def __init__(self, database_url: str):
        self.db_url = database_url
        self._tables_initialized = False
        self._initializing = False
        self._db_created = False
        # Parse connection params once for reliable connections
        self._conn_params = self._parse_db_url(database_url)
        # Log masked URL for debugging
        masked = database_url
        try:
            from urllib.parse import urlparse
            p = urlparse(database_url.replace("postgresql+psycopg2://", "postgresql://"))
            masked = f"postgresql://{p.username}:****@{p.hostname or '(socket)'}/{p.path.lstrip('/')}?{p.query}"
        except Exception:
            pass
        print(f"🔗 CloudSQLChatHistoryStore initialized (lazy table init enabled)", flush=True)
        print(f"   DB URL (masked): {masked}", flush=True)
        print(f"   Parsed conn params: dbname={self._conn_params.get('dbname')}, "
              f"user={self._conn_params.get('user')}, "
              f"host={self._conn_params.get('host', '(default)')}", flush=True)

    @staticmethod
    def _parse_db_url(url: str) -> dict:
        """Parse DATABASE_URL into explicit psycopg2 connection params.
        This avoids issues with URL-encoding (%23 for #) that psycopg2's
        URI parser may not handle correctly."""
        from urllib.parse import urlparse, unquote, parse_qs
        clean = url.replace("postgresql+psycopg2://", "postgresql://")
        parsed = urlparse(clean)
        params = {}
        params["user"] = unquote(parsed.username) if parsed.username else "postgres"
        params["password"] = unquote(parsed.password) if parsed.password else ""
        params["dbname"] = parsed.path.lstrip("/") or "postgres"
        # Check for Unix socket host in query params (Cloud SQL proxy)
        qs = parse_qs(parsed.query)
        if "host" in qs:
            params["host"] = qs["host"][0]
        elif parsed.hostname:
            params["host"] = parsed.hostname
            if parsed.port:
                params["port"] = str(parsed.port)
        return params

    def _make_conn(self, dbname_override: str = None):
        """Create a psycopg2 connection using parsed params.
        Optionally override the database name (for admin ops like CREATE DATABASE)."""
        import psycopg2
        params = dict(self._conn_params)
        if dbname_override:
            params["dbname"] = dbname_override
        print(f"   🔌 Connecting to DB: dbname={params['dbname']}, "
              f"user={params['user']}, host={params.get('host', '(default)')}", flush=True)
        conn = psycopg2.connect(**params)
        print(f"   ✅ Connected! server_version={conn.server_version}", flush=True)
        return conn

    def _ensure_database_exists(self):
        """Auto-create the target database if it doesn't exist."""
        if self._db_created:
            return
        import traceback
        target_db = self._conn_params.get("dbname", "postgres")
        if not target_db or target_db == "postgres":
            self._db_created = True
            return
        try:
            print(f"🔍 Checking if database '{target_db}' exists...", flush=True)
            admin_conn = self._make_conn(dbname_override="postgres")
            admin_conn.autocommit = True  # CREATE DATABASE cannot run inside a transaction
            cur = admin_conn.cursor()
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
            if not cur.fetchone():
                print(f"📦 Creating database '{target_db}'...", flush=True)
                cur.execute(f'CREATE DATABASE "{target_db}"')
                print(f"✅ Database '{target_db}' created.", flush=True)
            else:
                print(f"✅ Database '{target_db}' already exists.", flush=True)
            cur.close()
            admin_conn.close()
            self._db_created = True
        except Exception as e:
            print(f"⚠️ Database auto-create check failed: {e}", flush=True)
            traceback.print_exc()
            self._db_created = True  # Don't retry endlessly

    def _get_conn(self):
        """Get a connection, lazily initializing tables on first call."""
        import traceback

        # Ensure the database itself exists before connecting
        self._ensure_database_exists()

        # Lazy initialization check
        if not self._tables_initialized and not self._initializing:
            self._initializing = True
            try:
                print("🛠️ Lazily initializing database tables...", flush=True)
                self._init_tables()
                self._tables_initialized = True
                print("✅ Database tables initialized successfully.", flush=True)
            except Exception as e:
                print(f"⚠️ Lazy Table Init Failed: {e}", flush=True)
                traceback.print_exc()
                # We don't set _tables_initialized=True so it retries on next call
            finally:
                self._initializing = False

        try:
            return self._make_conn()
        except Exception as e:
            print(f"❌ Db connection failed: {e}", flush=True)
            traceback.print_exc()
            raise e

    def _init_tables(self):
        # Create tables and seed guest user
        print("📋 [DB] _init_tables() starting...", flush=True)
        conn = self._make_conn()  # Direct connection (bypasses _get_conn to avoid recursion)
        try:
            with conn.cursor() as cur:
                print("   Creating users table...", flush=True)
                # Users
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        display_name TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        verified INTEGER DEFAULT 1,
                        verification_code TEXT,
                        verification_expires_at TIMESTAMP
                    )
                """)
                # Messages
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES users(id),
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        sources TEXT,
                        ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_session ON messages(user_id, session_id);")
                # Feedback
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id SERIAL PRIMARY KEY,
                        for_id TEXT NOT NULL,
                        thread_id TEXT,
                        user_id INTEGER,
                        value INTEGER NOT NULL,
                        comment TEXT,
                        ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # Seed guest user if it doesn't exist or ensure it is verified
                print("   Seeding/updating guest user...", flush=True)
                cur.execute("SELECT id FROM users WHERE email='guest@app.local'")
                row = cur.fetchone()
                guest_hash = pbkdf2_hash("guest")
                if not row:
                    cur.execute(
                        "INSERT INTO users (email, password_hash, display_name, verified) VALUES (%s, %s, %s, 1)",
                        ("guest@app.local", guest_hash, "Guest User")
                    )
                    print("   ✅ Guest user created (verified=1)", flush=True)
                else:
                    # Ensure it's verified and has the correct password if it exists
                    cur.execute(
                        "UPDATE users SET verified = 1, password_hash = %s WHERE email = 'guest@app.local'",
                        (guest_hash,)
                    )
                    print(f"   ✅ Guest user updated (id={row[0]}, verified=1)", flush=True)
                # Verify guest user was created/updated correctly
                cur.execute("SELECT id, email, verified, length(password_hash) FROM users WHERE email='guest@app.local'")
                check = cur.fetchone()
                print(f"   Guest user check: id={check[0]}, email={check[1]}, verified={check[2]}, hash_len={check[3]}", flush=True)
            conn.commit()
            print("📋 [DB] _init_tables() completed successfully!", flush=True)
        except Exception as e:
            print(f"❌ [DB] _init_tables() FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()
            raise
        finally:
            conn.close()

    def authenticate(self, email: str, password: str) -> Tuple[Optional[int], Optional[str]]:
        import traceback
        print(f"🔐 [DB] authenticate() called for: {email}", flush=True)
        try:
            conn = self._get_conn()
        except Exception as e:
            print(f"❌ [DB] authenticate() connection failed: {e}", flush=True)
            traceback.print_exc()
            return None, f"Database connection error: {e}"
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id, password_hash, display_name, verified FROM users WHERE email=%s", (email,))
                row = cur.fetchone()
                if not row:
                    # List all users for debugging
                    cur.execute("SELECT id, email, verified FROM users ORDER BY id")
                    all_users = cur.fetchall()
                    print(f"🔍 [DB] User not found: {email}. Existing users: {all_users}", flush=True)
                    return None, "User not found"
                uid, pw_hash, name, verified = row
                print(f"🔍 [DB] Found user: id={uid}, name={name}, verified={verified}, hash_len={len(pw_hash) if pw_hash else 0}", flush=True)
                if not pbkdf2_verify(password, pw_hash):
                    print(f"🔍 [DB] Password verify FAILED for {email}. Password length: {len(password)}", flush=True)
                    return None, "Invalid password"
                print(f"✅ [DB] Auth success: {email} (UID: {uid}, verified={verified})", flush=True)
                return int(uid), (name or email)
        except Exception as e:
            print(f"❌ [DB] authenticate() query failed: {e}", flush=True)
            traceback.print_exc()
            return None, f"Database error: {e}"
        finally:
            conn.close()

    def create_user(self, email: str, password: str, display_name: Optional[str] = None) -> Tuple[Optional[int], Optional[str]]:
        import psycopg2
        import traceback
        print(f"📝 [DB] create_user() called: email={email}, display_name={display_name}", flush=True)
        try:
            conn = self._get_conn()
        except Exception as e:
            print(f"❌ [DB] create_user() connection failed: {e}", flush=True)
            traceback.print_exc()
            return None, f"Database connection error: {e}"
        try:
            with conn.cursor() as cur:
                pw_hash = pbkdf2_hash(password)
                try:
                    cur.execute(
                        "INSERT INTO users (email, password_hash, display_name) VALUES (%s, %s, %s) RETURNING id",
                        (email, pw_hash, display_name)
                    )
                    uid = cur.fetchone()[0]
                    conn.commit()
                    print(f"✅ [DB] User created: {email} (UID: {uid})", flush=True)
                    return uid, None
                except psycopg2.IntegrityError:
                    conn.rollback()
                    print(f"⚠️ [DB] User already exists: {email}", flush=True)
                    return None, "Email already exists"
        except Exception as e:
            print(f"❌ [DB] create_user() failed: {e}", flush=True)
            traceback.print_exc()
            return None, f"Database error: {e}"
        finally:
            conn.close()

    def get_recent(self, user_id: int, session_id: str, limit: int = 20) -> List[Dict]:
        import json
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT role, content, sources, extract(epoch from ts) as ts FROM messages WHERE user_id=%s AND session_id=%s ORDER BY id DESC LIMIT %s",
                    (user_id, session_id, limit)
                )
                rows = cur.fetchall()
            
            # Rows are tuples (role, content, sources, ts)
            # Reverse to get chronological order
            rows.reverse()
            result = []
            for r in rows:
                msg = {"role": r[0], "content": r[1], "ts": r[3]}
                if r[2]:
                    try:
                        msg["sources"] = json.loads(r[2])
                    except:
                        pass
                result.append(msg)
            return result
        finally:
            conn.close()

    def add_message(self, user_id: int, session_id: str, role: str, content: str, sources: str = None) -> None:
        conn = self._get_conn()
        # Ensure sources is a valid JSON string or None
        if sources is not None and not isinstance(sources, str):
             import json
             sources = json.dumps(sources)

        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO messages (user_id, session_id, role, content, sources) VALUES (%s, %s, %s, %s, %s)",
                    (user_id, session_id, role, content, sources)
                )
            conn.commit()
        finally:
            conn.close()

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id, email, display_name, extract(epoch from created_at) as created_at FROM users WHERE email=%s", (email,))
                row = cur.fetchone()
                if not row: return None
                return {"id": int(row[0]), "email": row[1], "name": row[2] or row[1], "created_at": row[3]}
        finally:
            conn.close()

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id, email, display_name, extract(epoch from created_at) as created_at FROM users WHERE id=%s", (user_id,))
                row = cur.fetchone()
                if not row: return None
                return {"id": int(row[0]), "email": row[1], "name": row[2] or row[1], "created_at": row[3]}
        finally:
            conn.close()

    def get_user_sessions(self, user_id: int, search_term: str = None) -> List[Dict]:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                if search_term:
                    cur.execute("""
                        SELECT session_id, MAX(extract(epoch from ts)) as last_ts
                        FROM messages
                        WHERE user_id = %s AND content LIKE %s
                        GROUP BY session_id
                        ORDER BY last_ts DESC
                    """, (user_id, f"%{search_term}%"))
                else:
                    cur.execute("""
                        SELECT session_id, MAX(extract(epoch from ts)) as last_ts
                        FROM messages
                        WHERE user_id = %s
                        GROUP BY session_id
                        ORDER BY last_ts DESC
                    """, (user_id,))
                rows = cur.fetchall()
            
            sessions = []
            for r in rows:
                sid = r[0]
                ts = r[1]
                # Get first message for name
                with conn.cursor() as cur2:
                    cur2.execute("SELECT content FROM messages WHERE session_id=%s AND role='user' ORDER BY id ASC LIMIT 1", (sid,))
                    first_msg = cur2.fetchone()
                    name = (first_msg[0][:50] if first_msg else "New Chat")
                sessions.append({"id": sid, "name": name, "createdAt": ts})
            return sessions
        finally:
            conn.close()
            
    def get_session_by_id(self, session_id: str) -> Optional[Dict]:
        # Need userId for Chainlit logic
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id, MAX(extract(epoch from ts)) as ts FROM messages WHERE session_id=%s GROUP BY user_id", (session_id,))
                row = cur.fetchone()
                if not row: return None
                return {"id": session_id, "userId": str(row[0]), "createdAt": row[1], "name": "Chat"}
        finally:
            conn.close()

    def upsert_feedback(self, feedback: Dict) -> None:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO feedback (for_id, thread_id, user_id, value, comment)
                    VALUES (%s, %s, %s, %s, %s)
                """, (feedback.get("forId"), feedback.get("threadId"), feedback.get("userId"), int(feedback.get("value", 0)), feedback.get("comment")))
            conn.commit()
        finally:
            conn.close()

    def get_feedback_stats(self) -> Dict:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT value FROM feedback")
                rows = cur.fetchall()
                total = len(rows)
                thumbs_up = sum(1 for r in rows if r[0] == 1)
                thumbs_down = total - thumbs_up
                return {
                    "total": total,
                    "thumbs_up": thumbs_up,
                    "thumbs_down": thumbs_down,
                    "up_rate": (thumbs_up / total * 100) if total > 0 else 0
                }
        finally:
            conn.close()

    def get_recent_feedback(self, limit: int = 50) -> List[Dict]:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT f.value, f.comment, f.thread_id, extract(epoch from f.ts),
                           COALESCE(u.email, 'Anonymous') as user_email,
                           (SELECT content FROM messages m2 
                            WHERE m2.session_id = f.thread_id AND m2.role = 'user' AND m2.ts <= m_asst.ts 
                            ORDER BY m2.ts DESC LIMIT 1) as question,
                           m_asst.sources as sources
                    FROM feedback f
                    LEFT JOIN messages m_asst ON (f.for_id = CAST(m_asst.id AS TEXT) OR f.for_id = m_asst.id)
                    LEFT JOIN users u ON m_asst.user_id = u.id
                    ORDER BY f.ts DESC LIMIT %s
                """, (limit,))
                rows = cur.fetchall()
                return [
                    {
                        "value": r[0],
                        "comment": r[1],
                        "thread_id": r[2],
                        "ts": r[3],
                        "user": r[4],
                        "question": r[5],
                        "sources": r[6]
                    }
                    for r in rows
                ]
        finally:
            conn.close()

    def verify_email(self, email: str) -> None:
        # Generate code
        import random
        code = f"{random.randint(0, 999999):06d}"
        import time
        expires = time.time() + 900 # 15 mins
        
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE users SET verification_code=%s, verification_expires_at=to_timestamp(%s) WHERE email=%s", (code, expires, email))
            conn.commit()
        finally:
            conn.close()
            
        _send_verification_email(email, code)

    def check_verification(self, email: str, code: str) -> bool:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT verification_code, extract(epoch from verification_expires_at) FROM users WHERE email=%s", (email,))
                row = cur.fetchone()
                if not row: return False
                saved_code, expires = row
                import time
                if saved_code == code and expires > time.time():
                    cur.execute("UPDATE users SET verified=1 WHERE email=%s", (email,))
                    conn.commit()
                    return True
                return False
        finally:
            conn.close()
            
    def set_verified(self, user_id: int) -> None:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE users SET verified=1 WHERE id=%s", (user_id,))
            conn.commit()
        finally:
            conn.close()

    def is_verified(self, user_id: int) -> bool:
        print(f"🔍 [DB] is_verified() called for user_id={user_id}", flush=True)
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT verified FROM users WHERE id=%s", (user_id,))
                row = cur.fetchone()
                result = bool(row[0]) if row else False
                print(f"   is_verified result: {result} (raw={row})", flush=True)
                return result
        except Exception as e:
            print(f"❌ [DB] is_verified() failed: {e}", flush=True)
            import traceback; traceback.print_exc()
            return False
        finally:
            conn.close()


def get_chat_history_store() -> BaseChatHistoryStore:
    """Return store: SQLite for local. Set DATABASE_URL (postgres) for Cloud SQL (add PGChatHistoryStore later)."""
    db_url = os.getenv("DATABASE_URL")
    print(f"🏗️ get_chat_history_store() called", flush=True)
    print(f"   DATABASE_URL present: {bool(db_url)}", flush=True)
    if db_url:
        print(f"   DATABASE_URL starts with: {db_url[:30]}...", flush=True)
    if db_url and db_url.startswith("postgres"):
        print(f"   → Creating CloudSQLChatHistoryStore (PostgreSQL)", flush=True)
        store = CloudSQLChatHistoryStore(db_url)
        # Eagerly test connection so errors show in startup logs
        try:
            print(f"   → Testing PostgreSQL connection...", flush=True)
            test_conn = store._make_conn(dbname_override="postgres")
            print(f"   ✅ PostgreSQL connection test PASSED (server_version={test_conn.server_version})", flush=True)
            test_conn.close()
        except Exception as e:
            print(f"   ❌ PostgreSQL connection test FAILED: {e}", flush=True)
            print(f"   ❌ HINT: Ensure the Cloud Run service account has 'roles/cloudsql.client' IAM role", flush=True)
            print(f"   ❌ HINT: Run: gcloud projects add-iam-policy-binding PROJECT_ID "
                  f"--member='serviceAccount:cloud-run-sa@PROJECT_ID.iam.gserviceaccount.com' "
                  f"--role='roles/cloudsql.client'", flush=True)
            import traceback; traceback.print_exc()
            # Don't fallback — let the real error surface to the user
        return store
    # On Cloud Run, filesystem is read-only except /tmp
    _default_db = "/tmp/chat_history.sqlite3" if os.getenv("K_SERVICE") else os.path.join(os.path.dirname(__file__), "..", "chat_history.sqlite3")
    db_path = os.getenv("CHAT_DB_PATH", _default_db)
    print(f"   → Using SQLiteChatHistoryStore at: {db_path}", flush=True)
    return SQLiteChatHistoryStore(db_path)
