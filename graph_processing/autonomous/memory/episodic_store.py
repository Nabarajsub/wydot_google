"""SQLite-backed episodic memory for completed tasks."""
from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_DB_PATH = Path(__file__).parent.parent / "task_artifacts" / "episodic.db"


class EpisodicStore:
    def __init__(self, db_path: Optional[str] = None):
        self._db_path = Path(db_path) if db_path is not None else _DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    goal TEXT,
                    outcome TEXT,
                    corrections TEXT,
                    created_at REAL
                )
            """)
            conn.commit()

    def save_episode(self, task_id: str, goal: str, outcome: str,
                     corrections: Optional[List[str]] = None):
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT INTO episodes (task_id, goal, outcome, corrections, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (task_id, goal, outcome,
                     json.dumps(corrections or []), time.time())
                )
                conn.commit()
        except Exception as e:
            print(f"[EpisodicStore] save error: {e}")

    def get_similar(self, goal: str, limit: int = 3) -> List[Dict[str, Any]]:
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT task_id, goal, outcome FROM episodes ORDER BY created_at DESC LIMIT ?",
                    (limit * 5,)
                ).fetchall()
            goal_lower = goal.lower()
            scored = []
            for row in rows:
                score = sum(w in row[1].lower() for w in goal_lower.split() if len(w) > 3)
                if score > 0:
                    scored.append((score, {"task_id": row[0], "goal": row[1], "outcome": row[2]}))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [r for _, r in scored[:limit]]
        except Exception as e:
            print(f"[EpisodicStore] get error: {e}")
            return []
