"""
Conversation memory: Redis (fast) with fallback to DB/session.
- Local: use Redis if REDIS_URL or REDIS_SERVER is set; else caller uses store/session.
"""
import os
import json
from typing import List, Dict, Optional, Callable

# Key prefix and TTL
MEMORY_KEY_PREFIX = "wydot:chat:"
DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # 7 days
MAX_MESSAGES_STORED = 50


def _redis_client():
    """Return Redis client if configured, else None."""
    url = os.getenv("REDIS_URL")
    if url:
        try:
            import redis
            return redis.from_url(url, decode_responses=True)
        except Exception:
            return None
    host = os.getenv("REDIS_SERVER", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    if not host:
        return None
    try:
        import redis
        return redis.Redis(host=host, port=port, decode_responses=True)
    except Exception:
        return None


def _memory_key(user_id: int, session_id: str) -> str:
    return f"{MEMORY_KEY_PREFIX}{user_id}:{session_id}"


def get_recent(user_id: int, session_id: str, limit: int = 20, fallback: Optional[Callable[[int, str, int], List[Dict[str, str]]]] = None) -> List[Dict[str, str]]:
    """
    Get recent messages from Redis, or from fallback (e.g. DB get_recent) on miss.
    fallback: callable(user_id, session_id, limit) -> list of {role, content}.
    """
    r = _redis_client()
    if r is None:
        if fallback:
            return fallback(user_id, session_id, limit)
        return []

    key = _memory_key(user_id, session_id)
    try:
        raw = r.lrange(key, -limit, -1)
        if not raw and fallback:
            return fallback(user_id, session_id, limit)
        out = []
        for s in raw:
            try:
                out.append(json.loads(s))
            except Exception:
                pass
        return out[-limit:]
    except Exception:
        if fallback:
            return fallback(user_id, session_id, limit)
        return []


def append(user_id: int, session_id: str, role: str, content: str, ttl: int = DEFAULT_TTL_SECONDS) -> None:
    """Append one message and trim to MAX_MESSAGES_STORED. No-op if Redis unavailable."""
    r = _redis_client()
    if r is None:
        return

    key = _memory_key(user_id, session_id)
    try:
        r.rpush(key, json.dumps({"role": role, "content": content}))
        r.ltrim(key, -MAX_MESSAGES_STORED, -1)
        r.expire(key, ttl)
    except Exception:
        pass


def clear(user_id: int, session_id: str) -> None:
    """Clear conversation memory for this user/session."""
    r = _redis_client()
    if r is None:
        return
    try:
        r.delete(_memory_key(user_id, session_id))
    except Exception:
        pass
