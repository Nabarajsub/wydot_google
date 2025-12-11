# features_plus.py
# Additive utilities that patch into the existing WYDOT Streamlit app
# No external network calls; relies on functions/objects defined in app.py's globals()

from __future__ import annotations
import re, time, io, json, math
from typing import Any, Dict, List, Tuple, Optional
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------
# 1) ASSIGNMENTS & @MENTIONS
# ---------------------------

def _ensure_assignments_table(conn):
    try:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS assignments(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            message_id INTEGER,
            to_email TEXT NOT NULL,
            note TEXT,
            payload TEXT,                -- JSON: {question, answer, sources}
            created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_assign_sid ON assignments (user_id, session_id, id)")
        conn.commit()
    except Exception:
        pass

def parse_mentions(text: str) -> List[str]:
    if not text: return []
    # simplest acceptable pattern: @name@example.com or @First.Last
    emails = re.findall(r'@([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})', text)
    return list(dict.fromkeys([e.lower() for e in emails]))

def save_assignment(chat_db, session_id: str, to_email: str, note: str, payload: Dict[str, Any]):
    # Reuse the same SQLite connection as your Chat DB
    conn = chat_db._conn  # uses your existing object; safe in your app context
    _ensure_assignments_table(conn)
    uid = int(st.session_state.get("user_id", 0) or 0)
    conn.execute(
        "INSERT INTO assignments (user_id, session_id, to_email, note, payload) VALUES (?,?,?,?,?)",
        (uid, session_id or "default", (to_email or "").strip().lower(), note or "", json.dumps(payload, ensure_ascii=False)),
    )
    conn.commit()

def list_assignments(chat_db, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    conn = chat_db._conn
    _ensure_assignments_table(conn)
    uid = int(st.session_state.get("user_id", 0) or 0)
    rows = conn.execute(
        "SELECT id, to_email, note, payload, created_at FROM assignments WHERE user_id=? AND session_id=? ORDER BY id DESC LIMIT ?",
        (uid, session_id or "default", int(limit)),
    ).fetchall()
    out = []
    for r in rows:
        try:
            payload = json.loads(r[3]) if r[3] else {}
        except Exception:
            payload = {}
        out.append({
            "id": int(r[0]),
            "to_email": r[1],
            "note": r[2],
            "payload": payload,
            "created_at": float(r[4] or 0.0),
        })
    return out

def render_assignments_ui(chat_db, session_id: str, last_q: str, last_a: str, last_sources: List[Dict[str, Any]], user_message_text: str):
    st.markdown("### 📬 Share this result")
    mentions = parse_mentions(user_message_text or "")
    if mentions:
        st.caption(f"Detected @mentions → {', '.join(mentions)}")

    c1, c2 = st.columns([0.5, 0.5])
    with c1:
        to_emails = st.text_input("To (comma-separated emails)", value=", ".join(mentions) if mentions else "", key="assign_to")
    with c2:
        note = st.text_input("Note (optional)", key="assign_note", placeholder="What should they review?")

    if st.button("Share & log assignment", use_container_width=True, key="btn_share_assign"):
        payload = {
            "question": last_q or "",
            "answer": last_a or "",
            "sources": last_sources or [],
            "model": st.session_state.get("model_choice"),
            "answer_mode": st.session_state.get("answer_mode"),
            "ts": time.time(),
        }
        any_ok = False
        for addr in [a.strip() for a in (to_emails or "").split(",") if a.strip()]:
            save_assignment(chat_db, session_id, addr, note, payload)
            any_ok = True
        if any_ok:
            st.success("Shared & logged ✅ (audit record stored locally).")
        else:
            st.warning("Please provide at least one email.")

    with st.expander("📁 Recent assignments for this conversation", expanded=False):
        rows = list_assignments(chat_db, session_id, limit=20)
        if not rows:
            st.caption("No assignments yet.")
        else:
            for r in rows:
                pl = r["payload"] or {}
                st.markdown(f"**→ {r['to_email']}** — _{time.strftime('%Y-%m-%d %H:%M', time.localtime(r['created_at']))}_")
                st.write(pl.get("question") or "")
                st.caption((r.get("note") or "").strip() or "—")
                st.divider()

# --------------------------------------
# 2) CITATIONS BAR (CLICKABLE CHIP FILTER)
# --------------------------------------

def _confidence_from_score(score: Optional[float]) -> float:
    # For cosine similarity in Milvus, higher is better; bound to [0,1]
    if score is None: return 0.0
    try:
        # Some Milvus configs yield [-1,1]; map to [0,1] safely
        return max(0.0, min(1.0, (float(score) + 1.0) / 2.0))
    except Exception:
        return 0.0

def render_citations_bar(sources: List[Dict[str, Any]]) -> Optional[Tuple[str, int]]:
    """Render chips. Returns (doc_id, page) if a chip is clicked, else None."""
    if not sources: 
        return None
    st.markdown("####  Citations")
    chip_cols = st.columns(min(4, len(sources)))
    selected = None
    for i, s in enumerate(sources[:8]):  # show top 8
        label = s.get("section") or f"p.{s.get('page','?')}"
        conf = _confidence_from_score(s.get("score") or s.get("distance"))
        pct = int(round(conf * 100))
        with chip_cols[i % len(chip_cols)]:
            if st.button(f"{label} · {pct}%", key=f"cite_chip_{i}"):
                selected = (s.get("doc_id") or "", int(s.get("page") or 0))
    if selected:
        st.session_state["citations_filter"] = {"doc_id": selected[0], "page": selected[1]}
    return selected

def apply_citations_filter(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flt = st.session_state.get("citations_filter")
    if not flt:
        return sources
    doc_id = flt.get("doc_id")
    page = flt.get("page")
    out = []
    for s in sources:
        if str(s.get("doc_id")) == str(doc_id):
            if page is None or int(s.get("page") or -1) == int(page):
                out.append(s)
    return out or sources

# ----------------------------
# 3) KEYBOARD-FIRST & VOICE UI
# ----------------------------

def render_keyboard_and_voice():
    # Cmd/Ctrl-K focuses the main query input; floating mic toggles audio_uploader tab
    components.html("""
    <script>
      // Focus input on Cmd/Ctrl+K
      document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
          const inputs = parent.document.querySelectorAll('input[type="text"]');
          for (const el of inputs) {
            if (el.placeholder && el.placeholder.toLowerCase().includes('ask anything')) {
              el.focus(); el.select();
              e.preventDefault(); return false;
            }
          }
        }
      }, true);
    </script>
    """, height=0)

    st.markdown("""
    <style>
      .floating-mic {
        position: fixed; right: 22px; bottom: 22px; 
        background: #fff; border: 1px solid #e6e6e6; 
        padding: 10px 14px; border-radius: 999px; box-shadow: 0 2px 15px rgba(0,0,0,.08);
        cursor: pointer; z-index: 9999;
      }
    </style>
    """, unsafe_allow_html=True)
    # A small note: we reuse the existing audio recorder in your composer.
    st.markdown("<div class='floating-mic'>🎙️ Push-to-talk (uses in-page recorder)</div>", unsafe_allow_html=True)

# -----------------------
# 4) MULTI-HOP RAG LAYER
# -----------------------

def _decompose_query_to_subqueries(client, model_id: str, query: str) -> List[str]:
    try:
        prompt = (
            "You split complex transportation/spec questions into 2–4 brief sub-queries.\n"
            "Return JSON list only, no prose. Example: [\"Define asphalt content limits\",\"Construction temperature range\"]\n"
            f"Question: {query}"
        )
        resp = client.models.generate_content(
            model=model_id,
            contents=[{"role":"user","parts":[{"text":prompt}]}],
            config={"max_output_tokens": 256, "temperature": 0.2}
        )
        raw = (getattr(resp, "text", None) or "").strip()
        js = json.loads(raw) if raw.startswith("[") else json.loads(re.search(r"(\[.*\])", raw, re.S).group(1))
        out = [str(x).strip() for x in js if str(x).strip()]
        return out[:4] or [query]
    except Exception:
        return [query]

def multi_hop_search(ns: dict, query: str, k_per_hop: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
    """Returns (joined_context, deduped_sources) using multiple sub-queries."""
    if not query or not query.strip():
        return "", []
    client = ns["get_genai_client"]()
    model_id = ns["get_selected_model_id"]()
    subqs = _decompose_query_to_subqueries(client, model_id, query)
    seen = set()
    all_sources: List[Dict[str, Any]] = []
    contexts: List[str] = []
    for sq in subqs:
        ctx, srcs = ns["milvus_similarity_search"](sq, k=k_per_hop)
        if ctx:
            contexts.append(ctx)
        for s in srcs or []:
            key = (s.get("doc_id"), s.get("page"))
            if key not in seen:
                seen.add(key)
                all_sources.append(s)
    return "\n\n".join(contexts), all_sources

# --------------------------------
# 5) SECTION-AWARE CHUNK GROUPING
# --------------------------------

def group_sources_by_section(sources: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for s in sources or []:
        name = s.get("section") or f"Page {s.get('page','?')}"
        groups.setdefault(name, []).append(s)
    return groups

def render_section_aware_preview(sources: List[Dict[str, Any]], build_pdf_url_func):
    groups = group_sources_by_section(sources)
    for sec, items in groups.items():
        with st.expander(f"§ {sec} ({len(items)} hits)", expanded=(sec == list(groups.keys())[0])):
            for s in items:
                page = s.get("page")
                src  = s.get("source")
                url  = build_pdf_url_func(src, page)
                conf = _confidence_from_score(s.get("score") or s.get("distance"))
                st.caption(f"p.{page} · conf {int(conf*100)}%")
                if url:
                    st.markdown(f"[Open PDF @ page {page}]({url})")
                st.write(s.get("preview") or "_(no preview)_")
            st.markdown("---")

# ------------------------------------------------
# 6) PATCH THE APP (override retrieval + add state)
# ------------------------------------------------

def patch_app(app_ns: dict):
    """
    app_ns is the globals() of your main file.
    This adds:
    - score/confidence to Milvus retrieval
    - sidebar toggle for multi-hop
    - small helpers in session_state
    """
    # Wrap original Milvus retrieval to inject score/confidence + section-safe fields
    orig_search = app_ns.get("milvus_similarity_search")

    def _patched_milvus_similarity_search(query: str, k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        if not query or not query.strip():
            return "", []
        uri = st.session_state.get("milvus_uri", app_ns.get("DEFAULT_MILVUS_URI"))
        token = st.session_state.get("milvus_token", app_ns.get("DEFAULT_MILVUS_TOKEN"))
        collection = st.session_state.get("collection", app_ns.get("DEFAULT_COLLECTION"))
        col, dim, _ = app_ns["get_milvus_collection"](uri, token, collection)
        if not col or not dim:
            return "", []

        try:
            qv = app_ns["embed_query_vector"](query, dim)
        except Exception as e:
            st.warning(f"[Embed] {e}")
            return "", []

        try:
            res = col.search(
                data=[qv],
                anns_field=app_ns.get("VECTOR_FIELD","vector"),
                param={"metric_type": app_ns.get("METRIC_TYPE","COSINE"), "params": {"ef": 64}},
                limit=k,
                output_fields=["doc_id","chunk_id","page","section","source","content"],
            )
        except Exception as e:
            st.warning(f"[Milvus search] {e}")
            return "", []

        chunks, sources = [], []
        if res and len(res) > 0:
            for hit in res[0]:
                md = hit.entity
                content = md.get("content") or ""
                chunks.append(content)
                sources.append({
                    "doc_id": md.get("doc_id"),
                    "page": md.get("page"),
                    "section": md.get("section"),
                    "source": md.get("source"),
                    "preview": content[:300] if content else "",
                    "score": getattr(hit, "distance", None),  # Milvus names it "distance" but it's similarity for COSINE
                })
        return "\n\n".join(chunks), sources

    if callable(orig_search):
        app_ns["milvus_similarity_search"] = _patched_milvus_similarity_search

    # Sidebar toggle (create keys but actual UI is in your sidebar code)
    st.session_state.setdefault("enable_multihop", False)

def use_multihop_if_enabled(app_ns: dict, query: str, default_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    if not query or not query.strip():
        return "", []
    if st.session_state.get("enable_multihop", False):
        try:
            return multi_hop_search(app_ns, query, k_per_hop=max(2, default_k//2))
        except Exception as e:
            st.info(f"Multi-hop fallback: {e}")
            return app_ns["milvus_similarity_search"](query, k=default_k)
    return app_ns["milvus_similarity_search"](query, k=default_k)
