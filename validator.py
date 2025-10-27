#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib, json, os, sqlite3, re
from datetime import datetime
from typing import List
import pandas as pd
import streamlit as st

# --------------- Config ---------------
st.set_page_config(page_title="Q&A Validator", page_icon="✅", layout="wide")

# >>> Set your JSONL path here (absolute path is safest)
DATASET_DEFAULT = os.environ.get("QAV_DATASET", "geminituning_val.jsonl")
DB_PATH_DEFAULT = "feedback.db"
AUTOLOAD_ON_START = True  # auto-load DATASET_DEFAULT at startup if present

# --------------- DB Helpers ---------------
def get_conn(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id TEXT NOT NULL,
            item_index INTEGER,
            dataset_hash TEXT,
            timestamp TEXT,
            validator_id TEXT,
            correctness TEXT,
            suggested_answer TEXT,
            comments TEXT,
            tags TEXT,
            original_question TEXT,
            original_answer TEXT,
            original_context TEXT
        )
    """)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_item_validator ON feedback(item_id, validator_id)")
    return conn

def upsert_feedback(conn, row: dict):
    cur = conn.cursor()
    cur.execute("SELECT id FROM feedback WHERE item_id=? AND validator_id=?", (row["item_id"], row["validator_id"]))
    got = cur.fetchone()
    if got:
        fid = got[0]
        conn.execute("""
            UPDATE feedback
            SET item_index=?, dataset_hash=?, timestamp=?, correctness=?, suggested_answer=?, comments=?, tags=?,
                original_question=?, original_answer=?, original_context=?
            WHERE id=?
        """, (row["item_index"], row["dataset_hash"], row["timestamp"], row["correctness"], row["suggested_answer"],
              row["comments"], row["tags"], row["original_question"], row["original_answer"], row["original_context"], fid))
    else:
        conn.execute("""
            INSERT INTO feedback
            (item_id, item_index, dataset_hash, timestamp, validator_id, correctness, suggested_answer, comments, tags,
             original_question, original_answer, original_context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (row["item_id"], row["item_index"], row["dataset_hash"], row["timestamp"], row["validator_id"],
              row["correctness"], row["suggested_answer"], row["comments"], row["tags"],
              row["original_question"], row["original_answer"], row["original_context"]))
    conn.commit()

def read_all_feedback(conn, validator_id: str = None) -> pd.DataFrame:
    if validator_id:
        q = "SELECT * FROM feedback WHERE validator_id=? ORDER BY item_index"
        return pd.read_sql_query(q, conn, params=(validator_id,))
    return pd.read_sql_query("SELECT * FROM feedback ORDER BY item_index", conn)


def normalize_whitespace(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    # unify line endings
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    # trim each line’s edges
    s = "\n".join(line.strip() for line in s.split("\n"))
    # collapse runs of spaces/tabs inside lines
    s = re.sub(r"[ \t]+", " ", s)
    # collapse 3+ newlines to a single blank line (i.e., two newlines)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# --------------- Data Loading & Normalization ---------------
def sha1_text(s: str) -> str:
    import hashlib as _hashlib
    return _hashlib.sha1(s.encode("utf-8")).hexdigest()

def dataset_hash(records: List[dict]) -> str:
    blob = json.dumps(records[:50], sort_keys=True, ensure_ascii=False)
    return sha1_text(blob)

CANDIDATE_CONTEXT_KEYS = ["context", "passage", "doc", "document", "evidence", "source", "text"]
CANDIDATE_QUESTION_KEYS = ["question", "query", "prompt"]
CANDIDATE_ANSWER_KEYS   = ["answer", "response", "target", "ground_truth", "reference_answer", "output", "label"]

def choose_key(d: dict, candidates: List[str]) -> str | None:
    for k in candidates:
        if k in d:
            return k
    lowered = {k.lower(): k for k in d.keys()}
    for k in candidates:
        if k in lowered:
            return lowered[k]
    return None

def parse_gemini_contents(record: dict):
    """
    Parse Gemini-style {"contents": [{"role":"user","parts":[{"text":...}]}, {"role":"model","parts":[{"text":...}]}]}
    Expected 'CONTEXT:' ... 'QUESTION:' markers in the user text.
    """
    if "contents" not in record or not isinstance(record["contents"], list):
        return None

    user_texts, model_texts = [], []
    for m in record["contents"]:
        role = m.get("role")
        for p in m.get("parts", []):
            t = p.get("text")
            if not isinstance(t, str):
                continue
            if role == "user":
                user_texts.append(t)
            elif role in ("model", "assistant"):
                model_texts.append(t)

    user_blob = "\n\n".join(user_texts).strip()
    answer    = "\n\n".join(model_texts).strip()

    # Split on QUESTION:
    ctx = user_blob
    question = ""
    parts = re.split(r"(?i)\bQUESTION:\s*", user_blob, maxsplit=1)
    if len(parts) == 2:
        ctx, question = parts[0], parts[1]
    # Remove leading CONTEXT:
    ctx = re.sub(r"(?i)^\s*CONTEXT:\s*", "", ctx).strip()
    question = question.strip()
    return {
        "context": ctx,
        "question": question if question else user_blob,  # fallback if no QUESTION:
        "answer": answer
    }

def normalize_records(raw: List[dict]) -> List[dict]:
    """Return list of dicts with keys: index, context, question, answer, extra"""
    normalized = []
    for i, r in enumerate(raw):
        gem = parse_gemini_contents(r)
        if gem:
            normalized.append({
                "index": i,
                "context": gem["context"],
                "question": gem["question"],
                "answer": gem["answer"],
                "extra": {k: v for k, v in r.items() if k not in {"contents"}}
            })
            continue

        # Fallback: classic flat schema
        ctx_k = choose_key(r, CANDIDATE_CONTEXT_KEYS)
        q_k   = choose_key(r, CANDIDATE_QUESTION_KEYS)
        a_k   = choose_key(r, CANDIDATE_ANSWER_KEYS)

        context  = r.get(ctx_k, "") if ctx_k else ""
        question = r.get(q_k, "") if q_k else ""
        answer   = r.get(a_k, "") if a_k else ""

        normalized.append({
            "index": i,
            "context": context if isinstance(context, str) else json.dumps(context, ensure_ascii=False),
            "question": question if isinstance(question, str) else json.dumps(question, ensure_ascii=False),
            "answer": answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False),
            "extra": {k: v for k, v in r.items() if k not in {ctx_k, q_k, a_k} if k is not None}
        })
    return normalized

def load_jsonl_from_path(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# --------------- UI Helpers ---------------
def init_state():
    if "data" not in st.session_state: st.session_state.data = []
    if "norm" not in st.session_state: st.session_state.norm = []
    if "dataset_hash" not in st.session_state: st.session_state.dataset_hash = None
    if "i" not in st.session_state: st.session_state.i = 0
    if "db_path" not in st.session_state: st.session_state.db_path = DB_PATH_DEFAULT
    if "validator_id" not in st.session_state: st.session_state.validator_id = ""
    if "conn" not in st.session_state: st.session_state.conn = None
    if "dataset_path" not in st.session_state: st.session_state.dataset_path = DATASET_DEFAULT

def current_item():
    if not st.session_state.norm:
        return None
    i = max(0, min(st.session_state.i, len(st.session_state.norm) - 1))
    st.session_state.i = i
    return st.session_state.norm[i], i

def autoload_dataset_if_needed():
    if not AUTOLOAD_ON_START:
        return
    if st.session_state.norm:
        return
    path = st.session_state.dataset_path
    if path and os.path.exists(path):
        try:
            raw = load_jsonl_from_path(path)
            norm = normalize_records(raw)
            st.session_state.data = raw
            st.session_state.norm = norm
            st.session_state.dataset_hash = dataset_hash(raw)
            st.session_state.i = 0
            st.sidebar.success(f"Auto-loaded {len(norm)} items from {path}")
        except Exception as e:
            st.sidebar.error(f"Auto-load failed: {e}")
# --- Compact context formatting helpers ---
def normalize_whitespace(text: str) -> str:
    """Trim each line, collapse internal spaces, kill massive blank runs."""
    if not isinstance(text, str):
        text = str(text)
    # unify newlines & spaces
    s = (text
         .replace("\r\n", "\n").replace("\r", "\n")
         .replace("\u00A0", " "))  # NBSP -> normal space
    # trim each line
    s = "\n".join(line.strip() for line in s.split("\n"))
    # collapse multiple spaces/tabs inside a line
    s = re.sub(r"[ \t]+", " ", s)
    # remove leading/trailing blank lines
    s = s.strip()
    # collapse ANY 2+ blank lines down to ONE blank line
    s = re.sub(r"\n{2,}", "\n", s)
    return s

def remove_noise_headers(text: str) -> str:
    """Remove lines like '(page=680)' and stray 'Context' labels at top."""
    lines = []
    for ln in text.splitlines():
        raw = ln.strip()
        if re.match(r"^\(page=\d+\)$", raw, flags=re.I):
            continue
        if raw.lower() in {"context", "aggregate"}:
            # Drop standalone labels that clutter the view
            continue
        lines.append(ln)
    return "\n".join(lines)

def extract_tables_from_text(text: str):
    """
    Find blocks that start with 'Sieve    % Passing' and convert following rows
    into a DataFrame. Returns (text_with_placeholders, [dataframes]).
    """
    lines = text.splitlines()
    out_lines, tables = [], []
    i = 0
    while i < len(lines):
        line_norm = re.sub(r"\s+", " ", lines[i].strip()).lower()
        if line_norm == "sieve % passing":
            # collect table rows until a blank line or a header-like line
            rows = []
            i += 1
            while i < len(lines):
                row = lines[i].strip()
                if not row:
                    break
                # split by 2+ spaces or tabs; if that fails, split near the end
                parts = re.split(r"\s{2,}|\t+", row)
                if len(parts) < 2:
                    m = re.match(r"(.+?)\s+(\d+(?:\s*(?:to|-)\s*\d+)?[%µa-zA-Z]*)$", row)
                    if m:
                        parts = [m.group(1), m.group(2)]
                    else:
                        # fallback: last space split
                        if " " in row:
                            parts = [row.rsplit(" ", 1)[0], row.rsplit(" ", 1)[1]]
                        else:
                            parts = [row, ""]
                rows.append(parts[:2])
                i += 1
            # store the table
            import pandas as _pd
            df = _pd.DataFrame(rows, columns=["Sieve", "% Passing"])
            tables.append(df)
            out_lines.append("[[TABLE_BLOCK]]")
        else:
            out_lines.append(lines[i])
            i += 1
    return "\n".join(out_lines), tables

# --------------- App ---------------
def main():
    init_state()
    st.title("WYDOT SPECS Q&A Validator ✅")
    st.caption("Review question–answer pairs against context and store your feedback locally.")

    with st.sidebar:
        st.header("Setup")
        st.session_state.db_path = st.text_input("SQLite DB file", value=st.session_state.db_path)
        if st.session_state.conn is None:
            st.session_state.conn = get_conn(st.session_state.db_path)

        st.session_state.validator_id = st.text_input("Your Validator ID (e.g., name or email)", value=st.session_state.validator_id)

        st.divider()
        st.subheader("Dataset")
        st.session_state.dataset_path = st.text_input("JSONL path", value=st.session_state.dataset_path)
        if st.button("Load from path"):
            try:
                raw = load_jsonl_from_path(st.session_state.dataset_path)
                norm = normalize_records(raw)
                st.session_state.data = raw
                st.session_state.norm = norm
                st.session_state.dataset_hash = dataset_hash(raw)
                st.session_state.i = 0
                st.success(f"Loaded {len(norm)} items. Dataset hash: {st.session_state.dataset_hash[:10]}…")
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")

        if st.session_state.norm:
            total = len(st.session_state.norm)
            st.write(f"Dataset items: **{total}**")
            if st.session_state.validator_id:
                df_me = read_all_feedback(st.session_state.conn, st.session_state.validator_id)
                st.write(f"Your reviews saved: **{len(df_me)}**")
            if st.button("Export all feedback as CSV"):
                df = read_all_feedback(st.session_state.conn)
                if df.empty:
                    st.info("No feedback yet.")
                else:
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download CSV", data=csv, file_name="feedback_export.csv", mime="text/csv")

        st.caption("Tip: Use ←/→ buttons below to navigate. Click **Save** to write to the DB.")

    # Try auto-load at startup
    autoload_dataset_if_needed()

    if not st.session_state.norm:
        st.info("Load a JSONL dataset to start reviewing.")
        st.stop()

    item, idx = current_item()

    # Header / navigation
    cols = st.columns([1,1,1,6])
    with cols[0]:
        if st.button("⟵ Prev", use_container_width=True):
            st.session_state.i = max(0, st.session_state.i - 1); st.rerun()
    with cols[1]:
        if st.button("Next ⟶", use_container_width=True):
            st.session_state.i = min(len(st.session_state.norm) - 1, st.session_state.i + 1); st.rerun()
    with cols[2]:
        st.number_input("Jump to index", min_value=0, max_value=len(st.session_state.norm)-1,
                        value=st.session_state.i, key="jump_idx", step=1)
        if st.button("Go"):
            st.session_state.i = st.session_state.jump_idx; st.rerun()
    with cols[3]:
        st.progress((idx + 1) / len(st.session_state.norm), text=f"Item {idx + 1} / {len(st.session_state.norm)}")

    # Show the item
    st.subheader(f"Item #{idx}")
    with st.expander("Context", expanded=True):
        raw_ctx = item.get("context", "")
    # clean + compact text
        ctx = remove_noise_headers(raw_ctx)
        ctx = normalize_whitespace(ctx)
        # extract any sieve tables and render them nicely
        ctx_text, ctx_tables = extract_tables_from_text(ctx)

        # optional truncation for super long contexts (toggle here if you want)
        MAX_CHARS = 3000
        truncated = False
        if len(ctx_text) > MAX_CHARS:
            ctx_text = ctx_text[:MAX_CHARS].rstrip() + " …"
            truncated = True

        st.markdown(f"<div style='white-space: pre-wrap'>{ctx_text}</div>", unsafe_allow_html=True)
        if truncated:
            st.caption("Context truncated for compact view.")

        # show tables (if any) under the text
        for df in ctx_tables:
            st.table(df)



    qcol, acol = st.columns([1,1])
    with qcol:
        st.markdown("### Question")
        st.markdown(f"<div style='white-space: pre-wrap'>{item['question']}</div>", unsafe_allow_html=True)
    with acol:
        st.markdown("### Given Answer (to validate)")
        st.markdown(f"<div style='white-space: pre-wrap; border-left: 4px solid #ddd; padding-left: 10px;'>{item['answer']}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Your Review")

    default_idx = 3  # Unclear by default
    try:
        a_low = item["answer"].lower()
        default_idx = 0 if "correct" in a_low else (1 if "part" in a_low else 3)
    except Exception:
        pass

    correctness = st.radio(
        "Label",
        options=["Correct", "Partially correct", "Incorrect", "Unclear"],
        index=default_idx,
        horizontal=True,
        key=f"correctness_{idx}"
    )

    suggested_answer = st.text_area(
        "Suggested / Corrected Answer (optional)",
        value=item.get("answer", ""),
        height=120,
        key=f"suggested_{idx}"
    )

    tags = st.multiselect(
        "Quick tags (optional)",
        options=["Ambiguous question", "Insufficient context", "Hallucination", "Formatting issue", "Typos", "Policy concern"],
        default=[],
        key=f"tags_{idx}"
    )

    comments = st.text_area("Comments (optional)", key=f"comments_{idx}")

    # Save buttons
    save_col, skip_col = st.columns([1,1])
    with save_col:
        if st.button("💾 Save", type="primary"):
            if not st.session_state.validator_id.strip():
                st.warning("Please enter your Validator ID in the sidebar first.")
            else:
                iid = sha1_text(json.dumps({"q": item["question"], "a": item["answer"], "c": item["context"]}, sort_keys=True, ensure_ascii=False))
                row = {
                    "item_id": iid,
                    "item_index": idx,
                    "dataset_hash": st.session_state.dataset_hash or "",
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "validator_id": st.session_state.validator_id.strip(),
                    "correctness": correctness,
                    "suggested_answer": suggested_answer.strip(),
                    "comments": comments.strip(),
                    "tags": ",".join(tags),
                    "original_question": item["question"],
                    "original_answer": item["answer"],
                    "original_context": item["context"],
                }
                try:
                    upsert_feedback(st.session_state.conn, row)
                    st.success("Saved!")
                except Exception as e:
                    st.error(f"Save failed: {e}")

    with skip_col:
        if st.button("Skip → Next"):
            st.session_state.i = min(len(st.session_state.norm) - 1, st.session_state.i + 1)
            st.rerun()

    # Footer: your rows
    st.markdown("---")
    if st.session_state.validator_id:
        df_me = read_all_feedback(st.session_state.conn, st.session_state.validator_id)
        st.caption(f"Your saved reviews: {len(df_me)}")
        if not df_me.empty:
            st.dataframe(df_me[["item_index","correctness","tags","timestamp","comments"]].sort_values("item_index"),
                         use_container_width=True)

if __name__ == "__main__":
    main()
