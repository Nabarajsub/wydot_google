#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q&A Validator — Human/AI Split + Consistency Dashboard (API key first, robust fallback)

Quick start:
  - Create .env with: GEMINI_API_KEY=YOUR_KEY   (or GOOGLE_API_KEY=YOUR_KEY)
  - pip install streamlit google-generativeai python-dotenv
  - streamlit run qav_validator_ai_split.py

What’s included:
- Stable 80/20 split (AI/HUMAN) per item_id + dataset hash.
- Batch AI validation for the AI cohort.
- Per-item AI check.
- Cross-check AI on your human-labeled items.
- Consistency dashboard (confusion, % agreement, Cohen’s κ).

Auth strategy:
- Prefer AI Studio via API key (GEMINI_API_KEY or GOOGLE_API_KEY).
- Optional Vertex (set GEMINI_USE_VERTEX=1, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_REGION).
- If Vertex returns 403 scope/permission errors, auto-fallback to API key when available.
- DRY_RUN=1 to simulate labels without network.
"""

import hashlib, json, os, sqlite3, re, math
from datetime import datetime
from typing import List, Tuple, Optional
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ---------------- Load .env ----------------
load_dotenv()

# ---------------- Config ----------------
st.set_page_config(page_title="Q&A Validator", page_icon="✅", layout="wide")
DATASET_DEFAULT = os.environ.get("QAV_DATASET", "geminituning_val.jsonl")
DB_PATH_DEFAULT = "/tmp/feedback.db"#"feedback.db"
AUTOLOAD_ON_START = True

# AI config
a_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
# Prefer API key (AI Studio) if present; else Vertex if explicitly enabled
a_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
a_USE_VERTEX = os.environ.get("GEMINI_USE_VERTEX", "0") == "1"
a_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
a_REGION  = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
a_DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
AI_VALIDATOR_ID = f"AI:{a_MODEL}"

# ---------------- DB ----------------
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

def read_all_feedback(conn, validator_id: Optional[str] = None) -> pd.DataFrame:
    if validator_id:
        return pd.read_sql_query("SELECT * FROM feedback WHERE validator_id=? ORDER BY item_index", conn, params=(validator_id,))
    return pd.read_sql_query("SELECT * FROM feedback ORDER BY item_index", conn)

# ---------------- Text / data utils ----------------
def sha1_text(s: str) -> str:
    import hashlib as _hashlib
    return _hashlib.sha1(s.encode("utf-8")).hexdigest()

def dataset_hash(records: List[dict]) -> str:
    blob = json.dumps(records[:50], sort_keys=True, ensure_ascii=False)
    return sha1_text(blob)

CANDIDATE_CONTEXT_KEYS  = ["context","passage","doc","document","evidence","source","text"]
CANDIDATE_QUESTION_KEYS = ["question","query","prompt"]
CANDIDATE_ANSWER_KEYS   = ["answer","response","target","ground_truth","reference_answer","output","label"]

def choose_key(d: dict, candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in d: return k
    lowered = {k.lower(): k for k in d.keys()}
    for k in candidates:
        if k in lowered: return lowered[k]
    return None

def parse_gemini_contents(record: dict):
    if "contents" not in record or not isinstance(record["contents"], list):
        return None
    user_texts, model_texts = [], []
    for m in record["contents"]:
        role = m.get("role")
        for p in m.get("parts", []):
            t = p.get("text")
            if not isinstance(t, str): continue
            if role == "user": user_texts.append(t)
            elif role in ("model","assistant"): model_texts.append(t)
    user_blob = "\n\n".join(user_texts).strip()
    answer    = "\n\n".join(model_texts).strip()
    ctx = user_blob
    question = ""
    parts = re.split(r"(?i)\bQUESTION:\s*", user_blob, maxsplit=1)
    if len(parts) == 2:
        ctx, question = parts[0], parts[1]
    ctx = re.sub(r"(?i)^\s*CONTEXT:\s*", "", ctx).strip()
    return {"context": ctx, "question": question.strip() or user_blob, "answer": answer}

def normalize_records(raw: List[dict]) -> List[dict]:
    out = []
    for i, r in enumerate(raw):
        gem = parse_gemini_contents(r)
        if gem:
            out.append({"index": i, "context": gem["context"], "question": gem["question"], "answer": gem["answer"],
                        "extra": {k:v for k,v in r.items() if k!="contents"}})
            continue
        ctx_k = choose_key(r, CANDIDATE_CONTEXT_KEYS)
        q_k   = choose_key(r, CANDIDATE_QUESTION_KEYS)
        a_k   = choose_key(r, CANDIDATE_ANSWER_KEYS)
        context  = r.get(ctx_k, "") if ctx_k else ""
        question = r.get(q_k, "") if q_k else ""
        answer   = r.get(a_k, "") if a_k else ""
        out.append({
            "index": i,
            "context": context if isinstance(context, str) else json.dumps(context, ensure_ascii=False),
            "question": question if isinstance(question, str) else json.dumps(question, ensure_ascii=False),
            "answer":   answer   if isinstance(answer, str)   else json.dumps(answer, ensure_ascii=False),
            "extra": {k:r[k] for k in r.keys() if k not in {ctx_k,q_k,a_k} and k is not None}
        })
    return out

def load_jsonl_from_path(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# ---------------- Context helpers ----------------
def normalize_whitespace_compact(text: str) -> str:
    if not isinstance(text, str): text = str(text)
    s = text.replace("\r\n","\n").replace("\r","\n").replace("\u00A0"," ")
    s = "\n".join(line.strip() for line in s.split("\n"))
    s = re.sub(r"[ \t]+"," ", s).strip()
    s = re.sub(r"\n{2,}","\n", s)
    return s

def remove_noise_headers(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        raw = ln.strip()
        if re.match(r"^\(page=\d+\)$", raw, flags=re.I): continue
        if raw.lower() in {"context","aggregate"}: continue
        lines.append(ln)
    return "\n".join(lines)

def extract_tables_from_text(text: str):
    lines = text.splitlines()
    out_lines, tables = [], []
    i = 0
    while i < len(lines):
        line_norm = re.sub(r"\s+"," ", lines[i].strip()).lower()
        if line_norm == "sieve % passing":
            rows = []; i += 1
            while i < len(lines):
                row = lines[i].strip()
                if not row: break
                parts = re.split(r"\s{2,}|\t+", row)
                if len(parts) < 2:
                    m = re.match(r"(.+?)\s+(\d+(?:\s*(?:to|-)\s*\d+)?[%µa-zA-Z]*)$", row)
                    if m: parts = [m.group(1), m.group(2)]
                    else:
                        parts = [row.rsplit(" ",1)[0], row.rsplit(" ",1)[1]] if " " in row else [row,""]
                rows.append(parts[:2]); i += 1
            tables.append(pd.DataFrame(rows, columns=["Sieve","% Passing"]))
            out_lines.append("[[TABLE_BLOCK]]")
        else:
            out_lines.append(lines[i]); i += 1
    return "\n".join(out_lines), tables

# ---------------- Split ----------------
SPLIT_RATIO_AI = 0.80
def stable_partition(item_id: str, dataset_sig: str, ratio: float = SPLIT_RATIO_AI) -> str:
    h = hashlib.sha1((dataset_sig + "::" + item_id).encode("utf-8")).hexdigest()
    bucket = int(h[:6], 16) % 100
    return "AI" if bucket < int(ratio*100) else "HUMAN"

# ---------------- Gemini client (API-key first, with Vertex fallback) ----------------
class GeminiClient:
    """
    Strategy:
      - If GEMINI_API_KEY/GOOGLE_API_KEY exists → use AI Studio (google.generativeai).
      - Else if GEMINI_USE_VERTEX=1 → use Vertex (google.genai).
      - If Vertex returns 403/PermissionDenied scope errors and API key exists → retry via AI Studio.
      - If nothing works or DRY_RUN=1 → heuristic labels.
    """
    def __init__(self):
        self.ready = False
        self.err = None
        self.path = None
        self._studio_model = None
        self._vertex_client = None

        try:
            if a_DRY_RUN:
                self.ready = True
                self.path = "DRY_RUN"
                return

            if a_API_KEY:
                self._ensure_studio()
                if self._studio_model is not None:
                    self.ready = True
                    self.path = "studio:google.generativeai"
            elif a_USE_VERTEX:
                self._ensure_vertex()
                if self._vertex_client is not None:
                    self.ready = True
                    self.path = "vertex:google.genai"

            # If still not ready but both options exist, try Vertex second
            if not self.ready and a_USE_VERTEX:
                self._ensure_vertex()
                if self._vertex_client is not None:
                    self.ready = True
                    self.path = "vertex:google.genai"

            if not self.ready:
                self.err = "No working Gemini client (no API key, Vertex disabled, or SDKs missing)."
        except Exception as e:
            self.err = str(e)
            self.ready = False

    # Lazy initializers
    def _ensure_studio(self):
        if self._studio_model is not None:
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=a_API_KEY)
            self._studio_model = genai.GenerativeModel(a_MODEL)
        except Exception:
            self._studio_model = None

    def _ensure_vertex(self):
        if self._vertex_client is not None:
            return
        try:
            from google import genai  # unified Vertex SDK
            self._vertex_client = genai.Client(project=a_PROJECT, location=a_REGION)
        except Exception:
            self._vertex_client = None

    # Main call with fallback logic
    def judge(self, context: str, question: str, answer: str) -> dict:
        sys_prompt = (
            "You are a meticulous Q&A validator for civil engineering specifications. "
            "Decide if the GIVEN ANSWER is correct with respect to the CONTEXT for the QUESTION. "
            "Prefer 'Unclear' if the context does not support a definite judgment. If partially correct, say so. "
            "Return strict JSON only, with keys: label, suggested_answer, justification."
        )
        user_prompt = (
            f"CONTEXT\n-----\n{context}\n\n"
            f"QUESTION\n-----\n{question}\n\n"
            f"GIVEN ANSWER\n-----\n{answer}\n\n"
            "Return JSON with fields: label, suggested_answer, justification."
        )

        if a_DRY_RUN:
            return self._heuristic(context, answer)

        # Preferred path: Studio if API key present; else Vertex
        first = "studio" if (a_API_KEY and self._studio_model is not None) else ("vertex" if self._vertex_client is not None else None)

        if first == "studio":
            ok, resp = self._call_studio(sys_prompt, user_prompt)
            if ok:
                self.path = "studio:google.generativeai"
                return resp
            # try Vertex if available
            if self._vertex_client is not None:
                ok2, resp2 = self._call_vertex(sys_prompt, user_prompt)
                if ok2:
                    self.path = "vertex:google.genai"
                    return resp2
                return self._fallback(resp2)
            return self._fallback(resp)

        if first == "vertex":
            ok, resp = self._call_vertex(sys_prompt, user_prompt)
            if ok:
                self.path = "vertex:google.genai"
                return resp
            # if Vertex permission/scope error and API key exists, retry via Studio
            if self._is_scope_error(resp) and a_API_KEY:
                self._ensure_studio()
                if self._studio_model is not None:
                    ok2, resp2 = self._call_studio(sys_prompt, user_prompt)
                    if ok2:
                        self.path = "studio:google.generativeai"
                        return resp2
                    return self._fallback(resp2)
            # otherwise, try Studio if we have a key; else fallback
            if a_API_KEY:
                self._ensure_studio()
                if self._studio_model is not None:
                    ok2, resp2 = self._call_studio(sys_prompt, user_prompt)
                    if ok2:
                        self.path = "studio:google.generativeai"
                        return resp2
                    return self._fallback(resp2)
            return self._fallback(resp)

        # No client available: heuristic
        return self._heuristic(context, answer)

    # Helpers
    def _heuristic(self, context: str, answer: str) -> dict:
        ctx_lower = (context or "").lower()
        ans_lower = (answer or "").lower().strip()
        if ans_lower and ans_lower in ctx_lower:
            label = "Correct"
        elif len(ans_lower) > 20 and any(w in ctx_lower for w in ans_lower.split()[:4]):
            label = "Partially correct"
        else:
            label = "Incorrect" if (" not " in ans_lower and " not " not in ctx_lower) else "Unclear"
        return {"label": label, "suggested_answer": answer, "justification": f"heuristic={self.path or 'unavailable'}"}

    def _call_studio(self, sys_prompt: str, user_prompt: str):
        try:
            if self._studio_model is None:
                return False, {"error": "studio-not-initialized"}
            resp = self._studio_model.generate_content(
                [sys_prompt, user_prompt],
                generation_config={"response_mime_type": "application/json"},
            )
            return True, safe_json_parse(resp.text)
        except Exception as e:
            return False, {"error": f"studio:{type(e).__name__}:{e}"}

    def _call_vertex(self, sys_prompt: str, user_prompt: str):
        try:
            if self._vertex_client is None:
                return False, {"error": "vertex-not-initialized"}
            resp = self._vertex_client.models.generate_content(
                model=a_MODEL,
                contents=[
                    {"role": "system", "parts": [{"text": sys_prompt}]},
                    {"role": "user",   "parts": [{"text": user_prompt}]},
                ],
                config={"response_mime_type": "application/json"},
            )
            return True, safe_json_parse(resp.text)
        except Exception as e:
            return False, {"error": f"vertex:{type(e).__name__}:{e}"}

    def _is_scope_error(self, resp: dict) -> bool:
        if not isinstance(resp, dict):
            return False
        msg = json.dumps(resp, ensure_ascii=False).lower()
        return ("permissiondenied" in msg or
                "access_token_scope_insufficient" in msg or
                '"code": 403' in msg or
                "insufficient authentication scopes" in msg)

    def _fallback(self, resp):
        return {"label": "Unclear", "suggested_answer": "", "justification": f"error: {resp}"}

def safe_json_parse(txt: str) -> dict:
    try: return json.loads(txt)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", txt)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
        return {"label":"Unclear","suggested_answer":"","justification":"non-json response"}

# ---------------- State ----------------
def init_state():
    for k, v in {
        "data": [], "norm": [], "dataset_hash": None, "i": 0,
        "db_path": DB_PATH_DEFAULT, "validator_id": "", "conn": None,
        "dataset_path": DATASET_DEFAULT, "view_mode": "HUMAN", "ai_client": GeminiClient()
    }.items():
        if k not in st.session_state: st.session_state[k]=v

def current_item():
    if not st.session_state.norm: return None
    i = max(0, min(st.session_state.i, len(st.session_state.norm)-1))
    st.session_state.i = i
    return st.session_state.norm[i], i

def autoload_dataset_if_needed():
    if not AUTOLOAD_ON_START or st.session_state.norm: return
    path = st.session_state.dataset_path
    if path and os.path.exists(path):
        try:
            raw = load_jsonl_from_path(path); norm = normalize_records(raw)
            st.session_state.data = raw; st.session_state.norm = norm
            st.session_state.dataset_hash = dataset_hash(raw); st.session_state.i=0
            st.sidebar.success(f"Auto-loaded {len(norm)} items from {path}")
        except Exception as e:
            st.sidebar.error(f"Auto-load failed: {e}")

# -------- Utilities to GUARANTEE _cohort/_item_id --------
def make_item_id(item: dict) -> str:
    dset_sig = st.session_state.dataset_hash or ""
    payload = json.dumps(
        {
            "ds": dset_sig,
            "i": item.get("index"),
            "q": item.get("question",""),
            "a": item.get("answer",""),
            "c": item.get("context",""),
        },
        sort_keys=True, ensure_ascii=False
    )
    return sha1_text(payload)

def ensure_decorated(item: dict) -> dict:
    it = dict(item)
    iid = it.get("_item_id") or make_item_id(it)
    it["_item_id"] = iid
    dset_sig = st.session_state.dataset_hash or ""
    it["_cohort"] = it.get("_cohort") or stable_partition(iid, dset_sig)
    return it

def pick_filtered_item(filtered: List[dict]) -> Optional[dict]:
    if not filtered: return None
    target_idx = st.session_state.i
    for it in filtered:
        if it["index"] == target_idx:
            return ensure_decorated(it)
    st.session_state.i = filtered[0]["index"]
    return ensure_decorated(filtered[0])

# ---------------- Metrics ----------------
LABELS = ["Correct","Partially correct","Incorrect","Unclear"]
label_to_idx = {s:i for i,s in enumerate(LABELS)}

def cohens_kappa(y_true: List[int], y_pred: List[int]) -> float:
    n=len(y_true); 
    if n==0: return float("nan")
    K=len(LABELS); M=[[0]*K for _ in range(K)]
    for t,p in zip(y_true,y_pred):
        if 0<=t<K and 0<=p<K: M[t][p]+=1
    po = sum(M[i][i] for i in range(K))/n
    row=[sum(M[i][j] for j in range(K))/n for i in range(K)]
    col=[sum(M[i][j] for i in range(K))/n for j in range(K)]
    pe = sum(row[k]*col[k] for k in range(K))
    return (po-pe)/(1-pe) if (1-pe)!=0 else float("nan")

def consistency_report(conn, human_id: str):
    df_h = read_all_feedback(conn, human_id)
    df_ai = read_all_feedback(conn, AI_VALIDATOR_ID)
    if df_h.empty or df_ai.empty: return pd.DataFrame(), {}
    h = df_h[["item_id","item_index","correctness"]].rename(columns={"correctness":"human"})
    a = df_ai[["item_id","correctness"]].rename(columns={"correctness":"ai"})
    m = pd.merge(h,a,on="item_id",how="inner")
    if m.empty: return pd.DataFrame(), {}
    y_true=[label_to_idx.get(x,3) for x in m["human"].tolist()]
    y_pred=[label_to_idx.get(x,3) for x in m["ai"].tolist()]
    kappa = cohens_kappa(y_true,y_pred)
    agree = sum(int(t==p) for t,p in zip(y_true,y_pred))/len(y_true)
    K=len(LABELS); M=[[0]*K for _ in range(K)]
    for t,p in zip(y_true,y_pred):
        if 0<=t<K and 0<=p<K: M[t][p]+=1
    conf = pd.DataFrame(M, index=["H_"+s for s in LABELS], columns=["AI_"+s for s in LABELS])
    return conf, {"n_overlap":len(m),"agreement":agree,"cohens_kappa":kappa}

# ---------------- AI routines ----------------
def ai_validate_and_save(conn, client: GeminiClient, item: dict, item_id: str, idx: int, dset_hash: str):
    result = client.judge(item.get("context",""), item.get("question",""), item.get("answer",""))
    row = {
        "item_id": item_id,
        "item_index": idx,
        "dataset_hash": dset_hash or "",
        "timestamp": datetime.utcnow().isoformat(timespec="seconds")+"Z",
        "validator_id": AI_VALIDATOR_ID,
        "correctness": result.get("label","Unclear"),
        "suggested_answer": result.get("suggested_answer", item.get("answer","")),
        "comments": result.get("justification",""),
        "tags": "AI",
        "original_question": item.get("question",""),
        "original_answer": item.get("answer",""),
        "original_context": item.get("context",""),
    }
    upsert_feedback(conn, row)

# ---------------- App ----------------
def main():
    init_state()
    st.title("WYDOT SPECS Q&A Validator ✅")
    st.caption("80/20 Human–AI split, Gemini auto-validation, and consistency checks.")

    with st.sidebar:
        st.header("Setup")
        st.session_state.db_path = st.text_input("SQLite DB file", value=st.session_state.db_path)
        if st.session_state.conn is None:
            st.session_state.conn = get_conn(st.session_state.db_path)

        st.session_state.validator_id = st.text_input("Your Validator ID", value=st.session_state.validator_id)

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

        autoload_dataset_if_needed()

        total = len(st.session_state.norm)
        if total:
            dset_sig = st.session_state.dataset_hash or ""
            ids = [make_item_id(x) for x in st.session_state.norm]
            parts = [stable_partition(iid, dset_sig) for iid in ids]
            ai_ct = sum(1 for p in parts if p=="AI")
            st.info(f"Cohorts → AI: {ai_ct} | Human: {total-ai_ct} (total: {total})")

        st.divider()
        st.subheader("View mode")
        st.session_state.view_mode = st.radio("Show items assigned to…", ["HUMAN","AI","ALL"], index=0)

        st.divider()
        st.subheader("AI Validation")
        client = st.session_state.ai_client
        if client.ready: st.caption(f"AI client ready [{client.path}] — model={a_MODEL}")
        else:            st.warning(f"AI client not ready: {client.err or 'no SDK'}")

        if st.button("▶ Run AI on the 80% AI cohort") and st.session_state.norm:
            norm = st.session_state.norm
            dset_sig = st.session_state.dataset_hash or ""
            existing_ai = read_all_feedback(st.session_state.conn, AI_VALIDATOR_ID)
            have_ai = set(existing_ai["item_id"].tolist()) if not existing_ai.empty else set()
            prog = st.progress(0.0, text="AI validating…")
            done = 0; n = len(norm)
            for it in norm:
                iid = make_item_id(it)
                if stable_partition(iid, dset_sig) != "AI":
                    done+=1; prog.progress(done/n, text=f"Skipping human cohort: {done}/{n}"); continue
                if iid in have_ai:
                    done+=1; prog.progress(done/n, text=f"Already AI-labeled: {done}/{n}"); continue
                ai_validate_and_save(st.session_state.conn, client, it, iid, it["index"], dset_sig)
                done+=1; prog.progress(done/n, text=f"Processed: {done}/{n}")
            st.success("AI validation pass complete.")

        if st.session_state.validator_id and st.button("▶ Run AI on my HUMAN-reviewed items"):
            human_df = read_all_feedback(st.session_state.conn, st.session_state.validator_id)
            if human_df.empty:
                st.info("No human reviews found yet.")
            else:
                norm_by_id = {make_item_id(it): it for it in st.session_state.norm}
                dset_sig = st.session_state.dataset_hash or ""
                existed_ai = read_all_feedback(st.session_state.conn, AI_VALIDATOR_ID)
                have_ai = set(existed_ai["item_id"].tolist()) if not existed_ai.empty else set()
                rows = human_df.to_dict("records")
                prog = st.progress(0.0, text="AI cross-checking your reviewed items…")
                for i, r in enumerate(rows, 1):
                    iid = r["item_id"]
                    if iid not in norm_by_id or iid in have_ai:
                        prog.progress(i/len(rows)); continue
                    it = norm_by_id[iid]
                    ai_validate_and_save(st.session_state.conn, client, it, iid, it["index"], dset_sig)
                    prog.progress(i/len(rows))
                st.success("AI cross-checking finished.")

        st.divider()
        # ---------- Exports ----------
        st.subheader("Exports")

        def _dl_csv_button(label, df, fname):
            if df is None or df.empty:
                st.caption(f"{label}: no rows yet.")
                return
            st.download_button(label, data=df.to_csv(index=False).encode("utf-8"),
                            file_name=fname, mime="text/csv", use_container_width=True)

        # 1) AI-only
        df_ai = read_all_feedback(st.session_state.conn, AI_VALIDATOR_ID)
        _dl_csv_button("Download AI-only CSV", df_ai, "ai_feedback.csv")

        # 2) HUMAN-only (current validator)
        df_human = None
        if st.session_state.validator_id.strip():
            df_human = read_all_feedback(st.session_state.conn, st.session_state.validator_id.strip())
        _dl_csv_button("Download HUMAN-only CSV", df_human, "human_feedback.csv")

        # 3) AI vs HUMAN overlap rows + 4) confusion matrix + stats
        def build_overlap_and_conf(conn, human_id: str):
            if not human_id: return None, None, None
            df_h = read_all_feedback(conn, human_id)
            df_a = read_all_feedback(conn, AI_VALIDATOR_ID)
            if df_h.empty or df_a.empty: return None, None, None

            h = df_h[["item_id","item_index","correctness","original_question","original_answer","original_context"]] \
                    .rename(columns={"correctness":"human"})
            a = df_a[["item_id","correctness","suggested_answer","comments"]] \
                    .rename(columns={"correctness":"ai","comments":"ai_notes"})

            m = pd.merge(h, a, on="item_id", how="inner")
            if m.empty: return None, None, None
            m["agree"] = (m["human"] == m["ai"]).astype(int)

            # confusion + stats (reuse your functions)
            y_true = [label_to_idx.get(x,3) for x in m["human"].tolist()]
            y_pred = [label_to_idx.get(x,3) for x in m["ai"].tolist()]
            kappa = cohens_kappa(y_true, y_pred)
            agree = float(sum(int(t==p) for t,p in zip(y_true,y_pred)))/len(y_true)
            K=len(LABELS); M=[[0]*K for _ in range(K)]
            for t,p in zip(y_true,y_pred):
                if 0<=t<K and 0<=p<K: M[t][p]+=1
            conf = pd.DataFrame(M, index=["H_"+s for s in LABELS], columns=["AI_"+s for s in LABELS])
            stats = {"n_overlap": int(len(m)), "agreement": agree, "cohens_kappa": float(kappa)}
            return m, conf, stats

        overlap_df, conf_df, stats = (None, None, None)
        if st.session_state.validator_id.strip():
            overlap_df, conf_df, stats = build_overlap_and_conf(st.session_state.conn, st.session_state.validator_id.strip())

        _dl_csv_button("Download AI↔HUMAN Overlap CSV", overlap_df, "ai_vs_human_overlap.csv")

        # confusion matrix CSV
        if conf_df is not None and not conf_df.empty:
            st.download_button("Download Confusion Matrix CSV",
                data=conf_df.to_csv().encode("utf-8"),
                file_name="ai_vs_human_confusion.csv", mime="text/csv", use_container_width=True)

        # stats JSON
        if stats is not None:
            st.download_button("Download Stats (JSON)",
                data=json.dumps(stats, indent=2).encode("utf-8"),
                file_name="ai_vs_human_stats.json", mime="application/json", use_container_width=True)


        st.caption("Tip: Use ←/→ below to navigate. Click **Save** to write to the DB.")

    # ------------- Stop if no data -------------
    if not st.session_state.norm:
        st.info("Load a JSONL dataset to start reviewing.")
        st.stop()

    # ------------- Build filtered list (decorate) -------------
    dset_sig = st.session_state.dataset_hash or ""
    filtered: List[dict] = []
    for it in st.session_state.norm:
        iid = make_item_id(it)
        cohort = stable_partition(iid, dset_sig)
        if st.session_state.view_mode == "ALL" or cohort == st.session_state.view_mode:
            decorated = dict(it)
            decorated["_item_id"] = iid
            decorated["_cohort"]  = cohort
            filtered.append(decorated)

    if not filtered:
        st.info("No items in this view.")
        st.stop()

    # Map global index -> filtered position
    global_to_filtered = {it["index"]: k for k, it in enumerate(filtered)}

    # ------------- Pick current filtered item (guaranteed decorated) -------------
    item = pick_filtered_item(filtered)
    idx  = item["index"]

    # ------------- Navigation -------------
    cols = st.columns([1,1,1,6])
    with cols[0]:
        if st.button("⟵ Prev", use_container_width=True):
            pos = global_to_filtered[item["index"]]
            pos = max(0, pos-1)
            st.session_state.i = filtered[pos]["index"]
            st.rerun()
    with cols[1]:
        if st.button("Next ⟶", use_container_width=True):
            pos = global_to_filtered[item["index"]]
            pos = min(len(filtered)-1, pos+1)
            st.session_state.i = filtered[pos]["index"]
            st.rerun()
    with cols[2]:
        cur_pos = global_to_filtered[item["index"]]
        st.number_input("Jump to position (filtered)", min_value=0, max_value=len(filtered)-1,
                        value=cur_pos, key="jump_idx", step=1)
        if st.button("Go"):
            st.session_state.i = filtered[st.session_state.jump_idx]["index"]
            st.rerun()
    with cols[3]:
        st.progress((global_to_filtered[item["index"]]+1)/max(1,len(filtered)),
                    text=f"Item {global_to_filtered[item['index']]+1} / {len(filtered)} (view: {st.session_state.view_mode})")

    # ------------- Display item -------------
    cohort = item.get("_cohort") or stable_partition(item.get("_item_id") or make_item_id(item), dset_sig)
    st.subheader(f"Item #{idx} — Cohort: {cohort}")

    with st.expander("Context", expanded=True):
        raw_ctx = item.get("context","")
        ctx = normalize_whitespace_compact(remove_noise_headers(raw_ctx))
        ctx_text, ctx_tables = extract_tables_from_text(ctx)
        MAX_CHARS=3000; truncated=False
        if len(ctx_text)>MAX_CHARS:
            ctx_text = ctx_text[:MAX_CHARS].rstrip()+" …"; truncated=True
        st.markdown(f"<div style='white-space: pre-wrap'>{ctx_text}</div>", unsafe_allow_html=True)
        if truncated: st.caption("Context truncated for compact view.")
        for df in ctx_tables: st.table(df)

    qcol, acol = st.columns([1,1])
    with qcol:
        st.markdown("### Question")
        st.markdown(f"<div style='white-space: pre-wrap'>{item['question']}</div>", unsafe_allow_html=True)
    with acol:
        st.markdown("### Given Answer (to validate)")
        st.markdown(f"<div style='white-space: pre-wrap; border-left:4px solid #ddd; padding-left:10px;'>{item['answer']}</div>", unsafe_allow_html=True)

    # ------------- Per-item AI check -------------
    left, right = st.columns([1,1])
    with left:
        if st.button("🤖 AI check this item"):
            iid = item.get("_item_id") or make_item_id(item)
            ai_validate_and_save(st.session_state.conn, st.session_state.ai_client, item, iid, item["index"], dset_sig)
            st.success("AI validation saved for this item.")
    with right:
        df_ai_item = read_all_feedback(st.session_state.conn, AI_VALIDATOR_ID)
        if not df_ai_item.empty:
            row = df_ai_item[df_ai_item["item_id"] == (item.get("_item_id") or make_item_id(item))]
            if not row.empty:
                r = row.iloc[-1]
                st.markdown(f"**AI label:** {r['correctness']}  ")
                if str(r.get("comments","")).strip():
                    st.caption(f"AI notes: {r['comments'][:500]}")

    st.markdown("---")
    st.markdown("### Your Review")
    default_idx=3
    try:
        a_low = (item["answer"] or "").lower()
        default_idx = 0 if "correct" in a_low else (1 if "part" in a_low else 3)
    except Exception: pass

    correctness = st.radio("Label", options=LABELS, index=default_idx, horizontal=True, key=f"correctness_{idx}")
    suggested_answer = st.text_area("Suggested / Corrected Answer (optional)", value=item.get("answer",""),
                                    height=120, key=f"suggested_{idx}")
    tags = st.multiselect("Quick tags (optional)",
                          options=["Ambiguous question","Insufficient context","Hallucination","Formatting issue","Typos","Policy concern"],
                          default=[], key=f"tags_{idx}")
    comments = st.text_area("Comments (optional)", key=f"comments_{idx}")

    save_col, skip_col = st.columns([1,1])
    with save_col:
        if st.button("💾 Save", type="primary"):
            if not st.session_state.validator_id.strip():
                st.warning("Please enter your Validator ID in the sidebar first.")
            else:
                iid = item.get("_item_id") or make_item_id(item)
                row = {
                    "item_id": iid,
                    "item_index": idx,
                    "dataset_hash": dset_sig,
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds")+"Z",
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
            pos = global_to_filtered[item["index"]]
            pos = min(len(filtered)-1, pos+1)
            st.session_state.i = filtered[pos]["index"]
            st.rerun()

    # ------------- Footer + Consistency -------------
    st.markdown("---")
    if st.session_state.validator_id:
        df_me = read_all_feedback(st.session_state.conn, st.session_state.validator_id)
        st.caption(f"Your saved reviews: {len(df_me)}")
        if not df_me.empty:
            st.dataframe(df_me[["item_index","correctness","tags","timestamp","comments"]]
                         .sort_values("item_index"), use_container_width=True)

    st.markdown("---")
    st.subheader("🤝 Human vs AI — Consistency Dashboard")
    if st.session_state.validator_id:
        conf, stats = consistency_report(st.session_state.conn, st.session_state.validator_id)
        if not conf.empty:
            st.markdown("**Confusion matrix (rows=Human, cols=AI):**")
            st.table(conf)
            st.markdown(f"**Overlap:** {stats['n_overlap']}  |  **Agreement:** {stats['agreement']*100:.1f}%  |  **Cohen's κ:** {stats['cohens_kappa']:.3f}")
        else:
            st.caption("No overlapping items yet. Save some human reviews and/or run the AI checks.")

if __name__ == "__main__":
    main()
