# import os
# import json
# import re
# import streamlit as st
# from dotenv import load_dotenv

# import google.generativeai as genai
# from google.generativeai.types import HarmCategory, HarmBlockThreshold

# from pymilvus import connections, Collection, utility, DataType

# # =========================
# # PAGE SETUP
# # =========================
# st.set_page_config(page_title="WYDOT Assistant (Advanced RAG)", layout="wide")

# # =========================
# # CONFIG
# # =========================
# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
# MILVUS_URI = os.getenv("MILVUS_URI")
# MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
# MILVUS_COLLECTION = "metadata_specs"

# EMBED_MODEL = "models/text-embedding-004"
# TARGET_DIM = 768

# ROUTER_MODEL = "gemini-2.5-flash"
# GEN_MODEL = "gemini-2.0-flash"

# # =========================
# # SAFETY SETTINGS (avoid empty responses)
# # =========================
# SAFETY_SETTINGS = {
#     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
# }

# def ensure_genai():
#     if not GOOGLE_API_KEY:
#         st.error("❌ GEMINI_API_KEY missing in environment (.env).")
#         st.stop()
#     genai.configure(api_key=GOOGLE_API_KEY)

# # =========================
# # EMBEDDING
# # =========================
# def embed_query(text: str) -> list:
#     ensure_genai()
#     try:
#         res = genai.embed_content(
#             model=EMBED_MODEL,
#             content=text,
#             task_type="retrieval_query"
#         )
#         emb = res["embedding"]
#         if len(emb) != TARGET_DIM:
#             st.warning(f"⚠️ Embedding dim mismatch: got {len(emb)}, expected {TARGET_DIM}")
#         return emb
#     except Exception as e:
#         st.error(f"⚠️ Embedding failed: {e}")
#         return [0.0] * TARGET_DIM

# # =========================
# # MILVUS CONNECTION
# # =========================
# def connect_milvus():
#     if not MILVUS_URI or not MILVUS_TOKEN:
#         st.error("❌ MILVUS_URI / MILVUS_TOKEN missing.")
#         st.stop()
#     try:
#         connections.connect(
#             alias="default",
#             uri=MILVUS_URI,
#             token=MILVUS_TOKEN,
#             secure=True
#         )
#     except Exception as e:
#         st.error(f"❌ Milvus Connection Error: {e}")
#         st.stop()

# def get_collection() -> Collection:
#     connect_milvus()

#     if not utility.has_collection(MILVUS_COLLECTION):
#         st.error(f"❌ Collection '{MILVUS_COLLECTION}' not found. Did you run the ingestion script?")
#         st.stop()

#     col = Collection(MILVUS_COLLECTION)
#     col.load()
#     return col

# def pick_text_field(col: Collection) -> str | None:
#     names = [f.name for f in col.schema.fields]

#     for candidate in ["text", "content", "chunk", "doc_text", "passage"]:
#         if candidate in names:
#             return candidate

#     for f in col.schema.fields:
#         if f.dtype in (DataType.VARCHAR, DataType.STRING):
#             return f.name

#     return None

# def existing_field_set(col: Collection) -> set[str]:
#     return {f.name for f in col.schema.fields}

# # =========================
# # ROUTER
# # =========================
# def router_agent(query: str, available_years: list[int]) -> dict:
#     ensure_genai()
#     model = genai.GenerativeModel(ROUTER_MODEL)

#     yrs = sorted(set(available_years))
#     default_year = max(yrs) if yrs else 2023

#     prompt = f"""
# You are a Router for a WYDOT RAG system.
# Available report years in the database: {yrs}

# Classify the user query into one of:
# - CURRENT: asking about the latest year or present status(if the content is related to specs latest year is 2021 and if it is related to annual report latest year is 2023)
# - HISTORY: asking about a past year (not the latest)
# - COMPARE: comparing two or more years / asking for change or trends

# Also decide which years to use:
# - If the user specifies year(s), use them (only from available years).
# - If COMPARE and user doesn't specify years, use the latest two available years.
# - If CURRENT with no year specified, use the latest year.
# - If HISTORY with no year specified, use the earliest year.

# Return ONLY JSON:
# {{"route":"CURRENT"|"HISTORY"|"COMPARE","years":[YYYY,...]}}

# User query: {query}
# """
#     try:
#         res = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)
#         clean = res.text.replace("```json", "").replace("```", "").strip()
#         data = json.loads(clean)

#         route = data.get("route", "CURRENT")
#         years = [int(y) for y in data.get("years", []) if int(y) in yrs]

#         if route == "COMPARE" and len(years) < 2:
#             years = yrs[-2:] if len(yrs) >= 2 else [default_year]
#         elif route == "CURRENT" and len(years) == 0:
#             years = [default_year]
#         elif route == "HISTORY" and len(years) == 0:
#             years = [min(yrs)] if yrs else [default_year]

#         return {"route": route, "years": years}
#     except Exception:
#         return {"route": "CURRENT", "years": [default_year]}

# # =========================
# # DEBUG HELPERS (clickable citations -> debug anchors)
# # =========================
# def _safe_anchor_id(chunk_id: str) -> str:
#     return re.sub(r"[^A-Za-z0-9_-]", "_", str(chunk_id))

# def build_debug_context(metas: list[dict], docs: list[str], max_chars: int = 900) -> str:
#     if not metas or not docs:
#         return "### Retrieved chunks\nNone"

#     jump_links = []
#     for i, m in enumerate(metas, start=1):
#         cid = m.get("chunk_id", f"missing_{i}")
#         anchor = _safe_anchor_id(cid)
#         jump_links.append(f"[S{i}](#CID_{anchor})")

#     out = []
#     out.append("### Chunk jump links")
#     out.append(" ".join(jump_links))
#     out.append("\n---\n")
#     out.append("### Retrieved chunks (chunk_id + preview)")

#     for i, (m, d) in enumerate(zip(metas, docs), start=1):
#         cid = m.get("chunk_id", f"missing_{i}")
#         anchor = _safe_anchor_id(cid)

#         file_name = m.get("file_name", "N/A")
#         year = m.get("year", "N/A")
#         page = m.get("page", "N/A")
#         score = m.get("score", 0.0)

#         snippet = (d or "").strip()
#         more = ""
#         if len(snippet) > max_chars:
#             snippet = snippet[:max_chars]
#             more = "…"

#         out.append(f'<a id="CID_{anchor}"></a>')
#         out.append(f"#### S{i} — chunk_id: `{cid}` — {file_name} (year {year}, page {page}, score {score:.4f})")

#         if m.get("image_path") and m["image_path"] != "N/A":
#             out.append(f"- image_path: `{m['image_path']}`")

#         out.append(f"> {snippet}{more}\n")

#     return "\n".join(out)

# def linkify_citations(text: str, metas: list[dict]) -> str:
#     out = text or ""
#     for i, m in enumerate(metas, start=1):
#         cid = m.get("chunk_id", f"missing_{i}")
#         anchor = _safe_anchor_id(cid)
#         out = out.replace(f"[S{i}]", f"[S{i}](#CID_{anchor})")
#     return out

# # =========================
# # KEYWORD FILTER + RERANK (NEW)
# # =========================
# def _tokenize(s: str) -> list[str]:
#     s = re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())
#     return [t for t in s.split() if len(t) >= 3]

# def keyword_score(query: str, text: str) -> int:
#     q = set(_tokenize(query))
#     t = set(_tokenize(text))
#     return len(q.intersection(t))

# def filter_and_rerank(query: str, docs: list[str], metas: list[dict], min_kw: int = 1, topk: int = 5):
#     scored = []
#     for d, m in zip(docs, metas):
#         s = keyword_score(query, d or "")
#         scored.append((s, d, m))

#     # primary: keyword overlap, secondary: vector score (hit.score)
#     scored.sort(key=lambda x: (x[0], x[2].get("score", 0.0)), reverse=True)

#     kept = [(d, m, s) for s, d, m in scored if s >= min_kw][:topk]
#     filt_docs = [d for d, _, _ in kept]
#     filt_metas = [m for _, m, _ in kept]
#     filt_scores = [s for _, _, s in kept]
#     return filt_docs, filt_metas, filt_scores

# # =========================
# # MILVUS SEARCH
# # =========================
# def search_milvus(query_text: str, year: int | None, limit: int = 5):
#     col = get_collection()

#     text_field = pick_text_field(col)
#     if not text_field:
#         st.error("❌ Could not find any VARCHAR/STRING field to use as document text.")
#         return [], [], {"schema": [(f.name, str(f.dtype)) for f in col.schema.fields]}

#     existing = existing_field_set(col)

#     desired = [text_field, "file_name", "year", "page", "image_path"]
#     output_fields = [f for f in desired if f in existing]

#     query_vector = embed_query(query_text)

#     expr = None
#     if year is not None and "year" in existing:
#         expr = f"year == {int(year)}"

#     search_params = {"metric_type": "COSINE", "params": {}}

#     try:
#         results = col.search(
#             data=[query_vector],
#             anns_field="embedding",
#             param=search_params,
#             limit=limit,
#             expr=expr,
#             output_fields=output_fields
#         )
#     except Exception as e:
#         st.error(f"❌ Search Error: {e}")
#         return [], [], {"schema": [(f.name, str(f.dtype)) for f in col.schema.fields]}

#     if not results or len(results[0]) == 0:
#         return [], [], {"schema": [(f.name, str(f.dtype)) for f in col.schema.fields]}

#     docs, metas = [], []
#     for hit in results[0]:
#         docs.append(hit.entity.get(text_field))
#         metas.append({
#             "chunk_id": str(hit.id),
#             "file_name": hit.entity.get("file_name"),
#             "year": hit.entity.get("year"),
#             "page": hit.entity.get("page"),
#             "image_path": hit.entity.get("image_path"),
#             "score": float(hit.score),
#         })

#     debug = {"schema": [(f.name, str(f.dtype)) for f in col.schema.fields], "text_field": text_field, "expr": expr}
#     return docs, metas, debug

# # =========================
# # ANSWER GENERATION
# # =========================
# def grounded_answer(query: str, metas: list[dict], docs: list[str], mode: str):
#     ensure_genai()
#     model = genai.GenerativeModel(GEN_MODEL)

#     chunks = []
#     for i, (m, d) in enumerate(zip(metas, docs), start=1):
#         src = (
#             f"[S{i}] (chunk_id={m.get('chunk_id')}) "
#             f"{m.get('file_name')} (year {m.get('year')}, page {m.get('page')}, score {m.get('score'):.4f})"
#         )
#         chunks.append(f"{src}\n{d}")

#     context = "\n\n".join(chunks)

#     prompt = f"""
# You are a WYDOT document assistant. Answer ONLY using the provided sources.
# Rules:
# - If the answer is not present in the sources, say exactly: "I couldn't find that information in the documents."
# - When you state a fact, add citations like [S1], [S2].
# - If the user asks for comparison/trends, explicitly compare and cite both sides.
# - Be concise.

# Mode: {mode}

# Sources:
# {context}

# Question: {query}
# """
#     try:
#         res = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)
#         return linkify_citations(res.text, metas)
#     except Exception as e:
#         return f"Error generating answer: {e}"

# # =========================
# # SUPERVISOR (multi-year compare)
# # =========================
# def supervisor_compare(query: str, years: list[int]):
#     all_docs = []
#     all_metas = []
#     all_images = []

#     for y in years:
#         docs, metas, _ = search_milvus(query, year=y, limit=8)
#         for m in metas:
#             if m.get("image_path") and m["image_path"] != "N/A":
#                 all_images.append(m["image_path"])
#         all_docs.extend(docs)
#         all_metas.extend(metas)

#     if not all_docs:
#         return "I couldn't find that information in the documents.", [], "No context found."

#     answer = grounded_answer(query, all_metas, all_docs, mode=f"COMPARE {years}")
#     dbg = "\n".join([
#         f"Compare years: {years}, chunks: {len(all_docs)}",
#         build_debug_context(all_metas, all_docs)
#     ])
#     return answer, sorted(set(all_images)), dbg

# # =========================
# # SIDEBAR
# # =========================
# with st.sidebar:
#     st.title("WYDOT Intelligent RAG (Milvus)")
#     st.markdown("### Architecture")
#     st.info("1) Router → 2) Experts (year filtered) → 3) Supervisor (compare)")

#     st.divider()
#     st.caption("Config loaded from .env")
#     st.write("Collection:", MILVUS_COLLECTION)

# # =========================
# # UI STATE
# # =========================
# st.title(" WYDOT Assistant")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"], unsafe_allow_html=True)

#         if msg.get("images"):
#             with st.expander("📊 View referenced charts/images"):
#                 for img in msg["images"]:
#                     if os.path.exists(img):
#                         st.image(img, caption="Reference", width=650)
#                     else:
#                         st.caption(f"Image referenced but not found locally: {img}")

#         if msg.get("debug"):
#             with st.expander("Sources(route + years + filter + context preview)"):
#                 st.markdown(msg["debug"], unsafe_allow_html=True)

# # =========================
# # MAIN INPUT
# # =========================
# if prompt := st.chat_input("Ask about WYDOT specs or reports..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         with st.spinner("Routing + retrieving..."):
#             available_years = [2022, 2023, 2021, 2010]

#             route_data = router_agent(prompt, available_years)
#             route = route_data["route"]
#             years = route_data["years"]

#             images = []
#             debug_lines = [f"Route: {route}", f"Years: {years}"]

#             if route == "COMPARE":
#                 response_text, images, dbg = supervisor_compare(prompt, years)
#                 debug_lines.append(dbg)

#                 valid_images = []
#                 for img_path in images:
#                     if os.path.exists(img_path):
#                         valid_images.append(img_path)

#             else:
#                 target_year = years[0] if years else max(available_years)

#                 # ✅ NEW: Always get raw vector hits first
#                 raw_docs, raw_metas, debug = search_milvus(prompt, year=target_year, limit=15)

#                 debug_lines.append(f"Year filter expr: {debug.get('expr')}")
#                 debug_lines.append(f"Text field: {debug.get('text_field')}")
#                 debug_lines.append(f"Raw vector hits: {len(raw_docs)}")

#                 # ✅ NEW: keyword filter + rerank
#                 docs, metas, kw_scores = filter_and_rerank(prompt, raw_docs, raw_metas, min_kw=1, topk=5)
#                 debug_lines.append(f"Keyword overlap scores (kept): {kw_scores}")

#                 if not docs:
#                     # ✅ NEW behavior when nothing keyword-matches
#                     response_text = (
#                         "I couldn't find that information in the documents.\n\n"
#                         "**I didn’t find an exact match, but here is related information from the closest excerpts:**"
#                     )
#                     debug_lines.append("No keyword-matching chunks. Showing nearest vector hits below:")
#                     debug_lines.append(build_debug_context(raw_metas[:5], raw_docs[:5]))
#                     images = [m["image_path"] for m in raw_metas[:5] if m.get("image_path") and m["image_path"] != "N/A"]
#                     images = sorted(set(images))
#                 else:
#                     response_text = grounded_answer(prompt, metas, docs, mode=f"{route} {target_year}")
#                     debug_lines.append(build_debug_context(metas, docs))

#                     images = [m["image_path"] for m in metas if m.get("image_path") and m["image_path"] != "N/A"]
#                     images = sorted(set(images))

#                 valid_images = []
#                 for img_path in images:
#                     if os.path.exists(img_path):
#                         valid_images.append(img_path)

#         st.markdown(response_text, unsafe_allow_html=True)

#         if images:
#             with st.expander("📊 Click to view relevant charts/images"):
#                 for img_path in images:
#                     if os.path.exists(img_path):
#                         st.image(img_path, caption=f"Source: {img_path}", width=650)
#                     else:
#                         st.caption(f"Image referenced but file not found: {img_path}")

#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": response_text,
#         "images": valid_images,
#         "debug": "\n".join(debug_lines)
#     })
import os
import json
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from pymilvus import connections, Collection, utility, DataType

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="WYDOT Assistant (Advanced RAG)", layout="wide")

# =========================
# CONFIG
# =========================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_COLLECTION = "metadata_specs"

EMBED_MODEL = "models/text-embedding-004"
TARGET_DIM = 768

ROUTER_MODEL = "gemini-2.5-flash"
GEN_MODEL = "gemini-2.0-flash"

# =========================
# PATHS (for images inside repo/container)
# =========================
APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "output_images"

def resolve_image_path(p: str) -> str:
    """
    Converts whatever is stored in Milvus (Windows paths, relative paths, etc.)
    into a real path inside this running app: /app/output_images/<filename>.
    """
    if not p:
        return ""
    p = str(p).strip().replace("\\", "/")     # windows -> posix
    fname = Path(p).name                     # keep only filename
    return str((OUTPUT_DIR / fname).resolve())

# =========================
# SAFETY SETTINGS (avoid empty responses)
# =========================
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

def ensure_genai():
    if not GOOGLE_API_KEY:
        st.error("❌ GEMINI_API_KEY missing in environment (.env).")
        st.stop()
    genai.configure(api_key=GOOGLE_API_KEY)

# =========================
# EMBEDDING
# =========================
def embed_query(text: str) -> list:
    ensure_genai()
    try:
        res = genai.embed_content(
            model=EMBED_MODEL,
            content=text,
            task_type="retrieval_query"
        )
        emb = res["embedding"]
        if len(emb) != TARGET_DIM:
            st.warning(f"⚠️ Embedding dim mismatch: got {len(emb)}, expected {TARGET_DIM}")
        return emb
    except Exception as e:
        st.error(f"⚠️ Embedding failed: {e}")
        return [0.0] * TARGET_DIM

# =========================
# MILVUS CONNECTION
# =========================
def connect_milvus():
    if not MILVUS_URI or not MILVUS_TOKEN:
        st.error("❌ MILVUS_URI / MILVUS_TOKEN missing.")
        st.stop()
    try:
        connections.connect(
            alias="default",
            uri=MILVUS_URI,
            token=MILVUS_TOKEN,
            secure=True
        )
    except Exception as e:
        st.error(f"❌ Milvus Connection Error: {e}")
        st.stop()

def get_collection() -> Collection:
    connect_milvus()

    if not utility.has_collection(MILVUS_COLLECTION):
        st.error(f"❌ Collection '{MILVUS_COLLECTION}' not found. Did you run the ingestion script?")
        st.stop()

    col = Collection(MILVUS_COLLECTION)
    col.load()
    return col

def pick_text_field(col: Collection) -> str | None:
    names = [f.name for f in col.schema.fields]

    for candidate in ["text", "content", "chunk", "doc_text", "passage"]:
        if candidate in names:
            return candidate

    for f in col.schema.fields:
        if f.dtype in (DataType.VARCHAR, DataType.STRING):
            return f.name

    return None

def existing_field_set(col: Collection) -> set[str]:
    return {f.name for f in col.schema.fields}

# =========================
# ROUTER
# =========================
def router_agent(query: str, available_years: list[int]) -> dict:
    ensure_genai()
    model = genai.GenerativeModel(ROUTER_MODEL)

    yrs = sorted(set(available_years))
    default_year = max(yrs) if yrs else 2023

    prompt = f"""
You are a Router for a WYDOT RAG system.
Available report years in the database: {yrs}

Classify the user query into one of:
- CURRENT: asking about the latest year or present status (if related to specs latest is 2021; if annual report latest is 2023)
- HISTORY: asking about a past year (not the latest)
- COMPARE: comparing two or more years / asking for change or trends

Also decide which years to use:
- If the user specifies year(s), use them (only from available years).
- If COMPARE and user doesn't specify years, use the latest two available years.
- If CURRENT with no year specified, use the latest year.
- If HISTORY with no year specified, use the earliest year.

Return ONLY JSON:
{{"route":"CURRENT"|"HISTORY"|"COMPARE","years":[YYYY,...]}}

User query: {query}
"""
    try:
        res = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)
        clean = res.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)

        route = data.get("route", "CURRENT")
        years = [int(y) for y in data.get("years", []) if int(y) in yrs]

        if route == "COMPARE" and len(years) < 2:
            years = yrs[-2:] if len(yrs) >= 2 else [default_year]
        elif route == "CURRENT" and len(years) == 0:
            years = [default_year]
        elif route == "HISTORY" and len(years) == 0:
            years = [min(yrs)] if yrs else [default_year]

        return {"route": route, "years": years}
    except Exception:
        return {"route": "CURRENT", "years": [default_year]}

# =========================
# DEBUG HELPERS (clickable citations -> anchors)
# =========================
def _safe_anchor_id(chunk_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(chunk_id))

def build_debug_context(metas: list[dict], docs: list[str], max_chars: int = 900) -> str:
    if not metas or not docs:
        return "### Retrieved chunks\nNone"

    jump_links = []
    for i, m in enumerate(metas, start=1):
        cid = m.get("chunk_id", f"missing_{i}")
        anchor = _safe_anchor_id(cid)
        jump_links.append(f"[S{i}](#CID_{anchor})")

    out = []
    out.append("### Chunk jump links")
    out.append(" ".join(jump_links))
    out.append("\n---\n")
    out.append("### Retrieved chunks (chunk_id + preview)")

    for i, (m, d) in enumerate(zip(metas, docs), start=1):
        cid = m.get("chunk_id", f"missing_{i}")
        anchor = _safe_anchor_id(cid)

        file_name = m.get("file_name", "N/A")
        year = m.get("year", "N/A")
        page = m.get("page", "N/A")
        score = m.get("score", 0.0)

        snippet = (d or "").strip()
        more = ""
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars]
            more = "…"

        out.append(f'<a id="CID_{anchor}"></a>')
        out.append(f"#### S{i} — chunk_id: `{cid}` — {file_name} (year {year}, page {page}, score {score:.4f})")

        if m.get("image_path") and m["image_path"] != "N/A":
            out.append(f"- image_path(raw): `{m['image_path']}`")
            out.append(f"- image_path(resolved): `{resolve_image_path(m['image_path'])}`")

        out.append(f"> {snippet}{more}\n")

    return "\n".join(out)

def linkify_citations(text: str, metas: list[dict]) -> str:
    out = text or ""
    for i, m in enumerate(metas, start=1):
        cid = m.get("chunk_id", f"missing_{i}")
        anchor = _safe_anchor_id(cid)
        out = out.replace(f"[S{i}]", f"[S{i}](#CID_{anchor})")
    return out

# =========================
# KEYWORD FILTER + RERANK
# =========================
def _tokenize(s: str) -> list[str]:
    s = re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())
    return [t for t in s.split() if len(t) >= 3]

def keyword_score(query: str, text: str) -> int:
    q = set(_tokenize(query))
    t = set(_tokenize(text))
    return len(q.intersection(t))

def filter_and_rerank(query: str, docs: list[str], metas: list[dict], min_kw: int = 1, topk: int = 5):
    scored = []
    for d, m in zip(docs, metas):
        s = keyword_score(query, d or "")
        scored.append((s, d, m))

    # primary: keyword overlap, secondary: vector score
    scored.sort(key=lambda x: (x[0], x[2].get("score", 0.0)), reverse=True)

    kept = [(d, m, s) for s, d, m in scored if s >= min_kw][:topk]
    filt_docs = [d for d, _, _ in kept]
    filt_metas = [m for _, m, _ in kept]
    filt_scores = [s for _, _, s in kept]
    return filt_docs, filt_metas, filt_scores

# =========================
# MILVUS SEARCH
# =========================
def search_milvus(query_text: str, year: int | None, limit: int = 5):
    col = get_collection()

    text_field = pick_text_field(col)
    if not text_field:
        st.error("❌ Could not find any VARCHAR/STRING field to use as document text.")
        return [], [], {"schema": [(f.name, str(f.dtype)) for f in col.schema.fields]}

    existing = existing_field_set(col)

    desired = [text_field, "file_name", "year", "page", "image_path"]
    output_fields = [f for f in desired if f in existing]

    query_vector = embed_query(query_text)

    expr = None
    if year is not None and "year" in existing:
        expr = f"year == {int(year)}"

    search_params = {"metric_type": "COSINE", "params": {}}

    try:
        results = col.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=expr,
            output_fields=output_fields
        )
    except Exception as e:
        st.error(f"❌ Search Error: {e}")
        return [], [], {"schema": [(f.name, str(f.dtype)) for f in col.schema.fields]}

    if not results or len(results[0]) == 0:
        return [], [], {"schema": [(f.name, str(f.dtype)) for f in col.schema.fields]}

    docs, metas = [], []
    for hit in results[0]:
        docs.append(hit.entity.get(text_field))
        metas.append({
            "chunk_id": str(hit.id),
            "file_name": hit.entity.get("file_name"),
            "year": hit.entity.get("year"),
            "page": hit.entity.get("page"),
            "image_path": hit.entity.get("image_path"),
            "score": float(hit.score),
        })

    debug = {
        "schema": [(f.name, str(f.dtype)) for f in col.schema.fields],
        "text_field": text_field,
        "expr": expr
    }
    return docs, metas, debug

# =========================
# ANSWER GENERATION
# =========================
def grounded_answer(query: str, metas: list[dict], docs: list[str], mode: str):
    ensure_genai()
    model = genai.GenerativeModel(GEN_MODEL)

    chunks = []
    for i, (m, d) in enumerate(zip(metas, docs), start=1):
        src = (
            f"[S{i}] (chunk_id={m.get('chunk_id')}) "
            f"{m.get('file_name')} (year {m.get('year')}, page {m.get('page')}, score {m.get('score'):.4f})"
        )
        chunks.append(f"{src}\n{d}")

    context = "\n\n".join(chunks)

    prompt = f"""
You are a WYDOT document assistant. Answer ONLY using the provided sources.
Rules:
- If the answer is not present in the sources, say exactly: "I couldn't find that information in the documents."
- When you state a fact, add citations like [S1], [S2].
- If the user asks for comparison/trends, explicitly compare and cite both sides.
- Be concise.

Mode: {mode}

Sources:
{context}

Question: {query}
"""
    try:
        res = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)
        return linkify_citations(res.text, metas)
    except Exception as e:
        return f"Error generating answer: {e}"

# =========================
# SUPERVISOR (multi-year compare)
# =========================
def supervisor_compare(query: str, years: list[int]):
    all_docs = []
    all_metas = []

    for y in years:
        docs, metas, _ = search_milvus(query, year=y, limit=8)
        all_docs.extend(docs)
        all_metas.extend(metas)

    if not all_docs:
        return "I couldn't find that information in the documents.", [], "No context found."

    answer = grounded_answer(query, all_metas, all_docs, mode=f"COMPARE {years}")

    # collect images (RESOLVED paths)
    all_images = [
        resolve_image_path(m["image_path"])
        for m in all_metas
        if m.get("image_path") and m["image_path"] != "N/A"
    ]
    all_images = sorted(set([p for p in all_images if p]))

    dbg = "\n".join([
        f"Compare years: {years}, chunks: {len(all_docs)}",
        build_debug_context(all_metas, all_docs)
    ])
    return answer, all_images, dbg

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("WYDOT Intelligent RAG (Milvus)")
    st.markdown("### Architecture")
    st.info("1) Router → 2) Experts (year filtered) → 3) Supervisor (compare)")

    st.divider()
    st.caption("Config loaded from .env")
    st.write("Collection:", MILVUS_COLLECTION)

    st.caption("Image folder sanity check")
    st.write("APP_DIR:", str(APP_DIR))
    st.write("OUTPUT_DIR exists:", OUTPUT_DIR.exists())
    if OUTPUT_DIR.exists():
        st.write("JPG count:", len(list(OUTPUT_DIR.glob("*.jpg"))))

# =========================
# UI STATE
# =========================
st.title(" WYDOT Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

        if msg.get("images"):
            with st.expander("📊 View referenced charts/images"):
                for img in msg["images"]:
                    if img and os.path.exists(img):
                        st.image(img, caption="Reference", width=650)
                    else:
                        st.caption(f"Image referenced but not found locally: {img}")

        if msg.get("debug"):
            with st.expander("Sources(route + years + filter + context preview)"):
                st.markdown(msg["debug"], unsafe_allow_html=True)

# =========================
# MAIN INPUT
# =========================
if prompt := st.chat_input("Ask about WYDOT specs or reports..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Routing + retrieving..."):
            available_years = [2022, 2023, 2021, 2010]

            route_data = router_agent(prompt, available_years)
            route = route_data["route"]
            years = route_data["years"]

            debug_lines = [f"Route: {route}", f"Years: {years}"]
            valid_images = []
            response_text = ""

            if route == "COMPARE":
                response_text, images, dbg = supervisor_compare(prompt, years)
                debug_lines.append(dbg)

                valid_images = [p for p in images if p and os.path.exists(p)]

            else:
                target_year = years[0] if years else max(available_years)

                # Always get raw vector hits first
                raw_docs, raw_metas, debug = search_milvus(prompt, year=target_year, limit=15)

                debug_lines.append(f"Year filter expr: {debug.get('expr')}")
                debug_lines.append(f"Text field: {debug.get('text_field')}")
                debug_lines.append(f"Raw vector hits: {len(raw_docs)}")

                # keyword filter + rerank
                docs, metas, kw_scores = filter_and_rerank(prompt, raw_docs, raw_metas, min_kw=1, topk=5)
                debug_lines.append(f"Keyword overlap scores (kept): {kw_scores}")

                if not docs:
                    response_text = (
                        "I couldn't find that information in the documents.\n\n"
                        "**I didn’t find an exact match, but here is related information from the closest excerpts:**"
                    )
                    debug_lines.append("No keyword-matching chunks. Showing nearest vector hits below:")
                    debug_lines.append(build_debug_context(raw_metas[:5], raw_docs[:5]))

                    images = [
                        resolve_image_path(m["image_path"])
                        for m in raw_metas[:5]
                        if m.get("image_path") and m["image_path"] != "N/A"
                    ]
                    images = sorted(set([p for p in images if p]))
                else:
                    response_text = grounded_answer(prompt, metas, docs, mode=f"{route} {target_year}")
                    debug_lines.append(build_debug_context(metas, docs))

                    images = [
                        resolve_image_path(m["image_path"])
                        for m in metas
                        if m.get("image_path") and m["image_path"] != "N/A"
                    ]
                    images = sorted(set([p for p in images if p]))

                valid_images = [p for p in images if p and os.path.exists(p)]

        st.markdown(response_text, unsafe_allow_html=True)

        if valid_images:
            with st.expander("📊 Click to view relevant charts/images"):
                for img_path in valid_images:
                    st.image(img_path, caption=f"Source: {img_path}", width=650)
        else:
            # if images were referenced but not found, show what was attempted
            with st.expander("📊 Click to view relevant charts/images"):
                st.caption("No image files found for this answer (or none were referenced).")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "images": valid_images,            # store resolved + existing paths only
        "debug": "\n".join(debug_lines)
    })


