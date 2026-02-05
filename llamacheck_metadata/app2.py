# import os
# import streamlit as st
# from dotenv import load_dotenv

# import google.generativeai as genai
# from pymilvus import connections, Collection, utility, DataType

# # =========================
# # PAGE SETUP
# # =========================
# st.set_page_config(page_title="WYDOT Chatbot", layout="wide")

# # =========================
# # CONFIG
# # =========================
# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")   # ✅ FIXED (was os.getenv(""))
# MILVUS_URI = os.getenv("MILVUS_URI")
# MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
# MILVUS_COLLECTION = "metadata_specs"

# EMBED_MODEL = "models/text-embedding-004"
# TARGET_DIM = 768
# GEN_MODEL = "gemini-2.0-flash"

# # =========================
# # EMBEDDING LOGIC
# # =========================
# def embed_query(text: str) -> list:
#     if not GOOGLE_API_KEY:
#         st.error("❌ GEMINI_API_KEY missing in environment (.env).")
#         return [0.0] * TARGET_DIM

#     genai.configure(api_key=GOOGLE_API_KEY)

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

# def pick_text_field(col: Collection) -> str | None:
#     """
#     Try to find the field that contains your document chunk text.
#     Preference order: text -> content -> chunk -> first VARCHAR
#     """
#     field_names = [f.name for f in col.schema.fields]

#     # common names
#     for candidate in ["text", "content", "chunk", "doc_text", "passage"]:
#         if candidate in field_names:
#             return candidate

#     # fallback: first VARCHAR/STRING field
#     for f in col.schema.fields:
#         if f.dtype in (DataType.VARCHAR, DataType.STRING):
#             return f.name

#     return None

# # =========================
# # QUERY LOGIC
# # =========================
# def search_milvus(query_text):
#     connect_milvus()

#     if not utility.has_collection(MILVUS_COLLECTION):
#         st.error(f"❌ Collection '{MILVUS_COLLECTION}' not found. Did you run the ingestion script?")
#         st.stop()

#     col = Collection(MILVUS_COLLECTION)
#     col.load()

#     # ---- DEBUG: show schema in UI ----
#     schema_fields = [(f.name, str(f.dtype)) for f in col.schema.fields]
#     with st.expander("🔎 Milvus schema (debug)"):
#         st.write(schema_fields)

#     # 1) embedding
#     query_vector = embed_query(query_text)

#     # 2) figure out correct text field
#     text_field = pick_text_field(col)
#     if not text_field:
#         st.error("❌ Could not find any VARCHAR/STRING field to use as document text in this collection.")
#         return [], []

#     # 3) build output fields safely (only include fields that exist)
#     existing = {f.name for f in col.schema.fields}
#     desired = [text_field, "file_name", "year", "page", "image_path"]
#     output_fields = [f for f in desired if f in existing]

#     search_params = {"metric_type": "COSINE", "params": {}}

#     try:
#         results = col.search(
#             data=[query_vector],
#             anns_field="embedding",
#             param=search_params,
#             limit=5,
#             output_fields=output_fields
#         )
#     except Exception as e:
#         st.error(f"❌ Search Error: {e}")
#         return [], []

#     if not results or len(results[0]) == 0:
#         return [], []

#     docs, metas = [], []
#     for hit in results[0]:
#         docs.append(hit.entity.get(text_field))
#         metas.append({
#             "file_name": hit.entity.get("file_name"),
#             "year": hit.entity.get("year"),
#             "page": hit.entity.get("page"),
#             "image_path": hit.entity.get("image_path"),
#             "score": hit.score
#         })

#     return docs, metas

# # =========================
# # GENERATION LOGIC
# # =========================
# def generate_answer(query, context):
#     genai.configure(api_key=GOOGLE_API_KEY)
#     model = genai.GenerativeModel(GEN_MODEL)

#     prompt = f"""
# You are a helpful assistant for the Wyoming Department of Transportation (WYDOT).
# Use the following retrieved context to answer the user's question.
# If the answer is not in the context, say: "I couldn't find that information in the documents."

# Context:
# {context}

# Question:
# {query}
# """
#     try:
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"Error generating answer: {e}"

# # =========================
# # UI
# # =========================
# st.title("🚧 WYDOT Assistant")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
#         if "images" in msg and msg["images"]:
#             for img in msg["images"]:
#                 if os.path.exists(img):
#                     st.image(img, caption="Reference Chart/Image", width=500)

# if prompt := st.chat_input("Ask about WYDOT specs or reports..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         with st.spinner("Searching documents..."):
#             docs, metas = search_milvus(prompt)

#             if not docs:
#                 response_text = "I couldn't find any relevant documents in the database."
#                 images = []
#             else:
#                 context_text = "\n\n".join(
#                     [f"Source ({m.get('file_name')}, p{m.get('page')}): {d}" for m, d in zip(metas, docs)]
#                 )
#                 response_text = generate_answer(prompt, context_text)

#                 images = [m["image_path"] for m in metas if m.get("image_path") and m["image_path"] != "N/A"]
#                 images = list(set(images))

#         st.markdown(response_text)

#         if images:
#             with st.expander("See related charts/images"):
#                 for img in images:
#                     if os.path.exists(img):
#                         st.image(img, caption="Reference", width=500)
#                     else:
#                         st.caption(f"Image referenced but not found locally: {img}")

#     st.session_state.messages.append({"role": "assistant", "content": response_text, "images": images})
import os
import json
import re
import streamlit as st
from dotenv import load_dotenv

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from pymilvus import connections, Collection, utility, DataType
from pathlib import Path



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

# You can use a stronger model for routing/supervisor if you want:
ROUTER_MODEL = "gemini-2.5-flash"
GEN_MODEL = "gemini-2.0-flash"

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
# ROUTER (CURRENT / HISTORY / COMPARE) + YEARS
# =========================
def router_agent(query: str, available_years: list[int]) -> dict:
    """
    Returns: {"route": "...", "years": [..]}
    If user doesn't mention a year, defaults to max(available_years) for CURRENT.
    """
    ensure_genai()
    model = genai.GenerativeModel(ROUTER_MODEL)

    yrs = sorted(set(available_years))
    default_year = max(yrs) if yrs else 2023

    prompt = f"""
You are a Router for a WYDOT RAG system.
Available report years in the database: {yrs}

Classify the user query into one of:
- CURRENT: asking about the latest year or present status(if the content is related to specs latest year is 2021 and if it is related to annual report latest year is 2023)
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

        # Fallbacks if model returns empty/invalid years
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
# DEBUG HELPERS (clickable citations -> debug anchors)
# =========================
def normalize_path(p: str) -> str:
    if not p or p == "N/A":
        return None
    # 1. Replace Windows backslashes with Unix forward slashes
    clean_p = p.replace("\\", "/")
    # 2. Remove leading './' if present so Path() handles it correctly from the root
    if clean_p.startswith("./"):
        clean_p = clean_p[2:]
    
    # 3. Create a path object relative to your current working directory
    # (Assuming output_images is in your project folder)
    full_path = Path.cwd() / clean_p
    return str(full_path)

def _safe_anchor_id(chunk_id: str) -> str:
    # Make a safe HTML anchor id from chunk_id
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(chunk_id))

def build_debug_context(metas: list[dict], docs: list[str], max_chars: int = 900) -> str:
    if not metas or not docs:
        return "### Retrieved chunks\nNone"

    # Jump links to each chunk (unique per chunk_id)
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

        # Anchor target for clickable citations
        out.append(f'<a id="CID_{anchor}"></a>')
        out.append(f"#### S{i} — chunk_id: `{cid}` — {file_name} (year {year}, page {page}, score {score:.4f})")

        if m.get("image_path") and m["image_path"] != "N/A":
            out.append(f"- image_path: `{m['image_path']}`")

        out.append(f"> {snippet}{more}\n")

    return "\n".join(out)

def linkify_citations(text: str, metas: list[dict]) -> str:
    """
    Convert [S1] style citations into clickable links that jump to
    the corresponding chunk_id anchor in the Debug expander.
    """
    out = text or ""
    for i, m in enumerate(metas, start=1):
        cid = m.get("chunk_id", f"missing_{i}")
        anchor = _safe_anchor_id(cid)
        out = out.replace(f"[S{i}]", f"[S{i}](#CID_{anchor})")
    return out

# =========================
# MILVUS SEARCH (year-filtered)
# =========================
def search_milvus(query_text: str, year: int | None, limit: int = 5):
    col = get_collection()

    text_field = pick_text_field(col)
    if not text_field:
        st.error("❌ Could not find any VARCHAR/STRING field to use as document text.")
        return [], [], {"schema": [(f.name, str(f.dtype)) for f in col.schema.fields]}

    existing = existing_field_set(col)

    # build output fields safely
    desired = [text_field, "file_name", "year", "page", "image_path"]
    output_fields = [f for f in desired if f in existing]

    query_vector = embed_query(query_text)

    # Milvus filter expression for year (if field exists)
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
            # ✅ store Milvus chunk id for debug + linking
            "chunk_id": str(hit.id),
            "file_name": hit.entity.get("file_name"),
            "year": hit.entity.get("year"),
            "page": hit.entity.get("page"),
            "image_path": hit.entity.get("image_path"),
            "score": float(hit.score),
        })

    debug = {"schema": [(f.name, str(f.dtype)) for f in col.schema.fields], "text_field": text_field, "expr": expr}
    return docs, metas, debug

# =========================
# ANSWER GENERATION (grounded + citations)
# =========================
def grounded_answer(query: str, metas: list[dict], docs: list[str], mode: str):
    ensure_genai()
    model = genai.GenerativeModel(GEN_MODEL)

    # Add structured context with stable citation keys
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
        # Make citations clickable -> jump to Debug anchors by chunk_id
        return linkify_citations(res.text, metas)
    except Exception as e:
        return f"Error generating answer: {e}"

# =========================
# SUPERVISOR (multi-year compare)
# =========================
def supervisor_compare(query: str, years: list[int]):
    all_docs = []
    all_metas = []
    all_images = []

    # retrieve per year
    for y in years:
        docs, metas, _ = search_milvus(query, year=y, limit=4)
        for m in metas:
            if m.get("image_path") and m["image_path"] != "N/A":
                all_images.append(m["image_path"])
        all_docs.extend(docs)
        all_metas.extend(metas)

    # if no docs at all
    if not all_docs:
        return "I couldn't find that information in the documents.", [], "No context found."

    answer = grounded_answer(query, all_metas, all_docs, mode=f"COMPARE {years}")

    dbg = "\n".join([
        f"Compare years: {years}, chunks: {len(all_docs)}",
        build_debug_context(all_metas, all_docs)
    ])

    return answer, sorted(set(all_images)), dbg

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

# =========================
# UI STATE
# =========================
st.title(" WYDOT Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

        if msg.get("images"):
            with st.expander("📊 View referenced charts/images"):
                for img in msg["images"]:
                    # img is already normalized here because we saved valid_images to state
                    if os.path.exists(img):
                        st.image(img, caption=f"Source: {Path(img).name}", use_container_width=True)
                    else:
                        st.caption(f"⚠️ File still not found at: {img}")

        if msg.get("debug"):
            with st.expander("Sources(route + years + filter + context preview)"):
                # ✅ Use markdown so the [S1](#...) links + <a id="..."> anchors work
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
            # Determine available years quickly (from env or fallback list)
            # If your DB has more years, just add them here, or you can store a list in .env.
            available_years = [2022, 2023, 2021, 2010]  # <-- change if needed

            route_data = router_agent(prompt, available_years)
            route = route_data["route"]
            years = route_data["years"]

            images = []
            debug_lines = [f"Route: {route}", f"Years: {years}"]

            # ROUTE HANDLING
            if route == "COMPARE":
                response_text, images, dbg = supervisor_compare(prompt, years)
                debug_lines.append(dbg)

                # For storing message images
                valid_images = []
                for img_path in images:
                    if os.path.exists(img_path):
                        valid_images.append(img_path)

            else:
                target_year = years[0] if years else max(available_years)
                docs, metas, debug = search_milvus(prompt, year=target_year, limit=5)

                debug_lines.append(f"Year filter expr: {debug.get('expr')}")
                debug_lines.append(f"Text field: {debug.get('text_field')}")
                debug_lines.append(f"Retrieved chunks: {len(docs)}")

                # ✅ Add chunk id + chunk preview + jump anchors into Debug
                debug_lines.append(build_debug_context(metas, docs))

                if not docs:
                    response_text = "I couldn't find that information in the documents."
                else:
                    response_text = grounded_answer(prompt, metas, docs, mode=f"{route} {target_year}")

                    images = []
                    for m in metas:
                        raw_path = m.get("image_path")
                        if raw_path and raw_path != "N/A":
                            norm = normalize_path(raw_path)
                            if norm:
                                images.append(norm)
                    images = sorted(set(images))

                    # Validate images for storing in session
                   

                # Validate images for storing in session
                valid_images = []
                for img_path in images:
                    if os.path.exists(img_path):
                        valid_images.append(img_path)

        # Show assistant answer (citations are clickable and jump to debug anchors)
        st.markdown(response_text, unsafe_allow_html=True)

        # Image viewer
        if images:
            with st.expander("📊 Click to view relevant charts/images"):
                for img_path in images:
                    if os.path.exists(img_path):
                        st.image(img_path, caption=f"Source: {img_path}", width=650)
                    else:
                        st.caption(f"Image referenced but file not found: {img_path}")

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "images": valid_images,
        "debug": "\n".join(debug_lines)
    })

