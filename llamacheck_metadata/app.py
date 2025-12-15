# import streamlit as st
# import chromadb
# import google.generativeai as genai
# import json
# import os

# # --- PAGE CONFIG ---
# st.set_page_config(page_title="WYDOT Specs AI", layout="wide")

# # --- EMBEDDING WRAPPER ---
# class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
#     def __init__(self, api_key):
#         genai.configure(api_key=api_key)
#     def __call__(self, input: list) -> list:
#         model = "models/text-embedding-004"
#         embeddings = []
#         for text in input:
#             try:
#                 res = genai.embed_content(model=model, content=text, task_type="retrieval_query")
#                 embeddings.append(res['embedding'])
#             except:
#                 embeddings.append([0.0]*768)
#         return embeddings

# # --- SIDEBAR ---
# with st.sidebar:
#     st.title("WYDOT Intelligent RAG")
#     st.markdown("### Configuration")
    
#     # API Key Input
#     user_api_key = ""
    
#     os.environ["GOOGLE_API_KEY"] = user_api_key
#     genai.configure(api_key=user_api_key)


#     st.markdown("---")
#     st.markdown("**System Architecture:**")
#     st.info("1. **Router:** Classifies intent (Current vs History vs Compare)\n2. **Experts:** Retrieve specific year data\n3. **Supervisor:** Synthesizes multi-year logic")

# # --- DATABASE CONNECTION ---
# def get_db():
#     if not os.environ.get("GOOGLE_API_KEY"):
#         return None
#     try:
#         client = chromadb.PersistentClient(path="./wydot_vector_db")
#         ef = GeminiEmbeddingFunction(os.environ["GOOGLE_API_KEY"])
#         return client.get_collection(name="wydot_reports", embedding_function=ef)
#     except Exception as e:
#         st.error(f"DB Error: {e}")
#         return None

# # --- AGENT FUNCTIONS ---

# def router_agent(query):
#     """Decides which expert or supervisor to call."""
#     model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
#     prompt = f"""
#     You are a Router for WYDOT reports (2022 and 2023).
#     Classify the query:
#     1. CURRENT: Asks about 2023, active rules, or general status.
#     2. HISTORY: Asks about 2022 or past data.
#     3. COMPARE: Asks to compare years or find changes/differences.
    
#     Return JSON: {{ "route": "CURRENT" | "HISTORY" | "COMPARE", "years": [list of ints] }}
#     Query: {query}
#     """
#     try:
#         res = model.generate_content(prompt)
#         text = res.text.replace("```json", "").replace("```", "").strip()
#         return json.loads(text)
#     except:
#         return {"route": "CURRENT", "years": [2023]}

# def expert_agent(query, year, collection):
#     """Retrieves data for a specific year."""
#     results = collection.query(
#         query_texts=[query],
#         n_results=4,
#         where={"year": int(year)}
#     )
#     docs = results['documents'][0] if results['documents'] else []
#     context = "\n\n".join(docs)
    
#     if not context:
#         return f"No data found for {year}."

#     model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
#     prompt = f"""
#     You are the Expert for {year} WYDOT Reports.
#     Answer using ONLY this context:
#     {context}
    
#     Question: {query}
#     """
#     res = model.generate_content(prompt)
#     return res.text, context

# def supervisor_agent(query, collection):
#     """Retrieves 2022 AND 2023 data and compares."""
#     # Get 2022
#     r22 = collection.query(query_texts=[query], n_results=3, where={"year": 2022})
#     ctx22 = "\n".join(r22['documents'][0]) if r22['documents'] else "No 2022 data."
    
#     # Get 2023
#     r23 = collection.query(query_texts=[query], n_results=3, where={"year": 2023})
#     ctx23 = "\n".join(r23['documents'][0]) if r23['documents'] else "No 2023 data."
    
#     model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
#     prompt = f"""
#     You are the Supervisor (Comparison Engine).
#     Compare the following data for WYDOT.
    
#     [2022 Data]:
#     {ctx22}
    
#     [2023 Data]:
#     {ctx23}
    
#     User Question: {query}
#     Task: Highlight the differences, numbers, and trends between the two years.
#     """
#     res = model.generate_content(prompt)
#     return res.text, f"**2022 Context:**\n{ctx22}\n\n**2023 Context:**\n{ctx23}"

# # --- MAIN UI ---

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Chat container
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.write(msg["content"])
#         if "debug" in msg:
#             with st.expander("See Context & Routing"):
#                 st.text(msg["debug"])

# # Input
# if prompt := st.chat_input("Ask about WYDOT reports (e.g., 'Compare fatalities 2022 vs 2023')"):
#     if not user_api_key:
#         st.error("Please add your API Key in the sidebar first.")
#         st.stop()
        
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.write(prompt)
        
#     with st.chat_message("assistant"):
#         coll = get_db()
#         if not coll:
#             st.error("Database not initialized. Run `create_vector_db.py` first.")
#             st.stop()
            
#         with st.status("Processing...", expanded=True) as status:
#             # 1. Route
#             st.write("Routing query...")
#             route_data = router_agent(prompt)
#             route = route_data.get("route", "CURRENT")
#             years = route_data.get("years", [2023])
            
#             st.write(f"Route selected: **{route}**")
            
#             # 2. Retrieve & Generate
#             if route == "COMPARE":
#                 st.write("Supervisor Agent activated (Comparing 2022 vs 2023)...")
#                 response, context = supervisor_agent(prompt, coll)
#             elif route == "HISTORY":
#                 target = years[0] if years else 2022
#                 st.write(f"History Expert activated ({target})...")
#                 response, context = expert_agent(prompt, target, coll)
#             else:
#                 st.write("Current Expert activated (2023)...")
#                 response, context = expert_agent(prompt, 2023, coll)
                
#             status.update(label="Complete", state="complete", expanded=False)
            
#         st.write(response)
        
#         debug_info = f"Route: {route}\nYears: {years}\n\nContext Used:\n{context}"
#         st.session_state.messages.append({
#             "role": "assistant", 
#             "content": response,
#             "debug": debug_info
#         })

import streamlit as st
import chromadb
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="WYDOT Reports AI", layout="wide")

# --- SAFETY SETTINGS ---
# Crucial to prevent empty responses on government docs
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- EMBEDDING CLASS ---
class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
    def __call__(self, input: list) -> list:
        model = "models/text-embedding-004"
        embeddings = []
        for text in input:
            try:
                res = genai.embed_content(model=model, content=text, task_type="retrieval_query")
                embeddings.append(res['embedding'])
            except:
                embeddings.append([0.0]*768)
        return embeddings

# --- SIDEBAR ---
with st.sidebar:
    st.title("WYDOT Intelligent RAG")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Wyoming_Athletics_logo.svg/1200px-Wyoming_Athletics_logo.svg.png", width=80)
    
    # Secure API Key Input
    user_api_key = os.environ.get("GEMINI_API_KEY")
    
    os.environ["GOOGLE_API_KEY"] = user_api_key
    genai.configure(api_key=user_api_key)
    
    st.divider()
    st.markdown("### Architecture")
    st.info("""
    **1. Router:** Classifies Intent
    **2. Supervisor:** Handles Comparisons
    **3. Experts:** Search specific years
    """)

# --- DATABASE CONNECTION ---
def get_db():
    if not os.environ.get("GOOGLE_API_KEY"):
        return None
    try:
        # Connect to the persistent DB
        client = chromadb.PersistentClient(path="./wydot_vector_db")
        ef = GeminiEmbeddingFunction(os.environ["GOOGLE_API_KEY"])
        return client.get_collection(name="wydot_reports", embedding_function=ef)
    except Exception as e:
        return None

# --- AGENTS ---

def router_agent(query):
    # Use 1.5 Flash for stability
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""
    You are the Router for a RAG system containing WYDOT Annual Reports for 2022 and 2023.
    Classify the user query:
    
    1. CURRENT: Asks about 2023, active projects, current budget, or the latest report.
    2. HISTORY: Asks about 2022, past data, or last year's stats.
    3. COMPARE: Asks to compare years, find trends, or difference between 2022 and 2023.
    
    Return ONLY JSON: {{ "route": "CURRENT" | "HISTORY" | "COMPARE" }}
    
    Query: {query}
    """
    try:
        res = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)
        clean_text = res.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except:
        return {"route": "CURRENT"}

def expert_agent(query, year, collection):
    """Retrieves text AND images for a specific year."""
    results = collection.query(
        query_texts=[query],
        n_results=4,
        where={"year": int(year)}
    )
    
    docs = results['documents'][0] if results['documents'] else []
    metas = results['metadatas'][0] if results['metadatas'] else []
    
    context_text = "\n\n".join(docs)
    
    # Extract image paths found in metadata
    images_found = []
    for m in metas:
        if m.get('image_path'):
            images_found.append(m['image_path'])

    if not context_text:
        return f"I could not find specific information for {year}.", [], ""

    model = genai.GenerativeModel('gemini-2.5-flash')
    sys_prompt = f"You are the {year} Annual Report Expert. Answer using the context provided."
    
    try:
        res = model.generate_content(
            f"{sys_prompt}\nContext: {context_text}\nQuestion: {query}",
            safety_settings=SAFETY_SETTINGS
        )
        return res.text, images_found, context_text
    except Exception as e:
        return f"Error: {e}", images_found, context_text

def supervisor_agent(query, collection):
    """Retrieves from both years and compares."""
    # Get 2022 Data
    r22 = collection.query(query_texts=[query], n_results=3, where={"year": 2022})
    ctx22 = "\n".join(r22['documents'][0]) if r22['documents'] else "No data."
    img22 = [m['image_path'] for m in r22['metadatas'][0] if m.get('image_path')] if r22['metadatas'] else []

    # Get 2023 Data
    r23 = collection.query(query_texts=[query], n_results=3, where={"year": 2023})
    ctx23 = "\n".join(r23['documents'][0]) if r23['documents'] else "No data."
    img23 = [m['image_path'] for m in r23['metadatas'][0] if m.get('image_path')] if r23['metadatas'] else []

    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""
    You are the Supervisor Agent (Comparison Engine).
    Compare the following data sets.
    
    [2022 Data]:
    {ctx22}
    
    [2023 Data]:
    {ctx23}
    
    Question: {query}
    Task: Highlight differences, trends, and specific numbers. Be concise.
    """
    try:
        res = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)
        return res.text, img22 + img23, f"2022 Len: {len(ctx22)} | 2023 Len: {len(ctx23)}"
    except Exception as e:
        return f"Error: {e}", img22 + img23, ""

# --- MAIN UI LOGIC ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # UI CHANGE: Hide history images in expander
        if "images" in msg and msg["images"]:
            with st.expander("📊 View Referenced Charts & Visuals"):
                for img in msg["images"]:
                    if os.path.exists(img):
                        st.image(img, caption="Reference Visual", width=600)

# Chat Input
if prompt := st.chat_input("Ask about WYDOT reports..."):
    if not user_api_key:
        st.error("Please enter API Key in sidebar.")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        
    with st.chat_message("assistant"):
        coll = get_db()
        if not coll:
            st.error("Database not initialized. Run `create_vector_db.py`.")
            st.stop()
            
        with st.spinner("Routing and Retrieving..."):
            # 1. Route
            route_info = router_agent(prompt)
            route = route_info.get("route", "CURRENT")
            
            response_text = ""
            images = []
            debug_ctx = ""
            
            # 2. Execute
            if route == "COMPARE":
                response_text, images, debug_ctx = supervisor_agent(prompt, coll)
            elif route == "HISTORY":
                response_text, images, debug_ctx = expert_agent(prompt, 2022, coll)
            else: # CURRENT
                response_text, images, debug_ctx = expert_agent(prompt, 2023, coll)
            
            # 3. Display Text
            st.write(response_text)
            
            # 4. Display Images (UI CHANGE: Dropdown)
            valid_images = []
            if images:
                # Deduplicate images
                images = list(set(images))
                
                # The Expander creates the "Dropdown" effect
                with st.expander("📊 Click to view Relevant Charts & Visuals"):
                    for img_path in images:
                        if os.path.exists(img_path):
                            st.image(img_path, caption=f"Source: {img_path}", width=600)
                            valid_images.append(img_path)
                        else:
                            st.caption(f"*(Image referenced but file not found: {img_path})*")
            
            # 5. Save to History
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "images": valid_images,
                "debug": f"Route: {route}"
            })