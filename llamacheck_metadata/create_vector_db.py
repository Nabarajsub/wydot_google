# import json
# import chromadb
# import os
# import google.generativeai as genai
# from dotenv import load_dotenv

# # --- SETUP ---
# # Load environment variables if you have a .env file, otherwise set api_key directly
# load_dotenv()
# api_key = ""

# # NOTE: If you don't have a .env file, paste your key below:
# # api_key = "YOUR_GOOGLE_API_KEY_HERE"

# if not api_key:
#     print("❌ Error: Google API Key is missing. Set it in .env or the script.")
#     exit()

# class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
#     def __init__(self, api_key):
#         genai.configure(api_key=api_key)

#     def __call__(self, input: list) -> list:
#         model = "models/text-embedding-004"
#         embeddings = []
#         # Process one by one for safety, though batching is possible
#         for text in input:
#             try:
#                 # Ensure text is a string and not empty
#                 if not text or not isinstance(text, str):
#                     embeddings.append([0.0] * 768) # Placeholder for empty/bad text
#                     continue
                    
#                 result = genai.embed_content(
#                     model=model,
#                     content=text,
#                     task_type="retrieval_document"
#                 )
#                 embeddings.append(result['embedding'])
#             except Exception as e:
#                 print(f"⚠️ Warning: Failed to embed a chunk: {e}")
#                 embeddings.append([0.0] * 768) 
#         return embeddings

# def create_db():
#     print("🚀 Initializing Vector Store Creation...")

#     # 1. Setup ChromaDB
#     try:
#         gemini_ef = GeminiEmbeddingFunction(api_key=api_key)
#         client = chromadb.PersistentClient(path="./wydot_vector_db")
        
#         # Reset collection to ensure clean state
#         try:
#             client.delete_collection(name="wydot_reports")
#         except:
#             pass
            
#         collection = client.create_collection(
#             name="wydot_reports",
#             embedding_function=gemini_ef
#         )
#     except Exception as e:
#         print(f"❌ Failed to init DB: {e}")
#         return

#     # 2. Load Your JSON File
#     json_path = 'final_extracted_data_visual.json'
#     try:
#         with open(json_path, 'r') as f:
#             data = json.load(f)
#     except FileNotFoundError:
#         print(f"❌ Error: '{json_path}' not found. Make sure it is in the same folder.")
#         return

#     print(f"📂 Loaded {len(data)} documents. Processing...")

#     # 3. Prepare Data for DB
#     ids = []
#     documents = []
#     metadatas = []

#     for item in data:
#         # Safety check for required fields
#         if 'text' not in item or not item['text']:
#             continue

#         # Use the ID from JSON or generate one
#         doc_id = item.get('id_', str(hash(item['text'])))
#         ids.append(doc_id)
#         documents.append(item['text'])

#         # --- FLATTEN METADATA ---
#         # Chroma requires flat dictionaries (no nested dicts)
#         # We assume 2023 is "Active"
#         raw_meta = item.get('metadata', {})
        
#         flat_meta = {
#             "file_name": raw_meta.get('file_name', "Unknown"),
#             "year": int(raw_meta.get('year', 0)),
#             "page": int(raw_meta.get('page', 0)),
#             "is_active": True if raw_meta.get('year') == 2023 else False
#         }
#         metadatas.append(flat_meta)

#     # 4. Ingest into Chroma
#     if documents:
#         print(f"🧠 Generating Embeddings for {len(documents)} chunks (this may take a moment)...")
#         # Add in batches of 50 to prevent timeouts
#         batch_size = 50
#         for i in range(0, len(documents), batch_size):
#             end = min(i + batch_size, len(documents))
#             collection.add(
#                 ids=ids[i:end],
#                 documents=documents[i:end],
#                 metadatas=metadatas[i:end]
#             )
#             print(f"   Processed batch {i} to {end}...")
            
#         print(f"✅ Success! Database created at './wydot_vector_db' with {collection.count()} items.")
#     else:
#         print("❌ No valid documents found to add.")

# if __name__ == "__main__":
#     create_db()

import json
import chromadb
import os
import google.generativeai as genai
from dotenv import load_dotenv

# --- CONFIGURATION ---
# Load API key from .env or environment variable
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    # Fallback for manual entry if needed during testing
    # api_key = "YOUR_API_KEY_HERE"
    pass

if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found. Please set it in your environment.")
    exit()

class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, api_key):
        genai.configure(api_key=api_key)

    def __call__(self, input: list) -> list:
        model = "models/text-embedding-004"
        embeddings = []
        # Process sequentially to handle potential API limits or errors gracefully
        for text in input:
            try:
                # Basic validation
                if not text or not isinstance(text, str):
                    embeddings.append([0.0] * 768) # Placeholder for empty/bad text
                    continue
                    
                result = genai.embed_content(
                    model=model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                print(f"⚠️ Warning: Failed to embed chunk. Error: {e}")
                embeddings.append([0.0] * 768) 
        return embeddings

def create_db():
    print("🚀 Initializing Vector Store Creation...")

    # 1. Initialize ChromaDB
    try:
        gemini_ef = GeminiEmbeddingFunction(api_key=api_key)
        client = chromadb.PersistentClient(path="./wydot_vector_db")
        
        # Reset collection to ensure we don't have duplicate data from previous runs
        try:
            client.delete_collection(name="wydot_reports")
        except:
            pass
            
        collection = client.create_collection(
            name="wydot_reports",
            embedding_function=gemini_ef
        )
    except Exception as e:
        print(f"❌ Failed to initialize DB: {e}")
        return

    # 2. Load the JSON Data
    json_path = 'final_extracted_data_visual.json'
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: '{json_path}' not found. Please ensure the file is in the same directory.")
        return

    print(f"📂 Loaded {len(data)} documents. Processing metadata...")

    # 3. Prepare Data for Ingestion
    ids = []
    documents = []
    metadatas = []

    for item in data:
        # Skip items without text
        if 'text' not in item or not item['text']:
            continue

        # Use existing ID or generate one
        doc_id = item.get('id_', str(hash(item['text'])))
        ids.append(doc_id)
        documents.append(item['text'])

        # --- METADATA PROCESSING ---
        # ChromaDB requires flat dictionaries (no nested objects)
        # We assume 2023 documents are "Active" specs
        raw_meta = item.get('metadata', {})
        
        # Handle image path: Ensure it's a valid string, default to empty if missing or "N/A"
        img_path = raw_meta.get('image_path', "")
        if img_path == "N/A":
            img_path = ""

        flat_meta = {
            "file_name": raw_meta.get('file_name', "Unknown"),
            "year": int(raw_meta.get('year', 0)),
            "page": int(raw_meta.get('page', 0)),
            "image_path": img_path,
            "is_active": True if raw_meta.get('year') == 2023 else False
        }
        metadatas.append(flat_meta)

    # 4. Add to Vector Store
    if documents:
        print(f"🧠 Generating Embeddings for {len(documents)} chunks...")
        
        # Batch processing to prevent timeouts/memory issues
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end]
            )
            print(f"   Processed batch {i} to {end}...")
            
        print(f"✅ Success! Database created at './wydot_vector_db' with {collection.count()} items.")
    else:
        print("❌ No valid documents found to process.")

if __name__ == "__main__":
    create_db()