
# import os
# import json
# import nest_asyncio
# import pandas as pd
# from llama_parse import LlamaParse
# from llama_index.core.schema import TextNode
# import google.generativeai as genai
# from PIL import Image
# import io
# import re

# # --- CONFIGURATION ---
# # API KEYS
# # ⚠️ REPLACE THESE WITH YOUR ACTUAL KEYS
# os.environ["LLAMA_CLOUD_API_KEY"] = "llx-70lfJlK1UztwgKFgNPlUA7NLU1AhJnsFtLyLJEFrRh6wdCIe" 
# GOOGLE_API_KEY = "AIzaSyBJCo4A_oJENi45ShXYyeZHRBorKa121Ho" # Your Gemini Key

# # DIRECTORIES
# DATA_DIR = "./data_raw"
# IMG_DIR = "./output_images"
# OUTPUT_JSON_FILE = "final_extracted_data.json" # <--- New Output File
# os.makedirs(IMG_DIR, exist_ok=True)

# # SETUP GEMINI
# genai.configure(api_key=GOOGLE_API_KEY)
# vision_model = genai.GenerativeModel('gemini-1.5-flash') # Updated to 1.5-flash as 2.5 is not standard yet

# nest_asyncio.apply()

# def get_year_from_filename(filename):
#     """Simple regex to grab the first 4-digit year from filename."""
#     match = re.search(r'\d{4}', filename)
#     return int(match.group(0)) if match else "Unknown"

# def caption_image_with_gemini(image_path):
#     """Sends image to Gemini for a detailed caption."""
#     try:
#         img = Image.open(image_path)
#         response = vision_model.generate_content([
#             "Analyze this chart, graph, or image from an annual report. "
#             "Extract key figures, trends, and titles. Output a detailed summary paragraph.", 
#             img
#         ])
#         return response.text
#     except Exception as e:
#         print(f"Error captioning {image_path}: {e}")
#         return "Image processing failed."

# def process_pdf(file_path):
#     file_name = os.path.basename(file_path)
#     print(f"Processing: {file_name}...")
    
#     # 1. PARSE WITH LLAMAPARSE
#     parser = LlamaParse(
#         result_type="markdown",  
#         premium_mode=True,       
#         parsing_instruction="This is an annual report. Keep tables structured. Extract all charts.",
#         api_key=os.environ["LLAMA_CLOUD_API_KEY"],
#         verbose=True,
#         # CRITICAL CHANGE: This ensures images are downloaded to your local folder
#         download_images=True  
#     )
    
#     # LlamaParse returns a list of 'Document' objects (JSON-like wrapper)
#     json_objs = parser.get_json_result(file_path)
#     json_data = json_objs[0] 

#     nodes = []
    
#     # 2. ITERATE PAGES
#     for page in json_data["pages"]:
#         page_num = page["page"]
        
#         # --- A. HANDLE TEXT ---
#         text_content = page["md"] 
#         if text_content.strip():
#             text_node = TextNode(
#                 text=text_content,
#                 metadata={
#                     "file_name": file_name,
#                     "year": get_year_from_filename(file_name),
#                     "page": page_num,
#                     "type": "text_block",
#                     "image_path": "N/A" # Consistent schema
#                 }
#             )
#             nodes.append(text_node)

#         # --- B. HANDLE IMAGES ---
#         for img_info in page.get("images", []):
#             img_name = img_info["name"]
            
#             # LlamaParse downloads images to the current working directory by default
#             local_img_path = f"./{img_name}" 
            
#             # Only process if the file was actually downloaded
#             if os.path.exists(local_img_path):
#                 # Move to our organized folder with a unique name
#                 new_img_name = f"{os.path.splitext(file_name)[0]}_p{page_num}_{img_name}"
#                 final_path = os.path.join(IMG_DIR, new_img_name)
                
#                 # Rename/Move the file
#                 os.rename(local_img_path, final_path)
                
#                 # CAPTION IT
#                 print(f"   -> Captioning image: {new_img_name}")
#                 caption = caption_image_with_gemini(final_path)
                
#                 # Create a specific NODE for this image
#                 img_node = TextNode(
#                     text=f"[IMAGE SUMMARY]\nType: Chart/Figure\nCaption: {caption}",
#                     metadata={
#                         "file_name": file_name,
#                         "year": get_year_from_filename(file_name),
#                         "page": page_num,
#                         "type": "image_caption",
#                         "image_path": final_path # <-- THIS IS KEY FOR RAG
#                     }
#                 )
#                 nodes.append(img_node)

#     return nodes

# # --- MAIN EXECUTION ---
# all_nodes = []
# pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]

# for pdf in pdf_files:
#     nodes = process_pdf(os.path.join(DATA_DIR, pdf))
#     all_nodes.extend(nodes)

# # --- 3. SAVE TO JSON (NEW STEP) ---
# print(f"\nSaving {len(all_nodes)} nodes to {OUTPUT_JSON_FILE}...")

# # We must convert TextNode objects to standard Dictionaries to save them as JSON
# nodes_as_dicts = [node.to_dict() for node in all_nodes]

# with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
#     json.dump(nodes_as_dicts, f, indent=4, ensure_ascii=False)

# print(f"✅ Data saved successfully to {OUTPUT_JSON_FILE}")

# # --- 4. METADATA AUDIT ---
# print("\n--- METADATA AUDIT ---")
# df = pd.DataFrame([n.metadata for n in all_nodes])
# if not df.empty:
#     print(df.head())
#     print("\nSummary of Extracted Years:")
#     print(df['year'].value_counts())
#     print("\nSummary of Node Types:")
#     print(df['type'].value_counts())

#     if df['year'].isnull().any() or (df['year'] == "Unknown").any():
#         print("⚠️ WARNING: Some documents have missing years!")
#     else:
#         print("✅ SUCCESS: Metadata looks valid.")
# else:
#     print("❌ ERROR: No nodes extracted.")




# --- CONFIGURATION ---
# ⚠️ REPLACE WITH REAL KEYS


# DIRECTORIES
import os
import json
import nest_asyncio
import re
from llama_parse import LlamaParse
from llama_index.core.schema import TextNode
import google.generativeai as genai
from pdf2image import convert_from_path
from PIL import Image

# --- CONFIGURATION ---
# ⚠️ REPLACE WITH YOUR REAL KEYS
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-..." 
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-70lfJlK1UztwgKFgNPlUA7NLU1AhJnsFtLyLJEFrRh6wdCIe" 
GOOGLE_API_KEY = "AIzaSyBJCo4A_oJENi45ShXYyeZHRBorKa121Ho" # Your Gemini Key

# DIRECTORIES
DATA_DIR = "./data_raw"
IMG_DIR = "./output_images"
OUTPUT_JSON_FILE = "final_extracted_data_visual.json" 
os.makedirs(IMG_DIR, exist_ok=True)

# SETUP GEMINI
genai.configure(api_key=GOOGLE_API_KEY)
vision_model = genai.GenerativeModel('gemini-2.5-flash') 

nest_asyncio.apply()

def get_year_from_filename(filename):
    match = re.search(r'\d{4}', filename)
    return int(match.group(0)) if match else "Unknown"

def analyze_page_with_gemini(image_path):
    """
    Sends page image to Gemini. 
    Returns: (Has_Chart_Bool, Description_String)
    """
    try:
        img = Image.open(image_path)
        # We ask Gemini to act as a filter AND a captioner
        response = vision_model.generate_content([
            "Analyze this page image. Does it contain a data chart, graph, map, or significant engineering diagram? "
            "Answer starting with 'YES' or 'NO'. "
            "If YES, provide a detailed description of the chart data/trends for a blind user. "
            "If NO, just say NO.", 
            img
        ])
        text = response.text.strip()
        
        if text.upper().startswith("YES"):
            # Remove the "YES" part and return the description
            return True, text[3:].strip(" :.-")
        else:
            return False, None
    except Exception as e:
        print(f"   ⚠️ Gemini Error: {e}")
        return False, None

def process_pdf(file_path):
    file_name = os.path.basename(file_path)
    print(f"\nProcessing: {file_name}...")
    
    # 1. PARSE TEXT ONLY (Standard Mode - Most Reliable for Text)
    parser = LlamaParse(
        result_type="markdown",  
        premium_mode=True,
        verbose=True,
        language="en",
        download_images=False # We will handle images ourselves locally
    )
    
    # Get Text Data
    try:
        json_objs = parser.get_json_result(file_path)
        if not json_objs:
            print("   ❌ LlamaParse returned empty.")
            return []
        json_data = json_objs[0]
    except Exception as e:
        print(f"   ❌ LlamaParse Error: {e}")
        return []

    nodes = []
    
    # 2. CONVERT PDF TO IMAGES (Local Snapshot)
    print("   📷 Converting PDF pages to images for analysis...")
    try:
        # dpi=150 is a good balance between quality and speed
        pdf_images = convert_from_path(file_path, dpi=150)
    except Exception as e:
        print(f"   ❌ PDF2Image failed (Is Poppler installed?): {e}")
        return []

    # 3. ITERATE PAGES
    for page in json_data["pages"]:
        page_num = page["page"]
        text_content = page["md"]
        
        # --- A. TEXT NODE ---
        if text_content.strip():
            nodes.append(TextNode(
                text=text_content,
                metadata={
                    "file_name": file_name,
                    "year": get_year_from_filename(file_name),
                    "page": page_num,
                    "type": "text_block",
                    "image_path": "N/A"
                }
            ))

        # --- B. VISUAL ANALYSIS (The "Force" Strategy) ---
        # Note: page_num is usually 1-based, pdf_images list is 0-based
        idx = page_num - 1
        
        if 0 <= idx < len(pdf_images):
            # Save temporary image for Gemini to look at
            temp_img_name = f"temp_{file_name}_{page_num}.jpg"
            temp_img_path = os.path.join(IMG_DIR, temp_img_name)
            pdf_images[idx].save(temp_img_path, "JPEG")
            
            # Ask Gemini: "Is there a chart here?"
            has_chart, description = analyze_page_with_gemini(temp_img_path)
            
            if has_chart:
                # Keep the file, rename it to permanent structure
                final_img_name = f"{os.path.splitext(file_name)[0]}_p{page_num}_CHART.jpg"
                final_path = os.path.join(IMG_DIR, final_img_name)
                
                # If reusing the temp file
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.rename(temp_img_path, final_path)
                
                print(f"   ✅ Chart detected on Page {page_num}. Saved {final_img_name}")
                
                nodes.append(TextNode(
                    text=f"[VISUAL DATA EXTRACT]\nType: Chart/Graph/Map\nSummary: {description}",
                    metadata={
                        "file_name": file_name,
                        "year": get_year_from_filename(file_name),
                        "page": page_num,
                        "type": "image_caption", # <--- THIS is what you were missing
                        "image_path": final_path # <--- THIS is your RAG link
                    }
                ))
            else:
                # No chart, delete the temp image to save space
                os.remove(temp_img_path)

    return nodes

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    all_nodes = []
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    
    for pdf in pdf_files:
        try:
            file_nodes = process_pdf(os.path.join(DATA_DIR, pdf))
            all_nodes.extend(file_nodes)
        except Exception as e:
            print(f"CRITICAL ERROR on {pdf}: {e}")

    # Save Final JSON
    print(f"\n💾 Saving {len(all_nodes)} nodes to {OUTPUT_JSON_FILE}...")
    nodes_as_dicts = [node.to_dict() for node in all_nodes]
    
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(nodes_as_dicts, f, indent=4, ensure_ascii=False)
    
    # Final Audit
    count_images = sum(1 for n in nodes_as_dicts if n['metadata']['type'] == 'image_caption')
    print(f"✅ Process Complete.")
    print(f"📊 Total Text Blocks: {len(nodes_as_dicts) - count_images}")
    print(f"🖼️  Total Charts Detected & Saved: {count_images}")