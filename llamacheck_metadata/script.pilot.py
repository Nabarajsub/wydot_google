
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
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-70lfJlK1UztwgKFgNPlUA7NLU1AhJnsFtLyLJEFrRh6wdCIe" # Replace with your LlamaCloud Key
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") # Replace with your Gemini Key

# DIRECTORIES
# You mentioned your files are in 'raw_data', so I updated this.
DATA_DIR = "./data_raw" 
IMG_DIR = "./output_images"
OUTPUT_JSON_FILE = "final_extracted_data_visual_withspecs.json" 

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# SETUP GEMINI
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    vision_model = genai.GenerativeModel('gemini-2.5-flash') # Use 1.5 Flash (stable)
except ImportError:
    print("❌ Error: 'google.generativeai' not installed. Run: pip install google-generativeai")
    exit()

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
        # If poppler is not in PATH, you might need to install it: 
        # Windows: http://blog.alivate.com.au/poppler-windows/ (Add bin to PATH)
        # Mac: brew install poppler
        # Linux: sudo apt-get install poppler-utils
        pdf_images = convert_from_path(file_path, dpi=150)
    except Exception as e:
        print(f"   ❌ PDF2Image failed (Is Poppler installed?): {e}")
        # Continue with just text if image conversion fails
        pdf_images = []

    # 3. ITERATE PAGES
    # Safety check: ensure we don't index out of bounds if pdf_images failed
    num_images = len(pdf_images)
    
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
        
        if 0 <= idx < num_images:
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
                
                # If reusing the temp file (overwrite if exists)
                if os.path.exists(final_path):
                    try:
                        os.remove(final_path)
                    except:
                        pass # Handle open file errors gracefully
                        
                os.rename(temp_img_path, final_path)
                
                print(f"   ✅ Chart detected on Page {page_num}. Saved {final_img_name}")
                
                nodes.append(TextNode(
                    text=f"[VISUAL DATA EXTRACT]\nType: Chart/Graph/Map\nSummary: {description}",
                    metadata={
                        "file_name": file_name,
                        "year": get_year_from_filename(file_name),
                        "page": page_num,
                        "type": "image_caption", 
                        "image_path": final_path 
                    }
                ))
            else:
                # No chart, delete the temp image to save space
                try:
                    os.remove(temp_img_path)
                except:
                    pass

    return nodes

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    all_nodes = []
    
    if not os.path.exists(DATA_DIR):
         print(f"❌ Data directory '{DATA_DIR}' not found. Please create it and add PDFs.")
         exit()

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"⚠️ No PDF files found in '{DATA_DIR}'. Please add your 2 WYDOT specs there.")
    else:
        print(f"Found {len(pdf_files)} PDFs to process: {pdf_files}")
        
        for pdf in pdf_files:
            try:
                file_nodes = process_pdf(os.path.join(DATA_DIR, pdf))
                all_nodes.extend(file_nodes)
            except Exception as e:
                print(f"CRITICAL ERROR on {pdf}: {e}")

        # Save Final JSON
        if all_nodes:
            print(f"\n💾 Saving {len(all_nodes)} nodes to {OUTPUT_JSON_FILE}...")
            # Convert TextNodes to dicts for JSON serialization
            nodes_as_dicts = []
            for node in all_nodes:
                # TextNode object to dict conversion
                node_dict = {
                    "id_": node.node_id,
                    "text": node.text,
                    "metadata": node.metadata
                }
                nodes_as_dicts.append(node_dict)
            
            with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
                json.dump(nodes_as_dicts, f, indent=4, ensure_ascii=False)
            
            # Final Audit
            count_images = sum(1 for n in nodes_as_dicts if n['metadata'].get('type') == 'image_caption')
            print(f"✅ Process Complete.")
            print(f"📊 Total Text Blocks: {len(nodes_as_dicts) - count_images}")
            print(f"🖼️  Total Charts Detected & Saved: {count_images}")
        else:
            print("⚠️ No nodes were created. Check input files and keys.")


