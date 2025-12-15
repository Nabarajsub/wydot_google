# 

import os
import json
import re
import uuid
import nest_asyncio  # not required anymore, but keeping it won't hurt

import pdfplumber
from pypdf import PdfReader

# --- CONFIGURATION ---
DATA_DIR = "./large_file_data"
OUTPUT_JSON_FILE = "final_extracted_data_visual_withspecs.json"

# Chunk / saving controls
PAGES_PER_SAVE = 25          # save JSON after every N pages (safer for big PDFs)
MIN_TEXT_CHARS = 30          # skip pages with too little extracted text


nest_asyncio.apply()


def get_year_from_filename(filename):
    """Extracts year from filename (e.g., '2010' from '2010 Standard Specs')"""
    match = re.search(r"\d{4}", filename)
    return int(match.group(0)) if match else "Unknown"


def get_pdf_page_count(file_path):
    """Fast local page count."""
    try:
        reader = PdfReader(file_path)
        return len(reader.pages)
    except Exception as e:
        print(f"   ⚠️ Could not read page count with pypdf: {e}")
        return None


def load_existing_json(output_file):
    """Load existing JSON and return (data_list, processed_files_set)."""
    if not os.path.exists(output_file):
        return [], set()

    print(f"📂 Loading existing data from {output_file}...")
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        processed = set(item["metadata"]["file_name"] for item in data if "metadata" in item and "file_name" in item["metadata"])
        print(f"   Found {len(data)} existing blocks from: {processed}")
        return data, processed
    except json.JSONDecodeError:
        print("   ⚠️ JSON file exists but is corrupt/empty. Starting fresh.")
        return [], set()


def save_json(output_file, data):
    """Save JSON safely."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def extract_text_from_pdf_pdfplumber(file_path, file_name, year):
    """
    Extract text from PDF using pdfplumber.
    Returns list of node dicts (same schema you used before).
    """
    nodes = []
    total_pages = get_pdf_page_count(file_path)
    if total_pages is not None:
        print(f"   ↳ PDF pages (local): {total_pages}")

    empty_pages = 0

    # pdfplumber pages are 0-indexed in python list,
    # but your metadata page numbers can be 1-indexed (more human friendly).
    with pdfplumber.open(file_path) as pdf:
        n = len(pdf.pages)
        print(f"   ↳ pdfplumber opened {n} pages.")

        for i, page in enumerate(pdf.pages):
            page_num_human = i + 1

            # Basic text extraction
            text = page.extract_text() or ""
            text = text.strip()

            if len(text) < MIN_TEXT_CHARS:
                empty_pages += 1
                continue

            nodes.append({
                "id_": str(uuid.uuid4()),
                "text": text,
                "metadata": {
                    "file_name": file_name,
                    "year": year,
                    "page": page_num_human,
                    "type": "text_block_spec",
                    "image_path": "N/A"
                }
            })

    return nodes, empty_pages, (total_pages if total_pages is not None else None)


def main():
    # 1) Load existing data
    existing_data, processed_files = load_existing_json(OUTPUT_JSON_FILE)

    # 2) Validate folder
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Directory '{DATA_DIR}' not found.")
        return

    all_pdfs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    files_to_process = [f for f in all_pdfs if f not in processed_files]

    if not files_to_process:
        print(f"\n✅ All files in '{DATA_DIR}' are already in the JSON. Nothing to do.")
        return

    print(f"\n🚀 Found {len(files_to_process)} new PDF spec files to process: {files_to_process}")

    # 3) Process each PDF, save incrementally
    final_data = existing_data[:]

    for pdf_file in files_to_process:
        full_path = os.path.join(DATA_DIR, pdf_file)
        file_name = os.path.basename(full_path)
        year = get_year_from_filename(file_name)

        print(f"\n📄 Processing (pdfplumber): {file_name}")

        try:
            # Extract all nodes first (simple & reliable)
            nodes, empty_pages, total_pages = extract_text_from_pdf_pdfplumber(full_path, file_name, year)

            if total_pages is not None:
                print(f"   ✅ Extracted blocks: {len(nodes)} | Empty/short-text pages skipped: {empty_pages} / {total_pages}")
            else:
                print(f"   ✅ Extracted blocks: {len(nodes)} | Empty/short-text pages skipped: {empty_pages}")

            if nodes:
                # Append and save
                final_data.extend(nodes)
                save_json(OUTPUT_JSON_FILE, final_data)
                print(f"💾 Saved. Total blocks now: {len(final_data)}")
            else:
                print("   ⚠️ No usable text extracted (PDF may be scanned images). JSON not updated for this file.")

        except Exception as e:
            print(f"   ❌ Failed to process {file_name}: {e}")

    print(f"\n✅ Done. Final total blocks: {len(final_data)}")


if __name__ == "__main__":
    main()
