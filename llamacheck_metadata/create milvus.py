import os
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
import google.generativeai as genai

from pymilvus import (
    connections, utility,
    FieldSchema, CollectionSchema, DataType,
    Collection
)

# =========================
# CONFIG
# =========================
load_dotenv()

# Gemini
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("❌ GOOGLE_API_KEY missing")

# Milvus/Zilliz Cloud
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_SECURE = True

MILVUS_COLLECTION = "metadata_specs"
JSON_PATH = "final_extracted_data_visual_withspecs.json"

# Gemini base embedding (768)
BASE_DIM = 768
TARGET_DIM = 3072   # ⭐ IMPORTANT

EMBED_MODEL = "models/text-embedding-004"
BATCH_SIZE = 64
DROP_COLLECTION_IF_EXISTS = True

# =========================
# EMBEDDING
# =========================
def expand_embedding(vec: List[float], target_dim: int) -> List[float]:
    """Repeat embedding deterministically to match target dim."""
    if not vec:
        return [0.0] * target_dim
    if len(vec) == target_dim:
        return vec
    if len(vec) > target_dim:
        return vec[:target_dim]

    reps = target_dim // len(vec)
    rem = target_dim % len(vec)
    return vec * reps + vec[:rem]


def embed_texts(texts: List[str]) -> List[List[float]]:
    out = []
    for t in texts:
        if not t or not isinstance(t, str):
            out.append([0.0] * TARGET_DIM)
            continue
        try:
            res = genai.embed_content(
                model=EMBED_MODEL,
                content=t,
                task_type="retrieval_document",
            )
            base = res["embedding"]          # 768
            out.append(expand_embedding(base, TARGET_DIM))  # ➜ 3072
        except Exception as e:
            print(f"⚠️ embed failed, zeros used: {e}")
            out.append([0.0] * TARGET_DIM)
    return out

# =========================
# MILVUS
# =========================
def connect_milvus():
    connections.connect(
        alias="default",
        uri=MILVUS_URI,
        token=MILVUS_TOKEN,
        secure=MILVUS_SECURE,
    )


def ensure_collection(name: str) -> Collection:
    if utility.has_collection(name):
        if DROP_COLLECTION_IF_EXISTS:
            print(f"🧹 Dropping existing collection: {name}")
            utility.drop_collection(name)
        else:
            col = Collection(name)
            return col

    print(f"🧱 Creating collection: {name}")

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="year", dtype=DataType.INT64),
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="is_active", dtype=DataType.BOOL),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=TARGET_DIM),
    ]

    schema = CollectionSchema(fields, description="WYDOT specs + annual reports (3072d)")
    col = Collection(name, schema=schema)

    col.create_index(
        field_name="embedding",
        index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"}
    )

    col.load()
    return col

# =========================
# DATA
# =========================
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_meta(item):
    raw = item.get("metadata", {}) or {}
    year = int(raw.get("year", 0) or 0)
    return {
        "file_name": raw.get("file_name", "Unknown"),
        "year": year,
        "page": int(raw.get("page", 0) or 0),
        "image_path": "" if raw.get("image_path") == "N/A" else raw.get("image_path", ""),
        "is_active": True if year == 2023 else False,
    }

# =========================
# INGEST
# =========================
def create_milvus_db():
    genai.configure(api_key=GOOGLE_API_KEY)

    connect_milvus()
    col = ensure_collection(MILVUS_COLLECTION)

    data = load_json(JSON_PATH)
    print(f"📂 Loaded {len(data)} JSON items")

    buffer_texts, buffer_meta = [], []

    def flush():
        if not buffer_texts:
            return
        embs = embed_texts(buffer_texts)

        col.insert([
            [m[0] for m in buffer_meta],     # id
            buffer_texts,                    # text
            [m[1]["file_name"] for m in buffer_meta],
            [m[1]["year"] for m in buffer_meta],
            [m[1]["page"] for m in buffer_meta],
            [m[1]["image_path"] for m in buffer_meta],
            [m[1]["is_active"] for m in buffer_meta],
            embs
        ])
        col.flush()
        print(f"   ✅ Inserted {len(buffer_texts)}")
        buffer_texts.clear()
        buffer_meta.clear()

    for item in data:
        txt = item.get("text")
        if not txt:
            continue
        meta = flatten_meta(item)
        doc_id = item.get("id_") or str(abs(hash(txt)))

        buffer_texts.append(txt)
        buffer_meta.append((doc_id, meta))

        if len(buffer_texts) >= BATCH_SIZE:
            flush()

    flush()
    col.load()
    print(f"🎉 DONE. Rows ≈ {col.num_entities}")

if __name__ == "__main__":
    create_milvus_db()
