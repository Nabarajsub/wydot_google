import os
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global model variable
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model_name = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
    logger.info(f"Loading model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        # Warmup
        model.encode("warmup")
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

class PredictRequest(BaseModel):
    instances: List[str]

@app.post("/predict")
async def predict(request: Request):
    """
    Vertex AI Prediction format:
    Input: {"instances": ["text1", "text2"]}
    Output: {"predictions": [[emb1], [emb2]]}
    """
    try:
        body = await request.json()
    except Exception:
        return {"error": "Invalid JSON"}, 400

    instances = body.get("instances", [])
    if not instances:
        return {"predictions": []}

    if model is None:
        return {"error": "Model not loaded"}, 500

    try:
        # Encode
        embeddings = model.encode(instances)
        # Convert to list for JSON serialization
        return {"predictions": embeddings.tolist()}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    # Vertex AI sets AIP_HTTP_PORT
    port = int(os.getenv("AIP_HTTP_PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
