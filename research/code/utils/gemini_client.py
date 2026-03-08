"""
Shared Gemini API client with rate limiting and retry logic.
Reusable across all phases.
"""
import time
import json
import logging
from typing import Optional

import google.generativeai as genai

from config.settings import (
    GEMINI_API_KEY, GEMINI_FLASH_MODEL, GEMINI_EMBEDDING_MODEL,
    EMBEDDING_DIMS, MAX_RETRIES, BASE_RETRY_DELAY, GEMINI_RPM
)

logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Rate limiter state
_last_call_time = 0.0
_min_interval = 60.0 / GEMINI_RPM  # seconds between calls


def rate_limit():
    """Simple rate limiter for Gemini API calls."""
    global _last_call_time
    now = time.time()
    elapsed = now - _last_call_time
    if elapsed < _min_interval:
        time.sleep(_min_interval - elapsed)
    _last_call_time = time.time()


def retry_with_backoff(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """Execute function with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            rate_limit()
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource" in err_str or "quota" in err_str:
                delay = BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}), waiting {delay}s...")
                time.sleep(delay)
            else:
                raise
    raise Exception(f"Max retries ({max_retries}) exceeded")


def gemini_generate(prompt: str, temperature: float = 0.1,
                    response_schema=None, model_name: str = None) -> str:
    """
    Generate text with Gemini Flash.
    Optionally pass a Pydantic schema for structured output.
    """
    model_name = model_name or GEMINI_FLASH_MODEL
    model = genai.GenerativeModel(model_name)

    generation_config = {"temperature": temperature}
    if response_schema:
        generation_config["response_mime_type"] = "application/json"

    def _call():
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
        )
        return response.text

    return retry_with_backoff(_call)


def gemini_generate_json(prompt: str, temperature: float = 0.1,
                         model_name: str = None) -> dict:
    """Generate structured JSON from Gemini. Returns parsed dict."""
    raw = gemini_generate(prompt, temperature=temperature, model_name=model_name)
    # Clean markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return json.loads(raw.strip())


def gemini_embed(texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
    """
    Embed a list of texts using Gemini Embedding model.
    Returns list of embedding vectors.
    """
    embeddings = []
    batch_size = 50

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        def _embed_batch(b=batch):
            result = genai.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                content=b,
                task_type=task_type,
            )
            return result['embedding'] if isinstance(result['embedding'][0], list) else [result['embedding']]

        batch_embeddings = retry_with_backoff(_embed_batch)
        embeddings.extend(batch_embeddings)

        if i + batch_size < len(texts):
            time.sleep(0.5)  # batch delay

    return embeddings
