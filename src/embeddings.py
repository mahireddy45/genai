from __future__ import annotations
import os
from typing import List
import time
import logging

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


def _get_openai_client():
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Install 'openai' in requirements.txt")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    # Instantiate client; the client will read env vars if api_key not passed
    return OpenAI(api_key=key)


def get_text_embeddings(texts: List[str], model: str = "text-embedding-3-large") -> List[List[float]]:
    """Return embeddings for a list of texts using OpenAI.

    Args:
        texts: list of input strings
        model: embedding model name (default: text-embedding-3-large)

    Returns:
        list of embedding vectors
    """
    _ensure_openai()

    # OpenAI accepts up to a certain number of inputs; for simplicity
    # we send texts in a single request. Retry basic transient errors.
    tries = 3
    client = _get_openai_client()
    for attempt in range(tries):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            embeddings = [item["embedding"] for item in resp.data]
            return embeddings
        except Exception as exc:
            logger.warning("Embedding request failed (attempt %s): %s", attempt + 1, exc)
            if attempt + 1 == tries:
                raise
            time.sleep(1 * (attempt + 1))
