import os
from typing import List
import time
import logging
from config.secret import OPENAI_API_KEY
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from .schemas import DocumentMeta, IngestedDocument
from .logging_config import get_logger

logger = get_logger(__name__)
os.environ["OPENAI_API_KEY"] = "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"

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


def _get_openai_client():
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Install 'openai' in requirements.txt")
    
    # Try to get API key from environment first, then fall back to secret.py
    key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    
    if not key or "YOUR-ACTUAL" in key:
        raise RuntimeError(
            "\n" + "="*70 + "\n"
            "❌ OPENAI_API_KEY is not configured or is invalid!\n\n"
            "To use the RAG application:\n"
            "1. Get your API key: https://platform.openai.com/api-keys\n"
            "2. Update .env file: OPENAI_API_KEY=sk-proj-YOUR-KEY\n"
            "3. Or set env var: export OPENAI_API_KEY=sk-proj-YOUR-KEY\n"
            "4. Restart the application\n"
            "="*70 + "\n"
        )
    
    # Instantiate client with explicit api_key
    return OpenAI(api_key=key)


def validate_openai_key(model: str = "text-embedding-3-small") -> bool:
    """Validate OpenAI API key by making a test API call.
    
    Args:
        model: Model to use for validation test
        
    Returns:
        True if API key is valid, False otherwise
    """
    try:
        client = _get_openai_client()
    except Exception as e:
        logger.error("❌ OpenAI client could not be created: %s", str(e)[:200])
        return False

    try:
        # Make a small test payload to validate key
        resp = client.embeddings.create(model=model, input=["test"])
        logger.info("✅ OpenAI API key validated successfully")
        return True
    except Exception as exc:
        error_msg = str(exc)
        if "401" in error_msg or "invalid_api_key" in error_msg:
            logger.error("❌ OpenAI API key is invalid or expired. Error: %s", error_msg[:200])
        else:
            logger.error("❌ OpenAI API validation failed: %s", error_msg[:200])
        return False

# Function to chunk documents with dynamic chunk size and overlap
def chunk_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[dict]:
    # Initialize the text splitter with dynamic chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    logger.info("Chunking documents into size %d with overlap %d", chunk_size, chunk_overlap)
    for i, doc in enumerate(documents):
        # Handle both LangChain Document and custom IngestedDocument
        text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        
        # Sanitize metadata early: remove None values and non-string keys/values
        sanitized_metadata = {}
        for k, v in metadata.items():
            if v is not None and isinstance(k, str):
                # Convert non-string values to strings for ChromaDB compatibility
                if isinstance(v, (str, int, float, bool)):
                    sanitized_metadata[k] = v
                else:
                    sanitized_metadata[k] = str(v)
        
        split_texts = text_splitter.split_text(text)
        for j, chunk in enumerate(split_texts):
            chunks.append({
                "id": f"doc_{i}_chunk_{j}",
                "page_content": chunk,
                "metadata": {
                    "chunk_id": j,
                    "chunk_length": len(chunk),
                    **sanitized_metadata  # Merge sanitized metadata
                }
            })
    return chunks

# Function to generate embeddings
def generate_embeddings(docs: List, model: str = "text-embedding-3-large", chunk_size: int = 1000, chunk_overlap: int = 200) -> List[dict]:
    logger.info("chucksize: %d, chunk_overlap: %d", chunk_size, chunk_overlap)
    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    embedding_model = OpenAIEmbeddings(model=model)
    for chunk in chunks:
        chunk["embedding"] = embedding_model.embed_query(chunk["page_content"])
    return chunks

