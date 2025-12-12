from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List
import os
from .schemas import DocumentMeta, IngestedDocument
from .logging_config import get_logger

logger = get_logger(__name__)

# Function to store chunks in Chroma vector store
def store_in_vector_db(chunks: List[dict], persist_directory: str = "./db/chroma_db") -> int:
    """Store chunks in vector database and return count of chunks stored."""
    vector_store = Chroma(persist_directory=persist_directory)
    stored_count = 0
    
    for chunk in chunks:
        # Sanitize metadata: remove None values that ChromaDB cannot handle
        metadata = chunk.get("metadata", {})
        sanitized_metadata = {k: v for k, v in metadata.items() if v is not None}
        
        try:
            vector_store.add_texts(
                texts=[chunk["page_content"]],
                metadatas=[sanitized_metadata],
                ids=[chunk["id"]],
                embeddings=[chunk["embedding"]]  # Use precomputed embeddings
            )
            stored_count += 1
            logger.debug("Stored chunk %s with metadata: %s", chunk["id"], sanitized_metadata)
        except Exception as e:
            logger.error("Failed to store chunk %s: %s", chunk["id"], e)
    
    vector_store.persist()
    logger.info("Stored %d chunks in vector database", stored_count)
    return stored_count