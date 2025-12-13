from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List
import os
from .schemas import DocumentMeta, IngestedDocument
from .logging_config import get_logger
from .embedding_config import EmbeddingConfig

logger = get_logger(__name__)

# Function to store chunks in Chroma vector store
def store_in_vector_db(chunks: List[dict], persist_directory: str, embedding_model: str = "text-embedding-3-small") -> int:
    """Store chunks in vector database and return count of chunks stored."""
    # Create embeddings instance with the specified model
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment")
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key)
    
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    stored_count = 0
    logger.info("Starting to store %d chunks in vector database with embedding model: %s", len(chunks), embedding_model)
    
    # Save the embedding model config for future retrievals
    config = EmbeddingConfig(persist_directory)
    config.set_model(embedding_model)
    
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