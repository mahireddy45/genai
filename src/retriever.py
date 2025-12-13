from typing import List, Dict
import os
import logging
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

# API key is loaded by config.settings from .env file
# All modules can access it via os.getenv("OPENAI_API_KEY")

def retrieve(query: str, n_retrieve: int, db_path: str, retrieval_model_name: str) -> List[Dict]:
    logger.info("Starting retrieval for query using embedding_model: %s db_path: %s", retrieval_model_name, db_path)
    if not query.strip():
        logger.warning("Empty query received for retrieval.")
        return []

    logger.info("Retrieving top-%d chunks for query: %s", n_retrieve, query)

    try:
        # Get API key from environment (set by config.settings)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment")
            return []
        
        # Explicitly pass API key to OpenAIEmbeddings with hardcoded text-embedding-3-small model
        embeddings = OpenAIEmbeddings(model=retrieval_model_name, api_key=api_key)

        vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
        )

        results = vector_store.similarity_search(query, k=n_retrieve)

    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        return []

    formatted_results = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in results
    ]

    logger.info("Retrieved %d chunks", len(formatted_results))
    return formatted_results