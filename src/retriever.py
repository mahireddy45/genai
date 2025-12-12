from __future__ import annotations
from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from .logging_config import get_logger
import os

logger = get_logger(__name__)


class Retriever:
    """
    Handles semantic search using ChromaDB.
    - Loads the stored vector database.
    - Embeds the user query.
    - Returns top-K most relevant chunks with metadata.
    """

    def __init__(
        self,
        db_path: str = "./db/chroma_db",
        embedding_model: str = "text-embedding-3-small"
    ):
        self.db_path = db_path
        self.embedding_model = embedding_model

        logger.info(
            "Initializing Retriever | DB: %s | Embedding: %s",
            db_path,
            embedding_model
        )

        # Initialize embedding model used ONLY for query embedding
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)

        # Load existing Chroma vector store (DO NOT recreate)
        self.vstore = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings
        )

    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        """
        Perform top-k semantic search and return cleaned result.
        """
        if not query.strip():
            logger.warning("Empty query received for retrieval.")
            return []

        logger.info("Retrieving top-%d chunks for query: %s", k, query)

        try:
            results = self.vstore.similarity_search(query, k=k)
        except Exception as e:
            logger.error("Retrieval failed: %s", e)
            return []

        formatted_results = []
        for doc in results:
            formatted_results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
            )

        logger.info("Retrieved %d chunks", len(formatted_results))
        return formatted_results

    def as_retriever(self, k: int = 4):
        """Return LangChain retriever wrapper if needed by LC RAG chain."""
        return self.vstore.as_retriever(search_kwargs={"k": k})
