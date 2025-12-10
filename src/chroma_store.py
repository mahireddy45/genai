from __future__ import annotations
from typing import List, Dict, Optional
import os
import uuid
import logging

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
except Exception:  # pragma: no cover - optional dependency
    chromadb = None

from .embeddings import get_text_embeddings
from .schemas import IngestedDocument


def _ensure_chromadb():
    if chromadb is None:
        raise RuntimeError("chromadb package is not installed. Add 'chromadb' to requirements.txt")


class ChromaStore:
    def __init__(self, persist_directory: str = "./data/chroma_db", collection_name: str = "documents"):
        _ensure_chromadb()
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.Client(Settings(persist_directory=persist_directory))
        self.collection_name = collection_name
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)

    def get_collection_info(self) -> Dict:
        # Try to use the collection.count() method if available (preferred).
        try:
            count = self.collection.count()
        except Exception:
            # Fallback: attempt to fetch metadatas or documents and count them.
            try:
                data = self.collection.get(include=["metadatas"]) if hasattr(self.collection, "get") else None
                if isinstance(data, dict) and "metadatas" in data:
                    metas = data["metadatas"]
                    # metas might be a list of lists (per-query results) or a flat list
                    if metas and isinstance(metas[0], list):
                        count = sum(len(m) for m in metas)
                    else:
                        count = len(metas)
                else:
                    count = 0
            except Exception:
                count = 0

        return {
            "collection_name": self.collection_name,
            "db_path": self.persist_directory,
            "document_count": count,
        }

    def add_documents(self, docs: List[Dict], model: str = "text-embedding-3-large") -> int:
        # docs are expected to be dicts matching IngestedDocument
        texts = [d["text"] for d in docs]
        ids = [d.get("id") or str(uuid.uuid4()) for d in docs]
        metadatas = [d.get("meta") for d in docs]

        embeddings = get_text_embeddings(texts, model=model)

        # If collection.add accepts embeddings directly
        try:
            self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        except TypeError:
            # older/newer clients may use a different API
            self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
            # attempt to upsert embeddings via client if supported
        return len(ids)

    def query(self, query_text: str, n_results: int = 4):
        emb = get_text_embeddings([query_text])[0]
        results = self.collection.query(query_embeddings=[emb], n_results=n_results)
        return results
