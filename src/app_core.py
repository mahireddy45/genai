from __future__ import annotations
from typing import List, Dict, Iterable, Tuple
import os
import tempfile
import logging
from pathlib import Path
from .document_loader import load_documents
from .schemas import IngestedDocument
from .embeddings import generate_embeddings
from .chroma_store import store_in_vector_db
from .response_validation import validate_assistant_output
from .logging_config import get_logger
from config.secret import OPENAI_API_KEY
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

try:
    import openai
except Exception:
    openai = None

logger = get_logger(__name__)

# Set API key from secret.py if not already in environment
if not os.getenv("OPENAI_API_KEY") and OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Global ChatOpenAI instance (lazy-initialized). Use `init_chat_llm` to configure.
chat_llm = None

def init_chat_llm(llm_model: str = "gpt-4o", temperature: float = 0.0, max_tokens: int | None = None):
    """Initialize or reconfigure a global ChatOpenAI instance for reuse across the module.

    Call this from the UI or startup code when you know which model and settings to use.
    """
    global chat_llm
    try:
        # ChatOpenAI accepts model, temperature and optionally max_tokens depending on langchain version
        if max_tokens is not None:
            chat_llm = ChatOpenAI(model=llm_model, temperature=temperature, max_tokens=max_tokens)
        else:
            chat_llm = ChatOpenAI(model=llm_model, temperature=temperature)
    except Exception:
        # If ChatOpenAI isn't available or fails to construct, set to None and log
        logger.exception("Failed to initialize ChatOpenAI global instance")
        chat_llm = None
    return chat_llm


def get_chat_llm(default_model: str = "gpt-4o", default_temp: float = 0.0, default_max_tokens: int | None = None):
    """Return the global chat_llm, initializing with defaults if unset."""
    global chat_llm
    if chat_llm is None:
        return init_chat_llm(default_model, default_temp, default_max_tokens)
    return chat_llm


def init_vector_store_raw(db_path: str, embedding_model: str, collection_name: str) -> ChromaStore:
    logger.info("Initializing ChromaStore at %s with collection %s", db_path, collection_name)
    return ChromaStore(persist_directory=db_path, collection_name=collection_name)


def process_uploaded_files(uploaded_files: List, embedding_model: str, llm_model: str, db_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> tuple:
    """Process uploaded files and store them in the vector database.
    
    Returns:
        Tuple of (num_documents, num_chunks) successfully stored.
    """
    all_documents = []
    logger.info("Loading and processing %d uploaded files", len(uploaded_files))
    
    try:
        all_documents = load_documents(uploaded_files, llm_model)
        logger.info("Loaded %d documents from files", len(all_documents))
    except Exception as e:
        logger.error("Failed to load documents: %s", e)
        return (0, 0)

    # Chunk and ingest documents
    try:
        chunks = generate_embeddings(all_documents, embedding_model, chunk_size, chunk_overlap)
        logger.info("Generated %d chunks from documents", len(chunks))
    except Exception as e:
        logger.error("Failed to generate embeddings: %s", e)
        return (len(all_documents), 0)

    # Store in vector database
    try:
        stored_count = store_in_vector_db(chunks, db_path)
        logger.info("Successfully stored %d chunks from %d documents in vector database", stored_count, len(all_documents))
        return (len(all_documents), stored_count)
    except Exception as e:
        logger.error("Failed to store documents in vector database: %s", e)
        return (len(all_documents), 0)


def process_directory(path: str, embedding_model: str, llm_model: str, db_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> tuple:
    """Process all files in a directory and store them in the vector database.
    
    Returns:
        Tuple of (num_documents, num_chunks) successfully stored.
    """
    directory_path = Path(path)
    uploaded_files = [str(f) for f in directory_path.glob("**/*") if f.is_file()]
    logger.info("Found %d files in directory %s", len(uploaded_files), path)
    return process_uploaded_files(uploaded_files, embedding_model, llm_model, db_path, chunk_size, chunk_overlap)

def _prepare_context_from_results(results, n_retrieve: int):
    docs_texts = []
    sources = []
    if isinstance(results, dict):
        docs = results.get("documents") or results.get("results") or []
        metadatas = results.get("metadatas") or []
        if docs and isinstance(docs[0], list):
            for i, doc_list in enumerate(docs):
                for j, d in enumerate(doc_list):
                    docs_texts.append(d)
                    try:
                        m = metadatas[i][j]
                        sources.append(m)
                    except Exception:
                        pass
        else:
            for i, d in enumerate(docs):
                docs_texts.append(d)
                try:
                    sources.append(metadatas[i])
                except Exception:
                    pass

    context = "\n\n".join(docs_texts[:n_retrieve]) if docs_texts else ""
    return context, sources


def answer_question(vector_store: ChromaStore, question: str, n_retrieve: int = 3, max_tokens: int = 256, temperature: float = 0.7, llm_model: str = "gpt-3.5-turbo"):
    # try:
    #     results = vector_store.query(question, n_results=n_retrieve)
    # except Exception:
    #     logger.exception("Retrieval failed")
    #     return {"answer": "", "sources": []}

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # context, sources = _prepare_context_from_results(results, n_retrieve)

    prompt = (
        "Use the following context from retrieved documents to answer the question. "
        "If the answer is not contained, say you don't know. Keep the answer concise.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    key = os.getenv("OPENAI_API_KEY")
    if not key or openai is None:
        return {"answer": "OPENAI_API_KEY not set or openai package missing; cannot generate answer.", "sources": sources}

    openai.api_key = key
    llm = get_chat_llm(default_model=llm_model, default_temp=temperature, default_max_tokens=max_tokens)

    resp = openai.ChatCompletion.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    answer = resp["choices"][0]["message"]["content"].strip()

    parsed = validate_assistant_output(answer)
    if not parsed.sources and sources:
        parsed.sources = sources

    return {"answer": parsed.answer, "sources": parsed.sources}


def answer_without_context(question: str, max_tokens: int = 256, temperature: float = 0.7, llm_model: str = "gpt-4o"):
    """Generate an answer directly without retrieving documents from context."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return {"answer": "OPENAI_API_KEY not set; cannot generate answer.", "sources": []}
    
    llm = get_chat_llm(default_model=llm_model, default_temp=temperature, default_max_tokens=max_tokens)
    
    prompt_text = f"Answer the following question concisely. If you don't know, say so.\n\nQuestion: {question}\n\nAnswer:"
    
    try:
        response = llm.invoke(prompt_text)
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.exception("Error generating answer without context: %s", e)
        answer = f"Error: {str(e)}"
    
    return {"answer": answer, "sources": []}
