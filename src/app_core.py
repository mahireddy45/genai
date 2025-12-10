from __future__ import annotations
from typing import List, Dict, Iterable, Tuple
import os
import tempfile
import logging
from pathlib import Path

from .chroma_store import ChromaStore
#from .pdf_loader import load_and_process_pdfs
from .office_loader import load_docx
from .image_loader import load_image
from .schemas import IngestedDocument
from .embeddings import get_text_embeddings
from .response_validation import validate_assistant_output
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

try:
    import openai
except Exception:
    openai = None

logger = logging.getLogger(__name__)

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
    """Create or open a ChromaStore (no UI caching here)."""
    return ChromaStore(persist_directory=db_path, collection_name=collection_name)


def _chunk_text_simple(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> Iterable[Tuple[int, str]]:
    start = 0
    length = len(text)
    idx = 1
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            yield idx, chunk
            idx += 1
        start = max(end - chunk_overlap, end)


def ingest_text_to_store(vector_store: ChromaStore, text: str, filename: str, embedding_model: str, chunk_size: int = 500, chunk_overlap: int = 50) -> int:
    docs = []
    for i, chunk in _chunk_text_simple(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
        doc_id = f"{filename}::c{i}"
        meta = {"filename": filename, "page": None, "source": filename, "text_length": len(chunk)}
        d = {"id": doc_id, "text": chunk, "meta": meta}
        try:
            parsed = IngestedDocument.parse_obj(d)
            docs.append(parsed.dict())
        except Exception as e:
            logger.warning("Skipping invalid chunk: %s", e)

    if docs:
        added = vector_store.add_documents(docs, model=embedding_model)
        return added
    return 0


def process_uploaded_files(uploaded_files: List, vector_store: ChromaStore, embedding_model: str, chunk_size: int = 500, chunk_overlap: int = 50) -> int:
    all_documents = []
    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()
        # save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        try:
            if suffix == ".pdf":
                docs = load_and_process_pdfs(tmp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            elif suffix in [".txt", ".md"]:
                # Load plain text files and chunk them into documents
                docs = []
                try:
                    with open(tmp_path, "r", encoding="utf-8") as f:
                        text = f.read()
                except Exception:
                    # fallback to binary decode
                    with open(tmp_path, "rb") as f:
                        text = f.read().decode("utf-8", errors="ignore")

                for i, chunk in _chunk_text_simple(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
                    doc_id = f"{Path(uploaded_file.name).name}::c{i}"
                    meta = {"filename": uploaded_file.name, "page": None, "source": uploaded_file.name, "text_length": len(chunk)}
                    docs.append({"id": doc_id, "text": chunk, "meta": meta})
            elif suffix == ".docx":
                docs = load_docx(tmp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            elif suffix in [".png", ".jpg", ".jpeg", ".tiff"]:
                docs = load_image(tmp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            else:
                docs = []

            all_documents.extend(docs)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # Validate and add
    validated = []
    for d in all_documents:
        try:
            parsed = IngestedDocument.parse_obj(d)
            validated.append(parsed.dict())
        except Exception as e:
            logger.warning("Skipping invalid document: %s", e)

    if validated:
        added = vector_store.add_documents(validated, model=embedding_model)
        return added
    return 0


def process_directory(path: str, vector_store: ChromaStore, embedding_model: str, chunk_size: int = 500, chunk_overlap: int = 50) -> int:
    """Process all PDFs in `path` (a directory or single file) and ingest.

    Returns number of chunks added.
    """
    docs = []
    try:
        docs = load_and_process_pdfs(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except Exception:
        logger.exception("Failed to load PDFs from directory: %s", path)
        return 0

    validated = []
    for d in docs:
        try:
            parsed = IngestedDocument.parse_obj(d)
            validated.append(parsed.dict())
        except Exception as e:
            logger.warning("Skipping invalid document: %s", e)

    if validated:
        added = vector_store.add_documents(validated, model=embedding_model)
        return added
    return 0


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
