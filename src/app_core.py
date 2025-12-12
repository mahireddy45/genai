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
from .retriever import Retriever
from config.secret import OPENAI_API_KEY
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

try:
    import openai
except Exception:
    openai = None

logger = get_logger(__name__)
os.environ["OPENAI_API_KEY"] = "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"
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
    logger.info("chuck_size: %d, chunk_overlap: %d", chunk_size, chunk_overlap)
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
    logger.info("chuck_size: %d, chunk_overlap: %d", chunk_size, chunk_overlap)
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



def answer_question(question: str, db_path: str = None, collection_name: str = None, 
                   llm_model: str = "gpt-4o", temperature: float = 0.0, 
                   n_retrieve: int = 4) -> dict:
    """Answer a question using the RAG chain with ingested documents from ChromaDB.
    
    Args:
        question: The question to answer
        db_path: Path to ChromaDB database (uses settings default if None)
        collection_name: Collection name in ChromaDB (uses settings default if None)
        llm_model: LLM model to use (default: gpt-4o)
        temperature: Temperature for LLM (default: 0.0 for deterministic)
        n_retrieve: Number of documents to retrieve (default: 4)
        
    Returns:
        Dictionary with answer, sources, and retrieval stats
    """
    try:
        # Import here to avoid circular imports
        from .rag_chain import answer_question_from_docs
        from config.settings import CHROMA_DB_PATH, COLLECTION_NAME
        
        # Use provided paths or defaults from settings
        db_path = db_path or CHROMA_DB_PATH
        collection_name = collection_name or COLLECTION_NAME
        
        logger.info("Answering question using RAG chain: %s", question[:100])
        
        result = answer_question_from_docs(
            question=question,
            db_path=db_path,
            collection_name=collection_name,
            llm_model=llm_model,
            temperature=temperature,
            num_docs=n_retrieve
        )
        
        logger.info("Question answered successfully. Retrieved %d documents", result.get("num_docs_retrieved", 0))
        return result
        
    except Exception as e:
        logger.exception("Error answering question: %s", e)
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": [],
            "num_docs_retrieved": 0,
            "num_docs_used": 0
        }


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


def create_simple_rag_chain(db_path: str, question: str, llm_model: str = "gpt-3.5-turbo", temperature: float = 0.7, n_retrieve: int = 3, max_tokens: int = 256):
    """Create a simple RAG chain that retrieves context and generates an answer using invoke()."""
    logger.info(f"Creating RAG chain for question: {question[:50]}...")
    
    try:
        retriever = Retriever(db_path=db_path, embedding_model=llm_model)
        llm = ChatOpenAI(model=llm_model, temperature=temperature, max_tokens=max_tokens)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
If the context includes image descriptions, consider those as visual information.
Context: {context}"""),
            ("human", "{question}")
        ])
        
        # Create a callable that retrieves and formats context
        def get_context(q):
            try:
                top_chunks = retriever.retrieve(q, k=n_retrieve)
                context = "\n\n".join([c.get("content", str(c)) for c in top_chunks if c])
                return context
            except Exception as e:
                logger.error(f"Error retrieving context: {e}")
                return ""
        
        # Create chain using dictionary input instead of pipe operator
        def chain_invoke(input_dict):
            question_text = input_dict.get("question", question)
            context_text = get_context(question_text)
            
            # Create input for prompt
            prompt_input = {
                "context": context_text,
                "question": question_text
            }
            
            # Invoke prompt then LLM
            formatted_prompt = prompt.invoke(prompt_input)
            response = llm.invoke(formatted_prompt)
            return response
        
        # Return a callable object that has invoke method
        class SimpleRagChain:
            def invoke(self, input_dict):
                return chain_invoke(input_dict)
        
        return SimpleRagChain()
        
    except Exception as e:
        logger.exception(f"Error creating RAG chain: {e}")
        raise
