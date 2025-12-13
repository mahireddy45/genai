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
from .retriever import retrieve
from .prompts import prompt_library
from .embedding_config import EmbeddingConfig
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

try:
    import openai
except Exception:
    openai = None

logger = get_logger(__name__)

# Global ChatOpenAI instance (lazy-initialized). Use `init_chat_llm` to configure.
chat_llm = None

def init_chat_llm(llm_model: str = "gpt-4o", temperature: float = 0.0, max_tokens: int | None = None):
    
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

def process_uploaded_files(uploaded_files: List, embedding_model: str, llm_model: str, db_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> tuple:
    
    logger.info("chuck_size: %d, chunk_overlap: %d", chunk_size, chunk_overlap)
    all_documents = []
    logger.info("Loading and processing %d uploaded files", len(uploaded_files))
    
    try:
        all_documents = load_documents(uploaded_files, llm_model)
        logger.info("Loaded %d documents from files", len(all_documents))
    except Exception:
        logger.exception("Failed to load documents")
        return (0, 0)

    # Chunk and ingest documents\n    
    try:
        chunks = generate_embeddings(all_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap, embedding_model=embedding_model)
        logger.info("Generated %d chunks from documents", len(chunks))
    except Exception as e:
        logger.error("Failed to generate embeddings: %s", e)
        return (len(all_documents), 0)

    # Store in vector database
    try:
        stored_count = store_in_vector_db(chunks, db_path, embedding_model=embedding_model)
        logger.info("Successfully stored %d chunks from %d documents in vector database", stored_count, len(all_documents))
        return (len(all_documents), stored_count)
    except Exception as e:
        logger.error("Failed to store documents in vector database: %s", e)
        return (len(all_documents), 0)


def process_directory(path: str, embedding_model: str, llm_model: str, db_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> tuple:
    logger.info("chuck_size: %d, chunk_overlap: %d", chunk_size, chunk_overlap)
    directory_path = Path(path)
    uploaded_files = [str(f) for f in directory_path.glob("**/*") if f.is_file()]
    logger.info("Found %d files in directory %s", len(uploaded_files), path)
    return process_uploaded_files(uploaded_files, embedding_model, llm_model, db_path, chunk_size, chunk_overlap)

def create_simple_rag_chain(db_path: str, question: str, llm_model: str, temperature: float, n_retrieve: int = 3, max_tokens: int = 256):
    logger.info(f"Creating RAG chain for question: {question[:50]}...")
    logger.info(f"DB Path: {db_path}, LLM Model: {llm_model}, Temperature: {temperature}, n_retrieve: {n_retrieve}, max_tokens: {max_tokens}")
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment")
            raise ValueError("OPENAI_API_KEY is not set")
        
        # For grounded responses, use lower temperature
        effective_temperature = min(temperature, 0.2) if temperature > 0.2 else temperature
        logger.info("Using temperature: %s (effective: %s for grounding)", temperature, effective_temperature)
        
        llm = ChatOpenAI(api_key=api_key, model=llm_model, temperature=effective_temperature, max_tokens=max_tokens)
        
        # Use context-grounded prompt from prompts module to ensure response is grounded
        # grounded_prompt_template = prompt_library.get_prompt("context_grounded")
        grounded_prompt_template = prompt_library.get_prompt("rag_qa")
        # grounded_prompt_template = prompt_library.get_prompt("response_validation")
        # grounded_prompt_template = prompt_library.get_prompt("summarization")
        # grounded_prompt_template = prompt_library.get_prompt("clarification")
        
        if not grounded_prompt_template:
            logger.warning("Context-grounded prompt not found, using default")
            grounded_prompt_text = """You are a helpful assistant. Answer the following question using ONLY the provided context. If the answer cannot be found in the context, explicitly say you don't have that information.
            Context: {context}
            Question: {question}
            Answer:"""
        else:
            grounded_prompt_text = grounded_prompt_template.template
        
        prompt = ChatPromptTemplate.from_template(grounded_prompt_text)
        logger.info("Using prompt template for grounded response")
        
        # Create a callable that retrieves and formats context
        def get_context(q):
            try:
                # Load the embedding model that was used during ingestion
                config = EmbeddingConfig(db_path)
                embedding_model_for_retrieval = config.get_model()
                logger.info("Using embedding model for retrieval: %s", embedding_model_for_retrieval)
                
                # Call retrieve function directly (it returns a list of chunks)
                top_chunks = retrieve(query=q, k=n_retrieve, db_path=db_path, embedding_model=embedding_model_for_retrieval)
                logger.info("Retrieved %d chunks for grounding response", len(top_chunks))
                context = "\n\n".join([c.get("content", str(c)) for c in top_chunks if c])
                logger.info("Context length: %d characters, Context preview: %s", len(context), context[:200] if context else "EMPTY")
                
                # Log full context for debugging
                if top_chunks:
                    for idx, chunk in enumerate(top_chunks):
                        logger.info("Chunk %d: %s...", idx + 1, chunk.get("content", str(chunk))[:150])
                
                if not context.strip():
                    logger.warning("No relevant context retrieved for question: %s", q)
                    context = "No relevant context available in the knowledge base."
                return context
            except Exception as e:
                logger.error(f"Error retrieving context: {e}")
                return "Error retrieving context from knowledge base."
        
        # Create chain using dictionary input instead of pipe operator
        def chain_invoke(input_dict):
            question_text = input_dict.get("question", question)
            context_text = get_context(question_text)
            
            # Create input for prompt - response is grounded in context
            prompt_input = {
                "context": context_text,
                "question": question_text
            }
            
            logger.info("Generating grounded response using context and question")
            logger.info("Question: %s", question_text)
            logger.info("Context to be sent to LLM: %s", context_text[:300] if context_text else "EMPTY")
            
            # Invoke prompt then LLM
            formatted_prompt = prompt.invoke(prompt_input)
            logger.debug("Formatted prompt: %s", str(formatted_prompt)[:500])
            
            response = llm.invoke(formatted_prompt)
            
            # Extract text from response
            response_text = response.content if hasattr(response, 'content') else str(response)
            logger.info("Generated response (grounded in context): %s", response_text[:100])
            
            return response
        
        # Return a callable object that has invoke method
        class SimpleRagChain:
            def invoke(self, input_dict):
                return chain_invoke(input_dict)
        
        return SimpleRagChain()
        
    except Exception as e:
        logger.exception(f"Error creating RAG chain: {e}")
        raise
