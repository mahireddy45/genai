import streamlit as st
import logging
from pathlib import Path
import tempfile
import os

# Load .env file and set environment variables FIRST (before anything else)
# This makes OPENAI_API_KEY available to all modules via os.getenv("OPENAI_API_KEY")
from config.settings import OPENAI_API_KEY  # noqa: F401 - imported to trigger .env loading

# Configure logging
from src.logging_config import setup_logging

# Call once at startup
setup_logging(
    log_dir='./logs',
    log_level='INFO',
    module_levels={
        'src.app_core': 'DEBUG',
        'src.document_loader': 'DEBUG',
        'src.image_loader': 'DEBUG',
        'src.chroma_store': 'DEBUG',
        'src.embeddings': 'INFO',
        'src.schemas': 'WARNING',
        'src.response_validation': 'INFO',
        'src.guardrail_helpers': 'INFO',
    }
)
logger = logging.getLogger(__name__)

logger.info("Using OpenAI API key: %s", OPENAI_API_KEY if OPENAI_API_KEY else "None")

if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
    
# Quick validate OpenAI API key (if available) so user sees a clear message
try:
    from src.embeddings import validate_openai_key
    ok = validate_openai_key()
except Exception as e:
    logger.debug("Could not validate OpenAI API key at startup: %s", e)

from src.app_core import (
    process_uploaded_files,
    # process_directory,
    create_simple_rag_chain
)

from src.guardrail_helpers import (
    moderate_text,
    validate_user_query,
    check_pii_in_text,
    log_audit_entry,
    redact_pii
)

# Set page config
st.set_page_config(page_title="Capstone - Enterprise RAG Assistant", page_icon="📚", layout="wide")

# Custom CSS (narrow)
st.markdown(
    """
    <style>
    .main { max-width: 1200px; }
    .stTabs [data-baseweb="tab-list"] button { font-size: 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

def main():
    """Main Streamlit app."""
    logger.info("Starting Capstone - Enterprise RAG Assistant application")
    st.title("📚 Capstone - Enterprise RAG Assistant")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Database settings
        st.subheader("Vector Store")
        db_path = st.text_input("Database Path", value="./data/chroma_db")
        #collection_name = st.text_input("Collection Name", value="documents")

        # Embedding settings
        st.subheader("Embeddings")
        embedding_model = st.selectbox(
            "Select Embedding Model",
            ["text-embedding-3-small", "text-embedding-3-large"],
            help="Choose the embedding model to use"
        )        

        # LLM settings
        st.subheader("GPT Model")
        llm_backend = st.selectbox("LLM", ["gpt-4o", "gpt-3.5-turbo"]).lower()
        # device = st.radio("Device", ["CPU", "GPU"]).lower()

        # Generation settings
        st.subheader("Generation")
        n_retrieve = st.slider("Documents to Retrieve", 1, 10, 3)
        temperature = st.slider("Temperature", 0.0, 2.0, 0.3, step=0.1)
        max_tokens = st.slider("Max Tokens", 50, 1024, 256)

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📥 Ingest", "ℹ️ Info"]) 

    # Chat Tab
    with tab1:
        # Chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input (automatically positioned at bottom by Streamlit)
        question = st.chat_input("Ask a question ...")

        if question:
            # ========== GUARDRAIL CHECKS ==========
            # 1. Validate query format and length
            is_valid, validation_error = validate_user_query(question, max_len=1500)
            if not is_valid:
                st.error(f"❌ Query rejected: {validation_error}")
                logger.warning(f"Query validation failed: {validation_error}")
                log_audit_entry("query_rejected", {"reason": validation_error, "query": question[:100]})
            else:
                # 2. Check for PII in query
                pii_detected = check_pii_in_text(question)
                if pii_detected:
                    st.warning(f"⚠️ Sensitive information detected in your query: {list(pii_detected.keys())}")
                    logger.warning(f"PII detected in query: {pii_detected}")
                    log_audit_entry("pii_detected_in_query", {"pii_types": list(pii_detected.keys())})
                
                # 3. Run moderation check on query
                is_flagged, reasons = moderate_text(question, model = llm_backend)
                if is_flagged:
                    st.error(f"❌ Query flagged by content moderation: {', '.join(reasons)}")
                    logger.warning(f"Query flagged by moderation: {reasons}")
                    log_audit_entry("query_flagged_by_moderation", {"reasons": reasons, "query": question[:100]})
                else:
                    # ========== QUERY PASSED ALL GUARDRAILS - PROCEED ==========
                    log_audit_entry("query_accepted", {"query": question[:100]})
                    
                    # Add user message to history
                    st.session_state.messages.append({"role": "user", "content": question})

                    with st.chat_message("user"):
                        st.markdown(question)
                    
                    with st.chat_message("assistant"):
                        with st.spinner("🤖 Thinking..."):
                            try:
                                logger.info("Generating grounded answer for question: %s with llm model: %s", question[:50], llm_backend)
                                # Create RAG chain and invoke with question
                                chain = create_simple_rag_chain(
                                    db_path, question, llm_backend, temperature, n_retrieve, max_tokens
                                )
                                
                                # Use invoke() to get response
                                response = chain.invoke({
                                    "context": "",
                                    "question": question
                        })
                        
                                # Extract answer text
                                answer_text = response.content if hasattr(response, "content") else str(response)
                                
                                # Check for PII in response and redact if found
                                pii_in_response = check_pii_in_text(answer_text)
                                if pii_in_response:
                                    logger.warning(f"PII detected in LLM response: {pii_in_response}")
                                    st.warning(f"⚠️ Response contains sensitive information that will be redacted: {list(pii_in_response.keys())}")
                                    safe_answer = redact_pii(answer_text)
                                else:
                                    safe_answer = answer_text
                                
                                # Log audit entry for successful response
                                log_audit_entry("response_generated", {
                                    "query": redact_pii(question[:1000]),
                                    "response_length": len(answer_text),
                                    "pii_detected": bool(pii_in_response),
                                    "llm_model": llm_backend
                                })
                                
                                # Display redacted answer
                                st.markdown(safe_answer)
                                
                                # Add assistant message to history
                                st.session_state.messages.append({"role": "assistant", "content": answer_text})
                            except Exception as e:
                                error_msg = f"Error generating answer: {str(e)}"
                                st.error(error_msg)
                                logger.exception("Error in RAG chain invocation")
                                log_audit_entry("response_generation_error", {"error": str(e)[:100]})
    # Ingest Tab
    with tab2:
        st.subheader("Ingest Documents")
        # col1, col2 = st.columns([1, 1])

        # with col1:
        st.write("**Option 1: Upload Files**")
        uploaded_files = st.file_uploader(
            "Upload files (PDF, DOCX, TXT, images)",
            type=["pdf", "docx", "png", "jpg", "jpeg", "tiff", "txt", "md", "doc"],
            accept_multiple_files=True,
            help="Select one or more files to ingest"
        )
        chunk_size = st.number_input("Chunk Size", value=500, min_value=100)
        chunk_overlap = st.number_input("Chunk Overlap", value=50, min_value=0)

        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap
        logger.info("DB Path: %s and Embedding Model: %s", db_path, embedding_model)
        if uploaded_files:
            if st.button("📥 Ingest Uploaded Files"):
                with st.spinner("Processing..."):
                    try:
                        # ========== CHECK FOR PII IN DOCUMENTS ==========
                        pii_warnings = {}
                        for uploaded_file in uploaded_files:
                            file_content = uploaded_file.read().decode('utf-8', errors='ignore')
                            pii_found = check_pii_in_text(file_content)
                            if pii_found:
                                pii_warnings[uploaded_file.name] = pii_found
                        
                        if pii_warnings:
                            st.warning("⚠️ **PII Detected in Documents:**")
                            for filename, pii_types in pii_warnings.items():
                                st.write(f"  • **{filename}**: {', '.join(pii_types)}")
                            st.info("Ensure you have proper authorization before ingesting documents with sensitive information.")
                            log_audit_entry("pii_detected_in_documents", {"documents": list(pii_warnings.keys()), "pii_types": list(set([t for types in pii_warnings.values() for t in types]))})
                        
                        selected_model = embedding_model
                        docs_count, chunks_count = process_uploaded_files(
                            uploaded_files,
                            selected_model,
                            llm_backend,
                            db_path,
                            chunk_size=st.session_state.get("chunk_size", 500),
                            chunk_overlap=st.session_state.get("chunk_overlap", 50),
                        )
                        logger.info("Ingested %d documents into %d chunks from uploaded files", docs_count, chunks_count)
                        if chunks_count > 0:
                            st.success(f"✓ Added {docs_count} documents as {chunks_count} chunks")
                        else:
                            st.error("No valid documents were added from the uploaded files")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        logger.exception("Error ingesting files")

        # with col2:
        #     st.write("**Option 2: Ingest from Directory**")
        #     directory = st.text_input(
        #         "Directory Path",
        #         help="Full path to directory containing PDFs"
        #     )

        #     chunk_size = st.number_input("Chunk Size", value=500, min_value=100)
        #     chunk_overlap = st.number_input("Chunk Overlap", value=50, min_value=0)

        #     st.session_state.chunk_size = chunk_size
        #     st.session_state.chunk_overlap = chunk_overlap

        #     if st.button("📥 Ingest from Directory"):
        #         if not directory:
        #             st.error("Please enter a directory path")
        #         else:
        #             with st.spinner("Processing PDFs..."):
        #                 try:
        #                     added = process_directory(
        #                         directory,
        #                         embedding_model,
        #                         llm_backend,
        #                         chunk_size=chunk_size,
        #                         chunk_overlap=chunk_overlap,
        #                     )
        #                     if added:
        #                         st.success(f"✓ Added {added} document chunks")
        #                     else:
        #                         st.error("No valid documents found or added from directory")
        #                 except Exception as e:
        #                     st.error(f"Error: {e}")
        #                     logger.exception("Error ingesting directory")

    # Info Tab
    # with tab3:
    #     st.subheader("System Information")

    #     try:
    #         vector_store = init_vector_store(db_path, embedding_model)

    #         col1, col2, col3 = st.columns(3)

    #         with col1:
    #             st.metric("Documents", info["document_count"])

    #         with col2:
    #             st.text(f"Collection: {info['collection_name']}")

    #         with col3:
    #             st.text(f"Database: {info['db_path']}")

    #         st.divider()

    #         st.subheader("Available Models")

    #         col1, col2 = st.columns(2)

    #         with col1:
    #             st.write("**Local LLMs**")
    #             st.caption("GPT4All and Ollama models are managed locally; check your installation and model folder.")

    #         with col2:
    #             st.write("**Embedding Models**")
    #             st.caption("OpenAI: text-embedding-3-large (requires OPENAI_API_KEY)")

    #     except Exception as e:
    #         st.error(f"Error: {e}")


if __name__ == "__main__":
    main()