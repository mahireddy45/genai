import streamlit as st
import logging
from pathlib import Path
import tempfile
import os

# Load API key from secret.py before anything else
from config.secret import OPENAI_API_KEY
if not os.getenv("OPENAI_API_KEY") and OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

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
    }
)
logger = logging.getLogger(__name__)

if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
# Quick validate OpenAI API key (if available) so user sees a clear message
try:
    from src.embeddings import validate_openai_key
    ok = validate_openai_key()
    if not ok:
        st.warning("OpenAI API key not valid — embeddings requests will fail. Check `config/secret.py` or set the `OPENAI_API_KEY` environment variable.")
        logger.error("OpenAI API key validation failed at startup")
except Exception as e:
    logger.debug("Could not validate OpenAI API key at startup: %s", e)

from src.app_core import (
    process_uploaded_files,
    process_directory,
    create_simple_rag_chain,
)

# Set page config
st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")

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

@st.cache_resource
def init_vector_store(db_path: str, embedding_model: str, collection_name: str):
    return init_vector_store_raw(db_path, embedding_model, collection_name)

def main():
    """Main Streamlit app."""
    logger.info("Starting RAG Chatbot application")
    st.title("📚 RAG Chatbot")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Database settings
        st.subheader("Vector Store")
        db_path = st.text_input("Database Path", value="./data/chroma_db")
        #collection_name = st.text_input("Collection Name", value="documents")

        # Embedding settings
        st.subheader("Embeddings")
        embedding_model = st.selectbox("Embedding Model", ["text-embedding-3-small", "text-embedding-3-large"])

        # LLM settings
        st.subheader("GPT Model")
        llm_backend = st.selectbox("LLM", ["gpt-4o", "gpt-3.5-turbo"]).lower()
        device = st.radio("Device", ["CPU", "GPU"]).lower()

        # Generation settings
        st.subheader("Generation")
        n_retrieve = st.slider("Documents to Retrieve", 1, 10, 3)
        max_tokens = st.slider("Max Tokens", 50, 1024, 256)
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, step=0.1)

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
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": question})

            with st.chat_message("user"):
                st.markdown(question)
            
            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking..."):
                    try:
                        # Create RAG chain and invoke with question
                        chain = create_simple_rag_chain(
                            db_path=db_path,
                            question=question,
                            llm_model=llm_backend,
                            temperature=temperature,
                            n_retrieve=n_retrieve,
                            max_tokens=max_tokens
                        )
                        
                        # Use invoke() to get response
                        response = chain.invoke({
                            "context": "",
                            "question": question
                        })
                        
                        # Extract answer text
                        answer_text = response.content if hasattr(response, "content") else str(response)
                        
                        # Display answer
                        st.markdown(answer_text)
                        
                        # Add assistant message to history
                        st.session_state.messages.append({"role": "assistant", "content": answer_text})
                    except Exception as e:
                        error_msg = f"Error generating answer: {str(e)}"
                        st.error(error_msg)
                        logger.exception("Error in RAG chain invocation")
    # Ingest Tab
    with tab2:
        st.subheader("Ingest Documents")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("**Option 1: Upload Files**")
            uploaded_files = st.file_uploader(
                "Upload files (PDF, DOCX, TXT, images)",
                type=["pdf", "docx", "png", "jpg", "jpeg", "tiff", "txt", "md", "doc"],
                accept_multiple_files=True,
                help="Select one or more files to ingest"
            )
            logger.info("DB Path: %s and Embedding Model: %s", db_path, embedding_model)
            if uploaded_files:
                if st.button("📥 Ingest Uploaded Files"):
                    with st.spinner("Processing..."):
                        try:
                            docs_count, chunks_count = process_uploaded_files(
                                uploaded_files,
                                embedding_model,
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

        with col2:
            st.write("**Option 2: Ingest from Directory**")
            directory = st.text_input(
                "Directory Path",
                help="Full path to directory containing PDFs"
            )

            chunk_size = st.number_input("Chunk Size", value=500, min_value=100)
            chunk_overlap = st.number_input("Chunk Overlap", value=50, min_value=0)

            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap

            if st.button("📥 Ingest from Directory"):
                if not directory:
                    st.error("Please enter a directory path")
                else:
                    with st.spinner("Processing PDFs..."):
                        try:
                            added = process_directory(
                                pdf_directory,
                                vector_store,
                                embedding_model,
                                llm_backend,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                            )
                            if added:
                                st.success(f"✓ Added {added} document chunks")
                            else:
                                st.error("No valid documents found or added from directory")
                        except Exception as e:
                            st.error(f"Error: {e}")
                            logger.exception("Error ingesting directory")

    # Info Tab
    # with tab3:
    #     st.subheader("System Information")

    #     try:
    #         vector_store = init_vector_store(db_path, embedding_model, collection_name)
    #         info = vector_store.get_collection_info()

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