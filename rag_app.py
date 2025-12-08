"""Streamlit web interface for the local RAG chatbot."""

import streamlit as st
import logging
from pathlib import Path
import tempfile
import os

# from src.pdf_loader import load_and_process_pdfs
# from src.chroma_store import create_vector_store
# from src.local_llm import create_local_llm, LocalLLM
# from src.rag_pipeline import create_rag_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)


# @st.cache_resource
# def init_vector_store(db_path: str, embedding_model: str):
#     """Initialize vector store (cached)."""
#     return create_vector_store(
#         db_path=db_path,
#         embedding_model=embedding_model
#     )


# @st.cache_resource
# def init_local_llm(backend: str, model_name: str, device: str):
#     """Initialize local LLM (cached)."""
#     return create_local_llm(
#         backend=backend,
#         model_name=model_name,
#         device=device
#     )


def main():
    """Main Streamlit app."""
    st.title("📚 RAG Chatbot")
    st.markdown("Chat with your documents")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Database settings
        st.subheader("Vector Store")
        db_path = st.text_input(
            "Database Path",
            value="./data/chroma_db",
            help="Path to store Chroma vector database"
        )
        collection_name = st.text_input(
            "Collection Name",
            value="documents",
            help="Name of document collection"
        )

        # Embedding settings
        st.subheader("Embeddings")
        embedding_model = st.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "distiluse-base-multilingual-cased-v2"],
            help="Local embedding model to use"
        )

        # LLM settings
        st.subheader("Local LLM")
        llm_backend = st.radio(
            "LLM Backend",
            ["GPT4All", "Ollama"],
            help="Local LLM backend to use"
        )
        llm_backend = llm_backend.lower()

        if llm_backend == "gpt4all":
            llm_model = st.selectbox(
                "GPT4All Model",
                [
                    "orca-mini-3b.gguf",
                    "mistral-7b-openorca.Q4_0.gguf",
                    "neural-chat-7b-v3-1.Q4_0.gguf",
                    "ggml-model-q4_0.gguf"
                ],
                help="GPT4All model (auto-downloaded on first use)"
            )
        else:
            llm_model = st.text_input(
                "Ollama Model",
                value="mistral",
                help="Ollama model (must be pulled: ollama pull <model>)"
            )

        device = st.radio(
            "Device",
            ["CPU", "GPU"],
            help="Device to run LLM on"
        ).lower()

        # Generation settings
        st.subheader("Generation")
        n_retrieve = st.slider(
            "Documents to Retrieve",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of relevant documents to retrieve"
        )
        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=1024,
            value=256,
            help="Maximum tokens in response"
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Sampling temperature (creativity)"
        )

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📥 Ingest", "ℹ️ Info"])

    # Chat Tab
    with tab1:
        st.subheader("Chat with Your Documents")

        try:
            # Initialize components
            vector_store = init_vector_store(db_path, embedding_model)
            info = vector_store.get_collection_info()

            if info["document_count"] == 0:
                st.warning("⚠️ No documents in vector store. Use the 'Ingest' tab to add PDFs.")
            else:
                st.success(f"✓ Loaded {info['document_count']} document chunks")

                # Initialize LLM
                llm = init_local_llm(llm_backend, llm_model, device)

                # Create RAG pipeline
                rag_pipeline = create_rag_pipeline(
                    vector_store=vector_store,
                    local_llm=llm,
                    n_retrieve=n_retrieve
                )

                # Chat history
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Display chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Chat input
                question = st.chat_input("Ask a question about your documents...")

                if question:
                    # Add user message to history
                    st.session_state.messages.append(
                        {"role": "user", "content": question}
                    )

                    with st.chat_message("user"):
                        st.markdown(question)

                    # Generate response
                    with st.chat_message("assistant"):
                        with st.spinner("🤖 Thinking..."):
                            result = rag_pipeline.query(
                                question,
                                max_tokens=max_tokens,
                                temperature=temperature
                            )

                        # Display answer
                        st.markdown(result["answer"])

                        # Display sources
                        if result["sources"]:
                            with st.expander("📖 Sources"):
                                for i, source in enumerate(result["sources"], 1):
                                    st.caption(
                                        f"{i}. {source['filename']} (page {source['page']}, "
                                        f"distance: {source['distance']:.3f})"
                                    )

                    # Add assistant message to history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result["answer"]}
                    )

        except Exception as e:
            st.error(f"Error: {e}")
            logger.exception("Error in chat tab")

    # Ingest Tab
    with tab2:
        st.subheader("Ingest PDF Documents")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("**Option 1: Upload Files**")
            uploaded_files = st.file_uploader(
                "Upload PDF files",
                type="pdf",
                accept_multiple_files=True,
                help="Select one or more PDF files to ingest"
            )

            if uploaded_files:
                if st.button("📥 Ingest Uploaded Files"):
                    with st.spinner("Processing..."):
                        try:
                            all_documents = []

                            # Save uploaded files to temp directory
                            for uploaded_file in uploaded_files:
                                with tempfile.NamedTemporaryFile(
                                    delete=False,
                                    suffix=".pdf"
                                ) as tmp_file:
                                    tmp_file.write(uploaded_file.getbuffer())
                                    tmp_path = tmp_file.name

                                # Load PDF
                                docs = load_and_process_pdfs(
                                    tmp_path,
                                    chunk_size=st.session_state.get("chunk_size", 500),
                                    chunk_overlap=st.session_state.get("chunk_overlap", 50)
                                )
                                all_documents.extend(docs)

                                # Clean up
                                os.unlink(tmp_path)

                            if all_documents:
                                # Add to vector store
                                vector_store = init_vector_store(db_path, embedding_model)
                                added = vector_store.add_documents(all_documents)
                                st.success(f"✓ Added {added} document chunks")
                            else:
                                st.error("No documents extracted from PDFs")

                        except Exception as e:
                            st.error(f"Error: {e}")
                            logger.exception("Error ingesting files")

        with col2:
            st.write("**Option 2: Ingest from Directory**")
            pdf_directory = st.text_input(
                "PDF Directory Path",
                help="Full path to directory containing PDFs"
            )

            chunk_size = st.number_input("Chunk Size", value=500, min_value=100)
            chunk_overlap = st.number_input("Chunk Overlap", value=50, min_value=0)

            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap

            if st.button("📥 Ingest from Directory"):
                if not pdf_directory:
                    st.error("Please enter a directory path")
                else:
                    with st.spinner("Processing PDFs..."):
                        try:
                            documents = load_and_process_pdfs(
                                pdf_directory,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap
                            )

                            if documents:
                                vector_store = init_vector_store(db_path, embedding_model)
                                added = vector_store.add_documents(documents)
                                st.success(f"✓ Added {added} document chunks")
                            else:
                                st.error("No PDFs found in directory")

                        except Exception as e:
                            st.error(f"Error: {e}")
                            logger.exception("Error ingesting directory")

    # Info Tab
    with tab3:
        st.subheader("System Information")

        try:
            vector_store = init_vector_store(db_path, embedding_model)
            info = vector_store.get_collection_info()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Documents", info["document_count"])

            with col2:
                st.text(f"Collection: {info['collection_name']}")

            with col3:
                st.text(f"Database: {info['db_path']}")

            st.divider()

            st.subheader("Available Models")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**GPT4All Models** (auto-downloaded):")
                for model, desc in LocalLLM.get_gpt4all_models().items():
                    st.caption(f"• {model}: {desc}")

            with col2:
                st.write("**Ollama Models** (requires pull):")
                for model, desc in LocalLLM.get_ollama_models().items():
                    st.caption(f"• {model}: {desc}")

            st.divider()

            st.subheader("Embedding Models")
            from src.embeddings import LocalEmbeddings

            for model, desc in LocalEmbeddings.get_available_models().items():
                st.caption(f"• {model}: {desc}")

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()