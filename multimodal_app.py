import streamlit as st
import os
from pathlib import Path
from typing import List, Union
import tempfile
import shutil
from PIL import Image

from src.multimodal import (
    MultimodalEmbeddings,
    load_text_documents,
    load_image_documents,
    create_chroma_vectorstore,
    create_simple_rag_chain,
)


st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2.0rem; font-weight: 700; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .result-box { background-color: #f0f2f6; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .source-box { background-color: #e8f4f8; padding: 0.75rem; border-radius: 6px; margin: 0.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "multimodal_emb" not in st.session_state:
    st.session_state.multimodal_emb = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False


def _save_uploads_to_temp(uploaded_files):
    tmp_dir = tempfile.mkdtemp()
    paths = []
    for file in uploaded_files:
        tmp_path = os.path.join(tmp_dir, file.name)
        with open(tmp_path, "wb") as f:
            f.write(file.getbuffer())
        paths.append(tmp_path)
    return tmp_dir, paths


def main():
    st.markdown('<p class="main-header">🔍 Multimodal RAG System</p>', unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Configuration")

        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key",
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        st.markdown("---")
        st.header("📚 Load Documents")

        text_files = st.file_uploader(
            "Upload PDF, TXT, or MD files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        image_files = st.file_uploader(
            "Upload images",
            type=["png", "jpg", "jpeg", "gif", "bmp", "webp"],
            accept_multiple_files=True,
        )

        image_folder = st.text_input("Or enter image folder path", placeholder="C:/path/to/images")

        if st.button("🔄 Load Documents"):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("Please set your OpenAI API key in the sidebar first.")
            else:
                st.session_state.multimodal_emb = MultimodalEmbeddings()

                all_documents = []

                # Text documents
                temp_dirs = []
                if text_files:
                    with st.spinner("Loading text documents..."):
                        tmp_dir, text_paths = _save_uploads_to_temp(text_files)
                        temp_dirs.append(tmp_dir)
                        text_docs = load_text_documents(text_paths)
                        all_documents.extend(text_docs)
                        st.success(f"Loaded {len(text_docs)} text documents")

                # Image documents (uploaded)
                if image_files:
                    with st.spinner("Processing uploaded images..."):
                        tmp_dir, image_paths = _save_uploads_to_temp(image_files)
                        temp_dirs.append(tmp_dir)
                        image_docs = load_image_documents(image_paths, st.session_state.multimodal_emb)
                        all_documents.extend(image_docs)
                        st.success(f"Loaded {len(image_docs)} image documents")

                # Image documents (folder)
                if image_folder:
                    with st.spinner(f"Processing images from folder: {image_folder}..."):
                        folder_docs = load_image_documents(image_folder, st.session_state.multimodal_emb)
                        all_documents.extend(folder_docs)
                        st.success(f"Loaded {len(folder_docs)} images from folder")

                if all_documents:
                    with st.spinner("Creating vector store..."):
                        st.session_state.vectorstore = create_chroma_vectorstore(all_documents)
                        st.session_state.qa_chain = create_simple_rag_chain(st.session_state.vectorstore)
                        st.session_state.documents_loaded = True
                    st.success(f"✅ Successfully loaded {len(all_documents)} documents!")
                else:
                    st.warning("No documents were loaded. Please upload supported files or provide a valid folder path.")

                # Clean up temp dirs
                for d in temp_dirs:
                    try:
                        shutil.rmtree(d)
                    except Exception:
                        pass

        st.markdown("---")
        st.header("📊 Status")
        if st.session_state.documents_loaded:
            st.success("✅ Documents loaded and ready")
        else:
            st.info("⏳ No documents loaded yet")

    # Main area
    if not st.session_state.documents_loaded:
        st.info("👈 Please load documents from the sidebar to get started.")
        st.markdown(
            """
            ### How to use:
            1. Enter your OpenAI API key in the sidebar
            2. Upload text documents (PDF, TXT, MD) or images
            3. Optionally provide a folder path for images
            4. Click "Load Documents" to process them
            5. Start querying!
            """
        )
        return

    st.header("💬 Query Your Documents")
    query_type = st.radio("Query Type", ["Text Query", "Image Query"], horizontal=True)

    if query_type == "Text Query":
        query = st.text_area("Enter your question", height=100)
        if st.button("🔍 Search"):
            if not query:
                st.warning("Please enter a query")
            else:
                with st.spinner("Searching and generating answer..."):
                    try:
                        result = st.session_state.qa_chain(query)
                        answer = result.get("answer") if isinstance(result, dict) else str(result)

                        st.markdown("### 📝 Answer")
                        st.markdown(f'<div class="result-box">{answer}</div>', unsafe_allow_html=True)

                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
                        source_docs = retriever.get_relevant_documents(query)

                        st.markdown("### 📚 Source Documents")
                        for i, doc in enumerate(source_docs, 1):
                            with st.expander(f"Source {i}: {Path(doc.metadata.get('source','Unknown')).name}"):
                                st.write(f"**Type:** {doc.metadata.get('type', 'text')}")
                                st.write(f"**Path:** {doc.metadata.get('source','N/A')}")
                                st.write("**Content:**")
                                st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                if doc.metadata.get('type') == 'image':
                                    image_path = doc.metadata.get('image_path')
                                    if image_path and Path(image_path).exists():
                                        try:
                                            img = Image.open(image_path)
                                            st.image(img, caption=Path(image_path).name, use_container_width=True)
                                        except Exception:
                                            pass
                    except Exception as e:
                        st.error(f"Error: {e}")

    else:  # Image Query
        uploaded_image = st.file_uploader("Upload an image to search for similar content", type=["png", "jpg", "jpeg", "gif", "bmp", "webp"]) 
        if uploaded_image:
            img = Image.open(uploaded_image)
            st.image(img, caption="Query Image", use_container_width=True)

            if st.button("🔍 Search Similar Content"):
                with st.spinner("Processing image and searching..."):
                    try:
                        temp_dir, image_paths = _save_uploads_to_temp([uploaded_image])
                        temp_path = image_paths[0]
                        image_description = st.session_state.multimodal_emb.get_image_description(temp_path)

                        query_text = f"Based on this image description: {image_description}. What related information can you find?"
                        result = st.session_state.qa_chain(query_text)
                        answer = result.get("answer") if isinstance(result, dict) else str(result)

                        st.markdown("### 📝 Answer")
                        st.markdown(f'<div class="result-box">{answer}</div>', unsafe_allow_html=True)

                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
                        source_docs = retriever.get_relevant_documents(image_description)

                        st.markdown("### 📚 Source Documents")
                        for i, doc in enumerate(source_docs, 1):
                            with st.expander(f"Source {i}: {Path(doc.metadata.get('source','Unknown')).name}"):
                                st.write(f"**Type:** {doc.metadata.get('type', 'text')}")
                                st.write(f"**Path:** {doc.metadata.get('source', 'N/A')}")
                                st.write("**Content:**")
                                st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                if doc.metadata.get('type') == 'image':
                                    image_path = doc.metadata.get('image_path')
                                    if image_path and Path(image_path).exists():
                                        try:
                                            img = Image.open(image_path)
                                            st.image(img, caption=Path(image_path).name, use_container_width=True)
                                        except Exception:
                                            pass

                        try:
                            shutil.rmtree(temp_dir)
                        except Exception:
                            pass

                    except Exception as e:
                        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
