from __future__ import annotations
import base64
from pathlib import Path
from typing import List, Union
import logging

try:
    import streamlit as st
except Exception:  # streamlit is optional for these helpers
    st = None

try:
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, Document
    from langchain.document_loaders import PyPDFLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Chroma
except Exception:
    OpenAIEmbeddings = None
    ChatOpenAI = None
    HumanMessage = None
    Document = None
    PyPDFLoader = None
    TextLoader = None
    RecursiveCharacterTextSplitter = None
    Chroma = None


def _ensure_langchain_installed():
    if OpenAIEmbeddings is None:  # type: ignore[name-defined]
        raise RuntimeError(
            "Required langchain packages are not installed.\n"
            "Install via: .\\.venv\\Scripts\\pip.exe install langchain langchain-openai langchain-community langchain-core"
        )

logger = logging.getLogger(__name__)


class MultimodalEmbeddings:

    def __init__(self, model_name: str = "text-embedding-3-large", llm_model: str = "gpt-4o"):
        self.text_embeddings = OpenAIEmbeddings(model=model_name)
        self.llm = ChatOpenAI(model=llm_model, temperature=0)

    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_image_description(self, image_path: str) -> str:
        """Get text description of image using a chat-model vision prompt.

        Implementation note: we embed the image by creating a data URL and
        asking the chat model to describe it. This is a lightweight approach
        that avoids adding a separate vision-specific SDK.
        """
        base64_image = self.encode_image_to_base64(image_path)
        prompt = (
            "Describe this image in detail, focusing on key visual elements, text, and "
            "context that would be useful for semantic retrieval.\n\n"
            f"IMAGE_DATA_URL: data:image/jpeg;base64,{base64_image}"
        )

        try:
            # ChatOpenAI provides a `predict` convenience method that returns a string
            description = self.llm.predict(prompt)
        except Exception:
            # Fallback to calling the model with a HumanMessage
            try:
                msg = HumanMessage(content=prompt)
                resp = self.llm([msg])
                # resp may be a string or an object depending on LangChain version
                description = getattr(resp, "content", str(resp))
            except Exception as e:
                logger.exception("Failed to get image description: %s", e)
                description = ""

        return description or ""

    def embed_text(self, text: str) -> List[float]:
        """Embed text using OpenAI embeddings."""
        return self.text_embeddings.embed_query(text)

    def embed_image(self, image_path: str) -> List[float]:
        """Embed image by first converting to text description, then embedding."""
        description = self.get_image_description(image_path)
        return self.embed_text(description)


def load_text_documents(file_paths: List[str]) -> List[Document]:
    """Load text documents from various formats using LangChain loaders."""
    documents: List[Document] = []

    for file_path in file_paths:
        path = Path(file_path)
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(path))
            elif suffix in [".txt", ".md"]:
                loader = TextLoader(str(path))
            else:
                logger.debug("Skipping unsupported text file: %s", path)
                continue

            docs = loader.load()
            documents.extend(docs)
        except Exception:
            logger.exception("Failed to load text document: %s", path)

    return documents


def load_image_documents(image_paths_or_folder: Union[str, List[str]], multimodal_emb: MultimodalEmbeddings) -> List[Document]:
    """
    Load images and create LangChain `Document` objects with descriptions.

    Args:
        image_paths_or_folder: folder path or list of image file paths
        multimodal_emb: instance of `MultimodalEmbeddings` used to describe images
    """
    documents: List[Document] = []
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.svg'}

    if isinstance(image_paths_or_folder, str):
        path = Path(image_paths_or_folder)
        if path.is_dir():
            image_paths = [str(p) for p in path.rglob('*') if p.is_file() and p.suffix.lower() in image_extensions]
            if st:
                st.info(f"Found {len(image_paths)} images in folder: {image_paths_or_folder}")
        elif path.is_file() and path.suffix.lower() in image_extensions:
            image_paths = [str(path)]
        else:
            if st:
                st.warning(f"{image_paths_or_folder} is not a valid folder or image file.")
            return documents
    else:
        image_paths = image_paths_or_folder

    total = len(image_paths)
    if total == 0:
        return documents

    if st:
        progress_bar = st.progress(0)
        status_text = st.empty()
    else:
        progress_bar = None
        status_text = None

    for idx, image_path in enumerate(image_paths):
        path = Path(image_path)
        if not path.exists() or path.suffix.lower() not in image_extensions:
            continue

        try:
            if status_text:
                status_text.text(f"Processing image {idx+1}/{total}: {path.name}")
            description = multimodal_emb.get_image_description(str(path))

            doc = Document(
                page_content=description,
                metadata={
                    "source": str(path),
                    "type": "image",
                    "image_path": str(path),
                },
            )
            documents.append(doc)
            if progress_bar:
                progress_bar.progress((idx + 1) / total)
        except Exception as e:
            if st:
                st.error(f"Error processing {image_path}: {str(e)}")
            logger.exception("Error processing image %s: %s", image_path, e)
            continue

    if progress_bar:
        progress_bar.empty()
    if status_text:
        status_text.empty()

    return documents


def create_chroma_vectorstore(documents: List[Document], persist_directory: str = "./chroma_db", embedding_model: str = "text-embedding-3-large"):
    """Create a Chroma vectorstore from a list of LangChain Documents, handling image docs specially."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    split_docs: List[Document] = []
    for doc in documents:
        if doc.metadata.get("type") == "image":
            # images already represented as a single textual description
            split_docs.append(doc)
        else:
            try:
                splits = text_splitter.split_documents([doc])
                split_docs.extend(splits)
            except Exception:
                logger.exception("Failed to split document: %s", doc.metadata.get("source"))
                split_docs.append(doc)

    embeddings = OpenAIEmbeddings(model=embedding_model)

    vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings, persist_directory=persist_directory)
    return vectorstore


def create_simple_rag_chain(vectorstore: Chroma):
    """Return a simple callable that performs retrieval and answers with an LLM.

    This keeps the implementation dependency-light and easy to call from the UI.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def answer(question: str, k: int = 4, max_tokens: int = 256, temperature: float = 0.0):
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join(d.page_content for d in docs)
        prompt = (
            "Use the following pieces of context to answer the question at the end.\n"
            "If you don't know the answer, just say that you don't know. Do not hallucinate.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        try:
            answer_text = llm.predict(prompt)
        except Exception:
            # fallback to calling with a message object
            msg = HumanMessage(content=prompt)
            resp = llm([msg])
            answer_text = getattr(resp, "content", str(resp))

        sources = [d.metadata.get("source") for d in docs]
        return {"answer": answer_text, "sources": sources}

    return answer
