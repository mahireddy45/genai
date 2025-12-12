import os
import tempfile
import logging

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from .image_loader import load_image
from .logging_config import get_logger

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

logger = get_logger(__name__)

def preprocess(docs: List[Document]) -> List[Document]:
    preprocessed_docs = []
    for doc in docs:
        text = doc.page_content
        text = text.strip()                # remove leading/trailing whitespace
        text = " ".join(text.split())      # normalize multiple spaces/newlines
        text = text.lower()                # lowercase for consistency        
        doc.page_content = text
        preprocessed_docs.append(doc)
    return preprocessed_docs

def load_documents(files: List, llm_model: str) -> List[Document]:
    logger.info("Loading %d files with LLM model %s", len(files), llm_model)
    for f in files:
        logger.debug("File uploaded: %s", f.name)
    
    docs = []
    
    for file in files:
        suffix = Path(file.name).suffix.lower()
        original_filename = file.name  # Preserve original filename with spaces
        logger.debug("Processing file %s with suffix %s", original_filename, suffix)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file.getbuffer())
            tmp_path = tmp_file.name

        try:
            file_path = Path(tmp_path)

            if suffix in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"]:
                documents = load_image(file_path, llm_model)
                docs.append(documents)
            else:            
                if suffix in [".docx", ".doc"]:
                    try:
                        # attempt to use the langchain loader (requires docx2txt)
                        loader = Docx2txtLoader(file_path)
                        documents = loader.load()
                    except Exception as e:
                        logger.debug("Docx2txtLoader failed: %s; falling back to python-docx", e)
                        try:
                            from docx import Document as _PyDocxDocument
                            _doc = _PyDocxDocument(str(file_path))
                            txt = "\n\n".join([p.text for p in _doc.paragraphs if p.text])
                            documents = [Document(page_content=txt, metadata={})]
                        except Exception as e2:
                            logger.exception("Failed to parse .docx file with both docx2txt and python-docx: %s", e2)
                            raise
                elif suffix in [".txt", ".md"]:
                    loader = TextLoader(file_path)
                    documents = loader.load()
                elif suffix == ".pdf":
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()

                # Update metadata for all documents in the list
                for doc in documents:
                    # Preserve original filename (may contain spaces)
                    doc.metadata = doc.metadata or {}
                    doc.metadata["source"] = original_filename
                    doc.metadata["path"] = original_filename
                    doc.metadata["file_ext"] = suffix
                    doc.metadata["file_type"] = "document"
                docs.extend(documents)
                
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    if docs:
        docs = preprocess(docs)

    return docs