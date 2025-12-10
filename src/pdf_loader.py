import os
from pathlib import Path
from typing import List, Union
import tempfile

def load_text_documents(file_paths: List[str]) -> List[Document]:
    """Load text documents from various formats."""
    documents = []
    
    for file_path in file_paths:
        path = Path(file_path)
        loader = PyPDFLoader(str(path))

        docs = loader.load()
        documents.extend(docs)
    
    return documents