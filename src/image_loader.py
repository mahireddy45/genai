"""Simple image loader with OCR using pytesseract.

Returns list of ingested document chunks extracted from images.
"""
from __future__ import annotations
from typing import List, Dict
from pathlib import Path
from PIL import Image
from .schemas import DocumentMeta, IngestedDocument


def load_image(path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
    p = Path(path)
    if not p.is_file():
        return []
    text = pytesseract.image_to_string(Image.open(p)) or ""
    if not text.strip():
        return []

    # chunk text
    results = []
    start = 0
    length = len(text)
    idx = 1
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            doc_id = f"{p.name}::c{idx}"
            meta = DocumentMeta(filename=p.name, page=None, source=str(p), text_length=len(chunk))
            results.append(IngestedDocument(id=doc_id, text=chunk, meta=meta).dict())
            idx += 1
        start = max(end - chunk_overlap, end)

    return results
