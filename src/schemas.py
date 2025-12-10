from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List


class DocumentMeta(BaseModel):
    filename: str
    page: Optional[int] = None
    source: Optional[str] = None
    text_length: Optional[int] = None


class IngestedDocument(BaseModel):
    id: str
    text: str
    embedding: Optional[List[float]] = None
    meta: DocumentMeta
