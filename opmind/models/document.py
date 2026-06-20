from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    source: str
    source_type: str
    title: str
    content: str
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0"
    status: str = "active"


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    context_prefix: Optional[str] = None
    embedding: Optional[list[float]] = None
    start_line: int = 0
    end_line: int = 0
    section_path: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    index_version: str = "v1"


class Citation(BaseModel):
    citation_id: str
    chunk_id: str
    doc_id: str
    doc_title: str
    excerpt: str
    source_url: Optional[str] = None
    relevance_score: float = 0.0


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    doc_title: str
    score: float
    start_line: int = 0
    end_line: int = 0
    metadata: dict = Field(default_factory=dict)
