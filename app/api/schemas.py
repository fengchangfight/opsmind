from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    session_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Optional[dict] = None


class ResumeRequest(BaseModel):
    session_id: str
    human_input: str = Field(..., min_length=1, max_length=4096)
    option: str = Field(default="continue", pattern="^(continue|modify|transfer)$")


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=10, ge=1, le=50)
    filters: Optional[dict] = None
