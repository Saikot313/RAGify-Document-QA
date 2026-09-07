"""
Pydantic models defining the shape of API requests and responses.
"""

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    filename: str
    pages_extracted: int
    chunks_indexed: int
    message: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question about the uploaded documents.")


class SourceCitation(BaseModel):
    source: str
    page: int | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]


class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool
