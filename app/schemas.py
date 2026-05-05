from typing import Literal, Optional
from pydantic import BaseModel, Field


class BookOut(BaseModel):
    id: str
    title: str
    author: Optional[str] = None
    char_count: int
    chunk_count: int


class ChunkRef(BaseModel):
    chunk_id: str
    idx: int
    chapter: Optional[str]
    char_start: int
    char_end: int
    snippet: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    k: int = Field(default=6, ge=1, le=20)
    position_chunk_id: Optional[str] = None  # where the user is in the book


class Citation(BaseModel):
    chunk_id: str
    idx: int
    snippet: str


class ChatResponse(BaseModel):
    session_id: str
    message: str
    citations: list[Citation]


class AnnotationCreate(BaseModel):
    kind: Literal["text", "voice"] = "text"
    body: str
    chunk_id: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    timestamp_s: Optional[float] = None


class AnnotationOut(BaseModel):
    id: str
    book_id: str
    kind: str
    anchor: dict
    body: str
    created_at: str


class OverviewRequest(BaseModel):
    style: Literal["brief", "detailed", "outline"] = "brief"


class GraphicRequest(BaseModel):
    topic: str
    format: Literal["mermaid", "svg"] = "mermaid"


class NotebookEntry(BaseModel):
    id: str
    book_id: str
    kind: str
    title: Optional[str]
    body: str
    meta: Optional[dict] = None
    created_at: str


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
