from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    content_type: str
    size_bytes: int
    char_count: int
    chunk_count: int
    embedding_model: str | None
    created_at: datetime


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    position: int
    char_start: int
    char_end: int
    text: str


class DocumentDetail(DocumentOut):
    chunks: list[ChunkOut] | None = None


class SearchResult(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    position: int
    score: float
    matched_by: list[str]
    text: str


class SearchResponse(BaseModel):
    query: str
    mode: str
    note: str | None = None
    results: list[SearchResult]


class EmbedResult(BaseModel):
    model: str
    documents_embedded: int
    chunks_embedded: int
    failed: list[int]


class Source(BaseModel):
    n: int
    chunk_id: int
    document_id: int
    filename: str
    position: int
    score: float
    excerpt: str
    cited: bool
    matched_by: list[str] = []


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    mode: str | None
    sources: list[Source]
    error: str | None
    search_mode: str | None = None
    created_at: datetime


class ConversationCreate(BaseModel):
    title: str | None = Field(None, max_length=200)


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]


class QuestionIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int | None = Field(None, ge=1, le=20, description="How many passages to retrieve")
    document_ids: list[int] | None = Field(None, description="Only search these documents")


class AskIn(QuestionIn):
    conversation_id: int | None = Field(None, description="Continue this conversation, or omit to start one")


class AnswerOut(BaseModel):
    conversation_id: int
    question: MessageOut
    answer: MessageOut
