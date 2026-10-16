import logging
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session, selectinload

from . import models
from .config import settings
from .database import get_db, init_db, vector_backend
from .embeddings import Embedder, EmbeddingError, default_embedder
from .errors import AppError, EmbeddingsNotConfigured, FileTooLarge, InvalidDocument, NotFound
from .ingest import chunk_text, content_type_for, extract_text
from .llm import LLMClient, default_client
from .logging_config import request_id_var, setup_logging
from .rag import answer_question
from .retrieval import keyword_backend, search
from .schemas import (
    AnswerOut,
    AskIn,
    ConversationCreate,
    ConversationDetail,
    ConversationOut,
    DocumentDetail,
    DocumentOut,
    EmbedResult,
    MessageOut,
    QuestionIn,
    SearchResponse,
)

setup_logging(settings.log_level)
log = logging.getLogger("app")

init_db()

app = FastAPI(
    title="AI Knowledge Assistant",
    version="1.0.0",
    description="Upload documents, search them, and ask questions answered from their contents.",
)
STATIC_DIR = Path(__file__).parent / "static"


def get_llm_client() -> LLMClient | None:
    return default_client()


def get_embedder() -> Embedder | None:
    return default_embedder()


# ---------- middleware & error handling ----------


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    try:
        try:
            response = await call_next(request)
        except Exception:
            log.exception("Unhandled error on %s %s", request.method, request.url.path)
            response = _error(500, "internal_error", "Something went wrong on our side.")
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        if request.url.path != "/favicon.ico":
            log.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, response.status_code, elapsed_ms)
        return response
    finally:
        request_id_var.reset(token)


def _error(status: int, code: str, message: str, details=None) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "request_id": request_id_var.get()}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    log.info("%s: %s", exc.code, exc.message)
    return _error(exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    details = [
        {"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
        for e in exc.errors()
    ]
    return _error(422, "validation_error", "The request is invalid.", details)


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _error(500, "internal_error", "Something went wrong on our side.")


# ---------- helpers ----------


def _get_or_404(db: Session, model, obj_id: int, label: str):
    obj = db.get(model, obj_id)
    if obj is None:
        raise NotFound(f"{label} {obj_id} not found")
    return obj


def _conversation_out(conv: models.Conversation, count: int) -> ConversationOut:
    out = ConversationOut.model_validate(conv)
    out.message_count = count
    return out


# ---------- pages & health ----------


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(
    db: Session = Depends(get_db),
    llm: LLMClient | None = Depends(get_llm_client),
    embedder: Embedder | None = Depends(get_embedder),
):
    db.execute(text("SELECT 1"))
    return {
        "status": "ok",
        "database": db.get_bind().dialect.name,
        "keyword_search": keyword_backend(db),
        "semantic_search": vector_backend() if embedder else "disabled",
        "default_search_mode": "hybrid" if embedder else "keyword",
        "llm_configured": llm is not None,
        "model": getattr(llm, "model", None),
        "embedding_model": getattr(embedder, "model", None),
    }


# ---------- documents ----------


@app.post("/documents", response_model=DocumentOut, status_code=201, tags=["documents"])
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    embedder: Embedder | None = Depends(get_embedder),
):
    filename = Path(file.filename or "").name
    if not filename:
        raise InvalidDocument("The uploaded file needs a filename.")
    content_type = content_type_for(filename)

    data = file.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise FileTooLarge(f"Files must be {settings.max_upload_bytes // (1024 * 1024)} MB or smaller.")

    body = extract_text(filename, data)
    pieces = chunk_text(body, settings.chunk_size, settings.chunk_overlap)

    doc = models.Document(
        filename=filename,
        content_type=content_type,
        size_bytes=len(data),
        char_count=len(body),
        chunk_count=len(pieces),
        chunks=[
            models.Chunk(position=i, text=p.text, char_start=p.start, char_end=p.end)
            for i, p in enumerate(pieces)
        ],
    )
    if embedder is not None:
        _embed_chunks(doc, embedder)
    db.add(doc)
    db.commit()
    log.info(
        "Ingested document %s '%s': %d chars, %d chunks, embedded=%s",
        doc.id, filename, len(body), len(pieces), doc.embedding_model is not None,
    )
    return doc


def _embed_chunks(doc: models.Document, embedder: Embedder) -> bool:
    """Attach a vector to every chunk. On failure, keep the document (keyword search still works)."""
    try:
        vectors = embedder.embed([c.text for c in doc.chunks], input_type="document")
    except EmbeddingError as e:
        log.warning("Embedding failed for '%s'; it can be retried with POST /documents/embed: %s", doc.filename, e)
        return False
    for chunk, vector in zip(doc.chunks, vectors):
        chunk.embedding = vector
    doc.embedding_model = embedder.model
    return True


@app.post("/documents/embed", response_model=EmbedResult, tags=["documents"])
def embed_missing(db: Session = Depends(get_db), embedder: Embedder | None = Depends(get_embedder)):
    """Embed documents that have no vectors yet, or were embedded with a different model."""
    if embedder is None:
        raise EmbeddingsNotConfigured("Set VOYAGE_API_KEY to enable embeddings.")
    docs = db.scalars(
        select(models.Document).where(
            (models.Document.embedding_model.is_(None)) | (models.Document.embedding_model != embedder.model)
        )
    ).all()
    result = EmbedResult(model=embedder.model, documents_embedded=0, chunks_embedded=0, failed=[])
    for doc in docs:
        if _embed_chunks(doc, embedder):
            db.commit()
            result.documents_embedded += 1
            result.chunks_embedded += len(doc.chunks)
        else:
            db.rollback()
            result.failed.append(doc.id)
    return result


@app.get("/documents", response_model=list[DocumentOut], tags=["documents"])
def list_documents(db: Session = Depends(get_db)):
    return db.scalars(select(models.Document).order_by(models.Document.id.desc())).all()


@app.get("/documents/{document_id}", response_model=DocumentDetail, tags=["documents"])
def get_document(
    document_id: int,
    include_chunks: bool = Query(False, description="Include the document's chunks"),
    db: Session = Depends(get_db),
):
    doc = _get_or_404(db, models.Document, document_id, "Document")
    out = DocumentDetail.model_validate(doc)
    if not include_chunks:
        out.chunks = None
    return out


@app.delete("/documents/{document_id}", status_code=204, tags=["documents"])
def delete_document(document_id: int, db: Session = Depends(get_db)):
    doc = _get_or_404(db, models.Document, document_id, "Document")
    db.delete(doc)
    db.commit()
    log.info("Deleted document %s", document_id)
    return Response(status_code=204)


# ---------- search ----------


@app.get("/search", response_model=SearchResponse, tags=["search"])
def search_documents(
    q: str = Query(..., min_length=1, max_length=500),
    k: int = Query(None, ge=1, le=50),
    document_id: list[int] | None = Query(None, description="Repeat to search several documents"),
    mode: str = Query("auto", pattern="^(auto|hybrid|keyword|semantic)$",
                      description="auto = hybrid when embeddings are configured, else keyword"),
    db: Session = Depends(get_db),
    embedder: Embedder | None = Depends(get_embedder),
):
    outcome = search(db, q, k or settings.top_k, document_id, embedder, mode)
    return SearchResponse(
        query=q, mode=outcome.mode, note=outcome.note, results=[h.to_dict() for h in outcome.hits]
    )


# ---------- conversations ----------


@app.post("/conversations", response_model=ConversationOut, status_code=201, tags=["conversations"])
def create_conversation(payload: ConversationCreate, db: Session = Depends(get_db)):
    conv = models.Conversation(title=(payload.title or "New conversation").strip() or "New conversation")
    db.add(conv)
    db.commit()
    return _conversation_out(conv, 0)


@app.get("/conversations", response_model=list[ConversationOut], tags=["conversations"])
def list_conversations(db: Session = Depends(get_db)):
    counts = (
        select(models.Message.conversation_id, func.count().label("n"))
        .group_by(models.Message.conversation_id)
        .subquery()
    )
    rows = db.execute(
        select(models.Conversation, func.coalesce(counts.c.n, 0))
        .outerjoin(counts, counts.c.conversation_id == models.Conversation.id)
        .order_by(models.Conversation.updated_at.desc(), models.Conversation.id.desc())
    ).all()
    return [_conversation_out(conv, n) for conv, n in rows]


@app.get("/conversations/{conversation_id}", response_model=ConversationDetail, tags=["conversations"])
def get_conversation(conversation_id: int, db: Session = Depends(get_db)):
    conv = db.scalars(
        select(models.Conversation)
        .where(models.Conversation.id == conversation_id)
        .options(selectinload(models.Conversation.messages))
    ).first()
    if conv is None:
        raise NotFound(f"Conversation {conversation_id} not found")
    out = ConversationDetail.model_validate(conv)
    out.message_count = len(conv.messages)
    return out


@app.delete("/conversations/{conversation_id}", status_code=204, tags=["conversations"])
def delete_conversation(conversation_id: int, db: Session = Depends(get_db)):
    db.delete(_get_or_404(db, models.Conversation, conversation_id, "Conversation"))
    db.commit()
    return Response(status_code=204)


@app.post(
    "/conversations/{conversation_id}/messages",
    response_model=AnswerOut,
    status_code=201,
    tags=["conversations"],
)
def ask_in_conversation(
    conversation_id: int,
    payload: QuestionIn,
    db: Session = Depends(get_db),
    llm: LLMClient | None = Depends(get_llm_client),
    embedder: Embedder | None = Depends(get_embedder),
):
    conv = _get_or_404(db, models.Conversation, conversation_id, "Conversation")
    return _ask(db, conv, payload, llm, embedder)


@app.post("/ask", response_model=AnswerOut, status_code=201, tags=["conversations"])
def ask(
    payload: AskIn,
    db: Session = Depends(get_db),
    llm: LLMClient | None = Depends(get_llm_client),
    embedder: Embedder | None = Depends(get_embedder),
):
    """Ask a question. Continues `conversation_id` if given, otherwise starts a new conversation."""
    if payload.conversation_id is not None:
        conv = _get_or_404(db, models.Conversation, payload.conversation_id, "Conversation")
    else:
        title = payload.question.strip()
        conv = models.Conversation(title=title[:80] + ("…" if len(title) > 80 else ""))
        db.add(conv)
        db.flush()
    return _ask(db, conv, payload, llm, embedder)


def _ask(
    db: Session,
    conv: models.Conversation,
    payload: QuestionIn,
    llm: LLMClient | None,
    embedder: Embedder | None,
) -> AnswerOut:
    question = payload.question.strip()
    history = [
        {"role": m.role, "content": m.content}
        for m in db.scalars(
            select(models.Message).where(models.Message.conversation_id == conv.id).order_by(models.Message.id)
        )
    ]

    found = search(db, question, payload.k or settings.top_k, payload.document_ids, embedder)
    result = answer_question(question, found.hits, history, llm, settings.history_messages)
    log.info(
        "Answered in conversation %s: mode=%s, search=%s, %d passages retrieved",
        conv.id, result.mode, found.mode, len(found.hits),
    )

    user_msg = models.Message(conversation=conv, role="user", content=question, sources=[])
    bot_msg = models.Message(
        conversation=conv,
        role="assistant",
        content=result.content,
        mode=result.mode,
        sources=result.sources,
        error=result.error,
        search_mode=found.mode,
    )
    db.add_all([user_msg, bot_msg])
    if conv.title == "New conversation":
        conv.title = question[:80]
    conv.updated_at = models.utcnow()
    db.commit()
    return AnswerOut(
        conversation_id=conv.id,
        question=MessageOut.model_validate(user_msg),
        answer=MessageOut.model_validate(bot_msg),
    )
