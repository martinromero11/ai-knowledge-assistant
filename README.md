# AI Knowledge Assistant

A FastAPI service that answers questions about your own documents. Upload `.txt`, `.md`, or `.pdf` files. The service splits them into passages, stores them in PostgreSQL, and indexes them two ways: by **keywords** (full-text search) and by **meaning** (vector embeddings with pgvector). When you ask a question, it finds the most relevant passages with **hybrid search** and sends them to an LLM. The LLM is told to answer **only** from those passages and cite them by number. This pattern is called retrieval-augmented generation (RAG).

**Stack:** Python, FastAPI, SQLAlchemy, PostgreSQL + pgvector, Voyage AI embeddings, Anthropic or OpenAI API, pypdf, Docker, pytest.

## How it works

```
upload ──► extract text ──► split into overlapping chunks ──► embed each chunk ──► PostgreSQL
                                                                                    ├─ tsvector + GIN index   (keywords)
                                                                                    └─ vector + HNSW index    (meaning)

question ──┬─► full-text search ──► ranked list ─┐
           └─► embed question ──► vector search ──► ranked list ─┴─► reciprocal rank fusion ──► top-k passages
                                                                                                       │
            answer with [1][2] citations, saved to conversation history ◄── LLM with numbered sources ◄┘
```

1. **Ingestion** (`app/ingest.py`): Extracts text from the file and splits it into chunks of about 1,000 characters with a 150-character overlap. Chunks end at paragraph or sentence boundaries when possible.
2. **Embeddings** (`app/embeddings.py`): Each chunk is sent to an embedding model (Voyage AI by default), which returns a vector of 1,024 numbers that represents its meaning. Chunks are labeled `input_type="document"` and questions `input_type="query"`, as Voyage recommends for retrieval. The document records which model produced its vectors.
3. **Search** (`app/retrieval.py`), with three modes:
   - **keyword:** PostgreSQL full-text search. Each chunk has a generated `tsvector` column with a GIN index, ranked by `ts_rank_cd`. On SQLite, BM25 scoring in Python is used instead.
   - **semantic:** Cosine distance between the question's vector and each chunk's vector, using pgvector's `<=>` operator and an HNSW index. Matches below a minimum similarity are dropped, so unrelated questions still get "not found."
   - **hybrid** (the default when embeddings are configured): Runs both searches and merges them with **reciprocal rank fusion**. Each appearance in a list adds `1 / (60 + rank)`. This rewards passages that rank well in either list, and especially in both, without having to compare the two methods' scores directly, since they're on different scales.
4. **Answering** (`app/rag.py`): The top passages are numbered and sent to the model with the last few turns of the conversation. Citations in the reply are matched back to their passages.

### Why hybrid search

Keyword search is precise for exact terms ("Trail Club," "$75") but misses paraphrases. For example, "Can I get my money back?" shares no keywords with a policy that says "refund." Semantic search catches that, but it can blur exact terms. Hybrid search gets both. Each source in a response shows how it was found (`matched_by: ["keyword", "semantic"]`).

### Reliability

| Failure | What happens |
|---|---|
| No passages match | Replies "not found" without calling the LLM |
| LLM down, timed out, or no key | Returns the retrieved passages instead, and records the error |
| Embedding API rate-limited (429) or briefly down | Waits and retries up to 3 times, honoring `Retry-After` |
| Embedding API still failing during upload | The document is still saved and keyword-searchable; `POST /documents/embed` retries later |
| Embedding API fails during a question | Falls back to keyword search, and the response includes a `note` |
| Embedding model changed | Vectors from the old model are ignored (they can't be compared); `POST /documents/embed` re-embeds them |
| PostgreSQL without pgvector | Vectors are stored as JSON and compared in Python |

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/documents` | Upload a file (multipart `file`); chunks and embeds it |
| `GET` | `/documents` | List documents, including `embedding_model` |
| `GET` | `/documents/{id}?include_chunks=true` | Document metadata, optionally with its chunks |
| `DELETE` | `/documents/{id}` | Delete a document and its chunks |
| `POST` | `/documents/embed` | Embed any documents missing vectors or using an old model |
| `GET` | `/search?q=...&mode=hybrid&k=5&document_id=1` | Ranked passages; `mode` is `auto`, `hybrid`, `keyword`, or `semantic` |
| `POST` | `/ask` | Ask a question; starts a conversation or continues `conversation_id` |
| `POST` | `/conversations` | Create a conversation |
| `GET` | `/conversations` | List conversations with message counts |
| `GET` | `/conversations/{id}` | Full history, including sources and search mode for each answer |
| `POST` | `/conversations/{id}/messages` | Ask a follow-up in a conversation |
| `DELETE` | `/conversations/{id}` | Delete a conversation |
| `GET` | `/health` | Database, search backends, LLM, and embedding status |

Interactive docs are at `/docs`. A simple chat page is served at `/`.

**Errors** use one consistent shape:

```json
{ "error": { "code": "unsupported_file_type", "message": "...", "request_id": "3f9c2a1b7d4e" } }
```

**Logging:** Every request gets an `X-Request-ID` (or reuses the one sent by the client). The ID appears in every log line for that request, along with the status code and duration.

## Data model

| Table | Purpose |
|---|---|
| `documents` | Filename, type, size, chunk count, and which embedding model was used |
| `chunks` | Passage text, position, and offsets; `embedding vector(1024)` with HNSW index; `tsv` full-text column with GIN index |
| `conversations` | Title and timestamps |
| `messages` | Questions and answers, with answer mode, search mode, cited sources (JSON), and any LLM error |

## Run with Docker (PostgreSQL + pgvector)

```bash
cp .env.example .env        # add ANTHROPIC_API_KEY and VOYAGE_API_KEY
docker compose up --build
python seed.py              # in another terminal: upload sample docs and ask a question
```

Open http://localhost:8000. The database uses the `pgvector/pgvector:pg16` image, which is PostgreSQL 16 with pgvector installed.

## Run locally without Docker

This uses SQLite, BM25 keyword search, and Python vector search, so no database setup is needed.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # add your API keys
uvicorn app.main:app --reload
python seed.py
```

Without `VOYAGE_API_KEY`, search is keyword-only. Without `ANTHROPIC_API_KEY`, answers show the retrieved passages instead of a written answer.

## Tests

```bash
pytest -q                                                            # SQLite
TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/test pytest -q  # PostgreSQL (+ pgvector if installed)
```

The 44 tests run without network access. They use a fake LLM, a fake embedder (which maps related words like "money" and "refund" close together, the way real embeddings do), and stubbed HTTP for the API clients. They cover:
- chunking and file extraction
- upload validation
- keyword, semantic, and hybrid ranking
- reciprocal rank fusion
- paraphrased questions that only semantic search can find
- rate-limit retries, embedding failures, and re-embedding
- model changes
- grounded answers with citations
- conversation history
- LLM fallback
- error formats and request IDs

## Configuration

All settings come from environment variables or `.env`. See `.env.example`.

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./knowledge.db` | Database connection |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | — | LLM provider key |
| `LLM_MODEL` | `claude-haiku-5-5` | Answering model |
| `VOYAGE_API_KEY` | — | Enables embeddings and semantic/hybrid search |
| `EMBEDDING_MODEL` | `voyage-4-lite` | Embedding model |
| `EMBEDDING_DIM` | `1024` | Vector size; can't change after the database is created |
| `SEMANTIC_MIN_SIMILARITY` | `0.3` | Cosine similarity below which a semantic match is ignored |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Chunking in characters |
| `TOP_K` | `5` | Passages retrieved per question |
| `MAX_UPLOAD_MB` | `10` | Upload size limit |

## Possible improvements

- **Tune and evaluate retrieval:** Build a small set of questions with known correct passages, and measure hit rate as `SEMANTIC_MIN_SIMILARITY`, chunk size, and `TOP_K` change.
- **Database migrations:** Use Alembic. The app currently adds missing nullable columns on startup, which only covers simple changes.
- **Background ingestion:** Process large uploads in a background job.
- **Reranking:** Re-score the merged candidates with a reranker model before sending them to the LLM.
- **Authentication:** Add per-user document collections.