"""Finding the chunks most relevant to a question.

Three search modes:

- keyword: matches words. On PostgreSQL this uses full-text search (a generated
  tsvector column with a GIN index, ranked by ts_rank_cd). Elsewhere it falls
  back to BM25 scoring in Python.
- semantic: matches meaning. The question is turned into an embedding vector and
  compared with each chunk's vector by cosine similarity. On PostgreSQL with
  pgvector this runs in the database with an HNSW index; elsewhere in Python.
- hybrid (default when embeddings are configured): runs both and merges the two
  ranked lists with reciprocal rank fusion, so a chunk that ranks well in either
  list, and especially in both, comes out on top.
"""

import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from sqlalchemy import bindparam, select, text
from sqlalchemy.orm import Session

from . import database
from .config import settings
from .embeddings import Embedder, EmbeddingError
from .errors import EmbeddingsNotConfigured
from .models import Chunk, Document

log = logging.getLogger("app.retrieval")

MODES = ("auto", "hybrid", "keyword", "semantic")
RRF_K = 60  # standard constant for reciprocal rank fusion

STOPWORDS = {
    "a", "about", "after", "all", "also", "am", "an", "and", "any", "are", "as", "at", "be",
    "been", "before", "but", "by", "can", "could", "did", "do", "does", "for", "from", "get",
    "got", "had", "has", "have", "he", "her", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "just", "me", "my", "no", "not", "of", "on", "or", "our", "she", "should", "so",
    "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to",
    "too", "us", "was", "we", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "you", "your",
}
_TOKEN = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    """Very light stemming so 'refunds' matches 'refund' and 'policies' matches 'policy'."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(value: str, stem: bool = True) -> list[str]:
    words = [w for w in _TOKEN.findall(value.lower()) if w not in STOPWORDS and len(w) > 1]
    return [_stem(w) for w in words] if stem else words


@dataclass
class SearchHit:
    chunk_id: int
    document_id: int
    filename: str
    position: int
    text: str
    score: float
    matched_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SearchOutcome:
    hits: list[SearchHit]
    mode: str  # the mode actually used: "keyword", "semantic", or "hybrid"
    note: str | None = None  # set when we had to fall back


def keyword_backend(db: Session) -> str:
    return "postgres_fts" if db.get_bind().dialect.name == "postgresql" else "bm25"


def search(
    db: Session,
    query: str,
    k: int = 5,
    document_ids: list[int] | None = None,
    embedder: Embedder | None = None,
    mode: str = "auto",
) -> SearchOutcome:
    if mode == "auto":
        mode = "hybrid" if embedder else "keyword"
    if mode in ("hybrid", "semantic") and embedder is None:
        raise EmbeddingsNotConfigured(
            f"'{mode}' search needs an embeddings API key (set VOYAGE_API_KEY)."
        )

    if mode == "keyword":
        return SearchOutcome(keyword_search(db, query, k, document_ids), "keyword")

    try:
        query_vector = embedder.embed([query], input_type="query")[0]
    except EmbeddingError as e:
        log.warning("Query embedding failed, using keyword search only: %s", e)
        return SearchOutcome(
            keyword_search(db, query, k, document_ids), "keyword", f"semantic search unavailable: {e}"
        )

    pool = max(k * 3, 10)  # fetch extra candidates from each method before merging
    semantic = semantic_search(db, query_vector, embedder.model, pool, document_ids)
    if mode == "semantic":
        return SearchOutcome(semantic[:k], "semantic")
    keyword = keyword_search(db, query, pool, document_ids)
    return SearchOutcome(reciprocal_rank_fusion([("keyword", keyword), ("semantic", semantic)], k), "hybrid")


def reciprocal_rank_fusion(ranked: list[tuple[str, list[SearchHit]]], k: int) -> list[SearchHit]:
    """Merge ranked lists: each appearance adds 1 / (60 + rank). Scores are scaled to 0-1."""
    merged: dict[int, SearchHit] = {}
    for name, hits in ranked:
        for rank, h in enumerate(hits, start=1):
            m = merged.get(h.chunk_id)
            if m is None:
                m = merged[h.chunk_id] = SearchHit(
                    h.chunk_id, h.document_id, h.filename, h.position, h.text, 0.0, []
                )
            m.score += 1 / (RRF_K + rank)
            m.matched_by.append(name)

    best_possible = len(ranked) / (RRF_K + 1)  # ranked #1 in every list
    results = sorted(merged.values(), key=lambda h: (-h.score, h.chunk_id))[:k]
    for h in results:
        h.score = round(h.score / best_possible, 4)
    return results


# ---------- keyword search ----------


def keyword_search(db: Session, query: str, k: int, document_ids: list[int] | None = None) -> list[SearchHit]:
    if keyword_backend(db) == "postgres_fts":
        return _keyword_postgres(db, query, k, document_ids)
    return _keyword_bm25(db, query, k, document_ids)


def _keyword_postgres(db, query, k, document_ids) -> list[SearchHit]:
    terms = tokenize(query, stem=False)  # Postgres does its own stemming
    if not terms:
        return []
    # OR the terms together so a chunk doesn't need every word to match.
    tsquery = " | ".join(dict.fromkeys(terms))
    sql = """
        SELECT c.id, c.document_id, d.filename, c.position, c.text,
               ts_rank_cd(c.tsv, q, 32) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id,
             to_tsquery('english', :tsq) AS q
        WHERE c.tsv @@ q
    """
    params: dict = {"tsq": tsquery, "k": k}
    if document_ids:
        sql += " AND c.document_id IN :ids"
        params["ids"] = document_ids
    sql += " ORDER BY score DESC, c.id LIMIT :k"
    return _run(db, sql, params, document_ids, "keyword")


def _keyword_bm25(db, query, k, document_ids, k1: float = 1.5, b: float = 0.75) -> list[SearchHit]:
    terms = list(dict.fromkeys(tokenize(query)))
    if not terms:
        return []

    stmt = select(Chunk.id, Chunk.document_id, Document.filename, Chunk.position, Chunk.text).join(Document)
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))
    rows = db.execute(stmt).all()
    if not rows:
        return []

    tokenized = [Counter(tokenize(r.text)) for r in rows]
    lengths = [sum(c.values()) for c in tokenized]
    avg_len = (sum(lengths) / len(lengths)) or 1.0
    n = len(rows)
    doc_freq = {t: sum(1 for c in tokenized if t in c) for t in terms}

    scored = []
    for row, counts, length in zip(rows, tokenized, lengths):
        score = 0.0
        for t in terms:
            tf = counts.get(t, 0)
            if not tf:
                continue
            idf = math.log(1 + (n - doc_freq[t] + 0.5) / (doc_freq[t] + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg_len))
        if score > 0:
            scored.append(
                SearchHit(row.id, row.document_id, row.filename, row.position, row.text, round(score, 4), ["keyword"])
            )

    scored.sort(key=lambda h: (-h.score, h.chunk_id))
    return scored[:k]


# ---------- semantic search ----------


def semantic_search(
    db: Session, query_vector: list[float], model: str, k: int, document_ids: list[int] | None = None
) -> list[SearchHit]:
    """Chunks closest in meaning to the query, among documents embedded with the same model."""
    if database.VECTOR_AVAILABLE and db.get_bind().dialect.name == "postgresql":
        hits = _semantic_pgvector(db, query_vector, model, k, document_ids)
    else:
        hits = _semantic_python(db, query_vector, model, k, document_ids)
    return [h for h in hits if h.score >= settings.semantic_min_similarity]


def _semantic_pgvector(db, query_vector, model, k, document_ids) -> list[SearchHit]:
    sql = """
        SELECT c.id, c.document_id, d.filename, c.position, c.text,
               1 - (c.embedding <=> CAST(:qv AS vector)) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.embedding IS NOT NULL AND d.embedding_model = :model
    """
    params: dict = {"qv": "[" + ",".join(f"{x:.8g}" for x in query_vector) + "]", "model": model, "k": k}
    if document_ids:
        sql += " AND c.document_id IN :ids"
        params["ids"] = document_ids
    sql += " ORDER BY c.embedding <=> CAST(:qv AS vector), c.id LIMIT :k"
    return _run(db, sql, params, document_ids, "semantic")


def _semantic_python(db, query_vector, model, k, document_ids) -> list[SearchHit]:
    stmt = (
        select(Chunk.id, Chunk.document_id, Document.filename, Chunk.position, Chunk.text, Chunk.embedding)
        .join(Document)
        .where(Chunk.embedding.is_not(None), Document.embedding_model == model)
    )
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))

    q_norm = math.sqrt(sum(x * x for x in query_vector)) or 1.0
    hits = []
    for r in db.execute(stmt):
        vec = r.embedding
        if not vec or len(vec) != len(query_vector):
            continue
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        similarity = sum(a * b for a, b in zip(query_vector, vec)) / (q_norm * norm)
        hits.append(SearchHit(r.id, r.document_id, r.filename, r.position, r.text, round(similarity, 4), ["semantic"]))
    hits.sort(key=lambda h: (-h.score, h.chunk_id))
    return hits[:k]


def _run(db, sql, params, document_ids, label) -> list[SearchHit]:
    stmt = text(sql)
    if document_ids:
        stmt = stmt.bindparams(bindparam("ids", expanding=True))
    return [
        SearchHit(r.id, r.document_id, r.filename, r.position, r.text, round(float(r.score), 4), [label])
        for r in db.execute(stmt, params).all()
    ]
