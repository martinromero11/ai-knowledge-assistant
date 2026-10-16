"""All settings come from environment variables (or a .env file)."""

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass
class Settings:
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./knowledge.db"))

    # LLM
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or None)
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY") or None)
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "").lower())
    llm_model: str | None = field(default_factory=lambda: os.getenv("LLM_MODEL") or None)
    llm_timeout_seconds: int = field(default_factory=lambda: _int("LLM_TIMEOUT_SECONDS", 30))

    # Embeddings (semantic search)
    voyage_api_key: str | None = field(default_factory=lambda: os.getenv("VOYAGE_API_KEY") or None)
    embedding_provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "").lower())
    embedding_model: str | None = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL") or None)
    # Fixed when the chunks table is first created on PostgreSQL (vector column size).
    embedding_dim: int = field(default_factory=lambda: _int("EMBEDDING_DIM", 1024))
    # Semantic matches below this cosine similarity are treated as unrelated.
    semantic_min_similarity: float = field(
        default_factory=lambda: float(os.getenv("SEMANTIC_MIN_SIMILARITY", "0.3"))
    )

    # Ingestion
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 150))
    max_upload_bytes: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 10) * 1024 * 1024)

    # Retrieval / answering
    top_k: int = field(default_factory=lambda: _int("TOP_K", 5))
    history_messages: int = field(default_factory=lambda: _int("HISTORY_MESSAGES", 6))

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())

    def __post_init__(self):
        if not 0 <= self.chunk_overlap < self.chunk_size // 2:
            raise ValueError("CHUNK_OVERLAP must be at least 0 and less than half of CHUNK_SIZE")


settings = Settings()
