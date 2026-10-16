"""Embedding clients: turn text into vectors so passages can be matched by meaning.

Each client exposes embed(texts, input_type) where input_type is "document" for
chunks being stored and "query" for a user's question. Both must come from the
same model, or the vectors can't be compared.
"""

import logging
import time
from typing import Protocol

import httpx

from .config import settings

log = logging.getLogger("app.embeddings")

BATCH_SIZE = 128
MAX_RETRIES = 3  # extra attempts after a rate-limit (429) or temporary server error
MAX_WAIT_SECONDS = 25.0
RETRY_STATUSES = {429, 500, 502, 503, 504}


class EmbeddingError(Exception):
    pass


class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str], input_type: str) -> list[list[float]]: ...


def _post_with_retry(url: str, headers: dict, payload: dict, timeout: float, provider: str) -> dict:
    """POST, waiting and retrying when the API says it's rate-limited or temporarily down.

    New Voyage accounts have a low requests-per-minute limit, so short bursts (like
    uploading several files) can hit 429s. Waiting a few seconds usually fixes it.
    """
    for attempt in range(MAX_RETRIES + 1):
        resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code not in RETRY_STATUSES or attempt == MAX_RETRIES:
            resp.raise_for_status()
            return resp.json()
        wait = _retry_after(resp, attempt)
        log.warning(
            "%s API returned %s, retrying in %.0fs (attempt %d of %d)",
            provider, resp.status_code, wait, attempt + 1, MAX_RETRIES,
        )
        time.sleep(wait)
    raise AssertionError("unreachable")


def _retry_after(resp: httpx.Response, attempt: int) -> float:
    try:
        wait = float(resp.headers.get("retry-after", ""))
    except ValueError:
        wait = 2.0 * 3**attempt  # 2s, 6s, 18s
    return min(max(wait, 1.0), MAX_WAIT_SECONDS)


def _parse_vectors(body: dict, expected: int) -> list[list[float]]:
    if "data" in body:  # OpenAI-style: {"data": [{"embedding": [...], "index": 0}, ...]}
        items = sorted(body["data"], key=lambda d: d.get("index", 0))
        vectors = [d["embedding"] for d in items]
    else:  # {"embeddings": [[...], ...]}
        vectors = body["embeddings"]
    if len(vectors) != expected:
        raise EmbeddingError(f"expected {expected} embeddings, got {len(vectors)}")
    return vectors


class VoyageEmbedder:
    def __init__(self, api_key: str, model: str | None = None, dim: int = 1024, timeout: float = 30):
        self.api_key = api_key
        self.model = model or "voyage-4-lite"
        self.dim = dim
        self.timeout = timeout

    def embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            payload = {"input": batch, "model": self.model, "input_type": input_type}
            if self.dim != 1024:
                payload["output_dimension"] = self.dim
            vectors += self._post(payload, len(batch))
        return vectors

    def _post(self, payload: dict, expected: int) -> list[list[float]]:
        try:
            body = _post_with_retry(
                "https://api.voyageai.com/v1/embeddings",
                {"Authorization": f"Bearer {self.api_key}"},
                payload,
                self.timeout,
                "Voyage",
            )
            return _parse_vectors(body, expected)
        except httpx.HTTPStatusError as e:
            raise EmbeddingError(f"Voyage API returned {e.response.status_code}: {e.response.text[:200]}") from e
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as e:
            raise EmbeddingError(f"Voyage API request failed: {type(e).__name__}: {e}") from e


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str | None = None, dim: int = 1024, timeout: float = 30):
        self.api_key = api_key
        self.model = model or "text-embedding-3-small"
        self.dim = dim
        self.timeout = timeout

    def embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            try:
                body = _post_with_retry(
                    "https://api.openai.com/v1/embeddings",
                    {"Authorization": f"Bearer {self.api_key}"},
                    {"input": batch, "model": self.model, "dimensions": self.dim},
                    self.timeout,
                    "OpenAI",
                )
                vectors += _parse_vectors(body, len(batch))
            except httpx.HTTPStatusError as e:
                raise EmbeddingError(
                    f"OpenAI API returned {e.response.status_code}: {e.response.text[:200]}"
                ) from e
            except (httpx.HTTPError, KeyError, ValueError, TypeError) as e:
                raise EmbeddingError(f"OpenAI API request failed: {type(e).__name__}: {e}") from e
        return vectors


def default_embedder() -> Embedder | None:
    provider = settings.embedding_provider
    args = (settings.embedding_model, settings.embedding_dim, settings.llm_timeout_seconds)
    if provider in ("", "voyage") and settings.voyage_api_key:
        return VoyageEmbedder(settings.voyage_api_key, *args)
    if provider == "openai" and settings.openai_api_key:
        return OpenAIEmbedder(settings.openai_api_key, *args)
    return None