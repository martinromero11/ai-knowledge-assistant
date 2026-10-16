"""The embedding API clients: request format, batching, and error handling (no real network calls)."""

import httpx
import pytest

from app import embeddings
from app.embeddings import EmbeddingError, OpenAIEmbedder, VoyageEmbedder


class FakeHTTP:
    def __init__(self, status=200, shape="data", fail_first=0):
        self.status = status
        self.shape = shape
        self.fail_first = fail_first  # return 429 this many times before succeeding
        self.requests = []

    def post(self, url, headers, json, timeout):
        self.requests.append({"url": url, "headers": headers, "json": json})
        request = httpx.Request("POST", url)
        if self.fail_first:
            self.fail_first -= 1
            return httpx.Response(429, text="rate limited", headers={"retry-after": "4"}, request=request)
        if self.status != 200:
            return httpx.Response(self.status, text="rate limited", request=request)
        vectors = [[float(i), 1.0] for i in range(len(json["input"]))]
        if self.shape == "data":  # return out of order to check sorting by index
            body = {"data": [{"embedding": v, "index": i} for i, v in reversed(list(enumerate(vectors)))]}
        else:
            body = {"embeddings": vectors}
        return httpx.Response(200, json=body, request=request)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    waits = []
    monkeypatch.setattr(embeddings.time, "sleep", waits.append)
    return waits


@pytest.fixture
def fake_http(monkeypatch):
    fake = FakeHTTP()
    monkeypatch.setattr(embeddings.httpx, "post", fake.post)
    return fake


def test_voyage_request_and_batching(fake_http, monkeypatch):
    monkeypatch.setattr(embeddings, "BATCH_SIZE", 2)
    vectors = VoyageEmbedder("key", dim=1024).embed(["a", "b", "c"], input_type="document")
    assert vectors == [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]  # 2 batches, each in order
    first = fake_http.requests[0]
    assert first["url"] == "https://api.voyageai.com/v1/embeddings"
    assert first["headers"]["Authorization"] == "Bearer key"
    assert first["json"] == {"input": ["a", "b"], "model": "voyage-4-lite", "input_type": "document"}


def test_voyage_custom_dimension(fake_http):
    VoyageEmbedder("key", dim=512).embed(["a"], input_type="query")
    assert fake_http.requests[0]["json"]["output_dimension"] == 512


def test_openai_request(fake_http):
    vectors = OpenAIEmbedder("key", dim=256).embed(["a", "b"], input_type="query")
    assert vectors == [[0.0, 1.0], [1.0, 1.0]]
    assert fake_http.requests[0]["json"] == {"input": ["a", "b"], "model": "text-embedding-3-small", "dimensions": 256}


def test_alternate_response_shape(monkeypatch):
    fake = FakeHTTP(shape="embeddings")
    monkeypatch.setattr(embeddings.httpx, "post", fake.post)
    assert VoyageEmbedder("key").embed(["a"], input_type="query") == [[0.0, 1.0]]


def test_api_errors_become_embedding_errors(monkeypatch):
    monkeypatch.setattr(embeddings.httpx, "post", FakeHTTP(status=429).post)
    with pytest.raises(EmbeddingError, match="429"):
        VoyageEmbedder("key").embed(["a"], input_type="query")

    def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(embeddings.httpx, "post", timeout)
    with pytest.raises(EmbeddingError, match="ReadTimeout"):
        OpenAIEmbedder("key").embed(["a"], input_type="query")


def test_rate_limit_is_retried(monkeypatch, no_sleep):
    fake = FakeHTTP(fail_first=2)
    monkeypatch.setattr(embeddings.httpx, "post", fake.post)
    assert VoyageEmbedder("key").embed(["a"], input_type="query") == [[0.0, 1.0]]
    assert len(fake.requests) == 3
    assert no_sleep == [4.0, 4.0]  # honors the Retry-After header


def test_gives_up_after_max_retries(monkeypatch, no_sleep):
    fake = FakeHTTP(status=429)
    monkeypatch.setattr(embeddings.httpx, "post", fake.post)
    with pytest.raises(EmbeddingError, match="429"):
        VoyageEmbedder("key").embed(["a"], input_type="query")
    assert len(fake.requests) == embeddings.MAX_RETRIES + 1
    assert no_sleep == [2.0, 6.0, 18.0]  # backoff when no Retry-After header