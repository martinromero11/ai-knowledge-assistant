import os

# Configure before the app is imported. Set TEST_DATABASE_URL to run against PostgreSQL.
os.environ["DATABASE_URL"] = os.getenv("TEST_DATABASE_URL", "sqlite:///./test_knowledge.db")
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["VOYAGE_API_KEY"] = ""
os.environ["EMBEDDING_DIM"] = "64"

import pytest
from fastapi.testclient import TestClient

from app.database import Base, engine, init_db
from app.main import app, get_embedder, get_llm_client


class FakeLLM:
    """Records what it was sent and returns scripted replies (or raises)."""

    model = "fake-model"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, system, messages):
        self.calls.append({"system": system, "messages": messages})
        reply = self.replies.pop(0) if self.replies else "Answer [1]."
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.drop_all(bind=engine)
    init_db()
    app.dependency_overrides[get_llm_client] = lambda: None
    app.dependency_overrides[get_embedder] = lambda: None
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def use_embedder():
    def _use(fake):
        app.dependency_overrides[get_embedder] = lambda: fake
        return fake

    return _use


@pytest.fixture
def use_llm():
    def _use(fake):
        app.dependency_overrides[get_llm_client] = lambda: fake
        return fake

    return _use


REFUNDS = """# Refund policy

Customers may return items within 30 days of delivery for a full refund.
Card refunds appear within 5 to 7 business days after the return is approved.

Final sale items cannot be returned."""

PTO = """# Time off

Full-time employees accrue 15 vacation days per year.
Up to 5 unused days carry over into the next year."""


@pytest.fixture
def upload(client):
    def _upload(name, content, mime="text/plain"):
        data = content.encode() if isinstance(content, str) else content
        r = client.post("/documents", files={"file": (name, data, mime)})
        return r

    return _upload


@pytest.fixture
def two_docs(upload):
    a = upload("refunds.md", REFUNDS).json()
    b = upload("pto.md", PTO).json()
    return a, b


class FakeEmbedder:
    """Deterministic 64-dim vectors. Words in the same concept group (like "refund"
    and "money") share a dimension, so paraphrases land close together, the way a
    real embedding model behaves. Other words are hashed into the remaining dims."""

    CONCEPTS = {
        "refund": 0, "refunds": 0, "money": 0, "reimburse": 0, "reimbursed": 0,
        "return": 1, "returns": 1, "returned": 1, "send": 1,
        "vacation": 2, "pto": 2, "holiday": 2, "leave": 2,
    }

    def __init__(self, model="fake-embed", fail=False):
        self.model = model
        self.dim = 64
        self.fail = fail
        self.calls: list[tuple[str, int]] = []

    def embed(self, texts, input_type):
        import math
        import re
        import zlib

        from app.embeddings import EmbeddingError
        from app.retrieval import STOPWORDS

        self.calls.append((input_type, len(texts)))
        if self.fail:
            raise EmbeddingError("Voyage API returned 503: service unavailable")
        vectors = []
        for t in texts:
            vec = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if w in STOPWORDS:
                    continue
                if w in self.CONCEPTS:
                    vec[self.CONCEPTS[w]] += 3.0
                else:
                    vec[8 + zlib.crc32(w.encode()) % (self.dim - 8)] += 1.0
            if not any(vec):
                vec[-1] = 1.0
            norm = math.sqrt(sum(x * x for x in vec))
            vectors.append([x / norm for x in vec])
        return vectors


@pytest.fixture
def embedded_docs(use_embedder, upload):
    fake = use_embedder(FakeEmbedder())
    a = upload("refunds.md", REFUNDS).json()
    b = upload("pto.md", PTO).json()
    return fake, a, b
