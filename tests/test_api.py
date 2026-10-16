from app.config import settings
from app.llm import LLMError

from .conftest import FakeLLM
from .test_ingest import make_pdf


# ---------- health & errors ----------

def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["llm_configured"] is False


def test_request_id_and_error_shape(client):
    r = client.get("/documents/999", headers={"X-Request-ID": "abc123"})
    assert r.status_code == 404
    assert r.headers["X-Request-ID"] == "abc123"
    assert r.json() == {"error": {"code": "not_found", "message": "Document 999 not found", "request_id": "abc123"}}


def test_unexpected_error_returns_500_without_details(client, monkeypatch):
    import app.main

    def boom(*args, **kwargs):
        raise RuntimeError("database exploded")

    monkeypatch.setattr(app.main, "search", boom)
    r = client.get("/search", params={"q": "refund"}, headers={"X-Request-ID": "req-500"})
    assert r.status_code == 500
    assert r.json()["error"] == {
        "code": "internal_error",
        "message": "Something went wrong on our side.",
        "request_id": "req-500",
    }


def test_validation_error_shape(client):
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "validation_error" and err["details"][0]["field"] == "question"


# ---------- documents ----------

def test_upload_list_get_delete(client, upload):
    r = upload("refunds.md", "Returns accepted within 30 days. " * 80)
    assert r.status_code == 201
    doc = r.json()
    assert doc["content_type"] == "text/markdown" and doc["chunk_count"] >= 2

    assert [d["id"] for d in client.get("/documents").json()] == [doc["id"]]

    detail = client.get(f"/documents/{doc['id']}", params={"include_chunks": True}).json()
    assert len(detail["chunks"]) == doc["chunk_count"]
    assert client.get(f"/documents/{doc['id']}").json()["chunks"] is None

    assert client.delete(f"/documents/{doc['id']}").status_code == 204
    assert client.get("/documents").json() == []
    assert client.get("/search", params={"q": "returns"}).json()["results"] == []


def test_upload_pdf(upload):
    r = upload("warranty.pdf", make_pdf("Warranty covers two years"), "application/pdf")
    assert r.status_code == 201 and r.json()["chunk_count"] == 1


def test_upload_errors(upload, monkeypatch):
    assert upload("notes.docx", "hi").json()["error"]["code"] == "unsupported_file_type"
    assert upload("notes.docx", "hi").status_code == 415
    assert upload("empty.txt", "   ").status_code == 422
    monkeypatch.setattr(settings, "max_upload_bytes", 10)
    assert upload("big.txt", "x" * 11).status_code == 413


# ---------- search ----------

def test_search_ranks_relevant_document_first(client, two_docs):
    refunds, pto = two_docs
    r = client.get("/search", params={"q": "how long until my refund shows up?"}).json()
    assert r["results"][0]["document_id"] == refunds["id"]

    r = client.get("/search", params={"q": "vacation days carry over"}).json()
    assert r["results"][0]["document_id"] == pto["id"]


def test_search_filters_and_edge_cases(client, two_docs):
    refunds, pto = two_docs
    r = client.get("/search", params={"q": "refund days", "document_id": pto["id"]}).json()
    assert {h["document_id"] for h in r["results"]} == {pto["id"]}
    assert client.get("/search", params={"q": "what is it"}).json()["results"] == []
    assert client.get("/search", params={"q": "zebra"}).json()["results"] == []


# ---------- asking ----------

def test_ask_with_llm_grounds_answer_in_sources(client, two_docs, use_llm):
    fake = use_llm(FakeLLM("Refunds take 5 to 7 business days after approval [1]."))
    r = client.post("/ask", json={"question": "When will my refund arrive?"})
    assert r.status_code == 201
    body = r.json()
    answer = body["answer"]
    assert answer["mode"] == "llm" and "[1]" in answer["content"]
    assert answer["sources"][0]["filename"] == "refunds.md" and answer["sources"][0]["cited"] is True

    prompt = fake.calls[0]["messages"][-1]["content"]
    assert "5 to 7 business days" in prompt and "Question: When will my refund arrive?" in prompt
    assert "ONLY" in fake.calls[0]["system"]

    convo = client.get(f"/conversations/{body['conversation_id']}").json()
    assert [m["role"] for m in convo["messages"]] == ["user", "assistant"]
    assert convo["title"] == "When will my refund arrive?"


def test_follow_up_question_includes_history(client, two_docs, use_llm):
    fake = use_llm(FakeLLM("Within 30 days [1].", "Card refunds take 5-7 days [1]."))
    cid = client.post("/ask", json={"question": "How long do I have to return items?"}).json()["conversation_id"]
    r = client.post(f"/conversations/{cid}/messages", json={"question": "And the refund timing?"})
    assert r.status_code == 201

    sent = fake.calls[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[0]["content"] == "How long do I have to return items?"
    assert client.get("/conversations").json()[0]["message_count"] == 4


def test_falls_back_to_passages_when_llm_fails(client, two_docs, use_llm):
    use_llm(FakeLLM(LLMError("Anthropic API returned 529: overloaded")))
    answer = client.post("/ask", json={"question": "refund timing"}).json()["answer"]
    assert answer["mode"] == "retrieval_only"
    assert "529" in answer["error"] and "refunds.md" in answer["content"]


def test_retrieval_only_without_llm(client, two_docs):
    answer = client.post("/ask", json={"question": "refund timing"}).json()["answer"]
    assert answer["mode"] == "retrieval_only" and answer["sources"]


def test_no_relevant_documents_skips_llm(client, two_docs, use_llm):
    fake = use_llm(FakeLLM())
    answer = client.post("/ask", json={"question": "zebra migration patterns"}).json()["answer"]
    assert answer["mode"] == "no_context" and fake.calls == []


def test_ask_limited_to_documents(client, two_docs, use_llm):
    refunds, pto = two_docs
    use_llm(FakeLLM())
    answer = client.post("/ask", json={"question": "days", "document_ids": [pto["id"]]}).json()["answer"]
    assert {s["document_id"] for s in answer["sources"]} == {pto["id"]}


# ---------- conversations ----------

def test_conversation_crud(client):
    conv = client.post("/conversations", json={"title": "Onboarding"}).json()
    assert conv["message_count"] == 0
    assert client.get("/conversations").json()[0]["title"] == "Onboarding"
    assert client.delete(f"/conversations/{conv['id']}").status_code == 204
    assert client.get(f"/conversations/{conv['id']}").status_code == 404
    assert client.post("/conversations/999/messages", json={"question": "hi"}).status_code == 404


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "Knowledge Assistant" in r.text
