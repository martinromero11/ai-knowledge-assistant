"""Semantic and hybrid search, embedding failures, and re-embedding."""

from app.retrieval import SearchHit, reciprocal_rank_fusion

from .conftest import PTO, REFUNDS, FakeEmbedder, FakeLLM

PARAPHRASE = "Can I get my money back?"  # shares no keywords with the refund policy


def _search(client, q, **params):
    r = client.get("/search", params={"q": q, **params})
    assert r.status_code == 200, r.text
    return r.json()


def test_upload_embeds_chunks(client, embedded_docs):
    fake, refunds, pto = embedded_docs
    assert refunds["embedding_model"] == "fake-embed"
    assert ("document", refunds["chunk_count"]) in fake.calls

    health = client.get("/health").json()
    assert health["default_search_mode"] == "hybrid"
    assert health["semantic_search"] in ("pgvector", "python")


def test_semantic_finds_paraphrase_that_keyword_misses(client, embedded_docs):
    fake, refunds, pto = embedded_docs
    assert _search(client, PARAPHRASE, mode="keyword")["results"] == []

    semantic = _search(client, PARAPHRASE, mode="semantic")
    assert semantic["results"][0]["document_id"] == refunds["id"]
    assert ("query", 1) in fake.calls

    hybrid = _search(client, PARAPHRASE)
    assert hybrid["mode"] == "hybrid"
    assert hybrid["results"][0]["document_id"] == refunds["id"]
    assert hybrid["results"][0]["matched_by"] == ["semantic"]


def test_hybrid_ranks_chunks_found_both_ways_highest(client, embedded_docs):
    fake, refunds, pto = embedded_docs
    top = _search(client, "refund")["results"][0]
    assert top["document_id"] == refunds["id"]
    assert top["matched_by"] == ["keyword", "semantic"]
    assert top["score"] == 1.0  # ranked first in both lists


def test_reciprocal_rank_fusion():
    def hit(i):
        return SearchHit(i, 1, "f", 0, "t", 0.0, [])

    keyword = [hit(1), hit(2), hit(3)]
    semantic = [hit(3), hit(4)]
    fused = reciprocal_rank_fusion([("keyword", keyword), ("semantic", semantic)], k=4)
    assert [h.chunk_id for h in fused] == [3, 1, 2, 4]  # 3 is in both lists; 2 and 4 tie, lower id first
    assert fused[0].matched_by == ["keyword", "semantic"]


def test_unrelated_question_is_still_no_context(client, embedded_docs, use_llm):
    llm = use_llm(FakeLLM())
    answer = client.post("/ask", json={"question": "zebra migration patterns"}).json()["answer"]
    assert answer["mode"] == "no_context" and llm.calls == []


def test_ask_uses_hybrid_search(client, embedded_docs, use_llm):
    use_llm(FakeLLM("Yes, within 30 days you can get a full refund [1]."))
    body = client.post("/ask", json={"question": PARAPHRASE}).json()
    answer = body["answer"]
    assert answer["mode"] == "llm" and answer["search_mode"] == "hybrid"
    assert answer["sources"][0]["filename"] == "refunds.md"
    assert answer["sources"][0]["matched_by"] == ["semantic"]

    stored = client.get(f"/conversations/{body['conversation_id']}").json()["messages"][1]
    assert stored["search_mode"] == "hybrid"


def test_upload_survives_embedding_failure_then_backfill(client, use_embedder, upload):
    use_embedder(FakeEmbedder(fail=True))
    r = upload("refunds.md", REFUNDS)
    assert r.status_code == 201 and r.json()["embedding_model"] is None

    use_embedder(FakeEmbedder())
    assert _search(client, PARAPHRASE, mode="semantic")["results"] == []  # no vectors yet

    result = client.post("/documents/embed").json()
    assert result == {"model": "fake-embed", "documents_embedded": 1, "chunks_embedded": 1, "failed": []}
    assert _search(client, PARAPHRASE, mode="semantic")["results"][0]["filename"] == "refunds.md"
    assert client.post("/documents/embed").json()["documents_embedded"] == 0  # nothing left to do


def test_query_embedding_failure_falls_back_to_keyword(client, embedded_docs, use_embedder):
    use_embedder(FakeEmbedder(fail=True))
    body = _search(client, "refund timing")
    assert body["mode"] == "keyword" and "503" in body["note"]
    assert body["results"][0]["filename"] == "refunds.md"


def test_vectors_from_another_model_are_ignored_until_reembedded(client, embedded_docs, use_embedder):
    use_embedder(FakeEmbedder(model="fake-embed-v2"))
    assert _search(client, PARAPHRASE, mode="semantic")["results"] == []

    assert client.post("/documents/embed").json()["documents_embedded"] == 2
    docs = client.get("/documents").json()
    assert {d["embedding_model"] for d in docs} == {"fake-embed-v2"}
    assert _search(client, PARAPHRASE, mode="semantic")["results"][0]["filename"] == "refunds.md"


def test_semantic_mode_requires_embeddings(client, two_docs):
    r = client.get("/search", params={"q": "refund", "mode": "semantic"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "embeddings_not_configured"
    assert client.post("/documents/embed").status_code == 400
    assert _search(client, "refund")["mode"] == "keyword"  # auto mode without embeddings


def test_semantic_search_respects_document_filter(client, embedded_docs):
    fake, refunds, pto = embedded_docs
    results = _search(client, PARAPHRASE, mode="semantic", document_id=pto["id"])["results"]
    assert all(r["document_id"] == pto["id"] for r in results)
