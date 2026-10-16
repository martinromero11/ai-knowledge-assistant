"""Retrieval-augmented generation: answer a question using retrieved passages.

The model only sees the passages we retrieved, numbered [1], [2], ...
It is told to answer from those passages alone and cite them by number,
which keeps answers grounded in the uploaded documents.
"""

import logging
import re
from dataclasses import dataclass, field

from .llm import LLMClient, LLMError
from .retrieval import SearchHit

log = logging.getLogger("app.rag")

SYSTEM_PROMPT = """You are a knowledge assistant. Answer the user's question using ONLY the numbered sources provided in their message.

Rules:
- Cite the sources you use inline with bracketed numbers, like [1] or [2][3].
- If the sources do not contain the answer, say you couldn't find it in the uploaded documents. Do not use outside knowledge or guess.
- If sources disagree, point that out.
- Be concise and direct. Use short paragraphs or a short list when it helps."""

NO_CONTEXT_ANSWER = (
    "I couldn't find anything in the uploaded documents related to that question. "
    "Try rephrasing it, or upload a document that covers this topic."
)

EXCERPT_CHARS = 300


@dataclass
class Answer:
    content: str
    mode: str  # "llm" | "retrieval_only" | "no_context"
    sources: list[dict] = field(default_factory=list)
    error: str | None = None


def build_sources(hits: list[SearchHit]) -> list[dict]:
    return [
        {
            "n": i,
            "chunk_id": h.chunk_id,
            "document_id": h.document_id,
            "filename": h.filename,
            "position": h.position,
            "score": h.score,
            "excerpt": _excerpt(h.text),
            "cited": False,
            "matched_by": list(h.matched_by),
        }
        for i, h in enumerate(hits, start=1)
    ]


def build_prompt(question: str, hits: list[SearchHit]) -> str:
    blocks = [f"[{i}] (from {h.filename}, part {h.position + 1})\n{h.text}" for i, h in enumerate(hits, start=1)]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def trim_history(history: list[dict], max_messages: int) -> list[dict]:
    """Keep the most recent turns, starting on a user message (APIs require it)."""
    recent = history[-max_messages:] if max_messages > 0 else []
    while recent and recent[0]["role"] != "user":
        recent = recent[1:]
    return recent


def answer_question(
    question: str,
    hits: list[SearchHit],
    history: list[dict],
    llm: LLMClient | None,
    max_history: int = 6,
) -> Answer:
    if not hits:
        return Answer(NO_CONTEXT_ANSWER, "no_context")

    sources = build_sources(hits)
    if llm is None:
        return _retrieval_only(sources, "no LLM API key configured")

    messages = trim_history(history, max_history) + [{"role": "user", "content": build_prompt(question, hits)}]
    try:
        text = llm.complete(SYSTEM_PROMPT, messages).strip()
    except LLMError as e:
        log.warning("LLM call failed, falling back to retrieved passages: %s", e)
        return _retrieval_only(sources, str(e))
    if not text:
        return _retrieval_only(sources, "the model returned an empty answer")

    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    for s in sources:
        s["cited"] = s["n"] in cited
    return Answer(text, "llm", sources)


def _retrieval_only(sources: list[dict], reason: str) -> Answer:
    lines = ["I couldn't generate a written answer right now, but these passages look most relevant:"]
    lines += [f"[{s['n']}] {s['filename']}: {s['excerpt']}" for s in sources[:3]]
    return Answer("\n\n".join(lines), "retrieval_only", sources, reason)


def _excerpt(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= EXCERPT_CHARS:
        return text
    cut = text.rfind(" ", 0, EXCERPT_CHARS)
    return text[: cut if cut > 0 else EXCERPT_CHARS] + "…"
