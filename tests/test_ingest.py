import pytest

from app.errors import InvalidDocument, UnsupportedFileType
from app.ingest import chunk_text, extract_text
from app.retrieval import tokenize

LONG = "\n\n".join(
    f"Paragraph {i}. " + " ".join(f"word{i}_{j}" for j in range(40)) + "." for i in range(30)
)


def test_chunks_respect_size_and_offsets():
    chunks = chunk_text(LONG, size=500, overlap=80)
    assert len(chunks) > 5
    for c in chunks:
        assert len(c.text) <= 500
        assert LONG[c.start:c.end] == c.text  # offsets point back to the source text


def test_chunks_cover_whole_text_with_overlap():
    chunks = chunk_text(LONG, size=500, overlap=80)
    assert chunks[0].start == 0 and chunks[-1].end == len(LONG)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start < prev.end  # overlapping
        assert nxt.start > prev.start  # always moving forward


def test_chunks_prefer_paragraph_breaks():
    chunks = chunk_text(LONG, size=500, overlap=0)
    assert all(c.text.endswith(".") for c in chunks[:-1])


def test_short_text_is_one_chunk():
    chunks = chunk_text("Just one short line.", size=1000, overlap=150)
    assert [c.text for c in chunks] == ["Just one short line."]


def test_bad_chunk_settings():
    with pytest.raises(ValueError):
        chunk_text(LONG, size=100, overlap=60)


def test_extract_text_decoding_and_cleanup():
    assert extract_text("a.txt", "caf\xe9\r\n\r\n\r\n\r\nend   \n".encode("latin-1")) == "café\n\nend"
    assert extract_text("a.md", "﻿# Title".encode("utf-8")) == "# Title"


def test_extract_rejects_bad_input():
    with pytest.raises(UnsupportedFileType):
        extract_text("a.docx", b"data")
    with pytest.raises(InvalidDocument):
        extract_text("a.txt", b"   \n\n  ")
    with pytest.raises(InvalidDocument):
        extract_text("a.pdf", b"not really a pdf")


def make_pdf(text: str) -> bytes:
    """Build a minimal one-page PDF containing `text`."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return out


def test_extract_pdf():
    assert "Warranty covers two years" in extract_text("w.pdf", make_pdf("Warranty covers two years"))


def test_tokenize():
    assert tokenize("What is the refund policy for returns?") == ["refund", "policy", "return"]
    assert tokenize("Policies and refunds") == ["policy", "refund"]
    assert tokenize("what is it?") == []
