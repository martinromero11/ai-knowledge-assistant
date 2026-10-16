"""Turning uploaded files into searchable chunks.

1. extract_text(): get plain text out of .txt, .md, or .pdf bytes.
2. chunk_text(): split the text into overlapping passages of about CHUNK_SIZE
   characters, breaking at paragraph, line, or sentence boundaries when possible.
"""

import io
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .errors import InvalidDocument, UnsupportedFileType

SUPPORTED_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
}


@dataclass
class TextChunk:
    text: str
    start: int  # character offsets into the extracted text
    end: int


def content_type_for(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_TYPES:
        allowed = ", ".join(SUPPORTED_TYPES)
        raise UnsupportedFileType(f"'{ext or filename}' files are not supported. Upload one of: {allowed}")
    return SUPPORTED_TYPES[ext]


def extract_text(filename: str, data: bytes) -> str:
    content_type = content_type_for(filename)

    if content_type == "application/pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except (PdfReadError, ValueError, KeyError) as e:
            raise InvalidDocument(f"Could not read PDF: {e}") from e
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("latin-1")

    text = _normalize(text)
    if not text:
        raise InvalidDocument(
            "No text could be extracted. If this is a scanned PDF, it needs OCR first."
        )
    return text


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)  # trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse big gaps
    return text.strip()


# Preferred places to end a chunk, best first.
_BREAKS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[TextChunk]:
    if size <= 0 or not 0 <= overlap < size // 2:
        raise ValueError("need size > 0 and 0 <= overlap < size / 2")

    chunks: list[TextChunk] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            # Look for a natural break in the second half of the window.
            for sep in _BREAKS:
                idx = window.rfind(sep, size // 2)
                if idx != -1:
                    end = start + idx + len(sep)
                    break

        raw = text[start:end]
        piece = raw.strip()
        if piece:
            lead = len(raw) - len(raw.lstrip())
            chunk_start = start + lead
            chunks.append(TextChunk(piece, chunk_start, chunk_start + len(piece)))

        if end >= n:
            break

        # Start the next chunk a little before this one ended, at a word boundary.
        if overlap:
            next_start = max(end - overlap, start + 1)
            space = text.find(" ", next_start, end)
            start = space + 1 if space != -1 else next_start
        else:
            start = end

    return chunks
