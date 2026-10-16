"""Upload the sample documents and ask a test question.

Usage:  python seed.py                     (server must be running on :8000)
        python seed.py http://host:port
"""

import sys
from pathlib import Path

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
SAMPLES = Path(__file__).parent / "sample_docs"

with httpx.Client(base_url=BASE, timeout=90) as client:
    for path in sorted(SAMPLES.iterdir()):
        with path.open("rb") as f:
            r = client.post("/documents", files={"file": (path.name, f)})
        r.raise_for_status()
        d = r.json()
        print(f"Uploaded #{d['id']} {d['filename']}: {d['chunk_count']} chunks")

    question = "How long do I have to return something, and when will I get my money back?"
    print(f"\nQ: {question}")
    r = client.post("/ask", json={"question": question})
    r.raise_for_status()
    answer = r.json()["answer"]
    print(f"A ({answer['mode']}, {answer.get('search_mode')} search): {answer['content']}\n")
    for s in answer["sources"]:
        mark = "*" if s["cited"] else " "
        how = " + ".join(s.get("matched_by", []))
        print(f" {mark}[{s['n']}] {s['filename']} part {s['position'] + 1} ({how}, score {s['score']})")

    print(f"\nOpen {BASE} to chat with your documents.")
