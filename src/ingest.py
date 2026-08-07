"""
ingest.py
Loads .txt files from a directory, splits them into overlapping chunks,
and returns a list of chunk dicts ready for indexing.

Design notes (for interview talking points):
- Chunking is done by sentence-boundary-aware windowing rather than a hard
  character cut, so we don't split a sentence in half mid-fact.
- Overlap between chunks (default 1 sentence) reduces the chance that a
  fact gets stranded across a chunk boundary and becomes unretrievable.
"""

import os
import re
import glob
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    source: str
    position: int


def _split_sentences(text: str) -> List[str]:
    """Very small sentence splitter (no heavy NLP dependency)."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, max_sentences: int = 5, overlap: int = 1) -> List[str]:
    """Group sentences into overlapping chunks."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks = []
    step = max(1, max_sentences - overlap)
    for i in range(0, len(sentences), step):
        window = sentences[i : i + max_sentences]
        if window:
            chunks.append(" ".join(window))
        if i + max_sentences >= len(sentences):
            break
    return chunks


def load_documents(data_dir: str) -> List[Chunk]:
    """Read every .txt file in data_dir and return a flat list of Chunks."""
    all_chunks: List[Chunk] = []
    paths = sorted(glob.glob(os.path.join(data_dir, "*.txt")))

    if not paths:
        raise FileNotFoundError(
            f"No .txt files found in '{data_dir}'. Drop some documents there first."
        )

    for path in paths:
        doc_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        pieces = chunk_text(raw)
        for idx, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}::chunk_{idx}",
                    text=piece,
                    source=path,
                    position=idx,
                )
            )

    return all_chunks


if __name__ == "__main__":
    chunks = load_documents(os.path.join(os.path.dirname(__file__), "..", "data"))
    print(f"Loaded {len(chunks)} chunks from data/")
    for c in chunks[:3]:
        print(f"\n[{c.chunk_id}]\n{c.text[:200]}...")
