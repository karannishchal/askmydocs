"""
sparse_retriever.py
BM25-based lexical retrieval. This is the "classic" search baseline —
fast, needs no model download, and is a strong baseline against which
to measure the dense retriever in eval.py.
"""

from typing import List, Tuple
from rank_bm25 import BM25Okapi

from ingest import Chunk


class BM25Retriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self._tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self._tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
