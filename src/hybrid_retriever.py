"""
hybrid_retriever.py
Combines BM25 (sparse) and embedding (dense) rankings using Reciprocal
Rank Fusion (RRF) — a simple, parameter-light way to merge two ranked
lists without needing to normalize incompatible score scales
(BM25 scores and cosine similarities aren't on the same axis, so
averaging them directly would be wrong; RRF sidesteps that by using
rank position instead of raw score).

RRF formula: score(d) = sum over retrievers of 1 / (k + rank(d))
k=60 is the standard constant from the original RRF paper (Cormack et al.)
"""

from typing import List, Tuple, Dict
from ingest import Chunk
from sparse_retriever import BM25Retriever
from dense_retriever import DenseRetriever

RRF_K = 60


class HybridRetriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Retriever(chunks)
        self.dense = DenseRetriever(chunks)

    def search(self, query: str, top_k: int = 5, candidate_pool: int = 20) -> List[Tuple[Chunk, float]]:
        bm25_results = self.bm25.search(query, top_k=candidate_pool)
        dense_results = self.dense.search(query, top_k=candidate_pool)

        rrf_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, Chunk] = {}

        for rank, (chunk, _) in enumerate(bm25_results):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + 1 / (RRF_K + rank + 1)
            chunk_lookup[chunk.chunk_id] = chunk

        for rank, (chunk, _) in enumerate(dense_results):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + 1 / (RRF_K + rank + 1)
            chunk_lookup[chunk.chunk_id] = chunk

        ranked_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(chunk_lookup[cid], score) for cid, score in ranked_ids[:top_k]]
