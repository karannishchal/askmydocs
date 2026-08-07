"""
dense_retriever.py
Embedding-based retrieval using sentence-transformers + FAISS.

Why both a sparse (BM25) and dense retriever exist in this repo:
BM25 is exact-keyword matching — great for rare terms, bad for paraphrases
("car" vs "automobile"). Dense retrieval embeds meaning, not just words, so
it catches semantically similar chunks even with no word overlap. A
production search system (this is the core of what this project is
modelling) typically blends both — see hybrid_retriever.py.

Note: the first run downloads the embedding model (~80MB) from
HuggingFace, so it needs an internet connection once. After that it's
cached locally and works offline.
"""

from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from ingest import Chunk

DEFAULT_MODEL = "all-MiniLM-L6-v2"  # small, fast, good baseline quality


class DenseRetriever:
    def __init__(self, chunks: List[Chunk], model_name: str = DEFAULT_MODEL):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)

        texts = [c.text for c in chunks]
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        self.dim = embeddings.shape[1]

        # Inner product on normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings.astype("float32"))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        query_vec = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results
