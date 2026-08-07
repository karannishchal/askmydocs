"""
eval.py
Small offline evaluation harness comparing BM25 vs Dense vs Hybrid retrieval
using standard IR metrics: Recall@k and MRR (Mean Reciprocal Rank).

You need a labelled query set to run this meaningfully: a list of
(question, expected_chunk_id) pairs. See eval_queries.json for the format.
This is the piece that turns "I built a RAG demo" into "I built a RAG demo
and measured which retrieval strategy actually performs better" — which is
exactly the kind of offline evaluation this role asks about.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from ingest import load_documents
from sparse_retriever import BM25Retriever
from dense_retriever import DenseRetriever
from hybrid_retriever import HybridRetriever

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EVAL_FILE = os.path.join(os.path.dirname(__file__), "..", "eval_queries.json")


def recall_at_k(results, expected_id, k):
    top_ids = [c.chunk_id for c, _ in results[:k]]
    return 1.0 if expected_id in top_ids else 0.0


def reciprocal_rank(results, expected_id):
    for rank, (chunk, _) in enumerate(results, start=1):
        if chunk.chunk_id == expected_id:
            return 1.0 / rank
    return 0.0


def evaluate(retriever, queries, k=5):
    recalls, rrs = [], []
    for q in queries:
        results = retriever.search(q["question"], top_k=k)
        recalls.append(recall_at_k(results, q["expected_chunk_id"], k))
        rrs.append(reciprocal_rank(results, q["expected_chunk_id"]))
    return {
        "recall@k": sum(recalls) / len(recalls),
        "mrr": sum(rrs) / len(rrs),
        "n_queries": len(queries),
    }


def main():
    if not os.path.exists(EVAL_FILE):
        print(f"No eval_queries.json found at {EVAL_FILE}. See README for the format.")
        return

    with open(EVAL_FILE) as f:
        queries = json.load(f)

    chunks = load_documents(DATA_DIR)

    retrievers = {
        "BM25": BM25Retriever(chunks),
        "Dense": DenseRetriever(chunks),
        "Hybrid": HybridRetriever(chunks),
    }

    print(f"Evaluating {len(queries)} queries across {len(chunks)} chunks...\n")
    for name, retriever in retrievers.items():
        metrics = evaluate(retriever, queries)
        print(f"{name:8s} | Recall@5: {metrics['recall@k']:.3f} | MRR: {metrics['mrr']:.3f}")


if __name__ == "__main__":
    main()
