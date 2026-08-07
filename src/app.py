"""
app.py
Command-line entry point for AskMyDocs.

Usage:
    python src/app.py "What did the dissertation find about label noise?"
    python src/app.py "..." --retriever bm25   # sparse only
    python src/app.py "..." --retriever dense  # embeddings only
    python src/app.py "..." --retriever hybrid # RRF fusion (default)
    python src/app.py "..." --no-generate      # just show retrieved chunks, skip Claude call
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from ingest import load_documents
from sparse_retriever import BM25Retriever
from dense_retriever import DenseRetriever
from hybrid_retriever import HybridRetriever
from generate import answer_question

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def build_retriever(kind: str, chunks):
    if kind == "bm25":
        return BM25Retriever(chunks)
    if kind == "dense":
        return DenseRetriever(chunks)
    return HybridRetriever(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ask questions over your documents.")
    parser.add_argument("question", type=str, help="The question to ask")
    parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-generate", action="store_true", help="Skip the LLM call, just show retrieved chunks")
    args = parser.parse_args()

    print(f"Loading documents from {DATA_DIR} ...")
    chunks = load_documents(DATA_DIR)
    print(f"Indexed {len(chunks)} chunks. Building '{args.retriever}' retriever...")

    retriever = build_retriever(args.retriever, chunks)
    results = retriever.search(args.question, top_k=args.top_k)

    print("\n--- Retrieved chunks ---")
    for chunk, score in results:
        print(f"\n[{chunk.chunk_id}]  score={score:.4f}")
        print(chunk.text[:300])

    if args.no_generate:
        return

    print("\n--- Generating grounded answer ---")
    answer = answer_question(args.question, results)
    print(f"\n{answer}")


if __name__ == "__main__":
    main()
