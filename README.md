# AskMyDocs

A small Retrieval-Augmented Generation (RAG) search system: sparse (BM25),
dense (embeddings + FAISS), and hybrid (Reciprocal Rank Fusion) retrieval,
plus an offline evaluation harness measuring Recall@k and MRR, and a
generation step that answers questions grounded in retrieved sources.

Built to explore the core mechanics behind modern search/RAG systems —
chunking, sparse vs. dense retrieval, hybrid fusion, offline evaluation,
and grounded generation — at a scale that's easy to read end to end.

## Why both a sparse and a dense retriever?

BM25 (sparse) matches exact keywords — fast, needs no training, but misses
paraphrases ("automobile" vs "car"). Dense retrieval embeds meaning via a
neural encoder, so it catches semantic matches BM25 would miss, but can
underperform on rare terms or exact identifiers that keyword search handles
easily. This repo implements both, plus a hybrid retriever that merges
their rankings with Reciprocal Rank Fusion — a common pattern in real
search systems, since sparse and dense signals are complementary rather
than one strictly beating the other.

## Architecture

```
data/*.txt --> ingest.py (chunk into overlapping windows)
                    |
        +-----------+-----------+
        |                       |
  sparse_retriever.py     dense_retriever.py
   (BM25Okapi)            (sentence-transformers + FAISS)
        |                       |
        +-----------+-----------+
                    |
          hybrid_retriever.py (Reciprocal Rank Fusion)
                    |
              generate.py (Claude, grounded answer + citations)
                    |
                 app.py (CLI)

eval.py + eval_queries.json --> Recall@k / MRR per retriever
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # then add your ANTHROPIC_API_KEY
```

The first run of the dense retriever downloads a small embedding model
(`all-MiniLM-L6-v2`, ~80MB) from HuggingFace. That needs an internet
connection once; after that it's cached locally.

## Usage

Drop your own `.txt` files into `data/` (sample docs about label noise,
retrieval/ranking, and RAG systems are included so it runs out of the box).

```bash
# Ask a question (hybrid retrieval + Claude-generated answer)
python src/app.py "How does correcting label noise affect model performance?"

# Compare retrieval strategies directly
python src/app.py "What is BM25?" --retriever bm25 --no-generate
python src/app.py "What is BM25?" --retriever dense --no-generate
python src/app.py "What is BM25?" --retriever hybrid --no-generate

# Run offline evaluation (Recall@5, MRR) across all three retrievers
python src/eval.py
```

## Evaluation

`eval_queries.json` holds a small labelled query set — each entry maps a
question to the chunk ID that should be retrieved. `eval.py` reports
Recall@5 and MRR for BM25, Dense, and Hybrid, so retrieval quality is
measured rather than assumed. Extend `eval_queries.json` with your own
questions once you swap in your own documents.

## What I'd extend next

- Cross-encoder re-ranking on top of the hybrid retriever's candidate pool
- Faithfulness/groundedness scoring on generated answers (does the answer
  actually follow from the retrieved chunks?)
- Swapping the embedding model to compare quality/latency trade-offs
- A tiny FastAPI wrapper so this can run as a service instead of a CLI

## Project structure

```
askmydocs/
├── data/                  # source .txt documents (swap in your own)
├── eval_queries.json      # labelled queries for offline evaluation
├── src/
│   ├── ingest.py          # chunking
│   ├── sparse_retriever.py    # BM25
│   ├── dense_retriever.py     # embeddings + FAISS
│   ├── hybrid_retriever.py    # RRF fusion
│   ├── generate.py        # grounded generation via Claude
│   ├── eval.py             # Recall@k / MRR harness
│   └── app.py              # CLI entry point
├── requirements.txt
└── .env.example
```
