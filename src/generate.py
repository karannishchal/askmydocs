"""
generate.py
Takes retrieved chunks + a user question and calls Claude to produce a
grounded answer with inline source citations. If retrieved chunks don't
actually contain the answer, the prompt instructs the model to say so
rather than guessing — this is the difference between "RAG" and just
"an LLM that sometimes reads documents."
"""

import os
from typing import List, Tuple
from anthropic import Anthropic
from ingest import Chunk

SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's \
question using ONLY the provided source chunks. For every claim, cite the \
source using [source_id] notation. If the chunks do not contain enough \
information to answer, say so explicitly instead of guessing."""


def build_context(results: List[Tuple[Chunk, float]]) -> str:
    blocks = []
    for chunk, score in results:
        blocks.append(f"[{chunk.chunk_id}] (relevance={score:.3f})\n{chunk.text}")
    return "\n\n".join(blocks)


def answer_question(question: str, results: List[Tuple[Chunk, float]], model: str = "claude-sonnet-4-6") -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set ANTHROPIC_API_KEY in your environment (or .env file) to run generation."
        )

    client = Anthropic(api_key=api_key)
    context = build_context(results)

    user_message = f"Sources:\n\n{context}\n\nQuestion: {question}"

    response = client.messages.create(
        model=model,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text
