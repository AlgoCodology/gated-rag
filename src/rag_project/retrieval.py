"""Step 3 of RAG: given a question, find the most relevant chunks.

HOW SEARCH WORKS
    1. Embed the question into a vector (same model as the chunks!).
    2. Score every chunk: score = dot(chunk_vector, question_vector).
       Because all vectors have length 1, this equals cosine similarity:
       1.0 = same direction (very similar), 0 = unrelated.
    3. Return the top-k highest-scoring chunks.

ACCESS CONTROL HAPPENS HERE, NOT IN THE PROMPT
    Telling an AI "don't reveal secrets to customers" is only a polite request;
    a clever question can talk it out of it. Instead we remove forbidden chunks
    BEFORE ranking, so the model never even sees them. You can't leak what you
    were never given.
"""

import numpy as np

from . import config
from .chunking import Chunk
from .index import Index, normalize
from .ollama_client import embed


def top_chunks(
    index: Index,
    question_vector: np.ndarray,
    role: str,
    k: int = config.TOP_K,
) -> list[tuple[Chunk, float]]:
    """Pure-math part of retrieval (no network), so it is easy to test.

    Returns up to k (chunk, score) pairs, best first.
    """
    allowed = config.ROLE_PERMISSIONS[role]
    # Boolean mask: True for chunks this role may see.
    visible = np.array([c.access in allowed for c in index.chunks])

    scores = index.vectors @ normalize(question_vector)  # one score per chunk
    scores = np.where(visible, scores, -np.inf)  # hidden chunks can never win

    best_first = np.argsort(scores)[::-1][:k]
    return [(index.chunks[i], float(scores[i])) for i in best_first if visible[i]]


def retrieve(
    index: Index, question: str, role: str, k: int = config.TOP_K
) -> list[tuple[Chunk, float]]:
    """Embed the question and return the top-k visible chunks with scores."""
    question_vector = embed([question])[0]
    return top_chunks(index, question_vector, role, k)
