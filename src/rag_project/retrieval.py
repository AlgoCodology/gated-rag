"""Step 3 of RAG: given a question, find the most relevant context.

HOW SEARCH WORKS
    1. Embed the question into a vector (same model as the chunks, with the
       "search_query: " prefix the model was trained with).
    2. Score every small CHILD chunk: score = dot(chunk_vector, question_vector).
       Because all vectors have length 1, this equals cosine similarity.
    3. Keep the best CANDIDATES children and expand each to its PARENT section
       ("small-to-big"): we matched on a precise sentence, but the chat model
       gets the paragraph(s) around it. Several children of one parent count once.
    4. If even the best child scores under SCORE_THRESHOLD, return nothing, so
       the caller can say "I don't know" without bothering the chat model.
    5. Order the contexts for the prompt (see `u_shaped`).

ACCESS CONTROL HAPPENS HERE, NOT IN THE PROMPT
    Telling an AI "don't reveal secrets to customers" is only a polite request;
    a clever question can talk it out of it. Instead we remove forbidden chunks
    BEFORE ranking, so the model never even sees them. Parents are built only
    from blocks of the same access level (chunking.py), so expanding a hit to its
    parent cannot leak anything either.
"""

from dataclasses import dataclass

import numpy as np

from . import config
from .chunking import Chunk
from .index import Index, normalize
from .ollama_client import embed


@dataclass(frozen=True)
class Context:
    """One section of text that will be shown to the chat model."""

    text: str
    score: float  # similarity of the best child chunk that pointed here
    source: str
    header_path: str
    kind: str
    page: int | None
    access: str

    @property
    def label(self) -> str:
        """A short human-readable citation, e.g. 'public/student_handbook.md > Library (p.2)'."""
        where = self.header_path or ""
        page = f" (p.{self.page})" if self.page else ""
        return f"{self.source}{' > ' + where if where else ''}{page}"


def top_chunks(
    index: Index,
    question_vector: np.ndarray,
    role: str,
    k: int = config.CANDIDATES,
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


def expand_to_parents(
    index: Index,
    hits: list[tuple[Chunk, float]],
    max_contexts: int = config.TOP_K,
    use_parents: bool = True,
) -> list[Context]:
    """Turn child hits into de-duplicated contexts, best first."""
    contexts: list[Context] = []
    seen: set[str] = set()
    for chunk, score in hits:
        key = chunk.parent_id if use_parents and chunk.parent_id else f"{chunk.source}|{chunk.text}"
        if key in seen:
            continue
        seen.add(key)
        text = index.parents.get(chunk.parent_id, chunk.text) if use_parents else chunk.text
        contexts.append(
            Context(text, score, chunk.source, chunk.header_path, chunk.kind, chunk.page, chunk.access)
        )
        if len(contexts) == max_contexts:
            break
    return contexts


def u_shaped(items: list) -> list:
    """Reorder best-first items so the best sit at the START and END of the list.

    Language models read the beginning and end of a long prompt most carefully
    and tend to miss the middle ("Lost in the Middle", Liu et al. 2023).
    [1, 2, 3, 4, 5] (1 = best)  ->  [1, 3, 5, 4, 2]: best first, second-best last,
    the weakest in the middle.
    """
    return items[0::2] + items[1::2][::-1]


def retrieve(
    index: Index,
    question: str,
    role: str,
    k: int = config.TOP_K,
    threshold: float = config.SCORE_THRESHOLD,
    use_parents: bool = True,
    reranker: str | None = None,
) -> list[Context]:
    """Embed the question and return up to k visible contexts, best first.

    Returns an empty list when nothing is similar enough (below `threshold`),
    or, with a re-ranker on, when it finds nothing that answers the question.
    `reranker` overrides config.RERANKER ("none", "bge" or "judge").
    """
    reranker = reranker or config.RERANKER
    wanted = config.CANDIDATES
    if reranker != "none":
        from .rerank import candidates_for, rerank  # imported here: re-ranking is optional

        wanted = max(wanted, candidates_for(reranker))
    question_vector = embed([config.EMBED_QUERY_PREFIX + question])[0]
    hits = top_chunks(index, question_vector, role, wanted)
    if not hits or hits[0][1] < threshold:
        return []
    if reranker != "none":
        scored = rerank(question, hits, reranker)
        if scored[0][2] < config.RERANK_THRESHOLD:
            return []
        hits = [(chunk, score) for chunk, score, _ in scored]
    return expand_to_parents(index, hits, k, use_parents)
