"""Optional step 3b: re-rank the candidates by reading question and chunk TOGETHER.

WHY RE-RANK?
    Embedding search compares two vectors that were computed SEPARATELY (the
    question never saw the chunk). It is fast but coarse: it finds the right
    neighbourhood, not always the right paragraph, and it scores an off-topic
    question about "tuition fees of Medicine" almost as high as a real one.
    A re-ranker reads the question and one chunk side by side and says how well
    the chunk answers it. That is slower but sharper. It runs only on the few
    best candidates, so its cost does NOT grow with the size of the corpus.

TWO BACKENDS (config.RERANKER), both returning relevance on a 0-1 scale
    "bge"    a real cross-encoder (BAAI/bge-reranker-base): one small model that takes
             "question + chunk" and outputs a score. Fast (~0.1 s per chunk on a CPU),
             so it can read 20+ candidates. Weaker on world knowledge: it may not
             know that "ChatGPT" is "generative AI".
    "judge"  the chat model itself, prompted to answer a single digit 0-9. Slow (~4 s
             per chunk), but it understands paraphrases and common knowledge.

    Candidates are sorted by relevance, with the embedding score breaking ties.
    If even the best relevance is below config.RERANK_THRESHOLD the question is
    not answerable from the corpus, so the caller says "I don't know".

SECURITY
    A re-ranker only ever sees chunks that already passed the access filter in
    retrieval.py, so re-ranking cannot expose anything a role may not see.
"""

import hashlib
import json
import re

from . import config
from .chunking import Chunk
from .ollama_client import ask_llm

JUDGE_PROMPT = """You judge search results. Decide whether the passage helps answer the question.

Question: {question}

Passage:
{passage}

Rate the passage from 0 to 9: 0 = unrelated, 3 = same topic but does not answer, 6 = partly answers, 9 = clearly contains the answer.
Reply with the single digit only."""

MAX_PASSAGE_CHARS = 1200
BACKENDS = ("bge", "judge")
_bge_model = None  # loaded on first use (loading takes a while)


def passage_text(chunk: Chunk) -> str:
    """What the re-ranker reads: the heading path (context) plus the chunk text."""
    body = f"{chunk.header_path}\n{chunk.text}" if chunk.header_path else chunk.text
    return body[:MAX_PASSAGE_CHARS]


def parse_score(reply: str) -> int:
    """First digit in the model's reply (0-9); 0 if it did not give one."""
    match = re.search(r"\d", reply)
    return int(match.group()) if match else 0


# --- judge backend ----------------------------------------------------------------


def _load_cache() -> dict[str, int]:
    try:
        return json.loads(config.JUDGE_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict[str, int]) -> None:
    config.JUDGE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.JUDGE_CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")


def judge_scores(question: str, chunks: list[Chunk]) -> list[float]:
    """Ask the chat model for a 0-9 digit per chunk, scaled to 0-1.

    One model call per chunk (about 4 s each on a laptop CPU), so every verdict is
    cached on disk by (model, question, passage): repeating a question is free.
    """
    cache = _load_cache()
    before = len(cache)
    scores = []
    for chunk in chunks:
        passage = passage_text(chunk)
        key = hashlib.sha256(f"{config.CHAT_MODEL}\x00{question}\x00{passage}".encode("utf-8")).hexdigest()
        if key not in cache:
            reply = ask_llm(JUDGE_PROMPT.format(question=question, passage=passage), max_tokens=3, temperature=0)
            cache[key] = parse_score(reply)
        scores.append(cache[key] / 9)
    if len(cache) != before:
        _save_cache(cache)
    return scores


# --- cross-encoder backend --------------------------------------------------------


def _get_bge():
    global _bge_model
    if _bge_model is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("The 'bge' re-ranker needs an extra install: uv sync --extra rerank") from exc
        _bge_model = CrossEncoder(config.BGE_MODEL, device="cpu")
    return _bge_model


def bge_scores(question: str, chunks: list[Chunk]) -> list[float]:
    """Cross-encoder relevance per chunk, already on a 0-1 scale (sigmoid of the model's logit)."""
    if not chunks:
        return []
    pairs = [(question, passage_text(c)) for c in chunks]
    return [float(s) for s in _get_bge().predict(pairs, show_progress_bar=False)]


# --- public API ---------------------------------------------------------------------


def candidates_for(backend: str) -> int:
    return config.RERANK_CANDIDATES[backend]


def rerank(
    question: str,
    hits: list[tuple[Chunk, float]],
    backend: str | None = None,
    n: int | None = None,
) -> list[tuple[Chunk, float, float]]:
    """Score the first `n` hits and sort them best first.

    Returns (chunk, embedding_score, relevance). Hits beyond `n` are not read:
    they keep their order after the scored ones, with relevance -1.
    """
    backend = backend or config.RERANKER
    if backend not in BACKENDS:
        raise ValueError(f"unknown re-ranker {backend!r} (use one of {BACKENDS})")
    n = n or candidates_for(backend)
    chunks = [chunk for chunk, _ in hits[:n]]
    scores = (bge_scores if backend == "bge" else judge_scores)(question, chunks)
    scored = [(chunk, emb, rel) for (chunk, emb), rel in zip(hits[:n], scores)]
    scored.sort(key=lambda t: (t[2], t[1]), reverse=True)  # relevance first, embedding score breaks ties
    return scored + [(chunk, emb, -1.0) for chunk, emb in hits[n:]]
