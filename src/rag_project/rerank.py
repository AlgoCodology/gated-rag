"""Optional step 3b: re-rank the candidates with a "judge" model.

WHY RE-RANK?
    Embedding search compares two vectors that were computed SEPARATELY (the
    question never saw the chunk). It is fast but coarse: it finds the right
    neighbourhood, not always the right paragraph, and it scores an off-topic
    question about "tuition fees of Medicine" almost as high as a real one.
    A judge reads the question and one chunk TOGETHER and says how well the
    chunk answers it. That is slower but much sharper (this is the idea of a
    cross-encoder re-ranker, built here from the chat model we already have).

HOW
    For each of the best JUDGE_CANDIDATES chunks we ask the chat model for one digit,
    0 (unrelated) to 9 (clearly answers it). We sort by that digit, with the
    embedding score breaking ties. If even the best digit is below
    JUDGE_THRESHOLD the question is not answerable from the corpus, so
    the caller can say "I don't know" without writing an answer.

COST
    One short model call per candidate (about 4 s each on a laptop CPU), so every
    verdict is cached on disk by (model, question, passage): asking the same
    thing again, or re-running the evaluation, costs nothing.

SECURITY
    The judge only ever sees chunks that already passed the access filter in
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


def passage_text(chunk: Chunk) -> str:
    body = f"{chunk.header_path}\n{chunk.text}" if chunk.header_path else chunk.text
    return body[:MAX_PASSAGE_CHARS]


def parse_score(reply: str) -> int:
    """First digit in the model's reply (0-9); 0 if it did not give one."""
    match = re.search(r"\d", reply)
    return int(match.group()) if match else 0


def _load_cache() -> dict[str, int]:
    try:
        return json.loads(config.JUDGE_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict[str, int]) -> None:
    config.JUDGE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.JUDGE_CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")


def judge(question: str, chunk: Chunk, cache: dict[str, int]) -> int:
    """Score one chunk for one question (0-9), using and filling `cache`."""
    passage = passage_text(chunk)
    key = hashlib.sha256(f"{config.CHAT_MODEL}\x00{question}\x00{passage}".encode("utf-8")).hexdigest()
    if key not in cache:
        reply = ask_llm(JUDGE_PROMPT.format(question=question, passage=passage), max_tokens=3, temperature=0)
        cache[key] = parse_score(reply)
    return cache[key]


def rerank(
    question: str, hits: list[tuple[Chunk, float]], n: int = config.JUDGE_CANDIDATES
) -> list[tuple[Chunk, float, int]]:
    """Judge the first `n` hits and sort them best first.

    Returns (chunk, embedding_score, judge_score). Hits beyond `n` are not
    judged: they keep their order after the judged ones with judge_score -1.
    """
    cache = _load_cache()
    before = len(cache)
    judged = [(chunk, score, judge(question, chunk, cache)) for chunk, score in hits[:n]]
    if len(cache) != before:
        _save_cache(cache)
    judged.sort(key=lambda t: (t[2], t[1]), reverse=True)  # judge first, embedding score breaks ties
    return judged + [(chunk, score, -1) for chunk, score in hits[n:]]
