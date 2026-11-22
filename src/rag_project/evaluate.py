"""Measure retrieval quality, so every change can be judged by numbers, not hunches.

Run it (needs Ollama for the embeddings):
    uv run python -m rag_project.evaluate                # score the current setup
    uv run python -m rag_project.evaluate --ablate       # compare setups side by side

The question file (eval/questions.jsonl) has three kinds of question:
    answerable    a fact that is in the corpus. We check whether it appears in the
                  retrieved context, and at which rank (Recall@1/3/5 and MRR).
    restricted    a public user asks for an internal secret. PASS = the secret is
                  nowhere in the retrieved context (a leak would be a FAIL).
    unanswerable  not in the corpus at all. We record the best similarity score,
                  which helps choose SCORE_THRESHOLD (abstaining is better than guessing).

"Rank" is the position of the first retrieved context that contains the expected
text (and comes from the expected source, if the question names one).
"""

import argparse
import json
import statistics
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from . import config, index as index_mod
from .ollama_client import OllamaError, embed
from .retrieval import expand_to_parents, top_chunks

QUESTIONS_FILE = config.PROJECT_ROOT / "eval" / "questions.jsonl"


def load_questions(path: Path = QUESTIONS_FILE) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass
class Report:
    name: str
    recall: dict[int, float]
    mrr: float
    leaks: int
    restricted: int
    answerable_scores: list[float]
    unanswerable_scores: list[float]
    misses: list[str]


def evaluate(index: index_mod.Index, questions: list[dict], name: str = "current", use_parents: bool = True) -> Report:
    ranks: list[int | None] = []
    misses: list[str] = []
    leaks = restricted = 0
    answerable_scores: list[float] = []
    unanswerable_scores: list[float] = []

    for item in questions:
        question_vector = embed([config.EMBED_QUERY_PREFIX + item["q"]])[0]
        hits = top_chunks(index, question_vector, item["role"], config.CANDIDATES)
        best = hits[0][1] if hits else 0.0
        contexts = expand_to_parents(index, hits, config.TOP_K, use_parents)

        if item["type"] == "answerable":
            answerable_scores.append(best)
            rank = None
            for position, c in enumerate(contexts, 1):
                if item["contains"].lower() in c.text.lower() and item.get("source", "") in c.source:
                    rank = position
                    break
            ranks.append(rank)
            if rank is None or rank > 3:
                misses.append(f"rank {rank}: {item['q']}")
        elif item["type"] == "restricted":
            restricted += 1
            if any(item["secret"].lower() in c.text.lower() or c.access != "public" for c in contexts):
                leaks += 1
                misses.append(f"LEAK: {item['q']}")
        else:
            unanswerable_scores.append(best)

    n = len(ranks)
    return Report(
        name=name,
        recall={k: sum(1 for r in ranks if r and r <= k) / n for k in (1, 3, 5)},
        mrr=sum(1 / r for r in ranks if r) / n,
        leaks=leaks,
        restricted=restricted,
        answerable_scores=answerable_scores,
        unanswerable_scores=unanswerable_scores,
        misses=misses,
    )


def print_reports(reports: list[Report]) -> None:
    print(f"\n{'setup':<26}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'MRR':>7}{'leaks':>8}")
    for r in reports:
        print(f"{r.name:<26}{r.recall[1]:>7.2f}{r.recall[3]:>7.2f}{r.recall[5]:>7.2f}{r.mrr:>7.2f}{r.leaks:>5}/{r.restricted}")
    main = reports[0]
    if main.unanswerable_scores and main.answerable_scores:
        low, high = min(main.answerable_scores), max(main.unanswerable_scores)
        print(f"\nBest-child score, answerable questions:   min {low:.3f}  median {statistics.median(main.answerable_scores):.3f}")
        print(f"Best-child score, unanswerable questions: max {high:.3f}  median {statistics.median(main.unanswerable_scores):.3f}")
        if high < low:
            print(f"A threshold between {high:.3f} and {low:.3f} separates them perfectly "
                  f"(try SCORE_THRESHOLD = {(high + low) / 2:.2f}).")
        else:
            print("The score ranges overlap: any threshold will wrongly reject or accept some questions; "
                  f"a cautious choice is just below the weakest answerable score ({low:.3f}).")
    if main.misses:
        print(f"\nProblems in '{main.name}':")
        for line in main.misses:
            print("  -", line)


def build(name: str, **patches) -> index_mod.Index:
    """Build an index with some settings patched, using a separate cache file per setup."""
    with ExitStack() as stack:
        for target, value in patches.items():
            stack.enter_context(mock.patch(target, value))
        return index_mod.build_index(cache_file=config.PROJECT_ROOT / ".cache" / f"eval_{name}.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--ablate", action="store_true", help="also score variants with one feature switched off")
    parser.add_argument("--vision", action="store_true", help="also score vision-model image captions")
    args = parser.parse_args()
    questions = load_questions()
    print(f"{len(questions)} questions from {QUESTIONS_FILE.name}")

    try:
        full = build("full")
        reports = [evaluate(full, questions, "full (current)")]
        reports.append(evaluate(full, questions, "  children only (no parents)", use_parents=False))
        if args.ablate:
            no_header = build(
                "noheader",
                **{"rag_project.index.embedding_text": lambda c: config.EMBED_DOC_PREFIX + c.text},
            )
            reports.append(evaluate(no_header, questions, "  no header path"))
            no_prefix = build("noprefix", **{"rag_project.config.EMBED_DOC_PREFIX": ""})
            with mock.patch.object(config, "EMBED_QUERY_PREFIX", ""):
                reports.append(evaluate(no_prefix, questions, "  no nomic prefixes"))
        if args.ablate or args.vision:
            vision = build("vision", **{"rag_project.config.IMAGE_DESCRIPTIONS": "vision"})
            reports.append(evaluate(vision, questions, "  vision-model captions"))
            both = build("both", **{"rag_project.config.IMAGE_DESCRIPTIONS": "both"})
            reports.append(evaluate(both, questions, "  hand-written + vision"))
    except OllamaError as exc:
        raise SystemExit(f"Error: {exc}")
    print_reports(reports)


if __name__ == "__main__":
    main()
