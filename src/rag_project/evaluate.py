"""Measure retrieval quality, so every change can be judged by numbers, not hunches.

Run it (needs Ollama for the embeddings):
    uv run python -m rag_project.evaluate                # score the current setup
    uv run python -m rag_project.evaluate --ablate       # compare setups side by side
    uv run python -m rag_project.evaluate --vision       # also compare image-description modes

The question file (eval/questions.jsonl) has three kinds of question, and each has a
"tier" ("easy" if missing, or "hard"):
    answerable    a fact that is in the corpus. We check whether it appears in the
                  retrieved context, and at which rank (Recall@1/3/5 and MRR).
                  A question can need SEVERAL facts ("needs": [...]): multi-hop and
                  aggregation questions only count as found when ALL of them are in
                  the retrieved context, and the rank is that of the last one found.
    restricted    a public user asks for an internal secret. PASS = the secret is
                  nowhere in the retrieved context (a leak would be a FAIL).
    unanswerable  not in the corpus at all. We record the best similarity score, and how
                  many the SCORE_THRESHOLD would correctly turn away.

"Rank" is the position of the first retrieved context that contains the expected
text (and comes from the expected source, if the question names one).
"""

import argparse
import json
import statistics
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

from . import config, index as index_mod
from .ollama_client import OllamaError, embed
from .retrieval import expand_to_parents, top_chunks

QUESTIONS_FILE = config.PROJECT_ROOT / "eval" / "questions.jsonl"
CUTOFF = 3  # an answerable question that is not found in the top 3 is listed as a problem


def load_questions(path: Path = QUESTIONS_FILE) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metrics(ranks: list[int | None]) -> dict:
    """Recall@1/3/5 and MRR for a list of ranks (None = not found)."""
    n = len(ranks)
    if not n:
        return {"n": 0, "R@1": 0.0, "R@3": 0.0, "R@5": 0.0, "MRR": 0.0}
    return {
        "n": n,
        **{f"R@{k}": sum(1 for r in ranks if r and r <= k) / n for k in (1, 3, 5)},
        "MRR": sum(1 / r for r in ranks if r) / n,
    }


@dataclass
class Report:
    name: str
    ranks: dict[str, list[int | None]] = field(default_factory=lambda: defaultdict(list))  # by tier
    leaks: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))  # tier -> [leaks, total]
    answerable_scores: list[float] = field(default_factory=list)
    unanswerable_scores: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    misses: list[str] = field(default_factory=list)

    def all_ranks(self) -> list[int | None]:
        return [r for ranks in self.ranks.values() for r in ranks]

    def total_leaks(self) -> tuple[int, int]:
        return sum(v[0] for v in self.leaks.values()), sum(v[1] for v in self.leaks.values())


def find_rank(contexts, needs: list[dict]) -> int | None:
    """Rank of the answer: for several needed facts, the position of the LAST one found."""
    ranks = []
    for need in needs:
        position = next(
            (
                i
                for i, c in enumerate(contexts, 1)
                if need["contains"].lower() in c.text.lower() and need.get("source", "") in c.source
            ),
            None,
        )
        if position is None:
            return None
        ranks.append(position)
    return max(ranks)


def evaluate(index: index_mod.Index, questions: list[dict], name: str = "current", use_parents: bool = True) -> Report:
    report = Report(name)
    for item in questions:
        tier = item.get("tier", "easy")
        question_vector = embed([config.EMBED_QUERY_PREFIX + item["q"]])[0]
        hits = top_chunks(index, question_vector, item["role"], config.CANDIDATES)
        best = hits[0][1] if hits else 0.0
        contexts = expand_to_parents(index, hits, config.TOP_K, use_parents)

        if item["type"] == "answerable":
            needs = item.get("needs") or [{"contains": item["contains"], "source": item.get("source", "")}]
            rank = find_rank(contexts, needs)
            report.answerable_scores.append(best)
            report.ranks[tier].append(rank)
            if rank is None or rank > CUTOFF:
                report.misses.append(f"[{tier}] rank {rank}: {item['q']}")
        elif item["type"] == "restricted":
            report.leaks[tier][1] += 1
            if any(item["secret"].lower() in c.text.lower() or c.access != "public" for c in contexts):
                report.leaks[tier][0] += 1
                report.misses.append(f"[{tier}] LEAK: {item['q']}")
        else:
            report.unanswerable_scores[tier].append(best)
    return report


def print_reports(reports: list[Report]) -> None:
    print(f"\n{'setup':<30}{'n':>4}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'MRR':>7}{'leaks':>8}")
    for r in reports:
        m = metrics(r.all_ranks())
        leaks, total = r.total_leaks()
        print(f"{r.name:<30}{m['n']:>4}{m['R@1']:>7.2f}{m['R@3']:>7.2f}{m['R@5']:>7.2f}{m['MRR']:>7.2f}{leaks:>5}/{total}")

    main = reports[0]
    print(f"\nBy difficulty tier, '{main.name}':")
    print(f"  {'tier':<8}{'n':>4}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'MRR':>7}")
    for tier in sorted(main.ranks):
        m = metrics(main.ranks[tier])
        print(f"  {tier:<8}{m['n']:>4}{m['R@1']:>7.2f}{m['R@3']:>7.2f}{m['R@5']:>7.2f}{m['MRR']:>7.2f}")
    for tier, (leaks, total) in sorted(main.leaks.items()):
        print(f"  leak test, {tier}: {leaks}/{total} leaked")

    if main.answerable_scores and main.unanswerable_scores:
        weakest = min(main.answerable_scores)
        print(f"\nAbstention (threshold {config.SCORE_THRESHOLD:.2f}; weakest answerable score {weakest:.3f}):")
        for tier, scores in sorted(main.unanswerable_scores.items()):
            rejected = sum(1 for s in scores if s < config.SCORE_THRESHOLD)
            print(f"  {tier:<6} unanswerable: {rejected}/{len(scores)} turned away; "
                  f"best scores median {statistics.median(scores):.3f}, max {max(scores):.3f}")
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
            for mode, label in [("sidecar", "hand-written image text"), ("vision", "vision-model captions")]:
                index = build(mode, **{"rag_project.config.IMAGE_DESCRIPTIONS": mode})
                reports.append(evaluate(index, questions, f"  {label}"))
    except OllamaError as exc:
        raise SystemExit(f"Error: {exc}")
    print_reports(reports)


if __name__ == "__main__":
    main()
