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
from .rerank import BACKENDS
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
    rel_answerable: list[float] = field(default_factory=list)  # best re-ranker relevance (0-1) per answerable question
    rel_unanswerable: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    n_questions: dict[str, int] = field(default_factory=lambda: defaultdict(int))  # by question type
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


def evaluate(
    index: index_mod.Index,
    questions: list[dict],
    name: str = "current",
    use_parents: bool = True,
    reranker: str = "none",
) -> Report:
    """Score one setup. `reranker` is "none", "bge" or "judge" (see rerank.py)."""
    report = Report(name)
    wanted = config.CANDIDATES
    if reranker != "none":
        from .rerank import candidates_for, rerank

        wanted = max(wanted, candidates_for(reranker))
    for item in questions:
        tier = item.get("tier", "easy")
        report.n_questions[item["type"]] += 1
        question_vector = embed([config.EMBED_QUERY_PREFIX + item["q"]])[0]
        hits = top_chunks(index, question_vector, item["role"], wanted)
        best = hits[0][1] if hits else 0.0
        best_rel = None
        if reranker != "none" and hits:
            scored = rerank(item["q"], hits, reranker)
            best_rel = scored[0][2]
            hits = [(chunk, score) for chunk, score, _ in scored]
        contexts = expand_to_parents(index, hits, config.TOP_K, use_parents)

        if item["type"] == "answerable":
            needs = item.get("needs") or [{"contains": item["contains"], "source": item.get("source", "")}]
            rank = find_rank(contexts, needs)
            report.answerable_scores.append(best)
            if best_rel is not None:
                report.rel_answerable.append(best_rel)
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
            if best_rel is not None:
                report.rel_unanswerable[tier].append(best_rel)
    return report


def print_reports(reports: list[Report]) -> None:
    print(f"\n{'setup':<30}{'n':>4}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'MRR':>7}{'leaks':>8}")
    for r in reports:
        m = metrics(r.all_ranks())
        leaks, total = r.total_leaks()
        print(f"{r.name:<30}{m['n']:>4}{m['R@1']:>7.2f}{m['R@3']:>7.2f}{m['R@5']:>7.2f}{m['MRR']:>7.2f}{leaks:>5}/{total}")

    main = reports[0]
    counts = ", ".join(f"{n} {kind}" for kind, n in sorted(main.n_questions.items()))
    print(
        f"\nAll {sum(main.n_questions.values())} questions = {counts}. The table above and the tier table "
        f"cover only the {main.n_questions['answerable']} answerable ones; restricted questions are scored "
        "as leaks and unanswerable ones by abstention."
    )
    print("\nBy difficulty tier, every setup:")
    print(f"  {'setup':<30}{'tier':<7}{'n':>4}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'MRR':>7}")
    for tier in sorted({t for r in reports for t in r.ranks}):
        for r in reports:
            m = metrics(r.ranks[tier])
            print(f"  {r.name:<30}{tier:<7}{m['n']:>4}{m['R@1']:>7.2f}{m['R@3']:>7.2f}{m['R@5']:>7.2f}{m['MRR']:>7.2f}")
    for tier, (leaks, total) in sorted(main.leaks.items()):
        print(f"  leak test, {tier}: {leaks}/{total} leaked")

    if main.answerable_scores and main.unanswerable_scores:
        weakest = min(main.answerable_scores)
        print(f"\nAbstention (threshold {config.SCORE_THRESHOLD:.2f}; weakest answerable score {weakest:.3f}):")
        for tier, scores in sorted(main.unanswerable_scores.items()):
            rejected = sum(1 for s in scores if s < config.SCORE_THRESHOLD)
            print(f"  {tier:<6} unanswerable: {rejected}/{len(scores)} turned away; "
                  f"best scores median {statistics.median(scores):.3f}, max {max(scores):.3f}")
    for r in reports:
        if not r.rel_answerable:
            continue
        total = sum(len(v) for v in r.rel_unanswerable.values())
        print(
            f"\nRelevance (0-1) of the best candidate, '{r.name.strip()}': answerable min "
            f"{min(r.rel_answerable):.3f}, median {statistics.median(r.rel_answerable):.3f}"
        )
        for tier, values in sorted(r.rel_unanswerable.items()):
            print(f"  {tier:<6} unanswerable: {[round(v, 2) for v in sorted(values)]}")
        for cut in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            turned = sum(1 for vs in r.rel_unanswerable.values() for v in vs if v < cut)
            lost = sum(1 for v in r.rel_answerable if v < cut)
            print(
                f"  RERANK_THRESHOLD {cut:.2f}: turns away {turned}/{total} unanswerable, "
                f"wrongly rejects {lost}/{len(r.rel_answerable)} answerable"
            )
    if main.misses:
        print(f"\nProblems in '{main.name}':")
        for line in main.misses:
            print("  -", line)


def build(name: str, cache_name: str | None = None, **patches) -> index_mod.Index:
    """Build an index with some settings patched, using a separate cache file per setup."""
    with ExitStack() as stack:
        for target, value in patches.items():
            stack.enter_context(mock.patch(target, value))
        return index_mod.build_index(cache_file=config.PROJECT_ROOT / ".cache" / f"eval_{cache_name or name}.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--ablate", action="store_true", help="also score variants with one feature switched off")
    parser.add_argument("--vision", action="store_true", help="also score vision-model image captions")
    parser.add_argument(
        "--rerank", nargs="+", choices=BACKENDS, default=[],
        help="also score these re-rankers (the judge is slow the first time; verdicts are cached)",
    )
    parser.add_argument("--compare-stores", action="store_true", help="also score the vector store that was not auto-selected")
    args = parser.parse_args()
    questions = load_questions()
    print(f"{len(questions)} questions from {QUESTIONS_FILE.name}")

    try:
        full = build("full")
        store = "qdrant" if full.store is not None else "numpy"
        reports = [evaluate(full, questions, f"full (current, {store})")]
        reports.append(evaluate(full, questions, "  children only (no parents)", use_parents=False))
        for backend in args.rerank:
            reports.append(evaluate(full, questions, f"  + {backend} re-ranking", reranker=backend))
        if args.compare_stores:  # the store the size rule did NOT pick, with the same vectors
            other = "numpy" if full.store is not None else "qdrant"
            limit = 10**9 if other == "numpy" else 0
            alt = build(other, cache_name="full", **{"rag_project.config.QDRANT_MIN_CHUNKS": limit})
            reports.append(evaluate(alt, questions, f"  {other} store (forced)"))
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
