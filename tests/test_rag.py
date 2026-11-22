"""Offline tests: none of these need Ollama. Run with:  uv run python -m unittest discover tests

Fake 2-D vectors stand in for real embeddings so you can see the maths by hand.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rag_project import config, index as index_mod
from rag_project.chunking import Block, Chunk, Section, parse_markdown, sections_to_chunks, split_block
from rag_project.index import Index, embedding_text, normalize
from rag_project.loaders import default_access_for, load_chunks, load_csv, load_html, load_python
from rag_project.retrieval import expand_to_parents, retrieve, top_chunks, u_shaped


def chunks_of(markdown: str, default: str = "public"):
    return sections_to_chunks(parse_markdown(markdown, "f.md", root="Doc"), "f.md", default)


class ChunkingTests(unittest.TestCase):
    def test_tags_are_parsed_and_stripped(self):
        text = "<!-- access: public -->\nHello there, this is a public paragraph that is long enough to stay alone.\n\n" \
               "<!-- access: internal -->\nSecret paragraph that is also long enough to stay alone in its own chunk.\n\n" \
               "Untagged paragraph that is long enough to stay alone and so inherits the default access level."
        chunks, _ = chunks_of(text, default="internal")
        self.assertEqual([c.access for c in chunks], ["public", "internal", "internal"])
        self.assertTrue(chunks[0].text.startswith("Hello"))  # the tag itself is gone

    def test_unknown_level_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_markdown("<!-- access: secret -->\nx", "f.md")

    def test_header_path_follows_headings(self):
        text = "# A\n\n## B\n\nUnder B, a paragraph long enough that it will not be merged into anything else at all.\n\n" \
               "## C\n\n### D\n\nUnder D, another paragraph long enough that it will not be merged into anything else."
        chunks, _ = chunks_of(text)
        self.assertEqual([c.header_path for c in chunks], ["A > B", "A > C > D"])  # H1 replaces the file title

    def test_tables_and_lists_stay_whole(self):
        text = "## T\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\n- one\n- two\n- three"
        chunks, _ = chunks_of(text)
        self.assertEqual([c.kind for c in chunks], ["table", "list"])
        self.assertIn("| 3 | 4 |", chunks[0].text)

    def test_long_table_splits_and_repeats_header(self):
        rows = "\n".join(f"| row{i} | {'x' * 40} |" for i in range(60))
        pieces = split_block(Block(f"| h1 | h2 |\n|---|---|\n{rows}", kind="table"))
        self.assertGreater(len(pieces), 1)
        for p in pieces:
            self.assertTrue(p.text.startswith("| h1 | h2 |\n|---|---|"))

    def test_parent_never_mixes_access_levels(self):
        # Regression: a public paragraph and an internal one in the SAME section.
        text = "## S\n\n<!-- access: public -->\nPublic paragraph that is long enough to stand on its own as a chunk here.\n\n" \
               "<!-- access: internal -->\nInternal secret paragraph that is long enough to stand on its own as a chunk."
        chunks, parents = chunks_of(text)
        public = next(c for c in chunks if c.access == "public")
        self.assertNotIn("secret", parents[public.parent_id])
        self.assertNotEqual(public.parent_id, next(c for c in chunks if c.access == "internal").parent_id)

    def test_tiny_blocks_never_merge_across_access_levels(self):
        # Regression: a tiny internal line must not be glued onto the public paragraph after it.
        text = "<!-- access: internal -->\nTiny secret.\n\n<!-- access: public -->\n" + \
               "A public paragraph that is comfortably longer than the tiny-block merge limit so it stays separate."
        chunks, parents = chunks_of(text)
        public = next(c for c in chunks if c.access == "public")
        self.assertNotIn("Tiny secret", public.text)
        self.assertNotIn("Tiny secret", parents[public.parent_id])

    def test_parents_are_capped(self):
        paragraphs = "\n\n".join(f"Paragraph {i}. " + "word " * 100 for i in range(20))
        chunks, parents = chunks_of(paragraphs)
        self.assertGreater(len(parents), 1)
        self.assertTrue(all(len(p) <= config.PARENT_MAX_CHARS + config.CHILD_MAX_CHARS for p in parents.values()))


class LoaderTests(unittest.TestCase):
    def test_default_access_comes_from_folder_and_fails_closed(self):
        self.assertEqual(default_access_for(Path("public/a.md")), "public")
        self.assertEqual(default_access_for(Path("internal/a.md")), "internal")
        self.assertEqual(default_access_for(Path("elsewhere/a.md")), "internal")
        self.assertEqual(default_access_for(Path("a.md")), "internal")  # loose file: hidden

    def test_html_becomes_headings_lists_and_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "p.html"
            f.write_text("<h1>T</h1><h2>Sec</h2><p>Hello world.</p><ul><li>a</li><li>b</li></ul>"
                         "<table><tr><th>x</th><th>y</th></tr><tr><td>1</td><td>2</td></tr></table>", encoding="utf-8")
            sections = load_html(f)
        kinds = [b.kind for s in sections for b in s.blocks]
        self.assertEqual(kinds, ["text", "list", "table"])
        self.assertEqual(sections[0].header_path, "T > Sec")

    def test_csv_rows_and_parent_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "c.csv"
            f.write_text("code,title\nA1,Alpha\nB2,Beta\n", encoding="utf-8")
            chunks, parents = sections_to_chunks(load_csv(f), "c.csv", "public")
        self.assertEqual(chunks[0].text, "code: A1; title: Alpha")
        self.assertIn("| B2 | Beta |", parents[chunks[0].parent_id])  # parent shows neighbouring rows

    def test_python_is_chunked_by_function(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "m.py"
            f.write_text('"""Module doc."""\nLIMIT = 3\n\ndef one():\n    return 1\n\nclass K:\n    x = 2\n', encoding="utf-8")
            paths = [s.header_path for s in load_python(f)]
        self.assertEqual(paths, ["m.py > module overview", "m.py > one", "m.py > K", "m.py > constants"])

    def test_images_need_a_sidecar_or_a_vision_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp)
            (docs / "public").mkdir()
            (docs / "public" / "a.png").write_bytes(b"not really a png")
            (docs / "public" / "a.alt.txt").write_text("A red bicycle.", encoding="utf-8")
            (docs / "public" / "b.png").write_bytes(b"no description")
            with mock.patch.object(config, "VISION_MODEL", None):
                chunks, _ = load_chunks(docs)
        self.assertEqual([(c.source, c.kind, c.text) for c in chunks], [("public/a.png", "image", "A red bicycle.")])

    def test_real_corpus_loads_and_internal_stays_in_internal_folder(self):
        chunks, parents = load_chunks()
        self.assertGreater(len(chunks), 100)
        for c in chunks:
            self.assertIn(c.parent_id, parents)
            if c.source.startswith("internal/"):
                self.assertEqual(c.access, "internal")


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        chunks = [Chunk("pub", "f", "public", parent_id="p1"), Chunk("sec", "f", "internal", parent_id="p2")]
        # Both vectors point the same way as the question: "sec" is equally relevant.
        self.index = Index(chunks, normalize(np.array([[1.0, 0.0], [2.0, 0.0]])), {"p1": "PARENT-PUB", "p2": "PARENT-SEC"})
        self.question = np.array([1.0, 0.0])

    def test_public_role_never_sees_internal(self):
        texts = [c.text for c, _ in top_chunks(self.index, self.question, "public")]
        self.assertEqual(texts, ["pub"])

    def test_internal_role_sees_everything(self):
        texts = {c.text for c, _ in top_chunks(self.index, self.question, "internal")}
        self.assertEqual(texts, {"pub", "sec"})

    def test_ranking_prefers_similar_direction(self):
        chunks = [Chunk("x-axis", "f", "public"), Chunk("y-axis", "f", "public")]
        idx = Index(chunks, normalize(np.array([[1.0, 0.0], [0.0, 1.0]])))
        best = top_chunks(idx, np.array([0.1, 5.0]), "public", k=1)[0][0]
        self.assertEqual(best.text, "y-axis")

    def test_public_context_never_contains_internal_parent(self):
        hits = top_chunks(self.index, self.question, "public")
        self.assertEqual([c.text for c in expand_to_parents(self.index, hits)], ["PARENT-PUB"])

    def test_children_of_one_parent_count_once(self):
        chunks = [Chunk("a", "f", "public", parent_id="P"), Chunk("b", "f", "public", parent_id="P"),
                  Chunk("c", "f", "public", parent_id="Q")]
        idx = Index(chunks, normalize(np.eye(3)), {"P": "parent P", "Q": "parent Q"})
        hits = [(chunks[0], 0.9), (chunks[1], 0.8), (chunks[2], 0.7)]
        contexts = expand_to_parents(idx, hits)
        self.assertEqual([c.text for c in contexts], ["parent P", "parent Q"])
        self.assertEqual(contexts[0].score, 0.9)  # the best child sets the score

    def test_u_shaped_puts_best_at_the_edges(self):
        self.assertEqual(u_shaped([1, 2, 3, 4, 5]), [1, 3, 5, 4, 2])
        self.assertEqual(u_shaped([1, 2]), [1, 2])
        self.assertEqual(u_shaped([]), [])

    def test_threshold_makes_retrieve_return_nothing(self):
        with mock.patch("rag_project.retrieval.embed", return_value=np.array([[1.0, 0.0]])):
            self.assertEqual(len(retrieve(self.index, "q", "public", threshold=0.5)), 1)
            self.assertEqual(retrieve(self.index, "q", "public", threshold=1.5), [])

    def test_query_gets_the_search_prefix(self):
        fake = mock.Mock(return_value=np.array([[1.0, 0.0]]))
        with mock.patch("rag_project.retrieval.embed", fake):
            retrieve(self.index, "hello", "public")
        self.assertEqual(fake.call_args.args[0], [config.EMBED_QUERY_PREFIX + "hello"])


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(config, "JUDGE_CACHE_FILE", Path(self._tmp.name) / "judge.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)
        self.hits = [(Chunk("weak", "f", "public"), 0.9), (Chunk("strong", "f", "public"), 0.5),
                     (Chunk("tail", "f", "public"), 0.4)]

    def fake_llm(self, prompt, **kwargs):
        return "9" if "strong" in prompt.split("Passage:")[1] else "2"

    def test_parse_score(self):
        from rag_project.rerank import parse_score
        self.assertEqual(parse_score("7"), 7)
        self.assertEqual(parse_score("Score: 4."), 4)
        self.assertEqual(parse_score("no idea"), 0)

    def test_judge_overrules_embedding_order_and_skips_the_tail(self):
        from rag_project.rerank import rerank
        with mock.patch("rag_project.rerank.ask_llm", side_effect=self.fake_llm):
            out = rerank("q", self.hits, n=2)
        self.assertEqual([(c.text, j) for c, _, j in out], [("strong", 9), ("weak", 2), ("tail", -1)])

    def test_verdicts_are_cached(self):
        from rag_project.rerank import rerank
        fake = mock.Mock(side_effect=self.fake_llm)
        with mock.patch("rag_project.rerank.ask_llm", fake):
            rerank("q", self.hits, n=2)
            rerank("q", self.hits, n=2)
        self.assertEqual(fake.call_count, 2)  # the second call came entirely from the cache

    def test_retrieve_abstains_when_judge_finds_nothing(self):
        index = Index([c for c, _ in self.hits], normalize(np.eye(3)))
        with mock.patch("rag_project.retrieval.embed", return_value=np.array([[1.0, 0.0, 0.0]])), \
             mock.patch("rag_project.rerank.ask_llm", return_value="1"), \
             mock.patch.object(config, "JUDGE_THRESHOLD", 5):
            self.assertEqual(retrieve(index, "q", "public", threshold=0.0, use_judge=True), [])
            with mock.patch.object(config, "JUDGE_THRESHOLD", 0):
                self.assertTrue(retrieve(index, "q", "public", threshold=0.0, use_judge=True))

    def test_judge_never_sees_hidden_chunks(self):
        index = Index([Chunk("pub", "f", "public"), Chunk("SECRET", "f", "internal")], normalize(np.eye(2)))
        seen = []
        with mock.patch("rag_project.retrieval.embed", return_value=np.array([[1.0, 1.0]])), \
             mock.patch("rag_project.rerank.ask_llm", side_effect=lambda p, **k: seen.append(p) or "5"):
            retrieve(index, "q", "public", threshold=0.0, use_judge=True)
        self.assertTrue(seen)
        self.assertFalse(any("SECRET" in p for p in seen))


class IndexTests(unittest.TestCase):
    def test_embedding_text_has_prefix_and_header_path(self):
        c = Chunk("Opening hours", "f", "public", header_path="Doc > Hours")
        self.assertEqual(embedding_text(c), config.EMBED_DOC_PREFIX + "Doc > Hours\nOpening hours")

    def test_second_build_uses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            (docs / "public").mkdir(parents=True)
            (docs / "public" / "a.md").write_text("Hi there, this is a paragraph.\n\nAnd another one right here.", encoding="utf-8")
            cache = Path(tmp) / "cache" / "i.npz"
            fake = mock.Mock(side_effect=lambda texts: np.ones((len(texts), 3)))
            with mock.patch.object(index_mod, "embed", fake):
                index_mod.build_index(docs, cache)
                index_mod.build_index(docs, cache)
                self.assertEqual(fake.call_count, 1)  # second run came from cache
                (docs / "public" / "a.md").write_text("Changed", encoding="utf-8")
                index_mod.build_index(docs, cache)
                self.assertEqual(fake.call_count, 2)  # edit invalidated the cache

    def test_old_cafe_handbook_fixture_still_parses(self):
        text = (Path(__file__).parent / "fixtures" / "byte_bistro.md").read_text(encoding="utf-8")
        chunks, _ = sections_to_chunks(parse_markdown(text, "byte_bistro.md"), "byte_bistro.md", "internal")
        self.assertTrue(any(c.access == "public" for c in chunks))
        self.assertTrue(any("alarm" in c.text and c.access == "internal" for c in chunks))


if __name__ == "__main__":
    unittest.main()
