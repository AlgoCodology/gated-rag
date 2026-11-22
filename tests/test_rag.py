"""Offline tests: none of these need Ollama. Run with:  uv run python -m unittest discover tests

Fake 2-D vectors stand in for real embeddings so you can see the maths by hand.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rag_project import config, index as index_mod
from rag_project.chunking import Chunk, parse_paragraphs
from rag_project.index import Index, normalize
from rag_project.retrieval import top_chunks


class ChunkingTests(unittest.TestCase):
    def test_tags_are_parsed_and_stripped(self):
        text = "<!-- access: public -->\nHello\n\n<!-- access: internal -->\nSecret\n\nUntagged"
        chunks = parse_paragraphs(text, "f.md")
        self.assertEqual([(c.text, c.access) for c in chunks],
                         [("Hello", "public"), ("Secret", "internal"), ("Untagged", "internal")])

    def test_unknown_level_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_paragraphs("<!-- access: secret -->\nx", "f.md")


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        chunks = [Chunk("pub", "f", "public"), Chunk("sec", "f", "internal")]
        # Both vectors point the same way as the question: "sec" is equally relevant.
        self.index = Index(chunks, normalize(np.array([[1.0, 0.0], [2.0, 0.0]])))
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


class CacheTests(unittest.TestCase):
    def test_second_build_uses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            docs.mkdir()
            (docs / "a.md").write_text("<!-- access: public -->\nHi\n\nThere", encoding="utf-8")
            cache = Path(tmp) / "cache" / "i.npz"
            fake = mock.Mock(side_effect=lambda texts: np.ones((len(texts), 3)))
            with mock.patch.object(index_mod, "embed", fake):
                index_mod.build_index(docs, cache)
                index_mod.build_index(docs, cache)
                self.assertEqual(fake.call_count, 1)  # second run came from cache
                (docs / "a.md").write_text("Changed", encoding="utf-8")
                index_mod.build_index(docs, cache)
                self.assertEqual(fake.call_count, 2)  # edit invalidated the cache


if __name__ == "__main__":
    unittest.main()
