"""Tests for corpus loading and chunking."""

import os
import tempfile
import unittest

from rag.corpus import Chunk, chunk_text, load_dir, load_sample


class ChunkTextTests(unittest.TestCase):
    def test_splits_on_blank_lines(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird."
        chunks = chunk_text(text, source="doc")
        self.assertEqual([c.text for c in chunks],
                         ["First paragraph.", "Second paragraph.", "Third."])

    def test_joins_wrapped_lines(self):
        text = "a line\nthat wraps\n\nnext para"
        chunks = chunk_text(text, source="doc")
        self.assertEqual(chunks[0].text, "a line that wraps")

    def test_source_numbering_is_one_based(self):
        chunks = chunk_text("one\n\ntwo", source="handbook.md")
        self.assertEqual([c.source for c in chunks], ["handbook.md#1", "handbook.md#2"])

    def test_collapses_multiple_blank_lines(self):
        chunks = chunk_text("a\n\n\n\nb", source="d")
        self.assertEqual(len(chunks), 2)

    def test_trailing_text_kept(self):
        chunks = chunk_text("only one paragraph, no trailing blank", source="d")
        self.assertEqual(len(chunks), 1)

    def test_empty_text(self):
        self.assertEqual(chunk_text("\n\n  \n", source="d"), [])


class LoadDirTests(unittest.TestCase):
    def test_loads_txt_and_md_only(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.md"), "w") as fh:
                fh.write("alpha\n\nbeta")
            with open(os.path.join(d, "b.txt"), "w") as fh:
                fh.write("gamma")
            with open(os.path.join(d, "ignore.py"), "w") as fh:
                fh.write("delta = 1")
            chunks = load_dir(d)
        texts = {c.text for c in chunks}
        self.assertEqual(texts, {"alpha", "beta", "gamma"})
        self.assertNotIn("delta = 1", texts)

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(load_dir(d), [])


class LoadSampleTests(unittest.TestCase):
    def test_sample_is_nonempty(self):
        chunks = load_sample()
        self.assertGreater(len(chunks), 5)
        self.assertTrue(all(isinstance(c, Chunk) for c in chunks))

    def test_sample_mentions_key_facts(self):
        blob = " ".join(c.text for c in load_sample())
        for fact in ("Helix Dynamics", "Kestrel-7", "AuroraCell", "Loomweave", "2019"):
            self.assertIn(fact, blob)


if __name__ == "__main__":
    unittest.main()
