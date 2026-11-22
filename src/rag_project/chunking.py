"""Step 1 of RAG: load documents and cut them into "chunks".

WHY CHUNK?
    A model compares a question against pieces of text. If a piece is a whole
    10-page document, its meaning is blurry and it wastes the model's limited
    reading space. If a piece is one sentence, it loses context. Here we use a
    simple rule: one paragraph (text between blank lines) = one chunk.

ACCESS TAGS
    Some paragraphs are fine for customers, others are for staff only. Mark a
    paragraph by putting a tag on its first line:

        <!-- access: public -->
        Opening hours are 7am to 9pm.

        <!-- access: internal -->
        The alarm code is 1234.

    The tag is stripped from the text; it is stored next to the chunk instead.
    Untagged paragraphs get `config.DEFAULT_ACCESS` (internal = hidden).
"""

import re
from dataclasses import dataclass
from pathlib import Path

from . import config

# Matches a first line like:  <!-- access: public -->
_TAG = re.compile(r"^\s*<!--\s*access:\s*(\w+)\s*-->\s*", re.IGNORECASE)
_VALID_LEVELS = {config.PUBLIC, config.INTERNAL}


@dataclass(frozen=True)
class Chunk:
    """One searchable piece of text plus where it came from."""

    text: str
    source: str  # file name, handy for debugging / citations
    access: str  # "public" or "internal"


def parse_paragraphs(markdown: str, source: str) -> list[Chunk]:
    """Split Markdown text into chunks (one per paragraph) and read access tags."""
    chunks = []
    for paragraph in markdown.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        access = config.DEFAULT_ACCESS
        match = _TAG.match(paragraph)
        if match:
            level = match.group(1).lower()
            if level not in _VALID_LEVELS:
                raise ValueError(f"{source}: unknown access level {level!r} (use public/internal)")
            access = level
            paragraph = paragraph[match.end() :].strip()
        if paragraph:  # a tag with no text after it is ignored
            chunks.append(Chunk(text=paragraph, source=source, access=access))
    return chunks


def load_chunks(docs_dir: Path = config.DOCS_DIR) -> list[Chunk]:
    """Read every *.md file in `docs_dir` and return all chunks.

    Files are sorted so the chunk order is the same on every run; the cache
    in index.py relies on that.
    """
    chunks = []
    for file in sorted(docs_dir.glob("*.md")):
        chunks += parse_paragraphs(file.read_text(encoding="utf-8"), source=file.name)
    return chunks
