"""Step 1 of RAG: cut documents into "chunks" (this file is pure logic, no file I/O).

TWO SIZES OF CHUNK ("small-to-big" / parent-child retrieval)
    * A CHILD is a small unit (one paragraph, one table, one list, one code
      function). Children are what we embed and search: small text = a sharp,
      focused vector.
    * A PARENT is the whole section a child lives in (up to PARENT_MAX_CHARS).
      Parents are what the chat model reads: it gets the surrounding context
      instead of a lone sentence.

STRUCTURE IS MEANING
    Every chunk remembers its `header_path`, e.g.
    "Student Handbook > IT services for students > Printing", and that path is
    embedded together with the text. "Passwords expire every 12 months" under
    "Students" and the same sentence under "Staff" now mean different things.
    Tables, lists and code are never cut in half unless they are very long.

ACCESS TAGS
    A paragraph can carry a tag on its first line, overriding the file's default:

        <!-- access: internal -->
        The alarm code is 1234.

    Untagged blocks get the default of the folder they live in (docs/public/ or
    docs/internal/, see loaders.py), and anything else is hidden (fail closed).

    Important: a parent is built ONLY from blocks with the same access level as
    its children. So expanding a hit to its parent can never smuggle an
    internal paragraph into a public answer.
"""

import re
from dataclasses import dataclass, field

from . import config

# Matches a first line like:  <!-- access: public -->
_TAG = re.compile(r"^\s*<!--\s*access:\s*(\w+)\s*-->\s*", re.IGNORECASE)
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_VALID_LEVELS = {config.PUBLIC, config.INTERNAL}

MIN_CHILD_CHARS = 80  # a shorter text block (e.g. "Possible sanctions:") is merged into the next one


@dataclass(frozen=True)
class Block:
    """One structural unit of a document, before it is sized into chunks."""

    text: str
    kind: str = "text"  # text | table | list | code | row | image
    access: str | None = None  # None = use the file's default
    page: int | None = None  # PDF page number, when known


@dataclass
class Section:
    """Blocks that share a heading path. `parent_text` overrides the parent (used for CSV)."""

    header_path: str
    blocks: list[Block] = field(default_factory=list)
    parent_text: str | None = None


@dataclass(frozen=True)
class Chunk:
    """One searchable piece of text plus where it came from."""

    text: str
    source: str  # file path relative to docs/, handy for citations
    access: str  # "public" or "internal"
    header_path: str = ""  # "Document > Section > Subsection"
    parent_id: str = ""  # key into Index.parents ("" = the chunk is its own parent)
    kind: str = "text"
    page: int | None = None


def check_level(level: str, source: str) -> str:
    level = level.lower()
    if level not in _VALID_LEVELS:
        raise ValueError(f"{source}: unknown access level {level!r} (use public/internal)")
    return level


# --- Markdown -> sections -------------------------------------------------------


def _make_block(raw: str, source: str) -> Block | None:
    """Turn raw lines into a Block: read the access tag and guess the kind."""
    access = None
    match = _TAG.match(raw)
    if match:
        access = check_level(match.group(1), source)
        raw = raw[match.end() :]
    text = raw.strip()
    if not text:  # a tag with no text after it is ignored
        return None
    first = text.splitlines()[0].lstrip()
    if first.startswith("```"):
        kind = "code"
    elif first.startswith("|"):
        kind = "table"
    elif _LIST_ITEM.match(first):
        kind = "list"
    else:
        kind = "text"
    return Block(text=text, kind=kind, access=access)


def parse_markdown(markdown: str, source: str, root: str = "") -> list[Section]:
    """Split Markdown into sections (one per heading), each holding its blocks.

    `root` is the first part of every header path (usually the file title).
    """
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # the current chain of headings
    current = Section(root)
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        block = _make_block("\n".join(buffer), source)
        buffer.clear()
        if block:
            current.blocks.append(block)

    def path() -> str:
        # The file title only prefixes the path when the document has no H1 of its own.
        has_title = bool(stack) and stack[0][0] == 1
        return " > ".join(([root] if root and not has_title else []) + [t for _, t in stack])

    for line in markdown.splitlines():
        if in_fence:
            buffer.append(line)
            if line.strip().startswith("```"):
                in_fence = False
                flush()
            continue
        if line.strip().startswith("```"):
            flush()
            in_fence = True
            buffer.append(line)
            continue
        heading = _HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            stack = [h for h in stack if h[0] < level] + [(level, heading.group(2))]
            if current.blocks:
                sections.append(current)
            current = Section(path())
        elif not line.strip():
            flush()
        else:
            buffer.append(line)
    flush()
    if current.blocks:
        sections.append(current)
    return sections


# --- Sizing: blocks -> children, children -> parents -------------------------------


def _pack(units: list[str], sep: str, limit: int) -> list[str]:
    """Greedily join units (with `sep`) into pieces no longer than `limit`."""
    pieces: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(sep) + len(unit) > limit:
            pieces.append(current)
            current = unit
        else:
            current = f"{current}{sep}{unit}" if current else unit
    if current:
        pieces.append(current)
    return pieces


def split_block(block: Block) -> list[Block]:
    """Split a block that is too long, respecting its structure."""
    limit = config.CHILD_MAX_CHARS
    text = block.text
    if block.kind == "table" and len(text) > 2 * limit:
        lines = text.splitlines()
        head, rows = "\n".join(lines[:2]), lines[2:]  # repeat header + separator row in every piece
        pieces = [f"{head}\n{body}" for body in _pack(rows, "\n", 2 * limit)]
    elif block.kind == "list" and len(text) > limit:
        items = re.split(r"\n(?=\s*(?:[-*+]|\d+[.)])\s+)", text)
        pieces = _pack(items, "\n", limit)
    elif block.kind == "text" and len(text) > limit:
        pieces = _pack(_SENTENCE_END.split(text), " ", limit)
    else:
        return [block]
    return [Block(p, block.kind, block.access, block.page) for p in pieces]


def _merge_tiny(blocks: list[Block]) -> list[Block]:
    """Glue very short text blocks onto the block that follows them."""
    merged: list[Block] = []
    carry: Block | None = None
    for b in blocks:
        if carry:
            if carry.access == b.access:
                b = Block(f"{carry.text}\n\n{b.text}", b.kind, b.access, carry.page or b.page)
            else:  # never merge across access levels
                merged.append(carry)
            carry = None
        if b.kind == "text" and len(b.text) < MIN_CHILD_CHARS:
            carry = b
            continue
        merged.append(b)
    if carry:  # a tiny block at the very end of a section: keep it by itself
        merged.append(carry)
    return merged


def sections_to_chunks(
    sections: list[Section], source: str, default_access: str
) -> tuple[list[Chunk], dict[str, str]]:
    """Turn sections into child chunks plus the parent texts they point to."""
    chunks: list[Chunk] = []
    parents: dict[str, str] = {}
    for section in sections:
        # Blocks of different access levels never share a parent (see module docstring).
        by_access: dict[str, list[Block]] = {}
        for block in _merge_tiny(section.blocks):
            by_access.setdefault(block.access or default_access, []).append(block)
        for access, blocks in by_access.items():
            pieces = [p for b in blocks for p in split_block(b)]
            # Pack consecutive children into parents of at most PARENT_MAX_CHARS.
            groups: list[list[Block]] = []
            size = 0
            for piece in pieces:
                if groups and size + len(piece.text) > config.PARENT_MAX_CHARS:
                    groups.append([])
                    size = 0
                if not groups:
                    groups.append([])
                groups[-1].append(piece)
                size += len(piece.text)
            for group in groups:
                parent_id = f"{source}#{len(parents)}"
                parents[parent_id] = section.parent_text or "\n\n".join(p.text for p in group)
                for piece in group:
                    chunks.append(
                        Chunk(
                            text=piece.text,
                            source=source,
                            access=access,
                            header_path=section.header_path,
                            parent_id=parent_id,
                            kind=piece.kind,
                            page=piece.page,
                        )
                    )
    return chunks, parents
