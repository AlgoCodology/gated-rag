"""File loaders: turn each kind of file in docs/ into Sections, then into chunks.

    .md .html   -> heading tree, with tables / lists kept whole
    .csv        -> one chunk per row; the parent is a block of neighbouring rows
    .pdf        -> headings recovered from font sizes, tables detected, text in reading order
    .py         -> one chunk per function / class (found with the `ast` module)
    .png .jpg   -> a text description of the image: `<name>.alt.txt` next to it,
                   or, when config.VISION_MODEL is set, a caption from a vision model

Every chunk of a file gets that file's default access level, taken from its
top-level folder (docs/public/..., docs/internal/...). A file anywhere else is
treated as internal (fail closed).
"""

import ast
import hashlib
import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

from . import config
from .chunking import Block, Chunk, Section, parse_markdown, sections_to_chunks

SUPPORTED = {".md", ".html", ".htm", ".csv", ".pdf", ".py", ".png", ".jpg", ".jpeg"}
CSV_ROWS_PER_PARENT = 8


def _title(path: Path) -> str:
    return path.stem.replace("_", " ").title()


# --- Markdown / HTML ---------------------------------------------------------------


def load_markdown(path: Path) -> list[Section]:
    return parse_markdown(path.read_text(encoding="utf-8"), path.name, root=_title(path))


def _md_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [[c.replace("|", "/").replace("\n", " ").strip() for c in r] + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * width) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


class _HtmlToMarkdown(HTMLParser):
    """Tiny HTML -> Markdown converter (headings, paragraphs, lists, tables)."""

    def __init__(self) -> None:
        super().__init__()
        self.out: list[str] = []
        self.buf: list[str] = []
        self.heading = 0
        self.in_li = False
        self.skip = 0
        self.rows: list[list[str]] | None = None
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def _text(self) -> str:
        text = " ".join("".join(self.buf).split())
        self.buf = []
        return text

    def _para(self) -> None:
        text = self._text()
        if text:
            self.out += [text, ""]

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "title"):
            self.skip += 1
        elif re.fullmatch(r"h[1-6]", tag):
            self._para()
            self.heading = int(tag[1])
        elif tag in ("p", "ul", "ol", "div"):
            self._para()
        elif tag == "li":
            self._para()
            self.in_li = True
        elif tag == "table":
            self._para()
            self.rows = []
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []
        elif tag == "br":
            (self.cell if self.cell is not None else self.buf).append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "title"):
            self.skip -= 1
        elif re.fullmatch(r"h[1-6]", tag):
            self.out += ["#" * self.heading + " " + self._text(), ""]
            self.heading = 0
        elif tag == "p":
            self._para()
        elif tag == "li":
            text = self._text()
            if text:
                self.out.append("- " + text)
            self.in_li = False
        elif tag in ("ul", "ol"):
            self.out.append("")
        elif tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None and self.rows is not None:
            self.rows.append(self.row)
            self.row = None
        elif tag == "table" and self.rows is not None:
            self.out += [_md_table(self.rows), ""]
            self.rows = None

    def handle_data(self, data):
        if self.skip:
            return
        (self.cell if self.cell is not None else self.buf).append(data)


def load_html(path: Path) -> list[Section]:
    parser = _HtmlToMarkdown()
    parser.feed(path.read_text(encoding="utf-8"))
    parser._para()
    return parse_markdown("\n".join(parser.out), path.name, root=_title(path))


# --- CSV ----------------------------------------------------------------------------


def load_csv(path: Path) -> list[Section]:
    import pandas as pd

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    columns = list(frame.columns)
    sections = []
    for start in range(0, len(frame), CSV_ROWS_PER_PARENT):
        part = frame.iloc[start : start + CSV_ROWS_PER_PARENT]
        rows = part.values.tolist()
        sections.append(
            Section(
                header_path=_title(path),
                blocks=[
                    Block("; ".join(f"{c}: {v}" for c, v in zip(columns, row)), kind="row")
                    for row in rows
                ],
                parent_text=_md_table([columns] + rows),
            )
        )
    return sections


# --- Python source -------------------------------------------------------------------


def load_python(path: Path) -> list[Section]:
    """One chunk per top-level function/class, so code is cut along its own structure."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    root = path.name
    sections = []
    docstring = ast.get_docstring(tree)
    if docstring:
        sections.append(Section(f"{root} > module overview", [Block(docstring)]))
    constants = []
    for node in tree.body:
        segment = ast.get_source_segment(source, node) or ""
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            sections.append(Section(f"{root} > {node.name}", [Block(segment, kind="code")]))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            constants.append(segment)
    if constants:
        sections.append(Section(f"{root} > constants", [Block("\n".join(constants), kind="code")]))
    return sections


# --- Images -------------------------------------------------------------------------


def _load_captions() -> dict[str, str]:
    try:
        return json.loads(config.CAPTION_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def caption_image(image_bytes: bytes) -> str | None:
    """Describe an image with the vision model (cached by content hash). None if disabled."""
    if not config.VISION_MODEL:
        return None
    from .ollama_client import describe_image

    key = hashlib.sha256(image_bytes).hexdigest() + "|" + config.VISION_MODEL
    captions = _load_captions()
    if key not in captions:
        captions[key] = describe_image(
            image_bytes,
            "Describe this image for a search index. Mention every visible text label, "
            "number, chart axis and trend, and what the image shows.",
            config.VISION_MODEL,
        )
        config.CAPTION_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.CAPTION_CACHE_FILE.write_text(json.dumps(captions), encoding="utf-8")
    return captions[key]


def load_image(path: Path) -> list[Section]:
    sidecar = path.with_suffix(".alt.txt")
    written = sidecar.read_text(encoding="utf-8").strip() if sidecar.exists() else None
    if config.IMAGE_DESCRIPTIONS == "both":  # exact human facts + the model's wider coverage
        generated = caption_image(path.read_bytes())
        caption = "\n\n".join(filter(None, [written, generated and f"Detailed description: {generated}"]))
    elif config.IMAGE_DESCRIPTIONS == "vision":  # prefer the model, fall back to the human text
        caption = caption_image(path.read_bytes()) or written
    else:  # prefer the human text, fall back to the model
        caption = written or caption_image(path.read_bytes())
    if not caption:
        print(f"  (skipping {path.name}: no {sidecar.name} and no vision model configured)")
        return []
    return [Section(f"Image: {path.name}", [Block(caption, kind="image")])]


# --- PDF ----------------------------------------------------------------------------


def load_pdf(path: Path) -> list[Section]:
    """Recover structure from a PDF: headings by font size, tables, reading order.

    Idea: the most common font size is the body text; noticeably bigger, short
    lines are headings, and bigger = higher level. Never cut across a heading.
    """
    import pymupdf

    doc = pymupdf.open(path)
    sizes: Counter = Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    sizes[round(span["size"], 1)] += len(span["text"])
    if not sizes:
        return []
    body = sizes.most_common(1)[0][0]
    heading_sizes = sorted((s for s in sizes if s >= body * 1.15), reverse=True)

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    root = _title(path)
    current = Section(root)

    def path_text() -> str:
        return " > ".join(t for _, t in stack) or root

    def table_block(table, page, page_no) -> Block | None:
        # Read each cell by its bounding box: more reliable than table.extract() for generated PDFs.
        rows = [
            [" ".join(page.get_textbox(pymupdf.Rect(c)).split()) if c else "" for c in row.cells]
            for row in table.rows
        ]
        return Block(_md_table(rows), kind="table", page=page_no) if rows else None

    for page in doc:
        page_no = page.number + 1
        tables = sorted(page.find_tables().tables, key=lambda t: t.bbox[1])
        table_boxes = [pymupdf.Rect(t.bbox) for t in tables]
        for block in page.get_text("dict", sort=True)["blocks"]:
            if block.get("type") != 0:
                continue
            # Emit tables that sit above this block, so they land in the right section.
            while tables and tables[0].bbox[1] <= block["bbox"][1]:
                if (tb := table_block(tables.pop(0), page, page_no)):
                    current.blocks.append(tb)
            if any(pymupdf.Rect(block["bbox"]).intersects(box) for box in table_boxes):
                continue  # this text belongs to a table
            lines = ["".join(s["text"] for s in ln["spans"]).strip() for ln in block["lines"]]
            text = re.sub(r"-\n(?=[a-z])", "", "\n".join(lines))
            text = " ".join(text.split())
            if len(text) < 3 or text.isdigit():
                continue
            size = max(round(s["size"], 1) for ln in block["lines"] for s in ln["spans"])
            if size in heading_sizes and len(text) < 120:
                level = heading_sizes.index(size) + 1
                stack = [h for h in stack if h[0] < level] + [(level, text)]
                if current.blocks:
                    sections.append(current)
                current = Section(path_text())
            else:
                current.blocks.append(Block(text, page=page_no))
        for table in tables:  # tables below the last text block of the page
            if (tb := table_block(table, page, page_no)):
                current.blocks.append(tb)
        if config.VISION_MODEL:
            for info in page.get_image_info(xrefs=True):
                if info["width"] < 200 or info["height"] < 150 or not info.get("xref"):
                    continue
                caption = caption_image(doc.extract_image(info["xref"])["image"])
                if caption:
                    current.blocks.append(Block(f"Figure on page {page_no}: {caption}", "image", page=page_no))
    if current.blocks:
        sections.append(current)
    return sections


# --- Dispatch -----------------------------------------------------------------------

_LOADERS = {
    ".md": load_markdown,
    ".html": load_html,
    ".htm": load_html,
    ".csv": load_csv,
    ".pdf": load_pdf,
    ".py": load_python,
    ".png": load_image,
    ".jpg": load_image,
    ".jpeg": load_image,
}


def default_access_for(relative: Path) -> str:
    """public/ -> public, internal/ -> internal, anywhere else -> fail closed."""
    top = relative.parts[0] if len(relative.parts) > 1 else ""
    return config.FOLDER_ACCESS.get(top, config.DEFAULT_ACCESS)


def load_file(path: Path, docs_dir: Path) -> tuple[list[Chunk], dict[str, str]]:
    relative = path.relative_to(docs_dir)
    sections = _LOADERS[path.suffix.lower()](path)
    return sections_to_chunks(sections, relative.as_posix(), default_access_for(relative))


def load_chunks(docs_dir: Path = config.DOCS_DIR) -> tuple[list[Chunk], dict[str, str]]:
    """Read every supported file under `docs_dir`; return (child chunks, parent texts).

    Files are sorted so the chunk order is the same on every run; the cache
    in index.py relies on that.
    """
    chunks: list[Chunk] = []
    parents: dict[str, str] = {}
    for path in sorted(p for p in docs_dir.rglob("*") if p.suffix.lower() in SUPPORTED):
        file_chunks, file_parents = load_file(path, docs_dir)
        chunks += file_chunks
        parents |= file_parents
    return chunks, parents
