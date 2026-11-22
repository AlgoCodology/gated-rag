"""Step 2 of RAG: build the "index" (chunks + their vectors) and cache it.

WHAT IS AN EMBEDDING?
    An embedding model reads text and outputs a list of numbers (a vector,
    e.g. 768 numbers). Texts with similar MEANING get vectors that point in
    similar directions, even if they share no words. "When do you open?" lands
    near "Opening hours are 7am to 9pm".

WHY NORMALIZE?
    The usual similarity measure is cosine similarity: the angle between two
    vectors. If every vector is scaled to length 1 ("normalized"), cosine
    similarity is just a dot product. We normalize ONCE here, so each question
    needs only one cheap matrix multiplication instead of recomputing lengths.

WHY CACHE?
    Embedding is the slow part of startup. The vectors only change if the
    documents (or the embedding model) change, so we save them to disk and
    reload them next time. A "fingerprint" tells us when the cache is stale.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .chunking import Chunk, load_chunks
from .ollama_client import embed


@dataclass
class Index:
    chunks: list[Chunk]
    vectors: np.ndarray  # shape (n_chunks, dimensions), each row has length 1


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to length 1 (see module docstring)."""
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(lengths, 1e-12)  # max() avoids dividing by zero


def fingerprint(chunks: list[Chunk]) -> str:
    """A short hash of everything that affects the vectors.

    If the text, the embedding model, or the order of chunks changes, the
    hash changes and the cache is rebuilt. (Access tags don't change the
    vectors, but we include them so the cache always mirrors the files.)
    """
    payload = json.dumps(
        {"model": config.EMBED_MODEL, "chunks": [(c.text, c.source, c.access) for c in chunks]}
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _save(path: Path, key: str, vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, key=np.array(key), vectors=vectors)


def _load(path: Path, key: str) -> np.ndarray | None:
    """Return cached vectors if the cache exists and matches `key`, else None."""
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            if str(data["key"]) == key:
                return data["vectors"]
    except (OSError, ValueError, KeyError):
        pass  # corrupt or unreadable cache: just rebuild it
    return None


def build_index(
    docs_dir: Path = config.DOCS_DIR,
    cache_file: Path = config.CACHE_FILE,
    use_cache: bool = True,
) -> Index:
    """Load documents, embed them (or reuse the cache) and return an Index."""
    chunks = load_chunks(docs_dir)
    if not chunks:
        raise RuntimeError(f"No chunks found. Add .md files to {docs_dir}")

    key = fingerprint(chunks)
    vectors = _load(cache_file, key) if use_cache else None
    if vectors is None:
        print(f"Embedding {len(chunks)} chunks (this is the slow part)...")
        vectors = normalize(embed([c.text for c in chunks]))
        if use_cache:
            _save(cache_file, key, vectors)
    else:
        print("Loaded embeddings from cache.")
    return Index(chunks=chunks, vectors=vectors)
