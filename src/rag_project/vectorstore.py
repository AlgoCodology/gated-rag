"""Optional vector database: Qdrant instead of the in-memory NumPy matrix.

WHY?
    NumPy compares the question with EVERY chunk on every search. That is perfect for a
    few thousand chunks and gets slow for hundreds of thousands. A vector database keeps
    an approximate-nearest-neighbour graph (HNSW), so a search touches only a small part
    of the data, and it lives on disk instead of in RAM.

ACCESS CONTROL STAYS IN THE DATABASE QUERY
    The role filter is sent WITH the query (`access in allowed`), so forbidden chunks are
    never candidates and never leave the database. This is the same fail-closed rule as in
    retrieval.py, just enforced by the database.

HOW IT IS USED
    Chosen automatically by corpus size (config.QDRANT_MIN_CHUNKS, see index.choose_store). It runs in Qdrant's embedded local mode (a folder
    under .cache/, no server). To use a real server later, set config.QDRANT_URL.
    Only the vector and the chunk's position and access level are stored in Qdrant; the
    chunk text stays in the Index (it is re-read from the files at every start).
"""

from pathlib import Path

import numpy as np

from . import config
from .chunking import Chunk


class QdrantStore:
    def __init__(self, client, collection: str):
        self.client = client
        self.collection = collection

    @classmethod
    def build(cls, chunks: list[Chunk], vectors: np.ndarray, key: str, location: Path) -> "QdrantStore":
        """Open (or fill) the collection for this exact set of vectors.

        The collection name contains the fingerprint, so a changed corpus or embedding
        model gets a fresh collection and the stale ones are deleted.
        """
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise RuntimeError("The qdrant store needs: uv sync") from exc

        client = QdrantClient(url=config.QDRANT_URL) if config.QDRANT_URL else QdrantClient(path=str(location))
        name = f"chunks_{key[:16]}"
        existing = {c.name for c in client.get_collections().collections}
        for stale in existing - {name}:
            client.delete_collection(stale)
        info = client.get_collection(name) if name in existing else None
        if info is None or (info.points_count or 0) != len(chunks):
            if name in existing:
                client.delete_collection(name)
            client.create_collection(
                name, vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE)
            )
            if config.QDRANT_URL:  # the embedded mode ignores payload indexes; a server uses them to filter fast
                client.create_payload_index(name, "access", models.PayloadSchemaType.KEYWORD)
            points = [
                models.PointStruct(id=i, vector=vectors[i].tolist(), payload={"access": chunks[i].access})
                for i in range(len(chunks))
            ]
            for start in range(0, len(points), 256):
                client.upsert(name, points[start : start + 256])
        return cls(client, name)

    def search(self, question_vector: np.ndarray, allowed: set[str], k: int) -> list[tuple[int, float]]:
        """Best k (chunk position, cosine score) among chunks whose access level is allowed."""
        from qdrant_client import models

        flt = models.Filter(must=[models.FieldCondition(key="access", match=models.MatchAny(any=sorted(allowed)))])
        result = self.client.query_points(
            self.collection, query=question_vector.tolist(), query_filter=flt, limit=k
        )
        return [(int(p.id), float(p.score)) for p in result.points]
