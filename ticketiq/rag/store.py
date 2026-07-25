"""An in-memory cosine-similarity vector store.

Two interchangeable backends:

* ``tfidf`` (default) — the project's own :class:`TfidfVectorizer` fitted over
  the knowledge-base chunks, compared by cosine on sparse L2-normalised
  vectors. On a corpus this small it beats a hashed dense embedding
  comfortably, because rare high-signal terms ("SAML", "chargeback", "VAT")
  keep their own dimension instead of colliding into a shared bucket.
* ``dense`` — any :class:`~ticketiq.rag.embeddings.Embedder`: the offline
  hashing embedder, or ``sentence-transformers`` when installed. Select with
  ``RAG_BACKEND=dense``.

FAISS/Chroma would be the choice at scale; with ~25 chunks a brute-force scan
is exact and instant, and the interface (:meth:`add` / :meth:`search`) is the
same shape, so swapping the backend later touches only this file.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from functools import lru_cache

from ticketiq.config import get_settings
from ticketiq.ml.text import tokenize
from ticketiq.ml.vectorizer import SparseVector, TfidfVectorizer
from ticketiq.rag.embeddings import Embedder, cosine
from ticketiq.rag.kb import Chunk, load_kb

HEADING_BONUS = 0.15


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


def sparse_cosine(a: SparseVector, b: SparseVector) -> float:
    """Dot product of two L2-normalised sparse vectors."""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(idx, 0.0) for idx, value in a.items())


def _heading_overlap(query_tokens: set[str], heading_tokens: set[str]) -> float:
    if not query_tokens or not heading_tokens:
        return 0.0
    return len(query_tokens & heading_tokens) / len(query_tokens)


class VectorStore:
    """Brute-force cosine index over knowledge-base chunks."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder
        self._chunks: list[Chunk] = []
        self._sparse: list[SparseVector] = []
        self._dense: list[list[float]] = []
        self._heading_tokens: list[set[str]] = []
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def backend(self) -> str:
        return "dense" if self.embedder is not None else "tfidf"

    def add(self, chunks: list[Chunk]) -> None:
        """Index chunks. Re-fits the TF-IDF vocabulary over the full corpus."""
        if not chunks:
            return
        with self._lock:
            self._chunks.extend(chunks)
            self._heading_tokens.extend(set(tokenize(f"{c.heading} {c.title}")) for c in chunks)
            texts = [c.embedding_text for c in self._chunks]
            if self.embedder is not None:
                self._dense = self.embedder.embed_many(texts)
            else:
                self._sparse = self._vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        """Return the ``top_k`` chunks most similar to ``query``."""
        if top_k <= 0 or not self._chunks:
            return []
        query_tokens = set(tokenize(query))

        if self.embedder is not None:
            query_dense = self.embedder.embed(query)
            similarities = [cosine(query_dense, vec) for vec in self._dense]
        else:
            query_sparse = self._vectorizer.transform_one(query)
            similarities = [sparse_cosine(query_sparse, vec) for vec in self._sparse]

        hits = [
            SearchHit(
                chunk=chunk,
                score=round(
                    similarity + HEADING_BONUS * _heading_overlap(query_tokens, heading),
                    6,
                ),
            )
            for chunk, similarity, heading in zip(
                self._chunks, similarities, self._heading_tokens, strict=True
            )
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return hits[:top_k]


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Build the index once per process from the on-disk knowledge base."""
    settings = get_settings()
    embedder: Embedder | None = None
    if os.environ.get("RAG_BACKEND", "tfidf").lower() == "dense":
        from ticketiq.rag.embeddings import get_embedder

        embedder = get_embedder()
    store = VectorStore(embedder=embedder)
    store.add(
        load_kb(
            settings.kb_dir,
            chunk_size=settings.chunk_size_chars,
            overlap=settings.chunk_overlap_chars,
        )
    )
    return store


def reset_vector_store() -> None:
    """Test hook."""
    get_vector_store.cache_clear()
