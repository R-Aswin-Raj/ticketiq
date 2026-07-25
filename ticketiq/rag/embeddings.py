"""Embedding backends.

The default is a deterministic hashing embedder: no model download, no network,
identical vectors on every machine and in CI. It is a signed random-projection
of an IDF-free bag of n-grams, which is weak semantically but perfectly
adequate for a five-document knowledge base and keeps the whole service
reproducible.

Set ``EMBEDDING_BACKEND=sentence-transformers`` to swap in a real dense model
when the dependency is installed; the rest of the RAG layer is unchanged.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from ticketiq.config import get_settings
from ticketiq.ml.text import featurize

Vector = list[float]


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> Vector: ...

    def embed_many(self, texts: Sequence[str]) -> list[Vector]: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; inputs are expected but not required to be unit norm."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class HashingEmbedder:
    """Signed feature hashing (a.k.a. the hashing trick) into ``dim`` buckets."""

    def __init__(self, dim: int = 256, ngram_range: tuple[int, int] = (1, 2)) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.ngram_range = ngram_range

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def embed(self, text: str) -> Vector:
        vec = [0.0] * self.dim
        tokens = featurize(text, ngram_range=self.ngram_range)
        if not tokens:
            return vec
        for token, count in Counter(tokens).items():
            idx, sign = self._bucket(token)
            # Sublinear term frequency damps long-document dominance.
            vec[idx] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_many(self, texts: Sequence[str]) -> list[Vector]:
        return [self.embed(t) for t in texts]


class SentenceTransformerEmbedder:  # pragma: no cover - optional dependency
    """Adapter for ``sentence-transformers`` models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> Vector:
        return [float(x) for x in self._model.encode(text, normalize_embeddings=True)]

    def embed_many(self, texts: Sequence[str]) -> list[Vector]:
        return [
            [float(x) for x in row]
            for row in self._model.encode(list(texts), normalize_embeddings=True)
        ]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    settings = get_settings()
    backend = os.environ.get("EMBEDDING_BACKEND", "hashing").lower()
    if backend in ("sentence-transformers", "st"):
        try:
            return SentenceTransformerEmbedder(
                os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            )
        except Exception:  # pragma: no cover - falls back when unavailable
            pass
    return HashingEmbedder(dim=settings.embedding_dim)
