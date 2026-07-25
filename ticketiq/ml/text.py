"""Tokenisation utilities shared by the vectoriser and the aspect extractor."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are aren't as at be because been
    before being below between both but by can cannot could couldn't did didn't do does
    doesn't doing don't down during each few for from further had hadn't has hasn't have
    haven't having he her here hers herself him himself his how i if in into is isn't it
    its itself let's me more most my myself of off on once only or other ought our ours
    ourselves out over own same she should shouldn't so some such than that the their
    theirs them themselves then there these they this those through to too under until up
    very was wasn't we were weren't what when where which while who whom why with would
    wouldn't you your yours yourself yourselves hi hello hey thanks thank please regards
    """.split()
)


def normalise(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str, *, remove_stopwords: bool = True, min_len: int = 2) -> list[str]:
    """Split text into normalised word tokens."""
    tokens = _TOKEN_RE.findall(normalise(text))
    out = [t for t in tokens if len(t) >= min_len]
    if remove_stopwords:
        out = [t for t in out if t not in STOPWORDS]
    return out


def ngrams(tokens: Sequence[str], n: int) -> list[str]:
    """Return joined n-grams of order ``n``."""
    if n <= 1:
        return list(tokens)
    return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def featurize(text: str, *, ngram_range: tuple[int, int] = (1, 2)) -> list[str]:
    """Tokenise and expand into the requested n-gram range."""
    tokens = tokenize(text)
    features: list[str] = []
    lo, hi = ngram_range
    for n in range(lo, hi + 1):
        features.extend(ngrams(tokens, n))
    return features


def join_ticket(subject: str, body: str) -> str:
    """Subject carries more signal than body, so it is repeated once."""
    return f"{subject} {subject} {body}"


def unique(items: Iterable[str]) -> list[str]:
    """Order-preserving de-duplication."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
