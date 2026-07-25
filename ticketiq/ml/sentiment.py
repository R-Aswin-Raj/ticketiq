"""Sentence-level sentiment scoring.

Uses ``vaderSentiment`` when it is installed (it ships its lexicon in the
wheel, so no network access is needed) and otherwise falls back to a small
built-in lexicon scorer with negation and intensifier handling. Both backends
return a compound score in ``[-1, 1]``, so downstream code is agnostic.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Protocol

from ticketiq.ml.text import tokenize

NEGATIONS = frozenset(
    {"not", "no", "never", "cannot", "cant", "can't", "wont", "won't", "dont", "don't",
     "isnt", "isn't", "arent", "aren't", "without", "hardly", "barely", "neither", "nor"}
)

INTENSIFIERS: dict[str, float] = {
    "very": 1.5, "extremely": 1.9, "really": 1.4, "so": 1.3, "totally": 1.5,
    "completely": 1.7, "absolutely": 1.8, "incredibly": 1.7, "super": 1.4,
    "quite": 1.2, "slightly": 0.6, "somewhat": 0.7, "a_bit": 0.7, "constantly": 1.6,
    "always": 1.4, "repeatedly": 1.5, "still": 1.2, "again": 1.2,
}

LEXICON: dict[str, float] = {
    # negative — service / product experience
    "broken": -2.4, "bug": -1.8, "buggy": -2.1, "crash": -2.6, "crashes": -2.6,
    "crashing": -2.7, "fails": -2.2, "failed": -2.2, "failing": -2.3, "failure": -2.4,
    "error": -1.9, "errors": -2.0, "slow": -2.0, "sluggish": -2.1, "lag": -1.9,
    "laggy": -2.1, "timeout": -2.1, "timeouts": -2.2, "unusable": -3.0, "useless": -3.0,
    "terrible": -3.0, "awful": -3.0, "horrible": -3.0, "worst": -3.2, "bad": -1.9,
    "poor": -2.0, "disappointed": -2.4, "disappointing": -2.4, "frustrated": -2.6,
    "frustrating": -2.6, "angry": -2.8, "annoyed": -2.2, "annoying": -2.2,
    "unacceptable": -3.1, "ridiculous": -2.6, "outrageous": -2.9, "overcharged": -2.7,
    "charged": -0.9, "double": -0.8, "unexpected": -1.3, "wrong": -2.0, "denied": -2.2,
    "blocked": -2.1, "locked": -1.9, "stuck": -2.0, "cannot": -1.6, "unable": -1.8,
    "missing": -1.8, "lost": -1.9, "downtime": -2.4, "outage": -2.7, "urgent": -1.4,
    "critical": -1.8, "escalate": -1.7, "refund": -1.0, "cancel": -1.6, "churn": -2.2,
    "ignored": -2.5, "waiting": -1.5, "delay": -1.8, "delayed": -1.9, "confusing": -1.7,
    "unclear": -1.4, "difficult": -1.6, "hard": -1.2, "expensive": -1.6, "pricey": -1.4,
    # positive
    "great": 2.4, "good": 1.8, "excellent": 2.8, "amazing": 2.9, "awesome": 2.8,
    "love": 2.6, "loved": 2.5, "helpful": 2.2, "fast": 1.8, "quick": 1.6,
    "responsive": 2.0, "smooth": 1.9, "reliable": 2.2, "stable": 1.9, "easy": 1.8,
    "intuitive": 2.0, "thanks": 1.2, "appreciate": 2.0, "happy": 2.4, "pleased": 2.2,
    "perfect": 2.7, "works": 1.4, "resolved": 2.1, "fixed": 1.9, "improved": 1.8,
    "recommend": 2.0, "nice": 1.6, "solid": 1.7,
}


class SentimentBackend(Protocol):
    def score(self, text: str) -> float: ...


class LexiconSentiment:
    """Small VADER-inspired scorer: valence + negation flip + intensifiers."""

    window = 3

    def score(self, text: str) -> float:
        tokens = tokenize(text, remove_stopwords=False)
        if not tokens:
            return 0.0
        total = 0.0
        for i, token in enumerate(tokens):
            valence = LEXICON.get(token)
            if valence is None:
                continue
            start = max(0, i - self.window)
            for j in range(start, i):
                prior = tokens[j]
                if prior in NEGATIONS:
                    valence *= -0.75
                elif prior in INTENSIFIERS:
                    boost = INTENSIFIERS[prior]
                    valence *= boost if valence > 0 else boost
            total += valence
        if total == 0.0:
            return 0.0
        # Same squashing shape VADER uses: x / sqrt(x^2 + alpha).
        return round(max(-1.0, min(1.0, total / math.sqrt(total * total + 15.0))), 4)


class VaderSentiment:
    """Adapter around ``vaderSentiment`` when the package is available."""

    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> float:
        return round(float(self._analyzer.polarity_scores(text)["compound"]), 4)


@lru_cache(maxsize=1)
def get_sentiment_backend() -> SentimentBackend:
    try:
        return VaderSentiment()
    except Exception:  # pragma: no cover - depends on optional install
        return LexiconSentiment()


def label_for(score: float, threshold: float = 0.15) -> str:
    if score <= -threshold:
        return "negative"
    if score >= threshold:
        return "positive"
    return "neutral"
