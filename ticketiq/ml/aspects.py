"""Aspect-based sentiment analysis.

Aspects are spotted with a curated keyword lexicon, then sentiment is scored
**only over the sentences that mention the aspect** — that is what makes the
result aspect-level rather than a single document polarity. A ticket saying
"support was great but the app keeps crashing" yields a positive ``support``
aspect and a negative ``performance`` aspect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ticketiq.ml.sentiment import get_sentiment_backend, label_for

_SENTENCE_RE = re.compile(r"(?<=[.!?;\n])\s+|\n+")
# Contrastive conjunctions flip polarity mid-sentence ("support was great but
# the app crashes"), so clauses are scored separately before attribution.
_CLAUSE_RE = re.compile(r"\s+(?:but|however|although|though|whereas|yet)\s+", re.IGNORECASE)

ASPECT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "billing": (
        "bill", "billed", "billing", "invoice", "charge", "charged", "charges", "payment",
        "card", "refund", "price", "pricing", "cost", "subscription", "plan", "renewal",
        "receipt", "vat", "tax", "overcharge", "double charged",
    ),
    "login": (
        "login", "log in", "logged out", "sign in", "signin", "password", "sso", "saml",
        "mfa", "2fa", "otp", "authenticate", "authentication", "credentials", "locked out",
        "access", "reset link",
    ),
    "performance": (
        "slow", "latency", "lag", "laggy", "timeout", "timed out", "load", "loading",
        "spinner", "hang", "hangs", "freeze", "frozen", "crash", "crashes", "crashing",
        "downtime", "outage", "throughput", "performance", "speed",
    ),
    "support response time": (
        "support", "ticket", "agent", "reply", "response", "responded", "waiting",
        "no answer", "chased", "follow up", "sla", "escalation", "escalated",
    ),
    "data & reporting": (
        "report", "reports", "dashboard", "export", "csv", "analytics", "chart",
        "metrics", "data", "sync", "synced", "integration", "webhook", "api",
    ),
    "account management": (
        "account", "seat", "seats", "user", "users", "permission", "permissions", "role",
        "roles", "team", "workspace", "admin", "invite", "deactivate", "delete account",
    ),
    "feature availability": (
        "feature", "roadmap", "request", "support for", "would be great", "wish",
        "missing", "add", "ability to", "option to", "dark mode", "mobile app",
    ),
    "documentation": (
        "docs", "documentation", "guide", "tutorial", "knowledge base", "article",
        "instructions", "readme",
    ),
}


@dataclass(frozen=True)
class Aspect:
    aspect: str
    sentiment: float
    label: str
    evidence: str


def split_sentences(text: str) -> list[str]:
    """Split into sentences, then further into contrastive clauses."""
    sentences = [p.strip() for p in _SENTENCE_RE.split(text) if p and p.strip()]
    parts: list[str] = []
    for sentence in sentences:
        parts.extend(c.strip() for c in _CLAUSE_RE.split(sentence) if c and c.strip())
    return parts or ([text.strip()] if text.strip() else [])


def _mentions(sentence_lc: str, keywords: tuple[str, ...]) -> str | None:
    for kw in keywords:
        if " " in kw:
            if kw in sentence_lc:
                return kw
        elif re.search(rf"\b{re.escape(kw)}\b", sentence_lc):
            return kw
    return None


def extract_aspects(
    subject: str,
    body: str,
    *,
    max_aspects: int = 3,
    category_hint: str | None = None,
) -> list[Aspect]:
    """Return 1–3 aspects with independent sentiment scores.

    Ranking prefers aspects with strong (mostly negative) sentiment and more
    mentions, since those are what should drive urgency and escalation.
    """
    backend = get_sentiment_backend()
    text = f"{subject}. {body}"
    sentences = split_sentences(text)

    hits: dict[str, list[str]] = {}
    for sentence in sentences:
        lc = sentence.lower()
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if _mentions(lc, keywords):
                hits.setdefault(aspect, []).append(sentence)

    if not hits:
        fallback = category_hint if category_hint in ASPECT_KEYWORDS else "general"
        score = backend.score(text)
        return [
            Aspect(
                aspect=fallback,
                sentiment=score,
                label=label_for(score),
                evidence=sentences[0][:200] if sentences else "",
            )
        ]

    scored: list[Aspect] = []
    for aspect, aspect_sentences in hits.items():
        joined = " ".join(aspect_sentences)
        score = backend.score(joined)
        scored.append(
            Aspect(
                aspect=aspect,
                sentiment=score,
                label=label_for(score),
                evidence=aspect_sentences[0][:200],
            )
        )

    def rank(a: Aspect) -> tuple[float, int, str]:
        boost = 0.35 if a.aspect == category_hint else 0.0
        return (-(abs(a.sentiment) + boost), -len(hits[a.aspect]), a.aspect)

    scored.sort(key=rank)
    return scored[:max_aspects]
