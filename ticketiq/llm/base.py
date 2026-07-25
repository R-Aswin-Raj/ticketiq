"""LLM client abstraction.

Every backend implements :class:`LLMClient`. Nothing outside this package
knows whether it is talking to Ollama, a hosted API, or the offline stub.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    latency_s: float
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal async completion interface."""

    name: str

    async def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> LLMResult: ...


class LLMError(RuntimeError):
    """Raised when a backend fails and no usable completion was produced."""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _loads_dict(candidate: str) -> dict[str, Any] | None:
    """json.loads that returns a dict or None, never raises."""
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Small local models routinely wrap JSON in prose or code fences, so this is
    deliberately forgiving. A near-miss (trailing comma, single-quoted keys) is
    repaired and re-parsed; each repair is only accepted if it now parses, so a
    bad guess can never corrupt a valid decision. Callers must still handle an
    empty dict.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?|```$", "", cleaned).strip()

    parsed = _loads_dict(cleaned)
    if parsed is not None:
        return parsed

    match = _JSON_BLOCK_RE.search(cleaned)
    if match is None:
        return {}
    candidate = match.group(0)

    for attempt in (
        candidate,
        _TRAILING_COMMA_RE.sub(r"\1", candidate),
        _TRAILING_COMMA_RE.sub(r"\1", candidate).replace("'", '"'),
    ):
        # ponytail: covers trailing commas and single quotes, the two malformations
        # small models emit most. Python literals (True/None) not handled — add a
        # token-level pass only if logs show them.
        parsed = _loads_dict(attempt)
        if parsed is not None:
            return parsed
    return {}
