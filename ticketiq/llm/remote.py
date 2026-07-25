"""HTTP-backed LLM clients: Ollama (local) and an OpenAI-compatible API (cloud).

Both are thin on purpose. Retries are a single attempt with a short backoff:
under a bandit that is measuring latency, aggressive retrying would corrupt the
reward signal more than it would improve availability.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ticketiq.llm.base import LLMError, LLMResult

logger = logging.getLogger(__name__)


def _require_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise LLMError("httpx is required for local/cloud LLM modes") from exc
    return httpx


class OllamaLLM:
    """Client for the Ollama ``/api/chat`` endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 30.0,
    ) -> None:
        self.name = f"ollama:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> LLMResult:
        httpx = _require_httpx()
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            for attempt in range(2):
                try:
                    response = await client.post(f"{self.base_url}/api/chat", json=payload)
                    response.raise_for_status()
                    data = response.json()
                    break
                except Exception as exc:  # noqa: BLE001 - surfaced as LLMError below
                    if attempt == 1:
                        raise LLMError(f"ollama request failed: {exc}") from exc
                    logger.warning("ollama attempt %s failed: %s", attempt + 1, exc)
                    await asyncio.sleep(0.25)
        text = str(data.get("message", {}).get("content", "")).strip()
        if not text:
            raise LLMError("ollama returned an empty completion")
        return LLMResult(
            text=text,
            model=self.name,
            latency_s=round(time.perf_counter() - started, 4),
            usage={
                "prompt_tokens": int(data.get("prompt_eval_count", 0)),
                "completion_tokens": int(data.get("eval_count", 0)),
            },
        )


class CloudLLM:
    """Client for an OpenAI-compatible ``/chat/completions`` endpoint.

    Anthropic's API is also supported by pointing ``CLOUD_BASE_URL`` at a
    compatibility gateway, or by subclassing and overriding :meth:`_request`.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        timeout_s: float = 30.0,
    ) -> None:
        self.name = f"cloud:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    async def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> LLMResult:
        if not self.api_key:
            raise LLMError("CLOUD_API_KEY is not set")
        httpx = _require_httpx()
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
                response.raise_for_status()
                data = response.json()
            except Exception as exc:  # noqa: BLE001
                raise LLMError(f"cloud request failed: {exc}") from exc
        try:
            text = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as exc:
            raise LLMError(f"unexpected cloud response shape: {data}") from exc
        usage = data.get("usage", {}) or {}
        return LLMResult(
            text=text,
            model=self.name,
            latency_s=round(time.perf_counter() - started, 4),
            usage={
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            },
        )
