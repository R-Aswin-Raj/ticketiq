"""LLM client construction.

``LLM_MODE`` decides the backend once, at startup:

===========  ==========================================================
``mock``     :class:`~ticketiq.llm.mock.MockLLM` — offline, deterministic
``local``    :class:`~ticketiq.llm.remote.OllamaLLM`
``cloud``    :class:`~ticketiq.llm.remote.CloudLLM`
===========  ==========================================================

Clients are cached per prompt variant because the mock needs per-variant
latency characteristics and the HTTP clients are cheap but not free.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from ticketiq.config import get_settings
from ticketiq.llm.base import LLMClient
from ticketiq.llm.configs import PromptVariant, get_arm
from ticketiq.llm.mock import MockLLM
from ticketiq.llm.remote import CloudLLM, OllamaLLM

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def get_llm_client(variant_id: str) -> LLMClient:
    """Return the client for a prompt variant, honouring ``LLM_MODE``."""
    from ticketiq.llm.configs import VARIANTS

    variant: PromptVariant = VARIANTS[variant_id]
    settings = get_settings()

    if settings.llm_mode == "local":
        return OllamaLLM(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            timeout_s=settings.llm_timeout_s,
        )
    if settings.llm_mode == "cloud":
        return CloudLLM(
            model=settings.cloud_model,
            base_url=settings.cloud_base_url,
            api_key=settings.cloud_api_key,
            timeout_s=settings.llm_timeout_s,
        )
    return MockLLM(
        name=f"mock:{variant_id}",
        base_latency_s=variant.mock_base_latency_s,
        seed=abs(hash(variant_id)) % 10_000,
    )


def client_for_arm(arm_id: str) -> LLMClient:
    return get_llm_client(get_arm(arm_id).variant_id)


def describe_model() -> str:
    settings = get_settings()
    if settings.llm_mode == "local":
        return f"ollama:{settings.ollama_model}"
    if settings.llm_mode == "cloud":
        return f"cloud:{settings.cloud_model}"
    return "mock"


def reset_llm_cache() -> None:
    """Test hook."""
    get_llm_client.cache_clear()
