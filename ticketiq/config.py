"""Runtime configuration.

Everything is driven from environment variables (optionally loaded from a
``.env`` file). The single most important knob is ``LLM_MODE``:

* ``mock``  — deterministic offline stub. Used by tests and CI.
* ``local`` — Ollama HTTP API on ``OLLAMA_BASE_URL``.
* ``cloud`` — an OpenAI-compatible / Anthropic HTTP API.

The rest of the system never branches on the mode; it asks
:func:`ticketiq.llm.factory.get_llm_client` for a client and programs against
the :class:`ticketiq.llm.base.LLMClient` protocol.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

LLMMode = Literal["mock", "local", "cloud"]

# ticketiq/config.py lives one level under the repo root now (flat layout),
# so the root is parents[1]. Every default data path hangs off this.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (avoids a hard dependency on python-dotenv)."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # --- LLM -----------------------------------------------------------
    llm_mode: LLMMode = "mock"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    cloud_base_url: str = "https://api.openai.com/v1"
    cloud_model: str = "gpt-4o-mini"
    cloud_api_key: str = ""
    llm_timeout_s: float = 30.0

    # --- Storage -------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    state_db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "ticketiq.db")
    kb_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "kb")
    dataset_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "tickets.jsonl")
    model_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "classifier.json")

    # --- RL ------------------------------------------------------------
    bandit_epsilon: float = 0.15
    bandit_optimistic_init: float = 0.0

    # --- RAG -----------------------------------------------------------
    embedding_dim: int = 256
    chunk_size_chars: int = 480
    chunk_overlap_chars: int = 80

    @property
    def is_offline(self) -> bool:
        return self.llm_mode == "mock"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process. Cache is clearable in tests."""
    _load_dotenv(PROJECT_ROOT / ".env")
    mode = _env("LLM_MODE", "mock").lower()
    if mode not in ("mock", "local", "cloud"):
        raise ValueError(f"LLM_MODE must be mock|local|cloud, got {mode!r}")
    data_dir = Path(_env("DATA_DIR", str(PROJECT_ROOT / "data")))
    return Settings(
        llm_mode=mode,  # type: ignore[arg-type]
        ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=_env("OLLAMA_MODEL", "llama3.2"),
        cloud_base_url=_env("CLOUD_BASE_URL", "https://api.openai.com/v1"),
        cloud_model=_env("CLOUD_MODEL", "gpt-4o-mini"),
        cloud_api_key=_env("CLOUD_API_KEY", ""),
        llm_timeout_s=_env_float("LLM_TIMEOUT_S", 30.0),
        data_dir=data_dir,
        state_db_path=Path(_env("STATE_DB_PATH", str(data_dir / "ticketiq.db"))),
        kb_dir=Path(_env("KB_DIR", str(data_dir / "kb"))),
        dataset_path=Path(_env("DATASET_PATH", str(data_dir / "tickets.jsonl"))),
        model_path=Path(_env("MODEL_PATH", str(data_dir / "classifier.json"))),
        bandit_epsilon=_env_float("BANDIT_EPSILON", 0.15),
        bandit_optimistic_init=_env_float("BANDIT_OPTIMISTIC_INIT", 0.0),
        embedding_dim=_env_int("EMBEDDING_DIM", 256),
        chunk_size_chars=_env_int("CHUNK_SIZE_CHARS", 480),
        chunk_overlap_chars=_env_int("CHUNK_OVERLAP_CHARS", 80),
    )
