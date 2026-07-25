"""Shared fixtures.

Every test gets its own data directory, SQLite file and singleton state. The
singletons are process-wide by design (they cache an expensive model and index),
so they must be reset explicitly or tests leak into each other.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_DATA = PROJECT_ROOT / "data"


def _reset_singletons() -> None:
    from ticketiq.config import get_settings
    from ticketiq.llm.factory import reset_llm_cache
    from ticketiq.ml.classifier import reset_classifier_cache
    from ticketiq.rag.store import reset_vector_store
    from ticketiq.rl.service import reset_bandit
    from ticketiq.workflow.state import reset_state_store

    get_settings.cache_clear()
    reset_classifier_cache()
    reset_vector_store()
    reset_bandit()
    reset_state_store()
    reset_llm_cache()


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the app at a throwaway data directory seeded from the repo."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copytree(REPO_DATA / "kb", data_dir / "kb")
    shutil.copy(REPO_DATA / "tickets.jsonl", data_dir / "tickets.jsonl")

    monkeypatch.setenv("LLM_MODE", "mock")
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("KB_DIR", str(data_dir / "kb"))
    monkeypatch.setenv("DATASET_PATH", str(data_dir / "tickets.jsonl"))
    monkeypatch.setenv("STATE_DB_PATH", str(data_dir / "state.db"))
    monkeypatch.setenv("MODEL_PATH", str(data_dir / "classifier.json"))
    monkeypatch.setenv("BANDIT_EPSILON", "0.1")

    _reset_singletons()
    yield data_dir
    _reset_singletons()


@pytest.fixture
def state_store(isolated_env: Path):
    from ticketiq.workflow.state import StateStore

    store = StateStore(isolated_env / "test.db")
    yield store
    store.close()


@pytest.fixture
def client(isolated_env: Path):
    from fastapi.testclient import TestClient
    from ticketiq.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_tickets() -> list[dict[str, str]]:
    return [
        {
            "subject": "Double charged this month",
            "body": "We were billed twice on invoice INV-1201 and want the duplicate refunded.",
            "tier": "pro",
        },
        {
            "subject": "SSO login broken",
            "body": "Our SAML sign in fails for every user since this morning.",
            "tier": "enterprise",
        },
        {
            "subject": "Please add dark mode",
            "body": "It would be great if the product had a dark theme for late shifts.",
            "tier": "free",
        },
    ]
