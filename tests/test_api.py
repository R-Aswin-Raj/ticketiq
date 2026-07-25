"""API and end-to-end tests.

All of these run with ``LLM_MODE=mock`` (set in ``conftest``), so no network
call is ever made. ``test_end_to_end_with_patched_llm`` goes further and
injects an explicit fake client, asserting the pipeline works against a
completely arbitrary LLM implementation.
"""

from __future__ import annotations

import asyncio

import pytest

from ticketiq.llm.base import LLMResult


# --------------------------------------------------------------------------
# Health and introspection
# --------------------------------------------------------------------------
def test_healthz(client) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["llm_mode"] == "mock"


def test_pipeline_graph_exposes_the_parallel_level(client) -> None:
    body = client.get("/pipeline/graph").json()
    level_zero = body["levels"][0]
    assert level_zero["parallel"] is True
    assert {s["name"] for s in level_zero["stages"]} == {"classify", "aspects"}
    assert body["order"][-1] == "respond"


# --------------------------------------------------------------------------
# POST /ticket
# --------------------------------------------------------------------------
def test_post_ticket_returns_every_required_field(client, sample_tickets) -> None:
    response = client.post("/ticket", json=sample_tickets[0])
    assert response.status_code == 200
    body = response.json()

    assert body["transaction_id"].startswith("txn_")
    assert body["classification"]["category"] in (
        "billing", "technical", "account", "feature_request"
    )
    assert 0.0 <= body["classification"]["confidence"] <= 1.0
    assert 1 <= len(body["aspects"]) <= 3
    assert 0.0 <= body["urgency"] <= 1.0
    assert body["snippets"]
    assert body["agent"]["action"] in ("answer_directly", "tool_call", "escalate")
    assert body["response_text"]
    assert body["config"]["arm_id"]
    assert body["config"]["top_k"] in (2, 5)
    assert body["latency_s"] > 0


def test_billing_ticket_is_classified_as_billing(client, sample_tickets) -> None:
    body = client.post("/ticket", json=sample_tickets[0]).json()
    assert body["classification"]["category"] == "billing"


def test_retrieved_snippet_count_matches_the_chosen_top_k(client, sample_tickets) -> None:
    body = client.post("/ticket", json=sample_tickets[1]).json()
    assert len(body["snippets"]) == body["config"]["top_k"]


def test_enterprise_outage_is_escalated(client) -> None:
    body = client.post(
        "/ticket",
        json={
            "subject": "Production is down",
            "body": "Production is down for all our users and we have lost data.",
            "tier": "enterprise",
        },
    ).json()
    assert body["agent"]["action"] == "escalate"
    assert body["agent"]["escalation_reason"]
    assert body["urgency"] > 0.6


def test_agent_trace_is_inspectable(client, sample_tickets) -> None:
    body = client.post("/ticket", json=sample_tickets[0]).json()
    assert body["agent"]["reasoning"]
    for call in body["agent"]["tool_calls"]:
        assert call["tool"] and "result" in call


def test_transaction_ids_are_unique(client, sample_tickets) -> None:
    ids = {client.post("/ticket", json=sample_tickets[0]).json()["transaction_id"] for _ in range(3)}
    assert len(ids) == 3


@pytest.mark.parametrize(
    "payload",
    [
        {"subject": "", "body": "text", "tier": "free"},
        {"subject": "text", "body": "", "tier": "free"},
        {"subject": "text", "body": "text", "tier": "platinum"},
        {"body": "missing subject", "tier": "free"},
    ],
)
def test_invalid_payloads_are_rejected(client, payload) -> None:
    assert client.post("/ticket", json=payload).status_code == 422


# --------------------------------------------------------------------------
# GET /ticket/{id}/status
# --------------------------------------------------------------------------
def test_status_reflects_real_stage_state(client, sample_tickets) -> None:
    txn = client.post("/ticket", json=sample_tickets[0]).json()["transaction_id"]
    body = client.get(f"/ticket/{txn}/status").json()

    assert body["pipeline_status"] == "completed"
    names = [s["name"] for s in body["stages"]]
    assert names == ["classify", "aspects", "urgency", "select_config", "retrieve", "agent", "respond"]
    assert all(s["status"] == "completed" for s in body["stages"])
    # Not a hardcoded response: each stage carries its own persisted output.
    classify = next(s for s in body["stages"] if s["name"] == "classify")
    assert classify["output"]["category"]
    assert classify["duration_s"] >= 0


def test_status_404s_for_an_unknown_transaction(client) -> None:
    assert client.get("/ticket/txn_nope/status").status_code == 404


def test_rerunning_a_stage_keeps_upstream_state(client, sample_tickets) -> None:
    txn = client.post("/ticket", json=sample_tickets[1]).json()["transaction_id"]
    before = client.get(f"/ticket/{txn}/status").json()
    classify_before = next(s for s in before["stages"] if s["name"] == "classify")

    response = client.post(f"/ticket/{txn}/rerun/agent")
    assert response.status_code == 200
    after = response.json()
    classify_after = next(s for s in after["stages"] if s["name"] == "classify")
    agent_after = next(s for s in after["stages"] if s["name"] == "agent")

    assert classify_after["finished_at"] == classify_before["finished_at"]
    assert agent_after["finished_at"] > classify_before["finished_at"]


def test_rerunning_an_unknown_stage_is_a_400(client, sample_tickets) -> None:
    txn = client.post("/ticket", json=sample_tickets[0]).json()["transaction_id"]
    assert client.post(f"/ticket/{txn}/rerun/nonsense").status_code == 400


# --------------------------------------------------------------------------
# POST /feedback
# --------------------------------------------------------------------------
def test_feedback_updates_the_bandit(client, sample_tickets) -> None:
    posted = client.post("/ticket", json=sample_tickets[0]).json()
    txn, latency, arm = posted["transaction_id"], posted["latency_s"], posted["config"]["arm_id"]

    response = client.post("/feedback", json={"transaction_id": txn, "score": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["arm_id"] == arm
    assert body["reward"] == pytest.approx(10.0 - latency, abs=1e-3)
    assert body["pulls"] == 1
    assert body["mean_reward"] == pytest.approx(body["reward"], abs=1e-3)


def test_negative_feedback_produces_a_negative_reward(client, sample_tickets) -> None:
    txn = client.post("/ticket", json=sample_tickets[0]).json()["transaction_id"]
    body = client.post("/feedback", json={"transaction_id": txn, "score": 0}).json()
    assert body["reward"] < 0


def test_feedback_is_visible_in_rl_stats(client, sample_tickets) -> None:
    posted = client.post("/ticket", json=sample_tickets[0]).json()
    client.post("/feedback", json={"transaction_id": posted["transaction_id"], "score": 1})

    stats = client.get("/rl/stats").json()
    assert stats["table"]["total_pulls"] == 1
    assert sum(stats["action_distribution"].values()) == pytest.approx(1.0)


def test_feedback_404s_for_an_unknown_transaction(client) -> None:
    response = client.post("/feedback", json={"transaction_id": "txn_nope", "score": 1})
    assert response.status_code == 404


def test_feedback_rejects_a_non_binary_score(client, sample_tickets) -> None:
    txn = client.post("/ticket", json=sample_tickets[0]).json()["transaction_id"]
    assert client.post("/feedback", json={"transaction_id": txn, "score": 5}).status_code == 422


def test_repeated_feedback_averages_over_pulls(client, sample_tickets) -> None:
    first = client.post("/ticket", json=sample_tickets[0]).json()
    client.post("/feedback", json={"transaction_id": first["transaction_id"], "score": 1})
    second = client.post("/ticket", json=sample_tickets[0]).json()
    body = client.post(
        "/feedback", json={"transaction_id": second["transaction_id"], "score": 0}
    ).json()
    if body["pulls"] == 2:  # same arm was chosen twice
        assert body["mean_reward"] < 10.0


# --------------------------------------------------------------------------
# End to end with an injected fake LLM
# --------------------------------------------------------------------------
def test_end_to_end_with_patched_llm(isolated_env, monkeypatch, sample_tickets) -> None:
    """Full pipeline against a fake client, asserting no real backend is used."""
    calls: list[str] = []

    class FakeLLM:
        name = "fake"

        async def complete(self, prompt: str, *, system: str = "", **kwargs) -> LLMResult:
            calls.append(prompt[:40])
            if "Respond with JSON only" in prompt:
                return LLMResult(
                    text='{"thought": "context is enough", "action": "answer_directly"}',
                    model="fake",
                    latency_s=0.01,
                )
            return LLMResult(text="Here is your answer.", model="fake", latency_s=0.01)

    monkeypatch.setattr(
        "ticketiq.pipeline.stages.client_for_arm", lambda arm_id: FakeLLM()
    )

    from ticketiq.pipeline.service import apply_feedback, get_status, process_ticket
    from ticketiq.schemas import TicketRequest

    response = asyncio.run(process_ticket(TicketRequest(**sample_tickets[2])))

    assert len(calls) == 2  # one decision call, one answer call
    assert response.response_text == "Here is your answer."
    assert response.agent.action == "answer_directly"
    assert response.classification.category == "feature_request"

    status = get_status(response.transaction_id)
    assert status.pipeline_status == "completed"
    assert len(status.stages) == 7

    feedback = apply_feedback(response.transaction_id, 1)
    assert feedback.reward == pytest.approx(10.0 - response.latency_s, abs=1e-3)


def test_pipeline_surfaces_a_stage_failure(isolated_env, monkeypatch, sample_tickets) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("classifier exploded")

    monkeypatch.setattr("ticketiq.pipeline.stages.get_classifier", boom)

    from ticketiq.pipeline.service import get_status, process_ticket
    from ticketiq.schemas import TicketRequest
    from ticketiq.workflow.engine import WorkflowError

    request = TicketRequest(**sample_tickets[0])
    with pytest.raises(WorkflowError):
        asyncio.run(process_ticket(request, transaction_id="txn_boom"))

    status = get_status("txn_boom")
    assert status.pipeline_status == "failed"
    failed = [s for s in status.stages if s.status == "failed"]
    assert failed and "classifier exploded" in (failed[0].error or "")
