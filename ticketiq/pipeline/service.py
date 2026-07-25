"""Application service layer sitting between the API and the workflow engine."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from ticketiq.pipeline.stages import get_workflow
from ticketiq.rl.bandit import compute_reward
from ticketiq.rl.service import get_bandit, persist_bandit
from ticketiq.schemas import (
    AgentTrace,
    AspectSentiment,
    ClassificationResult,
    FeedbackResponse,
    KnowledgeSnippet,
    PipelineConfig,
    StageStatus,
    TicketRequest,
    TicketResponse,
    TicketStatusResponse,
)
from ticketiq.workflow.state import StateStore, get_state_store

logger = logging.getLogger(__name__)


class UnknownTransaction(KeyError):
    """Raised when a transaction id has no recorded state."""


def new_transaction_id() -> str:
    return f"txn_{uuid.uuid4().hex[:16]}"


async def process_ticket(
    request: TicketRequest,
    *,
    transaction_id: str | None = None,
    store: StateStore | None = None,
) -> TicketResponse:
    """Run the full DAG for one ticket and assemble the API response."""
    store = store or get_state_store()
    txn = transaction_id or new_transaction_id()
    payload: dict[str, Any] = {
        "subject": request.subject,
        "body": request.body,
        "tier": request.tier.value,
        "customer_id": request.customer_id,
        "order_id": request.order_id,
    }

    started = time.perf_counter()
    ctx = await get_workflow().run(txn, payload, store)
    latency = round(time.perf_counter() - started, 4)

    classify = ctx.outputs["classify"]
    aspects = ctx.outputs["aspects"]["aspects"]
    urgency = ctx.outputs["urgency"]
    config = ctx.outputs["select_config"]
    retrieve = ctx.outputs["retrieve"]
    agent = ctx.outputs["agent"]
    respond = ctx.outputs["respond"]

    # The bandit is only updated on feedback, but the decision and the measured
    # latency must be durable now — the reward is not computable without them.
    store.record_decision(txn, config["state"], config["arm_id"], latency)

    return TicketResponse(
        transaction_id=txn,
        classification=ClassificationResult(
            category=classify["category"],
            confidence=classify["confidence"],
            scores=classify["scores"],
        ),
        aspects=[AspectSentiment(**a) for a in aspects],
        urgency=urgency["urgency"],
        snippets=[KnowledgeSnippet(**s) for s in retrieve["snippets"]],
        agent=AgentTrace(
            action=agent["action"],
            reasoning=agent["reasoning"],
            tool_calls=agent["tool_calls"],
            escalation_reason=agent["escalation_reason"],
        ),
        response_text=respond["response_text"],
        config=PipelineConfig(
            arm_id=config["arm_id"],
            llm_variant=config["llm_variant"],
            model=config["model"],
            top_k=config["top_k"],
        ),
        latency_s=latency,
    )


def apply_feedback(
    transaction_id: str, score: int, *, store: StateStore | None = None
) -> FeedbackResponse:
    """Turn user feedback into a reward and push it into the bandit."""
    store = store or get_state_store()
    decision = store.get_decision(transaction_id)
    if decision is None:
        raise UnknownTransaction(transaction_id)

    reward = compute_reward(score, decision.latency_s)
    stats = get_bandit().update(decision.state, decision.arm_id, reward)
    store.record_feedback(transaction_id, score, reward)
    persist_bandit()
    logger.info(
        "feedback txn=%s state=%s arm=%s reward=%.3f pulls=%d",
        transaction_id, decision.state, decision.arm_id, reward, stats.pulls,
    )
    return FeedbackResponse(
        transaction_id=transaction_id,
        arm_id=decision.arm_id,
        reward=reward,
        pulls=stats.pulls,
        mean_reward=round(stats.mean_reward, 4),
    )


def get_status(
    transaction_id: str, *, store: StateStore | None = None
) -> TicketStatusResponse:
    """Report real per-stage state from the workflow store."""
    store = store or get_state_store()
    run = store.get_run(transaction_id)
    if run is None:
        raise UnknownTransaction(transaction_id)

    records = {r.stage: r for r in store.get_stages(transaction_id)}
    workflow = get_workflow()
    stages: list[StageStatus] = []
    for name in workflow.order():
        record = records.get(name)
        if record is None:
            stages.append(StageStatus(name=name, status="pending"))
            continue
        stages.append(
            StageStatus(
                name=name,
                status=record.status,  # type: ignore[arg-type]
                started_at=record.started_at,
                finished_at=record.finished_at,
                duration_s=record.duration_s,
                error=record.error,
                output=record.output,
            )
        )
    return TicketStatusResponse(
        transaction_id=transaction_id,
        pipeline_status=run["status"],
        stages=stages,
    )


async def rerun_stage(
    transaction_id: str, stage: str, *, store: StateStore | None = None
) -> TicketStatusResponse:
    """Re-run one stage, replaying completed upstream stages from the store."""
    store = store or get_state_store()
    run = store.get_run(transaction_id)
    if run is None:
        raise UnknownTransaction(transaction_id)
    workflow = get_workflow()
    if stage not in workflow.stages:
        raise KeyError(f"unknown stage {stage!r}")
    downstream = [
        name
        for name in workflow.order()[workflow.order().index(stage) :]
    ]
    await workflow.run(transaction_id, run["request"], store, resume=True, only=downstream)
    return get_status(transaction_id, store=store)
