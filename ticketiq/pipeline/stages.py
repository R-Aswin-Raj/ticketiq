"""The TicketIQ pipeline, expressed as a DAG of stages.

    classify ─┐
              ├─> urgency ─> select_config ─> retrieve ─> agent ─> respond
    aspects ──┘

``classify`` and ``aspects`` have no dependency on each other and therefore run
concurrently in level 0. ``select_config`` is where the bandit chooses the arm,
and it must sit after ``urgency`` because urgency is part of the RL state and
before ``retrieve`` because the arm decides top-K.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ticketiq.agent.react import AgentContext, ReActAgent
from ticketiq.llm.configs import get_arm
from ticketiq.llm.factory import client_for_arm, describe_model
from ticketiq.ml.aspects import extract_aspects
from ticketiq.ml.classifier import get_classifier
from ticketiq.ml.urgency import urgency_bucket, urgency_score
from ticketiq.rag.store import get_vector_store
from ticketiq.rl.bandit import state_key
from ticketiq.rl.service import get_bandit
from ticketiq.workflow.engine import Stage, StageContext, Workflow

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Stage implementations
# --------------------------------------------------------------------------
async def stage_classify(ctx: StageContext) -> dict[str, Any]:
    """TF-IDF + logistic-regression category prediction (CPU-bound, off the event loop)."""
    classifier = get_classifier()
    prediction = await asyncio.to_thread(
        classifier.predict, ctx.request["subject"], ctx.request["body"]
    )
    return {
        "category": prediction.category,
        "confidence": round(prediction.confidence, 4),
        "scores": {k: round(v, 4) for k, v in prediction.scores.items()},
    }


async def stage_aspects(ctx: StageContext) -> dict[str, Any]:
    """Aspect extraction and per-aspect sentiment."""
    aspects = await asyncio.to_thread(
        extract_aspects, ctx.request["subject"], ctx.request["body"]
    )
    return {
        "aspects": [
            {
                "aspect": a.aspect,
                "sentiment": a.sentiment,
                "label": a.label,
                "evidence": a.evidence,
            }
            for a in aspects
        ]
    }


def stage_urgency(ctx: StageContext) -> dict[str, Any]:
    """Blend category, aspect sentiment and tier into the RL state."""
    from ticketiq.ml.aspects import Aspect

    category = ctx.output_of("classify")["category"]
    raw = ctx.output_of("aspects")["aspects"]
    aspects = [
        Aspect(
            aspect=a["aspect"], sentiment=a["sentiment"], label=a["label"], evidence=a["evidence"]
        )
        for a in raw
    ]
    text = f"{ctx.request['subject']} {ctx.request['body']}"
    score = urgency_score(category, aspects, ctx.request["tier"], text)
    bucket = urgency_bucket(score)
    return {
        "urgency": score,
        "bucket": bucket,
        "state": state_key(category, bucket, ctx.request["tier"]),
    }


def stage_select_config(ctx: StageContext) -> dict[str, Any]:
    """Bandit action selection: which prompt variant and RAG top-K to use."""
    state = ctx.output_of("urgency")["state"]
    selection = get_bandit().select(state)
    arm = get_arm(selection.arm_id)
    return {
        "state": state,
        "arm_id": arm.id,
        "llm_variant": arm.variant_id,
        "top_k": arm.top_k,
        "model": describe_model(),
        "explored": selection.explored,
        "reason": selection.reason,
    }


async def stage_retrieve(ctx: StageContext) -> dict[str, Any]:
    """Top-K retrieval from the knowledge base, using the arm's K."""
    config = ctx.output_of("select_config")
    category = ctx.output_of("classify")["category"]
    aspects = [a["aspect"] for a in ctx.output_of("aspects")["aspects"]]
    # The category and aspect names are appended to the query: they bridge the
    # vocabulary gap when the customer never uses the KB's own terminology
    # ("money back" vs "refund").
    query = " ".join(
        [ctx.request["subject"], ctx.request["body"], category.replace("_", " "), *aspects]
    )
    store = get_vector_store()
    hits = await asyncio.to_thread(store.search, query, config["top_k"])
    ctx.scratch["hits"] = hits
    return {
        "query_terms": len(query.split()),
        "top_k": config["top_k"],
        "snippets": [
            {
                "doc_id": hit.chunk.doc_id,
                "chunk_id": hit.chunk.chunk_id,
                "score": hit.score,
                "text": hit.chunk.text,
            }
            for hit in hits
        ],
    }


async def stage_agent(ctx: StageContext) -> dict[str, Any]:
    """ReAct loop: decide, optionally call a tool, then draft the reply."""
    from ticketiq.rag.kb import Chunk
    from ticketiq.rag.store import SearchHit

    config = ctx.output_of("select_config")
    arm = get_arm(config["arm_id"])
    urgency = ctx.output_of("urgency")

    hits = ctx.scratch.get("hits")
    if hits is None:
        # Rehydrate from persisted state when this stage is re-run in isolation.
        hits = [
            SearchHit(
                chunk=Chunk(
                    doc_id=s["doc_id"],
                    chunk_id=s["chunk_id"],
                    title=s["doc_id"].replace("_", " ").title(),
                    heading=s["doc_id"].replace("_", " ").title(),
                    text=s["text"],
                ),
                score=s["score"],
            )
            for s in ctx.output_of("retrieve")["snippets"]
        ]

    agent_ctx = AgentContext(
        subject=ctx.request["subject"],
        body=ctx.request["body"],
        tier=ctx.request["tier"],
        category=ctx.output_of("classify")["category"],
        urgency=urgency["urgency"],
        urgency_bucket=urgency["bucket"],
        aspects=ctx.output_of("aspects")["aspects"],
        snippets=hits,
        customer_id=ctx.request.get("customer_id"),
        order_id=ctx.request.get("order_id"),
    )
    agent = ReActAgent(client=client_for_arm(arm.id), arm=arm)
    result = await agent.run(agent_ctx)
    return {
        "action": result.action,
        "reasoning": result.reasoning,
        "tool_calls": result.tool_calls,
        "escalation_reason": result.escalation_reason,
        "response_text": result.response_text,
    }


def stage_respond(ctx: StageContext) -> dict[str, Any]:
    """Assemble the customer-facing reply."""
    agent_output = ctx.output_of("agent")
    text = (agent_output.get("response_text") or "").strip()
    if not text:
        text = (
            "Thanks for contacting support. Your ticket has been received and a "
            "specialist will follow up on this thread."
        )
    if agent_output.get("action") == "escalate":
        text = f"{text}\n\n— This ticket has been escalated to a human specialist."
    return {"response_text": text, "action": agent_output.get("action")}


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------
def build_workflow() -> Workflow:
    return Workflow(
        name="ticket_triage",
        stages=[
            Stage("classify", stage_classify, (), "TF-IDF + logistic-regression category prediction"),
            Stage("aspects", stage_aspects, (), "Aspect extraction and per-aspect sentiment"),
            Stage("urgency", stage_urgency, ("classify", "aspects"), "Urgency score and RL state"),
            Stage("select_config", stage_select_config, ("urgency",), "Bandit arm selection"),
            Stage("retrieve", stage_retrieve, ("select_config",), "Top-K knowledge retrieval"),
            Stage("agent", stage_agent, ("retrieve",), "ReAct decision, tools and drafting"),
            Stage("respond", stage_respond, ("agent",), "Final response assembly"),
        ],
    )


_workflow: Workflow | None = None


def get_workflow() -> Workflow:
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow
