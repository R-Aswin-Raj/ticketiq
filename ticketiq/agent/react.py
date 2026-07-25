"""A small ReAct-style reasoning loop.

Flow per ticket:

1. **Guardrail** — a deterministic policy check runs *before* the model. Some
   escalations (data loss, legal, security, enterprise outage) must never
   depend on an LLM's judgement, so they short-circuit the loop. The rest is
   left to the model.
2. **Thought / Action** — one completion returning JSON: ``answer_directly``,
   ``tool_call``, or ``escalate``.
3. **Observation** — the tool runs and its output is appended to the context.
4. **Answer** — a second completion writes the customer-facing reply using the
   retrieved context plus any observation.

The loop is capped at ``max_steps`` tool calls (default 1). Unbounded ReAct
loops are a latency and cost risk, and the reward function penalises latency
directly, so the cap is a design decision rather than a shortcut.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ticketiq.agent.tools import describe_tools, run_tool
from ticketiq.llm.base import LLMClient, LLMError, extract_json
from ticketiq.llm.configs import Arm
from ticketiq.rag.store import SearchHit

logger = logging.getLogger(__name__)

VALID_ACTIONS = ("answer_directly", "tool_call", "escalate")

HARD_ESCALATION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bdata loss\b|\blost all (our|my) data\b", "Possible data loss reported."),
    (r"\b(lawyer|legal action|regulator|gdpr complaint|chargeback)\b", "Legal or regulatory exposure."),
    (r"\bsecurity (incident|breach)\b|\bbreach\b|\bunauthori[sz]ed access\b", "Possible security incident."),
    (r"\bproduction (is )?down\b|\bcomplete outage\b|\ball users (are )?(down|blocked)\b", "Full outage reported."),
)


@dataclass
class AgentContext:
    subject: str
    body: str
    tier: str
    category: str
    urgency: float
    urgency_bucket: str
    aspects: list[dict[str, Any]]
    snippets: list[SearchHit]
    customer_id: str | None = None
    order_id: str | None = None

    @property
    def full_text(self) -> str:
        return f"{self.subject}\n{self.body}"

    def render_context(self) -> str:
        if not self.snippets:
            return "(no knowledge-base context retrieved)"
        return "\n\n".join(
            f"[{hit.chunk.chunk_id}] {hit.chunk.title} — {hit.chunk.heading}\n{hit.chunk.text}"
            for hit in self.snippets
        )

    def render_summary(self) -> str:
        aspects = ", ".join(
            f"{a['aspect']}={a['sentiment']:+.2f}" for a in self.aspects
        ) or "none detected"
        return (
            f"Subject: {self.subject}\n"
            f"Body: {self.body}\n"
            f"Customer tier: {self.tier}\n"
            f"Predicted category: {self.category}\n"
            f"Urgency: {self.urgency_bucket} ({self.urgency:.2f})\n"
            f"Aspect sentiment: {aspects}"
        )


@dataclass
class AgentResult:
    action: str
    reasoning: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    escalation_reason: str | None = None
    response_text: str = ""


def hard_escalation_reason(text: str) -> str | None:
    """Non-negotiable escalation triggers, checked before the model runs."""
    lc = text.lower()
    for pattern, reason in HARD_ESCALATION_PATTERNS:
        if re.search(pattern, lc):
            return reason
    return None


DECISION_TEMPLATE = """You are triaging a support ticket. Decide the next action.

TICKET
{summary}

RETRIEVED KNOWLEDGE BASE CONTEXT
{context}

AVAILABLE TOOLS
{tools}

Choose exactly one action:
- "answer_directly": the context above is sufficient to resolve the ticket.
- "tool_call": you need live account or order data first.
- "escalate": policy requires a human (outage, data loss, legal, security,
  enterprise blocker, refund over $500, or a stated intent to cancel).

Respond with JSON only, no prose, in this shape:
{{"thought": "<one sentence of reasoning>", "action": "<one of the three>",
 "tool": "<tool name or null>", "arguments": {{}}, "escalation_reason": "<or null>"}}
"""

ANSWER_TEMPLATE = """Write the reply to send to the customer.

TICKET
{summary}

RETRIEVED KNOWLEDGE BASE CONTEXT
{context}
{observation}
Rules: ground every factual claim in the context above; do not invent policy,
prices, or dates. If the context does not answer the question, say it is being
escalated to a specialist.
"""

ESCALATION_TEMPLATE = """This ticket is being escalated to a human specialist.
Reason: {reason}

TICKET
{summary}

RETRIEVED KNOWLEDGE BASE CONTEXT
{context}

Write a short holding reply: acknowledge the specific impact, state that a
specialist now owns it, and set expectations for the next update using the
response targets in the context if they are present. Do not promise a fix time.
"""


class ReActAgent:
    """Bounded reason-act-observe loop over the mock tools."""

    def __init__(self, client: LLMClient, arm: Arm, max_steps: int = 1) -> None:
        self.client = client
        self.arm = arm
        self.max_steps = max_steps

    async def run(self, ctx: AgentContext) -> AgentResult:
        summary = ctx.render_summary()
        context = ctx.render_context()
        result = AgentResult(action="answer_directly")

        forced = hard_escalation_reason(ctx.full_text)
        if forced:
            result.reasoning.append(f"Guardrail: {forced} Escalating without consulting the model.")
            result.action = "escalate"
            result.escalation_reason = forced
            result.response_text = await self._complete(
                ESCALATION_TEMPLATE.format(reason=forced, summary=summary, context=context)
            )
            return result

        decision = await self._decide(summary, context, result)
        action = decision.get("action")
        observation_block = ""

        if action == "escalate":
            reason = str(decision.get("escalation_reason") or "Model judged human review necessary.")
            result.action = "escalate"
            result.escalation_reason = reason
            result.reasoning.append(f"Decision: escalate — {reason}")
            result.response_text = await self._complete(
                ESCALATION_TEMPLATE.format(reason=reason, summary=summary, context=context)
            )
            return result

        if action == "tool_call" and self.max_steps > 0:
            observation_block = self._invoke_tool(decision, ctx, result)

        result.action = "tool_call" if result.tool_calls else "answer_directly"
        result.response_text = await self._complete(
            ANSWER_TEMPLATE.format(
                summary=summary, context=context, observation=observation_block
            )
        )
        return result

    # -- internals ------------------------------------------------------
    async def _decide(
        self, summary: str, context: str, result: AgentResult
    ) -> dict[str, Any]:
        prompt = DECISION_TEMPLATE.format(
            summary=summary, context=context, tools=describe_tools()
        )
        try:
            completion = await self.client.complete(
                prompt,
                system=self.arm.variant.system,
                temperature=self.arm.variant.temperature,
                max_tokens=220,
            )
        except LLMError as exc:
            logger.warning("decision step failed (%s); defaulting to escalation", exc)
            result.reasoning.append(f"Model unavailable ({exc}); defaulting to human review.")
            return {"action": "escalate", "escalation_reason": "Automated triage unavailable."}

        decision = extract_json(completion.text)
        if decision.get("action") not in VALID_ACTIONS:
            result.reasoning.append(
                "Model returned an unparseable or invalid action; answering from context instead."
            )
            return {"action": "answer_directly"}
        thought = str(decision.get("thought") or "").strip()
        if thought:
            result.reasoning.append(f"Thought: {thought}")
        return decision

    def _invoke_tool(
        self, decision: dict[str, Any], ctx: AgentContext, result: AgentResult
    ) -> str:
        tool_name = str(decision.get("tool") or "")
        raw_args = decision.get("arguments")
        arguments: dict[str, Any] = dict(raw_args) if isinstance(raw_args, dict) else {}

        # The model does not see real identifiers, so they are injected here.
        if tool_name == "check_account_status":
            arguments["customer_id"] = ctx.customer_id or f"anon-{abs(hash(ctx.subject)) % 99999}"
        elif tool_name == "check_refund_eligibility":
            arguments["order_id"] = ctx.order_id or f"order-{abs(hash(ctx.body)) % 99999}"

        observation = run_tool(tool_name, arguments)
        result.tool_calls.append(
            {"tool": tool_name, "arguments": arguments, "result": observation}
        )
        result.reasoning.append(f"Action: {tool_name}({arguments})")
        result.reasoning.append(f"Observation: {observation}")

        if observation.get("requires_human_approval") or observation.get("status") == "locked":
            result.reasoning.append(
                "Observation requires a human (approval threshold or locked account), "
                "but the reply still explains the position."
            )
        return f"\nTOOL OBSERVATION ({tool_name})\n{observation}\n"

    async def _complete(self, prompt: str) -> str:
        try:
            completion = await self.client.complete(
                prompt,
                system=self.arm.variant.system,
                temperature=self.arm.variant.temperature,
                max_tokens=self.arm.variant.max_tokens,
            )
            return completion.text.strip()
        except LLMError as exc:
            logger.warning("answer step failed: %s", exc)
            return (
                "Thanks for getting in touch. We have received your ticket and a support "
                "specialist will follow up here shortly."
            )
