"""Deterministic offline LLM stub.

This is what ``LLM_MODE=mock`` selects, and what the tests mock against. It is
rule-based but honours the two prompt variants (so the bandit still has a real
choice with measurably different latency and quality), and it emits the same
JSON contract the real backends are prompted for, so the agent loop is
exercised end to end without a network call.

Simulated latency is deliberate: variant B is "slower but better", which is
exactly the trade-off the reward function ``10 * feedback - latency`` exists to
resolve.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any

from ticketiq.llm.base import LLMResult

_ESCALATION_MARKERS = (
    "outage",
    "production down",
    "production is down",
    "data loss",
    "security",
    "legal",
    "lawyer",
    "regulator",
    "chargeback",
    "cancel",
    "churn",
    "escalate",
)
_REFUND_MARKERS = ("refund", "double charged", "overcharge", "duplicate", "money back")
_ACCOUNT_MARKERS = ("locked", "sso", "saml", "password", "mfa", "sign in", "log in", "login")


class MockLLM:
    """Offline stand-in for a chat completion endpoint."""

    def __init__(
        self,
        name: str = "mock",
        *,
        base_latency_s: float = 0.05,
        jitter_s: float = 0.02,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.base_latency_s = base_latency_s
        self.jitter_s = jitter_s
        self._rng = random.Random(seed)

    async def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> LLMResult:
        started = time.perf_counter()
        delay = max(0.0, self.base_latency_s + self._rng.uniform(-self.jitter_s, self.jitter_s))
        await asyncio.sleep(delay)
        text = (
            self._decide(prompt)
            if "respond with json" in prompt.lower() or '"action"' in prompt
            else self._answer(prompt, system)
        )
        return LLMResult(
            text=text,
            model=self.name,
            latency_s=round(time.perf_counter() - started, 4),
            usage={"prompt_chars": len(prompt) + len(system), "completion_chars": len(text)},
        )

    # -- decision step --------------------------------------------------
    @staticmethod
    def _ticket_section(prompt: str) -> str:
        """Isolate the ticket text.

        The full prompt also contains the tool catalogue and the retrieved KB
        chunks, both of which mention words like "refund". Matching against the
        whole prompt would make the stub's decision depend on boilerplate
        rather than on the customer's actual ticket.
        """
        start = prompt.find("TICKET")
        if start == -1:
            return prompt
        end = prompt.find("RETRIEVED KNOWLEDGE", start)
        return prompt[start : end if end != -1 else len(prompt)]

    def _decide(self, prompt: str) -> str:
        lc = self._ticket_section(prompt).lower()
        payload: dict[str, Any]
        if any(marker in lc for marker in _ESCALATION_MARKERS) and "urgency: high" in lc:
            payload = {
                "thought": (
                    "High urgency plus an escalation trigger in the ticket text; "
                    "policy says a human owns this."
                ),
                "action": "escalate",
                "tool": None,
                "arguments": {},
                "escalation_reason": "Escalation rule matched: high urgency with a trigger phrase.",
            }
        elif any(marker in lc for marker in _REFUND_MARKERS):
            payload = {
                "thought": "A refund is requested, so eligibility must be checked before replying.",
                "action": "tool_call",
                "tool": "check_refund_eligibility",
                "arguments": {"order_id": "auto"},
            }
        elif any(marker in lc for marker in _ACCOUNT_MARKERS):
            payload = {
                "thought": "Access problem; the account state determines the right instructions.",
                "action": "tool_call",
                "tool": "check_account_status",
                "arguments": {"customer_id": "auto"},
            }
        else:
            payload = {
                "thought": "The retrieved policy fully answers this; no tool or human needed.",
                "action": "answer_directly",
                "tool": None,
                "arguments": {},
            }
        return json.dumps(payload)

    # -- answer step ----------------------------------------------------
    def _answer(self, prompt: str, system: str) -> str:
        detailed = "step-by-step" in system.lower() or "empathetic" in system.lower()
        snippet = self._first_context_line(prompt)
        if detailed:
            return (
                "Thanks for flagging this, and sorry for the disruption it has caused.\n\n"
                f"Here is what our policy says: {snippet}\n\n"
                "Next steps:\n"
                "1. We have recorded the details of your ticket.\n"
                "2. The relevant policy above has been applied to your case.\n"
                "3. You will get a further update from us on this thread.\n\n"
                "If anything above does not match what you are seeing, reply here and we "
                "will pick it straight back up."
            )
        return (
            f"Thanks for getting in touch. {snippet} "
            "We have applied this to your ticket and will follow up here with the outcome."
        )

    @staticmethod
    def _first_context_line(prompt: str) -> str:
        for line in prompt.splitlines():
            stripped = line.strip("-• ").strip()
            if len(stripped) > 60 and not stripped.lower().startswith(
                ("ticket", "subject", "body", "category", "urgency", "customer", "aspects")
            ):
                return stripped[:280]
        return "Our support policy covers this case."
