"""Mock tools available to the agent.

Deterministic: results are derived from a hash of the identifier, so the same
customer or order always returns the same state. That keeps end-to-end tests
stable and makes the bandit experiment reproducible.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ToolFn = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: tuple[str, ...]
    fn: ToolFn

    def signature(self) -> str:
        return f"{self.name}({', '.join(self.parameters)}) — {self.description}"


def _bucket(identifier: str, modulo: int) -> int:
    digest = hashlib.blake2b(identifier.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % modulo


def check_account_status(customer_id: str) -> dict[str, Any]:
    """Return the mock account state for a customer."""
    if not customer_id:
        return {"error": "customer_id is required", "found": False}
    states = ("active", "past_due", "locked", "read_only")
    state = states[_bucket(customer_id, len(states))]
    return {
        "found": True,
        "customer_id": customer_id,
        "status": state,
        "seats_used": 3 + _bucket(customer_id + "seats", 20),
        "mfa_enrolled": _bucket(customer_id + "mfa", 2) == 1,
        "sso_enabled": _bucket(customer_id + "sso", 3) > 0,
        "failed_logins_24h": _bucket(customer_id + "fails", 12),
        "notes": {
            "active": "Account is healthy; no billing or access blocks.",
            "past_due": "Payment failed; workspace becomes read-only after the third retry.",
            "locked": "Locked after ten failed sign-ins; an agent can clear it after ID checks.",
            "read_only": "Read-only due to unpaid invoice; restored on successful payment.",
        }[state],
    }


def check_refund_eligibility(order_id: str) -> dict[str, Any]:
    """Return mock refund eligibility for an order."""
    if not order_id:
        return {"error": "order_id is required", "found": False}
    reasons = (
        ("eligible", "Duplicate charge confirmed; full refund is automatic."),
        ("eligible_prorata", "Annual plan cancelled early; pro-rata refund applies."),
        ("ineligible_window", "Charge is older than the 60 day refund window."),
        ("needs_review", "Amount exceeds $500; billing operations must approve."),
    )
    outcome, reason = reasons[_bucket(order_id, len(reasons))]
    amount = 19 + _bucket(order_id + "amt", 60) * 12
    return {
        "found": True,
        "order_id": order_id,
        "eligibility": outcome,
        "amount_usd": amount,
        "days_since_charge": _bucket(order_id + "days", 120),
        "requires_human_approval": outcome == "needs_review" or amount > 500,
        "reason": reason,
    }


TOOLS: dict[str, Tool] = {
    "check_account_status": Tool(
        name="check_account_status",
        description="Look up subscription state, seats, SSO/MFA and lockout status for a customer.",
        parameters=("customer_id",),
        fn=check_account_status,
    ),
    "check_refund_eligibility": Tool(
        name="check_refund_eligibility",
        description="Check whether an order qualifies for a refund and whether approval is needed.",
        parameters=("order_id",),
        fn=check_refund_eligibility,
    ),
}


def describe_tools() -> str:
    return "\n".join(f"- {tool.signature()}" for tool in TOOLS.values())


def run_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name, tolerating a malformed argument dict."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"unknown tool {name!r}", "available": sorted(TOOLS)}
    kwargs = {key: arguments.get(key, "") for key in tool.parameters}
    try:
        return tool.fn(**kwargs)
    except Exception as exc:  # noqa: BLE001 - tool failures must not kill the pipeline
        return {"error": f"tool {name} failed: {exc}"}
