"""Generate the synthetic labelled ticket dataset (data/tickets.jsonl).

Deterministic given the seed, so the committed dataset is reproducible. Each
category draws from its own subject/body template pools with shared filler
vocabulary, which keeps the task non-trivial: templates overlap enough that a
bag-of-words model has to rely on discriminative terms rather than memorising
whole sentences.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TIERS = ("free", "pro", "enterprise")

TEMPLATES: dict[str, dict[str, list[str]]] = {
    "billing": {
        "subjects": [
            "Double charged this month",
            "Refund for duplicate invoice {inv}",
            "Question about my subscription price",
            "Invoice {inv} does not match my plan",
            "Unexpected charge on my card",
            "Please cancel my plan and refund",
            "VAT missing from receipt {inv}",
            "Upgrade cost is unclear",
            "Charged after cancelling",
            "Annual billing discount question",
        ],
        "bodies": [
            "We were billed twice for the same period on invoice {inv} and the amount is wrong. Please refund the duplicate charge.",
            "My card was charged {amount} but our plan should cost less. Can you explain the pricing on this invoice?",
            "I cancelled the subscription last month yet a new payment of {amount} was taken today. This is unacceptable.",
            "The receipt for invoice {inv} has no VAT breakdown, and our finance team cannot process the expense without it.",
            "I want to understand what the upgrade to the higher plan would cost per seat before we commit budget.",
            "There is an overcharge of {amount} on the latest invoice. Please issue a refund to the original payment method.",
            "Our renewal price increased without notice. We need a breakdown of the charges and the refund policy.",
            "Payment keeps failing with a card error even though the card is valid, and now the account shows an outstanding balance.",
        ],
    },
    "technical": {
        "subjects": [
            "Dashboard keeps crashing",
            "API returns 500 errors",
            "Exports are extremely slow",
            "Webhook deliveries failing",
            "App freezes when loading reports",
            "Production is down for our team",
            "Sync stopped working overnight",
            "Timeouts on every large query",
            "Charts render blank",
            "Mobile app crashes on launch",
        ],
        "bodies": [
            "The reporting dashboard crashes every time we load more than a few thousand rows. This is blocking our daily workflow.",
            "Since this morning the API returns a 500 error on roughly half of our requests. Latency is above {seconds} seconds when it does respond.",
            "CSV exports used to take a few seconds and now they time out completely. The performance has become unusable.",
            "Webhook deliveries have been failing with connection timeouts since yesterday and we are losing events.",
            "The application hangs on a spinner for over {seconds} seconds whenever we open the analytics page. Nothing loads.",
            "Our production integration is down. Every sync attempt fails with an error and no data is coming through.",
            "The mobile app crashes immediately on launch after the latest update. Reinstalling did not help.",
            "Queries that used to be fast are now extremely slow and frequently time out, which is critical for us.",
        ],
    },
    "account": {
        "subjects": [
            "Locked out of my account",
            "SSO login not working",
            "Password reset link never arrives",
            "Cannot invite new users",
            "MFA code always rejected",
            "Need to transfer workspace ownership",
            "Deactivated user still counted as a seat",
            "Permission changes not applying",
            "Access denied after email change",
            "Please delete my account",
        ],
        "bodies": [
            "I am locked out and the password reset link never arrives in my inbox. I have checked spam repeatedly.",
            "Our SAML single sign-on stopped authenticating users this morning and the whole team cannot log in.",
            "The MFA one time code is rejected every time even though the clock on my device is correct.",
            "I cannot invite new users to the workspace; the invite button returns an access denied message.",
            "We removed a user but the seat is still counted and their permissions appear to be active.",
            "The admin role changes I made are not applying and members still see restricted pages.",
            "After changing my email address I lost access to the workspace entirely and cannot sign in.",
            "Please delete my account and all associated personal data in line with your retention policy.",
        ],
    },
    "feature_request": {
        "subjects": [
            "Please add dark mode",
            "Request: scheduled report delivery",
            "Would be great to have bulk editing",
            "Feature request: Slack integration",
            "Ability to customise dashboards",
            "Add support for custom fields",
            "Roadmap question about mobile offline mode",
            "Request for granular permission roles",
            "Suggestion: keyboard shortcuts",
            "Can you add multi currency support",
        ],
        "bodies": [
            "It would be great if you could add a dark theme; our team works late and the bright interface is tiring.",
            "We would love the ability to schedule reports to be emailed weekly instead of exporting them by hand.",
            "Please consider adding bulk editing so we can update many records at once rather than one by one.",
            "A native Slack integration for notifications would save us building our own webhook relay.",
            "Is customisable dashboard layout on the roadmap? Being able to pin our key metrics would help a lot.",
            "We need custom fields on records to model our own workflow. Is there any plan to support this?",
            "Granular permission roles would let us give read only access to contractors, which we cannot do today.",
            "Keyboard shortcuts for the most common actions would make the product much faster to use day to day.",
        ],
    },
}

FILLERS = [
    "Let me know if you need more detail.",
    "Happy to provide screenshots.",
    "This is affecting several people on our team.",
    "We have been a customer for two years.",
    "Any update would be appreciated.",
    "I have already tried the basic troubleshooting steps.",
    "",
    "",
]

URGENT_SUFFIX = [
    "This is urgent, please escalate.",
    "We need this resolved as soon as possible.",
    "Our whole team is blocked.",
    "",
    "",
    "",
]


# Deliberately hard cases: vocabulary from one category, true label from
# another, plus terse tickets with little signal. Without these the templated
# corpus is linearly separable and the reported metrics are meaningless.
HARD_CASES: list[tuple[str, str, str]] = [
    ("billing", "Cannot access invoices", "I can log in fine but the billing page shows access denied when I try to download a receipt."),
    ("billing", "Refund request after outage", "The platform was down for two days so we are requesting a partial refund for the affected period."),
    ("billing", "Upgrade blocked", "The upgrade button errors out, so we are still on the old plan and being charged the old rate."),
    ("account", "Charged for a user I removed", "I deactivated a teammate but they still have access to the workspace. Seat management seems broken."),
    ("account", "Login page very slow", "The sign in page takes forever to load and eventually rejects my password. I am locked out."),
    ("account", "Please remove my data", "Delete my account and everything associated with it."),
    ("technical", "Report export shows wrong totals", "The numbers in the exported report do not match the dashboard, which affects the amounts we invoice clients."),
    ("technical", "Dark mode causes rendering bug", "After enabling the new dark theme the charts render blank and the page crashes."),
    ("technical", "Slow", "Everything is slow today."),
    ("feature_request", "Refund automation", "Could you add a way for admins to issue refunds from the billing dashboard themselves?"),
    ("feature_request", "SSO for smaller plans", "We would like SAML single sign on to be available on the pro plan, not just enterprise."),
    ("feature_request", "Alert on failed syncs", "It would help if the product notified us when a sync fails instead of us discovering it later."),
    ("billing", "Question", "Quick question about my last payment."),
    ("account", "Help", "I need help getting back into my workspace."),
    ("technical", "Errors everywhere", "Getting errors on nearly every page since this morning."),
    ("feature_request", "Idea", "Suggestion for a small improvement to the export flow."),
]


def generate(n_per_class: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for category, pools in TEMPLATES.items():
        for _ in range(n_per_class):
            inv = f"INV-{rng.randint(1000, 9999)}"
            amount = f"${rng.choice([19, 49, 99, 149, 249, 499])}"
            seconds = rng.choice([10, 20, 30, 45, 60])
            subject = rng.choice(pools["subjects"]).format(inv=inv)
            body = rng.choice(pools["bodies"]).format(inv=inv, amount=amount, seconds=seconds)
            extra = " ".join(x for x in (rng.choice(FILLERS), rng.choice(URGENT_SUFFIX)) if x)
            rows.append(
                {
                    "subject": subject,
                    "body": f"{body} {extra}".strip(),
                    "category": category,
                    "tier": rng.choice(TIERS),
                }
            )
    for category, subject, body in HARD_CASES:
        rows.append(
            {
                "subject": subject,
                "body": body,
                "category": category,
                "tier": rng.choice(TIERS),
            }
        )
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=45, help="tickets per category")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "tickets.jsonl")
    args = parser.parse_args()

    rows = generate(args.per_class, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} tickets to {args.out}")


if __name__ == "__main__":
    main()
