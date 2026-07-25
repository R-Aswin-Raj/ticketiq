---
id: account_access_troubleshooting
title: Account & Access Troubleshooting
tags: [account, login, sso, password, mfa, permissions]
---

# Account & Access Troubleshooting

## Password reset link not arriving
1. Confirm the address is the one on the account (Settings → Profile).
2. Reset emails are sent from `no-reply@ticketiq.example`; ask the customer to check spam
   and any corporate quarantine.
3. Links expire after 60 minutes and are single use. Request a fresh link if the old one was opened.
4. If nothing arrives within 15 minutes, the address is likely on a bounce suppression list —
   this requires an agent to clear it.

## SSO / SAML failures
- The most common cause is an expired identity-provider signing certificate. Check the
  certificate expiry date in Settings → Security → SSO.
- A changed IdP entity ID or ACS URL also breaks authentication for every user at once.
- Break-glass: workspace owners can always sign in with email and password even when SSO is down.

## MFA codes rejected
Time-based codes fail when the device clock drifts more than 30 seconds. Ask the customer to
enable automatic time sync. Owners can reset a member's MFA enrolment from the admin panel.

## Locked out accounts
Ten consecutive failed sign-in attempts lock the account for 30 minutes. An agent can clear the
lock immediately after verifying identity.

## Seats and permissions
Deactivating a user releases the seat at the end of the billing period, not immediately — this is
expected behaviour and a frequent source of "I was charged for a removed user" tickets.
Permission changes take effect on the member's next session refresh, up to five minutes.

## Account deletion
Deletion requests are honoured within 30 days. Only a workspace owner may request deletion, and
the request must be confirmed by email.
