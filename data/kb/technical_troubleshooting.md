---
id: technical_troubleshooting
title: Technical Troubleshooting & Performance
tags: [technical, performance, errors, api, sync, outage]
---

# Technical Troubleshooting & Performance

## First response checklist
Collect: workspace ID, approximate timestamp with timezone, one affected request ID, browser or
client version, and whether the problem affects one user or everyone. Without these the platform
team cannot correlate logs.

## Slow dashboards and exports
Reports spanning more than 90 days or 50,000 rows fall back to a slower query path and may take
up to 60 seconds. The supported workaround is to narrow the date range or use the asynchronous
export endpoint, which emails a download link when ready.

## HTTP 5xx from the API
- Sustained 5xx across many customers indicates a platform incident; check the status page first
  and link the customer to it rather than debugging individually.
- Isolated 500s usually come from a malformed payload on a nested field. Ask for the request ID.
- Rate limits return 429, not 500. The default limit is 600 requests per minute per workspace.

## Webhook delivery failures
Deliveries retry with exponential backoff for 24 hours. Endpoints that time out repeatedly are
disabled automatically and must be re-enabled in Settings → Integrations. Receivers must respond
within 5 seconds.

## Crashes on the mobile app
Clients older than two releases are unsupported. Ask for the build number before investigating.

## Sync stopped working
Check the integration credential expiry first — expired OAuth tokens are the single most common
cause and are self-service to reconnect.
