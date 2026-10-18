---
id: webhooks
version: 1
area: api
title: Webhooks
---
## Delivery and retries
Nimbus delivers webhook events with at least once semantics. Failed deliveries are retried for up to 24 hours with growing delays.

## Verifying signatures
Each webhook carries a signature header computed with your signing secret. Reject events whose signature does not match.
