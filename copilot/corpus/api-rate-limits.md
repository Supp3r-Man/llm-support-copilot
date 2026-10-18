---
id: api-rate-limits
version: 1
area: api
title: API rate limits
---
## Default limits
The Nimbus API allows 600 requests per minute per API key. Exceeding the limit returns HTTP status 429.

## Handling 429 responses
Read the Retry-After header and wait that many seconds before retrying. Use exponential backoff with jitter for repeated failures.
