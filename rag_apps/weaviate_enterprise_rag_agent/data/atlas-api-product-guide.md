---
title: Atlas API Product Guide
doc_type: product
team: platform
---
# Atlas API Product Guide

## Rate limits

The Standard plan allows 600 requests per minute per API key. The Enterprise plan allows 6,000 requests per minute. Requests above the limit receive HTTP 429 with a Retry-After header.

## API keys

Keys are scoped to one workspace and can be rotated without downtime: a rotated key stays valid for 24 hours alongside its replacement.

## Webhooks

Webhooks are signed with HMAC-SHA256 and sent in the X-Atlas-Signature header. Failed deliveries are retried with exponential backoff for up to 72 hours.
