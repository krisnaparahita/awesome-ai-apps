---
title: Database Failover Runbook
doc_type: runbook
team: platform
---
# Database Failover Runbook

## When to fail over

Fail over the primary Postgres cluster when it has been unreachable for 3 minutes or replication lag on the standby is under 30 seconds and the primary is degraded.

## Procedure

Run the command db-promote from the bastion host, then confirm the new primary with db-status. Update the pgbouncer target and restart connection pools one region at a time.

## Follow-up

Rebuild the old primary as a standby within 24 hours. The on-call engineer records the data loss window, if any, in the incident ticket.
