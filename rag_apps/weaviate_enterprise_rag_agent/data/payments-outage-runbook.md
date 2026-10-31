---
title: Payments Outage Runbook
doc_type: runbook
team: sre
---
# Payments Outage Runbook

## Severity

Declare SEV-1 when checkout error rate stays above 5 percent for 5 minutes. Page the payments on-call engineer and open an incident channel named inc-payments-<date>.

## Mitigation

First check the status of the card processor. If the primary processor is failing, enable the feature flag payments_failover to route traffic to the secondary processor. Failover takes about 2 minutes to apply.

## Recovery

Disable payments_failover only after the primary processor has been healthy for 30 minutes. Reconcile transactions from the failover window within 24 hours, and publish the postmortem within 5 business days.
