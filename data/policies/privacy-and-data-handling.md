---
title: Privacy and Data Handling
topic: privacy
version: 5.0
effective: 2026-05-01
owner: Aurelia Data Governance
---

# Privacy and Data Handling

## What this assistant can see

The assistant operates under the identity of the signed-in customer. It can read the catalogue, which is public, and it can read **only that customer's own orders**. It cannot enumerate other customers, look up an order that belongs to someone else, or read payment instrument details, which are held by the payment processor and never enter Aurelia systems.

Authorisation is enforced in the backend service layer, not in the model prompt. A request for another customer's order is refused by the data access layer regardless of what the conversation asks for.

## What is retained

| Data | Retention | Purpose |
| --- | --- | --- |
| Conversation transcripts | 90 days | Quality review and dispute resolution |
| Tool invocation audit records | 24 months | Traceability of transactional answers |
| Guardrail decision records | 24 months | Safety and governance review |
| Feedback ratings and comments | 24 months | Model and prompt improvement |

Transcripts are redacted for card numbers, government identifiers and email addresses before storage.

## Customer rights

Customers may request a copy of their conversation history, or its deletion, through Aurelia Customer Care. Deletion requests are honoured within 30 days. Audit records required for financial dispute resolution are retained in de-identified form.

## Human escalation

Any conversation can be escalated to a human agent on request. The assistant escalates automatically when it cannot answer from backend data, when a customer expresses distress, or when a request falls outside its defined scope of catalogue, orders and purchases.
