# Responsible AI and Governance

The brief asks for a lightweight but concrete responsible-AI approach covering explainability, feedback handling, and accuracy. This document describes what is actually implemented, and is explicit about where a control is real versus where it is best-effort.

The organising principle throughout: **a control that can be argued with is not a control.** Guardrails filter noise and create a record. The things that must not fail are enforced in SQL and in Python, where no conversation can reach them.

---

## 1. Explainability

### Every sentence has a receipt

For any reply the assistant produces, the interface can show the exact backend calls behind it: the arguments sent, a summary of what came back, per-step latency, and whether the answer was grounded. It is one click, under every message, for the customer as well as the operator.

This is not a rendering of what the model claims it did. It is a rendering of `tool_invocations`, the table the orchestrator wrote to while the turn was running.

```
1. [guardrail]  Input screening      clean (allow), injection score 0.0004      412ms
2. [reasoning]  Tool selection       6 of 17 tools offered to the model
3. [reasoning]  Planning (step 1)    Need to call get_order_status.
4. [tool_call]  get_order_status     {"order_number": "1234"}
                                     order 1234: Shipped                          9ms
5. [answer]     Answer composed      grounded in 1 backend call: get_order_status
```

### Three levels of audience

| Audience | Surface | What they get |
| --- | --- | --- |
| Customer | Trace under each reply | Which lookups produced this answer, in plain language |
| Operator | `GET /api/ops/audit/{turn_id}` | Full arguments and full result payloads for every call in a turn |
| Compliance | `tool_invocations`, `guardrail_events` tables | Queryable history with correlation ids, retained per the privacy policy |

### The model's own planning is shown, not hidden

`gpt-oss-120b` exposes its reasoning text, and the trace surfaces it verbatim. When the assistant does something surprising, the reason is usually visible there.

This is deliberately labelled as *planning*, not as *justification*. Model-reported reasoning is a description of a process, not a proof of one. The tool invocation rows are the evidence; the reasoning text is context.

### Citations for retrieved knowledge

Policy answers carry the source document and heading, rendered as chips beside the reply: `Returns, Exchanges and Refunds > Return window`. A customer told about a 30-day window can see which document said so.

---

## 2. Grounding: the accuracy mechanism

The brief's core requirement is that transactional responses come from backend APIs rather than AI guesses. That is enforced in three independent layers.

### Layer 1: Structural, the strongest

Facts the model could get wrong are computed in Python and handed over pre-formatted.

| Fact | How the model receives it | What this removes |
| --- | --- | --- |
| Price | `"$26.99"`, already formatted | Arithmetic and currency-format errors |
| Delivery date | A complete written sentence | Date arithmetic errors |
| Order status | `"Out for delivery"` | Enum-to-prose translation errors |
| Stock | Per-variant integers plus an explicit `any_available` boolean | Rounding "3 left" up to "in stock" |

The model cannot make a date arithmetic error if it never does date arithmetic. This converts a category of hallucination into one that is structurally impossible, which is stronger than any instruction.

### Layer 2: Behavioural

The system prompt names the tool to call before each class of claim, and the tool descriptions reinforce it. This is genuinely effective and genuinely not sufficient.

### Layer 3: Verification, the backstop

[`app/guardrails/output_guard.py`](../app/guardrails/output_guard.py) checks the finished reply. If it makes a price, stock, order-status, delivery-date or policy claim and no supporting tool ran, the reply is **replaced**, not merely flagged, with a message specific to the claim that fired.

**What this catches:** the model answering a price, stock or order question from its own weights when the backend was never consulted. That is the failure mode the brief is concerned with, and it is caught reliably.

**What it does not catch:** whether the specific number in the sentence matches the specific number in the tool result. Verifying that needs span-level attribution. This limitation is stated plainly rather than glossed, and is in [ACCURACY_AND_LIMITATIONS.md](ACCURACY_AND_LIMITATIONS.md) with a concrete plan.

---

## 3. Guardrails

### Inbound, cheapest first

| Order | Layer | Catches | Cost |
| --- | --- | --- | --- |
| 1 | Sliding-window rate limit | Abuse, runaway clients | Microseconds |
| 2 | Shape checks | Empty and oversized input | Microseconds |
| 3 | Deterministic pattern match | Known injection phrasings, schema probes, privilege escalation | Microseconds |
| 4 | Prompt Guard 2 classifier | Novel phrasings the pattern list does not know | ~300ms, ~$0 |

Layers 3 and 4 exist together on purpose. The regex list is precise but only knows what we thought of. The classifier generalises but is a model, and models have false negatives. Neither is trusted alone. The ordering means an obvious rejection never costs a model call: the injection probe in the smoke test is blocked in 0 ms.

**False positives are treated as failures.** A guardrail that blocks "forget what I said earlier about the size" is not a safe guardrail, it is a broken shopping assistant. The test suite includes explicit false-positive probes for ordinary shopping language that brushes against injection vocabulary: *"can you ignore the colour and just show me all of them"*, *"what are the rules for returns"*, *"I need a table for my order details"*.

**Degradation is logged, not silent.** If the classifier is unavailable, `classify_injection` returns a sentinel, the deterministic layer stands alone, and `guardrail.classifier_unavailable` is logged. Failing closed would take the assistant down over a provider hiccup; failing silently open would leave no record. Neither is acceptable.

### Outbound

1. **Schema disclosure.** Internal table names, ORM terms and SQL verbs in a customer-facing reply block it outright.
2. **Internals disclosure.** System-prompt and tool-schema content blocks it outright.
3. **PII redaction.** Card numbers (Luhn-validated), emails, phone numbers, IBANs, API keys.
4. **Grounding verification.** As above.
5. **Truncation.** A reply cut off by the token limit is labelled, not shown as if complete.

Redaction is deliberately conservative in two places, because **over-redaction is a real failure here, not a safe default**. Card numbers are confirmed by Luhn checksum and phone numbers require separators or an international prefix. Without both, tracking numbers and order numbers - the identifiers this assistant exists to discuss - get redacted out of the answer.

### What the guardrails are not

They are a filter and an audit record. They are **not** what prevents data disclosure. That is:

```python
select(Order).where(Order.order_number == cleaned, Order.customer_id == customer_id)
```

Both predicates, always, with no code path in the repository that omits the second. If every guardrail in this system were disabled, a customer still could not read another customer's order.

---

## 4. Data protection

### Authorisation

Enforced in the data access layer. `ToolContext` injects `customer_id` from the authenticated session, and it appears in no tool schema, so *the model has no vocabulary for requesting another customer's data*. A test asserts no tool exposes an identity parameter, so a future tool cannot quietly reintroduce one.

### No enumeration oracle

A nonexistent order and a real order belonging to someone else return the **same** `ORDER_NOT_FOUND`, with the same recovery hint. Distinguishing them would let anyone discover which order numbers are real by asking about them one at a time. There is a test asserting the two responses are identical apart from the echoed number.

### Data minimisation

Every object crossing the service boundary is an explicit Pydantic model in [`app/schemas.py`](../app/schemas.py). A new database column cannot become a new field in the model's context, or in the browser, without someone editing that file. Internal identifiers, cost columns and other customers' data have no path outward.

### PII handling

The system stores no real personal data: the dataset is synthetic, and payment details never enter it. For data a customer might volunteer mid-conversation, redaction runs **before** the message is written to the transcript, so a pasted card number never reaches storage.

---

## 5. Feedback handling

### It joins back to evidence

Ratings are recorded against `turn_id`. Because every turn already has a complete tool-invocation trail keyed by the same id, a "not helpful" rating is not an opaque complaint. It can be joined to the exact tool calls, arguments and results that produced the reply.

```sql
SELECT f.reason, t.tool_name, t.arguments_json, t.status
FROM feedback f
JOIN tool_invocations t ON t.turn_id = f.turn_id
WHERE f.rating = 'not_helpful';
```

That join is what makes the signal usable for improvement rather than merely countable. Three examples of what it distinguishes, which a rating alone cannot:

- Negative ratings clustering on turns where `search_products` returned zero results is a **retrieval** problem: synonyms, or catalogue gaps.
- Clustering on turns where every tool succeeded is a **prompt or response-quality** problem.
- Clustering on turns with a `ToolError` is a **coverage** problem: a capability customers expect and the assistant lacks.

### Structured reasons

A "not helpful" rating prompts for a category, so aggregation is meaningful rather than free text nobody reads. `GET /api/feedback/summary` returns totals, helpful rate, and the top reasons.

### The loop back to the product

Implemented: capture, storage, joinable audit, aggregation endpoint.

Not implemented, and it should be said plainly: automated retraining, prompt A/B testing, and a human review queue. What exists is the data foundation those need, which is the part that is hard to retrofit. [SCALING.md](SCALING.md#the-improvement-loop) describes the full loop.

---

## 6. Human escalation

The assistant escalates rather than improvising when:

- A tool returns `TOOL_EXECUTION_ERROR`. The recovery hint explicitly instructs it to offer a human agent and not invent the answer.
- The language model is unreachable. The turn ends with an honest failure message offering a human.
- A request falls outside catalogue, orders and purchases.
- A warranty claim is disputed - `request_return` returns a recovery hint that routes genuine faults to Customer Care rather than refusing on the return window.

The privacy policy in the RAG corpus commits to escalation on request, on distress, and on out-of-scope requests, so the assistant can state the escalation policy from retrieved text rather than from its own assumptions about what it is allowed to promise.

---

## 7. Governance surfaces

| Surface | Answers |
| --- | --- |
| Governance tab in the interface | Tool call volumes, latencies, guardrail decisions by rule, block rate |
| `GET /api/ops/metrics` | The same, as JSON, for a monitoring system |
| `GET /api/ops/audit/{turn_id}` | Full evidence trail for one turn |
| `GET /api/ops/tools` | The tool contract as the model sees it, for review |
| `guardrail_events` table | Every decision, allow and block, with score and detail |
| Structured JSON logs | One object per line, correlation id on every record |

Guardrail events record allows as well as blocks. A review that sees only blocks has no denominator and cannot tell a well-calibrated guardrail from one that is blocking a tenth of legitimate traffic.

In a real deployment these routes sit behind an operator role. That is called out in [SCALING.md](SCALING.md#access-control-for-operations) rather than faked here with a hardcoded password.

---

## 8. Honest limitations

Stated here rather than buried, and expanded in [ACCURACY_AND_LIMITATIONS.md](ACCURACY_AND_LIMITATIONS.md).

- **Grounding is class-level, not span-level.** It verifies that a *kind* of claim had a *kind* of supporting call. It does not verify that the number in the sentence matches the number in the result.
- **Injection defence is probabilistic.** Prompt Guard 2 has a false-negative rate. This is why authorisation does not depend on it.
- **PII detection is regex-based.** Precise on structured identifiers, and it makes no attempt at unstructured PII such as names in free text.
- **Identity is simulated.** A session cookie binds to a demo customer. The shape is right - identity arrives from outside the model's reach - but there is no real authentication provider.
- **No adversarial red-team programme.** The suite covers known attack classes. It is not a substitute for a dedicated exercise, which is what an enterprise deployment would require before launch.
