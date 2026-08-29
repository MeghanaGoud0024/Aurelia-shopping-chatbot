# Architecture

## The organising idea

There is exactly one architectural commitment, and everything else follows from it:

> **The language model is a router and a writer. It is never a source of truth.**

The model decides which backend function to call and how to phrase the result. It does not decide what a product costs, whether something is in stock, where an order is, or what the returns policy says. Those facts are read from the database, formatted in Python, and handed to the model to repeat.

This is not a stylistic preference. It is what makes the system auditable: for any sentence the assistant produces, there is a row in `tool_invocations` holding the exact call and the exact result behind it.

---

## System diagram

```
                            BROWSER
   +----------------------------------------------------------------+
   |  Chat transcript   Product/order cards   Trace panel   Bag      |
   |  Governance panel  Checkout confirm button                      |
   +----------------------------------------------------------------+
            |  POST /api/chat                  ^  reply + cards + trace
            v                                  |
   +----------------------------------------------------------------+
   |  FastAPI                                                        |
   |  correlation id -> structured logging -> session identity       |
   +----------------------------------------------------------------+
            |
            v
   +----------------------------------------------------------------+
   |  INBOUND GUARDRAILS                    app/guardrails/          |
   |  1 rate limit   2 shape   3 injection patterns   4 Prompt Guard |
   |  cheapest first, so an obvious rejection costs no model call    |
   +----------------------------------------------------------------+
            |  allowed
            v
   +----------------------------------------------------------------+
   |  ORCHESTRATOR                       app/agent/orchestrator.py   |
   |                                                                 |
   |    +------------------------------------------------------+     |
   |    |  loop, bounded by iterations and tool-call budget    |     |
   |    |                                                      |     |
   |    |   tool routing  --->  LLM plans  --->  tool calls    |     |
   |    |   (token budget)      (what to look up)     |        |     |
   |    |         ^                                   v        |     |
   |    |         +------- results fed back ----------+        |     |
   |    +------------------------------------------------------+     |
   |                          |  no more tool calls                  |
   |                          v                                      |
   |                    LLM writes the reply from tool results       |
   +----------------------------------------------------------------+
            |                                        |
            |  every call                            |  every decision
            v                                        v
   +---------------------------+        +----------------------------+
   |  TOOL LAYER               |        |  AUDIT TRAIL               |
   |  app/agent/tools.py       |        |  tool_invocations          |
   |  17 tools, schema and     |        |  guardrail_events          |
   |  executor declared        |        |  chat_messages             |
   |  together                 |        |  feedback                  |
   |                           |        |                            |
   |  ToolContext injects      |        |  same transaction as the   |
   |  customer_id - the model  |        |  business data, so audit   |
   |  cannot express it        |        |  and ledger cannot diverge |
   +---------------------------+        +----------------------------+
            |
            v
   +----------------------------------------------------------------+
   |  SERVICE LAYER                              app/services/       |
   |  catalog.py    orders.py    cart.py                             |
   |                                                                 |
   |  * authorisation is a SQL WHERE clause, not a prompt rule       |
   |  * money is integer cents, formatted once                       |
   |  * state machines enforced here, not described to the model     |
   |  * failures returned as typed errors with recovery hints        |
   +----------------------------------------------------------------+
            |                                    |
            v                                    v
   +---------------------------+     +------------------------------+
   |  RETRIEVAL                |     |  DATABASE (SQLite)           |
   |  app/retrieval/           |     |  products, variants,         |
   |  BM25 + RRF over          |     |  customers, orders,          |
   |  catalogue and policies   |     |  order_items, order_events,  |
   |  ranks, never answers     |     |  cart_items, audit tables    |
   +---------------------------+     +------------------------------+
            |
            v
   +----------------------------------------------------------------+
   |  OUTBOUND GUARDRAILS                   app/guardrails/          |
   |  disclosure -> PII redaction -> GROUNDING CHECK -> truncation   |
   |                                                                 |
   |  grounding: a reply making a price, stock, order, delivery or   |
   |  policy claim with no supporting tool call is replaced          |
   +----------------------------------------------------------------+
            |
            v
                            BROWSER
```

---

## The lifecycle of one turn

Take *"What is the status of my order 1234?"*.

**1. Identity, before anything else.** `app/api/deps.py` resolves the session cookie to exactly one `customer_id`. This never comes from the request body, so a client cannot select whose orders it reads by editing JSON.

**2. Inbound guardrails**, cheapest first. Rate limit, then length and emptiness, then a deterministic injection-pattern match, then the Prompt Guard 2 classifier. The ordering matters: an obvious rejection never costs a model call. Every decision, allow or block, is written to `guardrail_events` - a governance review needs the denominator, not just the blocks.

**3. Tool routing.** The message is matched to tool groups, so the model receives six schemas rather than seventeen. This is a token-budget optimisation, described in [PROMPT_DESIGN.md](PROMPT_DESIGN.md#4-tool-routing); it never removes a capability the customer is entitled to.

**4. The model plans.** It sees the system prompt, the recent conversation, and the routed tool schemas. It emits a call: `get_order_status(order_number="1234")`.

**5. The tool executes.** `ToolContext` carries `session`, `session_id`, `customer_id` and `customer_name`. The model supplied only `order_number`; identity was injected. The service runs:

```sql
SELECT ... FROM orders WHERE order_number = '1234' AND customer_id = :id
```

Both predicates, always. There is no code path in the repository that fetches an order by number alone.

**6. The result is audited and returned.** Arguments, result, status and latency go to `tool_invocations` in the same transaction as any business change. The result is serialised back into the model's context.

**7. The model writes the reply** from the tool result. It repeats the pre-formatted delivery sentence and money values rather than deriving them.

**8. Outbound guardrails.** Disclosure check, PII redaction, punctuation normalisation, then the grounding check: this reply makes an order-status claim, so at least one of `get_order_status`, `track_shipment`, `list_my_orders`, `cancel_order`, `request_return` or `place_order` must have run. It did.

**9. The response carries its own evidence.** Reply text, structured cards, and the full trace, so the interface can show exactly how the answer was produced.

---

## Where each guarantee lives

The most important property of this design is that **each guarantee is enforced in exactly one place, and that place is not the prompt.**

| Guarantee | Enforced in | Not enforced by |
| --- | --- | --- |
| A customer sees only their own orders | `WHERE customer_id = :id` in `services/orders.py` | The system prompt saying so |
| A shipped order cannot be cancelled | `OrderStatus.is_cancellable` checked in `cancel_order` | The model knowing the rules |
| A purchase needs human confirmation | Single-use token in `services/cart.py`, redeemed over a separate HTTP route | The model being asked to wait |
| Prices are correct | Integer cents, formatted once in `schemas.Money` | The model doing division |
| Delivery dates are correct | Computed in `orders._delivery_message` | The model doing date arithmetic |
| Stock is correct | Per-variant rows, re-checked inside the write transaction | The search index being fresh |
| No schema leaks to the browser | Explicit Pydantic response models plus the output guard | Hoping the model stays quiet |
| Every claim is backed by a lookup | Grounding check in `output_guard.py` | Instructions alone |

The system prompt does state these rules, because a model that understands the boundary produces better answers inside it. But if the prompt were deleted entirely, none of the guarantees above would break. That is the test of whether a control is real.

---

## Module responsibilities

### `app/agent/`

| Module | Responsibility |
| --- | --- |
| `orchestrator.py` | Runs one turn end to end. Owns the loop bounds, the audit calls, and the assembly of the response. |
| `tools.py` | The model-facing contract. Schema and executor for each tool declared together so they cannot drift. |
| `prompts.py` | System prompt. Deliberately short; behavioural guidance lives in tool descriptions. |
| `routing.py` | Selects which tool groups to expose. Token optimisation, never a security control. |
| `llm.py` | Transport. Timeouts, retry classification, provider-aware rate-limit backoff, defensive argument parsing. |
| `fallback.py` | Deterministic planner for running with no API key. |

### `app/services/`

The deterministic core. Everything the assistant asserts is produced here. Authorisation, state machines and pricing all live at this layer, which means the REST API and the tool layer get the same guarantees without duplicating them.

Failures are returned as typed `ToolError` values with a `recovery_hint`, not raised. A raised exception gives the model nothing to act on; a typed error lets it retry with corrected arguments or explain the limitation. Ambiguity is treated as a failure too: adding a product available in two colours returns `NEEDS_COLOR` listing the real options, so the assistant asks rather than assumes.

### `app/retrieval/`

Two corpora with genuinely different access patterns.

**Catalogue** retrieval is a *ranking* step. BM25 narrows ~1,100 products to a shortlist; the shortlist is then re-read from SQL, so price and stock in the reply are live values rather than indexed snapshots. A stale index can affect *which* products are shown, never whether their details are correct.

**Policy** retrieval is classical RAG: chunk on `##` headings, index, retrieve, cite. Heading-aligned chunking rather than fixed windows, because every section here is already a self-contained rule and a fixed window routinely severs a markdown table from its header row.

Both use title/body index fusion via reciprocal rank fusion. RRF combines by rank position rather than score, so two BM25 indices on different scales merge without normalisation.

### `app/guardrails/`

Layered, and honest about what each layer is for. Details in [RESPONSIBLE_AI.md](RESPONSIBLE_AI.md#3-guardrails).

### `app/observability/`

`tool_invocations` answers *what evidence supports this sentence*. `guardrail_events` answers *why did the assistant behave that way*. Both are written on the same session as the business data, so an audit record and the transaction it describes commit or roll back together.

---

## Data model

```
                    +------------+
                    |  Product   |   brand, category, subcategory,
                    +------------+   price_cents, rating, tags
                          | 1
                          |
                          | N
                 +-----------------+
                 | ProductVariant  |   size, colour, stock
                 +-----------------+   <-- stock lives HERE
                          | 1
                          |
              +-----------+-----------+
              | N                     | N
        +------------+         +------------+
        | OrderItem  |         | CartItem   |   session-scoped
        +------------+         +------------+
              | N
              |
              | 1
        +------------+  1     N  +--------------+
        |   Order    |-----------|  OrderEvent  |   append-only timeline
        +------------+           +--------------+
              | N
              | 1
        +------------+
        |  Customer  |
        +------------+

   Audit, keyed by turn_id:
        ChatMessage      ToolInvocation      GuardrailEvent      Feedback
```

Three decisions worth defending:

**Stock lives on the variant, not the product.** "Is the Nike tee available in medium?" is a variant question. Collapsing stock onto the product would force the assistant to guess, and it would guess confidently, which is the worst failure mode available.

**Money is integer cents.** Float currency is a correctness bug waiting for a rounding boundary. `Money.of()` formats once, and the model repeats `display`.

**`OrderItem.unit_price_cents` is a snapshot, not a join.** Price is a fact at the time of sale. A later price change must not retroactively alter what a customer paid.

---

## Request and failure handling

Every request gets a correlation id, either from an inbound `X-Correlation-Id` or newly generated, carried in a `ContextVar` so every log line and audit record produced while handling it is stamped with the same id. A customer report of "the assistant said something odd at 14:32" is traceable to the exact tool calls behind it.

Failure is bounded at three levels:

- **Tool level.** Exceptions are caught in `execute_tool` and converted to a typed error. A defect in one tool cannot abandon the customer mid-conversation.
- **Loop level.** Iteration and tool-call budgets. When exhausted, the model is asked for a final answer with tools removed, so it must conclude from what it gathered rather than starting another round.
- **Provider level.** Retries are classified: transient statuses back off with jitter and honour the provider's own wait hint; 401 and malformed requests fail immediately, because retrying them only wastes the customer's time.

If the model is unreachable entirely, the turn ends with an honest message offering a human agent. It never ends with an invented answer.
