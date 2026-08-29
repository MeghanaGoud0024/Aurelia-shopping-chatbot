# Scaling: Larger Datasets, More Users, Stricter Governance

The brief asks for a clear explanation of how this design would scale, even though the POC does not need to. This document names what breaks first, at what point, and what replaces it.

The general shape of the answer is that **the layering holds and the implementations change**. Nothing below requires re-architecting the request path, because every component that would need replacing is already behind an interface that isolates it.

---

## 1. What breaks first

In the order it would actually happen.

| # | Limit | Breaks at roughly | Symptom | Fix |
| --- | --- | --- | --- | --- |
| 1 | Provider tokens per minute | 2-3 concurrent users on a free tier | HTTP 429, slow turns | Paid tier, then semantic caching |
| 2 | Process-local state | The second worker | Lost carts, lost context, rate limit multiplied | Redis |
| 3 | SQLite write concurrency | ~50 concurrent writers | Write lock contention | PostgreSQL |
| 4 | In-memory retrieval index | ~100k products | Slow startup, memory pressure | OpenSearch or a vector store |
| 5 | Audit table growth | ~10M turns | Slow governance queries | Partitioning and archival |
| 6 | Single-region latency | Global users | 300ms+ added latency | Regional deployment |

Note the ordering. The first thing to break is not the database or the index; it is the LLM provider quota. That is worth internalising, because engineering effort naturally goes to the database.

---

## 2. Larger datasets

### Catalogue: 1k to 10M products

**What holds.** The service layer already treats retrieval as a *ranking* step and re-reads the shortlist from SQL. Price and stock in a reply are live values, so index freshness affects which products are shown but never whether their details are correct. That property is what makes the index swappable.

**What changes.**

| Catalogue size | Retrieval |
| --- | --- |
| Under 10k | Current in-memory BM25. Sub-second rebuild. |
| 10k to 500k | SQLite FTS5 or PostgreSQL `tsvector`. Retrieval moves into the database; no rebuild step. |
| 500k+ | OpenSearch or Elasticsearch for lexical, plus a vector store (pgvector, Qdrant) for dense. Fuse with the RRF already in `bm25.py`. |

`RetrievalService` is the seam. `search_products(query, limit) -> list[ProductHit]` is the whole contract the catalogue service depends on; everything above it is unchanged.

**Index freshness.** A full rebuild at startup stops being viable past ~100k products. Replace with change-data-capture: publish product updates to a queue, consume into the index. This also fixes a limitation that exists today - a price change mid-process is invisible to the index until restart, though not to the answer, because the answer re-reads from SQL.

**Facets.** `_facets()` currently runs `GROUP BY` over the filtered set. Past ~100k rows this becomes the slowest part of a search. Search engines compute facets as part of the query; that is one of the main reasons to move.

### Orders: 420 to 100M

Already indexed on `(customer_id)` and `(order_number)`, and every query filters on `customer_id`, which is the natural partition key.

- Partition `orders`, `order_items` and `order_events` by `customer_id` hash, or by `placed_at` range if the reporting workload dominates.
- `order_events` is append-only and grows fastest. Time-partition it and archive partitions older than the dispute window to object storage.
- `_next_order_number()` uses `MAX() + 1`. Correct here because SQLite serialises writers and checkout is one transaction. At scale this becomes a database sequence or a Snowflake-style id. This is called out in the code where it lives.

### Policy corpus: 7 documents to 10,000

Heading-based chunking holds well up to a few hundred documents. Beyond that:

- Add a topic classifier ahead of retrieval so the search space is narrowed before ranking.
- Move to dense retrieval, where policy text benefits far more than catalogue text does, because customers phrase policy questions in language that shares few words with the policy.
- Version the corpus and pin an answer to the policy version in force at the time. A refund dispute is about what the policy said in March, not what it says today.

---

## 3. More users

### The immediate blocker: LLM quota

At ~3,500 tokens per turn, an 8,000 TPM tier supports roughly two turns per minute. This is the binding constraint long before any infrastructure limit.

1. **Paid tier.** Millions of tokens per minute. Solves it up to real scale.
2. **Semantic caching.** Catalogue questions repeat heavily. Cache on normalised intent plus filters rather than on the raw string, and invalidate on price and stock change. In retail support this is typically a large hit rate, because a long tail of phrasings maps onto a short list of intents.
3. **Model tiering.** Route simple intents to a small fast model and reserve the large model for multi-step reasoning. The routing logic already exists in `routing.py`; it would be extended to select a model as well as a toolset.
4. **Keep tool routing on.** Already halves per-call tokens.

### Stateless workers

Three pieces of process-local state block horizontal scaling. Each is documented in the code, and each has the same fix.

| State | Today | Replacement |
| --- | --- | --- |
| Checkout quote store (`services/cart.py`) | `dict` with TTL | Redis `SETEX`, with the same issue/consume interface |
| Rate limiter (`guardrails/input_guard.py`) | Per-process deque | Redis sorted set sliding window |
| Conversation history (`agent/orchestrator.py`) | `dict` keyed by session | Redis list, or the `chat_messages` table already being written |

Conversation history is the interesting one: the data is *already* persisted to `chat_messages` for audit. Reading history from that table instead of memory removes the state without adding a store.

Sessions become sticky-optional rather than sticky-required, which is what allows workers to be replaced during a deployment without dropping conversations.

### Database

PostgreSQL, read replicas for catalogue and order reads, primary for writes. Connection pooling via PgBouncer, since FastAPI workers each hold a pool.

The SQLAlchemy layer is already portable. The SQLite-specific pieces are the pragmas in `db/session.py` and the `MAX() + 1` allocation; both are isolated.

### Concurrency correctness

One real race exists and is already handled at POC scale: stock between quoting and confirming. `place_order` re-checks inside the write transaction. At scale that needs to become a `SELECT ... FOR UPDATE` on the variant rows, or optimistic concurrency with a version column and a retry. The check is in the right *place* already, which is the part that is expensive to move later.

Inventory reservation is the fuller answer: hold stock for the life of a checkout quote, release on expiry. The quote store's TTL is already the right lifecycle hook.

### Observed latency budget

| Stage | Now | At scale |
| --- | --- | --- |
| Guardrails, patterns | under 1ms | unchanged |
| Guardrails, classifier | ~300ms | cache benign hashes; skip for known-good sessions |
| LLM planning call | ~400ms | unchanged; provider-bound |
| Tool execution | 3-15ms | 10-50ms across a network to PostgreSQL |
| LLM composition call | ~600ms | streaming makes this feel like ~200ms |
| **Total** | **~1.3s** | **~1.5s, or ~300ms to first token with streaming** |

Streaming is the single largest perceived-latency win and does not exist yet.

---

## 4. Stricter enterprise governance

### Secrets and configuration

This repository ships a working `.env` with an API key, at the reviewer's explicit request so the application runs on clone. That is not the production shape.

- `.env` in `.gitignore`, secrets from AWS Secrets Manager, Vault or the platform's secret store, injected at runtime.
- Automatic rotation, with the client re-reading credentials rather than requiring a restart.
- Secret scanning in CI to prevent reintroduction.
- The application already reads every setting from the environment, so this is a deployment change and not a code change.

### Access control for operations

`/api/ops/*` is open in the POC. In production:

- Operator role required, enforced as a FastAPI dependency on the router.
- `/api/ops/audit/{turn_id}` returns customer conversation content and needs a stronger role plus its own access log. Reading an audit trail is itself an auditable event.
- Separate the read-only metrics role from the audit-read role.

### Data residency and retention

The privacy policy in the corpus commits to 90-day transcripts and 24-month audit records. Enforcing that needs:

- A scheduled retention job, deleting transcripts past the window and de-identifying audit rows required for financial dispute resolution.
- Regional database deployment for residency requirements, with `customer_id` already the natural sharding key.
- Right-to-erasure implemented as a real operation across `chat_messages`, `feedback` and the de-identifiable parts of `tool_invocations`.

### Model governance

- **Pin model versions.** `AURELIA_LLM_MODEL` accepts a version already. A silent provider-side model update is an unreviewed change to a production system.
- **Evaluate before promotion.** The regression set from [DESIGN_RATIONALE.md](DESIGN_RATIONALE.md) becomes a gate: no model or prompt reaches production without passing it.
- **Prompt versioning.** Stamp a prompt version on every turn in `chat_messages`, so a behaviour change can be attributed to a specific prompt revision.
- **Model cards and DPIA.** Record intended use, evaluation results, known limitations and the human-escalation path. The material in [ACCURACY_AND_LIMITATIONS.md](ACCURACY_AND_LIMITATIONS.md) is most of the content.

### Compliance posture

| Requirement | Already present | Still needed |
| --- | --- | --- |
| Auditability of automated decisions | Full tool trace per turn, with correlation ids | Immutable storage, retention enforcement |
| Explainability to the customer | Trace in the interface, citations on policy | Plain-language summary on request |
| Human oversight | Escalation paths defined | Staffed review queue, SLA |
| Data minimisation | Explicit response models, PII redaction | Formal data map, DPIA |
| Non-discrimination | No demographic inputs to ranking | Bias testing once personalisation exists |
| Right to explanation | Audit endpoint per turn | Customer-facing export |

The point of the table is that the expensive half - the evidence trail - exists. What remains is process and access control around it, which is the half that can be added later without re-architecting.

### The improvement loop

Feedback is captured and joinable to the tool trace. The full loop adds:

1. **Triage.** Route every `not_helpful` rating and every ungrounded-claim block into a review queue.
2. **Label.** An operator sees the reply beside its trace and labels a root cause: retrieval miss, wrong tool, prompt gap, genuine catalogue gap.
3. **Act.** Retrieval misses become synonym entries or embedding improvements. Wrong tool choices become schema description edits. Prompt gaps become prompt changes, gated by the regression set. Catalogue gaps go to merchandising, which is a business signal the AI system is uniquely placed to surface.
4. **Measure.** Track helpful rate, ungrounded-block rate and escalation rate as product metrics per release.

Step 3 is where most of the value is, and it is only possible because the rating joins back to the exact evidence.

---

## 5. What would not change

Worth stating, because it is the return on the design decisions.

- **Authorisation stays a SQL predicate.** It scales because the database scales.
- **The tool contract is stable.** The seventeen tools describe the business domain, not the implementation. Swapping SQLite for PostgreSQL, or BM25 for OpenSearch, changes no schema the model sees.
- **The audit trail shape is right.** More rows, same columns.
- **The grounding check is O(1) in catalogue size.** It is a regex pass over a reply and a set intersection.
- **The two-phase checkout guarantee is unchanged.** A server-issued single-use token bound to a basket fingerprint works identically behind Redis.
- **The layering holds.** Guardrails, orchestration, tools, services, data. Every scaling change above lands inside one layer.
