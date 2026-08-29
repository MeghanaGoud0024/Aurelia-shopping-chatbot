# Accuracy and Limitations

Required deliverable. The purpose of this document is to be useful to whoever operates this system, which means being specific about what can go wrong rather than reassuring about what cannot.

---

## 1. Hallucination risk, by class

The honest framing is not "does it hallucinate" but "which hallucinations are structurally impossible, which are caught, and which remain".

### Eliminated structurally

These cannot occur because the model is never in a position to produce them.

| Risk | Why it is impossible |
| --- | --- |
| Wrong price arithmetic | Money is integer cents, formatted once in `Money.of()`. The model repeats a string. |
| Wrong delivery date | Computed in `orders._delivery_message()` from real timestamps. The model never does date arithmetic. |
| Inventing a product that does not exist | Product cards render from `search_products` output. A product not in the result set has no card. |
| Reading another customer's order | `WHERE customer_id = :id`. No code path omits it. |
| Cancelling a shipped order | `OrderStatus.is_cancellable` is checked in the service, not described to the model. |
| Charging without confirmation | `place_order` requires a server-issued single-use token bound to the session and to a fingerprint of the exact basket. The token is stripped from the model's copy of the tool result, so the model cannot hold or replay it. |
| Placing an order for out-of-stock goods | Stock is re-checked inside the write transaction. |

### Caught by the grounding check

The model answering from its own weights when the backend was never consulted. A reply making a price, stock, order-status, delivery-date or policy claim with no supporting tool call is replaced.

**Coverage is class-level.** It verifies that a *kind* of claim is backed by a *kind* of call. It does not verify that the number in the sentence matches the number in the result.

### Remaining, and how to reduce them

| # | Risk | Likelihood | Impact | Mitigation now | Planned |
| --- | --- | --- | --- | --- | --- |
| 1 | **Numeric transcription drift.** A tool returns `$26.99` and the model writes `$29.99`. | Low. Values are pre-formatted strings, so this is a copying error rather than a computation. | High. A wrong price is a consumer-law problem. | Product cards render from structured data, so the card is always right even if the prose drifts. Low temperature. | Span-level attribution, section 4. |
| 2 | **Attribute conflation across products.** Three products in one result, and a colour from the second is attached to the first. | Moderate. This is the most likely remaining error. | Moderate. Misleading but recoverable. | Prompt instructs comparative prose over per-product rundowns. Cards carry authoritative attributes. | Per-product citation markers. |
| 3 | **Over-generalising a policy passage.** Retrieved text covers the general case; the model applies it to an edge case it does not cover. | Moderate. | Moderate. A wrongly promised refund is a real cost. | Citations shown so the customer can check. Policy corpus written with explicit exception sections. | Retrieve more passages for edge-case phrasing; confidence signal on weak retrieval. |
| 4 | **Silent retrieval miss.** BM25 finds nothing for an unusual phrasing, and the assistant reports "we do not have that" for something that exists. | Moderate. This is the failure mode most likely to lose a sale. | Moderate. | Synonym table, fuzzy brand resolution, and structured-filter fallback when the keyword query matches nothing. `total_matching_filters` is always exact. | Dense retrieval as a second recall path, section 4. |
| 5 | **Stale conversational reference.** "The navy one" resolved against a product from several turns ago whose stock has since changed. | Low. | Moderate. | Only prose is replayed in history, never tool results, so the model must re-look-up rather than trust a remembered value. | Explicit entity tracking with freshness stamps. |
| 6 | **Confident tone on an uncertain answer.** The assistant sounds equally sure whether retrieval was strong or marginal. | Moderate. | Low to moderate. | Prompt requires directness about unavailability. | Surface retrieval confidence and instruct hedging below a threshold. |

---

## 2. Where the system is deliberately conservative

Each of these trades some helpfulness for correctness, on purpose.

- **Ambiguity is an error.** Adding a product available in two colours returns `NEEDS_COLOR` rather than picking one. The assistant asks. A wrong colour shipped is worse than one extra question.
- **A capped count is never reported as a total.** When the retrieval window bounds a count, the tool says so and the model must say "at least N". `total_matching_filters` is computed separately and is always exact.
- **An unstocked brand is refused, with alternatives.** "Gucci" returns an explicit "not a brand Aurelia carries" plus the stocked list, rather than a fuzzy match onto something else.
- **Missing and forbidden are indistinguishable.** Better for a customer to re-check their order number than for the assistant to become an enumeration oracle.
- **Truncated replies are labelled.** A reply cut off by the token limit is a broken answer, not a short one.

---

## 3. Known limitations

### Product and scope

- **No authentication.** A session cookie binds to a demo customer. The *shape* is correct - identity arrives from outside the model's reach and everything downstream is written against it - but there is no identity provider. Swapping in OIDC means changing `resolve_identity` and nothing else.
- **No payment processing.** `place_order` records a payment method and transitions state. No processor is contacted, no money moves.
- **Order numbers are allocated with `MAX() + 1`.** Safe because SQLite serialises writers and checkout runs in one transaction. Not safe on a database with concurrent writers, where this becomes a sequence.
- **Cart is session-scoped, not customer-scoped.** Clearing cookies loses the cart. Real carts persist against the account.
- **Single language.** English only. No localisation of currency, dates or sizing conventions.
- **No product images.** Cards are typographic. The dataset is synthetic and stock photography would have been decorative rather than informative.

### Retrieval

- **Lexical, not semantic.** BM25 with a hand-built synonym table. Reasoning is in [`app/retrieval/bm25.py`](../app/retrieval/bm25.py): at ~1,100 products, lexical matching over a controlled vocabulary of brands, categories and colours is the stronger signal, and a transformer would add a large download plus a torch dependency against the "no special infrastructure" constraint. **The consequence is real:** a query like "something smart for a wedding" has no lexical anchor and will retrieve poorly.
- **Full index rebuild on startup.** Sub-second at this size. It does not survive a catalogue two orders of magnitude larger.
- **Policy chunking follows headings.** Good for this corpus, where every section is a self-contained rule. A document with long flowing prose under one heading would produce chunks too large to retrieve precisely.
- **Measured policy retrieval quality: 9/10 top-three, 5/10 rank-one** on a ten-query gold set, pinned as a regression test in `tests/test_retrieval_and_catalog.py`. Top-three is the metric that matters for the LLM path, since the model receives three passages and synthesises across them. Rank-one matters only for the rule-based fallback, which cannot synthesise and therefore renders all three.

  A hand-tuned duration-synonym expansion was tried against this set - "how long" is a support question that shares no token with "within 30 calendar days" - and **reverted**. It moved errors between queries without improving either metric, and injected a high-IDF term that acted as a topic signal rather than a duration signal. Recorded because a change that measures as neutral should be removed rather than kept on intuition.

### State that lives in process memory

Three things are process-local and would not survive horizontal scaling. Each is called out in the code where it lives:

| State | Where | Consequence with two workers |
| --- | --- | --- |
| Checkout quote store | `services/cart.py` | A token issued by worker A is unknown to worker B |
| Rate limiter | `guardrails/input_guard.py` | Effective limit multiplies by worker count |
| Conversation history | `agent/orchestrator.py` | Context lost when a turn lands on a different worker |

All three have the same fix - Redis behind the same interface - covered in [SCALING.md](SCALING.md).

### Guardrails

- **Injection defence is probabilistic.** Prompt Guard 2 has a false-negative rate, and the pattern list only knows the phrasings we anticipated. This is precisely why authorisation is not built on them.
- **PII detection is regex-based.** Precise on structured identifiers - cards, emails, IBANs, keys - and it makes no attempt at unstructured PII such as a name or address in free text.
- **Grounding is class-level.** Stated above and in section 4.
- **No jailbreak red-team programme.** The suite covers known classes. That is not a substitute for a dedicated adversarial exercise before an enterprise launch.

### Model dependency

- **Tool-calling quality varies by model.** Developed against `gpt-oss-120b`. A weaker model will select tools less reliably. The grounding check limits the damage but does not eliminate it.
- **Free-tier rate limits shape the design.** 8,000 tokens per minute drove tool routing and prompt trimming. Both are net improvements, but the constraint is why they exist.
- **No streaming.** Replies arrive complete. At ~1.3 seconds this is acceptable; at 5+ seconds it would not be.

---

## 4. What I would do next, in priority order

**1. Span-level attribution.** Every factual span in a reply linked to the tool result field it came from. Implementation: have the model emit lightweight markers, then verify each marker's value against the recorded tool result before display, and repair or strike unverifiable spans. This closes risks 1 and 2 and turns the trace from *which calls ran* into *which call produced this exact phrase*. This is the single highest-value improvement available.

**2. Hybrid dense retrieval.** Add embeddings as a second recall path fused with BM25 via the existing RRF, so the lexical path keeps its precision on brands and attributes while the dense path catches "something smart for a wedding". `RetrievalService` already isolates this behind one class.

**3. A regression evaluation set.** Roughly 150 question-and-expected-tool-call pairs, run in CI, asserting the right tools are chosen and the grounding check does not fire. Today a prompt change is validated by hand; that does not scale past one engineer.

**4. Retrieval confidence surfaced to the model.** Pass a calibrated score so the assistant can hedge on marginal retrieval instead of sounding equally certain either way. Closes risk 6.

**5. Shared state in Redis.** Quote store, rate limiter and conversation history behind the interfaces they already have. Prerequisite for more than one worker.

**6. Streaming responses.** Server-sent events, with tool-call steps streamed into the trace panel as they happen. Turns a 3-second wait into 300ms to first token.

**7. Real authentication and an operator role.** OIDC on `resolve_identity`; put `/api/ops/*` behind a role rather than leaving it open.
