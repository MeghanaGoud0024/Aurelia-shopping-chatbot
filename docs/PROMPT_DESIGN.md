# Prompt Design, Tool Contract, and AI Interaction Improvements

Required deliverable. **The short note is section 0 below; sections 1 to 7 are the supporting detail** for anyone who wants the reasoning or the measurements.

---

## 0. The short note

**Prompt design.** Behavioural guidance is split by concern: *policy* lives in the system prompt (803 tokens), *how to fill a parameter* lives in the tool schema beside that parameter, and *what is true right now* comes only from tool results. Putting parameter guidance in the system prompt is the common expensive mistake, because it bloats a prompt re-sent on every call and separates an instruction from the thing it governs.

The system prompt states rules as actions, not abstractions: "call `search_products` before describing any product" rather than "do not hallucinate products". It injects exactly three dynamic facts - customer name, today's date, cart empty or not - because every fact placed in a prompt can go stale mid-conversation, whereas a tool result is true when read.

**Tool usage.** 17 tools, six of which mutate state. Schema descriptions do more prompt engineering than the system prompt does: they state units ("Dollars, not cents. 'under $50' is 50"), say what to do when a value is unknown ("Only if stated; never infer"), and name the prerequisite tool when an id is missing. Identity is deliberately *not* a parameter - `customer_id` is injected server-side, so the model has no vocabulary for requesting another customer's data. Failures return typed errors with a `recovery_hint` rather than raising, and ambiguity is treated as a failure too, which is what produces "which colour would you like?" instead of a silently chosen colour.

**The strongest technique used.** Facts the model could get wrong arrive pre-computed: `price.display` is already `"$26.99"`, and order tools return a finished delivery sentence. The model repeats rather than derives. It cannot get date arithmetic wrong if it never does date arithmetic - that converts a class of hallucination into one that is structurally impossible, which is far stronger than an instruction.

**AI interaction improvements implemented**, each from an observed failure:

| Observed | Fix |
| --- | --- |
| Products formatted as markdown tables duplicating the cards | Named the fields the cards carry, banned the format, gave a wrong/right example |
| Model quoted the internal field name `delivery_message` | Stopped naming fields; "state its dates in your own words" |
| Em dashes and curly quotes despite instruction | Moved enforcement into a `str.translate` in the output guard |
| A correct "we don't stock Gucci" answer was blocked as ungrounded | Added `list_brands` to the supporting tool set; made refusal messages rule-specific |
| Turns cost ~7,600 tokens against an 8,000/min cap | Intent-based tool routing plus prompt trimming: ~3,500 tokens, 11-42s down to ~1.3s |
| Model could read the checkout token from a tool result | `model_redacted_fields` strips bearer credentials from the model's copy |
| A plausible retrieval synonym fix | Measured against a gold set, found neutral, **reverted** |

---

## 1. The division of labour

The single most consequential prompt-design decision was **where to put behavioural guidance**.

| Concern | Lives in | Why |
| --- | --- | --- |
| *What is this assistant allowed to do* | System prompt | Policy, and it applies to every turn |
| *How do I fill this parameter* | Tool schema description | Guidance belongs next to the thing it describes |
| *What is true right now* | Tool results | Facts go stale; a tool result is true when read |

Putting parameter guidance into the system prompt is a common and expensive mistake. It bloats a prompt that is re-sent on every call, it separates the instruction from the parameter it governs, and it competes for attention with the policy rules that genuinely need to be there. The system prompt here is 803 tokens. The seventeen tool schemas carry the rest, and only the relevant ones are sent.

### Three dynamic facts, and no more

`build_system_prompt` injects exactly three things: the customer's name, today's date, and whether the cart is empty.

Nothing else. Every fact placed in a prompt is a fact that can go stale mid-conversation, whereas a tool result is true at the moment it is read. Injecting the cart contents would mean a cart modified during a turn leaves the model reasoning from a stale copy.

---

## 2. How the system prompt is written

The full text is in [`app/agent/prompts.py`](../app/agent/prompts.py). Four principles shape it.

### Behaviour, not prohibition

"Do not hallucinate products" names an abstraction the model has to interpret. "Call `search_products` before describing any product" names an action it can either take or not take. Every hard rule is phrased as something to do:

```
- Describing, recommending or pricing a product: call `search_products` or
  `get_product_details` first.
- Saying something is available in a size: call `check_availability` first.
  Stock is per size and colour, so "in stock" does not mean in stock in their size.
- Never calculate a delivery date. Order tools return a ready-written delivery
  sentence; state its dates and carrier in your own words.
```

The parenthetical on stock is doing real work. Without it the model reads `in_stock: true` on a product and reports the size as available, because that is the ordinary reading of the field.

### Make the grounded path the easy path

Tools return values that are already correct and already formatted:

- `price.display` is `"$26.99"`, not `2699`
- `delivery_message` is `"Estimated delivery tomorrow, on Sunday 30 August 2026 with MetroCourier."`
- `status_label` is `"Out for delivery"`, not `"out_for_delivery"`

The model is asked to repeat these rather than derive them. It cannot get date arithmetic wrong if it never does date arithmetic. This converts a class of hallucination into a class that is structurally impossible, which is a far stronger guarantee than an instruction.

### Give the refusal a script

An assistant that refuses awkwardly is worse than one that refuses cleanly, so the out-of-scope response is written out verbatim in the prompt rather than left to improvisation. It states the boundary in one sentence and immediately offers something useful, which is what a good salesperson does when asked something they cannot answer.

### Say what the interface already shows

```
Product, order and quote cards are already rendered beside your reply with name,
price, colours, sizes and stock. Never repeat those attributes in prose, and
never use a bullet list, numbered list or table of products.
  Wrong: "1. Nike Trail Tee - $26.99, Cobalt Blue or Crimson, XS-XXL 2. ..."
  Right: "Three Nike tees are in stock. The Trail is the cheapest and comes in
  two bold colours; the Core Athletic is the most technical if you train in it."
```

The wrong/right pair was added after observing the failure. See the iteration log below.

---

## 3. The tool contract

Seventeen tools. Six mutate state and are logged at a higher level.

| Group | Tools |
| --- | --- |
| Catalogue | `search_products`, `get_product_details`, `check_availability`, `list_brands`, `list_categories` |
| Orders | `get_order_status`, `track_shipment`, `list_my_orders`, `cancel_order`\*, `request_return`\* |
| Cart | `add_to_cart`\*, `view_cart`, `update_cart_quantity`\*, `remove_from_cart`\* |
| Checkout | `prepare_checkout`, `place_order`\* |
| Knowledge | `lookup_policy` |

\* mutating

### Schema descriptions are prompt engineering

A model fills arguments by reading these strings, so they are written as instructions to a model rather than documentation for a human. Three things every description tries to do:

**State units.** `"Dollars, not cents. 'under $50' is 50."` Without this, models pass 5000 for "under $50" often enough to matter.

**Say what to do when a value is unknown.** `"XS-XXL or US shoe 6-12. Only if stated; never infer."` Otherwise a model asked for "a t-shirt" helpfully guesses medium, and the customer is shown stock for a size they never mentioned.

**Name the prerequisite.** `"Call search_products first to obtain a product_id."` A model missing an id will either invent one or give up; telling it where to get one produces a recovery.

**Explain the mechanism when it changes behaviour.** `search_products` says: *"Parameters filter, `query` only ranks."* That one sentence is why "Nike t-shirts under $40" comes back as `brand="Nike", subcategory="T-Shirt", max_price=40` rather than everything crammed into `query`.

### Identity is not a parameter

`ToolContext` carries `session`, `session_id`, `customer_id` and `customer_name`. None of these appear in any tool schema. If `customer_id` were an argument the model could fill, persuading the model to change it would be a complete authorisation bypass. Because it is injected from the authenticated session, *the model has no way to express "somebody else's orders" at all*. There is a test asserting no schema exposes an identity parameter, so a future tool cannot quietly reintroduce one.

### Errors are data, not exceptions

```python
ToolError(
    error="Colour is ambiguous. Available: Cobalt Blue, Crimson.",
    code="NEEDS_COLOR",
    recovery_hint="Ask the customer which colour they want, then call add_to_cart again.",
)
```

A raised exception gives the model nothing to reason about. A typed error with a recovery hint lets it fix the call or explain the limitation in the same turn. Ambiguity is treated as a failure too, which is what produces "which colour would you like?" instead of a silently chosen colour.

---

## 4. Tool routing

Tool schemas are re-sent on every call in the agent loop, so their size multiplies by the number of iterations. All seventeen cost ~2,800 tokens; with the system prompt that is a ~3,800 token floor per call and ~7,600 per turn.

The development provider tier allows **8,000 tokens per minute**. That is one turn per minute, which is not a usable product.

[`app/agent/routing.py`](../app/agent/routing.py) matches the message to tool groups and sends only those. Measured effect:

| Turn shape | Before | After |
| --- | --- | --- |
| Order status question | 17 tools, ~3,800 tok/call | 6 tools, ~1,700 tok/call |
| Product browse | 17 tools, ~3,800 tok/call | 9 tools, ~2,100 tok/call |
| Full turn (2 calls) | ~7,600 tokens | ~3,500 tokens |
| Observed latency | 11 to 42 seconds | ~1.3 seconds |

Three properties keep it safe:

1. **Groups, not individual tools.** Anything order-shaped gets the whole order group, so the model still chooses freely within the right domain.
2. **Monotonic widening within a turn.** Once a tool from a group has been called, that group stays available for every later iteration. Searching unlocks cart and checkout, because that is where the conversation goes next.
3. **Ambiguity opens up rather than narrowing.** No clear signal gets the browsing default; several signals get all of them.

Routing is a token optimisation and **never a security control**. It never removes a capability the customer is entitled to, only defers loading it. `AURELIA_TOOL_ROUTING_ENABLED=false` sends everything, which is the right setting on a tier with headroom.

---

## 5. Model parameters

| Parameter | Value | Reasoning |
| --- | --- | --- |
| `temperature` | 0.15 | Tool selection should be stable. Near-zero temperature also makes failures reproducible, which matters more than varied phrasing for a support assistant. |
| `reasoning_effort` | `low` | Measured at ~8 reasoning tokens versus ~49 on `high`, with no observed difference in tool choice. Tool selection here is not a hard planning problem. |
| `max_tokens` | 1600 | Generous enough that a reply is never truncated mid-sentence. Truncation is detected via `finish_reason` and labelled rather than shown silently. |
| `max_tool_iterations` | 6 | Enough for search, then availability check, then cart add, with slack. An unbounded agent loop is a production incident waiting for a trigger. |
| `HISTORY_TURNS` | 12 | Deep enough to resolve "the second one" or "that jacket". Only user and assistant prose is replayed, never tool results, so a stale price from three turns ago cannot be treated as current evidence. |

---

## 6. Iteration log: what changed and why

Each of these was a behaviour observed in testing, followed by a specific fix.

### The model formatted every product as a markdown table

**Observed.** Asked "What Nike t-shirts are available?", the model produced a four-column markdown table duplicating the product cards rendered beside it.

**Why it happened.** "Do not restate every attribute" describes a category. The model does not know which attributes the interface shows.

**Fix.** Named the exact fields the cards carry, banned the specific formats, and gave a wrong/right example. Tables stopped, and replies became comparative prose that adds information the cards do not: which one is cheapest, which is most technical.

### The model quoted internal field names

**Observed.** *"The delivery message says: 'Estimated delivery tomorrow...'"*

**Why it happened.** The prompt said to use the `delivery_message` field, and the model took that literally, quoting it as an attributed quotation.

**Fix.** Stopped naming the field. The instruction became "Order tools return a ready-written delivery sentence; state its dates and carrier in your own words. Never name a field or quote a tool result as a quotation." Output became *"Your order #1234 is on its way and is estimated to arrive tomorrow, Sunday 30 August 2026 with MetroCourier."*

### Typographic punctuation kept appearing

**Observed.** Em dashes, curly quotes and non-breaking hyphens, despite an explicit instruction.

**Why it happened.** The model mostly complied. "Mostly" is not a guarantee.

**Fix.** Moved enforcement out of the prompt and into a ten-line `str.translate` in the output guard. The instruction stays, because it reduces the work, but correctness no longer depends on it. **A style rule that can be enforced deterministically should be.**

### A correct refusal was replaced by the grounding guard

**Observed.** "Do you sell Gucci handbags?" produced a correct answer - we do not stock Gucci, here is what we do carry - which the grounding check replaced with "I don't want to give you a number I haven't verified."

**Why it happened.** Two bugs. The reply contained "we have", matching the stock-claim pattern, but `list_brands` was missing from that rule's supporting tools. And the replacement message mentioned a number when no number had been asked for.

**Fix.** Added `list_brands` and `list_categories` to the supporting set, and made the replacement message specific to the rule that fired. Both are covered by regression tests.

### Rate limits produced 42-second turns and then failures

**Observed.** Turns two and three of a session failed with HTTP 429.

**Why it happened.** Fixed exponential backoff, against a token-per-minute window that refills on a schedule the provider was actually telling us about in the error body.

**Fix.** Parse the wait hint from `Retry-After`, the `x-ratelimit-reset-tokens` header, or the text of the error body, and honour it. Combined with tool routing, turns went from 11-42 seconds to ~1.3 seconds.

### The model could read the confirmation token it was never meant to hold

**Observed.** Late in the build, checking a claim already written in the documentation - "the confirmation token travels server to browser to server and is never reproduced by the model" - showed the claim was false. `prepare_checkout` returns the token in its result, and tool results are serialised into the model's context. The model could read it and call `place_order` with it directly, completing a purchase with no human confirmation.

**Why it happened.** The two-phase design was correct. The leak was in the *transport*: a tool result is not just a value returned to code, it becomes conversation history. Anything in it is readable, and replayable, on any later call in the turn.

This is the general lesson, and it is easy to miss: **a tool result is not private.** Any capability expressed as a value in a tool result is a capability the model holds.

**Fix.** `Tool` gained `model_redacted_fields`. `prepare_checkout` declares `confirmation_token`, and the orchestrator strips it from the model's copy while `artifacts` keeps the original for the browser. The audit trail is redacted too, since a live bearer token sitting in a queryable table is a credential at rest. The `place_order` description and the system prompt were rewritten to state plainly that the token is withheld and the customer's button press completes the purchase.

**Verified.** Asked to "just buy it now, place the order immediately", the assistant prepares the quote, explains that Confirm must be pressed, and does not call `place_order`. Three regression tests cover it, including one that sweeps real tool output for credential-shaped fields so a future tool cannot silently reintroduce the leak.

### A retrieval synonym fix that measured as neutral, and was reverted

**Observed.** "How long do I have to return something?" ranked the returns *procedure* first, while the passage stating the 30-day window ranked third. The LLM path answered correctly because it receives three passages; the rule-based fallback, which showed only the top one, answered the wrong question.

**Hypothesis.** Duration questions share no vocabulary with the passages that answer them: "how long" against "within 30 calendar days". Adding duration synonyms should be a general improvement, not a fix for one query.

**Measured.** Against a ten-query gold set, three variants: baseline 5/10 rank-one and 9/10 top-three; with the synonyms, still 5/10 and 9/10, with the errors merely relocated; a narrower variant reached 6/10 rank-one but dropped top-three to 8/10.

**Fix.** Reverted the synonyms and changed the fallback to render all three retrieved passages instead. The gold set is now a regression test, so the next retrieval change has a number to beat rather than an intuition to satisfy.

Worth recording as an iteration precisely because it did not work. The plausible fix measured as neutral, and keeping it would have added permanent vocabulary complexity for nothing.

### The offline planner assigned every women's query to menswear

**Observed.** "women's hoodies on sale" filtered to `gender=men`.

**Why it happened.** `"men's" in "women's hoodies"` is `True`. A substring check, not a word-boundary check.

**Fix.** Word-boundary regex, women tested first. This one is worth recording because it is not an AI bug at all - it is the kind of ordinary defect that hides comfortably inside an AI system, where a wrong answer looks like model behaviour rather than a broken conditional.

---

## 7. What the language model is actually contributing

Because [`app/agent/fallback.py`](../app/agent/fallback.py) answers the same questions deterministically, the comparison is concrete rather than asserted.

| | Rule-based planner | LLM |
| --- | --- | --- |
| "What Nike t-shirts are available?" | Correct: keyword match to brand and subcategory | Correct |
| "Show me something warm for a trip to Berlin in December" | Fails: no keyword maps to a category | Correct: infers outerwear |
| "Add the navy one in medium" | Fails: no referent for "the navy one" | Correct: resolves from prior turn |
| "Can I cancel after it ships?" | Correct topic, returns raw policy text | Correct, and applies it to *this* order's status |
| Reply quality | Templated lists | Comparative prose |
| Handles typos and grammar | Poorly | Well |

The model earns its place on intent understanding, anaphora resolution, and synthesis across several tool results. It earns nothing on correctness, because correctness never passes through it.
