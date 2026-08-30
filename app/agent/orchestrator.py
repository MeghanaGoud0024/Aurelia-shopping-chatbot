"""The agent loop.

Shape of a turn
---------------

    inbound guardrails
        -> LLM plans
        -> tools execute against the database
        -> LLM writes the answer from the tool results
        -> outbound guardrails
        -> response + trace

The loop is bounded on three axes, because an unbounded agent loop is a
production incident waiting for a trigger: iterations, total tool calls, and the
LLM's own timeout and retry budget. When a bound is hit the turn ends with an
honest message rather than silently truncating.

Why the loop is written out rather than delegated to a framework: at seventeen
tools and a fixed conversational shape, an agent framework would add a
dependency and a layer of indirection while removing the two things this
codebase most needs to demonstrate - exactly where authorisation is injected,
and exactly what is recorded for audit. Both live in this file, in plain sight.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.agent.fallback import plan_with_pending
from app.agent.llm import LLMError, LLMResponse, llm_client
from app.agent.mode import assistant_mode
from app.agent.prompts import build_system_prompt
from app.agent.routing import select_tool_schemas
from app.agent.tools import (
    MUTATING_TOOLS, TOOLS_BY_NAME, ToolContext, execute_tool, redact_for_model,
    tool_schemas,
)
from app.config import settings
from app.guardrails.input_guard import screen_input
from app.guardrails.output_guard import screen_output
from app.logging_setup import correlation_id
from app.observability.trace import (
    record_guardrail_event, record_message, record_tool_invocation,
)
from app.schemas import ChatResponse, TraceStep
from app.services import cart as cart_service

logger = logging.getLogger(__name__)

#: How many prior messages of a conversation are replayed to the model. Deep
#: enough to resolve "the second one" or "that jacket", shallow enough that the
#: prompt does not grow without bound over a long session.
HISTORY_TURNS = 12

FAILURE_MESSAGE = (
    "I'm having trouble reaching our systems right now, so I'd rather not guess. "
    "Please try again in a moment, or I can pass you to a human agent."
)


@dataclass(slots=True)
class TurnArtifacts:
    """Structured tool output the interface renders as cards.

    Kept separate from the prose reply so the frontend never has to parse the
    model's words to find a product. The model describes; this carries the data.
    """

    products: list[dict[str, Any]] = field(default_factory=list)
    orders: list[dict[str, Any]] = field(default_factory=list)
    cart: dict[str, Any] | None = None
    checkout_quote: dict[str, Any] | None = None
    citations: list[str] = field(default_factory=list)

    def absorb(self, tool_name: str, result: Any) -> None:
        """Pull renderable payloads out of a tool result."""
        tool = TOOLS_BY_NAME.get(tool_name)
        if tool is None or tool.renders is None or not isinstance(result, dict):
            return
        if result.get("code"):  # a ToolError, nothing to render
            return

        kind = tool.renders
        if kind == "products":
            if "products" in result:
                self._merge(self.products, result["products"], "product_id")
            elif "product_id" in result:
                self._merge(self.products, [result], "product_id")
        elif kind == "orders":
            if "orders" in result:
                self._merge(self.orders, result["orders"], "order_number")
            elif "order_number" in result:
                self._merge(self.orders, [result], "order_number")
        elif kind == "cart":
            self.cart = result
        elif kind == "checkout":
            self.checkout_quote = result
        elif kind == "order_placed":
            # A placed order invalidates the quote card it came from.
            self.checkout_quote = None
            self.cart = None
        elif kind == "policy":
            for passage in result.get("passages", []):
                citation = passage.get("citation")
                if citation and citation not in self.citations:
                    self.citations.append(citation)

    @staticmethod
    def _merge(target: list[dict[str, Any]], incoming: list[Any], key: str) -> None:
        seen = {item.get(key) for item in target}
        for item in incoming:
            if isinstance(item, dict) and item.get(key) not in seen:
                target.append(item)
                seen.add(item.get(key))


class Orchestrator:
    """Runs one conversational turn end to end."""

    def __init__(self) -> None:
        self._history: dict[str, list[dict[str, Any]]] = {}
        #: Unfinished add_to_cart clarifications, per session. Only the
        #: rule-based planner uses this: the LLM path replays conversation
        #: history and resolves "black" against it on its own, whereas the
        #: deterministic planner classifies each message in isolation and
        #: would otherwise drop the answer to its own question. Cleared as
        #: soon as it is used or superseded, so it can never go stale.
        self._pending_clarification: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    # -- conversation memory ---------------------------------------------

    async def _get_history(self, session_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            return list(self._history.get(session_id, []))

    async def _set_history(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        async with self._lock:
            # Keep only user/assistant prose. Replaying tool call/result pairs
            # would grow the prompt quickly and, worse, let a stale price from
            # three turns ago be treated as current evidence.
            trimmed = [m for m in messages if m["role"] in {"user", "assistant"} and m.get("content")]
            self._history[session_id] = trimmed[-HISTORY_TURNS:]

    async def reset(self, session_id: str) -> None:
        async with self._lock:
            self._history.pop(session_id, None)
            # A pending clarification belongs to the conversation being
            # cleared; leaving it would let the first message of a fresh
            # conversation be read as an answer to a question nobody can see.
            self._pending_clarification.pop(session_id, None)

    # -- the turn ---------------------------------------------------------

    async def run_turn(
        self,
        *,
        session: Session,
        session_id: str,
        customer_id: int,
        customer_name: str,
        message: str,
    ) -> ChatResponse:
        turn_id = uuid.uuid4().hex[:16]
        started = time.perf_counter()
        trace: list[TraceStep] = []
        step = 0

        def add_step(**kwargs: Any) -> None:
            nonlocal step
            step += 1
            trace.append(TraceStep(step=step, **kwargs))

        # --- 1. Inbound guardrails --------------------------------------
        guard_started = time.perf_counter()
        decision = await screen_input(message, session_id)
        guard_ms = int((time.perf_counter() - guard_started) * 1000)

        record_guardrail_event(
            session, session_id=session_id, turn_id=turn_id, stage="input",
            rule=decision.rule, action=decision.action, score=decision.score,
            detail=decision.detail,
        )
        add_step(
            kind="guardrail",
            label="Input screening",
            detail=(
                f"{decision.rule} ({decision.action})"
                + (f", injection score {decision.score:.4f}" if decision.score > 0 else "")
                + (f", redacted: {', '.join(decision.redactions)}" if decision.redactions else "")
            ),
            status="blocked" if not decision.allowed else "ok",
            latency_ms=guard_ms,
        )
        record_message(
            session, session_id=session_id, turn_id=turn_id, role="user",
            content=decision.safe_message or message,
        )

        if not decision.allowed:
            add_step(kind="answer", label="Refused", detail=decision.rule, status="blocked")
            record_message(
                session, session_id=session_id, turn_id=turn_id,
                role="assistant", content=decision.user_message,
            )
            return ChatResponse(
                session_id=session_id, turn_id=turn_id, reply=decision.user_message,
                trace=trace, blocked=True, grounded=True,
                latency_ms=int((time.perf_counter() - started) * 1000),
                model=settings.llm_model if llm_client.available else "rule-based",
            )

        # --- 2. Plan and act --------------------------------------------
        context = ToolContext(
            session=session, session_id=session_id,
            customer_id=customer_id, customer_name=customer_name,
        )
        artifacts = TurnArtifacts()
        tools_called: set[str] = set()

        # A manually forced fallback (the toggle in the interface) and a
        # missing API key both land on the same deterministic path, but the
        # label distinguishes them: one is a capability the deployment lacks,
        # the other is an operator's live decision, and the trace should say
        # which.
        use_fallback = assistant_mode.forced_fallback or not llm_client.available
        if use_fallback:
            reply, finish_reason = self._run_without_llm(
                context=context, message=message, artifacts=artifacts,
                tools_called=tools_called, turn_id=turn_id, add_step=add_step,
            )
            model_label = (
                "rule-based planner (manual)"
                if assistant_mode.forced_fallback and llm_client.available
                else "rule-based planner"
            )
        else:
            try:
                reply, finish_reason = await self._run_with_llm(
                    context=context, message=message, artifacts=artifacts,
                    tools_called=tools_called, turn_id=turn_id, add_step=add_step,
                )
                model_label = settings.llm_model
            except LLMError as exc:
                logger.error("orchestrator.llm_failed", extra={"error": str(exc)})
                add_step(
                    kind="answer", label="Language model unavailable",
                    detail=str(exc)[:200], status="error",
                )
                record_message(
                    session, session_id=session_id, turn_id=turn_id,
                    role="assistant", content=FAILURE_MESSAGE,
                )
                return ChatResponse(
                    session_id=session_id, turn_id=turn_id, reply=FAILURE_MESSAGE,
                    trace=trace, grounded=True,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    model=settings.llm_model,
                )

        # --- 3. Outbound guardrails -------------------------------------
        output = screen_output(reply, tools_called=tools_called, finish_reason=finish_reason)
        record_guardrail_event(
            session, session_id=session_id, turn_id=turn_id, stage="output",
            rule=output.rule, action=output.action, detail=output.detail,
        )
        if output.rule != "clean":
            add_step(
                kind="guardrail", label="Output screening",
                detail=output.detail or output.rule,
                status="blocked" if output.action == "block" else "warn",
            )

        add_step(
            kind="answer",
            label="Answer composed",
            detail=(
                f"grounded in {_plural(len(tools_called), 'backend call')}: "
                f"{', '.join(sorted(tools_called))}"
                if tools_called else "no backend call was needed for this turn"
            ),
            status="ok" if output.grounded else "ungrounded",
        )

        record_message(
            session, session_id=session_id, turn_id=turn_id,
            role="assistant", content=output.text,
        )
        history = await self._get_history(session_id)
        history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": output.text},
        ])
        await self._set_history(session_id, history)

        return ChatResponse(
            session_id=session_id,
            turn_id=turn_id,
            reply=output.text,
            trace=trace,
            products=artifacts.products,
            orders=artifacts.orders,
            cart=artifacts.cart,
            checkout_quote=artifacts.checkout_quote,
            citations=artifacts.citations,
            blocked=not output.allowed,
            grounded=output.grounded,
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=model_label,
        )

    # -- LLM path ---------------------------------------------------------

    async def _run_with_llm(
        self,
        *,
        context: ToolContext,
        message: str,
        artifacts: TurnArtifacts,
        tools_called: set[str],
        turn_id: str,
        add_step: Any,
    ) -> tuple[str, str]:
        cart_count = cart_service.view_cart(context.session, context.session_id).item_count
        history = await self._get_history(context.session_id)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(context.customer_name,
                                                              cart_item_count=cart_count)},
            *history,
            {"role": "user", "content": message},
        ]

        total_tool_calls = 0
        finish_reason = "stop"

        for iteration in range(1, settings.max_tool_iterations + 1):
            # Re-select each iteration: tools already used this turn keep their
            # group available, so the toolset can only widen as the turn unfolds.
            schemas = (
                select_tool_schemas(message, tools_called)
                if settings.tool_routing_enabled
                else tool_schemas()
            )
            if iteration == 1:
                add_step(
                    kind="reasoning",
                    label="Tool selection",
                    detail=(
                        f"{len(schemas)} of {len(TOOLS_BY_NAME)} tools offered to the model"
                        + ("" if settings.tool_routing_enabled else " (routing disabled)")
                    ),
                )
            response: LLMResponse = await llm_client.complete(messages, tools=schemas)
            finish_reason = response.finish_reason

            if response.reasoning:
                add_step(
                    kind="reasoning",
                    label=f"Planning (step {iteration})",
                    detail=response.reasoning[:600],
                    latency_ms=response.latency_ms,
                )

            if not response.wants_tools:
                return response.content, finish_reason

            messages.append({
                "role": "assistant",
                "content": response.content or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.raw_arguments or "{}"},
                    }
                    for call in response.tool_calls
                ],
            })

            for call in response.tool_calls:
                total_tool_calls += 1
                if total_tool_calls > settings.max_tool_calls_per_turn:
                    messages.append({
                        "role": "tool", "tool_call_id": call.id, "name": call.name,
                        "content": json.dumps({
                            "error": "Tool call budget for this turn is exhausted.",
                            "code": "BUDGET_EXCEEDED",
                            "recovery_hint": "Answer with what you already have.",
                        }),
                    })
                    add_step(
                        kind="tool_call", label=f"{call.name} (skipped)",
                        detail="tool call budget exhausted", tool_name=call.name,
                        status="skipped",
                    )
                    continue

                result, status, latency_ms = self._execute(
                    context=context, call_name=call.name, arguments=call.arguments,
                    turn_id=turn_id,
                )
                tools_called.add(call.name)
                artifacts.absorb(call.name, result)

                add_step(
                    kind="tool_call",
                    label=call.name,
                    tool_name=call.name,
                    arguments=call.arguments,
                    result_summary=_summarise(call.name, result),
                    status=status,
                    latency_ms=latency_ms,
                    detail="mutating call" if call.name in MUTATING_TOOLS else "",
                )
                # The model receives a redacted copy. `artifacts` above kept the
                # original, so the browser still gets the real confirmation token.
                messages.append({
                    "role": "tool", "tool_call_id": call.id, "name": call.name,
                    "content": json.dumps(
                        redact_for_model(TOOLS_BY_NAME.get(call.name), result), default=str
                    )[:12_000],
                })

        # Iteration budget exhausted. Ask for a final answer with tools removed,
        # so the model must conclude from what it already gathered rather than
        # starting another round of calls.
        add_step(
            kind="reasoning", label="Iteration budget reached",
            detail=f"stopped after {settings.max_tool_iterations} planning steps",
            status="warn",
        )
        messages.append({
            "role": "user",
            "content": (
                "Answer the customer now using only the tool results already gathered. "
                "Do not request further lookups. If something is still unknown, say so plainly."
            ),
        })
        final = await llm_client.complete(messages, tools=None)
        return final.content, final.finish_reason

    def _execute(
        self, *, context: ToolContext, call_name: str, arguments: dict[str, Any], turn_id: str
    ) -> tuple[Any, str, int]:
        started = time.perf_counter()
        result, status = execute_tool(call_name, arguments, context)
        latency_ms = int((time.perf_counter() - started) * 1000)

        error_message = ""
        if isinstance(result, dict) and result.get("code"):
            error_message = f"{result.get('code')}: {result.get('error', '')}"

        # The audit trail is also redacted. A live bearer token sitting in a
        # queryable table is a credential at rest; the quote is already
        # correlatable to its order by session and timestamp.
        record_tool_invocation(
            context.session, session_id=context.session_id, turn_id=turn_id,
            correlation_id=correlation_id.get(), tool_name=call_name,
            arguments=arguments,
            result=redact_for_model(TOOLS_BY_NAME.get(call_name), result),
            status=status, latency_ms=latency_ms, error_message=error_message,
        )
        log = logger.warning if call_name in MUTATING_TOOLS else logger.info
        log(
            "tool.invoked",
            extra={
                "tool_name": call_name, "status": status, "latency_ms": latency_ms,
                "session_id": context.session_id, "mutating": call_name in MUTATING_TOOLS,
            },
        )
        return result, status, latency_ms

    # -- offline path -----------------------------------------------------

    def _run_without_llm(
        self,
        *,
        context: ToolContext,
        message: str,
        artifacts: TurnArtifacts,
        tools_called: set[str],
        turn_id: str,
        add_step: Any,
    ) -> tuple[str, str]:
        """Deterministic planner used when no LLM key is configured.

        This is not a toy stub. It routes the message to the same tools through
        the same executor and audit path, so the application remains genuinely
        demonstrable, and reviewable, with no credentials at all. Only the
        language quality differs.
        """
        pending = self._pending_clarification.pop(context.session_id, None)
        plan = plan_with_pending(message, pending)
        add_step(
            kind="reasoning", label="Rule-based planning",
            detail=(
                f"matched intent '{plan.intent}'"
                + (" (answering a pending question)" if pending and plan.intent == "add_to_cart" else "")
            ),
            status="ok",
        )
        for call_name, arguments in plan.calls:
            result, status, latency_ms = self._execute(
                context=context, call_name=call_name, arguments=arguments, turn_id=turn_id
            )
            tools_called.add(call_name)
            artifacts.absorb(call_name, result)
            add_step(
                kind="tool_call", label=call_name, tool_name=call_name,
                arguments=arguments, result_summary=_summarise(call_name, result),
                status=status, latency_ms=latency_ms,
            )
            plan.observe(call_name, result)
            self._remember_clarification(context.session_id, call_name, arguments, result)
        return plan.render(), "stop"

    def _remember_clarification(
        self, session_id: str, call_name: str, arguments: dict[str, Any], result: Any
    ) -> None:
        """Record an unanswered size/colour question so the next turn can
        resolve a bare reply against it.

        Only NEEDS_SIZE/NEEDS_COLOR qualify: they are the one case where the
        planner asks the customer something and needs their answer to finish
        an action it has already started. Anything else leaves no pending
        state, so an ordinary error can't cause the next unrelated message to
        be misread as an answer.
        """
        if call_name != "add_to_cart" or not isinstance(result, dict):
            return
        if result.get("code") not in {"NEEDS_SIZE", "NEEDS_COLOR"}:
            return
        needs_field = result.get("needs_field")
        options = result.get("options") or []
        product_name = arguments.get("product_name")
        if not (needs_field and options and product_name):
            return
        self._pending_clarification[session_id] = {
            "product_name": product_name,
            "field": needs_field,
            "options": options,
            # Preserve an already-supplied dimension so answering the second
            # question doesn't lose the answer to the first.
            "size": arguments.get("size"),
            "color": arguments.get("color"),
        }


def _plural(count: int, singular: str, plural_form: str | None = None) -> str:
    """"1 product" / "3 products". These strings are shown to the customer."""
    return f"{count:,} {singular if count == 1 else (plural_form or singular + 's')}"


def _summarise(tool_name: str, result: Any) -> str:
    """One-line description of a tool result, for the explainability panel."""
    if not isinstance(result, dict):
        return str(result)[:180]
    if result.get("code"):
        return f"{result['code']}: {str(result.get('error', ''))[:140]}"

    if "products" in result:
        total = result.get("total_matches", 0)
        prefix = "at least " if result.get("total_matches_capped") else ""
        return f"{_plural(len(result['products']), 'product')} returned, {prefix}{total:,} matching"
    if "orders" in result:
        return f"{_plural(result.get('count', len(result['orders'])), 'order')} returned"
    if "order_number" in result and "timeline" in result:
        return f"order {result['order_number']}: {result.get('status_label') or result.get('status')}"
    if "passages" in result:
        return f"{_plural(len(result['passages']), 'policy passage')}: " + "; ".join(
            p.get("citation", "") for p in result["passages"][:2]
        )
    if "lines" in result:
        total = (result.get("total") or {}).get("display", "")
        return f"cart: {_plural(result.get('item_count', 0), 'item')}, total {total}"
    if "confirmation_token" in result:
        return f"quote issued, total {(result.get('total') or {}).get('display', '')}"
    if "matching_variants" in result:
        available = result.get("any_available")
        return f"{_plural(len(result['matching_variants']), 'variant')}, available={available}"
    if "brands" in result:
        return f"{len(result['brands'])} brands"
    if "categories" in result:
        return f"{len(result['categories'])} categories"
    if "message" in result:
        return str(result["message"])[:180]
    return json.dumps(result, default=str)[:180]


orchestrator = Orchestrator()
