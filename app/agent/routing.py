"""Intent-based tool routing.

The problem this solves
-----------------------
Tool schemas are re-sent on every call in the agent loop, so their size is
multiplied by the number of iterations in a turn. All seventeen schemas cost
roughly 2,800 tokens; with the system prompt that is a ~3,800 token floor per
call, and a two-call turn spends ~7,600 tokens before the customer's message is
even counted. On a provider tier limited to 8,000 tokens per minute that is one
turn per minute, which is not a usable product.

Routing narrows the toolset to the groups the message plausibly needs. A typical
order question then carries five schemas instead of seventeen, cutting the
per-call floor by roughly half.

Why it is safe
--------------
The obvious risk is routing away a tool the model actually needed. Three things
contain that:

1. **Groups, not individual tools.** Anything order-shaped gets the whole order
   group, so the model still chooses freely within the right domain.
2. **Monotonic expansion within a turn.** Once a tool from a group has been
   called, that group stays available for every later iteration. Searching for a
   product unlocks cart and checkout, because that is where the conversation
   goes next.
3. **Ambiguity opens up, it does not narrow down.** A message with no clear
   signal gets the browsing default rather than a guess, and a message matching
   several groups gets all of them.

Routing is a token-budget optimisation, not a security control. It never removes
a capability the customer is entitled to, only defers loading it. Setting
`AURELIA_TOOL_ROUTING_ENABLED=false` sends every tool on every call, which is
the right setting on a provider tier with headroom.
"""

from __future__ import annotations

import logging
import re

from app.agent.tools import TOOLS_BY_NAME

logger = logging.getLogger(__name__)

#: Tools grouped by the customer intent they serve.
TOOL_GROUPS: dict[str, tuple[str, ...]] = {
    "catalog": (
        "search_products", "get_product_details", "check_availability",
        "list_brands", "list_categories",
    ),
    "orders": (
        "get_order_status", "track_shipment", "list_my_orders",
        "cancel_order", "request_return",
    ),
    "cart": ("add_to_cart", "view_cart", "update_cart_quantity", "remove_from_cart"),
    "checkout": ("prepare_checkout", "place_order", "view_cart"),
    "policy": ("lookup_policy",),
}

#: Group -> group. Selecting the key makes the value available too, because the
#: conversation reliably moves that way: you browse, then you add to cart; you
#: add to cart, then you check out.
GROUP_IMPLIES: dict[str, tuple[str, ...]] = {
    "catalog": ("cart",),
    "cart": ("checkout", "catalog"),
    "checkout": ("cart",),
    "orders": ("policy",),
}

#: Signals that activate a group. Ordered by how strongly they discriminate.
GROUP_SIGNALS: dict[str, re.Pattern[str]] = {
    "orders": re.compile(
        r"\b(?:my order|order\s*#?\s*\d+|order number|track|tracking|parcel|package|"
        r"shipment|delivered|delivery|arrive|arriving|dispatch|courier|cancel(?:led)?|"
        r"return(?:ing|ed)?|refund|where is|when will|my purchase|bought)\b", re.I),
    "cart": re.compile(
        r"\b(?:cart|basket|bag|add (?:it|this|that|to)|remove|quantity|how many.*cart)\b", re.I),
    "checkout": re.compile(
        r"\b(?:check ?out|buy|purchase|order (?:it|this|these|them)|pay|payment|"
        r"place (?:the |my )?order|i'?ll take|confirm)\b", re.I),
    "policy": re.compile(
        r"\b(?:policy|policies|can i (?:return|cancel|exchange)|how long|return window|"
        r"warranty|guarantee|refund(?:ed)? (?:take|policy)|shipping cost|free shipping|"
        r"size chart|sizing|runs (?:small|large)|privacy|my data|terms)\b", re.I),
    "catalog": re.compile(
        r"\b(?:show|find|search|looking for|do you (?:have|sell|stock)|what.*available|"
        r"price|cost|cheap|expensive|brand|colou?r|size|stock|available|recommend|"
        r"suggest|shirt|tee|jean|shoe|jacket|hoodie|dress|cap|watch|bag|sunglasses)\b", re.I),
}

#: Used when a message carries no recognisable signal at all. Browsing and
#: policy questions are the most common cold-open, and both are read-only.
DEFAULT_GROUPS: tuple[str, ...] = ("catalog", "policy")


def _groups_for_called_tools(called: set[str]) -> set[str]:
    return {
        group
        for group, members in TOOL_GROUPS.items()
        if called & set(members)
    }


def select_groups(message: str, already_called: set[str] | None = None) -> set[str]:
    """Choose which tool groups to expose for this call."""
    selected = {
        group for group, pattern in GROUP_SIGNALS.items() if pattern.search(message)
    }
    if not selected:
        selected = set(DEFAULT_GROUPS)

    # Anything already used this turn stays available, along with what it implies.
    selected |= _groups_for_called_tools(already_called or set())

    for group in list(selected):
        selected.update(GROUP_IMPLIES.get(group, ()))

    return selected


def select_tool_names(message: str, already_called: set[str] | None = None) -> list[str]:
    """Resolve the selected groups to a stable, de-duplicated tool name list."""
    names: list[str] = []
    seen: set[str] = set()
    for group in sorted(select_groups(message, already_called)):
        for name in TOOL_GROUPS[group]:
            if name not in seen and name in TOOLS_BY_NAME:
                names.append(name)
                seen.add(name)
    return names


def select_tool_schemas(
    message: str, already_called: set[str] | None = None
) -> list[dict]:
    names = select_tool_names(message, already_called)
    schemas = [TOOLS_BY_NAME[name].schema() for name in names]
    logger.debug(
        "routing.selected",
        extra={"tool_count": len(names), "tools": names},
    )
    return schemas
