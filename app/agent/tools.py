"""Tool registry.

Every transactional capability the assistant has is declared here, once, as a
`Tool` carrying both its JSON schema (what the model sees) and its executor
(what actually runs). Keeping the two together means a schema can never drift
away from the function it describes.

Schema design is prompt engineering
-----------------------------------
The parameter descriptions in this file do more work than the system prompt
does. A model chooses tools and fills arguments by reading these strings, so
they are written as instructions to the model, not as documentation for a human:
they state units ("dollars, not cents"), they say what to do when a value is
unknown ("omit rather than guessing"), and they name the tool to call first when
a prerequisite id is missing. Ambiguity here shows up as wrong tool calls.

Every executor returns a JSON-serialisable dict. Failures come back as
`ToolError` payloads rather than exceptions, so the model always has something
to reason about and can recover inside the same turn.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.schemas import ToolError
from app.services import cart as cart_service
from app.services import catalog as catalog_service
from app.services import orders as order_service

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ToolContext:
    """Everything an executor needs that does not come from the model.

    Identity lives here, never in the tool arguments. If `customer_id` were a
    parameter the model could fill in, then persuading the model to change it
    would be a complete authorisation bypass. Because it is injected from the
    authenticated session, the model has no way to express "somebody else's
    orders" at all.
    """

    session: Session
    session_id: str
    customer_id: int
    customer_name: str


@dataclass(slots=True, frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    executor: Callable[[ToolContext, dict[str, Any]], Any]
    #: Mutating tools are logged at a higher level and are candidates for
    #: human-in-the-loop review in a regulated deployment.
    mutating: bool = False
    #: Tools whose results should be surfaced as UI cards, not just prose.
    renders: str | None = None
    #: Result fields that must be stripped before the result enters the model's
    #: context. These are bearer credentials: values that grant an action simply
    #: by being presented. A tool result is serialised into the conversation, so
    #: without this the model can read a token it was never meant to hold and
    #: replay it on a later call. The browser still receives the real value
    #: through the artifact channel, which never passes through the model.
    model_redacted_fields: tuple[str, ...] = ()

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def redact_for_model(tool: Tool | None, result: Any) -> Any:
    """Strip bearer credentials from a tool result before the model sees it.

    Returns a copy; the caller keeps the original for the browser. The
    placeholder is written as an instruction rather than a blank, so the model
    understands why the field is missing and what happens instead.
    """
    if tool is None or not tool.model_redacted_fields or not isinstance(result, dict):
        return result
    safe = dict(result)
    for field in tool.model_redacted_fields:
        if field in safe:
            safe[field] = (
                "[withheld] This value is issued directly to the customer's browser. "
                "You cannot see it and must not guess it. The customer confirms by "
                "pressing the button on the quote card."
            )
    return safe


#: Fields on a product entry that exist for the UI card, not for the model's
#: reply. Every one of them is either purely cosmetic (sku), redundant with
#: something else already in the payload (this catalogue's product names
#: already say "Men's"/"Women's", so `gender` repeats it; `category` repeats
#: what `subcategory` already says more specifically), or something the model
#: is explicitly told never to use (the tool descriptions instruct it to quote
#: `price.display` verbatim, so the cents/currency behind that string cost
#: tokens without ever being read).
_PRODUCT_FIELDS_DROPPED_FOR_MODEL = frozenset({"sku", "category", "gender", "total_stock", "relevance"})


def _slim_product_for_model(product: dict[str, Any]) -> dict[str, Any]:
    slim = {k: v for k, v in product.items() if k not in _PRODUCT_FIELDS_DROPPED_FOR_MODEL}
    for money_field in ("price", "list_price"):
        value = slim.get(money_field)
        if isinstance(value, dict) and "display" in value:
            slim[money_field] = value["display"]
    return slim


def slim_for_model(tool_name: str, result: Any) -> Any:
    """Shrink a tool result before it enters the model's context.

    Distinct from `redact_for_model`: that function removes what the model
    must never see (a credential); this one removes what the model has no use
    for, purely to cut prompt size. Both run at the same call site and neither
    touches what `TurnArtifacts.absorb` keeps for the browser - the UI card
    always renders from the untouched original result.

    Product search is the one result shape big enough for this to matter: it
    can carry up to 24 products, so trimming ~40% off each entry compounds.
    `facets` is dropped entirely for the model for a more specific reason than
    size - nothing in the system prompt or any tool description tells the
    model it exists or what to do with it, so it is pure unused weight in this
    copy specifically, not a capability being removed.
    """
    if not isinstance(result, dict):
        return result
    if tool_name == "search_products" and isinstance(result.get("products"), list):
        slim = {k: v for k, v in result.items() if k != "facets"}
        slim["products"] = [_slim_product_for_model(p) for p in result["products"]]
        return slim
    if tool_name == "get_product_details" and isinstance(result.get("variants"), list):
        return _slim_product_for_model(result)
    return result


def _dump(value: Any) -> Any:
    """Normalise an executor's return value into JSON-safe data."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_dump(v) for v in value]
    if isinstance(value, dict):
        return {k: _dump(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# Argument coercion
#
# Tool arguments arrive as whatever the model chose to emit, and models differ
# sharply in how disciplined that is. A strong model omits optional parameters
# it has no value for; a smaller one - llama3.2:3b, observed directly - fills
# in every declared parameter with placeholder junk instead:
#
#   {"brand": "Adidas", "query": "", "category": "", "max_price": "0",
#    "min_rating": "0", "in_stock_only": "false", "limit": "6"}
#
# Three separate hazards in that one payload, each of which broke something:
#
#   * `"0"` is a *string*, and the service layer does `min_price * 100` -> a
#     TypeError that crashed the whole call.
#   * `max_price = 0` as a number is worse than a crash: it silently returns
#     nothing, because no product costs zero or less. A wrong-but-quiet answer
#     is the failure mode this codebase works hardest to avoid.
#   * `bool("false")` is **True** in Python. Taken literally, "false" would
#     turn a filter on rather than off.
#
# So coercion belongs here, at the boundary where untrusted model output
# becomes typed arguments, rather than being pushed onto every service
# function or assumed away. This hardens the tool layer against any weak
# model, not just the one that exposed it.
# ---------------------------------------------------------------------------

#: Strings a model might emit meaning "no value". Compared case-insensitively.
_NULLISH_STRINGS = frozenset({"", "none", "null", "nil", "n/a", "na", "undefined"})


def _opt_str(value: Any) -> str | None:
    """A trimmed string, or None when the model meant "nothing"."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _NULLISH_STRINGS else text


def _opt_number(value: Any, *, zero_means_absent: bool = False) -> float | None:
    """A float, or None when absent or unparseable.

    `zero_means_absent` is for bounds where zero carries no usable meaning:
    a minimum price or rating of 0 excludes nothing, and a *maximum* price of
    0 excludes everything. Neither is ever a real shopper intent, so a zero on
    those parameters is far more likely a model placeholder than a request -
    and treating it as absent turns a silently-empty result set into a correct
    one. Parameters where zero is meaningful (a quantity, say) must not set
    this.
    """
    text = _opt_str(value)
    if text is None:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if zero_means_absent and number == 0:
        return None
    return number


def _opt_int(value: Any, default: int) -> int:
    number = _opt_number(value)
    return default if number is None else int(number)


def _opt_int_or_none(value: Any) -> int | None:
    """An int, or None when absent - for ids, where "missing" is a real state
    the executors already branch on and must not become a crash instead."""
    number = _opt_number(value)
    return None if number is None else int(number)


def _opt_bool(value: Any, default: bool) -> bool:
    """Parse a boolean the way a model might express one.

    Guards specifically against `bool("false") is True`, which would invert
    the caller's intent rather than merely ignoring it.
    """
    if isinstance(value, bool):
        return value
    text = _opt_str(value)
    if text is None:
        return default
    lowered = text.lower()
    if lowered in {"true", "yes", "y", "1"}:
        return True
    if lowered in {"false", "no", "n", "0"}:
        return False
    return default


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------

def _search_products(ctx: ToolContext, args: dict[str, Any]) -> Any:
    return catalog_service.search_products(
        ctx.session,
        query=_opt_str(args.get("query")) or "",
        brand=_opt_str(args.get("brand")),
        category=_opt_str(args.get("category")),
        subcategory=_opt_str(args.get("subcategory")),
        gender=_opt_str(args.get("gender")),
        size=_opt_str(args.get("size")),
        color=_opt_str(args.get("color")),
        min_price=_opt_number(args.get("min_price"), zero_means_absent=True),
        max_price=_opt_number(args.get("max_price"), zero_means_absent=True),
        min_rating=_opt_number(args.get("min_rating"), zero_means_absent=True),
        in_stock_only=_opt_bool(args.get("in_stock_only"), True),
        on_sale_only=_opt_bool(args.get("on_sale_only"), False),
        sort=_opt_str(args.get("sort")) or "relevance",
        limit=_opt_int(args.get("limit"), 6),
    )


def _get_product_details(ctx: ToolContext, args: dict[str, Any]) -> Any:
    product_id = _opt_int_or_none(args.get("product_id"))
    if product_id is None:
        return ToolError(
            error="product_id is required.", code="MISSING_ARGUMENT",
            recovery_hint="Call search_products first to obtain a product_id.",
        )
    detail = catalog_service.get_product(ctx.session, int(product_id))
    if detail is None:
        return ToolError(
            error=f"No product with id {product_id} exists.", code="PRODUCT_NOT_FOUND",
            recovery_hint="Call search_products to find valid product ids.",
        )
    return detail


def _check_availability(ctx: ToolContext, args: dict[str, Any]) -> Any:
    product_id = _opt_int_or_none(args.get("product_id"))
    if product_id is None:
        return ToolError(
            error="product_id is required.", code="MISSING_ARGUMENT",
            recovery_hint="Call search_products first.",
        )
    result = catalog_service.check_availability(
        ctx.session, product_id,
        size=_opt_str(args.get("size")), color=_opt_str(args.get("color")),
    )
    if not result.get("found"):
        return ToolError(
            error=f"No product with id {product_id} exists.", code="PRODUCT_NOT_FOUND",
            recovery_hint="Call search_products to find valid product ids.",
        )
    return result


def _list_brands(ctx: ToolContext, args: dict[str, Any]) -> Any:
    return {"brands": catalog_service.list_brands(ctx.session)}


def _list_categories(ctx: ToolContext, args: dict[str, Any]) -> Any:
    return {"categories": catalog_service.list_categories(ctx.session)}


def _get_order_status(ctx: ToolContext, args: dict[str, Any]) -> Any:
    order_number = _opt_str(args.get("order_number")) or ""
    if not order_number:
        return ToolError(
            error="order_number is required.", code="MISSING_ARGUMENT",
            recovery_hint="Ask the customer for the order number, or call list_my_orders.",
        )
    return order_service.get_order_status(ctx.session, order_number, ctx.customer_id)


def _track_shipment(ctx: ToolContext, args: dict[str, Any]) -> Any:
    order_number = _opt_str(args.get("order_number")) or ""
    if not order_number:
        return ToolError(
            error="order_number is required.", code="MISSING_ARGUMENT",
            recovery_hint="Ask the customer for the order number, or call list_my_orders.",
        )
    return order_service.track_shipment(ctx.session, order_number, ctx.customer_id)


def _list_my_orders(ctx: ToolContext, args: dict[str, Any]) -> Any:
    return order_service.list_my_orders(
        ctx.session, ctx.customer_id,
        limit=_opt_int(args.get("limit"), 5), status=_opt_str(args.get("status")),
    )


def _cancel_order(ctx: ToolContext, args: dict[str, Any]) -> Any:
    order_number = _opt_str(args.get("order_number")) or ""
    if not order_number:
        return ToolError(
            error="order_number is required.", code="MISSING_ARGUMENT",
            recovery_hint="Ask which order to cancel, or call list_my_orders.",
        )
    return order_service.cancel_order(
        ctx.session, order_number, ctx.customer_id, reason=_opt_str(args.get("reason")) or ""
    )


def _request_return(ctx: ToolContext, args: dict[str, Any]) -> Any:
    order_number = _opt_str(args.get("order_number")) or ""
    if not order_number:
        return ToolError(
            error="order_number is required.", code="MISSING_ARGUMENT",
            recovery_hint="Ask which order to return, or call list_my_orders.",
        )
    return order_service.request_return(
        ctx.session, order_number, ctx.customer_id, reason=_opt_str(args.get("reason")) or ""
    )


def _add_to_cart(ctx: ToolContext, args: dict[str, Any]) -> Any:
    product_id = _opt_int_or_none(args.get("product_id"))
    product_name = _opt_str(args.get("product_name"))
    if product_id is None and product_name:
        product_id = catalog_service.find_product_id_by_name(ctx.session, product_name)
        if product_id is None:
            return ToolError(
                error=f"Nothing matching \"{product_name}\" was found in the catalogue.",
                code="PRODUCT_NOT_FOUND",
                recovery_hint="Call search_products to find the product, then use its product_id.",
            )
    return cart_service.add_to_cart(
        ctx.session, ctx.session_id,
        product_id=product_id,
        variant_id=_opt_int_or_none(args.get("variant_id")),
        size=_opt_str(args.get("size")), color=_opt_str(args.get("color")),
        quantity=_opt_int(args.get("quantity"), 1),
    )


def _view_cart(ctx: ToolContext, args: dict[str, Any]) -> Any:
    return cart_service.view_cart(ctx.session, ctx.session_id)


def _update_cart_quantity(ctx: ToolContext, args: dict[str, Any]) -> Any:
    variant_id = _opt_int_or_none(args.get("variant_id"))
    if variant_id is None:
        return ToolError(
            error="variant_id is required.", code="MISSING_ARGUMENT",
            recovery_hint="Call view_cart to get the variant_id of each line.",
        )
    # quantity 0 is meaningful here - it removes the line - so it must not be
    # coerced away as an absent value the way a zero price bound is.
    return cart_service.update_cart_quantity(
        ctx.session, ctx.session_id, variant_id, _opt_int(args.get("quantity"), 0)
    )


def _remove_from_cart(ctx: ToolContext, args: dict[str, Any]) -> Any:
    variant_id = _opt_int_or_none(args.get("variant_id"))
    if variant_id is None:
        return ToolError(
            error="variant_id is required.", code="MISSING_ARGUMENT",
            recovery_hint="Call view_cart to get the variant_id of each line.",
        )
    return cart_service.remove_from_cart(ctx.session, ctx.session_id, variant_id)


def _prepare_checkout(ctx: ToolContext, args: dict[str, Any]) -> Any:
    return cart_service.prepare_checkout(
        ctx.session, ctx.session_id, ctx.customer_id,
        payment_method=_opt_str(args.get("payment_method")) or "card",
        shipping_address=_opt_str(args.get("shipping_address")),
    )


def _place_order(ctx: ToolContext, args: dict[str, Any]) -> Any:
    token = str(args.get("confirmation_token") or "").strip()
    if not token:
        return ToolError(
            error="A confirmation_token from prepare_checkout is required.",
            code="MISSING_CONFIRMATION",
            recovery_hint=(
                "Call prepare_checkout, show the customer the quote, and wait for them to "
                "confirm. Never invent a token."
            ),
        )
    return cart_service.place_order(ctx.session, ctx.session_id, ctx.customer_id, token)


def _lookup_policy(ctx: ToolContext, args: dict[str, Any]) -> Any:
    return catalog_service.lookup_policy(
        _opt_str(args.get("query")) or "",
        topic=_opt_str(args.get("topic")),
        limit=_opt_int(args.get("limit"), 3),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SORT_ENUM = ["relevance", "price_low_to_high", "price_high_to_low", "rating", "newest", "discount"]

TOOLS: list[Tool] = [
    Tool(
        name="search_products",
        description=(
            "Search the product catalogue. Call this before describing or pricing any "
            "product; never answer from memory. Put descriptive words in `query`; put "
            "anything matching a real attribute (brand, size, colour, price) in its own "
            "parameter. Parameters filter, `query` only ranks."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Descriptive words only, e.g. 'lightweight running'. Omit if purely attribute-based."},
                "brand": {"type": "string",
                          "description": "As the shopper said it; misspellings are resolved server-side. Unstocked brands are reported explicitly."},
                "category": {"type": "string",
                             "enum": ["Topwear", "Bottomwear", "Footwear", "Outerwear", "Accessories"]},
                "subcategory": {"type": "string",
                                "description": "Product type, e.g. 'T-Shirt', 'Jeans', 'Running Shoes', 'Hoodie'. Prefer over `query` when named."},
                "gender": {"type": "string", "enum": ["men", "women", "unisex"]},
                "size": {"type": "string",
                         "description": "XS-XXL or US shoe 6-12. Only if stated; never infer."},
                "color": {"type": "string"},
                "min_price": {"type": "number", "description": "Dollars, not cents."},
                "max_price": {"type": "number", "description": "Dollars, not cents. 'under $50' is 50."},
                "min_rating": {"type": "number", "description": "0 to 5."},
                "in_stock_only": {"type": "boolean", "description": "Default true; false only if sold-out items are asked for."},
                "on_sale_only": {"type": "boolean"},
                "sort": {"type": "string", "enum": _SORT_ENUM,
                         "description": "price_low_to_high for 'cheapest', rating for 'best'."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 24, "description": "Default 6."},
            },
            "required": [],
        },
        executor=_search_products,
        renders="products",
    ),
    Tool(
        name="get_product_details",
        description=(
            "Full detail for one product: description, material, care instructions, and the "
            "complete size and colour grid with per-variant stock. Use when the shopper asks "
            "about a specific product they have already been shown."
        ),
        parameters={
            "type": "object",
            "properties": {
                "product_id": {"type": "integer", "description": "The product_id from a previous search_products result."},
            },
            "required": ["product_id"],
        },
        executor=_get_product_details,
        renders="products",
    ),
    Tool(
        name="check_availability",
        description=(
            "Check live stock for a specific product, optionally narrowed to one size and "
            "colour. Call this before telling a shopper that something is available in their "
            "size, and before adding to the cart. Stock is per size-and-colour variant, so a "
            "product being 'in stock' does not mean their size is."
        ),
        parameters={
            "type": "object",
            "properties": {
                "product_id": {"type": "integer", "description": "The product_id to check."},
                "size": {"type": "string", "description": "Size to check, e.g. 'M' or '9'. Omit to see all sizes."},
                "color": {"type": "string", "description": "Colour to check. Omit to see all colours."},
            },
            "required": ["product_id"],
        },
        executor=_check_availability,
    ),
    Tool(
        name="list_brands",
        description=(
            "List every brand Aurelia stocks, with product counts. Use when the shopper asks "
            "what brands are carried, or to check whether a brand exists before apologising "
            "for not having it."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        executor=_list_brands,
    ),
    Tool(
        name="list_categories",
        description="List all categories and subcategories with product counts. Use to orient a shopper who does not know what to ask for.",
        parameters={"type": "object", "properties": {}, "required": []},
        executor=_list_categories,
    ),
    Tool(
        name="get_order_status",
        description=(
            "Full detail for one of the signed-in customer's orders: status, line items, "
            "totals, shipping address, tracking and the complete event timeline. This is the "
            "only source of truth for order questions. The result includes a "
            "`delivery_message` field written in plain language: repeat that sentence rather "
            "than composing your own delivery claim, and never calculate dates yourself."
        ),
        parameters={
            "type": "object",
            "properties": {
                "order_number": {
                    "type": "string",
                    "description": "Order number as given by the customer. A leading '#' is fine.",
                },
            },
            "required": ["order_number"],
        },
        executor=_get_order_status,
        renders="orders",
    ),
    Tool(
        name="track_shipment",
        description=(
            "Delivery-focused view of one order: estimated delivery date, days remaining, "
            "carrier, tracking number and courier scan history. Use this specifically for "
            "'when will it arrive' and 'where is my parcel'. Quote the `delivery_message` "
            "field verbatim."
        ),
        parameters={
            "type": "object",
            "properties": {"order_number": {"type": "string", "description": "The order number to track."}},
            "required": ["order_number"],
        },
        executor=_track_shipment,
    ),
    Tool(
        name="list_my_orders",
        description=(
            "List the signed-in customer's recent orders, newest first. Use when they refer "
            "to 'my order' without a number, or ask what they have bought recently."
        ),
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "How many orders. Default 5."},
                "status": {
                    "type": "string",
                    "description": (
                        "Optional status filter: pending_payment, confirmed, packed, shipped, "
                        "out_for_delivery, delivered, cancelled, return_requested, returned."
                    ),
                },
            },
            "required": [],
        },
        executor=_list_my_orders,
        renders="orders",
    ),
    Tool(
        name="cancel_order",
        description=(
            "Cancel one of the customer's orders. Only possible before dispatch; the backend "
            "enforces this and will refuse a shipped order. Confirm with the customer that "
            "they want it cancelled before calling, and never call this speculatively."
        ),
        parameters={
            "type": "object",
            "properties": {
                "order_number": {"type": "string", "description": "The order number to cancel."},
                "reason": {"type": "string", "description": "Optional reason the customer gave, recorded on the order."},
            },
            "required": ["order_number"],
        },
        executor=_cancel_order,
        mutating=True,
        renders="orders",
    ),
    Tool(
        name="request_return",
        description=(
            "Open a return on a delivered order. The backend enforces the return window and "
            "will refuse an order that is outside it. Confirm the customer's intent first."
        ),
        parameters={
            "type": "object",
            "properties": {
                "order_number": {"type": "string", "description": "The delivered order to return."},
                "reason": {"type": "string", "description": "Why they are returning it, recorded on the order."},
            },
            "required": ["order_number"],
        },
        executor=_request_return,
        mutating=True,
        renders="orders",
    ),
    Tool(
        name="add_to_cart",
        description=(
            "Add a product to the shopping cart. Supply a variant_id, or a product_id, or "
            "(only if neither is known) a product_name to resolve by exact catalogue name. "
            "Prefer product_id from a prior search_products call when you have one. Combine "
            "product_id or product_name with size and colour. If size or colour is ambiguous "
            "the tool returns NEEDS_SIZE or NEEDS_COLOR listing the real options: ask the "
            "customer to choose rather than picking for them."
        ),
        parameters={
            "type": "object",
            "properties": {
                "product_id": {"type": "integer", "description": "Product to add. Combine with size and colour."},
                "product_name": {"type": "string", "description": "Exact product name, only if product_id is unknown."},
                "variant_id": {"type": "integer", "description": "Exact variant, from check_availability or get_product_details. Preferred when known."},
                "size": {"type": "string", "description": "Required when the product has more than one size in stock."},
                "color": {"type": "string", "description": "Required when the product has more than one colour in stock."},
                "quantity": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Default 1."},
            },
            "required": [],
        },
        executor=_add_to_cart,
        mutating=True,
        renders="cart",
    ),
    Tool(
        name="view_cart",
        description="Show the current cart with line items, subtotal, shipping, tax and total. Call before checkout so the customer sees what they are buying.",
        parameters={"type": "object", "properties": {}, "required": []},
        executor=_view_cart,
        renders="cart",
    ),
    Tool(
        name="update_cart_quantity",
        description="Change the quantity of a cart line. Setting quantity to 0 removes it.",
        parameters={
            "type": "object",
            "properties": {
                "variant_id": {"type": "integer", "description": "From view_cart."},
                "quantity": {"type": "integer", "minimum": 0, "maximum": 10},
            },
            "required": ["variant_id", "quantity"],
        },
        executor=_update_cart_quantity,
        mutating=True,
        renders="cart",
    ),
    Tool(
        name="remove_from_cart",
        description=(
            "Remove one line from the cart entirely, whatever its quantity. Use this when "
            "the customer changes their mind about an item; use update_cart_quantity if "
            "they only want fewer of it."
        ),
        parameters={
            "type": "object",
            "properties": {"variant_id": {"type": "integer", "description": "From view_cart."}},
            "required": ["variant_id"],
        },
        executor=_remove_from_cart,
        mutating=True,
        renders="cart",
    ),
    Tool(
        name="prepare_checkout",
        description=(
            "Price the cart against live stock and produce a confirmation quote. This does "
            "NOT place an order and does NOT charge anything. Call it when the customer says "
            "they want to buy, then present the returned totals and wait. The customer "
            "confirms by pressing the confirm button on the quote card in the interface."
        ),
        parameters={
            "type": "object",
            "properties": {
                "payment_method": {
                    "type": "string",
                    "enum": ["card", "upi", "net_banking", "wallet", "cash_on_delivery"],
                    "description": "Defaults to card. Only change if the customer names a method.",
                },
                "shipping_address": {
                    "type": "string",
                    "description": "Only if the customer gives a new address. Otherwise the account address is used.",
                },
            },
            "required": [],
        },
        executor=_prepare_checkout,
        renders="checkout",
        model_redacted_fields=("confirmation_token",),
    ),
    Tool(
        name="place_order",
        description=(
            "Commit the order. This is the only irreversible action available to you.\n"
            "You cannot call this successfully: the confirmation token is issued directly to "
            "the customer's browser and is deliberately withheld from you, so any value you "
            "supply will be rejected. The customer completes the purchase by pressing Confirm "
            "on the quote card. Call prepare_checkout, present the total, and let them press "
            "it. This tool exists so you can explain that step, not perform it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "confirmation_token": {
                    "type": "string",
                    "description": "The exact token string returned by prepare_checkout.",
                },
            },
            "required": ["confirmation_token"],
        },
        executor=_place_order,
        mutating=True,
        renders="order_placed",
    ),
    Tool(
        name="lookup_policy",
        description=(
            "Retrieve Aurelia's official policy text on shipping, returns, cancellation, "
            "sizing, payments, warranty or privacy. Use this for any 'can I', 'how long', "
            "'what happens if' question about store rules. Answer only from the returned "
            "passages and cite the document heading. Do not answer policy questions from "
            "general knowledge, because these rules are specific to Aurelia."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The customer's question, in their own words."},
                "topic": {
                    "type": "string",
                    "enum": ["shipping", "returns", "cancellation", "sizing", "payments", "warranty", "privacy"],
                    "description": "Optional topic hint to narrow retrieval.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Passages to retrieve. Default 3."},
            },
            "required": ["query"],
        },
        executor=_lookup_policy,
        renders="policy",
    ),
]

TOOLS_BY_NAME: dict[str, Tool] = {tool.name: tool for tool in TOOLS}
MUTATING_TOOLS: frozenset[str] = frozenset(t.name for t in TOOLS if t.mutating)


def tool_schemas() -> list[dict[str, Any]]:
    return [tool.schema() for tool in TOOLS]


def execute_tool(name: str, arguments: dict[str, Any], context: ToolContext) -> tuple[Any, str]:
    """Run one tool. Returns `(json_safe_result, status)`.

    Status is 'ok', 'error' (the tool ran and returned a typed failure) or
    'unknown_tool'. Exceptions are caught and converted, because a stack trace
    escaping into the agent loop would abandon the customer mid-conversation.
    """
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        logger.warning("tool.unknown", extra={"tool_name": name})
        return (
            _dump(ToolError(
                error=f"There is no tool named '{name}'.",
                code="UNKNOWN_TOOL",
                recovery_hint="Choose one of the tools that were provided.",
            )),
            "unknown_tool",
        )

    try:
        result = tool.executor(context, arguments or {})
    except Exception:  # noqa: BLE001 - the loop must survive any tool defect
        logger.exception("tool.failed", extra={"tool_name": name, "arguments": arguments})
        return (
            _dump(ToolError(
                error="That lookup failed unexpectedly.",
                code="TOOL_EXECUTION_ERROR",
                recovery_hint=(
                    "Tell the customer the system could not complete that request and offer "
                    "to connect them to a human agent. Do not invent the answer."
                ),
                retryable=True,
            )),
            "error",
        )

    status = "error" if isinstance(result, ToolError) else "ok"
    return _dump(result), status
