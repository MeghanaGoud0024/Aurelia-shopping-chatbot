"""Cart and checkout.

The purchase guardrail
----------------------
Placing an order is the only irreversible action this assistant can take, so it
is the one place where "the model decided to" is not an acceptable audit answer.
Checkout is therefore split into two phases:

1. `prepare_checkout` prices the basket, re-checks live stock, and returns a
   quote carrying a single-use `confirmation_token` with a short expiry.
2. `place_order` refuses to do anything without that exact token.

The model cannot invent a valid token, because tokens are issued by the server
from `secrets.token_urlsafe` and held in a server-side store. The frontend
surfaces the quote as an explicit confirm/cancel card, so the token only reaches
`place_order` after a human has clicked. The net effect is that no sequence of
words in a conversation can cause a charge on its own.

The token is also bound to the exact basket that was quoted. If the cart changes
between quote and confirmation, the token is rejected rather than silently
charging for a different set of items.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import Integer, cast, delete, func, select
from sqlalchemy.orm import Session

from app.db.models import (
    CartItem, Customer, Order, OrderEvent, OrderItem, OrderStatus,
    PaymentMethod, ProductVariant,
)
from app.schemas import CartLine, CartView, CheckoutQuote, Money, ToolError
from app.services.catalog import _size_rank

logger = logging.getLogger(__name__)

FREE_SHIPPING_THRESHOLD_CENTS = 7_500
STANDARD_SHIPPING_CENTS = 799
TAX_RATE = 0.10
MAX_LINE_QUANTITY = 10
MAX_CART_LINES = 20
QUOTE_TTL_SECONDS = 600


# ---------------------------------------------------------------------------
# Pending checkout quotes
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class _PendingQuote:
    session_id: str
    customer_id: int
    basket_fingerprint: str
    total_cents: int
    payment_method: str
    shipping_address: str
    expires_at: datetime


class _QuoteStore:
    """In-memory, TTL-bounded store of issued checkout quotes.

    A process-local dict is the right call for a single-process POC and the
    wrong call for a horizontally scaled deployment, where two workers would not
    share it. `docs/SCALING.md` covers the swap to Redis with the same
    interface; nothing outside this class would change.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._quotes: dict[str, _PendingQuote] = {}

    def issue(self, quote: _PendingQuote) -> str:
        token = secrets.token_urlsafe(24)
        with self._lock:
            self._prune()
            self._quotes[token] = quote
        return token

    def consume(self, token: str) -> _PendingQuote | None:
        """Redeem a token. Single use: a redeemed token is gone."""
        with self._lock:
            self._prune()
            return self._quotes.pop(token, None)

    def _prune(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [t for t, q in self._quotes.items() if q.expires_at <= now]
        for token in expired:
            del self._quotes[token]

    def clear(self) -> None:
        with self._lock:
            self._quotes.clear()


quote_store = _QuoteStore()


def _fingerprint(lines: list[CartLine], payment_method: str, address: str) -> str:
    """Stable hash of exactly what was quoted."""
    payload = json.dumps(
        {
            "lines": sorted((l.variant_id, l.quantity, l.unit_price.amount_cents) for l in lines),
            "payment_method": payment_method,
            "address": address,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def _price_basket(lines: list[CartLine]) -> tuple[int, int, int, int]:
    subtotal = sum(line.line_total.amount_cents for line in lines)
    shipping = 0 if (subtotal >= FREE_SHIPPING_THRESHOLD_CENTS or not lines) else STANDARD_SHIPPING_CENTS
    tax = round(subtotal * TAX_RATE)
    return subtotal, shipping, tax, subtotal + shipping + tax


def _line(item: CartItem) -> CartLine:
    variant = item.variant
    product = variant.product
    unit = product.price_cents
    return CartLine(
        variant_id=variant.id,
        product_id=product.id,
        product_name=product.name,
        brand=product.brand,
        size=variant.size,
        color=variant.color,
        quantity=item.quantity,
        unit_price=Money.of(unit, product.currency),
        line_total=Money.of(unit * item.quantity, product.currency),
        stock_available=variant.stock,
    )


def _load_lines(session: Session, session_id: str) -> list[CartLine]:
    items = session.scalars(
        select(CartItem).where(CartItem.session_id == session_id).order_by(CartItem.added_at)
    ).all()
    return [_line(item) for item in items]


def view_cart(session: Session, session_id: str) -> CartView:
    lines = _load_lines(session, session_id)
    subtotal, shipping, tax, total = _price_basket(lines)

    note = ""
    oversold = [l for l in lines if l.quantity > l.stock_available]
    if oversold:
        note = (
            "Stock has changed since these items were added: "
            + "; ".join(
                f"{l.product_name} ({l.size}/{l.color}) has {l.stock_available} left "
                f"but {l.quantity} are in the cart"
                for l in oversold
            )
        )
    elif lines and subtotal < FREE_SHIPPING_THRESHOLD_CENTS:
        gap = FREE_SHIPPING_THRESHOLD_CENTS - subtotal
        note = f"Add {Money.of(gap).display} more to qualify for free standard shipping."

    return CartView(
        lines=lines,
        item_count=sum(l.quantity for l in lines),
        subtotal=Money.of(subtotal),
        shipping=Money.of(shipping),
        tax=Money.of(tax),
        total=Money.of(total),
        free_shipping_threshold=Money.of(FREE_SHIPPING_THRESHOLD_CENTS),
        note=note,
    )


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

def add_to_cart(
    session: Session,
    session_id: str,
    *,
    product_id: int | None = None,
    variant_id: int | None = None,
    size: str | None = None,
    color: str | None = None,
    quantity: int = 1,
) -> CartView | ToolError:
    """Add one variant to the cart.

    A product plus a size is enough; the exact variant is resolved here. If the
    combination is ambiguous (several colours available) or unavailable, we
    return a typed error listing the real options rather than picking one, so
    the assistant asks instead of assuming.
    """
    quantity = max(1, min(int(quantity or 1), MAX_LINE_QUANTITY))

    variant: ProductVariant | None = None
    if variant_id is not None:
        variant = session.get(ProductVariant, variant_id)
        if variant is None:
            return ToolError(
                error=f"No product variant with id {variant_id} exists.",
                code="VARIANT_NOT_FOUND",
                recovery_hint="Call search_products or get_product_details to obtain a valid variant_id.",
            )
    elif product_id is not None:
        candidates = session.scalars(
            select(ProductVariant).where(ProductVariant.product_id == product_id)
        ).all()
        if not candidates:
            return ToolError(
                error=f"No product with id {product_id} exists.",
                code="PRODUCT_NOT_FOUND",
                recovery_hint="Call search_products to find a valid product_id.",
            )
        if size:
            candidates = [v for v in candidates if v.size.upper() == size.strip().upper()]
        if color:
            candidates = [v for v in candidates if v.color.lower() == color.strip().lower()]
        in_stock = [v for v in candidates if v.stock > 0]

        if not candidates:
            return ToolError(
                error=(
                    f"That product has no variant matching "
                    f"{'size ' + size if size else ''}"
                    f"{' and ' if size and color else ''}"
                    f"{'colour ' + color if color else ''}."
                ),
                code="VARIANT_NOT_FOUND",
                recovery_hint="Call check_availability to list the sizes and colours that exist.",
            )
        if not in_stock:
            return ToolError(
                error="Every matching variant of that product is out of stock.",
                code="OUT_OF_STOCK",
                recovery_hint=(
                    "Call check_availability and offer the sizes or colours that are in stock."
                ),
            )
        distinct_colors = {v.color for v in in_stock}
        distinct_sizes = {v.size for v in in_stock}
        if len(distinct_colors) > 1 and not color:
            return ToolError(
                error=f"Colour is ambiguous. Available: {', '.join(sorted(distinct_colors))}.",
                code="NEEDS_COLOR",
                recovery_hint="Ask the customer which colour they want, then call add_to_cart again.",
                needs_field="color",
                options=sorted(distinct_colors),
            )
        if len(distinct_sizes) > 1 and not size:
            return ToolError(
                error=f"Size is ambiguous. Available: {', '.join(sorted(distinct_sizes))}.",
                code="NEEDS_SIZE",
                recovery_hint="Ask the customer which size they want, then call add_to_cart again.",
                needs_field="size",
                options=sorted(distinct_sizes, key=_size_rank),
            )
        variant = in_stock[0]
    else:
        return ToolError(
            error="add_to_cart requires either a variant_id or a product_id.",
            code="MISSING_ARGUMENT",
            recovery_hint="Search for the product first to obtain an id.",
        )

    if variant.stock <= 0:
        return ToolError(
            error=(
                f"{variant.product.name} in {variant.size}/{variant.color} is out of stock."
            ),
            code="OUT_OF_STOCK",
            recovery_hint="Call check_availability and offer an in-stock alternative.",
        )

    existing = session.scalar(
        select(CartItem).where(
            CartItem.session_id == session_id, CartItem.variant_id == variant.id
        )
    )
    line_count = session.scalar(
        select(CartItem.id).where(CartItem.session_id == session_id).limit(MAX_CART_LINES + 1)
    )
    if existing is None and line_count is not None:
        current = len(session.scalars(select(CartItem).where(CartItem.session_id == session_id)).all())
        if current >= MAX_CART_LINES:
            return ToolError(
                error=f"The cart already holds the maximum of {MAX_CART_LINES} distinct items.",
                code="CART_FULL",
                recovery_hint="Ask the customer to remove something before adding more.",
            )

    desired = (existing.quantity if existing else 0) + quantity
    if desired > variant.stock:
        return ToolError(
            error=(
                f"Only {variant.stock} of {variant.product.name} in "
                f"{variant.size}/{variant.color} are available, and the cart would need {desired}."
            ),
            code="INSUFFICIENT_STOCK",
            recovery_hint=f"Offer the customer the {variant.stock} available, or a different size.",
        )
    desired = min(desired, MAX_LINE_QUANTITY)

    if existing:
        existing.quantity = desired
    else:
        session.add(CartItem(session_id=session_id, variant_id=variant.id, quantity=desired))
    session.flush()

    logger.info(
        "cart.item_added",
        extra={"session_id": session_id, "variant_id": variant.id, "quantity": desired},
    )
    return view_cart(session, session_id)


def update_cart_quantity(
    session: Session, session_id: str, variant_id: int, quantity: int
) -> CartView | ToolError:
    item = session.scalar(
        select(CartItem).where(CartItem.session_id == session_id, CartItem.variant_id == variant_id)
    )
    if item is None:
        return ToolError(
            error="That item is not in the cart.",
            code="NOT_IN_CART",
            recovery_hint="Call view_cart to see what is actually in the basket.",
        )
    if quantity <= 0:
        session.delete(item)
        session.flush()
        return view_cart(session, session_id)
    if quantity > item.variant.stock:
        return ToolError(
            error=f"Only {item.variant.stock} are in stock.",
            code="INSUFFICIENT_STOCK",
            recovery_hint="Offer the available quantity.",
        )
    item.quantity = min(quantity, MAX_LINE_QUANTITY)
    session.flush()
    return view_cart(session, session_id)


def remove_from_cart(session: Session, session_id: str, variant_id: int) -> CartView:
    session.execute(
        delete(CartItem).where(CartItem.session_id == session_id, CartItem.variant_id == variant_id)
    )
    session.flush()
    return view_cart(session, session_id)


def clear_cart(session: Session, session_id: str) -> CartView:
    session.execute(delete(CartItem).where(CartItem.session_id == session_id))
    session.flush()
    return view_cart(session, session_id)


# ---------------------------------------------------------------------------
# Checkout
# ---------------------------------------------------------------------------

def prepare_checkout(
    session: Session,
    session_id: str,
    customer_id: int,
    *,
    payment_method: str = "card",
    shipping_address: str | None = None,
) -> CheckoutQuote | ToolError:
    """Phase one: price the basket and issue a single-use confirmation token."""
    lines = _load_lines(session, session_id)
    if not lines:
        return ToolError(
            error="The cart is empty, so there is nothing to check out.",
            code="EMPTY_CART",
            recovery_hint="Help the customer find and add a product first.",
        )

    try:
        method = PaymentMethod(payment_method.strip().lower())
    except ValueError:
        return ToolError(
            error=f"'{payment_method}' is not an accepted payment method.",
            code="INVALID_PAYMENT_METHOD",
            recovery_hint="Accepted values: " + ", ".join(m.value for m in PaymentMethod),
        )

    customer = session.get(Customer, customer_id)
    if customer is None:
        return ToolError(
            error="No signed-in customer is associated with this session.",
            code="NOT_AUTHENTICATED",
            recovery_hint="Ask the customer to sign in before purchasing.",
        )

    warnings: list[str] = []
    for line in lines:
        if line.quantity > line.stock_available:
            return ToolError(
                error=(
                    f"{line.product_name} ({line.size}/{line.color}) has only "
                    f"{line.stock_available} in stock but {line.quantity} are in the cart."
                ),
                code="INSUFFICIENT_STOCK",
                recovery_hint="Reduce the quantity with update_cart_quantity, then retry checkout.",
            )
        if line.stock_available - line.quantity <= 3:
            warnings.append(f"{line.product_name} ({line.size}/{line.color}) is low on stock.")

    address = (shipping_address or "").strip() or f"{customer.city}, {customer.country}"
    subtotal, shipping, tax, total = _price_basket(lines)
    if method == PaymentMethod.COD and total >= 30_000:
        return ToolError(
            error="Cash on delivery is not available on orders of $300.00 or more.",
            code="COD_LIMIT_EXCEEDED",
            recovery_hint="Offer card, UPI, net banking or wallet instead.",
        )

    expires_at = datetime.now(timezone.utc) + timedelta(seconds=QUOTE_TTL_SECONDS)
    token = quote_store.issue(
        _PendingQuote(
            session_id=session_id,
            customer_id=customer_id,
            basket_fingerprint=_fingerprint(lines, method.value, address),
            total_cents=total,
            payment_method=method.value,
            shipping_address=address,
            expires_at=expires_at,
        )
    )

    logger.info(
        "checkout.quote_issued",
        extra={"session_id": session_id, "total_cents": total, "lines": len(lines)},
    )
    return CheckoutQuote(
        confirmation_token=token,
        expires_at=expires_at,
        lines=lines,
        subtotal=Money.of(subtotal),
        shipping=Money.of(shipping),
        tax=Money.of(tax),
        total=Money.of(total),
        payment_method=method.value,
        shipping_address=address,
        warnings=warnings,
    )


def place_order(
    session: Session, session_id: str, customer_id: int, confirmation_token: str
) -> dict | ToolError:
    """Phase two: redeem a confirmation token and commit the order.

    Every rejection path here is a deliberate safety property, not defensive
    boilerplate: an unknown or expired token, a token issued to another session,
    and a basket that changed after it was quoted all stop the charge.
    """
    quote = quote_store.consume((confirmation_token or "").strip())
    if quote is None:
        return ToolError(
            error="That confirmation is not valid. It may have expired or already been used.",
            code="INVALID_CONFIRMATION",
            recovery_hint=(
                "Call prepare_checkout again to produce a fresh quote for the customer to confirm. "
                "Never invent a confirmation token."
            ),
            retryable=True,
        )
    if quote.session_id != session_id or quote.customer_id != customer_id:
        logger.warning(
            "checkout.token_session_mismatch",
            extra={"session_id": session_id, "token_session": quote.session_id},
        )
        return ToolError(
            error="That confirmation does not belong to this session.",
            code="CONFIRMATION_MISMATCH",
            recovery_hint="Call prepare_checkout to issue a fresh quote.",
        )
    if quote.expires_at <= datetime.now(timezone.utc):
        return ToolError(
            error="That confirmation has expired.",
            code="CONFIRMATION_EXPIRED",
            recovery_hint="Call prepare_checkout again.",
            retryable=True,
        )

    lines = _load_lines(session, session_id)
    if not lines:
        return ToolError(
            error="The cart is now empty, so there is nothing to place.",
            code="EMPTY_CART",
            recovery_hint="Ask the customer to add items again.",
        )
    if _fingerprint(lines, quote.payment_method, quote.shipping_address) != quote.basket_fingerprint:
        return ToolError(
            error="The cart changed after the quote was produced, so the confirmation is void.",
            code="BASKET_CHANGED",
            recovery_hint="Call prepare_checkout again so the customer confirms the current basket.",
            retryable=True,
        )

    # Re-check stock inside the write transaction. The gap between quoting and
    # confirming is exactly where a concurrent order can empty the shelf.
    variants: dict[int, ProductVariant] = {}
    for line in lines:
        variant = session.get(ProductVariant, line.variant_id)
        if variant is None or variant.stock < line.quantity:
            available = variant.stock if variant else 0
            return ToolError(
                error=(
                    f"{line.product_name} ({line.size}/{line.color}) sold out while the order was "
                    f"being confirmed. {available} remain."
                ),
                code="STOCK_CHANGED",
                recovery_hint="Tell the customer, adjust the cart, and re-quote.",
                retryable=True,
            )
        variants[line.variant_id] = variant

    now = datetime.now(timezone.utc)
    subtotal, shipping, tax, total = _price_basket(lines)
    next_number = _next_order_number(session)

    order = Order(
        order_number=next_number,
        customer_id=customer_id,
        status=OrderStatus.CONFIRMED,
        payment_method=PaymentMethod(quote.payment_method),
        subtotal_cents=subtotal,
        shipping_cents=shipping,
        tax_cents=tax,
        total_cents=total,
        shipping_address=quote.shipping_address,
        carrier="Aurelia Express",
        placed_at=now,
        estimated_delivery_at=(now + timedelta(days=5)).replace(hour=18, minute=0, second=0, microsecond=0),
        channel="assistant",
    )
    for line in lines:
        variant = variants[line.variant_id]
        variant.stock -= line.quantity  # decrement live inventory
        order.items.append(
            OrderItem(
                variant_id=variant.id,
                product_name=line.product_name,
                brand=line.brand,
                size=line.size,
                color=line.color,
                quantity=line.quantity,
                unit_price_cents=line.unit_price.amount_cents,
            )
        )
    order.events.append(
        OrderEvent(
            status=OrderStatus.CONFIRMED,
            location="Aurelia Order Service",
            note="Payment authorised, order confirmed via the shopping assistant.",
            occurred_at=now,
        )
    )
    session.add(order)
    session.execute(delete(CartItem).where(CartItem.session_id == session_id))
    session.flush()

    logger.info(
        "checkout.order_placed",
        extra={
            "order_number": order.order_number,
            "customer_id": customer_id,
            "total_cents": total,
            "line_count": len(lines),
        },
    )
    eta = order.estimated_delivery_at
    return {
        "order_placed": True,
        "order_number": order.order_number,
        "status": order.status.value,
        "total": Money.of(total).model_dump(),
        "payment_method": quote.payment_method,
        "shipping_address": quote.shipping_address,
        "estimated_delivery_at": eta.isoformat(),
        "estimated_delivery_readable": f"{eta:%A %d %B %Y}",
        "items": [
            {
                "product_name": l.product_name, "size": l.size, "color": l.color,
                "quantity": l.quantity, "line_total": l.line_total.display,
            }
            for l in lines
        ],
        "message": (
            f"Order {order.order_number} is confirmed. Total {Money.of(total).display}, "
            f"estimated delivery {eta:%A %d %B %Y}."
        ),
    }


def _next_order_number(session: Session) -> str:
    """Allocate the next order number.

    MAX() plus one is safe here because SQLite serialises writers and the whole
    checkout runs in one transaction. On a database with concurrent writers this
    becomes a sequence; the note is in docs/SCALING.md so the assumption is not
    silently inherited.
    """
    highest = session.scalar(select(func.max(cast(Order.order_number, Integer))))
    return str((highest or 1000) + 1)
