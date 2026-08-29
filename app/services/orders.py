"""Order service.

Authorisation posture
---------------------
Every read and every mutation in this module takes an explicit `customer_id`
and filters on it in the SQL predicate. There is no code path that fetches an
order by number alone. This matters because the caller is, ultimately, a
language model acting on instructions from a user who may be adversarial: if
authorisation were expressed as a rule in the system prompt, a sufficiently
persuasive message could talk its way past it. Expressed as a WHERE clause, it
cannot be argued with.

The failure mode is deliberately uniform. An order that does not exist and an
order belonging to somebody else return the *same* `ORDER_NOT_FOUND` response.
Distinguishing them would turn the assistant into an oracle that confirms
whether an arbitrary order number is real, which is an enumeration vector.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db.models import Customer, Order, OrderEvent, OrderStatus
from app.schemas import (
    Money, OrderDetail, OrderEventOut, OrderItemOut, OrderSummary, ToolError,
)

logger = logging.getLogger(__name__)

RETURN_WINDOW_DAYS = 30
EXTENDED_RETURN_WINDOW_DAYS = 60
EXTENDED_TIERS = {"gold", "platinum"}

STATUS_LABELS: dict[OrderStatus, str] = {
    OrderStatus.PENDING_PAYMENT: "Awaiting payment",
    OrderStatus.CONFIRMED: "Confirmed",
    OrderStatus.PACKED: "Packed",
    OrderStatus.SHIPPED: "Shipped",
    OrderStatus.OUT_FOR_DELIVERY: "Out for delivery",
    OrderStatus.DELIVERED: "Delivered",
    OrderStatus.CANCELLED: "Cancelled",
    OrderStatus.RETURN_REQUESTED: "Return requested",
    OrderStatus.RETURNED: "Returned",
}


def _as_utc(value: datetime | None) -> datetime | None:
    """Normalise to timezone-aware UTC.

    SQLite discards tzinfo on write, so values read back are naive even though
    they were stored as UTC. Comparing a naive datetime with an aware one
    raises, which would surface as a 500 on an ordinary status question.
    """
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _order_not_found(order_number: str) -> ToolError:
    return ToolError(
        error=(
            f"No order numbered {order_number} was found on this account."
        ),
        code="ORDER_NOT_FOUND",
        recovery_hint=(
            "Ask the customer to confirm the order number, or call list_my_orders "
            "to show the orders that do exist on this account."
        ),
        retryable=False,
    )


# ---------------------------------------------------------------------------
# Derived facts
# ---------------------------------------------------------------------------

def _delivery_message(order: Order, now: datetime) -> str:
    """A plain-language delivery statement, computed rather than generated.

    'When will my order arrive' is the single highest-traffic question in
    e-commerce support and the easiest one to get subtly wrong. Deriving the
    sentence here from real timestamps, and having the model repeat it verbatim,
    removes an entire class of hallucination: the model never does date
    arithmetic, so it never gets date arithmetic wrong.
    """
    eta = _as_utc(order.estimated_delivery_at)
    delivered = _as_utc(order.delivered_at)

    if order.status == OrderStatus.DELIVERED and delivered:
        return f"Delivered on {delivered:%d %B %Y} to {order.shipping_address}."
    if order.status == OrderStatus.CANCELLED:
        cancelled = _as_utc(order.cancelled_at)
        when = f" on {cancelled:%d %B %Y}" if cancelled else ""
        return f"This order was cancelled{when}. No delivery is scheduled."
    if order.status == OrderStatus.RETURNED:
        return "This order was returned and the refund has been issued."
    if order.status == OrderStatus.RETURN_REQUESTED:
        return "A return is in progress for this order. Collection is being scheduled."
    if order.status == OrderStatus.PENDING_PAYMENT:
        return (
            "Payment has not been authorised yet, so no delivery date has been set. "
            "The estimate is generated once payment clears."
        )
    if order.status == OrderStatus.OUT_FOR_DELIVERY:
        return "Out for delivery today with the courier. Expect it before end of day."
    if eta is None:
        return "No delivery estimate is available for this order yet."

    days = (eta.date() - now.date()).days
    when = {
        0: "today",
        1: "tomorrow",
    }.get(days) or (f"in {days} days" if days > 1 else "shortly, the estimate has passed")
    carrier = f" with {order.carrier}" if order.carrier else ""
    return f"Estimated delivery {when}, on {eta:%A %d %B %Y}{carrier}."


def _is_returnable(order: Order, customer: Customer, now: datetime) -> bool:
    if order.status != OrderStatus.DELIVERED:
        return False
    delivered = _as_utc(order.delivered_at)
    if delivered is None:
        return False
    window = (
        EXTENDED_RETURN_WINDOW_DAYS
        if customer.loyalty_tier.lower() in EXTENDED_TIERS
        else RETURN_WINDOW_DAYS
    )
    return now - delivered <= timedelta(days=window)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _summary(order: Order) -> OrderSummary:
    return OrderSummary(
        order_number=order.order_number,
        status=order.status.value,
        status_label=STATUS_LABELS[order.status],
        placed_at=_as_utc(order.placed_at),
        total=Money.of(order.total_cents, order.currency),
        item_count=sum(item.quantity for item in order.items),
        estimated_delivery_at=_as_utc(order.estimated_delivery_at),
        delivered_at=_as_utc(order.delivered_at),
    )


def _detail(order: Order, customer: Customer, now: datetime) -> OrderDetail:
    return OrderDetail(
        **_summary(order).model_dump(),
        payment_method=order.payment_method.value,
        subtotal=Money.of(order.subtotal_cents, order.currency),
        shipping=Money.of(order.shipping_cents, order.currency),
        tax=Money.of(order.tax_cents, order.currency),
        shipping_address=order.shipping_address,
        carrier=order.carrier,
        tracking_number=order.tracking_number,
        items=[
            OrderItemOut(
                product_name=item.product_name, brand=item.brand, size=item.size,
                color=item.color, quantity=item.quantity,
                unit_price=Money.of(item.unit_price_cents, order.currency),
                line_total=Money.of(item.line_total_cents, order.currency),
            )
            for item in order.items
        ],
        timeline=[
            OrderEventOut(
                status=event.status.value, label=STATUS_LABELS[event.status],
                location=event.location, note=event.note,
                occurred_at=_as_utc(event.occurred_at),
            )
            for event in order.events
        ],
        is_cancellable=order.status.is_cancellable,
        is_returnable=_is_returnable(order, customer, now),
        delivery_message=_delivery_message(order, now),
    )


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def _fetch(session: Session, order_number: str, customer_id: int) -> Order | None:
    """The only order fetch in the codebase. Always scoped to one customer."""
    cleaned = order_number.strip().lstrip("#").strip()
    if not cleaned:
        return None
    return session.scalar(
        select(Order).where(
            Order.order_number == cleaned,
            Order.customer_id == customer_id,
        )
    )


def get_order_status(
    session: Session, order_number: str, customer_id: int
) -> OrderDetail | ToolError:
    order = _fetch(session, order_number, customer_id)
    if order is None:
        logger.info(
            "orders.not_found",
            extra={"order_number": order_number, "customer_id": customer_id},
        )
        return _order_not_found(order_number)
    customer = session.get(Customer, customer_id)
    return _detail(order, customer, datetime.now(timezone.utc))


def track_shipment(
    session: Session, order_number: str, customer_id: int
) -> dict | ToolError:
    """Delivery-focused view: ETA, carrier, tracking and the scan timeline."""
    order = _fetch(session, order_number, customer_id)
    if order is None:
        return _order_not_found(order_number)

    now = datetime.now(timezone.utc)
    eta = _as_utc(order.estimated_delivery_at)
    return {
        "order_number": order.order_number,
        "status": order.status.value,
        "status_label": STATUS_LABELS[order.status],
        "delivery_message": _delivery_message(order, now),
        "estimated_delivery_at": eta.isoformat() if eta else None,
        "estimated_delivery_date_readable": f"{eta:%A %d %B %Y}" if eta else None,
        "days_until_delivery": (eta.date() - now.date()).days if eta else None,
        "carrier": order.carrier or None,
        "tracking_number": order.tracking_number or None,
        "has_tracking": bool(order.tracking_number),
        "shipping_address": order.shipping_address,
        "timeline": [
            {
                "status": e.status.value,
                "label": STATUS_LABELS[e.status],
                "location": e.location,
                "note": e.note,
                "occurred_at": _as_utc(e.occurred_at).isoformat(),
            }
            for e in order.events
        ],
    }


def list_my_orders(
    session: Session, customer_id: int, limit: int = 5, status: str | None = None
) -> dict:
    statement = select(Order).where(Order.customer_id == customer_id)
    if status:
        try:
            statement = statement.where(Order.status == OrderStatus(status.strip().lower()))
        except ValueError:
            return {
                "orders": [],
                "count": 0,
                "note": (
                    f"'{status}' is not a valid order status. Valid values: "
                    + ", ".join(s.value for s in OrderStatus)
                ),
            }
    orders = session.scalars(
        statement.order_by(desc(Order.placed_at)).limit(max(1, min(limit, 20)))
    ).all()
    return {
        "orders": [_summary(o).model_dump(mode="json") for o in orders],
        "count": len(orders),
        "note": "" if orders else "This account has no orders matching that filter.",
    }


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

def cancel_order(
    session: Session, order_number: str, customer_id: int, reason: str = ""
) -> dict | ToolError:
    """Cancel an order, if its current state permits it.

    The state machine is enforced here rather than described to the model. A
    shipped parcel is physically in a carrier's van; no amount of conversational
    pressure should be able to mark it cancelled in the database.
    """
    order = _fetch(session, order_number, customer_id)
    if order is None:
        return _order_not_found(order_number)

    if not order.status.is_cancellable:
        return ToolError(
            error=(
                f"Order {order.order_number} is '{STATUS_LABELS[order.status]}' and can no "
                f"longer be cancelled."
            ),
            code="ORDER_NOT_CANCELLABLE",
            recovery_hint=(
                "Explain that cancellation is only possible before dispatch. If the order is "
                "shipped or out for delivery, the customer can refuse delivery; if it is "
                "delivered, direct them to the returns process."
            ),
            retryable=False,
        )

    now = datetime.now(timezone.utc)
    order.status = OrderStatus.CANCELLED
    order.cancelled_at = now
    order.estimated_delivery_at = None
    note = "Cancelled at customer request via the shopping assistant."
    if reason.strip():
        note = f"{note} Reason given: {reason.strip()[:120]}"
    session.add(
        OrderEvent(
            order_id=order.id, status=OrderStatus.CANCELLED,
            location="Aurelia Order Service", note=note, occurred_at=now,
        )
    )
    session.flush()

    logger.info(
        "orders.cancelled",
        extra={"order_number": order.order_number, "customer_id": customer_id},
    )
    refund_days = "1 to 3 business days" if order.payment_method.value != "wallet" else "immediately"
    return {
        "cancelled": True,
        "order_number": order.order_number,
        "cancelled_at": now.isoformat(),
        "refund_amount": Money.of(order.total_cents, order.currency).model_dump(),
        "refund_method": order.payment_method.value,
        "refund_expected": refund_days,
        "message": (
            f"Order {order.order_number} has been cancelled. A refund of "
            f"{Money.of(order.total_cents, order.currency).display} will be returned to your "
            f"{order.payment_method.value.replace('_', ' ')} {refund_days}."
        ),
    }


def request_return(
    session: Session, order_number: str, customer_id: int, reason: str = ""
) -> dict | ToolError:
    order = _fetch(session, order_number, customer_id)
    if order is None:
        return _order_not_found(order_number)

    customer = session.get(Customer, customer_id)
    now = datetime.now(timezone.utc)

    if order.status != OrderStatus.DELIVERED:
        return ToolError(
            error=(
                f"Order {order.order_number} is '{STATUS_LABELS[order.status]}'. Returns can "
                f"only be started once an order has been delivered."
            ),
            code="ORDER_NOT_DELIVERED",
            recovery_hint="Explain the current status and what the customer can do instead.",
        )
    if not _is_returnable(order, customer, now):
        delivered = _as_utc(order.delivered_at)
        window = (
            EXTENDED_RETURN_WINDOW_DAYS
            if customer.loyalty_tier.lower() in EXTENDED_TIERS
            else RETURN_WINDOW_DAYS
        )
        elapsed = (now - delivered).days if delivered else None
        return ToolError(
            error=(
                f"Order {order.order_number} is outside its {window}-day return window"
                + (f" (delivered {elapsed} days ago)." if elapsed is not None else ".")
            ),
            code="RETURN_WINDOW_EXPIRED",
            recovery_hint=(
                "Explain the window has closed. If the customer describes a fault rather than "
                "a change of mind, tell them faulty items are covered by warranty regardless "
                "of the return window and offer to escalate to Customer Care."
            ),
        )

    order.status = OrderStatus.RETURN_REQUESTED
    note = "Return requested via the shopping assistant."
    if reason.strip():
        note = f"{note} Reason: {reason.strip()[:120]}"
    session.add(
        OrderEvent(
            order_id=order.id, status=OrderStatus.RETURN_REQUESTED,
            location="Aurelia Returns", note=note, occurred_at=now,
        )
    )
    session.flush()

    logger.info(
        "orders.return_requested",
        extra={"order_number": order.order_number, "customer_id": customer_id},
    )
    return {
        "return_requested": True,
        "order_number": order.order_number,
        "requested_at": now.isoformat(),
        "refund_amount": Money.of(order.total_cents, order.currency).model_dump(),
        "next_step": "A prepaid return label will be emailed within one business day.",
        "message": (
            f"A return has been opened for order {order.order_number}. A prepaid label is on "
            f"its way by email, and the refund of "
            f"{Money.of(order.total_cents, order.currency).display} is issued once the items "
            f"pass inspection."
        ),
    }
