"""Dashboard aggregation.

Everything the home view renders is computed here rather than in the browser,
for the same reason the assistant never does arithmetic: money maths and date
maths belong in one tested place. The client receives values that are already
correct and already formatted, and its only job is to draw them.

Every figure is scoped to the signed-in customer, using the same predicate the
order service uses. The dashboard is not a privileged view.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Customer, Order, OrderStatus, Product
from app.schemas import Money
from app.services import cart as cart_service
from app.services.orders import STATUS_LABELS, _as_utc, _delivery_message

#: Statuses that mean a parcel is somewhere between the warehouse and the door.
IN_FLIGHT = {
    OrderStatus.CONFIRMED, OrderStatus.PACKED,
    OrderStatus.SHIPPED, OrderStatus.OUT_FOR_DELIVERY,
}

#: Ordered for display, so the donut segments do not reshuffle between loads.
STATUS_DISPLAY_ORDER = [
    OrderStatus.DELIVERED, OrderStatus.SHIPPED, OrderStatus.OUT_FOR_DELIVERY,
    OrderStatus.PACKED, OrderStatus.CONFIRMED, OrderStatus.PENDING_PAYMENT,
    OrderStatus.RETURN_REQUESTED, OrderStatus.RETURNED, OrderStatus.CANCELLED,
]


def _greeting(now: datetime) -> str:
    hour = now.hour
    if hour < 12:
        return "Good morning"
    if hour < 18:
        return "Good afternoon"
    return "Good evening"


def _hero(session: Session, session_id: str, orders: list[Order], now: datetime) -> dict:
    """The headline card.

    Priority reflects what the customer most likely came to find out: an
    unfinished basket first, then a parcel in flight, then an invitation to
    browse. A dashboard that leads with a static welcome when a parcel is
    arriving today is answering the wrong question.
    """
    cart = cart_service.view_cart(session, session_id)
    if cart.lines:
        threshold = cart_service.FREE_SHIPPING_THRESHOLD_CENTS
        subtotal = cart.subtotal.amount_cents
        remaining = max(0, threshold - subtotal)
        return {
            "kind": "bag",
            "eyebrow": "In your bag",
            "title": (
                "You have free shipping" if remaining == 0
                else f"{Money.of(remaining).display} from free shipping"
            ),
            "caption": (
                f"{cart.item_count} item{'' if cart.item_count == 1 else 's'}, "
                f"{cart.total.display} including shipping and tax"
            ),
            "progress_pct": 100 if remaining == 0 else round(subtotal / threshold * 100),
            "value_label": cart.subtotal.display,
            "target_label": Money.of(threshold).display,
            "action": "Ask me to check out when you are ready",
        }

    in_flight = [o for o in orders if o.status in IN_FLIGHT and o.estimated_delivery_at]
    if in_flight:
        soonest = min(in_flight, key=lambda o: _as_utc(o.estimated_delivery_at))
        eta = _as_utc(soonest.estimated_delivery_at)
        days = (eta.date() - now.date()).days
        # A five-day dispatch window is the reference span for the progress arc.
        elapsed = max(0, 5 - max(days, 0))
        return {
            "kind": "delivery",
            "eyebrow": f"Order {soonest.order_number}",
            "title": (
                "Arriving today" if days <= 0
                else "Arriving tomorrow" if days == 1
                else f"Arriving in {days} days"
            ),
            "caption": _delivery_message(soonest, now),
            "progress_pct": min(100, round(elapsed / 5 * 100)),
            "value_label": STATUS_LABELS[soonest.status],
            "target_label": "Delivered",
            "action": f"Ask me to track order {soonest.order_number}",
        }

    return {
        "kind": "explore",
        "eyebrow": "Nothing in flight",
        "title": "Let's find something",
        "caption": "No open orders and an empty bag. Tell me what you are looking for.",
        "progress_pct": 0,
        "value_label": "0",
        "target_label": "",
        "action": "Try 'show me Nike t-shirts'",
    }


def _spend_by_month(orders: list[Order], now: datetime, months: int = 6) -> dict:
    """Spend per calendar month, excluding money that was never taken."""
    counted = [
        o for o in orders
        if o.status not in {OrderStatus.CANCELLED, OrderStatus.PENDING_PAYMENT}
    ]

    buckets: list[tuple[str, int]] = []
    cursor = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    keys: list[tuple[int, int]] = []
    for _ in range(months):
        keys.append((cursor.year, cursor.month))
        buckets.append((f"{cursor:%b}", 0))
        # Step back one month without a calendar library.
        cursor = (cursor - timedelta(days=1)).replace(day=1)
    keys.reverse()
    buckets.reverse()

    totals = Counter()
    for order in counted:
        placed = _as_utc(order.placed_at)
        totals[(placed.year, placed.month)] += order.total_cents

    points = [
        {
            "label": label,
            "amount_cents": totals.get(key, 0),
            "display": Money.of(totals.get(key, 0)).display,
        }
        for (label, _zero), key in zip(buckets, keys, strict=True)
    ]
    peak = max((p["amount_cents"] for p in points), default=0)
    for point in points:
        point["pct"] = round(point["amount_cents"] / peak * 100) if peak else 0

    window_start = now - timedelta(days=365)
    twelve_month = sum(
        o.total_cents for o in counted if _as_utc(o.placed_at) >= window_start
    )
    return {
        "points": points,
        "twelve_month_total": Money.of(twelve_month).model_dump(),
        "lifetime_total": Money.of(sum(o.total_cents for o in counted)).model_dump(),
    }


def build_dashboard(session: Session, *, customer_id: int, session_id: str) -> dict:
    now = datetime.now(timezone.utc)
    customer = session.get(Customer, customer_id)

    orders = session.scalars(
        select(Order).where(Order.customer_id == customer_id).order_by(Order.placed_at.desc())
    ).all()

    counts = Counter(o.status for o in orders)
    total_orders = len(orders)
    breakdown = [
        {
            "status": status.value,
            "label": STATUS_LABELS[status],
            "count": counts[status],
            "pct": round(counts[status] / total_orders * 100) if total_orders else 0,
        }
        for status in STATUS_DISPLAY_ORDER
        if counts[status]
    ]

    in_flight = [o for o in orders if o.status in IN_FLIGHT and o.estimated_delivery_at]
    next_delivery = None
    if in_flight:
        soonest = min(in_flight, key=lambda o: _as_utc(o.estimated_delivery_at))
        eta = _as_utc(soonest.estimated_delivery_at)
        next_delivery = {
            "order_number": soonest.order_number,
            "status": soonest.status.value,
            "status_label": STATUS_LABELS[soonest.status],
            "carrier": soonest.carrier,
            "date_readable": f"{eta:%a %d %b}",
            "days": (eta.date() - now.date()).days,
            "message": _delivery_message(soonest, now),
        }

    recent = [
        {
            "order_number": o.order_number,
            "status": o.status.value,
            "status_label": STATUS_LABELS[o.status],
            "placed_readable": f"{_as_utc(o.placed_at):%d %b %Y}",
            "total": Money.of(o.total_cents, o.currency).model_dump(),
            "item_count": sum(i.quantity for i in o.items),
            "first_item": o.items[0].product_name if o.items else "",
        }
        for o in orders[:5]
    ]

    cart = cart_service.view_cart(session, session_id)
    product_count = session.scalar(
        select(func.count(Product.id)).where(Product.is_active.is_(True))
    ) or 0
    brand_count = session.scalar(
        select(func.count(func.distinct(Product.brand))).where(Product.is_active.is_(True))
    ) or 0

    first_name = (customer.full_name or "there").split()[0]
    initials = "".join(part[0] for part in (customer.full_name or "A").split()[:2]).upper()

    return {
        "customer": {
            "name": customer.full_name,
            "first_name": first_name,
            "initials": initials,
            "public_id": customer.public_id,
            "tier": customer.loyalty_tier,
            "city": customer.city,
            "country": customer.country,
        },
        "greeting": _greeting(now),
        "today_readable": f"{now:%A %d %B}",
        "hero": _hero(session, session_id, orders, now),
        "stats": [
            {"key": "orders", "label": "Orders", "value": str(total_orders)},
            {"key": "in_flight", "label": "In transit",
             "value": str(sum(counts[s] for s in IN_FLIGHT))},
            {"key": "delivered", "label": "Delivered", "value": str(counts[OrderStatus.DELIVERED])},
            {"key": "bag", "label": "In bag", "value": str(cart.item_count)},
        ],
        "status_breakdown": breakdown,
        "next_delivery": next_delivery,
        "recent_orders": recent,
        "spend": _spend_by_month(orders, now),
        "catalogue": {"products": product_count, "brands": brand_count},
    }
