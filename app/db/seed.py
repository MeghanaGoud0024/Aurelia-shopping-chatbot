"""Deterministic synthetic dataset generator.

The assignment permits public or synthesised data. We synthesise, because a
generated catalogue lets us guarantee three things a scraped Kaggle CSV cannot:

1. **Referential integrity.** Every order line points at a real variant, every
   variant at a real product. Retrieval and transaction paths exercise the same
   rows.
2. **Reproducibility.** A fixed RNG seed means the reviewer's database is
   byte-identical to ours, so the documented example questions return the
   documented answers.
3. **Zero PII risk.** No real person's name, address, card or contact data ever
   enters the system, which satisfies the data-handling constraint outright.

The generator produces a full commerce graph: catalogue -> variants -> customers
-> orders -> line items -> shipment timeline.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import (
    Customer, Order, OrderEvent, OrderItem, OrderStatus, PaymentMethod,
    Product, ProductVariant,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Catalogue vocabulary
# ---------------------------------------------------------------------------

APPAREL_SIZES = ["XS", "S", "M", "L", "XL", "XXL"]
SHOE_SIZES = ["6", "7", "8", "9", "10", "11", "12"]
ONE_SIZE = ["ONE SIZE"]

COLORS = [
    "Black", "White", "Navy", "Charcoal", "Olive", "Burgundy", "Sand",
    "Cobalt Blue", "Forest Green", "Crimson", "Slate Grey", "Cream",
]

# (brand, positioning, price multiplier, house style)
BRANDS = [
    ("Nike", "performance sportswear", 1.00, "Dri-FIT"),
    ("Adidas", "performance sportswear", 0.95, "AEROREADY"),
    ("Puma", "athleisure", 0.80, "dryCELL"),
    ("Under Armour", "training", 0.92, "HeatGear"),
    ("New Balance", "running", 0.98, "NB Dry"),
    ("Levi's", "denim heritage", 0.90, "Cotton Twill"),
    ("Uniqlo", "everyday essentials", 0.55, "AIRism"),
    ("Zara", "fast fashion", 0.70, "Studio Cut"),
    ("H&M", "fast fashion", 0.45, "Everyday Soft"),
    ("Tommy Hilfiger", "premium casual", 1.25, "Classic Fit"),
    ("Calvin Klein", "premium casual", 1.30, "Modern Cotton"),
    ("The North Face", "outdoor", 1.45, "DryVent"),
    ("Columbia", "outdoor", 1.10, "Omni-Shade"),
    ("Ray-Ban", "eyewear", 1.60, "Polarised"),
    ("Fossil", "accessories", 1.35, "Stainless Steel"),
]

# category -> (subcategory, base price USD, size set, gender pool)
CATALOG_PLAN: dict[str, list[tuple[str, int, list[str], list[str]]]] = {
    "Topwear": [
        ("T-Shirt", 29, APPAREL_SIZES, ["men", "women", "unisex"]),
        ("Polo Shirt", 45, APPAREL_SIZES, ["men", "women"]),
        ("Hoodie", 69, APPAREL_SIZES, ["men", "women", "unisex"]),
        ("Sweatshirt", 59, APPAREL_SIZES, ["men", "women", "unisex"]),
        ("Casual Shirt", 55, APPAREL_SIZES, ["men", "women"]),
        ("Tank Top", 24, APPAREL_SIZES, ["men", "women"]),
    ],
    "Bottomwear": [
        ("Jeans", 79, APPAREL_SIZES, ["men", "women"]),
        ("Chinos", 65, APPAREL_SIZES, ["men", "women"]),
        ("Joggers", 55, APPAREL_SIZES, ["men", "women", "unisex"]),
        ("Shorts", 39, APPAREL_SIZES, ["men", "women", "unisex"]),
        ("Track Pants", 49, APPAREL_SIZES, ["men", "women", "unisex"]),
    ],
    "Footwear": [
        ("Running Shoes", 119, SHOE_SIZES, ["men", "women", "unisex"]),
        ("Sneakers", 95, SHOE_SIZES, ["men", "women", "unisex"]),
        ("Training Shoes", 105, SHOE_SIZES, ["men", "women"]),
        ("Sandals", 45, SHOE_SIZES, ["men", "women", "unisex"]),
    ],
    "Outerwear": [
        ("Jacket", 149, APPAREL_SIZES, ["men", "women", "unisex"]),
        ("Windbreaker", 99, APPAREL_SIZES, ["men", "women", "unisex"]),
        ("Puffer Jacket", 189, APPAREL_SIZES, ["men", "women"]),
    ],
    "Accessories": [
        ("Backpack", 79, ONE_SIZE, ["unisex"]),
        ("Cap", 29, ONE_SIZE, ["unisex"]),
        ("Sunglasses", 149, ONE_SIZE, ["men", "women", "unisex"]),
        ("Watch", 199, ONE_SIZE, ["men", "women"]),
        ("Socks 3-Pack", 19, ONE_SIZE, ["men", "women", "unisex"]),
    ],
}

FITS = ["Regular Fit", "Slim Fit", "Relaxed Fit", "Oversized", "Athletic Fit"]
MATERIALS = {
    "Topwear": ["100% Combed Cotton", "Cotton-Polyester Blend", "Organic Cotton Jersey", "Recycled Polyester"],
    "Bottomwear": ["Stretch Denim", "Cotton Twill", "Brushed Fleece", "Nylon Ripstop"],
    "Footwear": ["Engineered Mesh", "Full-Grain Leather", "Knit Upper with EVA Midsole", "Suede and Textile"],
    "Outerwear": ["Water-Repellent Nylon", "Recycled Down Fill", "Softshell Laminate", "Waxed Cotton"],
    "Accessories": ["Recycled Polyester", "Stainless Steel", "Acetate Frame", "Ripstop Nylon"],
}
CARE = [
    "Machine wash cold, tumble dry low.",
    "Machine wash cold with like colours, do not bleach.",
    "Hand wash recommended, line dry in shade.",
    "Wipe clean with a damp cloth.",
    "Dry clean only.",
]

COLLECTIONS = ["Core", "Essentials", "Pro", "Studio", "Heritage", "Everyday", "Elevate", "Trail", "Club", "Origin"]

FIRST_NAMES = [
    "Ava", "Liam", "Meghana", "Noah", "Priya", "Ethan", "Sofia", "Arjun",
    "Isabella", "Kai", "Amara", "Lucas", "Yuki", "Mateo", "Zara", "Omar",
    "Hana", "Diego", "Nina", "Rohan", "Elena", "Tomas", "Aisha", "Felix",
]
LAST_NAMES = [
    "Reyes", "Nakamura", "Iyer", "Okafor", "Fernandes", "Kowalski", "Haddad",
    "Lindqvist", "Moreau", "Silva", "Novak", "Bianchi", "Duarte", "Ahmed",
    "Petrova", "Larsen", "Costa", "Verma", "Chen", "Bakker",
]
CITIES = [
    ("Melbourne", "Australia"), ("Sydney", "Australia"), ("Singapore", "Singapore"),
    ("London", "United Kingdom"), ("Toronto", "Canada"), ("Austin", "United States"),
    ("Berlin", "Germany"), ("Dublin", "Ireland"), ("Auckland", "New Zealand"),
    ("Amsterdam", "Netherlands"), ("Lisbon", "Portugal"), ("Kuala Lumpur", "Malaysia"),
]
STREETS = ["Alder Street", "Marlowe Lane", "Kingsway", "Rowan Court", "Bellevue Road",
           "Harbour Parade", "Juniper Way", "Foxglove Crescent", "Sable Avenue", "Quarry Road"]

# The assignment's worked examples reference order 1234 ("what is the status",
# "when will it be delivered"). Leaving that to chance would mean the documented
# questions might land on a cancelled order. We pin a small set of order numbers
# to specific states so the README examples are reproducible, and let the other
# ~415 orders be drawn from the weighted distribution.
SHOWCASE_ORDERS: dict[str, OrderStatus] = {
    "1234": OrderStatus.SHIPPED,           # in transit, has an ETA
    "1201": OrderStatus.OUT_FOR_DELIVERY,  # arriving today
    "1288": OrderStatus.DELIVERED,         # completed, returnable
    "1305": OrderStatus.CONFIRMED,         # early enough to cancel
    "1350": OrderStatus.RETURN_REQUESTED,  # in the returns flow
}

# Showcase orders must belong to the account the demo session signs in as, or
# the documented examples return ORDER_NOT_FOUND - correctly, since the order
# would belong to someone else. Owning them is what makes "cancel order 1305"
# demonstrable rather than a demonstration of the authorisation boundary.
#
# The history below gives that same account a plausible purchase record so the
# dashboard has something real to chart: statuses spread across the funnel and
# orders spread across eight months. Values are (status, days_ago).
DEMO_HISTORY_ORDERS: dict[str, tuple[OrderStatus, int]] = {
    "1019": (OrderStatus.DELIVERED, 232),
    "1044": (OrderStatus.DELIVERED, 201),
    "1071": (OrderStatus.RETURNED, 178),
    "1096": (OrderStatus.DELIVERED, 154),
    "1123": (OrderStatus.DELIVERED, 131),
    "1147": (OrderStatus.CANCELLED, 118),
    "1168": (OrderStatus.DELIVERED, 96),
    "1189": (OrderStatus.DELIVERED, 74),
    "1212": (OrderStatus.DELIVERED, 58),
    "1247": (OrderStatus.DELIVERED, 41),
    "1266": (OrderStatus.DELIVERED, 27),
    "1319": (OrderStatus.PACKED, 2),
    "1372": (OrderStatus.PENDING_PAYMENT, 0),
}

CARRIERS = ["Aurelia Express", "SwiftPost", "GlobalShip", "MetroCourier", "BluePeak Logistics"]
HUBS = ["Regional Sortation Hub", "National Distribution Centre", "Local Delivery Depot",
        "International Gateway", "City Fulfilment Centre"]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _unique_name(preferred: str, alternates: list[str], used: set[str]) -> str:
    """Pick the first unused product name, falling back to a numbered suffix.

    Two products with the same display name is a real usability defect: the
    customer cannot tell the search results apart and the assistant cannot refer
    to one unambiguously. We disambiguate with the fit or fabric technology,
    which is how retailers actually differentiate lines within a collection.
    """
    for candidate in [preferred, *alternates]:
        if candidate not in used:
            used.add(candidate)
            return candidate
    suffix = 2
    while f"{preferred} {suffix:02d}" in used:
        suffix += 1
    final = f"{preferred} {suffix:02d}"
    used.add(final)
    return final


def _color_code(color: str) -> str:
    """Stable short code for a colour name, used in variant SKUs.

    Truncating to two characters collides ("Cream"/"Crimson"), so we take six
    characters of the de-spaced name, which is unique across the palette.
    """
    return color.replace(" ", "").upper()[:6]


def _price_cents(base_usd: int, multiplier: float, rng: random.Random) -> tuple[int, int]:
    """Return (sale_price_cents, list_price_cents) with realistic .99 endings."""
    raw = base_usd * multiplier * rng.uniform(0.85, 1.35)
    list_price = max(9, round(raw)) - 0.01
    discount = rng.choice([0.0, 0.0, 0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
    sale = list_price * (1 - discount)
    sale = round(sale) - 0.01 if discount else list_price
    return int(round(sale * 100)), int(round(list_price * 100))


def _build_products(rng: random.Random) -> list[Product]:
    products: list[Product] = []
    used_names: set[str] = set()
    counter = 1000

    for category, plans in CATALOG_PLAN.items():
        for subcategory, base_price, sizes, genders in plans:
            for brand, positioning, multiplier, house_style in BRANDS:
                # Not every brand sells every subcategory. Keep it plausible.
                if brand in {"Ray-Ban"} and subcategory != "Sunglasses":
                    continue
                if brand == "Fossil" and subcategory not in {"Watch", "Backpack"}:
                    continue
                if subcategory == "Sunglasses" and brand not in {"Ray-Ban", "Nike", "Adidas", "Zara"}:
                    continue
                if subcategory == "Watch" and brand not in {"Fossil", "Calvin Klein", "Tommy Hilfiger"}:
                    continue
                if category == "Outerwear" and brand in {"H&M"} and rng.random() < 0.5:
                    continue

                # 2-4 distinct product lines per brand/subcategory pair, so a
                # query like "Nike t-shirts" returns a real shortlist rather
                # than a single row.
                for _ in range(rng.randint(3, 5)):
                    counter += 1
                    gender = rng.choice(genders)
                    collection = rng.choice(COLLECTIONS)
                    fit = rng.choice(FITS)
                    material = rng.choice(MATERIALS[category])
                    sale_cents, list_cents = _price_cents(base_price, multiplier, rng)

                    gender_label = {"men": "Men's", "women": "Women's", "unisex": "Unisex"}[gender]
                    name = _unique_name(
                        f"{brand} {collection} {gender_label} {subcategory}",
                        alternates=[
                            f"{brand} {collection} {fit.split()[0]} {gender_label} {subcategory}",
                            f"{brand} {collection} {house_style} {gender_label} {subcategory}",
                        ],
                        used=used_names,
                    )

                    description = (
                        f"{brand} {collection} {subcategory.lower()} built for {positioning}. "
                        f"{fit} silhouette in {material.lower()}, finished with {brand}'s {house_style} "
                        f"treatment for all-day comfort. A dependable {category.lower()} piece that "
                        f"moves easily between training, travel and everyday wear."
                    )
                    tags = ", ".join(
                        {
                            subcategory.lower(), category.lower(), brand.lower(), gender,
                            fit.lower(), collection.lower(), house_style.lower(),
                            positioning.split()[0],
                        }
                    )

                    product = Product(
                        sku=f"AUR-{counter}",
                        name=name,
                        brand=brand,
                        category=category,
                        subcategory=subcategory,
                        gender=gender,
                        description=description,
                        material=material,
                        care=rng.choice(CARE),
                        price_cents=sale_cents,
                        list_price_cents=list_cents,
                        currency="USD",
                        rating=round(rng.uniform(3.4, 4.9), 1),
                        review_count=rng.randint(8, 2400),
                        tags=tags,
                        is_active=True,
                    )

                    # Variants: 2-4 colours across the size run.
                    chosen_colors = rng.sample(COLORS, rng.randint(2, 4))
                    for color in chosen_colors:
                        for size in sizes:
                            # Deliberately leave some combinations out of stock so
                            # availability questions have a real answer.
                            roll = rng.random()
                            stock = 0 if roll < 0.13 else rng.randint(1, 60)
                            product.variants.append(
                                ProductVariant(
                                    sku=f"AUR-{counter}-{_color_code(color)}-{size.replace(' ', '')}",
                                    size=size,
                                    color=color,
                                    stock=stock,
                                )
                            )
                    products.append(product)
    return products


def _build_customers(rng: random.Random, count: int) -> list[Customer]:
    customers: list[Customer] = []
    used_emails: set[str] = set()
    for i in range(count):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        city, country = rng.choice(CITIES)
        email = f"{first.lower()}.{last.lower()}{i}@example.com"
        if email in used_emails:
            continue
        used_emails.add(email)
        customers.append(
            Customer(
                public_id=f"CUST-{5000 + i}",
                full_name=f"{first} {last}",
                email=email,
                phone=f"+1-555-{rng.randint(1000, 9999)}",
                city=city,
                country=country,
                loyalty_tier=rng.choices(
                    ["standard", "silver", "gold", "platinum"], weights=[60, 22, 13, 5]
                )[0],
            )
        )
    return customers


def _status_timeline(status: OrderStatus) -> list[OrderStatus]:
    """The ordered set of states an order passed through to reach `status`."""
    happy = [
        OrderStatus.CONFIRMED, OrderStatus.PACKED, OrderStatus.SHIPPED,
        OrderStatus.OUT_FOR_DELIVERY, OrderStatus.DELIVERED,
    ]
    if status == OrderStatus.PENDING_PAYMENT:
        return [OrderStatus.PENDING_PAYMENT]
    if status == OrderStatus.CANCELLED:
        return [OrderStatus.CONFIRMED, OrderStatus.CANCELLED]
    if status in {OrderStatus.RETURN_REQUESTED, OrderStatus.RETURNED}:
        chain = happy + [OrderStatus.RETURN_REQUESTED]
        if status == OrderStatus.RETURNED:
            chain.append(OrderStatus.RETURNED)
        return chain
    return happy[: happy.index(status) + 1]


def _event_note(status: OrderStatus, carrier: str) -> tuple[str, str]:
    return {
        OrderStatus.PENDING_PAYMENT: ("Checkout", "Order created, awaiting payment authorisation."),
        OrderStatus.CONFIRMED: ("Aurelia Order Service", "Payment authorised, order confirmed."),
        OrderStatus.PACKED: ("City Fulfilment Centre", "Items picked and packed."),
        OrderStatus.SHIPPED: (f"{carrier} Origin Hub", f"Handed to {carrier}, in transit."),
        OrderStatus.OUT_FOR_DELIVERY: ("Local Delivery Depot", "Out for delivery with the courier."),
        OrderStatus.DELIVERED: ("Delivery address", "Delivered and signed for."),
        OrderStatus.CANCELLED: ("Aurelia Order Service", "Cancelled at customer request, refund initiated."),
        OrderStatus.RETURN_REQUESTED: ("Aurelia Returns", "Return requested, pickup being scheduled."),
        OrderStatus.RETURNED: ("Returns Processing Centre", "Return received, refund issued."),
    }[status]


def _build_orders(
    rng: random.Random, customers: list[Customer], variants: list[ProductVariant], now: datetime
) -> list[Order]:
    orders: list[Order] = []
    order_number = 1000  # first generated order is #1001; the docs use #1234

    status_pool = [
        (OrderStatus.DELIVERED, 40), (OrderStatus.SHIPPED, 18),
        (OrderStatus.OUT_FOR_DELIVERY, 8), (OrderStatus.PACKED, 8),
        (OrderStatus.CONFIRMED, 12), (OrderStatus.PENDING_PAYMENT, 4),
        (OrderStatus.CANCELLED, 5), (OrderStatus.RETURN_REQUESTED, 3),
        (OrderStatus.RETURNED, 2),
    ]
    statuses = [s for s, _ in status_pool]
    weights = [w for _, w in status_pool]

    # The account the demo session signs in as. Deterministic, so the documented
    # examples always resolve to the signed-in customer.
    demo_customer = customers[0]

    for _ in range(420):
        order_number += 1
        key = str(order_number)
        days_override: int | None = None

        if key in SHOWCASE_ORDERS:
            customer, status = demo_customer, SHOWCASE_ORDERS[key]
        elif key in DEMO_HISTORY_ORDERS:
            customer = demo_customer
            status, days_override = DEMO_HISTORY_ORDERS[key]
        else:
            customer = rng.choice(customers)
            status = rng.choices(statuses, weights=weights)[0]
        carrier = rng.choice(CARRIERS)

        # Age the order so that in-flight statuses are recent and delivered
        # orders sit further back. An order cannot be "delivered" tomorrow.
        if status in {OrderStatus.DELIVERED, OrderStatus.RETURNED, OrderStatus.RETURN_REQUESTED}:
            days_ago = rng.randint(9, 120)
        elif status in {OrderStatus.SHIPPED, OrderStatus.OUT_FOR_DELIVERY}:
            days_ago = rng.randint(1, 6)
        elif status == OrderStatus.CANCELLED:
            days_ago = rng.randint(2, 60)
        else:
            days_ago = rng.randint(0, 3)
        if days_override is not None:
            days_ago = days_override
        placed_at = now - timedelta(days=days_ago, hours=rng.randint(0, 23), minutes=rng.randint(0, 59))

        street = f"{rng.randint(1, 480)} {rng.choice(STREETS)}"
        address = f"{street}, {customer.city}, {customer.country}"

        order = Order(
            order_number=str(order_number),
            customer_id=customer.id,
            status=status,
            payment_method=rng.choice(list(PaymentMethod)),
            currency="USD",
            shipping_address=address,
            carrier=carrier if status not in {OrderStatus.PENDING_PAYMENT} else "",
            tracking_number=(
                f"{carrier.split()[0][:3].upper()}{rng.randint(10**9, 10**10 - 1)}"
                if status in {OrderStatus.SHIPPED, OrderStatus.OUT_FOR_DELIVERY,
                              OrderStatus.DELIVERED, OrderStatus.RETURN_REQUESTED,
                              OrderStatus.RETURNED}
                else ""
            ),
            placed_at=placed_at,
            channel=rng.choices(["web", "mobile", "assistant"], weights=[55, 35, 10])[0],
        )

        subtotal = 0
        for variant in rng.sample(variants, rng.randint(1, 4)):
            quantity = rng.choices([1, 1, 1, 2, 3], weights=[55, 20, 10, 10, 5])[0]
            unit_price = variant.product.price_cents
            subtotal += unit_price * quantity
            order.items.append(
                OrderItem(
                    variant_id=variant.id,
                    product_name=variant.product.name,
                    brand=variant.product.brand,
                    size=variant.size,
                    color=variant.color,
                    quantity=quantity,
                    unit_price_cents=unit_price,
                )
            )

        shipping = 0 if subtotal >= 7500 else 799
        tax = round(subtotal * 0.10)
        order.subtotal_cents = subtotal
        order.shipping_cents = shipping
        order.tax_cents = tax
        order.total_cents = subtotal + shipping + tax

        # Build the shipment timeline, then derive delivery dates from it.
        timeline = _status_timeline(status)
        cursor = placed_at
        for index, step in enumerate(timeline):
            if index:
                cursor += timedelta(hours=rng.randint(6, 42))
            location, note = _event_note(step, carrier)
            if step in {OrderStatus.SHIPPED} and rng.random() < 0.6:
                location = f"{carrier} {rng.choice(HUBS)}"
            order.events.append(
                OrderEvent(status=step, location=location, note=note, occurred_at=cursor)
            )

        if status == OrderStatus.DELIVERED:
            order.delivered_at = cursor
            order.estimated_delivery_at = cursor - timedelta(hours=rng.randint(0, 20))
        elif status == OrderStatus.CANCELLED:
            order.cancelled_at = cursor
        elif status in {OrderStatus.RETURN_REQUESTED, OrderStatus.RETURNED}:
            order.delivered_at = order.events[4].occurred_at
            order.estimated_delivery_at = order.delivered_at
        elif status == OrderStatus.PENDING_PAYMENT:
            order.estimated_delivery_at = None
        else:
            remaining = {
                OrderStatus.CONFIRMED: rng.randint(4, 8),
                OrderStatus.PACKED: rng.randint(3, 6),
                OrderStatus.SHIPPED: rng.randint(1, 4),
                OrderStatus.OUT_FOR_DELIVERY: 0,
            }[status]
            eta = (now + timedelta(days=remaining)).replace(hour=18, minute=0, second=0, microsecond=0)
            order.estimated_delivery_at = eta

        orders.append(order)
    return orders


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def database_is_seeded(session: Session) -> bool:
    return session.scalar(select(Product.id).limit(1)) is not None


def seed_database(session: Session, *, force: bool = False) -> dict[str, int]:
    """Populate the database. Idempotent unless `force` is set."""
    if database_is_seeded(session) and not force:
        logger.info("seed.skipped", extra={"reason": "database already populated"})
        return _counts(session)

    if force:
        for model in (OrderEvent, OrderItem, Order, ProductVariant, Product, Customer):
            session.query(model).delete()
        session.flush()

    rng = random.Random(settings.seed)
    now = datetime.now(timezone.utc)

    products = _build_products(rng)
    session.add_all(products)
    session.flush()

    customers = _build_customers(rng, 180)
    session.add_all(customers)
    session.flush()

    variants = [v for p in products for v in p.variants]
    orders = _build_orders(rng, customers, variants, now)
    session.add_all(orders)
    session.flush()

    counts = _counts(session)
    logger.info("seed.completed", extra=counts)
    return counts


def _counts(session: Session) -> dict[str, int]:
    from sqlalchemy import func

    return {
        "products": session.scalar(select(func.count(Product.id))) or 0,
        "variants": session.scalar(select(func.count(ProductVariant.id))) or 0,
        "customers": session.scalar(select(func.count(Customer.id))) or 0,
        "orders": session.scalar(select(func.count(Order.id))) or 0,
        "order_items": session.scalar(select(func.count(OrderItem.id))) or 0,
        "order_events": session.scalar(select(func.count(OrderEvent.id))) or 0,
    }
