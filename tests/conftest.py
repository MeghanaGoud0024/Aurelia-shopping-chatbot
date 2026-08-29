"""Shared test fixtures.

Every test runs against a fresh in-memory SQLite database seeded with a small
deterministic catalogue. That keeps the suite fast and, more importantly, keeps
it independent of the developer's local `data/aurelia.db`: a test that passes
only because someone's database happens to contain the right row is not a test.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Configure the app before any of its modules are imported, because settings are
# read once at import time.
#
# The application database is a temporary *file*, not ":memory:". An in-memory
# SQLite database is private to a single connection, so the app would create its
# schema on one pooled connection and then serve a request on another that has
# no tables. Unit-test fixtures below use a separate in-memory engine pinned to
# one connection with StaticPool, which is safe because they hold that session
# open for the life of the test.
_TEST_DB = Path(tempfile.mkdtemp(prefix="aurelia-test-")) / "test.db"
os.environ["AURELIA_DATABASE_URL"] = f"sqlite:///{_TEST_DB}"
os.environ.setdefault("AURELIA_LOG_LEVEL", "WARNING")
# No credentials in the suite: turns run the deterministic planner, so CI needs
# no secrets and tests never depend on a live model's wording.
os.environ["AURELIA_LLM_API_KEY"] = ""

from sqlalchemy import create_engine                     # noqa: E402
from sqlalchemy.orm import sessionmaker                  # noqa: E402
from sqlalchemy.pool import StaticPool                   # noqa: E402

from app.db.models import (                              # noqa: E402
    Base, Customer, Order, OrderEvent, OrderItem, OrderStatus, PaymentMethod,
    Product, ProductVariant,
)
from app.retrieval.index import retrieval_service        # noqa: E402


@pytest.fixture(scope="session")
def engine():
    # StaticPool keeps every connection pointed at the same in-memory database.
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


@pytest.fixture
def session(engine):
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    db = factory()
    try:
        yield db
    finally:
        db.rollback()
        db.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def catalogue(session):
    """A small, fully specified catalogue.

    Hand-built rather than generated, so each test can assert on exact values
    and a change to the synthetic data generator cannot silently change what a
    test means.
    """
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)

    nike_tee = Product(
        sku="T-001", name="Nike Core Unisex T-Shirt", brand="Nike",
        category="Topwear", subcategory="T-Shirt", gender="unisex",
        description="Lightweight Dri-FIT cotton tee for training and everyday wear.",
        material="Cotton-Polyester Blend", care="Machine wash cold.",
        price_cents=2699, list_price_cents=3499, rating=4.6, review_count=812,
        tags="t-shirt, topwear, nike, unisex, dri-fit",
    )
    nike_tee.variants = [
        ProductVariant(sku="T-001-BLACK-M", size="M", color="Black", stock=14),
        ProductVariant(sku="T-001-BLACK-L", size="L", color="Black", stock=0),
        ProductVariant(sku="T-001-NAVY-M", size="M", color="Navy", stock=3),
    ]

    adidas_shoe = Product(
        sku="S-001", name="Adidas Pro Men's Running Shoes", brand="Adidas",
        category="Footwear", subcategory="Running Shoes", gender="men",
        description="Engineered mesh upper with responsive cushioning.",
        material="Engineered Mesh", care="Wipe clean.",
        price_cents=8999, list_price_cents=8999, rating=4.2, review_count=310,
        tags="running shoes, footwear, adidas, men",
    )
    adidas_shoe.variants = [
        ProductVariant(sku="S-001-WHITE-9", size="9", color="White", stock=6),
        ProductVariant(sku="S-001-WHITE-10", size="10", color="White", stock=0),
    ]

    levis = Product(
        sku="J-001", name="Levi's Heritage Men's Jeans", brand="Levi's",
        category="Bottomwear", subcategory="Jeans", gender="men",
        description="Straight-leg stretch denim.", material="Stretch Denim",
        care="Machine wash cold.", price_cents=7999, list_price_cents=9999,
        rating=4.4, review_count=1204, tags="jeans, denim, bottomwear, levis",
    )
    levis.variants = [ProductVariant(sku="J-001-INDIGO-32", size="M", color="Navy", stock=9)]

    session.add_all([nike_tee, adidas_shoe, levis])
    session.flush()

    alice = Customer(
        public_id="CUST-1", full_name="Alice Tester", email="alice@example.com",
        city="Melbourne", country="Australia", loyalty_tier="standard",
    )
    mallory = Customer(
        public_id="CUST-2", full_name="Mallory Other", email="mallory@example.com",
        city="Berlin", country="Germany", loyalty_tier="gold",
    )
    session.add_all([alice, mallory])
    session.flush()

    def make_order(number, customer, status, *, placed_days_ago=3, eta_days=None,
                   delivered_days_ago=None, variant=None, quantity=1):
        order = Order(
            order_number=number, customer_id=customer.id, status=status,
            payment_method=PaymentMethod.CARD, shipping_address="1 Test Street, Melbourne",
            carrier="Aurelia Express", tracking_number="AUR123456789",
            placed_at=now - timedelta(days=placed_days_ago),
            estimated_delivery_at=(now + timedelta(days=eta_days)) if eta_days is not None else None,
            delivered_at=(now - timedelta(days=delivered_days_ago)) if delivered_days_ago is not None else None,
        )
        variant = variant or nike_tee.variants[0]
        order.items.append(
            OrderItem(
                variant_id=variant.id, product_name=variant.product.name,
                brand=variant.product.brand, size=variant.size, color=variant.color,
                quantity=quantity, unit_price_cents=variant.product.price_cents,
            )
        )
        order.subtotal_cents = variant.product.price_cents * quantity
        order.shipping_cents = 799
        order.tax_cents = round(order.subtotal_cents * 0.10)
        order.total_cents = order.subtotal_cents + order.shipping_cents + order.tax_cents
        order.events.append(
            OrderEvent(status=status, location="Test", note="Seeded", occurred_at=order.placed_at)
        )
        session.add(order)
        return order

    make_order("1001", alice, OrderStatus.SHIPPED, eta_days=2)
    make_order("1002", alice, OrderStatus.CONFIRMED, placed_days_ago=0, eta_days=5)
    make_order("1003", alice, OrderStatus.DELIVERED, placed_days_ago=10, delivered_days_ago=5)
    make_order("1004", alice, OrderStatus.DELIVERED, placed_days_ago=120, delivered_days_ago=100)
    make_order("2001", mallory, OrderStatus.SHIPPED, eta_days=1)
    session.flush()

    retrieval_service.build(session)

    return {
        "nike_tee": nike_tee, "adidas_shoe": adidas_shoe, "levis": levis,
        "alice": alice, "mallory": mallory,
    }
