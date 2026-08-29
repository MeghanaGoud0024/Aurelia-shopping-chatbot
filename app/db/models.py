"""Relational schema for the shopping domain.

Design notes
------------
* Products are modelled with a **variant** table (size / colour / SKU-level
  stock) because "is the Nike tee available in medium?" is a variant question,
  not a product question. Collapsing them would force the LLM to guess.
* Orders carry an immutable `unit_price_cents` snapshot on the line item. Price
  is a fact at time of sale, not a live join to the catalogue.
* Money is stored as integer cents. Float currency is a correctness bug waiting
  to happen.
* `ToolInvocation` and `ChatMessage` give us a complete audit trail: for any
  assistant sentence we can show which backend call produced the underlying
  fact. That is the backbone of the explainability story.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, CheckConstraint, DateTime, Enum, Float, ForeignKey, Index,
    Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class OrderStatus(str, enum.Enum):
    PENDING_PAYMENT = "pending_payment"
    CONFIRMED = "confirmed"
    PACKED = "packed"
    SHIPPED = "shipped"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    RETURN_REQUESTED = "return_requested"
    RETURNED = "returned"

    @property
    def is_terminal(self) -> bool:
        return self in {OrderStatus.DELIVERED, OrderStatus.CANCELLED, OrderStatus.RETURNED}

    @property
    def is_cancellable(self) -> bool:
        return self in {OrderStatus.PENDING_PAYMENT, OrderStatus.CONFIRMED, OrderStatus.PACKED}


class PaymentMethod(str, enum.Enum):
    CARD = "card"
    UPI = "upi"
    NET_BANKING = "net_banking"
    COD = "cash_on_delivery"
    WALLET = "wallet"


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True)
    sku: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    brand: Mapped[str] = mapped_column(String(80), index=True)
    category: Mapped[str] = mapped_column(String(80), index=True)
    subcategory: Mapped[str] = mapped_column(String(80), index=True)
    gender: Mapped[str] = mapped_column(String(20), default="unisex")
    description: Mapped[str] = mapped_column(Text, default="")
    material: Mapped[str] = mapped_column(String(120), default="")
    care: Mapped[str] = mapped_column(String(200), default="")
    price_cents: Mapped[int] = mapped_column(Integer)
    list_price_cents: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    rating: Mapped[float] = mapped_column(Float, default=0.0)
    review_count: Mapped[int] = mapped_column(Integer, default=0)
    tags: Mapped[str] = mapped_column(Text, default="")  # comma separated
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    variants: Mapped[list["ProductVariant"]] = relationship(
        back_populates="product", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        CheckConstraint("price_cents >= 0", name="ck_product_price_non_negative"),
        Index("ix_product_brand_category", "brand", "category"),
    )

    @property
    def discount_pct(self) -> int:
        if self.list_price_cents <= 0 or self.list_price_cents <= self.price_cents:
            return 0
        return round((1 - self.price_cents / self.list_price_cents) * 100)

    @property
    def total_stock(self) -> int:
        return sum(v.stock for v in self.variants)

    @property
    def tag_list(self) -> list[str]:
        return [t.strip() for t in self.tags.split(",") if t.strip()]

    def search_document(self) -> str:
        """Flattened text used to build the lexical retrieval index."""
        return " ".join(
            [
                self.name, self.brand, self.category, self.subcategory,
                self.gender, self.material, self.description, self.tags,
                " ".join({v.color for v in self.variants}),
                " ".join({v.size for v in self.variants}),
            ]
        )


class ProductVariant(Base):
    __tablename__ = "product_variants"

    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"), index=True)
    sku: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    size: Mapped[str] = mapped_column(String(20))
    color: Mapped[str] = mapped_column(String(40))
    stock: Mapped[int] = mapped_column(Integer, default=0)

    product: Mapped[Product] = relationship(back_populates="variants")

    __table_args__ = (
        UniqueConstraint("product_id", "size", "color", name="uq_variant_combination"),
        CheckConstraint("stock >= 0", name="ck_variant_stock_non_negative"),
    )

    @property
    def in_stock(self) -> bool:
        return self.stock > 0


# ---------------------------------------------------------------------------
# Customers & orders
# ---------------------------------------------------------------------------

class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    phone: Mapped[str] = mapped_column(String(40), default="")
    city: Mapped[str] = mapped_column(String(80), default="")
    country: Mapped[str] = mapped_column(String(80), default="")
    loyalty_tier: Mapped[str] = mapped_column(String(20), default="standard")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    orders: Mapped[list["Order"]] = relationship(back_populates="customer", lazy="selectin")


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_number: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    status: Mapped[OrderStatus] = mapped_column(Enum(OrderStatus), default=OrderStatus.CONFIRMED, index=True)
    payment_method: Mapped[PaymentMethod] = mapped_column(Enum(PaymentMethod), default=PaymentMethod.CARD)
    subtotal_cents: Mapped[int] = mapped_column(Integer, default=0)
    shipping_cents: Mapped[int] = mapped_column(Integer, default=0)
    tax_cents: Mapped[int] = mapped_column(Integer, default=0)
    total_cents: Mapped[int] = mapped_column(Integer, default=0)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    shipping_address: Mapped[str] = mapped_column(Text, default="")
    carrier: Mapped[str] = mapped_column(String(60), default="")
    tracking_number: Mapped[str] = mapped_column(String(60), default="")
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    estimated_delivery_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    channel: Mapped[str] = mapped_column(String(20), default="web")

    customer: Mapped[Customer] = relationship(back_populates="orders")
    items: Mapped[list["OrderItem"]] = relationship(
        back_populates="order", cascade="all, delete-orphan", lazy="selectin"
    )
    events: Mapped[list["OrderEvent"]] = relationship(
        back_populates="order", cascade="all, delete-orphan", lazy="selectin",
        order_by="OrderEvent.occurred_at",
    )


class OrderItem(Base):
    __tablename__ = "order_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"), index=True)
    variant_id: Mapped[int] = mapped_column(ForeignKey("product_variants.id"), index=True)
    product_name: Mapped[str] = mapped_column(String(200))
    brand: Mapped[str] = mapped_column(String(80))
    size: Mapped[str] = mapped_column(String(20))
    color: Mapped[str] = mapped_column(String(40))
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    unit_price_cents: Mapped[int] = mapped_column(Integer)

    order: Mapped[Order] = relationship(back_populates="items")
    variant: Mapped[ProductVariant] = relationship(lazy="selectin")

    __table_args__ = (CheckConstraint("quantity > 0", name="ck_item_quantity_positive"),)

    @property
    def line_total_cents(self) -> int:
        return self.unit_price_cents * self.quantity


class OrderEvent(Base):
    """Append-only shipment/status timeline. Never updated in place."""

    __tablename__ = "order_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"), index=True)
    status: Mapped[OrderStatus] = mapped_column(Enum(OrderStatus))
    location: Mapped[str] = mapped_column(String(120), default="")
    note: Mapped[str] = mapped_column(String(240), default="")
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    order: Mapped[Order] = relationship(back_populates="events")


# ---------------------------------------------------------------------------
# Cart (session-scoped, pre-checkout)
# ---------------------------------------------------------------------------

class CartItem(Base):
    __tablename__ = "cart_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    variant_id: Mapped[int] = mapped_column(ForeignKey("product_variants.id"), index=True)
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    variant: Mapped[ProductVariant] = relationship(lazy="selectin")

    __table_args__ = (
        UniqueConstraint("session_id", "variant_id", name="uq_cart_session_variant"),
        CheckConstraint("quantity > 0", name="ck_cart_quantity_positive"),
    )


# ---------------------------------------------------------------------------
# Conversation, audit & feedback
# ---------------------------------------------------------------------------

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    turn_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class ToolInvocation(Base):
    """Audit record for every backend call the assistant made.

    This is what makes a transactional answer defensible: the reviewer can see
    the exact arguments, the exact rows returned, and the latency, for every
    claim in the reply.
    """

    __tablename__ = "tool_invocations"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    turn_id: Mapped[str] = mapped_column(String(64), index=True)
    correlation_id: Mapped[str] = mapped_column(String(64), default="")
    tool_name: Mapped[str] = mapped_column(String(64), index=True)
    arguments_json: Mapped[str] = mapped_column(Text, default="{}")
    result_json: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(String(20), default="ok")  # ok | error | denied
    error_message: Mapped[str] = mapped_column(String(400), default="")
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class GuardrailEvent(Base):
    """Every guardrail decision, allowed or blocked, for governance review."""

    __tablename__ = "guardrail_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    turn_id: Mapped[str] = mapped_column(String(64), default="")
    stage: Mapped[str] = mapped_column(String(20))          # input | output | authorization
    rule: Mapped[str] = mapped_column(String(80))
    action: Mapped[str] = mapped_column(String(20))          # allow | block | redact | warn
    score: Mapped[float] = mapped_column(Float, default=0.0)
    detail: Mapped[str] = mapped_column(String(400), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    turn_id: Mapped[str] = mapped_column(String(64), index=True)
    rating: Mapped[str] = mapped_column(String(16))          # helpful | not_helpful
    reason: Mapped[str] = mapped_column(String(60), default="")
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
