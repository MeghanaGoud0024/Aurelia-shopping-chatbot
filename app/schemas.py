"""Response contracts.

Every object crossing the boundary out of the service layer is defined here.
This is a deliberate guardrail, not ceremony: the tool results are serialised
into the LLM prompt and echoed to the browser, so an accidental
`return product.__dict__` would leak internal columns into both. Defining the
shape explicitly means a new database column cannot become a new field in the
model's context without someone editing this file.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Money(BaseModel):
    """Currency amounts are transported as both machine and display values.

    The model is not asked to divide cents by 100 and format a currency symbol.
    Arithmetic and formatting happen in Python where they are testable; the
    model only ever repeats `display`.
    """

    amount_cents: int
    currency: str = "USD"
    display: str = ""

    @classmethod
    def of(cls, cents: int, currency: str = "USD") -> "Money":
        symbol = {"USD": "$", "AUD": "A$", "EUR": "€", "GBP": "£"}.get(currency, "")
        return cls(amount_cents=cents, currency=currency, display=f"{symbol}{cents / 100:,.2f}")


class VariantOut(BaseModel):
    variant_id: int
    sku: str
    size: str
    color: str
    stock: int
    in_stock: bool


class ProductSummary(BaseModel):
    product_id: int
    sku: str
    name: str
    brand: str
    category: str
    subcategory: str
    gender: str
    price: Money
    list_price: Money
    discount_pct: int
    rating: float
    review_count: int
    in_stock: bool
    total_stock: int
    available_sizes: list[str]
    available_colors: list[str]
    relevance: float | None = None


class ProductDetail(ProductSummary):
    description: str
    material: str
    care: str
    tags: list[str]
    variants: list[VariantOut]


class ProductSearchResult(BaseModel):
    query: str
    applied_filters: dict[str, Any]
    total_matches: int
    #: True when `total_matches` is bounded by the retrieval candidate window
    #: rather than being the true catalogue count. The model is instructed to
    #: say "at least N" when this is set.
    total_matches_capped: bool = False
    #: Count of products matching the structured filters, ignoring keyword
    #: ranking entirely. Always exact.
    total_matching_filters: int = 0
    returned: int
    products: list[ProductSummary]
    facets: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    note: str = ""


class OrderItemOut(BaseModel):
    product_name: str
    brand: str
    size: str
    color: str
    quantity: int
    unit_price: Money
    line_total: Money


class OrderEventOut(BaseModel):
    status: str
    label: str
    location: str
    note: str
    occurred_at: datetime


class OrderSummary(BaseModel):
    order_number: str
    status: str
    status_label: str
    placed_at: datetime
    total: Money
    item_count: int
    estimated_delivery_at: datetime | None = None
    delivered_at: datetime | None = None


class OrderDetail(OrderSummary):
    payment_method: str
    subtotal: Money
    shipping: Money
    tax: Money
    shipping_address: str
    carrier: str
    tracking_number: str
    items: list[OrderItemOut]
    timeline: list[OrderEventOut]
    is_cancellable: bool
    is_returnable: bool
    delivery_message: str


class CartLine(BaseModel):
    variant_id: int
    product_id: int
    product_name: str
    brand: str
    size: str
    color: str
    quantity: int
    unit_price: Money
    line_total: Money
    stock_available: int


class CartView(BaseModel):
    lines: list[CartLine]
    item_count: int
    subtotal: Money
    shipping: Money
    tax: Money
    total: Money
    currency: str = "USD"
    free_shipping_threshold: Money | None = None
    note: str = ""


class CheckoutQuote(BaseModel):
    """A priced, stock-checked basket plus a single-use confirmation token."""

    confirmation_token: str
    expires_at: datetime
    lines: list[CartLine]
    subtotal: Money
    shipping: Money
    tax: Money
    total: Money
    payment_method: str
    shipping_address: str
    requires_user_confirmation: Literal[True] = True
    warnings: list[str] = Field(default_factory=list)


class PolicyPassage(BaseModel):
    document: str
    heading: str
    topic: str
    text: str
    citation: str


class PolicyAnswer(BaseModel):
    query: str
    passages: list[PolicyPassage]
    note: str = ""


class ToolError(BaseModel):
    """Structured failure returned to the model instead of an exception.

    The model must be able to recover: a tool that raises gives it nothing to
    work with, whereas a typed error with a `recovery_hint` lets it retry with
    corrected arguments or explain the limitation to the customer in plain
    language.
    """

    model_config = ConfigDict(populate_by_name=True)

    error: str
    code: str
    recovery_hint: str = ""
    retryable: bool = False


class TraceStep(BaseModel):
    """One entry in the user-visible explainability trace."""

    step: int
    kind: Literal["guardrail", "reasoning", "tool_call", "tool_result", "answer"]
    label: str
    detail: str = ""
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    result_summary: str | None = None
    status: str = "ok"
    latency_ms: int = 0


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    turn_id: str
    reply: str
    trace: list[TraceStep]
    products: list[ProductSummary] = Field(default_factory=list)
    orders: list[OrderDetail] = Field(default_factory=list)
    cart: CartView | None = None
    checkout_quote: CheckoutQuote | None = None
    citations: list[str] = Field(default_factory=list)
    blocked: bool = False
    grounded: bool = True
    latency_ms: int = 0
    model: str = ""


class FeedbackRequest(BaseModel):
    session_id: str
    turn_id: str
    rating: Literal["helpful", "not_helpful"]
    reason: str = ""
    comment: str = Field(default="", max_length=1000)
