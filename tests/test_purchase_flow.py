"""Checkout: the only irreversible action.

Each test here corresponds to a way a purchase could go wrong that would cost a
customer real money.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.db.models import Order, ProductVariant
from app.schemas import CheckoutQuote, ToolError
from app.services import cart as cart_service

SESSION = "test-session"


def _add_one(session, catalogue, quantity=1):
    variant = catalogue["nike_tee"].variants[0]  # Black / M, stock 14
    return cart_service.add_to_cart(session, SESSION, variant_id=variant.id, quantity=quantity)


def test_place_order_requires_a_token(session, catalogue):
    _add_one(session, catalogue)
    result = cart_service.place_order(session, SESSION, catalogue["alice"].id, "")
    assert isinstance(result, ToolError)
    assert result.code == "INVALID_CONFIRMATION"


def test_invented_token_is_rejected(session, catalogue):
    """The central purchase guarantee: no string the model can produce works."""
    _add_one(session, catalogue)
    for invented in ("confirm", "yes", "token123", "a" * 32):
        result = cart_service.place_order(session, SESSION, catalogue["alice"].id, invented)
        assert isinstance(result, ToolError), invented
        assert result.code == "INVALID_CONFIRMATION"


def test_happy_path_places_the_order_and_decrements_stock(session, catalogue):
    variant = catalogue["nike_tee"].variants[0]
    stock_before = variant.stock

    _add_one(session, catalogue, quantity=2)
    quote = cart_service.prepare_checkout(session, SESSION, catalogue["alice"].id)
    assert isinstance(quote, CheckoutQuote)

    result = cart_service.place_order(session, SESSION, catalogue["alice"].id, quote.confirmation_token)
    assert result["order_placed"] is True

    order = session.scalar(select(Order).where(Order.order_number == result["order_number"]))
    assert order.customer_id == catalogue["alice"].id
    assert order.channel == "assistant"
    assert session.get(ProductVariant, variant.id).stock == stock_before - 2
    assert cart_service.view_cart(session, SESSION).item_count == 0


def test_token_is_single_use(session, catalogue):
    _add_one(session, catalogue)
    quote = cart_service.prepare_checkout(session, SESSION, catalogue["alice"].id)
    first = cart_service.place_order(session, SESSION, catalogue["alice"].id, quote.confirmation_token)
    assert first["order_placed"] is True

    _add_one(session, catalogue)
    replay = cart_service.place_order(session, SESSION, catalogue["alice"].id, quote.confirmation_token)
    assert isinstance(replay, ToolError)
    assert replay.code == "INVALID_CONFIRMATION"


def test_token_is_void_if_the_basket_changed(session, catalogue):
    """Quote for one basket must not be redeemable against a different one."""
    _add_one(session, catalogue)
    quote = cart_service.prepare_checkout(session, SESSION, catalogue["alice"].id)

    cart_service.add_to_cart(
        session, SESSION, variant_id=catalogue["adidas_shoe"].variants[0].id, quantity=1
    )
    result = cart_service.place_order(session, SESSION, catalogue["alice"].id, quote.confirmation_token)
    assert isinstance(result, ToolError)
    assert result.code == "BASKET_CHANGED"


def test_token_from_another_session_is_rejected(session, catalogue):
    _add_one(session, catalogue)
    quote = cart_service.prepare_checkout(session, SESSION, catalogue["alice"].id)
    result = cart_service.place_order(session, "someone-elses-session", catalogue["alice"].id,
                                      quote.confirmation_token)
    assert isinstance(result, ToolError)
    assert result.code == "CONFIRMATION_MISMATCH"


def test_expired_token_is_rejected(session, catalogue, monkeypatch):
    _add_one(session, catalogue)
    quote = cart_service.prepare_checkout(session, SESSION, catalogue["alice"].id)

    # Age the stored quote past its TTL rather than sleeping.
    stored = cart_service.quote_store._quotes[quote.confirmation_token]
    cart_service.quote_store._quotes[quote.confirmation_token] = type(stored)(
        session_id=stored.session_id, customer_id=stored.customer_id,
        basket_fingerprint=stored.basket_fingerprint, total_cents=stored.total_cents,
        payment_method=stored.payment_method, shipping_address=stored.shipping_address,
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    result = cart_service.place_order(session, SESSION, catalogue["alice"].id, quote.confirmation_token)
    assert isinstance(result, ToolError)
    assert result.code == "INVALID_CONFIRMATION"  # pruned on read


def test_stock_sold_out_between_quote_and_confirm(session, catalogue):
    """The quote-to-confirm gap is where a concurrent order empties the shelf."""
    variant = catalogue["nike_tee"].variants[0]
    _add_one(session, catalogue, quantity=2)
    quote = cart_service.prepare_checkout(session, SESSION, catalogue["alice"].id)

    variant.stock = 1  # somebody else bought the rest
    session.flush()

    result = cart_service.place_order(session, SESSION, catalogue["alice"].id, quote.confirmation_token)
    assert isinstance(result, ToolError)
    assert result.code == "STOCK_CHANGED"


def test_cannot_check_out_an_empty_cart(session, catalogue):
    result = cart_service.prepare_checkout(session, SESSION, catalogue["alice"].id)
    assert isinstance(result, ToolError)
    assert result.code == "EMPTY_CART"


def test_out_of_stock_variant_cannot_be_added(session, catalogue):
    dead = catalogue["nike_tee"].variants[1]  # Black / L, stock 0
    result = cart_service.add_to_cart(session, SESSION, variant_id=dead.id)
    assert isinstance(result, ToolError)
    assert result.code == "OUT_OF_STOCK"


def test_cannot_add_more_than_available(session, catalogue):
    scarce = catalogue["nike_tee"].variants[2]  # Navy / M, stock 3
    result = cart_service.add_to_cart(session, SESSION, variant_id=scarce.id, quantity=5)
    assert isinstance(result, ToolError)
    assert result.code == "INSUFFICIENT_STOCK"
    assert "3" in result.error


def test_ambiguous_add_asks_rather_than_guessing(session, catalogue):
    """Nike tee has Black and Navy in stock. Adding without a colour must ask."""
    result = cart_service.add_to_cart(session, SESSION, product_id=catalogue["nike_tee"].id, size="M")
    assert isinstance(result, ToolError)
    assert result.code == "NEEDS_COLOR"
    assert "Black" in result.error and "Navy" in result.error


def test_add_to_cart_resolves_product_by_exact_name(session, catalogue):
    """The rule-based planner's add-to-cart intent has no product_id, only the
    name text it parsed - the tool must resolve that itself."""
    from app.agent.tools import ToolContext, execute_tool

    ctx = ToolContext(session=session, session_id=SESSION,
                      customer_id=catalogue["alice"].id, customer_name="Alice Tester")
    result, status = execute_tool(
        "add_to_cart", {"product_name": "Nike Core Unisex T-Shirt", "size": "M", "color": "Black"}, ctx
    )
    assert status == "ok"
    assert result["item_count"] == 1


def test_add_to_cart_by_name_reports_unknown_product(session, catalogue):
    from app.agent.tools import ToolContext, execute_tool

    ctx = ToolContext(session=session, session_id=SESSION,
                      customer_id=catalogue["alice"].id, customer_name="Alice Tester")
    result, status = execute_tool("add_to_cart", {"product_name": "Nonexistent Product Xyz123"}, ctx)
    assert status == "error"
    assert result["code"] == "PRODUCT_NOT_FOUND"


def test_free_shipping_threshold(session, catalogue):
    cheap = _add_one(session, catalogue, quantity=1)
    assert cheap.shipping.amount_cents == cart_service.STANDARD_SHIPPING_CENTS

    cart_service.add_to_cart(session, SESSION, variant_id=catalogue["adidas_shoe"].variants[0].id, quantity=1)
    rich = cart_service.view_cart(session, SESSION)
    assert rich.subtotal.amount_cents >= cart_service.FREE_SHIPPING_THRESHOLD_CENTS
    assert rich.shipping.amount_cents == 0


def test_cod_limit_is_enforced(session, catalogue):
    cart_service.add_to_cart(session, SESSION, variant_id=catalogue["adidas_shoe"].variants[0].id, quantity=4)
    result = cart_service.prepare_checkout(
        session, SESSION, catalogue["alice"].id, payment_method="cash_on_delivery"
    )
    assert isinstance(result, ToolError)
    assert result.code == "COD_LIMIT_EXCEEDED"


# ------------------------------------- the token must never reach the model

def test_confirmation_token_is_withheld_from_the_model(session, catalogue):
    """The purchase guarantee depends on the model never holding the token.

    Tool results are serialised into the model's context, so a token returned
    by prepare_checkout would otherwise be readable and replayable by the model
    on a later call - completing a purchase with no human confirmation.
    """
    from app.agent.tools import TOOLS_BY_NAME, redact_for_model

    _add_one(session, catalogue)
    quote = cart_service.prepare_checkout(session, SESSION, catalogue["alice"].id)
    assert isinstance(quote, CheckoutQuote)

    payload = quote.model_dump(mode="json")
    model_view = redact_for_model(TOOLS_BY_NAME["prepare_checkout"], payload)

    assert quote.confirmation_token not in json.dumps(model_view), (
        "the confirmation token leaked into the model-visible tool result"
    )
    # The rest of the quote must survive, or the model cannot state the total.
    assert model_view["total"] == payload["total"]
    assert model_view["lines"] == payload["lines"]


def test_redaction_does_not_mutate_the_original(session, catalogue):
    """The browser still needs the real token from the same object."""
    from app.agent.tools import TOOLS_BY_NAME, redact_for_model

    _add_one(session, catalogue)
    quote = cart_service.prepare_checkout(session, SESSION, catalogue["alice"].id)
    payload = quote.model_dump(mode="json")
    original = payload["confirmation_token"]

    redact_for_model(TOOLS_BY_NAME["prepare_checkout"], payload)
    assert payload["confirmation_token"] == original

    # And the untouched token still works.
    result = cart_service.place_order(session, SESSION, catalogue["alice"].id, original)
    assert result["order_placed"] is True


def test_no_tool_result_leaks_a_credential_to_the_model(session, catalogue):
    """Sweep every read-only tool's real output for credential-shaped fields.

    A future tool returning a token without declaring it in
    model_redacted_fields would silently reintroduce the bypass this test was
    written for, so the check runs over actual results rather than over the
    declaration.
    """
    from app.agent.tools import TOOLS_BY_NAME, ToolContext, execute_tool, redact_for_model

    context = ToolContext(
        session=session, session_id=SESSION,
        customer_id=catalogue["alice"].id, customer_name="Alice Tester",
    )
    _add_one(session, catalogue)

    credential_keys = {"confirmation_token", "token", "secret", "api_key", "password"}
    calls = [
        ("search_products", {"brand": "Nike"}),
        ("get_order_status", {"order_number": "1001"}),
        ("list_my_orders", {}),
        ("view_cart", {}),
        ("lookup_policy", {"query": "returns"}),
        ("prepare_checkout", {}),
    ]
    for name, arguments in calls:
        result, _status = execute_tool(name, arguments, context)
        model_view = redact_for_model(TOOLS_BY_NAME[name], result)
        if not isinstance(model_view, dict):
            continue
        for key in credential_keys & set(model_view):
            assert str(model_view[key]).startswith("[withheld]"), (
                f"{name} exposes '{key}' to the model without declaring it in "
                f"model_redacted_fields"
            )
