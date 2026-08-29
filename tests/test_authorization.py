"""Authorisation: the property everything else depends on.

If these fail, nothing else in the system matters. The claim being tested is
that customer scoping is enforced in the data access layer, so it holds no
matter what the model is persuaded to attempt.
"""

from __future__ import annotations

import pytest

from app.schemas import ToolError
from app.services import orders as order_service


def test_customer_reads_own_order(session, catalogue):
    result = order_service.get_order_status(session, "1001", catalogue["alice"].id)
    assert not isinstance(result, ToolError)
    assert result.order_number == "1001"
    assert result.status == "shipped"


def test_customer_cannot_read_another_customers_order(session, catalogue):
    """Order 2001 belongs to Mallory. Alice must not be able to read it."""
    result = order_service.get_order_status(session, "2001", catalogue["alice"].id)
    assert isinstance(result, ToolError)
    assert result.code == "ORDER_NOT_FOUND"


def test_forbidden_and_missing_are_indistinguishable(session, catalogue):
    """Enumeration defence: a real-but-forbidden order and a nonexistent one
    must produce byte-identical responses, or the assistant becomes an oracle
    for discovering which order numbers exist."""
    forbidden = order_service.get_order_status(session, "2001", catalogue["alice"].id)
    missing = order_service.get_order_status(session, "9999", catalogue["alice"].id)
    assert forbidden.code == missing.code
    assert forbidden.recovery_hint == missing.recovery_hint
    # Only the echoed order number differs.
    assert forbidden.error.replace("2001", "N") == missing.error.replace("9999", "N")


@pytest.mark.parametrize(
    "operation",
    [order_service.track_shipment, order_service.cancel_order, order_service.request_return],
)
def test_every_order_operation_is_scoped(session, catalogue, operation):
    """Scoping is not just on the read path. Mutations are scoped too."""
    result = operation(session, "2001", catalogue["alice"].id)
    assert isinstance(result, ToolError)
    assert result.code == "ORDER_NOT_FOUND"


def test_list_orders_returns_only_own_orders(session, catalogue):
    result = order_service.list_my_orders(session, catalogue["alice"].id, limit=20)
    numbers = {o["order_number"] for o in result["orders"]}
    assert "2001" not in numbers
    assert numbers == {"1001", "1002", "1003", "1004"}


def test_order_number_is_normalised(session, catalogue):
    """Customers type '#1001' and ' 1001 '. Both must resolve."""
    for spelling in ("#1001", " 1001 ", "1001"):
        result = order_service.get_order_status(session, spelling, catalogue["alice"].id)
        assert not isinstance(result, ToolError), spelling


def test_tool_context_does_not_accept_customer_from_arguments(session, catalogue):
    """The model can only pass tool arguments. If customer_id were among them,
    persuading the model would be a complete authorisation bypass. This asserts
    that the tool schema has no such parameter."""
    from app.agent.tools import TOOLS

    for tool in TOOLS:
        parameters = set((tool.parameters.get("properties") or {}).keys())
        leaky = parameters & {"customer_id", "customer", "user_id", "session_id", "account_id"}
        assert not leaky, f"{tool.name} exposes identity parameters: {leaky}"
