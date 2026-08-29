"""Order and cart endpoints.

Every route here resolves identity through `resolve_identity` and passes the
resulting `customer_id` into the service layer, which filters on it in SQL. The
route never accepts a customer id from the client.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import Identity, resolve_identity
from app.db.session import get_session
from app.schemas import CartView, CheckoutQuote, OrderDetail, ToolError
from app.services import cart as cart_service
from app.services import orders as order_service

router = APIRouter(tags=["orders"])


def _unwrap(result):
    """Convert a service-layer ToolError into an HTTP error for REST callers."""
    if isinstance(result, ToolError):
        status = 404 if result.code.endswith("NOT_FOUND") else 409
        raise HTTPException(status_code=status, detail=result.model_dump())
    return result


@router.get("/orders")
def my_orders(
    limit: int = Query(10, ge=1, le=20),
    status: str | None = None,
    identity: Identity = Depends(resolve_identity),
    session: Session = Depends(get_session),
) -> dict:
    return order_service.list_my_orders(session, identity.customer_id, limit=limit, status=status)


@router.get("/orders/{order_number}", response_model=OrderDetail)
def order_detail(
    order_number: str,
    identity: Identity = Depends(resolve_identity),
    session: Session = Depends(get_session),
) -> OrderDetail:
    return _unwrap(order_service.get_order_status(session, order_number, identity.customer_id))


@router.get("/cart", response_model=CartView)
def get_cart(
    identity: Identity = Depends(resolve_identity),
    session: Session = Depends(get_session),
) -> CartView:
    return cart_service.view_cart(session, identity.session_id)


@router.delete("/cart/{variant_id}", response_model=CartView)
def delete_cart_line(
    variant_id: int,
    identity: Identity = Depends(resolve_identity),
    session: Session = Depends(get_session),
) -> CartView:
    return cart_service.remove_from_cart(session, identity.session_id, variant_id)


@router.post("/checkout/confirm")
def confirm_checkout(
    payload: dict,
    identity: Identity = Depends(resolve_identity),
    session: Session = Depends(get_session),
) -> dict:
    """Redeem a checkout quote.

    This is the human confirmation step, and it deliberately lives on its own
    HTTP route rather than inside the chat turn. The token travels
    server -> browser -> server without the model ever being asked to hold or
    reproduce it, which means the purchase cannot be triggered by anything the
    model generates.
    """
    token = str(payload.get("confirmation_token") or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="confirmation_token is required")
    return _unwrap(
        cart_service.place_order(session, identity.session_id, identity.customer_id, token)
    )
