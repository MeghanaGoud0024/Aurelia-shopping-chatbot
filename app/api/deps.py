"""Request-scoped dependencies: identity, session binding and correlation ids.

Identity model
--------------
This is a POC without a real authentication provider, and pretending otherwise
would be dishonest. What it does instead is model the *shape* of the real thing:
a browser session cookie is bound server-side to exactly one customer id, and
that id is the only identity the tool layer ever sees.

The important property is that the binding is not something the conversation can
influence. Swapping the identity provider for OIDC or a session service means
changing `resolve_identity` and nothing else - the tool layer, the service layer
and the authorisation predicates are all already written against a
`customer_id` that arrives from outside the model's reach.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from fastapi import Cookie, Depends, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Customer
from app.db.session import get_session

logger = logging.getLogger(__name__)

SESSION_COOKIE = "aurelia_session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 7


@dataclass(slots=True, frozen=True)
class Identity:
    session_id: str
    customer_id: int
    customer_name: str
    customer_public_id: str
    loyalty_tier: str
    email: str


def _demo_customer(session: Session) -> Customer:
    """Pick the customer this demo session acts as.

    Deterministic on purpose: the reviewer following the README must land on the
    account that owns the worked examples, order 1234 included.
    """
    from app.db.models import Order

    owner = session.scalar(
        select(Customer)
        .join(Order, Order.customer_id == Customer.id)
        .where(Order.order_number == "1234")
        .limit(1)
    )
    return owner or session.scalar(select(Customer).order_by(Customer.id).limit(1))


def resolve_identity(
    response: Response,
    session: Session = Depends(get_session),
    aurelia_session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
) -> Identity:
    """Resolve the signed-in customer for this request.

    In production this validates a session token against an identity provider.
    Here it issues a stable browser session and binds it to the demo account.
    """
    session_id = aurelia_session or f"sess_{uuid.uuid4().hex[:20]}"
    if aurelia_session != session_id:
        response.set_cookie(
            SESSION_COOKIE, session_id,
            max_age=COOKIE_MAX_AGE, httponly=True, samesite="lax",
        )

    customer = _demo_customer(session)
    if customer is None:
        raise RuntimeError(
            "No customers exist. Run `python scripts/seed_db.py` before starting the app."
        )

    return Identity(
        session_id=session_id,
        customer_id=customer.id,
        customer_name=customer.full_name,
        customer_public_id=customer.public_id,
        loyalty_tier=customer.loyalty_tier,
        email=customer.email,
    )
