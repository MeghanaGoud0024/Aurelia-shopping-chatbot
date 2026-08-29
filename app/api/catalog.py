"""Catalogue and policy endpoints.

These exist so the browser can render products without going through the model
- pagination, a direct product link, the opening product grid. Nothing here is
privileged: the catalogue is public data.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.session import get_session
from app.schemas import PolicyAnswer, ProductDetail, ProductSearchResult
from app.services import catalog as catalog_service

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("/products", response_model=ProductSearchResult)
def search(
    q: str = Query("", description="Free-text query"),
    brand: str | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    gender: str | None = None,
    size: str | None = None,
    color: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    in_stock_only: bool = True,
    on_sale_only: bool = False,
    sort: str = "relevance",
    limit: int = Query(12, ge=1, le=24),
    session: Session = Depends(get_session),
) -> ProductSearchResult:
    return catalog_service.search_products(
        session, query=q, brand=brand, category=category, subcategory=subcategory,
        gender=gender, size=size, color=color, min_price=min_price, max_price=max_price,
        in_stock_only=in_stock_only, on_sale_only=on_sale_only, sort=sort, limit=limit,
    )


@router.get("/products/{product_id}", response_model=ProductDetail)
def product_detail(product_id: int, session: Session = Depends(get_session)) -> ProductDetail:
    detail = catalog_service.get_product(session, product_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return detail


@router.get("/brands")
def brands(session: Session = Depends(get_session)) -> dict:
    return {"brands": catalog_service.list_brands(session)}


@router.get("/categories")
def categories(session: Session = Depends(get_session)) -> dict:
    return {"categories": catalog_service.list_categories(session)}


@router.get("/policy", response_model=PolicyAnswer)
def policy(
    q: str = Query(..., min_length=2),
    topic: str | None = None,
    limit: int = Query(3, ge=1, le=5),
) -> PolicyAnswer:
    return catalog_service.lookup_policy(q, topic=topic, limit=limit)
