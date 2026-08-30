"""Catalogue service.

The single source of truth for anything the assistant says about a product.
Retrieval proposes a shortlist; this module reads the actual rows and applies
the structured filters. That ordering matters: it means a stale or imperfect
index can only ever affect *which* products are shown, never whether the price
and stock attached to them are correct.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.db.models import Product, ProductVariant
from app.retrieval.index import retrieval_service
from app.schemas import (
    Money, PolicyAnswer, PolicyPassage, ProductDetail, ProductSearchResult,
    ProductSummary, VariantOut,
)

logger = logging.getLogger(__name__)

MAX_RESULTS = 24
#: How many BM25 candidates we let through to the SQL filter stage. Anything
#: beyond this is not "no match", it is "not ranked highly enough to matter",
#: and the distinction is surfaced to the model via `total_matches_capped`.
RETRIEVAL_CANDIDATE_WINDOW = 200
SORT_OPTIONS = {"relevance", "price_low_to_high", "price_high_to_low", "rating", "newest", "discount"}


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _summary(product: Product, relevance: float | None = None) -> ProductSummary:
    in_stock_variants = [v for v in product.variants if v.stock > 0]
    return ProductSummary(
        product_id=product.id,
        sku=product.sku,
        name=product.name,
        brand=product.brand,
        category=product.category,
        subcategory=product.subcategory,
        gender=product.gender,
        price=Money.of(product.price_cents, product.currency),
        list_price=Money.of(product.list_price_cents, product.currency),
        discount_pct=product.discount_pct,
        rating=product.rating,
        review_count=product.review_count,
        in_stock=bool(in_stock_variants),
        total_stock=product.total_stock,
        available_sizes=_ordered_sizes({v.size for v in in_stock_variants}),
        available_colors=sorted({v.color for v in in_stock_variants}),
        relevance=round(relevance, 5) if relevance is not None else None,
    )


def _detail(product: Product) -> ProductDetail:
    base = _summary(product)
    return ProductDetail(
        **base.model_dump(),
        description=product.description,
        material=product.material,
        care=product.care,
        tags=product.tag_list,
        variants=[
            VariantOut(
                variant_id=v.id, sku=v.sku, size=v.size, color=v.color,
                stock=v.stock, in_stock=v.stock > 0,
            )
            for v in sorted(product.variants, key=lambda v: (v.color, _size_rank(v.size)))
        ],
    )


_SIZE_ORDER = ["XS", "S", "M", "L", "XL", "XXL", "ONE SIZE"]


def _size_rank(size: str) -> tuple[int, float]:
    """Sort sizes the way a size selector displays them, not alphabetically.

    Alphabetical ordering puts L before M before S before XL, which reads as a
    bug to any customer who has ever bought clothing.
    """
    if size in _SIZE_ORDER:
        return (0, _SIZE_ORDER.index(size))
    try:
        return (1, float(size))
    except ValueError:
        return (2, 0.0)


def _ordered_sizes(sizes: set[str]) -> list[str]:
    return sorted(sizes, key=_size_rank)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _apply_filters(
    statement: Select,
    *,
    brand: str | None,
    category: str | None,
    subcategory: str | None,
    gender: str | None,
    min_price_cents: int | None,
    max_price_cents: int | None,
    min_rating: float | None,
    on_sale_only: bool,
) -> Select:
    statement = statement.where(Product.is_active.is_(True))
    if brand:
        statement = statement.where(Product.brand == brand)
    if category:
        statement = statement.where(Product.category == category)
    if subcategory:
        statement = statement.where(Product.subcategory == subcategory)
    if gender:
        statement = statement.where(Product.gender.in_([gender, "unisex"]))
    if min_price_cents is not None:
        statement = statement.where(Product.price_cents >= min_price_cents)
    if max_price_cents is not None:
        statement = statement.where(Product.price_cents <= max_price_cents)
    if min_rating is not None:
        statement = statement.where(Product.rating >= min_rating)
    if on_sale_only:
        statement = statement.where(Product.price_cents < Product.list_price_cents)
    return statement


def search_products(
    session: Session,
    *,
    query: str = "",
    brand: str | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    gender: str | None = None,
    size: str | None = None,
    color: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_rating: float | None = None,
    in_stock_only: bool = True,
    on_sale_only: bool = False,
    sort: str = "relevance",
    limit: int = 8,
) -> ProductSearchResult:
    """Search the catalogue.

    Free text goes through BM25; structured attributes go through SQL. Keeping
    them separate is what makes "Nike t-shirts under $40 in medium" answerable:
    the price and size constraints are hard predicates that a similarity score
    must not be allowed to soften.
    """
    limit = max(1, min(limit, MAX_RESULTS))
    sort = sort if sort in SORT_OPTIONS else "relevance"

    # Resolve free-text attribute values against the real catalogue vocabulary
    # so a near-miss ("addidas", "tshirt") becomes a hit rather than zero rows.
    resolved_brand = retrieval_service.resolve_brand(brand) if brand else None
    resolved_category = (
        retrieval_service.resolve_vocabulary(category, retrieval_service.categories) if category else None
    )
    resolved_subcategory = (
        retrieval_service.resolve_vocabulary(subcategory, retrieval_service.subcategories)
        if subcategory else None
    )
    resolved_color = (
        retrieval_service.resolve_vocabulary(color, retrieval_service.colors, cutoff=0.7) if color else None
    )
    resolved_gender = gender.lower().strip() if gender else None
    if resolved_gender not in {None, "men", "women", "unisex"}:
        resolved_gender = None

    normalised_size = size.strip().upper() if size else None
    min_price_cents = int(round(min_price * 100)) if min_price is not None else None
    max_price_cents = int(round(max_price * 100)) if max_price is not None else None

    statement = _apply_filters(
        select(Product),
        brand=resolved_brand, category=resolved_category, subcategory=resolved_subcategory,
        gender=resolved_gender, min_price_cents=min_price_cents, max_price_cents=max_price_cents,
        min_rating=min_rating, on_sale_only=on_sale_only,
    )

    # Variant-level constraints (size / colour / stock) are an EXISTS subquery
    # so that one matching variant qualifies the product without duplicating it
    # across the result set, which a join would do.
    if normalised_size or resolved_color or in_stock_only:
        variant_filter = select(ProductVariant.id).where(ProductVariant.product_id == Product.id)
        if normalised_size:
            variant_filter = variant_filter.where(func.upper(ProductVariant.size) == normalised_size)
        if resolved_color:
            variant_filter = variant_filter.where(ProductVariant.color == resolved_color)
        if in_stock_only:
            variant_filter = variant_filter.where(ProductVariant.stock > 0)
        statement = statement.where(variant_filter.exists())

    # Count the structured-filter set before the keyword restriction narrows it.
    # Holding both numbers means we never have to guess whether the keyword
    # window hid anything: we can compare them directly.
    filter_only_total = session.scalar(
        select(func.count()).select_from(statement.subquery())
    ) or 0

    relevance: dict[int, float] = {}
    if query.strip():
        hits = retrieval_service.search_products(query, limit=RETRIEVAL_CANDIDATE_WINDOW)
        relevance = {hit.product_id: hit.score for hit in hits}
        if relevance:
            statement = statement.where(Product.id.in_(list(relevance)))
        else:
            # No lexical hit at all. Rather than returning nothing, fall back to
            # the structured filters alone and say so, which is far more useful
            # than a dead end for a query like "something warm".
            pass

    total_matches = session.scalar(
        select(func.count()).select_from(statement.subquery())
    ) or 0
    # The keyword count is a floor only when the retrieval window was actually
    # full *and* it excluded rows the structured filters would have admitted.
    # Comparing the two counts makes that exact rather than heuristic.
    total_matches_capped = (
        len(relevance) >= RETRIEVAL_CANDIDATE_WINDOW and total_matches < filter_only_total
    )

    if sort == "price_low_to_high":
        statement = statement.order_by(Product.price_cents.asc())
    elif sort == "price_high_to_low":
        statement = statement.order_by(Product.price_cents.desc())
    elif sort == "rating":
        statement = statement.order_by(Product.rating.desc(), Product.review_count.desc())
    elif sort == "newest":
        statement = statement.order_by(Product.id.desc())
    elif sort == "discount":
        statement = statement.order_by((Product.list_price_cents - Product.price_cents).desc())

    if sort == "relevance" and relevance:
        # Pull a generous candidate window, then order in Python by the fused
        # retrieval score. SQLite has no way to ORDER BY an arbitrary id list.
        candidates = session.scalars(statement.limit(RETRIEVAL_CANDIDATE_WINDOW)).all()
        candidates.sort(key=lambda p: relevance.get(p.id, 0.0), reverse=True)
        rows = candidates[:limit]
    else:
        rows = list(session.scalars(statement.limit(limit)).all())

    note = ""
    if total_matches_capped:
        note = (
            f"{total_matches} products match the keyword ranking, out of "
            f"{filter_only_total} that match the filters overall. Report this as "
            f"'at least {total_matches}', never as an exact catalogue total."
        )
    if query.strip() and not relevance:
        note = "No keyword match was found, so results reflect the structured filters only."
    if brand and not resolved_brand:
        note = (
            f"'{brand}' is not a brand Aurelia carries. "
            f"Stocked brands include: {', '.join(retrieval_service.brands[:8])}."
        ).strip()
        return ProductSearchResult(
            query=query, applied_filters={"brand": brand}, total_matches=0,
            returned=0, products=[], note=note,
        )

    applied = {
        k: v for k, v in {
            "brand": resolved_brand, "category": resolved_category,
            "subcategory": resolved_subcategory, "gender": resolved_gender,
            "size": normalised_size, "color": resolved_color,
            "min_price": min_price, "max_price": max_price, "min_rating": min_rating,
            "in_stock_only": in_stock_only, "on_sale_only": on_sale_only or None,
            "sort": sort,
        }.items() if v is not None
    }

    return ProductSearchResult(
        query=query,
        applied_filters=applied,
        total_matches=total_matches,
        total_matches_capped=total_matches_capped,
        total_matching_filters=filter_only_total,
        returned=len(rows),
        products=[_summary(p, relevance.get(p.id)) for p in rows],
        facets=_facets(session, statement),
        note=note,
    )


def _facets(session: Session, statement: Select) -> dict[str, list[dict[str, Any]]]:
    """Brand and category counts across the whole filtered set.

    Facets let the assistant say "there are also 14 Adidas options" instead of
    silently truncating at the page size, which is the difference between a
    helpful narrowing question and an answer that looks like the whole story.
    """
    subquery = statement.subquery()
    product = Product.__table__

    def top(column_name: str, cap: int = 6) -> list[dict[str, Any]]:
        column = subquery.c[column_name]
        rows = session.execute(
            select(column, func.count()).select_from(subquery)
            .group_by(column).order_by(func.count().desc()).limit(cap)
        ).all()
        return [{"value": value, "count": count} for value, count in rows if value]

    del product  # documentation of intent; the subquery carries the columns
    return {"brand": top("brand"), "subcategory": top("subcategory"), "category": top("category")}


def get_product(session: Session, product_id: int) -> ProductDetail | None:
    product = session.get(Product, product_id)
    if product is None or not product.is_active:
        return None
    return _detail(product)


def find_product_id_by_name(session: Session, name: str) -> int | None:
    """Resolve a product by its exact display name.

    Product names are guaranteed unique by the seed generator (see
    `app/db/seed.py::_unique_name`), so an exact case-insensitive match is
    unambiguous when it succeeds. Falls back to the top BM25 hit so a close
    paraphrase still resolves rather than failing outright.

    This exists for callers that only have a name in hand, not an id - the
    rule-based fallback planner in particular, which parses free text and has
    no reason to have already looked a product up through search_products.
    """
    cleaned = name.strip().strip('"').strip()
    if not cleaned:
        return None

    exact = session.scalar(
        select(Product.id).where(
            Product.is_active.is_(True), func.lower(Product.name) == cleaned.lower()
        )
    )
    if exact is not None:
        return exact

    hits = retrieval_service.search_products(cleaned, limit=1)
    return hits[0].product_id if hits else None


def check_availability(
    session: Session, product_id: int, size: str | None = None, color: str | None = None
) -> dict[str, Any]:
    """Answer 'can I actually buy this, in this size, right now'.

    Returned as an explicit structure rather than prose so the model cannot
    round "3 left" up to "in stock" or down to "unavailable".
    """
    product = session.get(Product, product_id)
    if product is None or not product.is_active:
        return {"found": False, "product_id": product_id}

    variants = product.variants
    if size:
        variants = [v for v in variants if v.size.upper() == size.strip().upper()]
    if color:
        resolved = retrieval_service.resolve_vocabulary(color, retrieval_service.colors, cutoff=0.7) or color
        variants = [v for v in variants if v.color.lower() == resolved.lower()]

    available = [v for v in variants if v.stock > 0]
    return {
        "found": True,
        "product_id": product.id,
        "product_name": product.name,
        "brand": product.brand,
        "price": Money.of(product.price_cents, product.currency).model_dump(),
        "requested_size": size,
        "requested_color": color,
        "matching_variants": [
            {
                "variant_id": v.id, "size": v.size, "color": v.color,
                "stock": v.stock, "in_stock": v.stock > 0,
            }
            for v in sorted(variants, key=lambda v: (v.color, _size_rank(v.size)))
        ],
        "any_available": bool(available),
        "all_sizes_in_stock": _ordered_sizes({v.size for v in product.variants if v.stock > 0}),
        "all_colors_in_stock": sorted({v.color for v in product.variants if v.stock > 0}),
    }


def list_brands(session: Session) -> list[dict[str, Any]]:
    rows = session.execute(
        select(Product.brand, func.count(Product.id))
        .where(Product.is_active.is_(True))
        .group_by(Product.brand).order_by(Product.brand)
    ).all()
    return [{"brand": brand, "product_count": count} for brand, count in rows]


def list_categories(session: Session) -> list[dict[str, Any]]:
    rows = session.execute(
        select(Product.category, Product.subcategory, func.count(Product.id))
        .where(Product.is_active.is_(True))
        .group_by(Product.category, Product.subcategory)
        .order_by(Product.category, Product.subcategory)
    ).all()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for category, subcategory, count in rows:
        grouped.setdefault(category, []).append({"subcategory": subcategory, "product_count": count})
    return [{"category": k, "subcategories": v} for k, v in grouped.items()]


def lookup_policy(query: str, topic: str | None = None, limit: int = 3) -> PolicyAnswer:
    """RAG over the policy corpus. Returns passages with citations, not prose."""
    hits = retrieval_service.search_policies(query, limit=limit, topic=topic)
    if not hits and topic:
        hits = retrieval_service.search_policies(query, limit=limit)
    passages = [
        PolicyPassage(
            document=chunk.document, heading=chunk.heading, topic=chunk.topic,
            text=chunk.text, citation=chunk.citation(),
        )
        for chunk, _score in hits
    ]
    note = (
        "" if passages else
        "No policy passage matched. Available topics: "
        + ", ".join(retrieval_service.policy_topics())
    )
    return PolicyAnswer(query=query, passages=passages, note=note)
