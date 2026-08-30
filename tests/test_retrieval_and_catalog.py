"""Retrieval quality and catalogue correctness."""

from __future__ import annotations

import pytest

from app.retrieval.bm25 import BM25Index, normalise, reciprocal_rank_fusion, tokenize
from app.retrieval.index import chunk_policy_document, retrieval_service
from app.services import catalog as catalog_service
from app.config import POLICY_DIR


# ------------------------------------------------------------------- BM25

def test_tokenizer_drops_stopwords_and_normalises():
    tokens = tokenize("What are the available Nike shoes for men?")
    assert "the" not in tokens and "are" not in tokens
    assert "nike" in tokens
    assert "shoe" in tokens        # 'shoes' normalised


def test_synonym_expansion_is_query_only():
    """Expanding documents too would distort term frequencies and IDF."""
    query = tokenize("sneakers", expand=True)
    document = tokenize("sneakers", expand=False)
    assert "footwear" in query
    assert "footwear" not in document


def test_brand_names_are_never_stemmed():
    assert normalise("adidas") == "adidas"
    assert normalise("levis") == "levis"


def test_bm25_ranks_the_relevant_document_first():
    index = BM25Index()
    index.add("tee", "Nike Core Unisex T-Shirt topwear cotton")
    index.add("shoe", "Adidas Pro Running Shoes footwear mesh")
    index.finalise()
    top = index.search("nike t-shirt")[0]
    assert top[0] == "tee"


def test_idf_never_goes_negative():
    """A term in most documents must not subtract from a score."""
    index = BM25Index()
    for i in range(10):
        index.add(str(i), "common term here")
    index.finalise()
    assert index._idf("common") > 0


def test_rrf_merges_rankings_by_position():
    a = [("x", 100.0), ("y", 90.0)]
    b = [("y", 0.5), ("x", 0.4)]
    fused = dict(reciprocal_rank_fusion([a, b]))
    # y is 2nd then 1st; x is 1st then 2nd. Equal positions, equal fused score.
    assert fused["x"] == pytest.approx(fused["y"])


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[("a", 1.0)]], weights=[1.0, 2.0])


# ------------------------------------------------------------- policy RAG

def test_policy_chunking_follows_headings():
    chunks = chunk_policy_document(POLICY_DIR / "returns-and-refunds.md")
    headings = {c.heading for c in chunks}
    assert "Return window" in headings
    assert "Non-returnable items" in headings
    assert all(c.document == "Returns, Exchanges and Refunds" for c in chunks)
    assert all(len(c.text) >= 40 for c in chunks)


@pytest.mark.parametrize("query,expected_document", [
    ("how long do I have to return something", "Returns, Exchanges and Refunds"),
    ("can I cancel after it has shipped", "Order Changes and Cancellation"),
    ("what size should I buy", "Sizing and Fit"),
    ("do you ship internationally", "Shipping and Delivery"),
    ("when is my card charged", "Payments, Pricing and Promotions"),
    ("is my watch under warranty", "Warranty and Product Care"),
])
def test_policy_retrieval_finds_the_right_document(query, expected_document, session, catalogue):
    answer = catalog_service.lookup_policy(query, limit=3)
    assert answer.passages, query
    assert expected_document in {p.document for p in answer.passages}, query


def test_policy_answer_carries_citations(session, catalogue):
    answer = catalog_service.lookup_policy("refund timing", limit=2)
    assert all(">" in p.citation for p in answer.passages)


# --------------------------------------------------------------- catalogue

def test_search_finds_brand_and_type(session, catalogue):
    result = catalog_service.search_products(session, query="nike t-shirt", limit=5)
    assert result.products
    assert result.products[0].brand == "Nike"


def test_misspelled_brand_is_resolved(session, catalogue):
    result = catalog_service.search_products(session, brand="addidas", limit=5)
    assert result.applied_filters["brand"] == "Adidas"
    assert result.products


def test_unstocked_brand_is_reported_not_faked(session, catalogue):
    result = catalog_service.search_products(session, brand="Gucci")
    assert result.products == []
    assert "not a brand Aurelia carries" in result.note
    assert "Nike" in result.note  # tells the model what is stocked


def test_price_filter_is_a_hard_constraint(session, catalogue):
    """Relevance must never soften a price ceiling."""
    result = catalog_service.search_products(session, query="shoes", max_price=50, limit=10)
    assert all(p.price.amount_cents <= 5000 for p in result.products)


def test_size_filter_uses_variant_stock(session, catalogue):
    """Nike tee has L only in Black with stock 0, so an in-stock L search
    must not return it."""
    result = catalog_service.search_products(session, brand="Nike", size="L", in_stock_only=True)
    assert result.products == []
    result_any = catalog_service.search_products(session, brand="Nike", size="M", in_stock_only=True)
    assert result_any.products


def test_counts_are_honest_about_the_retrieval_window(session, catalogue):
    result = catalog_service.search_products(session, query="nike", limit=2)
    assert result.total_matching_filters >= result.total_matches
    assert isinstance(result.total_matches_capped, bool)


def test_sizes_sort_in_wearing_order_not_alphabetically(session, catalogue):
    detail = catalog_service.get_product(session, catalogue["nike_tee"].id)
    black = [v.size for v in detail.variants if v.color == "Black"]
    assert black == ["M", "L"]   # not ["L", "M"]


def test_availability_reports_per_variant_stock(session, catalogue):
    result = catalog_service.check_availability(session, catalogue["nike_tee"].id, size="L")
    assert result["found"]
    assert result["any_available"] is False       # L is Black-only with 0 stock
    assert "M" in result["all_sizes_in_stock"]


def test_money_is_formatted_once_in_python(session, catalogue):
    """The model repeats `display`; it never divides cents by 100 itself."""
    result = catalog_service.search_products(session, brand="Nike", limit=1)
    price = result.products[0].price
    assert price.display == "$26.99"
    assert price.amount_cents == 2699


# ------------------------------------------------- retrieval quality baseline

#: Gold set for policy retrieval, used as a regression guard rather than as an
#: accuracy claim. The measured baseline is 5/10 at rank one and 9/10 within the
#: top three. Top-three is the metric that matters, because the assistant is
#: given three passages and synthesises across them; rank one only matters for
#: the rule-based fallback, which is why that renderer shows all three.
#:
#: Recorded here so a retrieval change has to justify itself against a number.
#: A hand-tuned duration-synonym expansion was tried and reverted: it moved
#: errors between queries without improving either metric.
POLICY_GOLD: list[tuple[str, str]] = [
    ("how long do I have to return something", "Return window"),
    ("how long do refunds take", "Refunds"),
    ("what is the return window", "Return window"),
    ("how long does delivery take", "Delivery speeds and cost"),
    ("can I cancel after it ships", "When an order can be cancelled"),
    ("do you ship internationally", "International shipping"),
    ("what size should I buy in Zara", "Brand variation"),
    ("is my watch under warranty", "Warranty periods"),
    ("when am I charged", "When you are charged"),
    ("what happens if delivery fails", "Failed delivery"),
]


def test_policy_retrieval_top3_recall_does_not_regress(session, catalogue):
    """The passages handed to the model must contain the answer."""
    hits = 0
    misses = []
    for query, expected_heading in POLICY_GOLD:
        headings = [c.heading for c, _score in retrieval_service.search_policies(query, limit=3)]
        if expected_heading in headings:
            hits += 1
        else:
            misses.append(f"{query!r} -> {headings}")
    assert hits >= 9, f"top-3 recall fell to {hits}/10. Misses: {misses}"


def test_policy_retrieval_rank1_does_not_regress(session, catalogue):
    """Weaker guarantee, tracked because the fallback planner depends on it."""
    hits = sum(
        1
        for query, expected in POLICY_GOLD
        if (found := retrieval_service.search_policies(query, limit=1))
        and found[0][0].heading == expected
    )
    assert hits >= 5, f"rank-1 accuracy fell to {hits}/10"


# --------------------------------------------- slim_for_model (token budget)

def test_slim_for_model_shrinks_search_results_substantially(session, catalogue):
    """Regression guard on the actual saving, not just presence/absence of
    fields: this is a real token-budget lever and a change that quietly
    erodes it back toward the original size should fail loudly."""
    import json

    from app.agent.tools import ToolContext, execute_tool, slim_for_model

    ctx = ToolContext(session=session, session_id="s", customer_id=catalogue["alice"].id,
                      customer_name="Alice Tester")
    result, _status = execute_tool("search_products", {"query": "shirt"}, ctx)
    full = len(json.dumps(result, default=str))
    slim = len(json.dumps(slim_for_model("search_products", result), default=str))
    assert slim < full * 0.7, f"expected at least 30% smaller, got {full} -> {slim}"


def test_slim_for_model_keeps_everything_the_model_is_told_to_use(session, catalogue):
    """Every field the system prompt or a tool description instructs the model
    to read and repeat must survive slimming - only fields nothing tells the
    model to use are fair game to drop."""
    from app.agent.tools import ToolContext, execute_tool, slim_for_model

    ctx = ToolContext(session=session, session_id="s", customer_id=catalogue["alice"].id,
                      customer_name="Alice Tester")
    result, _status = execute_tool("search_products", {"brand": "Nike"}, ctx)
    slim = slim_for_model("search_products", result)
    product = slim["products"][0]

    for field in ("product_id", "name", "brand", "subcategory", "price", "list_price",
                  "discount_pct", "rating", "review_count", "in_stock",
                  "available_sizes", "available_colors"):
        assert field in product, f"model-relevant field '{field}' was dropped"

    # Money fields flatten to the display string the model is told to quote
    # verbatim - never the cents/currency it's told never to compute with.
    assert product["price"] == "$26.99"
    assert isinstance(product["price"], str)


def test_slim_for_model_drops_fields_nothing_tells_the_model_to_use(session, catalogue):
    from app.agent.tools import ToolContext, execute_tool, slim_for_model

    ctx = ToolContext(session=session, session_id="s", customer_id=catalogue["alice"].id,
                      customer_name="Alice Tester")
    result, _status = execute_tool("search_products", {"brand": "Nike"}, ctx)
    slim = slim_for_model("search_products", result)

    assert "facets" not in slim
    product = slim["products"][0]
    for field in ("sku", "category", "gender", "total_stock", "relevance"):
        assert field not in product


def test_slim_for_model_does_not_touch_the_original_result(session, catalogue):
    """The artifact channel (product cards in the UI) reads the untouched
    result - slimming must return a copy, never mutate in place."""
    from app.agent.tools import ToolContext, execute_tool, slim_for_model

    ctx = ToolContext(session=session, session_id="s", customer_id=catalogue["alice"].id,
                      customer_name="Alice Tester")
    result, _status = execute_tool("search_products", {"brand": "Nike"}, ctx)
    slim_for_model("search_products", result)
    assert "facets" in result
    assert "sku" in result["products"][0]
    assert isinstance(result["products"][0]["price"], dict)


def test_slim_for_model_leaves_unrelated_tools_untouched(session, catalogue):
    from app.agent.tools import ToolContext, execute_tool, slim_for_model

    ctx = ToolContext(session=session, session_id="s", customer_id=catalogue["alice"].id,
                      customer_name="Alice Tester")
    result, _status = execute_tool("list_brands", {}, ctx)
    assert slim_for_model("list_brands", result) == result


def test_slim_for_model_passes_through_non_dict_and_error_results():
    from app.agent.tools import slim_for_model

    assert slim_for_model("search_products", "not a dict") == "not a dict"
    error = {"code": "PRODUCT_NOT_FOUND", "error": "no such product"}
    assert slim_for_model("search_products", error) == error
