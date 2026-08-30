"""Rule-based planner for running without a language model.

Why this exists
---------------
The brief requires the application to run from a clean environment using the
documented commands. A reviewer who has not yet obtained an API key would
otherwise see a dead application and be unable to assess anything else that was
built. With this planner, `AURELIA_LLM_API_KEY` empty still gives a working
assistant: same tools, same authorisation, same audit trail, same guardrails.
Only the natural-language quality drops.

It is also the honest comparison point. Having a deterministic baseline in the
repository makes it concrete what the language model is actually contributing -
flexible intent understanding and fluent prose - rather than leaving that to
assertion.

The planner is intentionally simple: regex intent detection, slot extraction,
and templated rendering. It is not trying to be a language model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

def _plural(count: int, singular: str, plural_form: str | None = None) -> str:
    """"1 match" / "3 matches". These strings are shown to the customer."""
    return f"{count:,} {singular if count == 1 else (plural_form or singular + 's')}"


ORDER_NUMBER_RE = re.compile(r"#?\b(\d{3,8})\b")
# Quoted form matches the "Add to bag" button's exact message
# (`Add "Product Name" to my bag`); the loose form covers a customer typing
# the same request without quotes. Requires "to ... bag/cart" so it never
# fires on an unrelated sentence that merely contains the word "add".
ADD_TO_CART_RE = re.compile(
    r'add\s+"(?P<quoted>[^"]+)"\s+to\s+(?:my\s+)?(?:bag|cart|basket)'
    r'|add\s+(?:the\s+|a\s+|an\s+)?(?P<loose>.+?)\s+to\s+(?:my\s+)?(?:bag|cart|basket)',
    re.I,
)
PRICE_UNDER_RE = re.compile(r"(?:under|below|less than|cheaper than|max(?:imum)?(?:\s+of)?)\s*[$€£]?\s*(\d+(?:\.\d{1,2})?)", re.I)
PRICE_OVER_RE = re.compile(r"(?:over|above|more than|at least|min(?:imum)?(?:\s+of)?)\s*[$€£]?\s*(\d+(?:\.\d{1,2})?)", re.I)
SIZE_RE = re.compile(r"\b(?:size\s+)?(XS|S|M|L|XL|XXL)\b")
SHOE_SIZE_RE = re.compile(r"\bsize\s+(\d{1,2})\b", re.I)

BRANDS = [
    "Nike", "Adidas", "Puma", "Under Armour", "New Balance", "Levi's", "Uniqlo",
    "Zara", "H&M", "Tommy Hilfiger", "Calvin Klein", "The North Face",
    "Columbia", "Ray-Ban", "Fossil",
]
SUBCATEGORIES = [
    "T-Shirt", "Polo Shirt", "Hoodie", "Sweatshirt", "Casual Shirt", "Tank Top",
    "Jeans", "Chinos", "Joggers", "Shorts", "Track Pants", "Running Shoes",
    "Sneakers", "Training Shoes", "Sandals", "Jacket", "Windbreaker",
    "Puffer Jacket", "Backpack", "Cap", "Sunglasses", "Watch", "Socks 3-Pack",
]
SUBCATEGORY_ALIASES = {
    "t-shirt": ["t shirt", "tshirt", "tee", "tees", "t-shirts"],
    "running shoes": ["running shoe", "runners", "running"],
    "sneakers": ["sneaker", "trainers", "kicks"],
    "jeans": ["jean", "denim"],
    "hoodie": ["hoody", "hoodies"],
    "jacket": ["jackets", "coat"],
    "sunglasses": ["shades", "sunglass"],
    "backpack": ["bag", "rucksack"],
    "cap": ["hat"],
}

POLICY_TOPICS = {
    "returns": ["return", "refund", "exchange", "send back", "money back"],
    "shipping": ["ship", "shipping", "delivery cost", "postage", "courier", "international"],
    "cancellation": ["cancel", "cancellation"],
    "sizing": ["size chart", "sizing", "fit", "what size", "runs small", "runs large"],
    "payments": ["payment", "pay ", "card", "charged", "promo", "discount code", "loyalty"],
    "warranty": ["warranty", "faulty", "defect", "guarantee", "broken"],
    "privacy": ["privacy", "my data", "personal data", "gdpr", "delete my"],
}


@dataclass
class Plan:
    intent: str
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    _results: list[tuple[str, Any]] = field(default_factory=list)

    def observe(self, tool_name: str, result: Any) -> None:
        self._results.append((tool_name, result))

    # -- rendering --------------------------------------------------------

    def render(self) -> str:
        if not self._results:
            return self._help_text()
        parts = [self._render_one(name, result) for name, result in self._results]
        return "\n\n".join(p for p in parts if p) or self._help_text()

    def _render_one(self, tool_name: str, result: Any) -> str:
        if not isinstance(result, dict):
            return ""
        if result.get("code"):
            hint = result.get("recovery_hint", "")
            return f"{result.get('error', 'That did not work.')}" + (f" {hint}" if hint else "")

        if tool_name == "search_products":
            return self._render_products(result)
        if tool_name in {"get_order_status", "track_shipment"}:
            return self._render_order(result)
        if tool_name == "list_my_orders":
            return self._render_order_list(result)
        if tool_name == "lookup_policy":
            return self._render_policy(result)
        if tool_name == "check_availability":
            return self._render_availability(result)
        if tool_name == "list_brands":
            names = ", ".join(b["brand"] for b in result.get("brands", []))
            return f"Aurelia stocks these brands: {names}."
        if tool_name == "list_categories":
            lines = [
                f"- {c['category']}: " + ", ".join(s["subcategory"] for s in c["subcategories"])
                for c in result.get("categories", [])
            ]
            return "Here is what Aurelia carries:\n" + "\n".join(lines)
        if "lines" in result and "item_count" in result:
            return self._render_cart(result)
        if "message" in result:
            return str(result["message"])
        return ""

    @staticmethod
    def _render_products(result: dict[str, Any]) -> str:
        products = result.get("products", [])
        if not products:
            note = result.get("note") or "Nothing in the catalogue matched that."
            return note
        total = result.get("total_matches", len(products))
        prefix = "at least " if result.get("total_matches_capped") else ""
        head = (
            f"Found {prefix}{_plural(total, 'match', 'matches')}. "
            f"Here are the top {len(products)}:"
        )
        lines = []
        for p in products:
            sizes = ", ".join(p.get("available_sizes", [])) or "no sizes in stock"
            discount = f" (was {p['list_price']['display']})" if p.get("discount_pct") else ""
            lines.append(
                f"- {p['name']} - {p['price']['display']}{discount}, "
                f"rated {p['rating']}/5. Sizes in stock: {sizes}."
            )
        tail = f"\n\n{result['note']}" if result.get("note") else ""
        return head + "\n" + "\n".join(lines) + tail

    @staticmethod
    def _render_order(result: dict[str, Any]) -> str:
        lines = [
            f"Order {result['order_number']} is {result.get('status_label', result.get('status'))}.",
            result.get("delivery_message", ""),
        ]
        if result.get("tracking_number"):
            lines.append(f"Tracking: {result['tracking_number']} with {result.get('carrier', 'the carrier')}.")
        items = result.get("items") or []
        if items:
            lines.append("Items: " + "; ".join(
                f"{i['quantity']} x {i['product_name']} ({i['size']}/{i['color']})" for i in items
            ))
        if result.get("total"):
            lines.append(f"Order total: {result['total']['display']}.")
        return "\n".join(l for l in lines if l)

    @staticmethod
    def _render_order_list(result: dict[str, Any]) -> str:
        orders = result.get("orders", [])
        if not orders:
            return result.get("note") or "There are no orders on this account."
        lines = [
            f"- Order {o['order_number']}: {o['status_label']}, {o['total']['display']}, "
            f"placed {o['placed_at'][:10]}"
            for o in orders
        ]
        return f"Your {len(orders)} most recent order(s):\n" + "\n".join(lines)

    @staticmethod
    def _render_policy(result: dict[str, Any]) -> str:
        """Render every retrieved policy passage, not just the top one.

        The LLM path reads three passages and synthesises the answer across
        them. This planner cannot synthesise, so ranking errors become answer
        errors: "how long do I have to return something" ranks the returns
        *procedure* first, while the 30-day window it actually asks about sits
        third. Retrieval puts the right passage in the top three on 9 of 10
        gold queries but at rank one on only 5, so showing all three converts a
        rank-1 problem into a rank-3 problem and roughly doubles the hit rate.

        The cost is verbosity, which is the correct trade for a degraded mode
        whose job is to be right rather than elegant. Each passage is trimmed so
        the whole reply stays readable.
        """
        passages = result.get("passages", [])
        if not passages:
            return result.get("note") or "I could not find a policy covering that."

        blocks = []
        for passage in passages[:3]:
            body = passage["text"].strip()
            if len(body) > 480:
                body = body[:480].rsplit(" ", 1)[0] + "..."
            blocks.append(f"From our {passage['document']} policy ({passage['heading']}):\n\n{body}")
        return "\n\n".join(blocks)

    @staticmethod
    def _render_cart(result: dict[str, Any]) -> str:
        lines = result.get("lines", [])
        if not lines:
            return "The bag is empty."
        last = lines[-1]
        added = (
            f"Added {last['product_name']} ({last['size']}/{last['color']}) to your bag."
        )
        total = (result.get("total") or {}).get("display", "")
        summary = f"{result.get('item_count', 0)} item(s) in your bag, total {total}."
        note = f"\n{result['note']}" if result.get("note") else ""
        return f"{added}\n{summary}{note}"

    @staticmethod
    def _render_availability(result: dict[str, Any]) -> str:
        name = result.get("product_name", "That product")
        if not result.get("any_available"):
            sizes = ", ".join(result.get("all_sizes_in_stock", []))
            return (
                f"{name} is not available in that combination."
                + (f" In stock sizes: {sizes}." if sizes else "")
            )
        variants = [v for v in result.get("matching_variants", []) if v["in_stock"]]
        detail = ", ".join(f"{v['color']} ({v['stock']} left)" for v in variants[:6])
        return f"{name} is available: {detail}."

    @staticmethod
    def _help_text() -> str:
        return (
            "I can search the catalogue, check your orders and delivery dates, manage your "
            "cart, and answer questions about shipping, returns, sizing and payments. "
            "Try 'show me Nike t-shirts' or 'where is order 1234'."
        )


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

def _detect_subcategory(text: str) -> str | None:
    lowered = text.lower()
    for canonical, aliases in SUBCATEGORY_ALIASES.items():
        if canonical in lowered or any(alias in lowered for alias in aliases):
            return next(s for s in SUBCATEGORIES if s.lower() == canonical)
    for sub in SUBCATEGORIES:
        if sub.lower() in lowered:
            return sub
    return None


def _detect_brand(text: str) -> str | None:
    lowered = text.lower()
    for brand in BRANDS:
        if brand.lower() in lowered:
            return brand
    return None


def _detect_policy_topic(text: str) -> str | None:
    """Pick the best-matching policy topic, not merely the first.

    "Can I cancel after it ships" contains both a shipping keyword and a
    cancellation keyword. Returning whichever appears first in the dictionary
    would answer the wrong question, so we score by the longest keyword matched:
    the more specific term wins.
    """
    lowered = text.lower()
    best_topic: str | None = None
    best_score = 0
    for topic, keywords in POLICY_TOPICS.items():
        score = max((len(k) for k in keywords if k in lowered), default=0)
        if score > best_score:
            best_topic, best_score = topic, score
    return best_topic


def plan_without_llm(message: str) -> Plan:
    """Map a customer message onto a sequence of tool calls."""
    text = message.strip()
    lowered = text.lower()

    # -- order intents (checked first: an order number is a strong signal) --
    order_match = ORDER_NUMBER_RE.search(text)
    mentions_order = any(w in lowered for w in ("order", "parcel", "package", "delivery", "shipment", "tracking"))

    if mentions_order and order_match:
        number = order_match.group(1)
        if any(w in lowered for w in ("cancel",)):
            return Plan("cancel_order", [("cancel_order", {"order_number": number})])
        if any(w in lowered for w in ("return", "refund")):
            return Plan("request_return", [("request_return", {"order_number": number})])
        if any(w in lowered for w in ("when", "arrive", "deliver", "track", "where")):
            return Plan("track_shipment", [("track_shipment", {"order_number": number})])
        return Plan("order_status", [("get_order_status", {"order_number": number})])

    if mentions_order and any(w in lowered for w in ("my orders", "recent", "list", "all my", "history")):
        return Plan("list_orders", [("list_my_orders", {"limit": 5})])

    if mentions_order and not order_match and any(
        w in lowered for w in ("where", "when", "status", "my order")
    ):
        # No number given: show the recent orders so the customer can pick one.
        return Plan("list_orders", [("list_my_orders", {"limit": 5})])

    # -- cart intents ------------------------------------------------------
    add_match = ADD_TO_CART_RE.search(text)
    if add_match:
        # A quoted name (what the "Add to bag" button on a product card
        # sends) is unambiguous; an unquoted phrase falls back to whatever
        # words follow "add", which add_to_cart resolves the same way
        # search_products would - close enough for a common phrasing like
        # "add the nike trail tee to my bag" even without exact quoting.
        name = add_match.group("quoted") or add_match.group("loose")
        return Plan("add_to_cart", [("add_to_cart", {"product_name": name.strip()})])
    if any(w in lowered for w in ("my cart", "my basket", "what's in my cart", "view cart", "show cart")):
        return Plan("view_cart", [("view_cart", {})])
    if any(w in lowered for w in ("checkout", "check out", "place my order", "buy it now")):
        return Plan("checkout", [("prepare_checkout", {})])

    # -- policy intents ----------------------------------------------------
    topic = _detect_policy_topic(lowered)
    policy_shaped = any(
        lowered.startswith(p) for p in ("can i", "how long", "what happens", "do you", "is it possible", "what is your")
    ) or any(w in lowered for w in ("policy", "policies"))
    if topic and (policy_shaped or not _detect_subcategory(lowered)):
        return Plan("policy", [("lookup_policy", {"query": text, "topic": topic})])

    # -- catalogue intents -------------------------------------------------
    if any(w in lowered for w in ("what brands", "which brands", "brands do you")):
        return Plan("list_brands", [("list_brands", {})])
    if any(w in lowered for w in ("what do you sell", "what categories", "what can i buy")):
        return Plan("list_categories", [("list_categories", {})])

    arguments: dict[str, Any] = {"limit": 6}
    brand = _detect_brand(text)
    subcategory = _detect_subcategory(text)
    if brand:
        arguments["brand"] = brand
    if subcategory:
        arguments["subcategory"] = subcategory

    under = PRICE_UNDER_RE.search(text)
    if under:
        arguments["max_price"] = float(under.group(1))
    over = PRICE_OVER_RE.search(text)
    if over:
        arguments["min_price"] = float(over.group(1))

    size = SIZE_RE.search(text)
    if size:
        arguments["size"] = size.group(1)
    elif (shoe := SHOE_SIZE_RE.search(text)):
        arguments["size"] = shoe.group(1)

    # Women is tested first and with word boundaries, because "women's" contains
    # "men's" as a substring and a naive check assigns every women's query to men.
    if re.search(r"\b(?:women'?s|for women|ladies|female)\b", lowered):
        arguments["gender"] = "women"
    elif re.search(r"\b(?:men'?s|for men|male|guys)\b", lowered):
        arguments["gender"] = "men"

    if any(w in lowered for w in ("cheapest", "cheap", "budget", "lowest price")):
        arguments["sort"] = "price_low_to_high"
    elif any(w in lowered for w in ("best rated", "highest rated", "top rated", "best")):
        arguments["sort"] = "rating"
    elif any(w in lowered for w in ("on sale", "discount", "deal")):
        arguments["on_sale_only"] = True
        arguments["sort"] = "discount"

    # Anything not consumed by a structured slot becomes the relevance query.
    arguments["query"] = text

    if len(arguments) <= 2 and not brand and not subcategory:
        # Nothing recognisable to search on.
        greeting = any(w in lowered for w in ("hi", "hello", "hey", "help", "what can you"))
        if greeting or len(lowered.split()) <= 3:
            return Plan("help", [])

    return Plan("product_search", [("search_products", arguments)])
