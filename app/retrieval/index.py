"""Retrieval indices for the catalogue and the policy knowledge base.

Two corpora, two access patterns:

* **Catalogue** is structured. The authoritative answer to "is it in stock" is a
  SQL row, never a retrieved passage. Retrieval here is a *ranking* step that
  narrows thousands of rows to a shortlist; the shortlist is then re-read from
  the database so that price and stock in the reply are live values.
* **Policy** is unstructured prose. This is classical RAG: chunk, index,
  retrieve, and hand the passages to the model with citations.

Both are wrapped in a single `RetrievalService` that the tool layer consumes, so
the tools never know which retrieval strategy is behind a query.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import POLICY_DIR
from app.db.models import Product
from app.retrieval.bm25 import BM25Index, normalise, reciprocal_rank_fusion, tokenize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy chunking
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class PolicyChunk:
    """One retrievable passage of a policy document."""

    chunk_id: str
    document: str
    topic: str
    heading: str
    text: str
    source_path: str

    def citation(self) -> str:
        return f"{self.document} > {self.heading}"


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, raw[match.end():]


def chunk_policy_document(path: Path) -> list[PolicyChunk]:
    """Split a policy markdown file on its `##` headings.

    Heading-aligned chunking beats fixed-size windows for this corpus because
    each section is already a self-contained rule ("When an order can be
    cancelled", "Refund timing"). A fixed 500-token window would routinely cut a
    table in half and strand the header row, which is precisely the content the
    model needs to answer correctly.
    """
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(raw)
    document = meta.get("title", path.stem.replace("-", " ").title())
    topic = meta.get("topic", path.stem)

    chunks: list[PolicyChunk] = []
    current_heading = document
    buffer: list[str] = []
    index = 0

    def flush() -> None:
        nonlocal index, buffer
        text = "\n".join(buffer).strip()
        if len(text) >= 40:  # skip empty or heading-only fragments
            index += 1
            chunks.append(
                PolicyChunk(
                    chunk_id=f"{path.stem}#{index}",
                    document=document,
                    topic=topic,
                    heading=current_heading,
                    text=text,
                    source_path=str(path.relative_to(path.parent.parent.parent)),
                )
            )
        buffer = []

    for line in body.splitlines():
        if line.startswith("## "):
            flush()
            current_heading = line[3:].strip()
            continue
        if line.startswith("# "):
            continue
        buffer.append(line)
    flush()
    return chunks


# ---------------------------------------------------------------------------
# Retrieval service
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class ProductHit:
    product_id: int
    score: float


class RetrievalService:
    """Owns both indices and the vocabulary derived from the catalogue.

    The index is built once at startup and rebuilt on demand. For a catalogue of
    this size a full rebuild is a sub-second operation, so incremental updates
    would be complexity without benefit. `docs/SCALING.md` covers what changes
    when the catalogue reaches a size where that stops being true.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._title_index = BM25Index(b=0.4)      # short field, less length penalty
        self._body_index = BM25Index(b=0.75)
        self._policy_heading_index = BM25Index(b=0.35)
        self._policy_index = BM25Index(b=0.6)
        self._policy_chunks: dict[str, PolicyChunk] = {}
        self._brands: list[str] = []
        self._categories: list[str] = []
        self._subcategories: list[str] = []
        self._colors: list[str] = []
        self._built = False

    # -- construction ----------------------------------------------------

    def build(self, session: Session) -> None:
        """Build both indices from the database and the policy directory."""
        with self._lock:
            title_index = BM25Index(b=0.4)
            body_index = BM25Index(b=0.75)
            brands: set[str] = set()
            categories: set[str] = set()
            subcategories: set[str] = set()
            colors: set[str] = set()

            products = session.scalars(select(Product).where(Product.is_active.is_(True))).all()
            for product in products:
                doc_id = str(product.id)
                title_index.add(doc_id, f"{product.name} {product.brand} {product.subcategory}")
                body_index.add(doc_id, product.search_document())
                brands.add(product.brand)
                categories.add(product.category)
                subcategories.add(product.subcategory)
                colors.update(v.color for v in product.variants)

            self._title_index = title_index.finalise()
            self._body_index = body_index.finalise()
            self._brands = sorted(brands)
            self._categories = sorted(categories)
            self._subcategories = sorted(subcategories)
            self._colors = sorted(colors)

            policy_index = BM25Index(b=0.6)
            heading_index = BM25Index(b=0.35)
            chunks: dict[str, PolicyChunk] = {}
            for path in sorted(POLICY_DIR.glob("*.md")):
                for chunk in chunk_policy_document(path):
                    chunks[chunk.chunk_id] = chunk
                    policy_index.add(chunk.chunk_id, f"{chunk.heading}\n{chunk.text}")
                    heading_index.add(chunk.chunk_id, f"{chunk.document} {chunk.topic} {chunk.heading}")
            self._policy_index = policy_index.finalise()
            self._policy_heading_index = heading_index.finalise()
            self._policy_chunks = chunks
            self._built = True

            logger.info(
                "retrieval.index_built",
                extra={
                    "products": len(self._body_index),
                    "policy_chunks": len(chunks),
                    "brands": len(self._brands),
                },
            )

    @property
    def ready(self) -> bool:
        return self._built

    def stats(self) -> dict[str, int]:
        return {
            "products_indexed": len(self._body_index),
            "policy_chunks_indexed": len(self._policy_index),
            "brands": len(self._brands),
            "categories": len(self._categories),
        }

    # -- vocabulary ------------------------------------------------------

    @property
    def brands(self) -> list[str]:
        return list(self._brands)

    @property
    def categories(self) -> list[str]:
        return list(self._categories)

    @property
    def subcategories(self) -> list[str]:
        return list(self._subcategories)

    @property
    def colors(self) -> list[str]:
        return list(self._colors)

    def resolve_brand(self, value: str) -> str | None:
        """Map free text onto a catalogue brand, tolerating spelling drift.

        The model is good at extracting "Nike" from a sentence but customers
        type "addidas" and "levis". Resolving against the real brand list here,
        rather than trusting the string straight into a SQL filter, turns a
        zero-result query into a correct one.
        """
        if not value:
            return None
        cleaned = value.strip().lower()
        by_lower = {brand.lower(): brand for brand in self._brands}
        if cleaned in by_lower:
            return by_lower[cleaned]
        # Punctuation-insensitive match: "levis" -> "Levi's".
        squashed = {re.sub(r"[^a-z0-9]", "", k): v for k, v in by_lower.items()}
        key = re.sub(r"[^a-z0-9]", "", cleaned)
        if key in squashed:
            return squashed[key]
        close = get_close_matches(cleaned, list(by_lower), n=1, cutoff=0.75)
        return by_lower[close[0]] if close else None

    def resolve_vocabulary(self, value: str, options: list[str], cutoff: float = 0.8) -> str | None:
        if not value:
            return None
        cleaned = value.strip().lower()
        by_lower = {option.lower(): option for option in options}
        if cleaned in by_lower:
            return by_lower[cleaned]
        singular = normalise(cleaned)
        for lowered, original in by_lower.items():
            if normalise(lowered) == singular:
                return original
        close = get_close_matches(cleaned, list(by_lower), n=1, cutoff=cutoff)
        return by_lower[close[0]] if close else None

    # -- search ----------------------------------------------------------

    def search_products(self, query: str, limit: int = 60) -> list[ProductHit]:
        """Rank product ids for a free-text query using title/body fusion.

        The title ranking is weighted higher than the body ranking: a customer
        asking for "Nike t-shirt" wants products whose *name* says Nike t-shirt,
        not products whose description happens to mention both words.
        """
        with self._lock:
            if not query.strip():
                return []
            title_hits = self._title_index.search(query, limit=limit * 2)
            body_hits = self._body_index.search(query, limit=limit * 2)
            fused = reciprocal_rank_fusion([title_hits, body_hits], weights=[1.6, 1.0])
            return [ProductHit(product_id=int(doc_id), score=score) for doc_id, score in fused[:limit]]

    def search_policies(self, query: str, limit: int = 4, topic: str | None = None) -> list[tuple[PolicyChunk, float]]:
        """Retrieve policy passages, optionally constrained to one topic.

        Section headings in this corpus are unusually informative ("Refund
        timing on cancellation", "Non-returnable items"), so they get their own
        index and are fused with the body ranking. Without that, a passage whose
        heading is a near-exact restatement of the question can be outranked by
        a longer passage that merely repeats the query terms more often.
        """
        with self._lock:
            body_hits = self._policy_index.search(query, limit=limit * 8)
            heading_hits = self._policy_heading_index.search(query, limit=limit * 8)
            hits = reciprocal_rank_fusion([heading_hits, body_hits], weights=[1.3, 1.0])
            results: list[tuple[PolicyChunk, float]] = []
            for chunk_id, score in hits:
                chunk = self._policy_chunks.get(chunk_id)
                if chunk is None:
                    continue
                if topic and chunk.topic != topic:
                    continue
                results.append((chunk, score))
                if len(results) >= limit:
                    break
            return results

    def policy_topics(self) -> list[str]:
        with self._lock:
            return sorted({chunk.topic for chunk in self._policy_chunks.values()})


#: Process-wide retrieval service. Built during application startup.
retrieval_service = RetrievalService()
