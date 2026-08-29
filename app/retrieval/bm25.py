"""Okapi BM25 over an in-memory corpus.

Why hand-rolled rather than a vector database
---------------------------------------------
The brief requires the application to run on standard developer hardware with
free tools and no special infrastructure. Shipping a transformer embedding model
would add a multi-hundred-megabyte download and a torch dependency that does not
have wheels on every Python version, for a catalogue of ~1k products where
lexical matching over a controlled vocabulary (brands, categories, colours,
sizes) is genuinely the stronger signal. Retail search is dominated by exact
brand and attribute matching; "Nike" must match Nike and not "Adidas" because
the vectors were close.

What we do instead is take the parts of semantic search that actually pay off at
this scale and implement them explicitly:

* a domain synonym table, so "tee" and "sneakers" reach the right documents;
* light suffix normalisation, so "shoes" and "shoe" collapse;
* field weighting, so a brand match in the title outranks the same token buried
  in a description;
* reciprocal rank fusion across a title index and a full-text index.

`app/retrieval/index.py` exposes this behind a `Retriever` protocol, so swapping
in a dense retriever later is a single class, not a rewrite. The scaling note in
docs/SCALING.md sets out exactly what that migration looks like.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

#: Words that carry no retrieval signal in a shopping query.
STOPWORDS = frozenset("""
a an the and or of for with in on at to from by is are was were be been being
do does did have has had i me my we our you your it its this that these those
please can could would should will shall want need looking look show me
available availability any some all what which who whom whose how when where
there here about into over under than then so if not no yes ok okay
""".split())

#: Retail vocabulary. Maps a surface form to the canonical tokens it should also
#: match. Bidirectional entries are listed explicitly for clarity.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "tee": ("t", "shirt", "tshirt"),
    "tees": ("t", "shirt", "tshirt"),
    "tshirt": ("t", "shirt"),
    "tshirts": ("t", "shirt"),
    "sneaker": ("sneakers", "shoe", "footwear", "trainer"),
    "sneakers": ("sneaker", "shoe", "footwear", "trainer"),
    "trainer": ("shoe", "sneakers", "footwear"),
    "trainers": ("shoe", "sneakers", "footwear"),
    "kicks": ("shoe", "sneakers", "footwear"),
    "shoe": ("footwear", "sneakers"),
    "shoes": ("footwear", "sneakers"),
    "runner": ("running", "shoe", "footwear"),
    "runners": ("running", "shoe", "footwear"),
    "pant": ("pants", "bottomwear", "trouser"),
    "pants": ("bottomwear", "trouser", "trousers"),
    "trouser": ("pants", "bottomwear"),
    "trousers": ("pants", "bottomwear"),
    "denim": ("jeans", "bottomwear"),
    "jean": ("jeans", "denim", "bottomwear"),
    "jeans": ("denim", "bottomwear"),
    "jumper": ("sweatshirt", "hoodie", "topwear"),
    "sweater": ("sweatshirt", "topwear"),
    "hoody": ("hoodie", "topwear"),
    "hoodie": ("topwear", "sweatshirt"),
    "coat": ("jacket", "outerwear"),
    "jacket": ("outerwear",),
    "puffer": ("jacket", "outerwear"),
    "windcheater": ("windbreaker", "jacket", "outerwear"),
    "shades": ("sunglasses", "eyewear", "accessories"),
    "sunglass": ("sunglasses", "eyewear"),
    "specs": ("sunglasses", "eyewear"),
    "bag": ("backpack", "accessories"),
    "rucksack": ("backpack", "accessories"),
    "hat": ("cap", "accessories"),
    "beanie": ("cap", "accessories"),
    "watch": ("accessories", "timepiece"),
    "top": ("topwear",),
    "tops": ("topwear",),
    "bottoms": ("bottomwear",),
    "activewear": ("sportswear", "performance", "training"),
    "sportswear": ("activewear", "performance"),
    "gym": ("training", "performance", "sportswear"),
    "workout": ("training", "performance", "sportswear"),
    "cheap": ("budget", "affordable", "sale"),
    "affordable": ("budget", "cheap"),
    "discounted": ("sale", "discount"),
    "womens": ("women", "woman", "female"),
    "mens": ("men", "man", "male"),
    "guys": ("men",),
    "ladies": ("women",),
    "waterproof": ("water", "repellent", "dryvent", "rain"),
    "breathable": ("dri", "fit", "aeroready", "airism"),
}

#: Irregular plural and suffix rules, applied in order. Cheaper and far more
#: predictable than a full Porter stemmer, and it never mangles a brand name.
_SUFFIXES = (("ies", "y"), ("sses", "ss"), ("shes", "sh"), ("ches", "ch"), ("s", ""))
_NEVER_STEM = frozenset({"adidas", "levis", "levi's", "dress", "class", "gucci", "puma", "s", "xs", "xxs"})


def normalise(token: str) -> str:
    """Collapse simple plurals so 'shoes' and 'shoe' share a term."""
    if len(token) <= 3 or token in _NEVER_STEM:
        return token
    for suffix, replacement in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + replacement
    return token


def tokenize(text: str, *, expand: bool = False) -> list[str]:
    """Lowercase, strip stopwords, normalise, and optionally expand synonyms.

    Synonym expansion is applied to *queries* only. Expanding documents too
    would inflate term frequencies and distort the IDF distribution.
    """
    raw = _TOKEN_RE.findall(text.lower())
    tokens: list[str] = []
    for token in raw:
        if token in STOPWORDS:
            continue
        if expand:
            for extra in SYNONYMS.get(token, ()):
                tokens.append(normalise(extra))
        tokens.append(normalise(token))
    return tokens


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BM25Index:
    """Okapi BM25 with the standard k1/b parameterisation.

    k1 controls term-frequency saturation and b controls length normalisation.
    The defaults (1.5 / 0.75) are the usual starting point; b is worth lowering
    for short title fields, which is why the title index is constructed with
    b=0.4 in `index.py`.
    """

    k1: float = 1.5
    b: float = 0.75

    doc_ids: list[str] = field(default_factory=list)
    doc_lengths: list[int] = field(default_factory=list)
    term_frequencies: list[Counter[str]] = field(default_factory=list)
    postings: dict[str, list[int]] = field(default_factory=dict)
    average_length: float = 0.0

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        index = len(self.doc_ids)
        self.doc_ids.append(doc_id)
        self.doc_lengths.append(len(tokens))
        counts = Counter(tokens)
        self.term_frequencies.append(counts)
        for term in counts:
            self.postings.setdefault(term, []).append(index)

    def finalise(self) -> "BM25Index":
        total = sum(self.doc_lengths)
        self.average_length = total / len(self.doc_lengths) if self.doc_lengths else 0.0
        return self

    def _idf(self, term: str) -> float:
        """Robertson/Sparck-Jones IDF with the +0.5 smoothing and a floor.

        The raw formula goes negative for terms present in more than half the
        corpus, which would let a common term subtract from a document's score.
        We clamp at a small positive epsilon instead.
        """
        n_docs = len(self.doc_ids)
        df = len(self.postings.get(term, ()))
        if df == 0:
            return 0.0
        value = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        return max(value, 1e-6)

    def search(self, query: str, limit: int = 50) -> list[tuple[str, float]]:
        """Return `(doc_id, score)` pairs ordered by descending relevance."""
        if not self.doc_ids:
            return []
        terms = tokenize(query, expand=True)
        if not terms:
            return []

        scores: dict[int, float] = {}
        for term in set(terms):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for doc_index in postings:
                freq = self.term_frequencies[doc_index][term]
                length = self.doc_lengths[doc_index]
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * (length / self.average_length if self.average_length else 1.0)
                )
                scores[doc_index] = scores.get(doc_index, 0.0) + idf * (freq * (self.k1 + 1)) / denominator

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [(self.doc_ids[i], score) for i, score in ranked]

    def __len__(self) -> int:
        return len(self.doc_ids)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    *,
    weights: list[float] | None = None,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge several ranked lists into one.

    RRF combines rankings by position rather than by score, which means it does
    not care that a title-index BM25 score and a body-index BM25 score live on
    different scales. That property is exactly why it is the standard choice for
    hybrid retrieval, and it is why we do not need to normalise the two indices
    against each other.
    """
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must align with rankings")

    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for position, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + position)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)
