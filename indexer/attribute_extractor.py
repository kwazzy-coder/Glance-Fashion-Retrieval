"""
Rule-based attribute extractor for fashion image captions.

Parses BLIP-2 generated captions to identify clothing items, colours,
environmental context, style, and accessories using the taxonomy and
vocabulary defined in ``config.py``.
"""

import logging
import re
from collections import Counter
from typing import Dict, List

import config

logger = logging.getLogger(__name__)

# Pre-compile word-boundary patterns for multi-word aliases that could
# otherwise produce false positives (e.g. "ash" inside "fashion").
# We sort aliases longest-first so that "royal blue" is matched before "blue".
_SORTED_COLOR_ALIASES: List[str] = sorted(
    config.ALIAS_TO_COLOR.keys(), key=len, reverse=True
)

_SORTED_CLOTHING_ITEMS: List[str] = sorted(
    config.ITEM_TO_CATEGORY.keys(), key=len, reverse=True
)

_SORTED_ENV_KEYWORDS: Dict[str, List[str]] = {
    env: sorted(kws, key=len, reverse=True)
    for env, kws in config.ENVIRONMENT_KEYWORDS.items()
}

_SORTED_STYLE_KEYWORDS: Dict[str, List[str]] = {
    style: sorted(kws, key=len, reverse=True)
    for style, kws in config.STYLE_KEYWORDS.items()
}

# Items from the taxonomy whose category is "accessories"
_ACCESSORY_ITEMS: List[str] = sorted(
    config.CLOTHING_TAXONOMY.get("accessories", []), key=len, reverse=True
)


def _word_present(phrase: str, text: str) -> bool:
    """Return ``True`` if *phrase* appears in *text* on a word boundary.

    Uses ``re.search`` with ``\\b`` anchors so that e.g. ``"ash"`` does
    not match inside ``"fashion"``.
    """
    return bool(re.search(r"\b" + re.escape(phrase) + r"\b", text))


class AttributeExtractor:
    """Extracts structured fashion attributes from a caption string.

    All detection is rule-based: the caption is lowercased and scanned
    against the taxonomy dictionaries in ``config.py``.
    """

    # ── public API ──────────────────────────────────────────────────────

    def extract_attributes(self, caption: str) -> Dict:
        """Parse *caption* into a structured attribute dictionary.

        Parameters
        ----------
        caption : str
            A natural-language image description (typically from BLIP-2).

        Returns
        -------
        dict
            Keys:
            - ``clothing_types``     : list[str] – detected clothing items
            - ``clothing_categories``: list[str] – inferred taxonomy categories
            - ``colors``             : list[str] – canonical colour names
            - ``environment``        : str       – best-matching environment
            - ``style``              : str       – inferred overall style
            - ``accessories``        : list[str] – detected accessory items
            - ``raw_caption``        : str       – original caption text
        """
        text = caption.lower()

        clothing_types = self._detect_clothing(text)
        clothing_categories = self._infer_categories(clothing_types)
        colors = self._detect_colors(text)
        environment = self._detect_environment(text)
        accessories = self._detect_accessories(text)
        style = self._infer_style(text, clothing_categories)

        attributes = {
            "clothing_types": clothing_types,
            "clothing_categories": clothing_categories,
            "colors": colors,
            "environment": environment,
            "style": style,
            "accessories": accessories,
            "raw_caption": caption,
        }

        logger.debug("Extracted attributes: %s", attributes)
        return attributes

    # ── internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _detect_clothing(text: str) -> List[str]:
        """Return all clothing items from the taxonomy found in *text*."""
        found: List[str] = []
        for item in _SORTED_CLOTHING_ITEMS:
            if _word_present(item, text):
                found.append(item)
        # Deduplicate while preserving order
        return list(dict.fromkeys(found)) if found else ["unknown"]

    @staticmethod
    def _infer_categories(clothing_types: List[str]) -> List[str]:
        """Map detected clothing items to their taxonomy categories."""
        if clothing_types == ["unknown"]:
            return ["unknown"]
        categories = [
            config.ITEM_TO_CATEGORY[item]
            for item in clothing_types
            if item in config.ITEM_TO_CATEGORY
        ]
        # Unique, order-preserving
        return list(dict.fromkeys(categories)) if categories else ["unknown"]

    @staticmethod
    def _detect_colors(text: str) -> List[str]:
        """Return canonical colour names for every alias found in *text*.

        Longer aliases are checked first so that *"royal blue"* is not
        shadowed by a bare *"blue"* match.
        """
        found_canonical: List[str] = []
        for alias in _SORTED_COLOR_ALIASES:
            if _word_present(alias, text):
                canonical = config.ALIAS_TO_COLOR[alias]
                if canonical not in found_canonical:
                    found_canonical.append(canonical)
        return found_canonical if found_canonical else ["unknown"]

    @staticmethod
    def _detect_environment(text: str) -> str:
        """Pick the environment category with the most keyword hits."""
        scores: Counter = Counter()
        for env, keywords in _SORTED_ENV_KEYWORDS.items():
            for kw in keywords:
                if _word_present(kw, text):
                    scores[env] += 1

        if not scores:
            return "unknown"
        return scores.most_common(1)[0][0]

    @staticmethod
    def _detect_accessories(text: str) -> List[str]:
        """Return accessory items found in *text*."""
        found: List[str] = []
        for item in _ACCESSORY_ITEMS:
            if _word_present(item, text):
                found.append(item)
        return list(dict.fromkeys(found)) if found else []

    @staticmethod
    def _infer_style(text: str, clothing_categories: List[str]) -> str:
        """Infer the dominant style from categories and keyword matches.

        Strategy:
        1. Count keyword hits per style in the caption.
        2. Count how many clothing items fall into each category (which
           maps roughly to a style).
        3. Combine the two signals; the style with the highest total wins.
        """
        style_scores: Counter = Counter()

        # Signal 1 – direct keyword matches in caption
        for style, keywords in _SORTED_STYLE_KEYWORDS.items():
            for kw in keywords:
                if _word_present(kw, text):
                    style_scores[style] += 1

        # Signal 2 – category-based inference
        category_to_style = {
            "formal": "formal",
            "casual": "casual",
            "activewear": "sporty",
            "outerwear": "casual",   # outerwear is style-neutral; default casual
            "accessories": "casual",
        }
        if clothing_categories != ["unknown"]:
            for cat in clothing_categories:
                mapped_style = category_to_style.get(cat)
                if mapped_style:
                    style_scores[mapped_style] += 1

        if not style_scores:
            return "unknown"
        return style_scores.most_common(1)[0][0]
