"""
Deterministic query decomposer for fashion retrieval.

Parses natural-language queries into structured attribute constraints
(clothing items with bound colours, accessories, environment, style, and
compositional pairs) without requiring an LLM. All vocabulary comes from
``config.py`` so the logic stays fast, reproducible, and CPU-only.
"""

import logging
import re
from typing import Any

import config

logger = logging.getLogger(__name__)


class QueryDecomposer:
    """Decomposes a free-text query into structured fashion attributes."""

    def __init__(self) -> None:
        self._clothing_items: list[str] = sorted(
            config.ITEM_TO_CATEGORY.keys(), key=len, reverse=True
        )
        self._color_aliases: list[str] = sorted(
            config.ALIAS_TO_COLOR.keys(), key=len, reverse=True
        )
        self._environment_keywords: dict[str, list[str]] = config.ENVIRONMENT_KEYWORDS
        self._style_keywords: dict[str, list[str]] = config.STYLE_KEYWORDS
        self._item_to_category: dict[str, str] = config.ITEM_TO_CATEGORY
        self._style_synonyms: dict[str, str] = {
            "formal": "professional business elegant sophisticated",
            "casual": "relaxed comfortable everyday laid-back",
            "sporty": "athletic active fitness workout",
            "streetwear": "urban trendy edgy street style",
            "bohemian": "boho hippie artistic eclectic vintage",
            "minimalist": "simple clean understated sleek",
        }
        logger.info("QueryDecomposer initialised (CPU-only, deterministic).")

    def decompose(self, query: str) -> dict[str, Any]:
        """Decompose a free-form query into structured fashion constraints."""
        lowered = query.lower().strip()
        clothing_items = self._extract_clothing_items_with_colors(lowered)
        colors = list(
            dict.fromkeys(
                item["color"] for item in clothing_items if item["color"] is not None
            )
        )
        clothing_types = list(dict.fromkeys(item["type"] for item in clothing_items))
        accessories = self._extract_accessories(lowered, clothing_items)
        environment = self._detect_environment(lowered)
        style = self._detect_style(lowered, clothing_types, accessories)
        enhanced_query = self._build_enhanced_query(
            query,
            clothing_items,
            colors,
            clothing_types,
            accessories,
            environment,
            style,
        )
        compositional_pairs = self._build_compositional_pairs(clothing_items)

        result: dict[str, Any] = {
            "clothing_items": clothing_items,
            "colors": colors,
            "clothing_types": clothing_types,
            "accessories": accessories,
            "environment": environment,
            "style": style,
            "compositional_pairs": compositional_pairs,
            "enhanced_query": enhanced_query,
            "attribute_summary": self._build_attribute_summary(
                colors,
                clothing_types,
                accessories,
                environment,
                style,
            ),
        }
        logger.debug("Decomposed query: %s", result)
        return result

    def _extract_clothing_items_with_colors(
        self, text: str
    ) -> list[dict[str, str | None]]:
        """Find clothing items and bind the nearest relevant colour.

        The parser prefers local context around each match so phrases such as
        "red shirt with blue pants" are handled better than a simple
        left-to-right scan.
        """
        found: list[dict[str, str | None]] = []
        consumed_spans: list[tuple[int, int]] = []
        item_matches: list[tuple[int, int, str]] = []

        for item in self._clothing_items:
            pattern = re.compile(r"\b" + re.escape(item) + r"\b")
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                if any(not (end <= cs or start >= ce) for cs, ce in consumed_spans):
                    continue
                item_matches.append((start, end, item))
                consumed_spans.append((start, end))

        item_matches.sort(key=lambda item: item[0])

        for item_start, item_end, item_name in item_matches:
            color = self._find_best_color_in_context(text, item_start, item_end)
            found.append({"type": item_name, "color": color})

        return found

    def _find_best_color_in_context(self, text: str, start: int, end: int) -> str | None:
        """Return the nearest colour before or after the item match."""
        left_context = text[max(0, start - 40) : start]
        right_context = text[end : min(len(text), end + 40)]

        best_color: str | None = None
        best_distance = float("inf")

        for alias in self._color_aliases:
            pattern = re.compile(r"\b" + re.escape(alias) + r"\b")
            for match in pattern.finditer(left_context):
                distance = start - match.end()
                if 0 <= distance < best_distance:
                    best_distance = distance
                    best_color = config.ALIAS_TO_COLOR[alias]

            for match in pattern.finditer(right_context):
                distance = match.start()
                if 0 <= distance < best_distance:
                    best_distance = distance
                    best_color = config.ALIAS_TO_COLOR[alias]

        return best_color

    def _extract_accessories(
        self, text: str, clothing_items: list[dict[str, str | None]]
    ) -> list[str]:
        """Extract accessory terms from the query when present."""
        accessory_terms: list[str] = []
        for item in clothing_items:
            category = self._item_to_category.get(item["type"], "")
            if category == "accessories":
                accessory_terms.append(item["type"])

        if accessory_terms:
            return list(dict.fromkeys(accessory_terms))

        for token in ["bag", "sunglasses", "watch", "hat", "scarf", "belt"]:
            if re.search(r"\b" + re.escape(token) + r"\b", text):
                accessory_terms.append(token)
        return list(dict.fromkeys(accessory_terms))

    def _detect_environment(self, text: str) -> str | None:
        """Return the environment label with the most keyword hits."""
        best_env: str | None = None
        best_score = 0
        for env, keywords in self._environment_keywords.items():
            score = sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", text))
            if score > best_score:
                best_score = score
                best_env = env
        return best_env

    def _detect_style(
        self, text: str, clothing_types: list[str], accessories: list[str]
    ) -> str | None:
        """Detect style from direct keywords first, then from garment type."""
        best_style: str | None = None
        best_score = 0
        for style, keywords in self._style_keywords.items():
            score = sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", text))
            if score > best_score:
                best_score = score
                best_style = style

        if best_style is not None:
            return best_style

        category_counts: dict[str, int] = {}
        for ctype in clothing_types:
            cat = self._item_to_category.get(ctype)
            if cat is not None:
                category_counts[cat] = category_counts.get(cat, 0) + 1

        if accessories:
            category_counts["accessories"] = category_counts.get("accessories", 0) + 1

        if not category_counts:
            return None

        majority_category = max(category_counts, key=lambda key: category_counts[key])
        if majority_category in self._style_keywords:
            return majority_category
        return None

    def _build_enhanced_query(
        self,
        original_query: str,
        clothing_items: list[dict[str, str | None]],
        colors: list[str],
        clothing_types: list[str],
        accessories: list[str],
        environment: str | None,
        style: str | None,
    ) -> str:
        """Build an expanded query string for better vector-search recall."""
        parts: list[str] = [original_query.strip()]

        for color in colors:
            if color.lower() not in original_query.lower():
                parts.append(color)

        for ctype in clothing_types:
            if ctype.lower() not in original_query.lower():
                parts.append(ctype)

        for accessory in accessories:
            if accessory.lower() not in original_query.lower():
                parts.append(accessory)

        if environment is not None:
            if environment.lower() not in original_query.lower():
                parts.append(environment)
            env_kws = self._environment_keywords.get(environment, [])
            for kw in env_kws[:3]:
                if kw.lower() not in original_query.lower():
                    parts.append(kw)

        if style is not None:
            synonyms = self._style_synonyms.get(style, "")
            for token in synonyms.split():
                if token.lower() not in original_query.lower():
                    parts.append(token)

        seen_categories: set[str] = set()
        for ctype in clothing_types:
            cat = self._item_to_category.get(ctype)
            if cat and cat not in seen_categories:
                seen_categories.add(cat)
                if cat.lower() not in original_query.lower():
                    parts.append(cat)

        enhanced = " ".join(parts)
        enhanced = re.sub(r"\s+", " ", enhanced).strip()
        return enhanced

    def _build_compositional_pairs(
        self, clothing_items: list[dict[str, str | None]]
    ) -> list[dict[str, str]]:
        """Create a compact list of colour→garment pairs for reranking."""
        pairs: list[dict[str, str]] = []
        for item in clothing_items:
            if item.get("color") and item.get("type"):
                pairs.append({"color": item["color"], "item": item["type"]})
        return pairs

    def _build_attribute_summary(
        self,
        colors: list[str],
        clothing_types: list[str],
        accessories: list[str],
        environment: str | None,
        style: str | None,
    ) -> str:
        """Return a concise, human-readable attribute summary."""
        parts: list[str] = []
        if colors:
            parts.append("colors=" + ",".join(colors))
        if clothing_types:
            parts.append("garments=" + ",".join(clothing_types))
        if accessories:
            parts.append("accessories=" + ",".join(accessories))
        if environment:
            parts.append("environment=" + environment)
        if style:
            parts.append("style=" + style)
        return " | ".join(parts) if parts else "no explicit attributes"
