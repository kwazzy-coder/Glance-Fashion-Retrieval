"""
Deterministic query decomposer for fashion retrieval.

Parses natural-language queries into structured attribute constraints
(clothing items with bound colors, environment, style) without requiring
an LLM.  All vocabulary comes from ``config.py`` so the logic is fast,
reproducible, and runs on CPU.
"""

import logging
import re
from typing import Any

import config

logger = logging.getLogger(__name__)


class QueryDecomposer:
    """Decomposes a free-text query into structured fashion attributes.

    The decomposer walks the lowercased query string and:
    1. Finds every clothing item from the taxonomy.
    2. Binds the nearest preceding color adjective to each item.
    3. Detects environment and style keywords.
    4. Builds an *enhanced query* string that adds taxonomy synonyms for
       better vector-search recall.
    """

    def __init__(self) -> None:
        # Pre-sort multi-word items longest-first so that "dress shirt"
        # is matched before "shirt" alone.
        self._clothing_items: list[str] = sorted(
            config.ITEM_TO_CATEGORY.keys(), key=len, reverse=True
        )
        self._color_aliases: list[str] = sorted(
            config.ALIAS_TO_COLOR.keys(), key=len, reverse=True
        )
        self._environment_keywords: dict[str, list[str]] = config.ENVIRONMENT_KEYWORDS
        self._style_keywords: dict[str, list[str]] = config.STYLE_KEYWORDS
        self._clothing_taxonomy: dict[str, list[str]] = config.CLOTHING_TAXONOMY
        self._item_to_category: dict[str, str] = config.ITEM_TO_CATEGORY

        # Synonym expansions used to pad the enhanced query.
        self._style_synonyms: dict[str, str] = {
            "formal": "professional business elegant sophisticated",
            "casual": "relaxed comfortable everyday laid-back",
            "sporty": "athletic active fitness workout",
            "streetwear": "urban trendy edgy street style",
            "bohemian": "boho hippie artistic eclectic vintage",
            "minimalist": "simple clean understated sleek",
        }

        logger.info("QueryDecomposer initialised (CPU-only, deterministic).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, query: str) -> dict[str, Any]:
        """Decompose *query* into structured fashion constraints.

        Parameters
        ----------
        query:
            Free-form natural-language retrieval query.

        Returns
        -------
        dict with keys:
            clothing_items  – list of ``{'type': str, 'color': str | None}``
            colors          – list of canonical colour names found
            clothing_types  – list of clothing-type strings found
            environment     – detected environment label or ``None``
            style           – detected style label or ``None``
            enhanced_query  – the original query expanded with synonyms
        """
        lowered = query.lower().strip()

        clothing_items = self._extract_clothing_items_with_colors(lowered)
        colors = list(dict.fromkeys(
            item["color"] for item in clothing_items if item["color"] is not None
        ))
        clothing_types = list(dict.fromkeys(
            item["type"] for item in clothing_items
        ))
        environment = self._detect_environment(lowered)
        style = self._detect_style(lowered, clothing_types)
        enhanced_query = self._build_enhanced_query(
            query, clothing_items, colors, clothing_types, environment, style
        )

        result: dict[str, Any] = {
            "clothing_items": clothing_items,
            "colors": colors,
            "clothing_types": clothing_types,
            "environment": environment,
            "style": style,
            "enhanced_query": enhanced_query,
        }
        logger.debug("Decomposed query: %s", result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_clothing_items_with_colors(
        self, text: str
    ) -> list[dict[str, str | None]]:
        """Find clothing items in *text* and bind the nearest preceding colour.

        For every clothing item matched in order of appearance, the method
        scans *backwards* through the text preceding the match to locate the
        closest colour alias.  Multi-word items and colours are handled by
        matching longest-first.

        Returns a list of ``{'type': <item>, 'color': <canonical>|None}``.
        """
        found: list[dict[str, str | None]] = []
        # Track already-consumed spans to avoid overlapping matches.
        consumed_spans: list[tuple[int, int]] = []

        # Collect (start, end, item) tuples for all items, longest first.
        item_matches: list[tuple[int, int, str]] = []
        for item in self._clothing_items:
            pattern = re.compile(r"\b" + re.escape(item) + r"\b")
            for m in pattern.finditer(text):
                start, end = m.start(), m.end()
                # Skip if this span overlaps with an already-accepted item.
                if any(
                    not (end <= cs or start >= ce)
                    for cs, ce in consumed_spans
                ):
                    continue
                item_matches.append((start, end, item))
                consumed_spans.append((start, end))

        # Sort matches by position in the query (left-to-right).
        item_matches.sort(key=lambda t: t[0])

        for item_start, _item_end, item_name in item_matches:
            preceding_text = text[:item_start]
            color = self._find_nearest_color(preceding_text)
            found.append({"type": item_name, "color": color})

        return found

    def _find_nearest_color(self, text: str) -> str | None:
        """Return the canonical colour of the colour alias nearest to the
        *end* of *text*, or ``None`` if no colour is found."""
        best_color: str | None = None
        best_pos: int = -1

        for alias in self._color_aliases:
            pattern = re.compile(r"\b" + re.escape(alias) + r"\b")
            for m in pattern.finditer(text):
                if m.end() > best_pos:
                    best_pos = m.end()
                    best_color = config.ALIAS_TO_COLOR[alias]
        return best_color

    def _detect_environment(self, text: str) -> str | None:
        """Return the environment label with the most keyword hits, or ``None``."""
        best_env: str | None = None
        best_score: int = 0

        for env, keywords in self._environment_keywords.items():
            score = sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", text))
            if score > best_score:
                best_score = score
                best_env = env
        return best_env

    def _detect_style(
        self, text: str, clothing_types: list[str]
    ) -> str | None:
        """Detect style from explicit keywords first, then infer from
        clothing categories.

        If a style keyword (e.g. *formal*, *casual*) appears directly in
        the query it takes precedence.  Otherwise the majority category
        of the detected clothing items is used.
        """
        # 1. Explicit keyword match — highest-scoring style wins.
        best_style: str | None = None
        best_score: int = 0
        for style, keywords in self._style_keywords.items():
            score = sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", text))
            if score > best_score:
                best_score = score
                best_style = style

        if best_style is not None:
            return best_style

        # 2. Infer from detected clothing categories.
        if not clothing_types:
            return None

        category_counts: dict[str, int] = {}
        for ctype in clothing_types:
            cat = self._item_to_category.get(ctype)
            if cat is not None:
                category_counts[cat] = category_counts.get(cat, 0) + 1

        if not category_counts:
            return None

        majority_category = max(category_counts, key=lambda k: category_counts[k])
        # Map taxonomy category → style label (they share names for most).
        if majority_category in self._style_keywords:
            return majority_category
        return None

    def _build_enhanced_query(
        self,
        original_query: str,
        clothing_items: list[dict[str, str | None]],
        colors: list[str],
        clothing_types: list[str],
        environment: str | None,
        style: str | None,
    ) -> str:
        """Build an expanded query string for better vector-search recall.

        Appends detected attribute tokens and their synonyms to the original
        query so the text encoder can surface more relevant candidates.
        """
        parts: list[str] = [original_query.strip()]

        # Add colour tokens.
        for color in colors:
            if color.lower() not in original_query.lower():
                parts.append(color)

        # Add clothing type tokens.
        for ctype in clothing_types:
            if ctype.lower() not in original_query.lower():
                parts.append(ctype)

        # Add environment + related keywords (up to 3).
        if environment is not None:
            if environment.lower() not in original_query.lower():
                parts.append(environment)
            env_kws = self._environment_keywords.get(environment, [])
            for kw in env_kws[:3]:
                if kw.lower() not in original_query.lower():
                    parts.append(kw)

        # Add style synonyms.
        if style is not None:
            synonyms = self._style_synonyms.get(style, "")
            for token in synonyms.split():
                if token.lower() not in original_query.lower():
                    parts.append(token)

        # Also expand clothing items into their parent category names.
        seen_categories: set[str] = set()
        for ctype in clothing_types:
            cat = self._item_to_category.get(ctype)
            if cat and cat not in seen_categories:
                seen_categories.add(cat)
                if cat.lower() not in original_query.lower():
                    parts.append(cat)

        enhanced = " ".join(parts)
        # Collapse multiple spaces.
        enhanced = re.sub(r"\s+", " ", enhanced).strip()
        return enhanced
