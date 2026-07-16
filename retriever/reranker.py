"""
Attribute-aware re-ranker for fashion retrieval.

After Stage-1 vector search, rescores candidates by comparing stored
metadata / captions against structured query constraints — including
*compositional* colour↔garment bindings (e.g. red tie + white shirt).
"""

from __future__ import annotations

import logging
import re
from typing import Any

import config

logger = logging.getLogger(__name__)


class AttributeReranker:
    """Post-retrieval re-ranker that verifies attribute constraints."""

    def __init__(self) -> None:
        self._alias_to_color: dict[str, str] = config.ALIAS_TO_COLOR
        self._vector_weight: float = config.VECTOR_SIM_WEIGHT
        self._attribute_weight: float = config.ATTRIBUTE_MATCH_WEIGHT
        logger.info(
            "AttributeReranker initialised  "
            "(vector_w=%.2f, attribute_w=%.2f).",
            self._vector_weight,
            self._attribute_weight,
        )

    def rerank(
        self,
        candidates: list[dict[str, Any]],
        decomposed_query: dict[str, Any],
        top_k: int = config.TOP_K_FINAL,
    ) -> list[dict[str, Any]]:
        """Blend vector similarity with attribute / compositional match."""
        scored: list[dict[str, Any]] = []

        for cand in candidates:
            vector_sim = 1.0 - float(cand["distance"])
            attr_score, matched = self._compute_attribute_score(
                cand.get("metadata", {}), decomposed_query
            )
            final = (
                self._vector_weight * vector_sim
                + self._attribute_weight * attr_score
            )
            scored.append(
                {
                    "image_id": cand["image_id"],
                    "image_path": cand.get("image_path", ""),
                    "final_score": round(final, 6),
                    "vector_similarity": round(vector_sim, 6),
                    "attribute_score": round(attr_score, 6),
                    "matched_attributes": matched,
                    "caption": cand.get("metadata", {}).get("caption", ""),
                }
            )

        scored.sort(key=lambda r: r["final_score"], reverse=True)
        logger.info(
            "Re-ranked %d candidates → returning top %d.", len(scored), top_k
        )
        return scored[:top_k]

    def _compute_attribute_score(
        self,
        metadata: dict[str, Any],
        decomposed: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        scores: list[float] = []
        matched: dict[str, Any] = {}
        caption = str(metadata.get("caption", "")).lower()

        # --- Compositional colour↔item bindings (highest priority) ---
        clothing_items = decomposed.get("clothing_items") or []
        bound_items = [it for it in clothing_items if it.get("color")]
        if bound_items:
            pair_hits: list[str] = []
            pair_scores: list[float] = []
            for item in bound_items:
                pair_score = self._score_color_item_pair(
                    caption, item["color"], item["type"], metadata
                )
                pair_scores.append(pair_score)
                if pair_score >= 0.5:
                    pair_hits.append(f"{item['color']} {item['type']}")
            comp_score = sum(pair_scores) / len(pair_scores)
            # Weight compositional matches more heavily by repeating them.
            scores.extend([comp_score, comp_score])
            matched["compositional"] = {
                "query": [
                    f"{it['color']} {it['type']}" for it in bound_items
                ],
                "hits": pair_hits,
                "score": round(comp_score, 4),
            }

        # --- Colour set match ---
        query_colors: list[str] = decomposed.get("colors", [])
        if query_colors:
            cand_colors_raw: str = metadata.get("colors", "")
            cand_canonical = self._canonicalize_colors(cand_colors_raw)
            # Also scan caption for colour aliases.
            for alias, canonical in self._alias_to_color.items():
                if re.search(r"\b" + re.escape(alias) + r"\b", caption):
                    cand_canonical.add(canonical)
            hits = [c for c in query_colors if c in cand_canonical]
            color_score = len(hits) / len(query_colors)
            scores.append(color_score)
            matched["colors"] = {
                "query": query_colors,
                "candidate": list(cand_canonical),
                "hits": hits,
                "score": round(color_score, 4),
            }

        # --- Clothing-type match ---
        query_types: list[str] = decomposed.get("clothing_types", [])
        if query_types:
            cand_types_raw: str = metadata.get("clothing_types", "")
            cand_types = {
                t.strip().lower()
                for t in cand_types_raw.split(",")
                if t.strip()
            }
            for t in query_types:
                if re.search(r"\b" + re.escape(t) + r"\b", caption):
                    cand_types.add(t)
            hits_t = [t for t in query_types if t in cand_types]
            # Fuzzy: "shirt" matches "dress shirt" / "t-shirt" in caption
            for t in query_types:
                if t not in hits_t and any(t in ct for ct in cand_types):
                    hits_t.append(t)
            type_score = len(hits_t) / len(query_types) if query_types else 0.0
            scores.append(type_score)
            matched["clothing_types"] = {
                "query": query_types,
                "candidate": list(cand_types),
                "hits": hits_t,
                "score": round(type_score, 4),
            }

        # --- Environment match ---
        query_env: str | None = decomposed.get("environment")
        if query_env:
            cand_env: str = str(metadata.get("environment", "")).strip().lower()
            env_score = 1.0 if cand_env == query_env else 0.0
            # Soft credit if caption mentions an env keyword.
            if env_score == 0.0:
                env_kws = config.ENVIRONMENT_KEYWORDS.get(query_env, [])
                if any(
                    re.search(r"\b" + re.escape(kw) + r"\b", caption)
                    for kw in env_kws
                ):
                    env_score = 0.6
            scores.append(env_score)
            matched["environment"] = {
                "query": query_env,
                "candidate": cand_env,
                "score": env_score,
            }

        # --- Style match ---
        query_style: str | None = decomposed.get("style")
        if query_style:
            cand_style: str = str(metadata.get("style", "")).strip().lower()
            style_score = 1.0 if cand_style == query_style else 0.0
            if style_score == 0.0:
                style_kws = config.STYLE_KEYWORDS.get(query_style, [])
                if any(
                    re.search(r"\b" + re.escape(kw) + r"\b", caption)
                    for kw in style_kws
                ):
                    style_score = 0.6
            scores.append(style_score)
            matched["style"] = {
                "query": query_style,
                "candidate": cand_style,
                "score": style_score,
            }

        if not scores:
            return 1.0, matched

        return sum(scores) / len(scores), matched

    def _score_color_item_pair(
        self,
        caption: str,
        color: str,
        item_type: str,
        metadata: dict[str, Any],
    ) -> float:
        """Score whether *color* is bound to *item_type* in the caption.

        Prefers proximity ("red tie") over bag-of-words co-occurrence so
        "red shirt + blue pants" is distinguished from the reverse.
        """
        color_aliases = [
            a
            for a, c in self._alias_to_color.items()
            if c == color
        ] or [color]

        # Strong: colour adjective adjacent / near the garment word.
        for alias in color_aliases:
            near = re.search(
                rf"\b{re.escape(alias)}\b(?:\s+\w+){{0,2}}\s+\b{re.escape(item_type)}\b",
                caption,
            )
            if near:
                return 1.0
            near_rev = re.search(
                rf"\b{re.escape(item_type)}\b(?:\s+\w+){{0,2}}\s+\b{re.escape(alias)}\b",
                caption,
            )
            if near_rev:
                return 0.9

        color_in_caption = any(
            re.search(r"\b" + re.escape(a) + r"\b", caption)
            for a in color_aliases
        )
        item_in_caption = bool(
            re.search(r"\b" + re.escape(item_type) + r"\b", caption)
        )

        # Metadata fallback (bags of attributes, weaker signal).
        cand_colors = self._canonicalize_colors(str(metadata.get("colors", "")))
        cand_types = {
            t.strip().lower()
            for t in str(metadata.get("clothing_types", "")).split(",")
            if t.strip()
        }
        color_in_meta = color in cand_colors
        item_in_meta = item_type in cand_types or any(
            item_type in t for t in cand_types
        )

        if (color_in_caption or color_in_meta) and (
            item_in_caption or item_in_meta
        ):
            return 0.55
        if item_in_caption or item_in_meta:
            return 0.25
        if color_in_caption or color_in_meta:
            return 0.15
        return 0.0

    def _canonicalize_colors(self, raw: str) -> set[str]:
        canonical: set[str] = set()
        for token in raw.split(","):
            token = token.strip().lower()
            if not token or token == "unknown":
                continue
            mapped = self._alias_to_color.get(token, token)
            canonical.add(mapped)
        return canonical
