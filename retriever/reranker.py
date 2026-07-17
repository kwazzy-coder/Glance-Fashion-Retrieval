"""
Attribute-aware re-ranker for fashion retrieval.

After Stage-1 vector search, the re-ranker combines vector similarity with
attribute evidence from captions / metadata so that compositional queries
such as "red shirt with blue pants" are handled more reliably than a
vanilla embedding-only system.
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
            "AttributeReranker initialised (vector_w=%.2f, attribute_w=%.2f).",
            self._vector_weight,
            self._attribute_weight,
        )

    def rerank(
        self,
        candidates: list[dict[str, Any]],
        decomposed_query: dict[str, Any],
        top_k: int = config.TOP_K_FINAL,
    ) -> list[dict[str, Any]]:
        """Blend vector similarity with attribute and compositional match."""
        scored: list[dict[str, Any]] = []

        for cand in candidates:
            # Chroma's cosine distance is ``1 - cosine_similarity`` and can
            # lie in [0, 2].  Convert it to a bounded relevance score before
            # blending it with metadata evidence; otherwise a negative cosine
            # similarity can dominate an otherwise good attribute match.
            vector_sim = self._normalise_cosine_distance(float(cand["distance"]))
            metadata = cand.get("metadata", {})
            attr_score, matched = self._compute_attribute_score(metadata, decomposed_query)
            final = self._vector_weight * vector_sim + self._attribute_weight * attr_score
            scored.append(
                {
                    "image_id": cand["image_id"],
                    "image_path": cand.get("image_path", ""),
                    "final_score": round(final, 6),
                    "vector_similarity": round(vector_sim, 6),
                    "attribute_score": round(attr_score, 6),
                    "matched_attributes": matched,
                    "caption": metadata.get("caption", ""),
                }
            )

        scored.sort(key=lambda item: item["final_score"], reverse=True)
        logger.info("Re-ranked %d candidates → returning top %d.", len(scored), top_k)
        return scored[:top_k]

    @staticmethod
    def _normalise_cosine_distance(distance: float) -> float:
        """Map Chroma cosine distance from [0, 2] to a stable [0, 1] score."""
        return max(0.0, min(1.0, 1.0 - (distance / 2.0)))

    def _compute_attribute_score(
        self,
        metadata: dict[str, Any],
        decomposed: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        """Compute a weighted attribute score from metadata and caption cues."""
        matched: dict[str, Any] = {}
        caption = str(metadata.get("caption", "")).lower()

        query_colors: list[str] = decomposed.get("colors", [])
        query_types: list[str] = decomposed.get("clothing_types", [])
        query_accessories: list[str] = decomposed.get("accessories", [])
        query_environment: str | None = decomposed.get("environment")
        query_style: str | None = decomposed.get("style")
        clothing_items = decomposed.get("clothing_items") or []

        cand_colors = self._canonicalize_colors(str(metadata.get("colors", "")))
        cand_types = {
            token.strip().lower()
            for token in str(metadata.get("clothing_types", "")).split(",")
            if token.strip()
        }
        cand_accessories = {
            token.strip().lower()
            for token in str(metadata.get("accessories", "")).split(",")
            if token.strip()
        }
        cand_environment = str(metadata.get("environment", "")).strip().lower()
        cand_style = str(metadata.get("style", "")).strip().lower()

        weighted_score = 0.0
        total_weight = 0.0

        comp_score = 0.0
        if clothing_items:
            pair_scores: list[float] = []
            pair_hits: list[str] = []
            pair_conflicts: list[str] = []
            caption_confidence = max(0.15, min(1.0, float(metadata.get("caption_confidence", 0.5))))
            for item in clothing_items:
                if item.get("color"):
                    caption_pair_score = self._score_color_item_pair(
                        caption, item["color"], item["type"], metadata
                    )
                    region_score, has_region_evidence, conflict = self._score_region_pair(
                        metadata, item["color"], item["type"]
                    )
                    # Annotation-grounded region evidence is preferred. A
                    # contradictory top/bottom colour receives a strong
                    # penalty instead of being rescued by caption co-occurrence.
                    if has_region_evidence:
                        pair_score = 0.80 * region_score + 0.20 * caption_pair_score
                    else:
                        pair_score = 0.50 * caption_confidence * caption_pair_score
                    if conflict:
                        pair_score *= 0.10
                        pair_conflicts.append(f"{item['color']} {item['type']}")
                    pair_scores.append(pair_score)
                    if pair_score >= 0.5:
                        pair_hits.append(f"{item['color']} {item['type']}")
            if pair_scores:
                comp_score = sum(pair_scores) / len(pair_scores)
                weighted_score += 0.55 * comp_score
                total_weight += 0.55
                matched["compositional"] = {
                    "query": [f"{item['color']} {item['type']}" for item in clothing_items if item.get("color")],
                    "hits": pair_hits,
                    "conflicts": pair_conflicts,
                    "score": round(comp_score, 4),
                }

        if query_colors:
            color_hits = [color for color in query_colors if color in cand_colors]
            color_score = len(color_hits) / len(query_colors) if query_colors else 0.0
            weighted_score += 0.13 * color_score
            total_weight += 0.13
            matched["colors"] = {
                "query": query_colors,
                "candidate": sorted(cand_colors),
                "hits": color_hits,
                "score": round(color_score, 4),
            }

        if query_types:
            type_hits: list[str] = []
            for item_type in query_types:
                variants = self._garment_variants(item_type)
                if any(
                    variant in cand_types
                    or any(variant in token or token in variant for token in cand_types)
                    for variant in variants
                ):
                    type_hits.append(item_type)
            type_score = len(type_hits) / len(query_types) if query_types else 0.0
            weighted_score += 0.18 * type_score
            total_weight += 0.18
            matched["clothing_types"] = {
                "query": query_types,
                "candidate": sorted(cand_types),
                "hits": type_hits,
                "score": round(type_score, 4),
            }

        if query_accessories:
            accessory_hits = [accessory for accessory in query_accessories if accessory in cand_accessories]
            accessory_score = len(accessory_hits) / len(query_accessories) if query_accessories else 0.0
            weighted_score += 0.05 * accessory_score
            total_weight += 0.05
            matched["accessories"] = {
                "query": query_accessories,
                "candidate": sorted(cand_accessories),
                "hits": accessory_hits,
                "score": round(accessory_score, 4),
            }

        if query_environment:
            env_score = 1.0 if cand_environment == query_environment else 0.0
            compatible = config.ENVIRONMENT_COMPATIBILITY.get(query_environment, set())
            if cand_environment in compatible:
                env_score = 1.0
            if env_score == 0.0:
                env_keywords = config.ENVIRONMENT_KEYWORDS.get(query_environment, [])
                if any(re.search(r"\b" + re.escape(keyword) + r"\b", caption) for keyword in env_keywords):
                    env_score = 0.6
            weighted_score += 0.05 * env_score
            total_weight += 0.05
            matched["environment"] = {
                "query": query_environment,
                "candidate": cand_environment,
                "score": round(env_score, 4),
            }

        if query_style:
            style_score = 1.0 if cand_style == query_style else 0.0
            if style_score == 0.0:
                style_keywords = config.STYLE_KEYWORDS.get(query_style, [])
                if any(re.search(r"\b" + re.escape(keyword) + r"\b", caption) for keyword in style_keywords):
                    style_score = 0.6
            weighted_score += 0.04 * style_score
            total_weight += 0.04
            matched["style"] = {
                "query": query_style,
                "candidate": cand_style,
                "score": round(style_score, 4),
            }

        if total_weight <= 0:
            return 1.0, matched
        return round(weighted_score / total_weight, 6), matched

    def _score_region_pair(self, metadata: dict[str, Any], color: str, item_type: str) -> tuple[float, bool, bool]:
        """Score an annotation-grounded top/bottom colour constraint.

        Returns ``(score, has_evidence, conflict)``.  A known, different
        colour in the requested garment region is an explicit conflict.
        """
        region = "bottom" if item_type.lower() in {"pants", "trousers", "slacks", "jeans", "skirt", "shorts", "leggings"} else "top"
        observed = str(metadata.get(f"{region}_color", "unknown")).lower()
        try:
            confidence = max(0.0, min(1.0, float(metadata.get(f"{region}_color_confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        if observed in {"", "unknown"} or confidence < 0.35:
            return 0.0, False, False
        if observed == color:
            return confidence, True, False
        return 0.0, True, True

    def _score_color_item_pair(
        self,
        caption: str,
        color: str,
        item_type: str,
        metadata: dict[str, Any],
    ) -> float:
        """Score whether color and garment are bound together in the caption."""
        color_aliases = [alias for alias, canonical in self._alias_to_color.items() if canonical == color] or [color]
        item_variants = self._garment_variants(item_type)

        for alias in color_aliases:
            for item_variant in item_variants:
                near = re.search(
                    rf"\b{re.escape(alias)}\b(?:\s+\w+){{0,2}}\s+\b{re.escape(item_variant)}\b",
                    caption,
                )
                if near:
                    return 1.0
                near_rev = re.search(
                    rf"\b{re.escape(item_variant)}\b(?:\s+\w+){{0,2}}\s+\b{re.escape(alias)}\b",
                    caption,
                )
                if near_rev:
                    return 0.9

        color_in_caption = any(re.search(r"\b" + re.escape(alias) + r"\b", caption) for alias in color_aliases)
        item_in_caption = any(
            re.search(r"\b" + re.escape(variant) + r"\b", caption)
            for variant in item_variants
        )
        cand_colors = self._canonicalize_colors(str(metadata.get("colors", "")))
        cand_types = {
            token.strip().lower()
            for token in str(metadata.get("clothing_types", "")).split(",")
            if token.strip()
        }

        color_in_meta = color in cand_colors
        item_in_meta = any(
            variant in cand_types
            or any(variant in token or token in variant for token in cand_types)
            for variant in item_variants
        )

        if (color_in_caption or color_in_meta) and (item_in_caption or item_in_meta):
            return 0.55
        if item_in_caption or item_in_meta:
            return 0.25
        if color_in_caption or color_in_meta:
            return 0.15
        return 0.0

    @staticmethod
    def _garment_variants(item_type: str) -> set[str]:
        """Return canonical garment spelling plus configured equivalents."""
        normalized = item_type.lower().strip()
        return set(config.GARMENT_EQUIVALENTS.get(normalized, {normalized}))

    def _canonicalize_colors(self, raw: str) -> set[str]:
        canonical: set[str] = set()
        for token in raw.split(","):
            token = token.strip().lower()
            if not token or token == "unknown":
                continue
            mapped = self._alias_to_color.get(token, token)
            canonical.add(mapped)
        return canonical
