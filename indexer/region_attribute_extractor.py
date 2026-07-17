"""Region-aware fashion evidence extracted from Fashionpedia annotations.

The image encoder remains CLIP.  This module supplies independent visual
metadata for reranking: garment regions and their dominant colours.  When the
Fashionpedia annotation file is unavailable, it returns no evidence instead
of fabricating a low-confidence caption-based attribute.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import config

_COLOR_RGB = {
    "red": (210, 55, 50), "blue": (55, 95, 200), "green": (55, 140, 75),
    "yellow": (225, 195, 45), "orange": (225, 125, 45), "purple": (135, 75, 165),
    "pink": (225, 120, 155), "black": (35, 35, 35), "white": (235, 235, 235),
    "gray": (135, 135, 135), "brown": (125, 85, 50),
}


class RegionAttributeExtractor:
    """Provides annotation-grounded top/bottom garment colour evidence."""

    def __init__(self, annotation_path: Path | None = None) -> None:
        self._by_filename: dict[str, list[dict[str, Any]]] = {}
        path = annotation_path or (config.DATA_DIR / "annotations_val2020.json")
        if path.is_file():
            self._load(path)

    def _load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        filename_by_id = {image["id"]: image["file_name"] for image in data["images"]}
        categories = {category["id"]: category for category in data["categories"]}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for annotation in data["annotations"]:
            filename = filename_by_id.get(annotation["image_id"])
            category = categories.get(annotation["category_id"])
            if filename and category and annotation.get("bbox"):
                grouped[filename].append({"bbox": annotation["bbox"], "category": category})
        self._by_filename = dict(grouped)

    def extract(self, image: Image.Image, image_filename: str) -> dict[str, Any]:
        """Return high-confidence region evidence, or explicit empty fields."""
        annotations = self._by_filename.get(image_filename, [])
        evidence: dict[str, Any] = {
            "top_color": "unknown", "bottom_color": "unknown",
            "top_color_confidence": 0.0, "bottom_color_confidence": 0.0,
            "top_garments": [], "bottom_garments": [], "region_source": "none",
        }
        if not annotations:
            return evidence

        regions: dict[str, list[tuple[tuple[float, float, float, float], str]]] = defaultdict(list)
        for annotation in annotations:
            category = annotation["category"]
            supercategory = category.get("supercategory", "")
            label = self._canonical_label(category.get("name", ""))
            if supercategory == "upperbody":
                regions["top"].append((tuple(annotation["bbox"]), label))
            elif supercategory == "lowerbody":
                regions["bottom"].append((tuple(annotation["bbox"]), label))

        for region, key in (("top", "top"), ("bottom", "bottom")):
            entries = regions.get(region, [])
            if not entries:
                continue
            colors = [self._dominant_color(image, bbox) for bbox, _ in entries]
            colors = [item for item in colors if item is not None]
            if colors:
                color, confidence = max(colors, key=lambda value: value[1])
                evidence[f"{key}_color"] = color
                evidence[f"{key}_color_confidence"] = round(confidence, 4)
            evidence[f"{key}_garments"] = list(dict.fromkeys(label for _, label in entries))

        evidence["region_source"] = "fashionpedia_annotation"
        return evidence

    @staticmethod
    def _canonical_label(label: str) -> str:
        lowered = label.lower()
        if "pants" in lowered:
            return "pants"
        if "shirt" in lowered or "blouse" in lowered:
            return "shirt"
        if "t-shirt" in lowered or "top" in lowered:
            return "top"
        return lowered.split(",")[0].strip()

    @staticmethod
    def _dominant_color(image: Image.Image, bbox: tuple[float, float, float, float]) -> tuple[str, float] | None:
        x, y, width, height = bbox
        left, top = max(0, int(x)), max(0, int(y))
        right, bottom = min(image.width, int(x + width)), min(image.height, int(y + height))
        if right - left < 4 or bottom - top < 4:
            return None
        pixels = np.asarray(image.convert("RGB").crop((left, top, right, bottom)).resize((32, 32)), dtype=np.float32)
        median = np.median(pixels.reshape(-1, 3), axis=0)
        labels = list(_COLOR_RGB)
        palette = np.asarray([_COLOR_RGB[label] for label in labels], dtype=np.float32)
        distances = np.linalg.norm(palette - median, axis=1)
        index = int(np.argmin(distances))
        # Confidence is intentionally conservative: colour is an auxiliary
        # visual signal, not an asserted fact from a synthetic caption.
        return labels[index], max(0.0, min(1.0, 1.0 - float(distances[index]) / 220.0))
