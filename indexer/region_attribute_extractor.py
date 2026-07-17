"""Visual upper/lower garment colour evidence for attribute-aware retrieval.

Fashionpedia boxes are used when their filenames match.  Every other image
falls back to a pretrained person detector, so colour-to-garment reasoning is
not silently disabled for a mixed or streamed dataset.  CLIP remains the
retrieval encoder; this module only produces transparent reranking metadata.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import config

_COLOR_RGB = {
    "red": (210, 55, 50), "blue": (55, 95, 200), "green": (55, 140, 75),
    "yellow": (225, 195, 45), "orange": (225, 125, 45), "purple": (135, 75, 165),
    "pink": (225, 120, 155), "black": (35, 35, 35), "white": (235, 235, 235),
    "gray": (135, 135, 135), "brown": (125, 85, 50),
}


class RegionAttributeExtractor:
    """Extract independently observed upper/lower garment colours."""

    def __init__(self, annotation_path: Path | None = None, device: str = config.DEVICE) -> None:
        self.device = device
        self._by_filename: dict[str, list[dict[str, Any]]] = {}
        self._detector = None
        path = annotation_path or (config.DATA_DIR / "annotations_val2020.json")
        if path.is_file():
            self._load_annotations(path)

    def _load_annotations(self, path: Path) -> None:
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
        """Return region colours from annotations or detected person geometry."""
        evidence = self._empty_evidence()
        annotations = self._by_filename.get(image_filename, [])
        regions = self._annotation_regions(annotations)
        if regions:
            self._populate(evidence, image, regions, confidence_multiplier=1.0)
            evidence["region_source"] = "fashionpedia_annotation"
            return evidence

        person_box, detector_confidence, source = self._person_box(image)
        regions = self._split_person_box(person_box)
        self._populate(evidence, image, regions, confidence_multiplier=detector_confidence)
        evidence["region_source"] = source
        return evidence

    @staticmethod
    def _empty_evidence() -> dict[str, Any]:
        return {
            "top_color": "unknown", "bottom_color": "unknown",
            "top_color_confidence": 0.0, "bottom_color_confidence": 0.0,
            "top_garments": [], "bottom_garments": [], "region_source": "none",
        }

    def _annotation_regions(self, annotations: list[dict[str, Any]]) -> dict[str, list[tuple[tuple[float, float, float, float], str]]]:
        regions: dict[str, list[tuple[tuple[float, float, float, float], str]]] = defaultdict(list)
        for annotation in annotations:
            category = annotation["category"]
            label = self._canonical_label(category.get("name", ""))
            if category.get("supercategory") == "upperbody":
                regions["top"].append((tuple(annotation["bbox"]), label))
            elif category.get("supercategory") == "lowerbody":
                regions["bottom"].append((tuple(annotation["bbox"]), label))
        return regions

    def _person_box(self, image: Image.Image) -> tuple[tuple[float, float, float, float], float, str]:
        """Return the strongest detected person box, with a safe fallback."""
        try:
            if self._detector is None:
                from torchvision.models.detection import (
                    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                    fasterrcnn_mobilenet_v3_large_320_fpn,
                )
                self._detector = fasterrcnn_mobilenet_v3_large_320_fpn(
                    weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
                ).to(self.device).eval()
            pixels = torch.from_numpy(np.asarray(image.convert("RGB"))).permute(2, 0, 1).float().div(255).to(self.device)
            with torch.no_grad():
                output = self._detector([pixels])[0]
            candidates = [
                (box.detach().cpu().tolist(), float(score))
                for box, label, score in zip(output["boxes"], output["labels"], output["scores"])
                if int(label) == 1 and float(score) >= 0.55
            ]
            if candidates:
                box, confidence = max(candidates, key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]))
                left, top, right, bottom = box
                return (left, top, right - left, bottom - top), confidence, "person_detector"
        except Exception:
            # Torchvision weights are optional in restricted/offline runtimes.
            pass

        # The fallback is intentionally down-weighted; it is better than
        # leaving all records without evidence, but never overrides a clear
        # annotation or detected person box.
        return (
            image.width * 0.20, image.height * 0.08,
            image.width * 0.60, image.height * 0.87,
        ), 0.42, "center_body_fallback"

    @staticmethod
    def _split_person_box(box: tuple[float, float, float, float]) -> dict[str, list[tuple[tuple[float, float, float, float], str]]]:
        x, y, width, height = box
        # Skip head/feet; these bands are deliberately conservative garment ROIs.
        return {
            "top": [((x + 0.08 * width, y + 0.20 * height, 0.84 * width, 0.35 * height), "top")],
            "bottom": [((x + 0.12 * width, y + 0.55 * height, 0.76 * width, 0.35 * height), "pants")],
        }

    def _populate(self, evidence: dict[str, Any], image: Image.Image, regions: dict[str, list[tuple[tuple[float, float, float, float], str]]], confidence_multiplier: float) -> None:
        for region in ("top", "bottom"):
            entries = regions.get(region, [])
            colours = [self._dominant_color(image, bbox) for bbox, _ in entries]
            colours = [colour for colour in colours if colour is not None]
            if colours:
                colour, confidence = max(colours, key=lambda item: item[1])
                evidence[f"{region}_color"] = colour
                evidence[f"{region}_color_confidence"] = round(confidence * confidence_multiplier, 4)
            evidence[f"{region}_garments"] = list(dict.fromkeys(label for _, label in entries))

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
        return labels[index], max(0.0, min(1.0, 1.0 - float(distances[index]) / 220.0))
