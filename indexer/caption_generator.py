"""
Fashion-aware captioning via FashionSigLIP zero-shot attribute probing.

Instead of downloading a separate captioner (BLIP ~2 GB / BLIP-2 ~15 GB),
we score each image against a compact bank of colour / garment / scene /
style prompts using the *same* FashionSigLIP encoder used for retrieval.
Top-scoring attributes are assembled into a natural-language caption that
feeds attribute extraction and fused embeddings.

This is faster to download, lighter on VRAM, and more fashion-specific
than generic captioning.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

import config

logger = logging.getLogger(__name__)


def _build_prompt_bank() -> List[Tuple[str, str, str]]:
    """Return (label, category, prompt) triples for zero-shot scoring."""
    bank: List[Tuple[str, str, str]] = []

    for color in config.COLOR_VOCABULARY:
        bank.append(
            (color, "color", f"a person wearing {color} clothing")
        )

    # Representative garments from each taxonomy bucket (keep bank small).
    garments = [
        "blazer", "suit", "dress shirt", "tie", "blouse", "shirt",
        "t-shirt", "jeans", "hoodie", "sweater", "sneakers", "shorts",
        "jacket", "coat", "raincoat", "trench coat", "leather jacket",
        "dress", "skirt", "trousers", "polo", "cardigan", "vest",
        "hat", "scarf", "handbag", "sunglasses", "boots",
    ]
    for g in garments:
        bank.append((g, "garment", f"a person wearing a {g}"))

    for env, kws in config.ENVIRONMENT_KEYWORDS.items():
        hint = kws[0] if kws else env
        bank.append((env, "environment", f"a person in a {hint} setting"))

    for style in config.STYLE_KEYWORDS:
        bank.append((style, "style", f"a person in {style} attire"))

    # ── Compositional (color, garment) pair prompts ────────────────────
    # Scoring "colors" and "garments" independently loses the binding
    # between them: two independent top-K lists can't tell you the tie is
    # red *and* the shirt is white rather than the reverse. These pair
    # prompts let us ask the encoder directly "is this garment red or
    # white?" for each detected garment, so the caption can bind a
    # specific color to a specific garment instead of listing both pools
    # side by side. Adds len(colors) * len(garments) prompts (~300), all
    # encoded once at startup — negligible cost, no per-image overhead.
    for color in config.COLOR_VOCABULARY:
        for g in garments:
            bank.append((f"{color}|{g}", "pair", f"a person wearing a {color} {g}"))

    return bank


class CaptionGenerator:
    """Builds fashion captions from FashionSigLIP zero-shot scores.

    Parameters
    ----------
    embedding_generator:
        Shared ``EmbeddingGenerator`` instance (preferred). If omitted,
        one is created (loads FashionSigLIP once).
    """

    def __init__(
        self,
        device: str = config.DEVICE,
        dtype: torch.dtype = config.DTYPE,  # unused; kept for API compat
        embedding_generator=None,
        top_k_colors: int = 3,
        top_k_garments: int = 4,
    ) -> None:
        self.device = device
        self.top_k_colors = top_k_colors
        self.top_k_garments = top_k_garments

        if embedding_generator is None:
            from indexer.embedding_generator import EmbeddingGenerator

            logger.info(
                "CaptionGenerator: using shared open_clip encoder for zero-shot captions "
                "(no BLIP / FashionSigLIP download)."
            )
            embedding_generator = EmbeddingGenerator(device=self.device)

        self.encoder = embedding_generator
        self._prompt_bank = _build_prompt_bank()
        self._prompt_texts = [p for _, _, p in self._prompt_bank]
        self._text_feats: Optional[np.ndarray] = None

        logger.info(
            "Zero-shot caption bank ready (%d prompts). Caching text embeds…",
            len(self._prompt_bank),
        )
        self._text_feats = self._encode_prompts(self._prompt_texts)
        logger.info("CaptionGenerator ready (open_clip zero-shot).")

    def _encode_prompts(self, texts: Sequence[str]) -> np.ndarray:
        feats: List[np.ndarray] = []
        # Encode in chunks to avoid huge tokenizer batches.
        chunk = 32
        for i in range(0, len(texts), chunk):
            batch = list(texts[i : i + chunk])
            tokens = self.encoder._tokenize(batch)
            with torch.no_grad():
                out = self.encoder.model.encode_text(tokens)
                out = out / out.norm(dim=-1, keepdim=True)
            feats.append(out.cpu().numpy().astype(np.float32))
        return np.concatenate(feats, axis=0)

    def _score_image(self, image: Image.Image) -> np.ndarray:
        img_feat = self.encoder.encode_image(image)  # (512,)
        # Cosine sim against all prompt embeds.
        return self._text_feats @ img_feat  # type: ignore[operator]

    def _select(self, scores: np.ndarray) -> dict:
        """Pick top attributes per category from prompt scores."""
        by_cat: dict[str, List[Tuple[str, float]]] = {
            "color": [],
            "garment": [],
            "environment": [],
            "style": [],
            "pair": [],
        }
        for idx, (label, cat, _) in enumerate(self._prompt_bank):
            by_cat[cat].append((label, float(scores[idx])))

        def top(items: List[Tuple[str, float]], k: int, floor: float) -> List[str]:
            items = sorted(items, key=lambda t: t[1], reverse=True)
            return [lab for lab, sc in items[:k] if sc >= floor]

        # Floors are soft — FashionSigLIP sims are typically 0.1–0.35.
        colors = top(by_cat["color"], self.top_k_colors, floor=0.0)
        garments = top(by_cat["garment"], self.top_k_garments, floor=0.0)
        env = top(by_cat["environment"], 1, floor=0.0)
        style = top(by_cat["style"], 1, floor=0.0)

        # Bind each detected garment to its own best-scoring color using the
        # (color, garment) pair prompts — e.g. for garments ["tie", "shirt"],
        # ask "red tie vs white tie vs ... " and "red shirt vs white shirt vs
        # ..." independently, instead of assuming a shared color pool.
        pair_scores = {label: sc for label, sc in by_cat["pair"]}
        bound_pairs: List[Tuple[str, str]] = []
        for g in garments:
            best_color, best_score = None, float("-inf")
            for color in config.COLOR_VOCABULARY:
                sc = pair_scores.get(f"{color}|{g}")
                if sc is not None and sc > best_score:
                    best_color, best_score = color, sc
            if best_color is not None:
                bound_pairs.append((best_color, g))

        return {
            "colors": colors,
            "garments": garments,
            "bound_pairs": bound_pairs,
            "environment": env[0] if env else "unknown",
            "style": style[0] if style else "unknown",
        }

    @staticmethod
    def _attrs_to_caption(attrs: dict) -> str:
        bound_pairs = attrs.get("bound_pairs") or []
        env = attrs.get("environment", "unknown")
        style = attrs.get("style", "unknown")

        if bound_pairs:
            # Compositional phrasing — each garment keeps its own bound
            # color, e.g. "a red tie and a white shirt", not "red and
            # white tie, shirt" (which can't tell which garment is which
            # color and is exactly the compositionality bug we fixed).
            garment_str = " and ".join(
                f"a {color} {garment}" for color, garment in bound_pairs[:3]
            )
        else:
            colors = attrs.get("colors") or []
            garments = attrs.get("garments") or []
            color_str = " and ".join(colors[:2]) if colors else "neutral"
            garment_str = ", ".join(garments[:3]) if garments else "clothing"
            garment_str = f"{color_str} {garment_str}"

        parts = [
            f"a person wearing {garment_str}",
            f"in a {style} style",
        ]
        if env and env != "unknown":
            env_hint = {
                "office": "modern office",
                "urban": "city street",
                "park": "park with benches and trees",
                "home": "home interior",
                "beach": "beach",
                "formal_venue": "formal setting",
                "gym": "gym",
            }.get(env, env)
            parts.append(f"in a {env_hint}")
        return " ".join(parts)

    def generate_caption(self, image: Image.Image) -> str:
        """Return a zero-shot fashion+context caption for *image*."""
        scores = self._score_image(image)
        attrs = self._select(scores)
        caption = self._attrs_to_caption(attrs)
        logger.debug("Zero-shot caption: %s", caption)
        return caption

    def generate_captions_batch(
        self,
        images: List[Image.Image],
        batch_size: int = config.BATCH_SIZE,
    ) -> List[str]:
        """Caption a list of images (batched image encoding)."""
        captions: List[str] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            img_feats = self.encoder.encode_images_batch(batch, batch_size=len(batch))
            for feat in img_feats:
                scores = self._text_feats @ feat  # type: ignore[operator]
                captions.append(self._attrs_to_caption(self._select(scores)))
        return captions