"""
Embedding generator: OpenAI CLIP (local .pt) or open_clip (Hub tags).

Local ViT-B-32.pt files from Azure CDN are TorchScript archives. PyTorch 2.6+
and open_clip's ``torch.load(weights_only=True)`` cannot load them, so we use
the official ``clip.load()`` path for ``*.pt`` checkpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import open_clip
import torch
from PIL import Image

import config

logger = logging.getLogger(__name__)


def _is_local_openai_checkpoint(path: Optional[str]) -> bool:
    return bool(path and str(path).lower().endswith(".pt") and Path(path).is_file())


class EmbeddingGenerator:
    """Creates normalised image/text embeddings."""

    def __init__(self, device: str = config.DEVICE, text_only: bool = False) -> None:
        self.device = device
        self.text_only = text_only
        self._use_openai_clip = False

        model_name = config.OPEN_CLIP_MODEL
        pretrained: Optional[str] = (
            getattr(config, "CLIP_CHECKPOINT_PATH", None)
            or config.resolve_clip_checkpoint()
            or config.OPEN_CLIP_PRETRAINED
        )

        if _is_local_openai_checkpoint(pretrained):
            logger.info(
                "Loading OpenAI CLIP from local checkpoint '%s' on %s…",
                pretrained,
                self.device,
            )
            self._load_openai_clip(str(pretrained))
        elif model_name.startswith("hf-hub:"):
            logger.info(
                "Loading open_clip model '%s' on %s…",
                model_name,
                self.device,
            )
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("open_clip model loaded from Hub.")
        else:
            logger.info(
                "Loading open_clip model '%s' (pretrained=%s) on %s…",
                model_name,
                pretrained,
                self.device,
            )
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("open_clip model loaded.")

    def _load_openai_clip(self, checkpoint_path: str) -> None:
        try:
            import clip
        except ImportError as exc:
            raise ImportError(
                "Local ViT-B-32.pt requires the OpenAI CLIP package. "
                "Install with: pip install ftfy regex && "
                "pip install git+https://github.com/openai/CLIP.git"
            ) from exc

        # OpenAI CLIP loader handles TorchScript format correctly,
        # avoiding PyTorch 2.6's weights_only=True incompatibility
        self.model, self.preprocess = clip.load(
            checkpoint_path,
            device=self.device,
            jit=False,  # PyTorch handles TorchScript automatically
        )
        self.model.eval()
        self._use_openai_clip = True
        self.tokenizer = clip.tokenize

        # Guard against a well-known Colab footgun: PyPI hosts an
        # unrelated package literally named `clip` (a CLI tool). In a
        # long-running kernel with repeated `pip install`/`%%writefile`
        # edits, an earlier `import clip` anywhere in the process can
        # leave the WRONG module cached in `sys.modules['clip']` — every
        # later `import clip` (even after correctly installing OpenAI's
        # package) then silently returns that same broken module.
        # `clip.load(...)` can still appear to "work" in that case, but
        # `clip.tokenize` ends up missing or None, which otherwise fails
        # much later and far from the real cause. Fail here instead, with
        # a fix the person can act on immediately.
        if not callable(self.tokenizer):
            raise RuntimeError(
                f"clip.tokenize is not callable (got {self.tokenizer!r}). "
                "This usually means the wrong 'clip' package is cached in "
                "this session — PyPI has an unrelated package also named "
                "'clip'. Fix: Runtime -> Restart session in Colab, then "
                "re-run all cells from the top (a code fix alone cannot "
                "un-cache an already-imported module in this process)."
            )
        logger.info("OpenAI CLIP loaded from %s", checkpoint_path)

    def _tokenize(self, texts: List[str]) -> torch.Tensor:
        if self._use_openai_clip:
            import clip

            return clip.tokenize(texts, truncate=True).to(self.device)
        return self.tokenizer(texts).to(self.device)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        pixel_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(pixel_tensor)
            features = features.float()
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        tokens = self._tokenize([text])
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features.float()
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_texts_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode many strings (used by zero-shot caption prompt bank)."""
        all_embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = self._tokenize(batch)
            with torch.no_grad():
                features = self.model.encode_text(tokens)
                features = features.float()
                features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy().astype(np.float32))
        return np.concatenate(all_embeddings, axis=0)

    def encode_images_batch(
        self,
        images: List[Image.Image],
        batch_size: int = config.BATCH_SIZE,
    ) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            pixel_tensors = torch.stack(
                [self.preprocess(img) for img in batch]
            ).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(pixel_tensors)
                features = features.float()
                features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy().astype(np.float32))
        return np.concatenate(all_embeddings, axis=0)

    def generate_fused_embedding(
        self,
        image: Image.Image,
        caption: str,
    ) -> np.ndarray:
        img_emb = self.encode_image(image)
        txt_emb = self.encode_text(caption)
        fused = (
            config.IMAGE_EMBED_WEIGHT * img_emb
            + config.CAPTION_EMBED_WEIGHT * txt_emb
        )
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm
        return fused.astype(np.float32)