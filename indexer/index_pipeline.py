"""
End-to-end indexing pipeline for the Fashion & Context Retrieval System.

Uses a **single** FashionSigLIP model for:
  - zero-shot fashion/context captions
  - image + caption embeddings

No separate BLIP / BLIP-2 download (~2–15 GB avoided).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List

from PIL import Image
from tqdm import tqdm

import config
from indexer.attribute_extractor import AttributeExtractor
from indexer.caption_generator import CaptionGenerator
from indexer.embedding_generator import EmbeddingGenerator
from indexer.region_attribute_extractor import RegionAttributeExtractor
from indexer.vector_store import VectorStore

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class IndexPipeline:
    """Orchestrates indexing from raw images to the vector store."""

    def __init__(self) -> None:
        logger.info("Initialising IndexPipeline (FashionSigLIP only — no BLIP)…")

        # Load FashionSigLIP once; share with the captioner.
        self.embedding_generator = EmbeddingGenerator(device=config.DEVICE)
        self.caption_generator = CaptionGenerator(
            device=config.DEVICE,
            embedding_generator=self.embedding_generator,
        )
        self.attribute_extractor = AttributeExtractor()
        self.region_attribute_extractor = RegionAttributeExtractor()

        # Probe live embedding width (ViT-B/32 → 512; other encoders may differ).
        probe = self.embedding_generator.encode_text("fashion probe")
        embedding_dim = int(probe.shape[0])

        self.vector_store = VectorStore(
            persist_dir=str(config.CHROMA_PERSIST_DIR),
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_dim=embedding_dim,
        )

        logger.info("IndexPipeline ready.")

    def index_directory(
        self,
        image_dir: str,
        max_images: int = config.MAX_IMAGES,
        batch_size: int = config.BATCH_SIZE,
    ) -> Dict[str, int | float]:
        """Index images in *image_dir* and return an auditable summary.

        Returning counts makes notebooks and services able to report a clear
        outcome without scraping log output.  The batched API remains the
        scalability boundary: for millions of images, feed batches from an
        object-store iterator and keep the same VectorStore interface.
        """
        image_dir_path = Path(image_dir)
        if not image_dir_path.is_dir():
            logger.error("Image directory does not exist: %s", image_dir)
            return {"processed": 0, "failed": 0, "total": 0, "seconds": 0.0}

        image_paths: List[Path] = sorted(
            p
            for p in image_dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )

        if not image_paths:
            logger.warning("No images found in '%s'.", image_dir)
            return {"processed": 0, "failed": 0, "total": 0, "seconds": 0.0}

        image_paths = image_paths[:max_images]
        total = len(image_paths)
        logger.info(
            "Found %d images in '%s' (capped at %d). Starting indexing…",
            total,
            image_dir,
            max_images,
        )

        succeeded = 0
        failed = 0
        t_start = time.perf_counter()

        for start in tqdm(
            range(0, total, batch_size),
            desc="Indexing batches",
            unit="batch",
        ):
            batch_paths = image_paths[start : start + batch_size]
            try:
                n = self._index_batch(batch_paths)
                succeeded += n
                failed += len(batch_paths) - n
            except Exception:
                logger.exception(
                    "Batch failed at offset %d; falling back to per-image.",
                    start,
                )
                for path in batch_paths:
                    try:
                        self._index_image(path)
                        succeeded += 1
                    except Exception:
                        failed += 1
                        logger.exception("Failed to index '%s'.", path)

        elapsed = time.perf_counter() - t_start
        summary = (
            f"\n{'=' * 60}\n"
            f"  Indexing complete\n"
            f"  Directory  : {image_dir}\n"
            f"  Processed  : {succeeded}/{total}\n"
            f"  Failed     : {failed}\n"
            f"  Total time : {elapsed:.1f}s\n"
            f"  Per image  : {elapsed / max(succeeded, 1):.2f}s\n"
            f"  Collection : {self.vector_store.get_collection_size()} records\n"
            f"{'=' * 60}"
        )
        logger.info(summary)
        print(summary)
        return {
            "processed": succeeded,
            "failed": failed,
            "total": total,
            "seconds": elapsed,
        }

    def index_single(self, image_path: str) -> Dict:
        """Index one image and return its metadata."""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self._index_image(path)

    def _index_batch(self, paths: List[Path]) -> int:
        images: List[Image.Image] = []
        ok_paths: List[Path] = []
        for path in paths:
            try:
                images.append(Image.open(path).convert("RGB"))
                ok_paths.append(path)
            except Exception:
                logger.exception("Could not open '%s'.", path)

        if not images:
            return 0

        caption_records = self.caption_generator.generate_caption_records_batch(images)
        captions = [record["caption"] for record in caption_records]
        attrs_list = [self.attribute_extractor.extract_attributes(caption) for caption in captions]
        for path, image, attrs, record in zip(ok_paths, images, attrs_list, caption_records):
            attrs.update(self.region_attribute_extractor.extract(image, path.name))
            attrs["caption_confidence"] = record["caption_confidence"]

        img_embs = self.embedding_generator.encode_images_batch(images)
        embeddings = []
        for img_emb, caption in zip(img_embs, captions):
            txt_emb = self.embedding_generator.encode_text(caption)
            fused = (
                config.IMAGE_EMBED_WEIGHT * img_emb
                + config.CAPTION_EMBED_WEIGHT * txt_emb
            )
            norm = float((fused ** 2).sum()) ** 0.5
            if norm > 0:
                fused = fused / norm
            embeddings.append(fused.astype("float32").tolist())

        image_ids = [p.name for p in ok_paths]
        metadatas = []
        for path, attrs in zip(ok_paths, attrs_list):
            store_meta = {k: v for k, v in attrs.items() if k != "raw_caption"}
            store_meta["image_path"] = str(path)
            store_meta["corpus_id"] = config.IMAGE_DIR.name
            store_meta["index_schema_version"] = config.INDEX_SCHEMA_VERSION
            metadatas.append(store_meta)

        self.vector_store.add_images_batch(
            image_ids=image_ids,
            embeddings=embeddings,
            metadatas=metadatas,
            captions=captions,
        )
        return len(ok_paths)

    def _index_image(self, path: Path) -> Dict:
        """Process a single image through the full pipeline."""
        image = Image.open(path).convert("RGB")
        image_id = path.name

        caption_record = self.caption_generator.generate_caption_record(image)
        caption = caption_record["caption"]
        attributes = self.attribute_extractor.extract_attributes(caption)
        attributes.update(self.region_attribute_extractor.extract(image, path.name))
        attributes["caption_confidence"] = caption_record["caption_confidence"]
        embedding = self.embedding_generator.generate_fused_embedding(
            image, caption
        )

        store_meta = {k: v for k, v in attributes.items() if k != "raw_caption"}
        store_meta["image_path"] = str(path)
        store_meta["corpus_id"] = config.IMAGE_DIR.name
        store_meta["index_schema_version"] = config.INDEX_SCHEMA_VERSION
        self.vector_store.add_image(
            image_id=image_id,
            embedding=embedding.tolist(),
            metadata=store_meta,
            caption=caption,
        )

        logger.debug("Indexed '%s'  caption='%s'", image_id, caption[:80])
        return attributes


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    print(f"Device: {config.DEVICE}")
    print(f"Dtype: {config.DTYPE}")
    print(f"Target images: {config.MAX_IMAGES}")
    print(f"Encoder: {config.OPEN_CLIP_MODEL} / {config.OPEN_CLIP_PRETRAINED}")
    print("Caption backend: open_clip zero-shot (no BLIP download)")
    print()

    pipeline = IndexPipeline()
    pipeline.index_directory(str(config.IMAGE_DIR))
