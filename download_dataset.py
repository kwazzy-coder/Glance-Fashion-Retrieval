"""
Download and prepare Fashionpedia images for the retrieval system.

Strategy (fast — avoids the 15 GB train zip):
1. Stream only the Hugging Face *val* split (~1.1k images, hundreds of MB).
2. Save images locally until we hit the requested count.
3. Optional fallback: download the smaller Fashionpedia val/test zip.
"""

from __future__ import annotations

import argparse
import logging
import random
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HF_DATASET_NAME = "detection-datasets/fashionpedia"
HF_SPLIT = "val"
VAL_ZIP_URL = config.FASHIONPEDIA_URLS["val_images"]


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _existing_image_count(image_dir: Path) -> int:
    if not image_dir.exists():
        return 0
    return sum(1 for p in image_dir.iterdir() if _is_image(p))


def _save_pil(image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(dest, "JPEG", quality=90)


def download_via_hf_streaming(max_images: int, image_dir: Path) -> int:
    """Stream the Fashionpedia val split and save up to *max_images* JPEGs."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install datasets: pip install datasets")
        return 0

    print(f"Streaming '{HF_DATASET_NAME}' [{HF_SPLIT}] (no full 15GB download)…")
    try:
        # streaming=True never materialises the train split (~15 GB).
        ds = load_dataset(
            HF_DATASET_NAME,
            split=HF_SPLIT,
            streaming=True,
        )
    except Exception as exc:
        logger.error("HF streaming failed: %s", exc)
        return 0

    ds = ds.shuffle(seed=42, buffer_size=500)

    already_have = {p.stem for p in image_dir.iterdir() if _is_image(p)}
    need = max_images - len(already_have)
    if need <= 0:
        return len(already_have)

    succeeded = 0
    failed = 0
    with tqdm(total=need, desc="Saving images", unit="img") as pbar:
        for example in ds:
            if succeeded >= need:
                break

            image_id = str(example.get("image_id", succeeded))
            if image_id in already_have:
                continue

            try:
                dest = image_dir / f"{image_id}.jpg"
                _save_pil(example["image"], dest)
                already_have.add(image_id)
                succeeded += 1
                pbar.update(1)
            except Exception:
                failed += 1
                logger.debug("Failed image_id=%s", image_id, exc_info=True)

    if failed:
        logger.warning("Skipped %d failed images during HF stream.", failed)
    return _existing_image_count(image_dir)


def _download_file(url: str, dest: Path, desc: str) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  Using cached {dest.name}")
        return True

    print(f"  Downloading {desc}…")
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as fh, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=desc,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def download_via_val_zip(max_images: int, image_dir: Path) -> int:
    """Fallback: download Fashionpedia val/test zip and sample images."""
    zip_path = config.DATA_DIR / "val_test2020.zip"
    if not _download_file(VAL_ZIP_URL, zip_path, "val_test2020.zip"):
        return _existing_image_count(image_dir)

    already_have = {p.name for p in image_dir.iterdir() if _is_image(p)}
    need = max_images - len(already_have)
    if need <= 0:
        return len(already_have)

    print("  Extracting sampled images from zip…")
    extracted = 0
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [
                m
                for m in zf.namelist()
                if Path(m).suffix.lower() in IMAGE_EXTENSIONS
            ]
            random.Random(42).shuffle(members)
            for member in tqdm(members, desc="Extracting", unit="img"):
                if extracted >= need:
                    break
                name = Path(member).name
                if name in already_have:
                    continue
                dest = image_dir / name
                with zf.open(member) as src, open(dest, "wb") as out:
                    out.write(src.read())
                already_have.add(name)
                extracted += 1
    except Exception as exc:
        logger.error("Zip extract failed: %s", exc)

    return _existing_image_count(image_dir)


def download_dataset(max_images: int = config.MAX_IMAGES) -> None:
    """Download real Fashionpedia images (stream first, zip fallback)."""
    image_dir: Path = config.IMAGE_DIR
    annotation_path = config.DATA_DIR / "annotations_val2020.json"

    # Region-aware reranking uses Fashionpedia's garment boxes to obtain
    # independent upper/lower-body colour evidence.  Download this compact
    # annotation file even when the image subset was already cached.
    if not annotation_path.is_file() or annotation_path.stat().st_size < 1_000_000:
        _download_file(
            config.FASHIONPEDIA_URLS["val_annotations"],
            annotation_path,
            "Fashionpedia validation annotations",
        )

    existing = _existing_image_count(image_dir)
    if existing >= max_images:
        print(f"\nOK  {existing} images already present in {image_dir}")
        print("   Skipping download. Delete the folder to re-download.\n")
        return

    image_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target: {max_images} images → {image_dir}")
    print(f"Already have: {existing}\n")

    count = download_via_hf_streaming(max_images, image_dir)
    if count < min(500, max_images):
        print("\nWARNING: HF stream under-delivered; trying val/test zip fallback…")
        count = download_via_val_zip(max_images, image_dir)

    print()
    print("=" * 50)
    print("  DATASET READY")
    print("=" * 50)
    print(f"  Images directory : {image_dir}")
    print(f"  Total images     : {count}")
    print("=" * 50)
    if count < 500:
        print("\nWARNING: Less than 500 images. Re-run or add images to data/images/")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Fashionpedia images for Glance (fast path).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=config.MAX_IMAGES,
        help=f"Number of images to download (default: {config.MAX_IMAGES})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(42)
    download_dataset(max_images=args.count)
