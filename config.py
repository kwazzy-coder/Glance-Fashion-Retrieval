"""
Configuration for the Multimodal Fashion & Context Retrieval System.

Centralizes all paths, model identifiers, embedding dimensions,
and hyperparameters used across the indexer and retriever pipelines.
"""

from __future__ import annotations

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
CHROMA_PERSIST_DIR = DATA_DIR / "chromadb"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

# Create directories if they don't exist
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
# Default: OpenAI CLIP ViT-B/32 via open_clip (~151 MB) — fast to download.
# Fashion quality comes from zero-shot attribute captions + compositional re-rank.
#
# Optional upgrade (slower first download, ~813 MB):
#   OPEN_CLIP_MODEL = "hf-hub:Marqo/marqo-fashionSigLIP"
#   OPEN_CLIP_PRETRAINED = None
OPEN_CLIP_MODEL = "ViT-B-32"
# Prefer a local OpenAI .pt file (Azure CDN) to avoid Hugging Face stalls.
# Set CLIP_CHECKPOINT=/path/to/ViT-B-32.pt or place the file in the project root.
OPEN_CLIP_PRETRAINED = "openai"
EMBEDDING_DIM = 512

_OPENAI_VIT_B32_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
)


def resolve_clip_checkpoint() -> str | None:
    """Return path to a local ViT-B-32.pt if present, else None."""
    candidates = []
    env = os.environ.get("CLIP_CHECKPOINT")
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            PROJECT_ROOT / "ViT-B-32.pt",
            PROJECT_ROOT / "checkpoints" / "ViT-B-32.pt",
            Path.cwd() / "ViT-B-32.pt",
            Path("/content/ViT-B-32.pt"),
            Path("/content/glance/ViT-B-32.pt"),
            Path("/content/Glance-Fashion-Retrieval/ViT-B-32.pt"),
        ]
    )
    for path in candidates:
        try:
            if path.is_file() and path.stat().st_size > 50_000_000:
                return str(path.resolve())
        except OSError:
            continue
    return None


# Resolved at import; re-assign after download in notebooks if needed.
CLIP_CHECKPOINT_PATH = resolve_clip_checkpoint()
if CLIP_CHECKPOINT_PATH:
    OPEN_CLIP_PRETRAINED = CLIP_CHECKPOINT_PATH

# Back-compat alias used in older docs / notebooks
FASHIONSIGLIP_MODEL = OPEN_CLIP_MODEL

# Captions: same open_clip encoder scores fashion prompts (no BLIP download).
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
BLIP2_MODEL = BLIP_MODEL
CAPTION_BACKEND = "zeroshot"

# ──────────────────────────────────────────────
# Indexing Configuration
# ──────────────────────────────────────────────
BATCH_SIZE = 16                  # Batch size for GPU inference
MAX_IMAGES = 1000                # Target dataset size
CAPTION_MAX_LENGTH = 128         # Max tokens for BLIP-2 caption generation
NUM_CAPTIONS = 1                 # Captions per image (we use prompted captioning)

# Fused embedding weights
IMAGE_EMBED_WEIGHT = 0.7         # Weight for visual embedding
CAPTION_EMBED_WEIGHT = 0.3       # Weight for caption text embedding

# ──────────────────────────────────────────────
# Retrieval Configuration
# ──────────────────────────────────────────────
TOP_K_INITIAL = 50               # Candidates from vector search (Stage 1)
TOP_K_FINAL = 10                 # Results after re-ranking (Stage 2)

# Re-ranking weights
VECTOR_SIM_WEIGHT = 0.65         # Weight for vector cosine similarity
ATTRIBUTE_MATCH_WEIGHT = 0.35    # Weight for attribute match score

# ──────────────────────────────────────────────
# ChromaDB Configuration
# ──────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "fashion_images"
# Bump when required metadata fields change; stale records are rebuilt safely.
INDEX_SCHEMA_VERSION = 2

# ──────────────────────────────────────────────
# Device Configuration
# ──────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ──────────────────────────────────────────────
# Dataset URLs (Fashionpedia)
# ──────────────────────────────────────────────
FASHIONPEDIA_URLS = {
    "train_images": "https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip",
    "val_images": "https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip",
    "train_annotations": "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json",
    "val_annotations": "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json",
}

# ──────────────────────────────────────────────
# Fashion Taxonomy (shared between indexer & retriever)
# ──────────────────────────────────────────────

# Clothing categories → style mapping
CLOTHING_TAXONOMY = {
    "formal": [
        "blazer", "suit", "tuxedo", "dress shirt", "button-down", "button-up",
        "dress pants", "slacks", "trousers", "pants", "pencil skirt", "skirt",
        "formal dress", "dress",
        "gown", "waistcoat", "vest", "necktie", "tie", "bow tie", "cufflinks",
        "oxford shoes", "heels", "pumps", "loafers", "blouse", "shirt",
    ],
    "casual": [
        "t-shirt", "tee", "top", "jeans", "denim", "cargo pants",
        "chinos", "khakis", "hoodie", "sweatshirt",
        "sneakers", "shorts", "tank top", "polo", "cardigan", "sweater",
        "pullover", "leggings", "joggers", "sweatpants", "flip-flops",
        "sandals", "cap", "baseball cap", "beanie", "casual dress",
        "sundress", "romper", "overalls", "shirt",
    ],
    "outerwear": [
        "jacket", "coat", "raincoat", "rain coat", "parka", "windbreaker",
        "trench coat", "overcoat", "puffer jacket", "down jacket",
        "bomber jacket", "leather jacket", "denim jacket", "fleece",
        "poncho", "cape", "anorak", "peacoat",
    ],
    "activewear": [
        "sports bra", "athletic shorts", "running shoes", "track pants",
        "yoga pants", "workout top", "jersey", "swimsuit", "bikini",
        "wetsuit", "cycling shorts", "compression",
    ],
    "accessories": [
        "hat", "scarf", "gloves", "belt", "watch", "sunglasses",
        "handbag", "purse", "backpack", "necklace", "bracelet",
        "earrings", "ring", "brooch", "tie clip", "pocket square",
        "umbrella", "tote bag", "clutch",
    ],
}

# Flattened lookup: item → category
ITEM_TO_CATEGORY = {}
for category, items in CLOTHING_TAXONOMY.items():
    for item in items:
        ITEM_TO_CATEGORY[item] = category

# Equivalent garment terms keep user phrasing aligned with the more
# fashion-specific wording that zero-shot captions may emit.
GARMENT_EQUIVALENTS = {
    "pants": {"pants", "trousers", "slacks", "dress pants", "chinos", "khakis", "cargo pants"},
    "shirt": {"shirt", "dress shirt", "button-down", "button-up"},
    "t-shirt": {"t-shirt", "tee"},
    "coat": {"coat", "overcoat", "trench coat", "peacoat", "raincoat"},
    "jacket": {"jacket", "bomber jacket", "leather jacket", "denim jacket", "puffer jacket"},
    "dress": {"dress", "formal dress", "casual dress", "sundress", "gown"},
    "top": {"top", "t-shirt", "tee", "tank top", "blouse", "shirt"},
    "skirt": {"skirt", "pencil skirt"},
    "blazer": {"blazer", "suit jacket", "jacket"},
    "jeans": {"jeans", "denim", "pants"},
}

# Color vocabulary with aliases
COLOR_VOCABULARY = {
    "red": ["red", "crimson", "scarlet", "ruby", "cherry", "maroon", "burgundy", "wine"],
    "blue": ["blue", "navy", "cobalt", "azure", "indigo", "teal", "cerulean", "royal blue", "sky blue", "baby blue"],
    "green": ["green", "olive", "emerald", "lime", "sage", "forest green", "mint", "jade", "hunter green"],
    "yellow": ["yellow", "gold", "golden", "mustard", "lemon", "amber", "canary", "saffron"],
    "orange": ["orange", "tangerine", "coral", "peach", "rust", "burnt orange", "apricot"],
    "purple": ["purple", "violet", "lavender", "plum", "magenta", "mauve", "lilac", "amethyst"],
    "pink": ["pink", "rose", "fuchsia", "blush", "salmon", "hot pink", "baby pink"],
    "black": ["black", "charcoal", "ebony", "onyx", "jet black"],
    "white": ["white", "ivory", "cream", "off-white", "pearl", "snow white", "eggshell"],
    "gray": ["gray", "grey", "silver", "slate", "ash", "charcoal gray", "pewter"],
    "brown": ["brown", "tan", "beige", "khaki", "camel", "chocolate", "coffee", "taupe", "mocha", "espresso"],
}

# Reverse lookup: alias → canonical color
ALIAS_TO_COLOR = {}
for canonical, aliases in COLOR_VOCABULARY.items():
    for alias in aliases:
        ALIAS_TO_COLOR[alias.lower()] = canonical

# Environment keywords → category
ENVIRONMENT_KEYWORDS = {
    "office": ["office", "desk", "computer", "meeting", "conference", "workspace",
               "cubicle", "boardroom", "corporate", "professional", "work"],
    "urban": ["street", "city", "sidewalk", "downtown", "building", "urban",
              "crosswalk", "traffic", "metropolitan", "alley", "pavement"],
    "park": ["park", "garden", "grass", "tree", "bench", "outdoor", "nature",
             "lake", "pond", "flowers", "path", "trail", "picnic"],
    "home": ["home", "house", "living room", "bedroom", "kitchen", "couch",
             "sofa", "interior", "apartment", "domestic", "cozy"],
    "beach": ["beach", "sand", "ocean", "sea", "coast", "shore", "tropical",
              "waves", "surfing", "seaside"],
    "formal_venue": ["restaurant", "gala", "wedding", "ceremony", "banquet",
                     "ballroom", "theater", "opera", "red carpet", "stage"],
    "gym": ["gym", "fitness", "workout", "exercise", "treadmill", "weights",
            "yoga studio", "sports", "athletic", "training"],
}

# Generic scene intents are query-side constraints.  Indexed images retain
# their more informative scene labels (park, office, urban, ...), while the
# re-ranker maps those labels back to indoor/outdoor when needed.
ENVIRONMENT_COMPATIBILITY = {
    "indoor": {"office", "home", "formal_venue", "gym"},
    "outdoor": {"urban", "park", "beach"},
}

# Style inference keywords
STYLE_KEYWORDS = {
    "formal": ["formal", "professional", "business", "elegant", "sophisticated",
               "classy", "dressed up", "polished", "refined", "executive"],
    "casual": ["casual", "relaxed", "comfortable", "laid-back", "everyday",
               "weekend", "easygoing", "informal", "chill", "effortless"],
    "sporty": ["sporty", "athletic", "active", "fitness", "workout", "sport",
               "training", "running", "exercise"],
    "streetwear": ["streetwear", "urban", "hip-hop", "edgy", "trendy",
                   "cool", "street style", "skate"],
    "bohemian": ["bohemian", "boho", "hippie", "free-spirited", "artistic",
                 "eclectic", "vintage"],
    "minimalist": ["minimalist", "simple", "clean", "understated", "neutral",
                   "monochrome", "sleek"],
}
