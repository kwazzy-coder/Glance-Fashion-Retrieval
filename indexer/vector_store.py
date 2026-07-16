"""
Vector store backed by ChromaDB.

Provides persistent storage and cosine-similarity retrieval of fashion
image embeddings together with their structured metadata.  All list-valued
metadata fields are serialised as comma-separated strings because ChromaDB
only supports scalar metadata values (str, int, float, bool).
"""

import logging
from typing import Any, Dict, List, Optional

try:
    import chromadb  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    chromadb = None

import config

logger = logging.getLogger(__name__)


def _flatten_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Convert list values in *metadata* to comma-separated strings.

    ChromaDB metadata values must be ``str``, ``int``, ``float``, or
    ``bool``.  Lists of strings are joined with ``","``; other
    unsupported types are cast to ``str``.
    """
    flat: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            flat[key] = ",".join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            flat[key] = str(value)
    return flat


class VectorStore:
    """Persistent vector store wrapping a ChromaDB collection.

    Each stored record consists of:
    * A unique ``image_id``
    * A float embedding vector
    * Flattened metadata (clothing types, colours, style, …)
    * The raw caption (stored inside metadata)
    """

    def __init__(
        self,
        persist_dir: str = str(config.CHROMA_PERSIST_DIR),
        collection_name: str = config.CHROMA_COLLECTION_NAME,
        embedding_dim: int | None = None,
    ) -> None:
        """Create (or re-open) a persistent ChromaDB collection.

        Parameters
        ----------
        persist_dir : str
            Filesystem path for ChromaDB's on-disk storage.
        collection_name : str
            Name of the collection inside the database.
        embedding_dim : int, optional
            Expected vector size for the active encoder. If the on-disk
            collection was built with a different model (e.g. 768 vs 512),
            it is cleared automatically.
        """
        if chromadb is None:
            raise ImportError(
                "chromadb is required for indexing. Install it with 'pip install chromadb'."
            )

        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim or config.EMBEDDING_DIM

        logger.info(
            "Initialising ChromaDB  persist_dir='%s'  collection='%s'",
            self.persist_dir,
            self.collection_name,
        )

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.ensure_embedding_dim(self.embedding_dim)

        logger.info(
            "ChromaDB collection '%s' ready – %d existing records.",
            self.collection_name,
            self.collection.count(),
        )

    def _get_stored_embedding_dim(self) -> int | None:
        """Return embedding width of stored vectors, or None if empty."""
        if self.collection.count() == 0:
            return None
        try:
            sample = self.collection.get(limit=1, include=["embeddings"])
            embeddings = sample.get("embeddings")
            # Chroma may return a NumPy array here; do not use its truth value
            # (which raises "ambiguous truth value" and previously hid the
            # stored 768-dimension index from the compatibility check).
            if embeddings is not None and len(embeddings) > 0 and embeddings[0] is not None:
                return len(embeddings[0])
        except Exception:
            logger.debug("Could not read stored embedding dim.", exc_info=True)
        return None

    def ensure_embedding_dim(self, expected_dim: int) -> None:
        """Recreate the collection if a prior model left incompatible vectors."""
        self.embedding_dim = expected_dim
        stored = self._get_stored_embedding_dim()
        if stored is not None and stored != expected_dim:
            logger.warning(
                "ChromaDB collection '%s' has %d-d embeddings but the "
                "current model produces %d-d vectors. Clearing stale index.",
                self.collection_name,
                stored,
                expected_dim,
            )
            self.clear()
        elif stored is None:
            logger.info(
                "ChromaDB collection '%s' is empty (or new). "
                "Ready to index with %d-d embeddings.",
                self.collection_name,
                expected_dim,
            )
        else:
            logger.info(
                "ChromaDB collection '%s' is compatible with %d-d embeddings.",
                self.collection_name,
                expected_dim,
            )

    # ── single insert ───────────────────────────────────────────────────

    def add_image(
        self,
        image_id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        caption: str,
    ) -> None:
        """Store a single image record.

        Parameters
        ----------
        image_id : str
            Unique identifier (typically the filename).
        embedding : list[float]
            The embedding vector.
        metadata : dict
            Structured attributes from the attribute extractor.
        caption : str
            The BLIP-2 generated caption.
        """
        flat_meta = _flatten_metadata(metadata)
        flat_meta["caption"] = caption

        # upsert so re-indexing the same files is safe
        self.collection.upsert(
            ids=[image_id],
            embeddings=[embedding],
            metadatas=[flat_meta],
        )
        logger.debug("Upserted image '%s' to collection.", image_id)

    # ── batch insert ────────────────────────────────────────────────────

    def add_images_batch(
        self,
        image_ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        captions: List[str],
    ) -> None:
        """Store multiple image records in one call.

        Parameters
        ----------
        image_ids : list[str]
            Unique identifiers.
        embeddings : list[list[float]]
            Embedding vectors (one per image).
        metadatas : list[dict]
            Structured attribute dicts (one per image).
        captions : list[str]
            Captions (one per image).
        """
        flat_metas: List[Dict[str, Any]] = []
        for meta, cap in zip(metadatas, captions):
            fm = _flatten_metadata(meta)
            fm["caption"] = cap
            flat_metas.append(fm)

        self.collection.upsert(
            ids=image_ids,
            embeddings=embeddings,
            metadatas=flat_metas,
        )
        logger.debug("Upserted batch of %d images to collection.", len(image_ids))

    # ── search ──────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: List[float],
        top_k: int = config.TOP_K_INITIAL,
    ) -> Dict:
        """Retrieve the *top_k* nearest neighbours by cosine similarity.

        Parameters
        ----------
        query_embedding : list[float]
            The query vector.
        top_k : int
            Number of results to return.

        Returns
        -------
        dict
            Raw ChromaDB results dict with keys ``ids``, ``distances``,
            ``metadatas``, and ``embeddings``.
        """
        count = self.collection.count()
        if count == 0:
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
        )
        return results

    def search_with_filter(
        self,
        query_embedding: List[float],
        where_filter: Dict,
        top_k: int = config.TOP_K_INITIAL,
    ) -> Dict:
        """Vector search restricted by a metadata ``where`` filter.

        Parameters
        ----------
        query_embedding : list[float]
            The query vector.
        where_filter : dict
            A ChromaDB ``where`` clause, e.g.
            ``{"style": "formal"}``.
        top_k : int
            Number of results to return.

        Returns
        -------
        dict
            Raw ChromaDB results dict.
        """
        current_count = self.collection.count()
        if current_count == 0:
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            where=where_filter,
            n_results=min(top_k, current_count),
        )
        return results

    # ── utilities ───────────────────────────────────────────────────────

    def get_collection_size(self) -> int:
        """Return the number of stored image records."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete all records and recreate the collection."""
        logger.warning(
            "Clearing collection '%s' (%d records).",
            self.collection_name,
            self.collection.count(),
        )
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' cleared and recreated.", self.collection_name)
