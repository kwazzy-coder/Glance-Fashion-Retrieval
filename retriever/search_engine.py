"""
Two-stage search engine for the fashion retrieval system.

Stage 1 encodes the enhanced query text with FashionSigLIP and performs
approximate nearest-neighbour search against the ChromaDB collection.
An optional metadata-filtered variant narrows results using structured
attributes before falling back to unfiltered search.
"""

import logging
from typing import Any

import config
from indexer.embedding_generator import EmbeddingGenerator
from indexer.vector_store import VectorStore

logger = logging.getLogger(__name__)

_MIN_FILTERED_RESULTS = 5  # Fall-back threshold for filtered search


class SearchEngine:
    """Vector search engine backed by FashionSigLIP embeddings and ChromaDB.

    Only the **text encoder** of FashionSigLIP is needed at query time.
    The heavy image encoder / BLIP-2 captioner are not loaded, keeping
    memory usage low.
    """

    def __init__(self) -> None:
        logger.info("Initialising SearchEngine (text-encoder only) …")
        self._embedding_generator = EmbeddingGenerator(text_only=True)
        self._vector_store = VectorStore()
        logger.info("SearchEngine ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        decomposed: dict[str, Any],
        top_k: int = config.TOP_K_INITIAL,
    ) -> list[dict[str, Any]]:
        """Pure vector search — no metadata filter.

        Parameters
        ----------
        query:
            Original user query (used only for logging).
        decomposed:
            Output of ``QueryDecomposer.decompose()``.  The
            ``enhanced_query`` field is encoded for search.
        top_k:
            Number of nearest neighbours to return.

        Returns
        -------
        List of result dicts with keys *image_id*, *image_path*,
        *distance*, *metadata*.
        """
        enhanced = decomposed.get("enhanced_query", query)
        query_embedding = self._encode_query(enhanced)
        raw_results = self._vector_store.search(
            query_embedding=query_embedding, top_k=top_k
        )
        results = self._format_results(raw_results)
        logger.info(
            "Vector search for '%s' returned %d candidates.", query, len(results)
        )
        return results

    def search_with_metadata_filter(
        self,
        query: str,
        decomposed: dict[str, Any],
        top_k: int = config.TOP_K_INITIAL,
    ) -> list[dict[str, Any]]:
        """Vector search narrowed by a ChromaDB ``where`` filter built from
        the decomposed attributes.

        If the filtered search returns fewer than
        ``_MIN_FILTERED_RESULTS`` candidates the method transparently
        falls back to an unfiltered search.
        """
        enhanced = decomposed.get("enhanced_query", query)
        query_embedding = self._encode_query(enhanced)
        where_filter = self._build_where_filter(decomposed)

        if where_filter is not None:
            raw_results = self._vector_store.search_with_filter(
                query_embedding=query_embedding,
                where_filter=where_filter,
                top_k=top_k,
            )
            results = self._format_results(raw_results)
            if len(results) >= _MIN_FILTERED_RESULTS:
                logger.info(
                    "Filtered search for '%s' returned %d candidates.",
                    query,
                    len(results),
                )
                return results
            logger.warning(
                "Filtered search returned only %d results (< %d); "
                "falling back to unfiltered search.",
                len(results),
                _MIN_FILTERED_RESULTS,
            )

        # Fall back to unfiltered search.
        return self.search(query, decomposed, top_k)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_query(self, text: str) -> list[float]:
        """Encode *text* via FashionSigLIP text encoder → 512-dim vector."""
        embedding = self._embedding_generator.encode_text(text)
        # ``encode_text`` may return a numpy array or a list; normalise.
        if hasattr(embedding, "tolist"):
            return embedding.tolist()  # type: ignore[union-attr]
        return list(embedding)

    @staticmethod
    def _build_where_filter(decomposed: dict[str, Any]) -> dict[str, Any] | None:
        """Construct a ChromaDB ``where`` clause from decomposed attributes.

        Only ``environment`` and ``style`` are filtered server-side here,
        because they are stored as true scalar strings and support ``$eq``
        directly.

        ``clothing_types`` and ``colors`` are deliberately NOT filtered via
        a ChromaDB ``where`` clause: ``vector_store.py`` stores them as
        comma-joined strings (not arrays), and the ``$contains`` operator
        either isn't supported on metadata at all (pre-1.5.0 ChromaDB — it's
        a ``where_document``-only operator there) or, on newer versions,
        requires true array-typed metadata and exact element matches, not
        substring search. Using it here would either raise or silently
        return zero results. Those two attributes are instead handled by
        ``AttributeReranker``'s fuzzy, alias-aware scoring in Stage 2, which
        already operates on the same comma-joined string format.

        Returns ``None`` if no filterable constraints exist, so the caller
        can skip filtering entirely.
        """
        conditions: list[dict[str, Any]] = []

        environment = decomposed.get("environment")
        if environment:
            conditions.append({"environment": {"$eq": environment}})

        style = decomposed.get("style")
        if style:
            conditions.append({"style": {"$eq": style}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _format_results(raw: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert ChromaDB query output into a flat list of result dicts.

        ChromaDB ``collection.query()`` returns::

            {
                "ids": [[id1, id2, …]],
                "distances": [[d1, d2, …]],
                "metadatas": [[{…}, {…}, …]],
            }
        """
        ids = raw.get("ids", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]

        results: list[dict[str, Any]] = []
        for img_id, dist, meta in zip(ids, distances, metadatas):
            results.append(
                {
                    "image_id": img_id,
                    "image_path": meta.get("image_path", ""),
                    "distance": float(dist),
                    "metadata": meta,
                }
            )
        return results