"""
End-to-end retrieval pipeline for the Multimodal Fashion & Context
Retrieval System.

Orchestrates query decomposition → vector search (with optional
metadata filtering) → attribute-based re-ranking, producing a final
ranked list of fashion images.
"""

import logging
import sys
from typing import Any

import config
from retriever.query_decomposer import QueryDecomposer
from retriever.reranker import AttributeReranker
from retriever.search_engine import SearchEngine

logger = logging.getLogger(__name__)


class RetrievePipeline:
    """Facade that wires together decomposer, search engine, and re-ranker.

    Usage::

        pipeline = RetrievePipeline()
        results  = pipeline.retrieve("blue shirt in a park", top_k=5)
    """

    def __init__(self) -> None:
        logger.info("Initialising RetrievePipeline …")
        self._decomposer = QueryDecomposer()
        self._search_engine = SearchEngine()
        self._reranker = AttributeReranker()
        logger.info("RetrievePipeline ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = config.TOP_K_FINAL,
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        """Run the full retrieval pipeline.

        Parameters
        ----------
        query:
            Natural-language retrieval query.
        top_k:
            Number of final results to return.
        verbose:
            If ``True``, prints intermediate diagnostics to stdout.

        Returns
        -------
        List of result dicts sorted by ``final_score`` (descending).
        Each dict contains *image_id*, *image_path*, *final_score*,
        *vector_similarity*, *attribute_score*, and
        *matched_attributes*.
        """
        # 1. Decompose --------------------------------------------------
        decomposed = self._decomposer.decompose(query)
        if verbose:
            self._print_decomposition(query, decomposed)

        # 2. Search (prefer filtered, fall back to pure vector) ----------
        has_constraints = any([
            decomposed.get("colors"),
            decomposed.get("clothing_types"),
            decomposed.get("environment"),
            decomposed.get("style"),
        ])

        if has_constraints:
            candidates = self._search_engine.search_with_metadata_filter(
                query, decomposed, top_k=config.TOP_K_INITIAL
            )
        else:
            candidates = self._search_engine.search(
                query, decomposed, top_k=config.TOP_K_INITIAL
            )

        if verbose:
            print(f"  Stage-1 candidates: {len(candidates)}")

        # 3. Re-rank -----------------------------------------------------
        results = self._reranker.rerank(candidates, decomposed, top_k=top_k)

        if verbose:
            self._print_results(results)

        return results

    # ------------------------------------------------------------------
    # Pretty-printing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_decomposition(query: str, decomposed: dict[str, Any]) -> None:
        """Print the decomposed query to stdout for debugging."""
        print("\n" + "=" * 72)
        print(f"  QUERY: {query}")
        print("=" * 72)
        print(f"  Clothing items : {decomposed.get('clothing_items', [])}")
        print(f"  Colours        : {decomposed.get('colors', [])}")
        print(f"  Clothing types : {decomposed.get('clothing_types', [])}")
        print(f"  Environment    : {decomposed.get('environment')}")
        print(f"  Style          : {decomposed.get('style')}")
        print(f"  Enhanced query : {decomposed.get('enhanced_query', query)}")
        print("-" * 72)

    @staticmethod
    def _print_results(results: list[dict[str, Any]]) -> None:
        """Print the final ranked results to stdout."""
        print(f"\n  Top {len(results)} results:")
        print(f"  {'Rank':<5} {'Score':>7} {'Vec':>7} {'Attr':>7}  {'Image'}")
        print(f"  {'─' * 5} {'─' * 7} {'─' * 7} {'─' * 7}  {'─' * 40}")
        for rank, r in enumerate(results, 1):
            print(
                f"  {rank:<5} "
                f"{r['final_score']:>7.4f} "
                f"{r['vector_similarity']:>7.4f} "
                f"{r['attribute_score']:>7.4f}  "
                f"{r['image_path']}"
            )
        print()


# ======================================================================
# CLI entry-point — run evaluation queries
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
        stream=sys.stderr,
    )

    eval_queries = [
        "A person in a bright yellow raincoat.",
        "Professional business attire inside a modern office.",
        "Someone wearing a blue shirt sitting on a park bench.",
        "Casual weekend outfit for a city walk.",
        "A red tie and a white shirt in a formal setting.",
    ]

    pipeline = RetrievePipeline()

    for q in eval_queries:
        try:
            results = pipeline.retrieve(q, top_k=5, verbose=True)
        except Exception:
            logger.exception("Error while retrieving for query: %s", q)
