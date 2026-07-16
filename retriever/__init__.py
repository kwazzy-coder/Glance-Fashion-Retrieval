"""
Retriever package for the Multimodal Fashion & Context Retrieval System.

Exposes the main classes for query decomposition, vector search,
attribute-based re-ranking, and end-to-end retrieval orchestration.

Imports are lazy to avoid triggering heavy model loading on package import.
"""


def __getattr__(name):
    """Lazy-load classes to avoid loading heavy models at import time."""
    if name == "QueryDecomposer":
        from retriever.query_decomposer import QueryDecomposer
        return QueryDecomposer
    elif name == "SearchEngine":
        from retriever.search_engine import SearchEngine
        return SearchEngine
    elif name == "AttributeReranker":
        from retriever.reranker import AttributeReranker
        return AttributeReranker
    elif name == "RetrievePipeline":
        from retriever.retrieve_pipeline import RetrievePipeline
        return RetrievePipeline
    raise AttributeError(f"module 'retriever' has no attribute {name!r}")


__all__ = [
    "QueryDecomposer",
    "SearchEngine",
    "AttributeReranker",
    "RetrievePipeline",
]
