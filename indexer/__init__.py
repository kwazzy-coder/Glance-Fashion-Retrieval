"""
Indexer package for the Multimodal Fashion & Context Retrieval System.

Provides the full indexing pipeline: caption generation, attribute extraction,
embedding generation, vector storage, and the orchestrating index pipeline.

Imports are lazy to avoid triggering heavy model loading on package import.
"""


def __getattr__(name):
    """Lazy-load classes to avoid loading heavy models at import time."""
    if name == "CaptionGenerator":
        from indexer.caption_generator import CaptionGenerator
        return CaptionGenerator
    elif name == "AttributeExtractor":
        from indexer.attribute_extractor import AttributeExtractor
        return AttributeExtractor
    elif name == "EmbeddingGenerator":
        from indexer.embedding_generator import EmbeddingGenerator
        return EmbeddingGenerator
    elif name == "VectorStore":
        from indexer.vector_store import VectorStore
        return VectorStore
    elif name == "IndexPipeline":
        from indexer.index_pipeline import IndexPipeline
        return IndexPipeline
    raise AttributeError(f"module 'indexer' has no attribute {name!r}")


__all__ = [
    "CaptionGenerator",
    "AttributeExtractor",
    "EmbeddingGenerator",
    "VectorStore",
    "IndexPipeline",
]
