"""Interactive CLI demo for the Glance retrieval pipeline.

The demo now surfaces more structured explanations for each query, including
attribute extraction and the reranking rationale.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List


def _hr(width: int = 70) -> str:
    """Return a thin horizontal rule."""
    return "─" * width


def _banner() -> str:
    """Return the startup banner."""
    lines = [
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║        GLANCE  —  Multimodal Fashion & Context Retrieval           ║",
        "║                        Interactive Demo                            ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
    ]
    return "\n".join(lines)


def _format_matched(matched: Dict[str, Any]) -> str:
    """Condense matched_attributes into a readable one-liner."""
    parts: List[str] = []
    for key, detail in matched.items():
        if isinstance(detail, dict):
            hits = detail.get("hits", [])
            score = detail.get("score", 0.0)
            if isinstance(hits, list) and hits:
                parts.append(f"{key}: {', '.join(str(h) for h in hits)} ({score:.2f})")
            elif score > 0:
                parts.append(f"{key}: ✓ ({score:.2f})")
        elif isinstance(detail, bool) and detail:
            parts.append(key)
        elif detail:
            parts.append(f"{key}: {detail}")
    return " | ".join(parts) if parts else "(no attribute matches)"


def display_results(results: List[Dict[str, Any]], decomposed: Dict[str, Any]) -> None:
    """Pretty-print the retrieval results for one query."""
    if decomposed:
        print("\n  Decomposed attributes:")
        for key, value in decomposed.items():
            if key == "enhanced_query":
                val_str = str(value)[:70] + ("…" if len(str(value)) > 70 else "")
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    items = []
                    for item in value:
                        color = item.get("color") or "any"
                        items.append(f"{color} {item.get('type', '?')}")
                    val_str = ", ".join(items)
                else:
                    val_str = ", ".join(str(v) for v in value) if value else "(none)"
            elif isinstance(value, dict):
                val_str = ", ".join(f"{k}={v}" for k, v in value.items()) if value else "(none)"
            elif value is None:
                val_str = "(none)"
            else:
                val_str = str(value) if value else "(none)"
            print(f"    {key:20s}: {val_str}")

    if not results:
        print("\n  No results found for this query.\n")
        return

    print(f"\n  Top {len(results)} results:\n")
    for rank, match in enumerate(results, start=1):
        image_path = match.get("image_path", "N/A")
        final_score = match.get("final_score", 0.0)
        vec_sim = match.get("vector_similarity", 0.0)
        attr_score = match.get("attribute_score", 0.0)
        matched_attrs = match.get("matched_attributes", {})
        caption = str(match.get("caption", ""))

        print(f"  [{rank}]  {Path(image_path).name}")
        print(f"       Score: {final_score:.4f}  (vec={vec_sim:.4f}, attr={attr_score:.4f})")
        print(f"       Matched: {_format_matched(matched_attrs)}")
        if caption:
            print(f"       Caption: {caption[:140]}{'…' if len(caption) > 140 else ''}")
        print()


def run_demo() -> None:
    """Launch the interactive demo loop."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from retriever.retrieve_pipeline import RetrievePipeline

    print(_banner())
    print("\n  Loading retrieval pipeline — this may take a moment on first run …\n")

    pipeline = RetrievePipeline()
    decomposer = pipeline._decomposer

    print("  Pipeline ready!\n")
    print("  Type a fashion query and press Enter.")
    print("  Type 'quit' or 'exit' to leave.\n")
    print(_hr())

    while True:
        try:
            query = input("\n  Query ▸ ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye!\n")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("\n  Goodbye!\n")
            break

        print(f"\n  Searching for: \"{query}\"")
        print(_hr())

        try:
            decomposed = decomposer.decompose(query)
            results: List[Dict[str, Any]] = pipeline.retrieve(query, top_k=5)
            display_results(results, decomposed)
        except Exception as exc:
            print(f"\n  Error during retrieval: {exc}\n")

        print(_hr())


if __name__ == "__main__":
    run_demo()
