"""Evaluation harness for the Glance retrieval system.

The evaluation now reports both the original score summary and lightweight
attribute-based metrics for a small manually curated validation set.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

EVAL_QUERIES: List[str] = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]

VALIDATION_SET: List[Dict[str, Any]] = [
    {
        "query": "red shirt with blue pants",
        "expected_terms": ["shirt", "pants", "red", "blue"],
        "expected_environment": None,
        "expected_style": "casual",
    },
    {
        "query": "blue shirt with red pants",
        "expected_terms": ["shirt", "pants", "blue", "red"],
        "expected_environment": None,
        "expected_style": "casual",
    },
    {
        "query": "formal office outfit with a red tie",
        "expected_terms": ["tie", "formal", "office"],
        "expected_environment": "office",
        "expected_style": "formal",
    },
]


def _hr(char: str = "─", width: int = 80) -> str:
    """Return a horizontal rule string."""
    return char * width


def _header(text: str, width: int = 80) -> str:
    """Return a centred header inside a box."""
    border = "═" * width
    padding = (width - len(text) - 2) // 2
    line = "║" + " " * padding + text + " " * (width - padding - len(text) - 2) + "║"
    return f"╔{border}╗\n{line}\n╚{border}╝"


def _section(title: str, width: int = 80) -> str:
    """Return a section divider."""
    return f"\n{'─' * 3} {title} {'─' * (width - len(title) - 5)}"


def format_attributes(attrs: Dict[str, Any]) -> str:
    """Pretty-format the decomposed attribute dictionary."""
    lines: List[str] = []
    for key, value in attrs.items():
        if key == "enhanced_query":
            val_str = str(value)[:80] + ("…" if len(str(value)) > 80 else "")
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                parts = []
                for item in value:
                    color = item.get("color") or "any"
                    parts.append(f"{color} {item.get('type', '?')}")
                val_str = ", ".join(parts)
            else:
                val_str = ", ".join(str(v) for v in value) if value else "(none)"
        elif isinstance(value, dict):
            val_str = ", ".join(f"{k}={v}" for k, v in value.items()) if value else "(none)"
        elif value is None:
            val_str = "(none)"
        else:
            val_str = str(value) if value else "(none)"
        lines.append(f"    {key:20s}: {val_str}")
    return "\n".join(lines)


def _format_matched(matched: Dict[str, Any]) -> str:
    """Condense the matched_attributes dict into a one-line summary."""
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
    return " | ".join(parts) if parts else "(no matches)"


def _compute_metrics(results: List[Dict[str, Any]], example: Dict[str, Any], top_k: int = 5) -> Dict[str, float]:
    """Compute Precision@K and Recall@K against a small manual validation set."""
    expected_terms = set(str(term).lower() for term in example.get("expected_terms", []))
    relevant_results = 0

    for result in results[:top_k]:
        image_path = str(result.get("image_path", "")).lower()
        caption = str(result.get("caption", "")).lower()
        matched = str(result.get("matched_attributes", {})).lower()
        text_blob = f"{image_path} {caption} {matched}"
        if expected_terms and any(term in text_blob for term in expected_terms):
            relevant_results += 1

    precision_at_k = relevant_results / max(1, min(top_k, len(results)))
    recall_at_k = relevant_results / max(1, len(expected_terms))
    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
    }


def print_result(rank: int, result: Dict[str, Any]) -> None:
    """Print a single retrieval result."""
    image_path = result.get("image_path", "N/A")
    final_score = result.get("final_score", 0.0)
    vec_sim = result.get("vector_similarity", 0.0)
    attr_score = result.get("attribute_score", 0.0)
    matched = result.get("matched_attributes", {})

    print(f"  [{rank}]  {Path(image_path).name}")
    print(f"       Final score       : {final_score:.4f}")
    print(f"       Vector similarity : {vec_sim:.4f}")
    print(f"       Attribute score   : {attr_score:.4f}")
    print(f"       Matched attrs     : {_format_matched(matched)}")
    print()


def run_evaluation() -> None:
    """Execute all evaluation queries and print a formatted report."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from retriever.retrieve_pipeline import RetrievePipeline

    print(_header("GLANCE  —  Evaluation Report"))
    print()
    print("  Loading retrieval pipeline — this may take a moment …\n")

    pipeline = RetrievePipeline()
    decomposer = pipeline._decomposer
    print("  Pipeline initialised.\n")

    summary_rows: List[Dict[str, Any]] = []

    for idx, query in enumerate(EVAL_QUERIES, start=1):
        print(_section(f"Query {idx}/{len(EVAL_QUERIES)}"))
        print(f'  "{query}"\n')
        attrs = decomposer.decompose(query)
        print("  Decomposed attributes:")
        print(format_attributes(attrs))
        print()

        start = time.perf_counter()
        results: List[Dict[str, Any]] = pipeline.retrieve(query, top_k=5)
        elapsed = time.perf_counter() - start

        if not results:
            print("  (no results)\n")
        for rank, match in enumerate(results, start=1):
            print_result(rank, match)

        top_score = results[0].get("final_score", 0.0) if results else 0.0
        summary_rows.append(
            {
                "query": query[:50],
                "hits": len(results),
                "top_score": top_score,
                "time_s": elapsed,
            }
        )
        print(_hr())

    print(_section("Manual validation metrics"))
    print()
    print("  Precision@5 and Recall@5 are based on a compact manual validation set.")
    print()
    for example in VALIDATION_SET:
        try:
            results = pipeline.retrieve(example["query"], top_k=5)
        except Exception:
            results = []
        metrics = _compute_metrics(results, example, top_k=5)
        print(f"  Query: {example['query']}")
        print(f"    Precision@5: {metrics['precision_at_k']:.3f}")
        print(f"    Recall@5:    {metrics['recall_at_k']:.3f}")
        print()

    print(_section("Summary"))
    print()
    header_fmt = "  {:<4s}  {:<52s}  {:>4s}  {:>9s}  {:>7s}"
    row_fmt = "  {:<4s}  {:<52s}  {:>4d}  {:>9.4f}  {:>6.2f}s"

    print(header_fmt.format("#", "Query", "Hits", "Top Score", "Time"))
    print("  " + "─" * 4 + "  " + "─" * 52 + "  " + "─" * 4 + "  " + "─" * 9 + "  " + "─" * 7)
    for index, row in enumerate(summary_rows, start=1):
        print(row_fmt.format(str(index), row["query"], row["hits"], row["top_score"], row["time_s"]))
    print()
    avg_time = sum(item["time_s"] for item in summary_rows) / max(len(summary_rows), 1)
    print(f"  Average retrieval time: {avg_time:.2f}s")
    print()
    print(_header("Evaluation Complete"))
    print()


if __name__ == "__main__":
    run_evaluation()
