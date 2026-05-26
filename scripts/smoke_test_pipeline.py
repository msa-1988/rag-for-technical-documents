#!/usr/bin/env python3

"""Quick smoke test for the local-first RAG pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_DIR))

from src.pipeline import answer_question, build_vector_store, get_index_status  # noqa: E402


QUESTIONS = [
    "What are the main categories of AI music generation systems?",
    "How do MusicLM and MusicGen differ in architecture and control?",
    "What does controllability mean in AI music generation?",
]


def main() -> int:
    print("Project root:", PROJECT_ROOT)
    print("Initial status:", get_index_status())
    _, stats = build_vector_store(force_rebuild=False)
    print("Index stats:", stats)

    for idx, question in enumerate(QUESTIONS, start=1):
        result = answer_question(question)
        answer = result["answer"].strip()
        sources = result["sources"]
        print(f"\nQuestion {idx}: {question}")
        print("Answer preview:", answer[:240].replace("\n", " "))
        print(
            "Sources:",
            [(source["source"], source["page"], source["retrieval"]) for source in sources],
        )
        if not answer:
            raise RuntimeError("Received an empty answer from the local model.")
        if not sources:
            raise RuntimeError("Received no sources for a smoke-test question.")

    print("\nSmoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
