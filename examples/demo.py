"""CLI demo: raw text -> LLMLingua-2 -> LLM core-memory facts.

All configuration is driven by `.env` (see `.env.example` in the project root).
The package auto-loads `.env` on import, so this file does no config wiring.

Usage:
    python -m examples.demo                 # built-in sample text
    python -m examples.demo path/to/file    # compress + summarize a text file
"""

from __future__ import annotations

import json
import os
import sys

from precompress import run_from_env


DEMO_TEXT = (
    "My name is Alice and I work as a high-school physics teacher in Boston. "
    "I have been teaching for about eight years and recently started a part-time "
    "master's degree in education. My favorite movies are Inception and Interstellar, "
    "and last summer I traveled to Paris with my friend John, who is currently "
    "studying medicine at Johns Hopkins. I drank a lot of espresso while there and "
    "decided I prefer Italian roasts to French ones. I also dislike decaf and rarely "
    "drink tea. By the way, today is a slow Tuesday morning and I am writing this "
    "from my classroom while waiting for my next class to begin."
)


def _read_input() -> str:
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            return f.read()
    return DEMO_TEXT


def main() -> None:
    text = _read_input()
    skip_llm = not bool(os.environ.get("OPENAI_API_KEY"))
    if skip_llm:
        print("[warn] OPENAI_API_KEY not set - running compression only.\n")

    result = run_from_env(text, skip_llm=skip_llm)

    print("=" * 72)
    print("ORIGINAL TEXT")
    print("=" * 72)
    print(result.original_messages[0]["content"])
    print()
    print("=" * 72)
    print("COMPRESSED TEXT (LLMLingua-2)")
    print("=" * 72)
    print(result.compressed_messages[0]["content"])
    print()
    print(
        f"tokens: {result.tokens_before} -> {result.tokens_after} "
        f"(ratio: {result.compression_ratio:.2%})"
    )

    if not skip_llm:
        print()
        print("=" * 72)
        print("CORE MEMORY FACTS")
        print("=" * 72)
        print(json.dumps(result.core_memory_facts, ensure_ascii=False, indent=2))
        if result.extraction_raw:
            print()
            print(f"llm usage: {result.extraction_raw.get('usage')}")


if __name__ == "__main__":
    main()
