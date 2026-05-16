from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from precompress import run_from_env
from precompress.longmemeval import (
    flatten_to_messages,
    iter_sessions,
    load_longmemeval,
    pick_sample,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="precompress LongMemEval demo")
    parser.add_argument("--data", default=None,
                        help="Path to longmemeval_*.json (defaults to LONGMEMEVAL_DATA env var).")
    parser.add_argument("--index", type=int, default=0,
                        help="Sample index when --qid is not given (default: 0).")
    parser.add_argument("--qid", default=None,
                        help="Pick the sample by question_id instead of --index.")
    parser.add_argument("--session-index", type=int, default=None,
                        help="If set, only run pipeline on this one haystack session "
                             "(useful when the full concatenation is too long).")
    return parser.parse_args()


def _select_messages(sample: dict, session_index):
    """Return chat-style messages to feed into the pipeline."""
    if session_index is None:
        return flatten_to_messages(sample)

    sessions = list(iter_sessions(sample))
    if not sessions:
        raise RuntimeError(f"Sample {sample.get('question_id')!r} has no non-empty sessions.")
    if session_index < 0 or session_index >= len(sessions):
        raise IndexError(
            f"--session-index={session_index} out of range "
            f"(this sample has {len(sessions)} non-empty sessions)."
        )
    return sessions[session_index]["messages"]


def main() -> None:
    args = _parse_args()

    samples = load_longmemeval(args.data)
    sample = pick_sample(samples, question_id=args.qid, index=args.index)
    messages = _select_messages(sample, args.session_index)

    skip_llm = not bool(os.environ.get("OPENAI_API_KEY"))
    if skip_llm:
        print("[warn] OPENAI_API_KEY not set - running compression only.\n")

    n_msgs = len(messages)
    n_chars = sum(len(m.get("content", "")) for m in messages)
    print("=" * 72)
    print("LONGMEMEVAL SAMPLE")
    print("=" * 72)
    print(f"question_id:   {sample.get('question_id')}")
    print(f"question_type: {sample.get('question_type')}")
    print(f"question:      {sample.get('question')}")
    print(f"answer:        {sample.get('answer')}")
    print(f"sessions:      {len(sample.get('haystack_sessions', []))} "
          f"(selected={'all' if args.session_index is None else args.session_index})")
    print(f"messages:      {n_msgs} ({n_chars:,} chars)")
    print()

    result = run_from_env(messages, skip_llm=skip_llm)

    print("=" * 72)
    print("COMPRESSED (LLMLingua-2)")
    print("=" * 72)
    print(
        f"tokens: {result.tokens_before} -> {result.tokens_after} "
        f"(ratio: {result.compression_ratio:.2%})"
    )
    preview = result.compressed_messages[0].get("content", "")[:600]
    print(f"\nfirst message preview (≤600 chars):\n{preview}")

    if not skip_llm:
        print()
        print("=" * 72)
        print("CORE MEMORY FACTS")
        print("=" * 72)
        facts = result.core_memory_facts or []
        print(f"{len(facts)} facts extracted")
        print(json.dumps(facts[:20], ensure_ascii=False, indent=2))
        if len(facts) > 20:
            print(f"... ({len(facts) - 20} more)")
        if result.extraction_raw:
            print()
            print(f"llm usage: {result.extraction_raw.get('usage')}")


if __name__ == "__main__":
    main()
