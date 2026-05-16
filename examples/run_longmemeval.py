from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from precompress import run_from_env
from precompress.longmemeval import flatten_to_messages, iter_sessions, load_longmemeval, pick_sample


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PreCompress on LongMemEval-style data")
    parser.add_argument("--data", default=None, help="Dataset JSON path. Defaults to LONGMEMEVAL_DATA from .env.")
    parser.add_argument("--output-dir", default="outputs/longmemeval", help="Directory for per-sample JSON results.")
    parser.add_argument("--mode", choices=("flatten", "session"), default="flatten",
                        help="flatten: concatenate all sessions per sample; session: run each session separately.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N samples.")
    parser.add_argument("--qid", default=None, help="Run one sample by question_id.")
    parser.add_argument("--skip-llm", action="store_true", help="Compression only; skip memory extraction.")
    return parser.parse_args()


def _build_payload_for_flat(sample: Dict[str, Any], skip_llm: bool) -> Dict[str, Any]:
    messages = flatten_to_messages(sample)
    result = run_from_env(messages, skip_llm=skip_llm)
    return {
        "question_id": sample.get("question_id"),
        "question": sample.get("question"),
        "question_type": sample.get("question_type"),
        "answer": sample.get("answer"),
        "mode": "flatten",
        "message_count": len(messages),
        "tokens_before": result.tokens_before,
        "tokens_after": result.tokens_after,
        "compression_ratio": result.compression_ratio,
        "compressed_messages": result.compressed_messages,
        "core_memory_facts": result.core_memory_facts,
        "extraction_raw": result.extraction_raw,
    }


def _build_payload_for_sessions(sample: Dict[str, Any], skip_llm: bool) -> Dict[str, Any]:
    outputs: List[Dict[str, Any]] = []
    for session in iter_sessions(sample):
        result = run_from_env(session["messages"], skip_llm=skip_llm)
        outputs.append(
            {
                "session_id": session["session_id"],
                "session_date": session["session_date"],
                "message_count": len(session["messages"]),
                "tokens_before": result.tokens_before,
                "tokens_after": result.tokens_after,
                "compression_ratio": result.compression_ratio,
                "compressed_messages": result.compressed_messages,
                "core_memory_facts": result.core_memory_facts,
                "extraction_raw": result.extraction_raw,
            }
        )
    return {
        "question_id": sample.get("question_id"),
        "question": sample.get("question"),
        "question_type": sample.get("question_type"),
        "answer": sample.get("answer"),
        "mode": "session",
        "sessions": outputs,
    }


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_longmemeval(args.data)
    if args.qid is not None:
        samples = [pick_sample(samples, question_id=args.qid)]
    elif args.limit is not None:
        samples = samples[:args.limit]

    skip_llm = args.skip_llm or not bool(os.environ.get("OPENAI_API_KEY"))
    for idx, sample in enumerate(samples):
        if args.mode == "flatten":
            payload = _build_payload_for_flat(sample, skip_llm=skip_llm)
        else:
            payload = _build_payload_for_sessions(sample, skip_llm=skip_llm)

        qid = sample.get("question_id") or f"sample_{idx}"
        out_path = output_dir / f"{qid}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
