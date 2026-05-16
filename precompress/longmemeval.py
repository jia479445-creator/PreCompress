from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


def resolve_data_path(path: Optional[Union[str, Path]] = None) -> Path:

    if path is not None:
        return Path(path).expanduser().resolve()

    env_path = os.environ.get("LONGMEMEVAL_DATA")
    if env_path:
        return Path(env_path).expanduser().resolve()

    return Path("data/longmemeval_s.json").resolve()


def load_longmemeval(path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Load `longmemeval_s.json` (or any LongMemEval variant) into a list of samples."""
    resolved = resolve_data_path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"LongMemEval data not found at {resolved}.\n"
            "Download `longmemeval_s.json` from:\n"
            "    https://huggingface.co/datasets/xiaowu0162/longmemeval\n"
            "and set LONGMEMEVAL_DATA in your .env to the local file path."
        )

    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array of samples in {resolved}, got {type(data).__name__}."
        )
    return data


def iter_sessions(sample: Dict[str, Any]) -> Iterator[Dict[str, Any]]:

    sessions = sample.get("haystack_sessions", []) or []
    dates = sample.get("haystack_dates", []) or []
    session_ids = sample.get("haystack_session_ids", []) or []

    for idx, raw_session in enumerate(sessions):
        if not raw_session:
            continue
        session_date = dates[idx] if idx < len(dates) else ""
        session_id = session_ids[idx] if idx < len(session_ids) else f"sess_{idx}"

        messages = []
        for j, msg in enumerate(raw_session):
            messages.append(
                {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                    "time_stamp": session_date,
                    "sequence_number": j * 2,
                }
            )

        yield {
            "session_id": session_id,
            "session_date": session_date,
            "messages": messages,
        }


def flatten_to_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Concatenate every haystack session into one flat message list.

    Sequence numbers are renumbered across the concatenation so they stay
    globally even and monotonically increasing.
    """
    flat: List[Dict[str, Any]] = []
    cursor = 0
    for session in iter_sessions(sample):
        for msg in session["messages"]:
            flat.append({**msg, "sequence_number": cursor})
            cursor += 2
    return flat


def pick_sample(
    samples: List[Dict[str, Any]],
    question_id: Optional[str] = None,
    index: int = 0,
) -> Dict[str, Any]:
    if question_id is not None:
        for s in samples:
            if s.get("question_id") == question_id:
                return s
        raise KeyError(f"No LongMemEval sample with question_id={question_id!r}")
    if not samples:
        raise IndexError("LongMemEval data is empty.")
    return samples[index]


__all__ = [
    "resolve_data_path",
    "load_longmemeval",
    "iter_sessions",
    "flatten_to_messages",
    "pick_sample",
]
