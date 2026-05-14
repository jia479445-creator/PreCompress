"""Shared pytest fixtures and integration-test gating.

This suite contains real-scenario tests only. They actually load the
LLMLingua-2 model and / or call an OpenAI-compatible API, so each test
class is gated behind an environment-variable check:

    * RUN_INTEGRATION=1     enable the LLMLingua-2 model tests
    * OPENAI_API_KEY=...    enable the OpenAI-API tests
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_messages():
    """Pre-canned three-turn dialogue used across the integration tests."""
    path = Path(__file__).resolve().parent / "data" / "sample_dialogue.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def locomo_messages():
    """Two-speaker dialogue used by the event-mode (factual + relational) test."""
    path = Path(__file__).resolve().parent / "data" / "locomo_dialogue.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


needs_llmlingua = pytest.mark.skipif(
    not _flag_enabled("RUN_INTEGRATION"),
    reason="set RUN_INTEGRATION=1 to run real LLMLingua-2 compression",
)

needs_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="set OPENAI_API_KEY to run real OpenAI-compatible API tests",
)
