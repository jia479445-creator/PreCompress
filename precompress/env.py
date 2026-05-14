"""Unified .env-based configuration layer.

This module is the **only** place outside `pipeline.py` that is not lifted
verbatim from LightMem. It provides:

    * `load_dotenv_if_present()` - a tiny zero-dependency `.env` parser.
    * `llmlingua_config_from_env()` - builds a `LlmLingua2Config` from env vars.
    * `manager_config_from_env()` - builds a `BaseMemoryManagerConfig` from env vars.
    * `run_from_env(text)` - convenience: load .env + run the full pipeline.

The `.env` file is loaded automatically when `precompress` is imported (see
`precompress/__init__.py`). Real OS environment variables always win over the
file, so you can still override per-process: `OPENAI_MODEL=gpt-4o python -m examples.demo`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .compressor import LlmLingua2Compressor, LlmLingua2Config
from .extractor import BaseMemoryManagerConfig, OpenaiManager
from .pipeline import PipelineResult, run_pipeline


# ---------------------------------------------------------------------------
# 1. Lightweight .env parser (no external dependency)
# ---------------------------------------------------------------------------
def load_dotenv_if_present(path: Union[str, Path, None] = None) -> bool:
    """Load `KEY=VALUE` lines from a `.env` file into `os.environ`.

    * Lines starting with `#` are comments.
    * Surrounding single / double quotes on values are stripped.
    * Real environment variables ALWAYS win - file values use `setdefault`,
      so you can override anything per-process.

    Args:
        path: path to the .env file. If None, searches for a `.env` file
              starting from the current working directory, walking up to
              the filesystem root.

    Returns:
        True if a file was found and parsed; False otherwise.
    """
    env_path = _resolve_env_path(path)
    if env_path is None:
        return False

    with open(env_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip a single layer of surrounding quotes if present.
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            os.environ.setdefault(key, value)
    return True


def _resolve_env_path(path: Union[str, Path, None]) -> Optional[Path]:
    if path is not None:
        p = Path(path)
        return p if p.exists() else None

    # Walk up from CWD looking for a .env file.
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        f = candidate / ".env"
        if f.exists():
            return f
    return None


# ---------------------------------------------------------------------------
# 2. Typed env-var readers
# ---------------------------------------------------------------------------
def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# 3. Config factories
# ---------------------------------------------------------------------------
def llmlingua_config_from_env() -> LlmLingua2Config:
    """Build a `LlmLingua2Config` from environment variables.

    Recognized variables (see `.env.example`):
        LLMLINGUA_MODEL, LLMLINGUA_DEVICE, LLMLINGUA_USE_V2,
        LLMLINGUA_MAX_BATCH_SIZE, LLMLINGUA_MAX_FORCE_TOKEN,
        LLMLINGUA_RATE, LLMLINGUA_TARGET_TOKEN, LLMLINGUA_INSTRUCTION
    """
    return LlmLingua2Config(
        llmlingua_config={
            "model_name": _env_str(
                "LLMLINGUA_MODEL",
                "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            ),
            "device_map": _env_str("LLMLINGUA_DEVICE", "cuda"),
            "use_llmlingua2": _env_bool("LLMLINGUA_USE_V2", True),
        },
        llmlingua2_config={
            "max_batch_size": _env_int("LLMLINGUA_MAX_BATCH_SIZE", 50),
            "max_force_token": _env_int("LLMLINGUA_MAX_FORCE_TOKEN", 100),
        },
        compress_config={
            "instruction": _env_str("LLMLINGUA_INSTRUCTION", "") or "",
            "rate": _env_float("LLMLINGUA_RATE", 0.5),
            "target_token": _env_int("LLMLINGUA_TARGET_TOKEN", -1),
        },
    )


def manager_config_from_env() -> BaseMemoryManagerConfig:
    """Build a `BaseMemoryManagerConfig` from environment variables.

    Recognized variables (see `.env.example`):
        OPENAI_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL,
        OPENROUTER_API_KEY, OPENROUTER_API_BASE,
        LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P,
        OPENROUTER_SITE_URL, OPENROUTER_APP_NAME
    """
    return BaseMemoryManagerConfig(
        model=_env_str("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=_env_float("LLM_TEMPERATURE", 0.1),
        max_tokens=_env_int("LLM_MAX_TOKENS", 2000),
        top_p=_env_float("LLM_TOP_P", 0.1),
        api_key=_env_str("OPENAI_API_KEY"),
        openai_base_url=_env_str("OPENAI_BASE_URL"),
        openrouter_base_url=_env_str("OPENROUTER_API_BASE"),
        site_url=_env_str("OPENROUTER_SITE_URL"),
        app_name=_env_str("OPENROUTER_APP_NAME"),
    )


# ---------------------------------------------------------------------------
# 4. End-to-end convenience entry point
# ---------------------------------------------------------------------------
def run_from_env(
    raw_input,
    *,
    skip_llm: bool = False,
    dotenv_path: Union[str, Path, None] = None,
) -> PipelineResult:
    """Load `.env` if present, then run the full pipeline with env-driven configs.

    The simplest possible "Python API" entry point - one function, one call,
    everything configured via `.env`:

        from precompress import run_from_env
        result = run_from_env("My long input text...")
        print(result.core_memory_facts)
    """
    load_dotenv_if_present(dotenv_path)
    return run_pipeline(
        raw_input,
        llmlingua_config=llmlingua_config_from_env(),
        manager_config=manager_config_from_env(),
        skip_llm=skip_llm,
    )


__all__ = [
    "load_dotenv_if_present",
    "llmlingua_config_from_env",
    "manager_config_from_env",
    "run_from_env",
]
