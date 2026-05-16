from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from .compressor import LlmLingua2Compressor, LlmLingua2Config
from .extractor import BaseMemoryManagerConfig, OpenaiManager


@dataclass
class PipelineResult:
    original_messages: List[Dict[str, Any]]
    compressed_messages: List[Dict[str, Any]]
    tokens_before: int
    tokens_after: int
    compression_ratio: float
    core_memory_facts: Optional[List[Dict[str, Any]]] = None
    extraction_raw: Optional[Dict[str, Any]] = None


def _normalize_input(
    raw_input: Union[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    if isinstance(raw_input, str):
        return [{"role": "user", "content": raw_input, "sequence_number": 0}]
    msgs = [dict(m) for m in raw_input]
    for i, m in enumerate(msgs):
        m.setdefault("sequence_number", i * 2)  # mirror original even-stride numbering
        m.setdefault("role", "user")
    return msgs


def _count_tokens(tokenizer: Any, text: str) -> int:
    return len(LlmLingua2Compressor._safe_encode(tokenizer, text))


def run_pipeline(
    raw_input: Union[str, List[Dict[str, Any]]],
    *,
    llmlingua_config: Optional[LlmLingua2Config] = None,
    manager_config: Optional[BaseMemoryManagerConfig] = None,
    skip_llm: bool = False,
) -> PipelineResult:
    """Compress raw text with LLMLingua-2, then extract core-memory facts with the LLM."""
    compressor = LlmLingua2Compressor(llmlingua_config or LlmLingua2Config())

    messages = _normalize_input(raw_input)
    original_messages = [dict(m) for m in messages]
    compressed_messages = compressor.compress(
        [dict(m) for m in messages], compressor.tokenizer
    )

    tok = compressor.tokenizer
    tokens_before = sum(_count_tokens(tok, m.get("content", "")) for m in original_messages)
    tokens_after = sum(_count_tokens(tok, m.get("content", "")) for m in compressed_messages)
    ratio = (tokens_after / tokens_before) if tokens_before else 0.0

    result = PipelineResult(
        original_messages=original_messages,
        compressed_messages=compressed_messages,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        compression_ratio=ratio,
    )

    if skip_llm:
        return result

    manager = OpenaiManager(manager_config or BaseMemoryManagerConfig(model="gpt-4o-mini"))
    extract_list = [[compressed_messages]]
    extracted = manager.meta_text_extract(
        extract_list=extract_list,
        messages_use="user_only",
        topic_id_mapping=[[1]],
        extraction_mode="flat",
    )
    first = extracted[0] if extracted else None
    if first is not None:
        result.core_memory_facts = first.get("cleaned_result", [])
        result.extraction_raw = first
    return result
