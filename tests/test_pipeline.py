"""End-to-end real pipeline test (gated by RUN_INTEGRATION=1 + OPENAI_API_KEY)."""

from __future__ import annotations

import pytest

from precompress import BaseMemoryManagerConfig, LlmLingua2Config, run_pipeline

from .conftest import needs_llmlingua, needs_openai


@needs_llmlingua
@needs_openai
class TestPipelineIntegration:
    def test_full_pipeline_alice_dialogue(self, sample_messages):
        llm_cfg = LlmLingua2Config(
            llmlingua_config={
                "model_name": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                "device_map": "cpu",
                "use_llmlingua2": True,
            },
            compress_config={"instruction": "", "rate": 0.5, "target_token": -1},
        )
        mgr_cfg = BaseMemoryManagerConfig(model="gpt-4o-mini")

        result = run_pipeline(
            sample_messages,
            llmlingua_config=llm_cfg,
            manager_config=mgr_cfg,
        )

        assert result.tokens_after <= result.tokens_before
        assert result.core_memory_facts, "expected facts from real model"
        joined = " ".join(f.get("fact", "") for f in result.core_memory_facts).lower()
        assert any(kw in joined for kw in ["alice", "boston", "physics", "paris", "john"])
