"""Real OpenAI-compatible API extraction test (gated by OPENAI_API_KEY)."""

from __future__ import annotations

import pytest

from precompress import BaseMemoryManagerConfig, OpenaiManager

from .conftest import needs_openai


@needs_openai
class TestExtractorIntegration:
    def test_real_openai_flat_mode_returns_facts(self, sample_messages):
        """Flat mode: single 'factual' prompt -> list of {source_id, fact} entries."""
        manager = OpenaiManager(BaseMemoryManagerConfig(model="gpt-4o-mini"))
        # meta_text_extract expects: [api_call][topic_segment][message]
        extract_list = [[sample_messages]]
        results = manager.meta_text_extract(
            extract_list=extract_list,
            messages_use="user_only",
            topic_id_mapping=[[1]],
            extraction_mode="flat",
        )

        assert len(results) == 1
        result = results[0]
        assert result is not None
        facts = result.get("cleaned_result", [])
        assert isinstance(facts, list)
        assert facts, "expected at least one fact from the real model"
        assert all(e.get("entry_type") == "factual" for e in facts)

        joined = " ".join(f.get("fact", "") for f in facts).lower()
        assert any(kw in joined for kw in ["alice", "boston", "physics", "paris", "john"])

    def test_real_openai_event_mode_merges_factual_and_relational(self, locomo_messages):
        """Event mode: two prompts (factual + relational), merged via
        `_merge_dual_perspective_results`. Verifies both entry_type values
        appear and that token usage is accumulated across the two calls.
        """
        manager = OpenaiManager(BaseMemoryManagerConfig(model="gpt-4o-mini"))
        extract_list = [[locomo_messages]]
        results = manager.meta_text_extract(
            extract_list=extract_list,
            messages_use="hybrid",
            topic_id_mapping=[[1]],
            extraction_mode="event",
        )

        assert len(results) == 1
        result = results[0]
        assert result is not None

        entries = result.get("cleaned_result", [])
        assert entries, "expected merged factual + relational entries"

        types = {e.get("entry_type") for e in entries}
        # Both perspectives should be represented in the merged list.
        assert "factual" in types
        assert "relational" in types

        # `_merge_dual_perspective_results` sums the per-call token usage.
        usage = result.get("usage", {})
        assert usage.get("total_tokens", 0) > 0
        # Merged output prompt contains both sub-headers.
        assert "Factual:" in result.get("output_prompt", "")
        assert "Relational:" in result.get("output_prompt", "")
