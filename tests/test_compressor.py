"""Real LLMLingua-2 compression test (gated by RUN_INTEGRATION=1).

The first run will download the model weights
(~200 MB for the bert-base-multilingual checkpoint).
"""

from __future__ import annotations

import pytest

from precompress import LlmLingua2Compressor, LlmLingua2Config

from .conftest import needs_llmlingua


@needs_llmlingua
class TestCompressorIntegration:
    def test_real_model_shortens_text(self):
        cfg = LlmLingua2Config(
            llmlingua_config={
                "model_name": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                "device_map": "cpu",
                "use_llmlingua2": True,
            },
            compress_config={"instruction": "", "rate": 0.4, "target_token": -1},
        )
        compressor = LlmLingua2Compressor(cfg)

        messages = [
            {
                "role": "user",
                "content": (
                    "My name is Alice and I work as a high-school physics teacher in "
                    "Boston. I have been teaching for about eight years and recently "
                    "started a part-time master's degree in education. I traveled to "
                    "Paris last summer with my friend John."
                ),
            }
        ]
        original_len = len(
            LlmLingua2Compressor._safe_encode(compressor.tokenizer, messages[0]["content"])
        )
        compressed = compressor.compress([dict(m) for m in messages], compressor.tokenizer)

        assert compressed[0]["content"].strip() != ""
        compressed_len = len(
            LlmLingua2Compressor._safe_encode(compressor.tokenizer, compressed[0]["content"])
        )
        assert compressed_len < original_len
