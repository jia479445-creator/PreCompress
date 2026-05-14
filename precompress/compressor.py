"""Near-verbatim port of `src/lightmem/factory/pre_compressor/llmlingua_2.py`.

The only changes vs. the original LightMem source are:
    * `LlmLingua2Config` (originally in
      `src/lightmem/configs/pre_compressor/llmlingua_2.py`) is inlined at the top
      of this file so the module does not depend on the `lightmem.*` package.
"""

import os
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, field_validator
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Inlined from src/lightmem/configs/pre_compressor/llmlingua_2.py
# ---------------------------------------------------------------------------
class LlmLingua2Config(BaseModel):
    llmlingua_config: Dict[str, Any] = Field(
        default={
            "model_name": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            "device_map": "cuda",
            "use_llmlingua2": True,
        },
        description="Configuration for LLMLingua, including model name, device, and whether to use LLMLingua-2."
    )

    llmlingua2_config: Dict[str, Any] = Field(
        default={
            "max_batch_size": 50,
            "max_force_token": 100,
        },
        description="Advanced configuration for LLMLingua-2 (batch size, token control)"
    )

    compress_config: Dict[str, Any] = Field(
        default={
            "instruction": "",
            "rate": 0.8,
            "target_token": -1
        },
        description="Additional instruction text to be included in the prompt, The maximum compression rate, "
    )

    @field_validator("llmlingua_config")
    @classmethod
    def validate_llmlingua_config(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        allowed_models = [
            "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            "NousResearch/Llama-2-7b-hf",
            None
        ]
        model_name = v.get("model_name")

        if model_name is not None:
            if model_name not in allowed_models and not os.path.exists(model_name):
                raise ValueError(
                    f"model_name must be one of {allowed_models} "
                    f"or a valid local path (got {model_name})"
                )

        if "use_llmlingua2" in v and not isinstance(v["use_llmlingua2"], bool):
            raise ValueError("use_llmlingua2 must be a boolean")

        return v

    @field_validator("llmlingua2_config")
    @classmethod
    def validate_llmlingua2_config(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(v.get("max_batch_size"), int) or v["max_batch_size"] <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        if not isinstance(v.get("max_force_token"), int) or v["max_force_token"] <= 0:
            raise ValueError("max_force_token must be a positive integer")
        return v


# ---------------------------------------------------------------------------
# Compressor (verbatim from src/lightmem/factory/pre_compressor/llmlingua_2.py)
# ---------------------------------------------------------------------------
class LlmLingua2Compressor:
    def __init__(self, config: Optional[LlmLingua2Config] = None):
        self.config = config
        self._disable_runtime_compression = False
        self._normalization_logged = False
        self._disable_logged = False

        try:
            import importlib
            importlib.import_module('llmlingua')
        except ImportError:
            raise ImportError(
                "Required package 'llmlingua' not found. "
                "Please install it with: pip install llmlingua\n"
                "Or for the latest version: pip install git+https://github.com/microsoft/LLMLingua.git"
            )

        try:
            from llmlingua import PromptCompressor
            if config.llmlingua_config['use_llmlingua2'] is True:
                self._compressor = PromptCompressor(
                    model_name=config.llmlingua_config['model_name'],
                    device_map=config.llmlingua_config['device_map'],
                    use_llmlingua2=config.llmlingua_config['use_llmlingua2'],
                    llmlingua2_config=config.llmlingua2_config
                )
            else:
                self._compressor = PromptCompressor(
                    model_name=config.llmlingua_config['model_name'],
                    device_map=config.llmlingua_config['device_map']
                )
            # LLMLingua-2 internals branch on compressor.model_name substring and may
            # raise NotImplementedError for local path aliases (e.g. "/root/.../llmlingua-2").
            self._normalize_llmlingua_model_name()
            # Expose tokenizer for downstream callers (e.g., LightMemory.add_memory).
            self.tokenizer = getattr(self._compressor, "tokenizer", None)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LlmLingua2Compressor: {str(e)}")

    def _normalize_llmlingua_model_name(self) -> None:
        compressor_name = str(getattr(self._compressor, "model_name", "") or "")
        lowered = compressor_name.lower()
        if "bert-base-multilingual-cased" in lowered or "xlm-roberta-large" in lowered:
            return

        model_type = str(
            getattr(getattr(getattr(self._compressor, "model", None), "config", None), "model_type", "") or ""
        ).lower()
        tokenizer_name = str(getattr(getattr(self._compressor, "tokenizer", None), "name_or_path", "") or "").lower()
        hint = f"{lowered} {model_type} {tokenizer_name}"

        if "xlm" in hint or "roberta" in hint:
            canonical_name = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
        else:
            canonical_name = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

        setattr(self._compressor, "model_name", canonical_name)
        if not self._normalization_logged:
            print(
                "llmlingua model_name normalized for local path compatibility: "
                f"{compressor_name} -> {canonical_name}"
            )
            self._normalization_logged = True

    @staticmethod
    def _safe_encode(tokenizer: Any, text: str) -> List[int]:
        """Best-effort tokenization that works across HF/tiktoken/custom tokenizers."""
        if tokenizer is None:
            return list(range(len(text)))

        try:
            return tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            return tokenizer.encode(text)
        except Exception:
            try:
                encoded = tokenizer(
                    text,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                input_ids = encoded.get("input_ids", [])
                if input_ids and isinstance(input_ids[0], list):
                    return input_ids[0]
                return input_ids
            except Exception:
                return list(range(len(text)))

    @staticmethod
    def _safe_decode(tokenizer: Any, token_ids: List[int]) -> str:
        if tokenizer is None or not hasattr(tokenizer, "decode"):
            return ""
        try:
            return tokenizer.decode(token_ids, skip_special_tokens=True)
        except TypeError:
            return tokenizer.decode(token_ids)
        except Exception:
            return ""

    def _resolve_token_budget(self, tokenizer: Any) -> int:
        """Infer safe max length from model/tokenizer; fallback to 512 for llmlingua-2."""
        fallback_limit = 512
        candidates: List[int] = []

        model = getattr(self._compressor, "model", None)
        model_cfg = getattr(model, "config", None)
        model_limit = getattr(model_cfg, "max_position_embeddings", None)
        if isinstance(model_limit, int) and model_limit > 0:
            candidates.append(model_limit)

        tok_limit = getattr(tokenizer, "model_max_length", None)
        if isinstance(tok_limit, int) and 0 < tok_limit < 10**8:
            candidates.append(tok_limit)

        return min(candidates) if candidates else fallback_limit

    def _truncate_to_limit(self, text: str, tokenizer: Any, limit: int) -> str:
        if not text or limit <= 0:
            return text

        token_ids = self._safe_encode(tokenizer, text)
        if len(token_ids) <= limit:
            return text

        decoded = self._safe_decode(tokenizer, token_ids[:limit]).strip()
        if decoded:
            return decoded

        ratio = max(0.05, min(1.0, float(limit) / max(1, len(token_ids))))
        return text[: max(1, int(len(text) * ratio))]

    def _split_for_compression(self, text: str, tokenizer: Any, limit: int) -> List[str]:
        """Split overlong content into token-bounded chunks for llmlingua-2."""
        if not text:
            return []
        if tokenizer is None:
            return [text]

        token_ids = self._safe_encode(tokenizer, text)
        if len(token_ids) <= limit:
            return [text]

        chunks: List[str] = []
        for start in range(0, len(token_ids), limit):
            chunk_ids = token_ids[start : start + limit]
            chunk_text = self._safe_decode(tokenizer, chunk_ids).strip()
            if chunk_text:
                chunks.append(chunk_text)

        if chunks:
            return chunks

        approx_chars = max(1, int(len(text) * limit / max(1, len(token_ids))))
        return [text[i : i + approx_chars] for i in range(0, len(text), approx_chars)]

    def compress(
        self,
        messages: List[Dict[str, str]],
        tokenizer: Union[PreTrainedTokenizerBase, Any, None],
    ) -> List[Dict[str, str]]:
        # TODO: Consider adding an extra field in the message, compressed_content, and put the compressed content in this field while keeping content unchanged.
        """
        Compress the content of each message.

        Args:
            messages: List of message dicts containing 'role' and 'content'.
            tokenizer: Tokenizer to check token length after compression.

        Returns:
            List of messages with compressed content.
        """
        active_tokenizer = tokenizer or getattr(self, "tokenizer", None) or getattr(self._compressor, "tokenizer", None)
        token_budget = self._resolve_token_budget(active_tokenizer)
        # Keep a small margin for special tokens.
        split_budget = max(32, token_budget - 8)

        for mes in messages:
            content = mes.get('content', '')
            if not content or not content.strip():
                # If content is empty, it doesn't need compression
                continue

            chunks = self._split_for_compression(content, active_tokenizer, split_budget)
            compressed_chunks: List[str] = []

            for chunk in chunks:
                compress_config = {
                    'context': [chunk],
                    **self.config.compress_config
                }
                if self._disable_runtime_compression:
                    comp_content = chunk
                else:
                    try:
                        comp_content = self._compressor.compress_prompt(**compress_config)['compressed_prompt']
                    except NotImplementedError:
                        # Retry once after normalizing runtime model name for local-path model ids.
                        self._normalize_llmlingua_model_name()
                        try:
                            comp_content = self._compressor.compress_prompt(**compress_config)['compressed_prompt']
                        except NotImplementedError as e:
                            self._disable_runtime_compression = True
                            if not self._disable_logged:
                                print(
                                    "compress disabled for current run: "
                                    f"{type(e).__name__}. falling back to non-compressed content."
                                )
                                self._disable_logged = True
                            comp_content = chunk
                    except Exception as e:
                        print(f"compress error, skip this message chunk: {type(e).__name__}: {e}")
                        comp_content = chunk

                # Secondary compression retries for overlong result.
                if active_tokenizer is not None and not self._disable_runtime_compression:
                    try:
                        retry_count = 0
                        while (
                            len(self._safe_encode(active_tokenizer, comp_content)) >= token_budget
                            and comp_content.strip()
                            and retry_count < 4
                        ):
                            retry_count += 1
                            new_compress_config = {
                                'context': [comp_content],  # NOTE: must be a list for llmlingua-2
                                **self.config.compress_config
                            }
                            next_comp = self._compressor.compress_prompt(**new_compress_config)['compressed_prompt']
                            if next_comp.strip() == comp_content.strip():
                                break
                            comp_content = next_comp
                    except Exception as e:
                        print(f"secondary compress error: {type(e).__name__}: {e}")

                    # Last-resort truncation to avoid downstream index errors.
                    if len(self._safe_encode(active_tokenizer, comp_content)) >= token_budget:
                        comp_content = self._truncate_to_limit(comp_content, active_tokenizer, split_budget)

                if comp_content.strip():
                    compressed_chunks.append(comp_content.strip())

            merged = "\n".join(compressed_chunks).strip() if compressed_chunks else content.strip()
            if active_tokenizer is not None and len(self._safe_encode(active_tokenizer, merged)) >= token_budget:
                merged = self._truncate_to_limit(merged, active_tokenizer, split_budget)
            if merged:
                mes['content'] = merged

        return messages

    @property
    def inner_compressor(self):
        """
        Access the underlying PromptCompressor instance directly.
        """
        return self._compressor
