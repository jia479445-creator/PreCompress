"""precompress: standalone LLMLingua-2 pre-compression + LLM core-memory extractor.

Source mapping (LightMem repo paths) - kept as close to verbatim as possible:
    compressor.py  <- src/lightmem/factory/pre_compressor/llmlingua_2.py
                      + src/lightmem/configs/pre_compressor/llmlingua_2.py
    extractor.py   <- src/lightmem/factory/memory_manager/openai.py
                      + src/lightmem/configs/memory_manager/base_config.py
    prompts.py     <- src/lightmem/memory/prompts.py
    utils.py       <- src/lightmem/memory/utils.py (clean_response only)
    pipeline.py    -- thin orchestrator (mirrors LightMemory.add_memory's
                      pre-compression -> meta_text_extract slice)
    env.py         -- .env loader + config factories (this layer is new)

On import, this package automatically loads `.env` from the current working
directory (walking up to the filesystem root). Real OS environment variables
always win over the file.
"""

from .compressor import LlmLingua2Compressor, LlmLingua2Config
from .env import (
    llmlingua_config_from_env,
    load_dotenv_if_present,
    manager_config_from_env,
    run_from_env,
)
from .extractor import BaseMemoryManagerConfig, OpenaiManager
from .pipeline import PipelineResult, run_pipeline
from .prompts import (
    EXTRACTION_PROMPTS,
    LoCoMo_Cross_Event_Consolidation,
    LoCoMo_Event_Binding_factual,
    LoCoMo_Event_Binding_relational,
    METADATA_GENERATE_PROMPT,
    METADATA_GENERATE_PROMPT_locomo,
    UPDATE_PROMPT,
)
from .utils import clean_response

# Auto-load .env on first import so simple scripts "just work".
load_dotenv_if_present()

__all__ = [
    # core API
    "LlmLingua2Compressor",
    "LlmLingua2Config",
    "OpenaiManager",
    "BaseMemoryManagerConfig",
    "PipelineResult",
    "run_pipeline",
    # env-driven convenience layer
    "load_dotenv_if_present",
    "llmlingua_config_from_env",
    "manager_config_from_env",
    "run_from_env",
    # prompts + utils
    "METADATA_GENERATE_PROMPT",
    "METADATA_GENERATE_PROMPT_locomo",
    "LoCoMo_Event_Binding_factual",
    "LoCoMo_Event_Binding_relational",
    "LoCoMo_Cross_Event_Consolidation",
    "UPDATE_PROMPT",
    "EXTRACTION_PROMPTS",
    "clean_response",
]

__version__ = "0.3.0"
