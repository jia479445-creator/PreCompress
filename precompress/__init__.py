from .compressor import LlmLingua2Compressor, LlmLingua2Config
from .env import (
    llmlingua_config_from_env,
    load_dotenv_if_present,
    manager_config_from_env,
    run_from_env,
)
from .extractor import BaseMemoryManagerConfig, OpenaiManager
from .longmemeval import (
    flatten_to_messages,
    iter_sessions,
    load_longmemeval,
    pick_sample,
    resolve_data_path,
)
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
    # LongMemEval data loader
    "load_longmemeval",
    "resolve_data_path",
    "iter_sessions",
    "flatten_to_messages",
    "pick_sample",
]

__version__ = "0.4.0"
