from enum import Enum
from dataclasses import dataclass
from typing import Optional


class LLMTypes(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    TOGETHERAI = "togetherai"


class LLMModels(Enum):
    # OpenAI models
    GPT_4O_MINI = "gpt-4o-mini"

    # Gemini models
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"

    # Together AI models (popular open source models)
    LLAMA_3_3_70B = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"


@dataclass
class TaskLLMConfig:
    """LLM configuration for a specific task"""
    temperature: float
    max_tokens: Optional[int]


class TaskLLMConfigs:
    """LLM configurations for different tasks"""
    
    # Author rebuttal generation
    AUTHOR_REBUTTAL = TaskLLMConfig(
        temperature=0.2,  # Default temperature for creative but controlled rebuttal
        max_tokens=3000
    )
    
    # Leader merge points task
    LEADER_MERGE = TaskLLMConfig(
        temperature=0.1,  # Low temperature for consistent merging
        max_tokens=10000
    )
    
    # Leader update with rebuttals task
    LEADER_UPDATE_REBUTTALS = TaskLLMConfig(
        temperature=0.1,  # Low temperature for consistent updates
        max_tokens=3000
    )
    
    # Base reviewer task (for all reviewer agents)
    REVIEWER_BASE = TaskLLMConfig(
        temperature=0.2,  # Default temperature for review generation
        max_tokens=None   # Use model default
    )
    
    # Verifier check task
    VERIFIER_CHECK = TaskLLMConfig(
        temperature=0.0,  # Very low temperature for binary verification
        max_tokens=50
    )
    
    # Related work reviewer (specific configuration)
    REVIEWER_RELATED = TaskLLMConfig(
        temperature=0.2,  # Standard review temperature
        max_tokens=None
    )