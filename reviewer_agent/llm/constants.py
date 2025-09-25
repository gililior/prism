from enum import Enum


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