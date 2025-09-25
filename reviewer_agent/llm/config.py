import os
from typing import Optional
from dotenv import load_dotenv

from .constants import LLMTypes

# Load environment variables from .env file
load_dotenv()


def load_api_key(model_type: LLMTypes) -> Optional[str]:
    """
    Load API key from environment variables based on model type
    
    Args:
        model_type: The type of LLM (openai, gemini, togetherai)
        
    Returns:
        API key string if found in environment, None otherwise
    """
    env_var_map = {
        LLMTypes.OPENAI: "OPENAI_API_KEY",
        LLMTypes.GEMINI: "GOOGLE_API_KEY",
        LLMTypes.TOGETHERAI: "TOGETHER_API_KEY"
    }
    
    env_var = env_var_map.get(model_type)
    if env_var:
        return os.getenv(env_var)
    return None
