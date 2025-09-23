
from typing import List, Dict, Any

class LLMClient:
    """Plug in your favorite LLM backend here.

    Implement .generate() to return a string for a given prompt.
    You can also implement batched generation if needed.
    """
    def __init__(self, model_name: str = "dummy"):
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> str:
        # TODO: Replace with real API call
        # For now, return a deterministic canned response to keep the scaffold runnable.
        return "DUMMY_MODEL_OUTPUT: " + prompt[:160]
