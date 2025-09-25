
import json
import re
from typing import List, Dict, Any, Optional

# Import all LLM clients at the top
import openai
from google import genai
from google.genai import types
from together import Together

from .constants import LLMTypes, LLMModels
from .config import load_api_key


class LLMClient:
    """
    LLM client supporting OpenAI, Gemini, and Together AI with proper provider-specific handling
    """
    
    def __init__(self, model_name: str, model_type: Optional[LLMTypes] = None, api_key: Optional[str] = None):
        """
        Initialize the LLM client
        
        Args:
            model_name: Specific model name to use (e.g., "gpt-4o-mini", "gemini-2.5-flash-lite", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
            model_type: LLM provider type (if None, auto-detects from model name)
            api_key: API key for the service (if None, loads from environment)
        """
        self.model_name = model_name
        self.model_type = model_type or self._get_model_type_from_name(model_name)
        
        # Load API key from environment if not provided
        if api_key is None:
            api_key = load_api_key(self.model_type)
            
        if not api_key:
            raise ValueError(f"No API key found for {self.model_type.value}. Please set the appropriate environment variable.")
        
        # Initialize the appropriate client
        self._init_client(api_key)
    
    def _get_model_type_from_name(self, model_name: str) -> LLMTypes:
        """Determine the LLM provider from the model name string."""
        model_name_lower = model_name.lower()
        if "gemini" in model_name_lower:
            return LLMTypes.GEMINI
        elif "gpt" in model_name_lower or model_name_lower.startswith("o1"):
            return LLMTypes.OPENAI
        elif any(x in model_name_lower for x in ["llama", "meta-llama", "mistral", "qwen", "mixtral"]):
            return LLMTypes.TOGETHERAI
        else:
            # Default to OpenAI for unknown models
            return LLMTypes.OPENAI
    
    def _init_client(self, api_key: str):
        """Initialize the specific LLM client based on model type"""
        if self.model_type == LLMTypes.OPENAI:
            self.client = openai.OpenAI(api_key=api_key)
        elif self.model_type == LLMTypes.GEMINI:
            self.client = genai.Client(api_key=api_key)
        elif self.model_type == LLMTypes.TOGETHERAI:
            self.client = Together(api_key=api_key)
    
    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.2, max_tokens: Optional[int] = None, **kwargs) -> str:
        """
        Generate completion and return raw text response
        
        Args:
            prompt: The user prompt
            system: Optional system message
            temperature: Temperature for generation (0.0-2.0, default: 0.2)
            max_tokens: Maximum tokens to generate (None for model default)
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        # Convert simple prompt to messages format
        messages = [{"role": "user", "content": prompt}]
        return self.generate_with_messages(messages, system, temperature, max_tokens, **kwargs)
    
    def generate_with_messages(self, messages: List[Dict[str, Any]], system: Optional[str] = None, temperature: float = 0.2, max_tokens: Optional[int] = None, **kwargs) -> str:
        """
        Generate completion with full messages format
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system: Optional system message
            temperature: Temperature for generation (0.0-2.0, default: 0.2)
            max_tokens: Maximum tokens to generate (None for model default)
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        if self.model_type == LLMTypes.OPENAI:
            return self._generate_openai(messages, system, temperature, max_tokens, **kwargs)
        elif self.model_type == LLMTypes.GEMINI:
            return self._generate_gemini(messages, system, temperature, max_tokens, **kwargs)
        elif self.model_type == LLMTypes.TOGETHERAI:
            return self._generate_together(messages, system, temperature, max_tokens, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _generate_openai(self, messages: List[Dict[str, Any]], system: Optional[str], temperature: float, max_tokens: Optional[int], **kwargs) -> str:
        """Generate using OpenAI API"""
        if system:
            messages = [{"role": "system", "content": system}] + messages
            
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content
    
    def _generate_gemini(self, messages: List[Dict[str, Any]], system: Optional[str], temperature: float, max_tokens: Optional[int], **kwargs) -> str:
        """Generate using Gemini API"""
        # Convert messages to Gemini format
        gemini_contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            gemini_contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        # Create a GenerateContentConfig for parameters and system instructions
        config_params = {"temperature": temperature,
                         "response_mime_type": "application/json",}
        if system:
            config_params["system_instruction"] = system
        if max_tokens is not None:
            config_params["max_output_tokens"] = max_tokens
        
        # Add any additional kwargs to config
        for key, value in kwargs.items():
            if key not in config_params:
                config_params[key] = value
        
        config = types.GenerateContentConfig(**config_params)

        # Use the client.models.generate_content call
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=gemini_contents,
            config=config
        )
        return response.text
    
    def _generate_together(self, messages: List[Dict[str, Any]], system: Optional[str], temperature: float, max_tokens: Optional[int], **kwargs) -> str:
        """Generate using Together AI API"""
        # Prepare messages for Together AI
        together_messages = []
        
        # Add system message if provided
        if system:
            together_messages.append({"role": "system", "content": system})

        # Add the rest of the messages
        together_messages.extend(messages)

        # Call the Together AI API
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_tokens or 8193,
            messages=together_messages,
            temperature=temperature,
            **kwargs
        )

        return response.choices[0].message.content
