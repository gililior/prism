
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
    Simplified LLM client supporting OpenAI, Claude, Gemini, and Together AI
    """
    
    def __init__(self, model_type: LLMTypes = LLMTypes.OPENAI, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the LLM client
        
        Args:
            model_type: The type of LLM (openai, claude, gemini, togetherai)
            model_name: Specific model name to use (if None, uses default from LLMModels)
            api_key: API key for the service (if None, loads from environment)
        """
        self.model_type = model_type
        
        # Set default model name if not provided
        if not model_name:
            if model_type == LLMTypes.OPENAI:
                self.model_name = LLMModels.GPT_4O_MINI.value  # Fast and cost-effective
            elif model_type == LLMTypes.GEMINI:
                self.model_name = LLMModels.GEMINI_2_5_FLASH.value  # Fast Gemini model
            elif model_type == LLMTypes.TOGETHERAI:
                self.model_name = LLMModels.LLAMA_3_3_70B.value  # Strong open source model
        else:
            self.model_name = model_name
        
        # Load API key from environment if not provided
        if api_key is None:
            api_key = load_api_key(model_type)
            
        if not api_key:
            raise ValueError(f"No API key found for {model_type.value}. Please set the appropriate environment variable.")
        
        # Initialize the appropriate client
        self._init_client(api_key)
    
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
        
        if self.model_type == LLMTypes.OPENAI:
            return self._generate_openai(messages, system, temperature, max_tokens, **kwargs)
        elif self.model_type == LLMTypes.GEMINI:
            return self._generate_gemini(messages, system, temperature, max_tokens, **kwargs)
        elif self.model_type == LLMTypes.TOGETHERAI:
            return self._generate_together(messages, system, temperature, max_tokens, **kwargs)
    
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
    
    def _generate_openai(self, messages: List[Dict[str, Any]], system: Optional[str], temperature: float, max_tokens: Optional[int], **kwargs) -> str:
        """Generate using OpenAI API"""
        # Build messages list and parameters
        openai_messages = ([{"role": "system", "content": system}] if system else []) + messages
        
        params = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        params.update(kwargs)
            
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content
    
    def _generate_gemini(self, messages: List[Dict[str, Any]], system: Optional[str], temperature: float, max_tokens: Optional[int], **kwargs) -> str:
        """Generate using Gemini API"""
        # Convert messages to Gemini format - simplified
        gemini_contents = [
            {"role": "model" if msg["role"] == "assistant" else msg["role"], 
             "parts": [{"text": msg["content"]}]}
            for msg in messages
        ]
        
        # Build config parameters
        config_params = {"temperature": temperature}
        if system:
            config_params["system_instruction"] = system
        if max_tokens is not None:
            config_params["max_output_tokens"] = max_tokens
        config_params.update(kwargs)
        
        config = types.GenerateContentConfig(**config_params)
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=gemini_contents,
            config=config
        )
        return response.text
    
    def _generate_together(self, messages: List[Dict[str, Any]], system: Optional[str], temperature: float, max_tokens: Optional[int], **kwargs) -> str:
        """Generate using Together AI API"""
        # Build messages list
        together_messages = ([{"role": "system", "content": system}] if system else []) + messages
        
        # Build parameters
        params = {
            "model": self.model_name,
            "messages": together_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,  # Default for Together AI
        }
        params.update(kwargs)
        
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content
