"""
Ollama Client - LLM integration with configurable generation parameters
Supports streaming, retries, and health checks
"""
import os
import json
import time
from typing import Dict, Optional, Generator, Any
from dataclasses import dataclass

import requests

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import get_config


@dataclass
class GenerationConfig:
    """Generation configuration for LLM"""
    temperature: float
    top_k: int
    top_p: float
    max_tokens: int
    
    def to_dict(self) -> Dict:
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "num_predict": self.max_tokens
        }


@dataclass
class GenerationResult:
    """Result from LLM generation"""
    text: str
    model: str
    done: bool
    total_duration: Optional[int] = None
    prompt_tokens: Optional[int] = None
    response_tokens: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "model": self.model,
            "done": self.done,
            "total_duration_ms": self.total_duration // 1_000_000 if self.total_duration else None,
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens
        }


class OllamaClient:
    """
    Ollama API client with configurable generation parameters
    Supports both streaming and non-streaming generation
    """
    
    def __init__(self):
        self.config = get_config()
        self.base_url = self.config.ollama.base_url
        self.model = self.config.ollama.model
        self._session = requests.Session()
        self._is_available = None
    
    def health_check(self) -> bool:
        """
        Check if Ollama server is available
        
        Returns:
            True if server is responsive
        """
        try:
            response = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            self._is_available = response.status_code == 200
            return self._is_available
        except Exception as e:
            print(f"Ollama health check failed: {e}")
            self._is_available = False
            return False
    
    def list_models(self) -> list:
        """
        List available models in Ollama
        
        Returns:
            List of model names
        """
        try:
            response = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        max_tokens: int = None,
        stream: bool = False
    ) -> GenerationResult:
        """
        Generate text using the LLM
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 to 1.0)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            GenerationResult with generated text
        """
        # Use config values as defaults
        gen_config = GenerationConfig(
            temperature=temperature if temperature is not None else self.config.ollama.temperature,
            top_k=top_k if top_k is not None else self.config.ollama.top_k,
            top_p=top_p if top_p is not None else self.config.ollama.top_p,
            max_tokens=max_tokens if max_tokens is not None else self.config.ollama.max_tokens
        )
        
        # Build request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": gen_config.to_dict()
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            if stream:
                return self._generate_stream(payload)
            else:
                return self._generate_sync(payload)
        except Exception as e:
            print(f"Generation error: {e}")
            return GenerationResult(
                text=f"Error: Unable to generate response. {str(e)}",
                model=self.model,
                done=True
            )
    
    def _generate_sync(self, payload: Dict) -> GenerationResult:
        """Synchronous generation"""
        response = self._session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        return GenerationResult(
            text=data.get("response", ""),
            model=data.get("model", self.model),
            done=data.get("done", True),
            total_duration=data.get("total_duration"),
            prompt_tokens=data.get("prompt_eval_count"),
            response_tokens=data.get("eval_count")
        )
    
    def _generate_stream(self, payload: Dict) -> GenerationResult:
        """Streaming generation - collects all chunks"""
        response = self._session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=120
        )
        response.raise_for_status()
        
        full_text = []
        final_data = {}
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "response" in data:
                        full_text.append(data["response"])
                    if data.get("done"):
                        final_data = data
                except json.JSONDecodeError:
                    continue
        
        return GenerationResult(
            text="".join(full_text),
            model=final_data.get("model", self.model),
            done=True,
            total_duration=final_data.get("total_duration"),
            prompt_tokens=final_data.get("prompt_eval_count"),
            response_tokens=final_data.get("eval_count")
        )
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        max_tokens: int = None
    ) -> Generator[str, None, None]:
        """
        Stream text generation token by token
        
        Yields:
            Generated text chunks
        """
        gen_config = GenerationConfig(
            temperature=temperature if temperature is not None else self.config.ollama.temperature,
            top_k=top_k if top_k is not None else self.config.ollama.top_k,
            top_p=top_p if top_p is not None else self.config.ollama.top_p,
            max_tokens=max_tokens if max_tokens is not None else self.config.ollama.max_tokens
        )
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": gen_config.to_dict()
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = self._session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            yield f"\n[Error: {str(e)}]"
    
    def chat(
        self,
        messages: list,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        max_tokens: int = None
    ) -> GenerationResult:
        """
        Chat completion with message history
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            top_k: Top-k parameter
            top_p: Top-p parameter
            max_tokens: Maximum tokens
            
        Returns:
            GenerationResult with assistant response
        """
        gen_config = GenerationConfig(
            temperature=temperature if temperature is not None else self.config.ollama.temperature,
            top_k=top_k if top_k is not None else self.config.ollama.top_k,
            top_p=top_p if top_p is not None else self.config.ollama.top_p,
            max_tokens=max_tokens if max_tokens is not None else self.config.ollama.max_tokens
        )
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": gen_config.to_dict()
        }
        
        try:
            response = self._session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            
            message = data.get("message", {})
            
            return GenerationResult(
                text=message.get("content", ""),
                model=data.get("model", self.model),
                done=data.get("done", True),
                total_duration=data.get("total_duration"),
                prompt_tokens=data.get("prompt_eval_count"),
                response_tokens=data.get("eval_count")
            )
            
        except Exception as e:
            print(f"Chat error: {e}")
            return GenerationResult(
                text=f"Error: Unable to complete chat. {str(e)}",
                model=self.model,
                done=True
            )
    
    def get_generation_params(self) -> Dict:
        """Get current generation parameters"""
        return {
            "model": self.model,
            "temperature": self.config.ollama.temperature,
            "top_k": self.config.ollama.top_k,
            "top_p": self.config.ollama.top_p,
            "max_tokens": self.config.ollama.max_tokens
        }
    
    def update_params(
        self,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        max_tokens: int = None
    ):
        """Update generation parameters at runtime"""
        self.config.update_generation_params(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens
        )


if __name__ == "__main__":
    # Test the Ollama client
    print("Testing Ollama Client...")
    print("-" * 50)
    
    client = OllamaClient()
    
    # Health check
    print(f"\nModel: {client.model}")
    print(f"Base URL: {client.base_url}")
    
    if client.health_check():
        print("✓ Ollama server is available")
        
        # List models
        models = client.list_models()
        print(f"\nAvailable models: {models}")
        
        # Test generation
        print("\n" + "-" * 50)
        print("Testing generation...")
        
        result = client.generate(
            prompt="What is Apple Inc's stock ticker symbol? Answer in one sentence.",
            temperature=0.3,
            max_tokens=50
        )
        
        print(f"\nResponse: {result.text}")
        print(f"Model: {result.model}")
        if result.total_duration:
            print(f"Duration: {result.total_duration // 1_000_000}ms")
        
        # Test streaming
        print("\n" + "-" * 50)
        print("Testing streaming generation...")
        print("Response: ", end="", flush=True)
        
        for chunk in client.generate_stream(
            prompt="List 3 major tech stocks.",
            max_tokens=100
        ):
            print(chunk, end="", flush=True)
        print()
        
    else:
        print("✗ Ollama server is not available")
        print("Make sure Ollama is running: ollama serve")
    
    # Print current params
    print("\n" + "-" * 50)
    print("Current generation parameters:")
    params = client.get_generation_params()
    for key, value in params.items():
        print(f"  {key}: {value}")
