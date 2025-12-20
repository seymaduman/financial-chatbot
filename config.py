"""
Configuration settings for RAG Financial Chatbot
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OllamaConfig:
    """Ollama LLM Configuration"""
    model: str = "gpt-oss:120b-cloud"
    host: str = "http://localhost"
    port: int = 11434
    
    # Generation parameters - configurable at runtime
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    max_tokens: int = 2048
    
    @property
    def base_url(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class VectorStoreConfig:
    """ChromaDB configuration"""
    persist_directory: str = "./data/chromadb"
    collection_name: str = "financial_documents"


@dataclass
class ScrapingConfig:
    """Web scraping configuration"""
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    request_timeout: int = 30
    cache_ttl_hours: int = 1  # Cache validity in hours
    cache_directory: str = "./data/cache"


@dataclass
class Config:
    """Main configuration container"""
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    scraping: ScrapingConfig = field(default_factory=ScrapingConfig)
    
    # Application settings
    debug: bool = False
    max_context_documents: int = 5
    
    def update_generation_params(
        self,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """Update generation parameters at runtime"""
        if temperature is not None:
            self.ollama.temperature = temperature
        if top_k is not None:
            self.ollama.top_k = top_k
        if top_p is not None:
            self.ollama.top_p = top_p
        if max_tokens is not None:
            self.ollama.max_tokens = max_tokens


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance"""
    return config
