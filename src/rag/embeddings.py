"""
Embedding Service - Text embeddings using sentence-transformers
Handles document chunking and batch embedding generation
"""
import os
from typing import List, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import get_config


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    content: str
    chunk_id: str
    source_id: str
    source_type: str  # stock, news, statement
    metadata: dict


class EmbeddingService:
    """
    Text embedding service using sentence-transformers
    Provides chunking and embedding generation for RAG
    """
    
    def __init__(self):
        self.config = get_config()
        self._model = None
        self._embedding_cache = {}
    
    @property
    def model(self):
        """Lazy load the embedding model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.config.embedding.model_name
                print(f"Loading embedding model: {model_name}")
                self._model = SentenceTransformer(model_name)
                print("Embedding model loaded successfully")
            except ImportError:
                print("Warning: sentence-transformers not installed. Using fallback embeddings.")
                self._model = "fallback"
        return self._model
    
    def chunk_text(
        self, 
        text: str, 
        source_id: str,
        source_type: str,
        metadata: dict = None
    ) -> List[TextChunk]:
        """
        Split text into overlapping chunks for embedding
        
        Args:
            text: Text to chunk
            source_id: Unique identifier for the source document
            source_type: Type of source (stock, news, statement)
            metadata: Additional metadata to attach to chunks
            
        Returns:
            List of TextChunk objects
        """
        if not text:
            return []
        
        chunk_size = self.config.embedding.chunk_size
        overlap = self.config.embedding.chunk_overlap
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size
            if current_length + sentence_length > chunk_size and current_chunk:
                # Create chunk
                chunk_content = " ".join(current_chunk)
                chunk_id = f"{source_id}_chunk_{chunk_index}"
                
                chunks.append(TextChunk(
                    content=chunk_content,
                    chunk_id=chunk_id,
                    source_id=source_id,
                    source_type=source_type,
                    metadata=metadata or {}
                ))
                
                chunk_index += 1
                
                # Keep overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            chunk_id = f"{source_id}_chunk_{chunk_index}"
            
            chunks.append(TextChunk(
                content=chunk_content,
                chunk_id=chunk_id,
                source_id=source_id,
                source_type=source_type,
                metadata=metadata or {}
            ))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        
        # Simple sentence splitting
        # Handle common abbreviations
        text = text.replace("Mr.", "Mr").replace("Mrs.", "Mrs").replace("Dr.", "Dr")
        text = text.replace("Inc.", "Inc").replace("Corp.", "Corp").replace("Ltd.", "Ltd")
        text = text.replace("vs.", "vs").replace("etc.", "etc")
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean up and restore abbreviations
        cleaned = []
        for s in sentences:
            s = s.strip()
            if s:
                s = s.replace("Mr", "Mr.").replace("Mrs", "Mrs.").replace("Dr", "Dr.")
                s = s.replace("Inc", "Inc.").replace("Corp", "Corp.").replace("Ltd", "Ltd.")
                s = s.replace("vs", "vs.").replace("etc", "etc.")
                cleaned.append(s)
        
        return cleaned
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not text:
            return []
        
        # Check cache
        cache_key = hash(text)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        if self.model == "fallback":
            # Simple fallback: use hash-based pseudo-embeddings
            embedding = self._fallback_embedding(text)
        else:
            embedding = self.model.encode(text, convert_to_numpy=True).tolist()
        
        # Cache the result
        self._embedding_cache[cache_key] = embedding
        
        return embedding
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing)
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        if self.model == "fallback":
            return [self._fallback_embedding(t) for t in texts]
        
        embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
        return embeddings
    
    def embed_chunks(self, chunks: List[TextChunk]) -> List[tuple]:
        """
        Embed a list of text chunks
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            List of (chunk, embedding) tuples
        """
        if not chunks:
            return []
        
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_texts(texts)
        
        return list(zip(chunks, embeddings))
    
    def _fallback_embedding(self, text: str, dim: int = 384) -> List[float]:
        """
        Generate a simple fallback embedding when sentence-transformers is not available
        This is NOT suitable for production - just a placeholder
        """
        import hashlib
        import math
        
        # Create a deterministic embedding based on text content
        hash_bytes = hashlib.sha384(text.encode()).digest()
        
        # Convert bytes to floats in range [-1, 1]
        embedding = []
        for i in range(dim):
            byte_val = hash_bytes[i % len(hash_bytes)]
            float_val = (byte_val / 255.0) * 2 - 1
            embedding.append(float_val)
        
        # Normalize
        magnitude = math.sqrt(sum(x*x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        import math
        
        if not embedding1 or not embedding2:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = math.sqrt(sum(a * a for a in embedding1))
        magnitude2 = math.sqrt(sum(b * b for b in embedding2))
        
        if magnitude1 * magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings"""
        if self.model == "fallback":
            return 384
        
        # Get dimension from model
        test_embedding = self.embed_text("test")
        return len(test_embedding)
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self._embedding_cache.clear()


if __name__ == "__main__":
    # Test the embedding service
    service = EmbeddingService()
    
    print("Testing Embedding Service...")
    print("-" * 50)
    
    # Test text chunking
    sample_text = """
    Apple Inc. reported strong earnings this quarter, beating analyst expectations. 
    Revenue grew 15% year-over-year to $95 billion. The company announced a new 
    dividend increase and stock buyback program worth $100 billion. iPhone sales 
    remained strong despite global supply chain challenges. CEO Tim Cook expressed 
    optimism about the company's future growth prospects in emerging markets.
    The stock price surged 5% in after-hours trading following the announcement.
    """
    
    chunks = service.chunk_text(
        sample_text, 
        source_id="test_doc",
        source_type="news",
        metadata={"ticker": "AAPL"}
    )
    
    print(f"Chunked text into {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"  {chunk.chunk_id}: {len(chunk.content)} chars")
    
    print("\n" + "-" * 50)
    
    # Test embedding generation
    print("\nGenerating embeddings...")
    embedding = service.embed_text("Apple stock price increased")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test similarity
    emb1 = service.embed_text("Apple stock price went up")
    emb2 = service.embed_text("AAPL shares increased in value")
    emb3 = service.embed_text("Microsoft released new software")
    
    sim_same = service.similarity(emb1, emb2)
    sim_diff = service.similarity(emb1, emb3)
    
    print(f"\nSimilarity (Apple up vs AAPL increased): {sim_same:.3f}")
    print(f"Similarity (Apple up vs Microsoft software): {sim_diff:.3f}")
