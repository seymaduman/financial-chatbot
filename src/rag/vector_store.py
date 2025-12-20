"""
Vector Store - ChromaDB operations for document storage and retrieval
Manages embeddings and metadata for RAG pipeline
"""
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import get_config
from src.rag.embeddings import TextChunk


@dataclass
class SearchResult:
    """Search result with relevance score"""
    chunk_id: str
    content: str
    source_id: str
    source_type: str
    metadata: Dict
    score: float  # Similarity score


class VectorStore:
    """
    ChromaDB vector store for document embeddings
    Supports document storage, updating, and similarity search
    """
    
    def __init__(self):
        self.config = get_config()
        self._client = None
        self._collection = None
        
        # Ensure persist directory exists
        os.makedirs(self.config.vector_store.persist_directory, exist_ok=True)
    
    @property
    def client(self):
        """Lazy load ChromaDB client"""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                self._client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=self.config.vector_store.persist_directory,
                    anonymized_telemetry=False
                ))
                print("ChromaDB client initialized")
            except ImportError:
                print("Warning: chromadb not installed. Using in-memory fallback.")
                self._client = "fallback"
            except Exception as e:
                print(f"Warning: ChromaDB initialization failed: {e}. Using fallback.")
                self._client = "fallback"
        
        return self._client
    
    @property
    def collection(self):
        """Get or create the main collection"""
        if self._collection is None:
            if self.client == "fallback":
                self._collection = FallbackCollection()
            else:
                self._collection = self.client.get_or_create_collection(
                    name=self.config.vector_store.collection_name,
                    metadata={"description": "Financial documents for RAG"}
                )
        return self._collection
    
    def add_documents(
        self, 
        chunks: List[TextChunk],
        embeddings: List[List[float]]
    ) -> int:
        """
        Add documents to the vector store
        
        Args:
            chunks: List of TextChunk objects
            embeddings: List of embedding vectors
            
        Returns:
            Number of documents added
        """
        if not chunks or not embeddings:
            return 0
        
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "source_id": chunk.source_id,
                "source_type": chunk.source_type,
                **chunk.metadata
            }
            for chunk in chunks
        ]
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            return len(chunks)
        except Exception as e:
            print(f"Error adding documents: {e}")
            return 0
    
    def upsert_documents(
        self,
        chunks: List[TextChunk],
        embeddings: List[List[float]]
    ) -> int:
        """
        Add or update documents in the vector store
        
        Args:
            chunks: List of TextChunk objects
            embeddings: List of embedding vectors
            
        Returns:
            Number of documents upserted
        """
        if not chunks or not embeddings:
            return 0
        
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "source_id": chunk.source_id,
                "source_type": chunk.source_type,
                **chunk.metadata
            }
            for chunk in chunks
        ]
        
        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            return len(chunks)
        except Exception as e:
            print(f"Error upserting documents: {e}")
            return 0
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = None,
        filter_dict: Dict = None
    ) -> List[SearchResult]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return (uses config default if not specified)
            filter_dict: Optional filter on metadata
            
        Returns:
            List of SearchResult objects ordered by relevance
        """
        if top_k is None:
            top_k = self.config.ollama.top_k
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_dict
            )
            
            search_results = []
            
            if results and results.get("ids"):
                ids = results["ids"][0]
                documents = results["documents"][0] if results.get("documents") else [""] * len(ids)
                metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
                distances = results["distances"][0] if results.get("distances") else [0] * len(ids)
                
                for i, doc_id in enumerate(ids):
                    # Convert distance to similarity score (1 - distance for cosine)
                    score = 1.0 - distances[i] if distances[i] <= 1 else 1.0 / (1.0 + distances[i])
                    
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    
                    search_results.append(SearchResult(
                        chunk_id=doc_id,
                        content=documents[i] if i < len(documents) else "",
                        source_id=metadata.get("source_id", ""),
                        source_type=metadata.get("source_type", ""),
                        metadata=metadata,
                        score=score
                    ))
            
            return search_results
            
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def search_by_source_type(
        self,
        query_embedding: List[float],
        source_type: str,
        top_k: int = None
    ) -> List[SearchResult]:
        """
        Search for documents of a specific source type
        
        Args:
            query_embedding: Query embedding vector  
            source_type: Type to filter by (stock, news, statement)
            top_k: Number of results
            
        Returns:
            List of SearchResult objects
        """
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict={"source_type": source_type}
        )
    
    def delete_by_source(self, source_id: str) -> bool:
        """
        Delete all documents from a specific source
        
        Args:
            source_id: Source ID to delete
            
        Returns:
            True if successful
        """
        try:
            self.collection.delete(
                where={"source_id": source_id}
            )
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False
    
    def get_document_count(self) -> int:
        """Get total number of documents in the store"""
        try:
            return self.collection.count()
        except:
            return 0
    
    def clear(self) -> bool:
        """Clear all documents from the store"""
        try:
            if self.client != "fallback":
                self.client.delete_collection(self.config.vector_store.collection_name)
                self._collection = None
            else:
                self._collection = FallbackCollection()
            return True
        except Exception as e:
            print(f"Error clearing store: {e}")
            return False


class FallbackCollection:
    """
    Simple in-memory fallback when ChromaDB is not available
    """
    
    def __init__(self):
        self._documents: Dict[str, Dict] = {}
    
    def add(self, ids, embeddings, documents, metadatas):
        for i, doc_id in enumerate(ids):
            self._documents[doc_id] = {
                "id": doc_id,
                "embedding": embeddings[i],
                "document": documents[i],
                "metadata": metadatas[i]
            }
    
    def upsert(self, ids, embeddings, documents, metadatas):
        self.add(ids, embeddings, documents, metadatas)
    
    def query(self, query_embeddings, n_results, where=None):
        import math
        
        query_emb = query_embeddings[0]
        
        # Calculate similarities
        results = []
        for doc_id, doc in self._documents.items():
            # Apply filter
            if where:
                match = True
                for key, value in where.items():
                    if doc["metadata"].get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            # Calculate cosine similarity
            doc_emb = doc["embedding"]
            dot = sum(a*b for a, b in zip(query_emb, doc_emb))
            mag1 = math.sqrt(sum(a*a for a in query_emb))
            mag2 = math.sqrt(sum(b*b for b in doc_emb))
            similarity = dot / (mag1 * mag2) if mag1 * mag2 > 0 else 0
            
            # Convert similarity to distance
            distance = 1 - similarity
            
            results.append({
                "id": doc_id,
                "document": doc["document"],
                "metadata": doc["metadata"],
                "distance": distance
            })
        
        # Sort by distance (lower is better)
        results.sort(key=lambda x: x["distance"])
        results = results[:n_results]
        
        return {
            "ids": [[r["id"] for r in results]],
            "documents": [[r["document"] for r in results]],
            "metadatas": [[r["metadata"] for r in results]],
            "distances": [[r["distance"] for r in results]]
        }
    
    def delete(self, where):
        to_delete = []
        for doc_id, doc in self._documents.items():
            match = True
            for key, value in where.items():
                if doc["metadata"].get(key) != value:
                    match = False
                    break
            if match:
                to_delete.append(doc_id)
        
        for doc_id in to_delete:
            del self._documents[doc_id]
    
    def count(self):
        return len(self._documents)


if __name__ == "__main__":
    # Test the vector store
    from src.rag.embeddings import EmbeddingService
    
    print("Testing Vector Store...")
    print("-" * 50)
    
    embedding_service = EmbeddingService()
    store = VectorStore()
    
    # Create test chunks
    test_texts = [
        "Apple stock price increased 5% today after earnings report",
        "Microsoft announced new AI features for Windows",
        "Tesla deliveries exceeded analyst expectations this quarter",
        "Federal Reserve signals potential rate cuts in 2024",
        "Amazon Web Services revenue growth accelerates"
    ]
    
    chunks = []
    for i, text in enumerate(test_texts):
        chunk = TextChunk(
            content=text,
            chunk_id=f"test_{i}",
            source_id=f"source_{i}",
            source_type="news",
            metadata={"index": i}
        )
        chunks.append(chunk)
    
    # Generate embeddings
    embeddings = embedding_service.embed_texts(test_texts)
    
    # Add to store
    added = store.add_documents(chunks, embeddings)
    print(f"Added {added} documents to store")
    print(f"Total documents: {store.get_document_count()}")
    
    # Test search
    print("\n" + "-" * 50)
    query = "Apple stock performance"
    query_emb = embedding_service.embed_text(query)
    
    results = store.search(query_emb, top_k=3)
    print(f"\nSearch results for '{query}':")
    for r in results:
        print(f"  [{r.score:.3f}] {r.content[:60]}...")
