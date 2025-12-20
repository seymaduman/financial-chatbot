"""
Retriever - Intelligent document retrieval for RAG pipeline
Supports multi-source retrieval with relevance scoring and top_p filtering
"""
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import get_config
from src.rag.embeddings import EmbeddingService, TextChunk
from src.rag.vector_store import VectorStore, SearchResult


@dataclass
class RetrievalResult:
    """Aggregated retrieval result with context"""
    query: str
    results: List[SearchResult]
    context_text: str
    sources: List[Dict]
    total_retrieved: int
    top_k_used: int
    top_p_used: float


class Retriever:
    """
    Intelligent document retriever for RAG
    Handles query expansion, multi-source retrieval, and relevance filtering
    """
    
    def __init__(self):
        self.config = get_config()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        top_p: float = None,
        source_types: List[str] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query
            top_k: Number of documents to retrieve (uses config if not specified)
            top_p: Cumulative probability threshold for filtering (uses config if not specified)
            source_types: Optional list of source types to filter (stock, news, statement)
            
        Returns:
            RetrievalResult with ranked documents and context
        """
        # Use config values if not specified
        if top_k is None:
            top_k = self.config.ollama.top_k
        if top_p is None:
            top_p = self.config.ollama.top_p
        
        # Expand query for better retrieval
        expanded_query = self._expand_query(query)
        
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(expanded_query)
        
        # Retrieve from vector store
        if source_types:
            # Retrieve from each source type and merge
            all_results = []
            for source_type in source_types:
                results = self.vector_store.search_by_source_type(
                    query_embedding=query_embedding,
                    source_type=source_type,
                    top_k=top_k
                )
                all_results.extend(results)
            
            # Re-rank merged results
            all_results.sort(key=lambda x: x.score, reverse=True)
            results = all_results[:top_k]
        else:
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k
            )
        
        # Apply top_p filtering
        filtered_results = self._apply_top_p_filter(results, top_p)
        
        # Build context text
        context_text = self._build_context(filtered_results)
        
        # Extract unique sources
        sources = self._extract_sources(filtered_results)
        
        return RetrievalResult(
            query=query,
            results=filtered_results,
            context_text=context_text,
            sources=sources,
            total_retrieved=len(filtered_results),
            top_k_used=top_k,
            top_p_used=top_p
        )
    
    def retrieve_for_stock(
        self,
        symbol: str,
        query: str = None,
        top_k: int = None
    ) -> RetrievalResult:
        """
        Retrieve documents related to a specific stock
        
        Args:
            symbol: Stock ticker symbol
            query: Optional additional query context
            top_k: Number of results
            
        Returns:
            RetrievalResult with stock-related documents
        """
        # Build stock-specific query
        if query:
            full_query = f"{symbol} stock {query}"
        else:
            full_query = f"{symbol} stock price performance news"
        
        return self.retrieve(
            query=full_query,
            top_k=top_k,
            source_types=["stock", "news", "statement"]
        )
    
    def retrieve_news(
        self,
        query: str,
        top_k: int = None
    ) -> RetrievalResult:
        """
        Retrieve news articles for a query
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            RetrievalResult with news articles
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            source_types=["news"]
        )
    
    def retrieve_financials(
        self,
        query: str,
        top_k: int = None
    ) -> RetrievalResult:
        """
        Retrieve financial statement data
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            RetrievalResult with financial data
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            source_types=["statement"]
        )
    
    def _expand_query(self, query: str) -> str:
        """
        Expand query with synonyms and related terms
        for better retrieval coverage
        """
        # Simple expansion rules
        expansions = {
            "price": "price stock value trading",
            "earnings": "earnings revenue profit income quarterly annual",
            "news": "news announcement update report",
            "buy": "buy purchase acquire long bullish",
            "sell": "sell short bearish decline",
            "up": "increase rise gain surge rally",
            "down": "decrease fall drop decline crash",
            "growth": "growth expansion increase rising",
            "debt": "debt liabilities leverage borrowing",
            "dividend": "dividend yield payout distribution"
        }
        
        query_lower = query.lower()
        expanded_terms = []
        
        for term, expansion in expansions.items():
            if term in query_lower:
                expanded_terms.append(expansion)
        
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"
        
        return query
    
    def _apply_top_p_filter(
        self,
        results: List[SearchResult],
        top_p: float
    ) -> List[SearchResult]:
        """
        Apply nucleus sampling (top_p) to filter results
        Keep results until cumulative probability exceeds top_p
        """
        if not results:
            return []
        
        # Normalize scores to probabilities
        total_score = sum(r.score for r in results)
        if total_score == 0:
            return results
        
        # Calculate cumulative probability and filter
        filtered = []
        cumulative_prob = 0.0
        
        for result in results:
            prob = result.score / total_score
            cumulative_prob += prob
            
            filtered.append(result)
            
            if cumulative_prob >= top_p:
                break
        
        return filtered
    
    def _build_context(self, results: List[SearchResult]) -> str:
        """Build context text from search results"""
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        
        for i, result in enumerate(results, 1):
            source_info = f"[Source {i}: {result.source_type}]"
            context_parts.append(f"{source_info}\n{result.content}")
        
        return "\n\n".join(context_parts)
    
    def _extract_sources(self, results: List[SearchResult]) -> List[Dict]:
        """Extract unique source information from results"""
        sources = []
        seen_sources = set()
        
        for result in results:
            source_key = result.source_id
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({
                    "source_id": result.source_id,
                    "source_type": result.source_type,
                    "score": result.score,
                    "metadata": result.metadata
                })
        
        return sources
    
    def index_stock_data(
        self,
        symbol: str,
        stock_text: str,
        metadata: Dict = None
    ) -> int:
        """
        Index stock data into the vector store
        
        Args:
            symbol: Stock ticker symbol
            stock_text: Text representation of stock data
            metadata: Additional metadata
            
        Returns:
            Number of chunks indexed
        """
        chunks = self.embedding_service.chunk_text(
            text=stock_text,
            source_id=f"stock_{symbol}",
            source_type="stock",
            metadata={"ticker": symbol, **(metadata or {})}
        )
        
        if not chunks:
            return 0
        
        embeddings = self.embedding_service.embed_texts([c.content for c in chunks])
        return self.vector_store.upsert_documents(chunks, embeddings)
    
    def index_news(
        self,
        article_id: str,
        news_text: str,
        metadata: Dict = None
    ) -> int:
        """
        Index a news article into the vector store
        
        Args:
            article_id: Unique article identifier
            news_text: Text content of the article
            metadata: Additional metadata (tickers, sentiment, etc.)
            
        Returns:
            Number of chunks indexed
        """
        chunks = self.embedding_service.chunk_text(
            text=news_text,
            source_id=f"news_{article_id}",
            source_type="news",
            metadata=metadata or {}
        )
        
        if not chunks:
            return 0
        
        embeddings = self.embedding_service.embed_texts([c.content for c in chunks])
        return self.vector_store.upsert_documents(chunks, embeddings)
    
    def index_financial_statement(
        self,
        symbol: str,
        statement_type: str,
        statement_text: str,
        metadata: Dict = None
    ) -> int:
        """
        Index a financial statement into the vector store
        
        Args:
            symbol: Stock ticker symbol
            statement_type: Type of statement (income, balance, cashflow)
            statement_text: Text representation of the statement
            metadata: Additional metadata
            
        Returns:
            Number of chunks indexed
        """
        chunks = self.embedding_service.chunk_text(
            text=statement_text,
            source_id=f"statement_{symbol}_{statement_type}",
            source_type="statement",
            metadata={"ticker": symbol, "statement_type": statement_type, **(metadata or {})}
        )
        
        if not chunks:
            return 0
        
        embeddings = self.embedding_service.embed_texts([c.content for c in chunks])
        return self.vector_store.upsert_documents(chunks, embeddings)
    
    def get_stats(self) -> Dict:
        """Get retriever statistics"""
        return {
            "total_documents": self.vector_store.get_document_count(),
            "embedding_dim": self.embedding_service.get_embedding_dim(),
            "default_top_k": self.config.ollama.top_k,
            "default_top_p": self.config.ollama.top_p
        }


if __name__ == "__main__":
    # Test the retriever
    print("Testing Retriever...")
    print("-" * 50)
    
    retriever = Retriever()
    
    # Index some test data
    print("\nIndexing test data...")
    
    retriever.index_stock_data(
        "AAPL",
        "Apple Inc. stock is trading at $180. The company reported strong iPhone sales. Market cap is $2.8 trillion.",
        {"sector": "Technology"}
    )
    
    retriever.index_news(
        "news_001",
        "Apple announces record quarterly earnings. Revenue grew 15% year-over-year. CEO Tim Cook expressed optimism.",
        {"tickers": ["AAPL"], "sentiment": "positive"}
    )
    
    retriever.index_financial_statement(
        "AAPL",
        "income",
        "Apple Income Statement: Revenue $95B, Net Income $25B, Gross Margin 43%, Operating Margin 30%",
        {"fiscal_year": "2023"}
    )
    
    print(f"Stats: {retriever.get_stats()}")
    
    # Test retrieval
    print("\n" + "-" * 50)
    print("\nTesting retrieval...")
    
    result = retriever.retrieve("What is Apple's revenue?", top_k=5)
    
    print(f"\nQuery: {result.query}")
    print(f"Retrieved: {result.total_retrieved} documents")
    print(f"Top K: {result.top_k_used}, Top P: {result.top_p_used}")
    
    print("\nContext:")
    print(result.context_text[:500])
    
    print("\nSources:")
    for source in result.sources:
        print(f"  - {source['source_id']} ({source['source_type']}) - Score: {source['score']:.3f}")
