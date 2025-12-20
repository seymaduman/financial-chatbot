"""
Chat Controller - Conversation management and query processing
"""
import os
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import get_config
from src.data_collection import YahooScraper, NewsCollector, StatementsScraper
from src.rag import Retriever
from src.llm import OllamaClient, PromptBuilder
from src.analysis import SentimentAnalyzer, MarketPredictor, TrendAnalyzer
from src.chatbot.formatter import ResponseFormatter


class QueryIntent(Enum):
    """Types of user queries"""
    STOCK_PRICE = "stock_price"
    STOCK_ANALYSIS = "stock_analysis"
    NEWS = "news"
    NEWS_ANALYSIS = "news_analysis"
    FINANCIALS = "financials"
    PREDICTION = "prediction"
    SENTIMENT = "sentiment"
    TREND = "trend"
    GENERAL = "general"
    HELP = "help"
    SETTINGS = "settings"


@dataclass
class QueryContext:
    """Parsed query context"""
    intent: QueryIntent
    tickers: List[str]
    keywords: List[str]
    time_period: Optional[str]
    original_query: str


@dataclass
class ChatResponse:
    """Chat response structure"""
    text: str
    sources: List[Dict]
    confidence: float
    intent: QueryIntent
    generation_params: Dict
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ChatController:
    """
    Main conversation controller for RAG financial chatbot
    Handles query understanding, data retrieval, and response generation
    """
    
    # Common stock tickers for detection
    COMMON_TICKERS = {
        "AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "META", "TSLA", "NVDA",
        "AMD", "INTC", "NFLX", "DIS", "JPM", "BAC", "GS", "WMT", "COST",
        "V", "MA", "PYPL", "CRM", "ORCL", "IBM", "CSCO", "ADBE"
    }
    
    def __init__(self):
        self.config = get_config()
        
        # Initialize components
        self.scraper = YahooScraper()
        self.news_collector = NewsCollector()
        self.statements_scraper = StatementsScraper()
        self.retriever = Retriever()
        self.llm_client = OllamaClient()
        self.prompt_builder = PromptBuilder()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.market_predictor = MarketPredictor()
        self.trend_analyzer = TrendAnalyzer()
        self.formatter = ResponseFormatter()
        
        # Session state
        self._last_ticker: Optional[str] = None
        self._context_data: Dict = {}
    
    def process_query(self, query: str, override_context: Dict = None) -> ChatResponse:
        """
        Process user query and generate response
        
        Args:
            query: User's question
            override_context: Optional context override (e.g. selected ticker)
            
        Returns:
            ChatResponse with answer and metadata
        """
        # Parse query
        context = self._parse_query(query)
        
        # Apply override context if provided
        if override_context and "ticker" in override_context:
            context.tickers = [override_context["ticker"]]
            print(f"[DEBUG] Using override ticker: {context.tickers[0]}")
        
        # Handle special commands
        if context.intent == QueryIntent.HELP:
            return self._handle_help()
        
        if context.intent == QueryIntent.SETTINGS:
            return self._handle_settings(query)
        
        # Store old ticker before updating for comparison
        old_ticker = self._last_ticker
        
        # Update context with detected ticker
        if context.tickers:
            self._last_ticker = context.tickers[0]
            # Clear previous context when switching to a new ticker
            if old_ticker and old_ticker != self._last_ticker:
                print(f"[DEBUG] Switching from {old_ticker} to {self._last_ticker} - clearing context")
                self._clear_context()
        elif self._last_ticker:
            context.tickers = [self._last_ticker]
        
        # Debug: Show what ticker we're using
        if context.tickers:
            print(f"[DEBUG] Processing query for ticker: {context.tickers[0]}")
        else:
            print(f"[DEBUG] No ticker detected in query")
        
        # Collect relevant data (fresh for each query)
        retrieved_context = self._collect_data(context)
        
        # Generate response
        response = self._generate_response(context, retrieved_context)
        
        # Update conversation history
        self.prompt_builder.add_user_message(query)
        self.prompt_builder.add_assistant_message(response.text)
        
        return response
    
    def _parse_query(self, query: str) -> QueryContext:
        """Parse and classify user query"""
        query_lower = query.lower()
        
        # Detect tickers
        tickers = self._extract_tickers(query)
        
        # Detect intent
        intent = self._classify_intent(query_lower)
        
        # Extract keywords
        keywords = self._extract_keywords(query_lower)
        
        # Detect time period
        time_period = self._detect_time_period(query_lower)
        
        return QueryContext(
            intent=intent,
            tickers=tickers,
            keywords=keywords,
            time_period=time_period,
            original_query=query
        )
    
    def _extract_tickers(self, query: str) -> List[str]:
        """Extract stock ticker symbols (including crypto and indices)"""
        tickers = []
        
        # Match $TICKER pattern
        dollar_tickers = re.findall(r'\$([A-Z\^][A-Z0-9\-\.]{0,10})\b', query.upper())
        tickers.extend(dollar_tickers)
        
        # Match crypto patterns (BTC-USD, ETH-USD, etc.)
        crypto_pattern = r'\b([A-Z]{3,4})-USD\b'
        crypto_tickers = re.findall(crypto_pattern, query.upper())
        tickers.extend([f"{c}-USD" for c in crypto_tickers])
        
        # Match index patterns (^IXIC, ^GSPC, etc.)
        index_pattern = r'\^([A-Z]{2,10})\b'
        index_tickers = re.findall(index_pattern, query.upper())
        tickers.extend([f"^{idx}" for idx in index_tickers])
        
        # Match common ticker names
        ticker_mappings = {
            "bitcoin": "BTC-USD",
            "btc": "BTC-USD",
            "ethereum": "ETH-USD",
            "eth": "ETH-USD",
            "nasdaq": "^IXIC",
            "s&p": "^GSPC",
            "sp500": "^GSPC",
            "dow": "^DJI"
        }
        
        query_lower = query.lower()
        for name, symbol in ticker_mappings.items():
            if name in query_lower:
                tickers.append(symbol)
        
        # Match common tickers in text
        words = query.upper().split()
        for word in words:
            clean_word = re.sub(r'[^\w\-\^]', '', word)
            if clean_word in self.COMMON_TICKERS:
                tickers.append(clean_word)
        
        # Dynamic search if no tickers found and query looks like it might contain a company name
        if not tickers and len(query.split()) < 10:  # Avoid searching long paragraphs
            cleaned_query = self._clean_query_for_search(query)
            
            # Basic validation to avoid searching garbage
            ignored_words = {"hello", "hi", "help", "thanks", "thank you", "goodbye", "bye", "hey", "yes", "no"}
            
            if cleaned_query and len(cleaned_query) > 1 and cleaned_query not in ignored_words:
                # print(f"[DEBUG] Searching for ticker with term: '{cleaned_query}'")
                found_ticker = self.scraper.get_ticker_from_name(cleaned_query)
                if found_ticker:
                    tickers.append(found_ticker)
        
        return list(set(tickers))  # Remove duplicates

    def _clean_query_for_search(self, query: str) -> str:
        """Remove common conversational phrases to isolate company name"""
        noise_phrases = [
            "what is the price of", "price of", "price for", "show me", "tell me about",
            "analyze", "analysis of", "news about", "news for", "sentiment for",
            "sentiment about", "trend for", "forecast for", "prediction for",
            "stock price", "stock", "share price", "value of", "how much is",
            "trading at", "quote for", "market cap of", "earnings for",
            "what is", "how about", "looking for", "search for"
        ]
        
        cleaned = query.lower()
        for phrase in noise_phrases:
            cleaned = cleaned.replace(phrase, "")
            
        # Remove special chars but keep spaces
        cleaned = re.sub(r'[^\w\s\-\.\&]', '', cleaned)
        return cleaned.strip()
    
    def _clear_context(self):
        """Clear retrieved context to prevent confusion from previous queries"""
        self._context_data.clear()
        
        # Clear the scraper's cache for fresh data
        if hasattr(self, 'scraper'):
            self.scraper.clear_cache()
        
        # Clear the vector store to prevent mixing old stock data
        try:
            if hasattr(self, 'retriever') and hasattr(self.retriever, 'vector_store'):
                # Clear all stock-related documents
                self.retriever.vector_store.collection.delete(where={"source_type": "stock"})
        except Exception as e:
            # Silently ignore if vector store doesn't support deletion
            pass

    
    def _classify_intent(self, query: str) -> QueryIntent:
        """Classify query intent"""
        # Help
        if any(w in query for w in ["help", "how to", "what can you"]):
            return QueryIntent.HELP
        
        # Settings
        if any(w in query for w in ["set temperature", "set top_k", "change param"]):
            return QueryIntent.SETTINGS
        
        # Price query
        if any(w in query for w in ["price", "trading at", "worth", "cost"]):
            return QueryIntent.STOCK_PRICE
        
        # Prediction
        if any(w in query for w in ["predict", "will go", "forecast", "outlook"]):
            return QueryIntent.PREDICTION
        
        # Sentiment
        if any(w in query for w in ["sentiment", "feeling", "mood"]):
            return QueryIntent.SENTIMENT
        
        # Trend
        if any(w in query for w in ["trend", "trending", "direction"]):
            return QueryIntent.TREND
        
        # News
        if any(w in query for w in ["news", "headline", "latest"]):
            return QueryIntent.NEWS
        
        # Financials
        if any(w in query for w in ["revenue", "earnings", "profit", "balance", "income", "cash flow"]):
            return QueryIntent.FINANCIALS
        
        # Stock analysis (if ticker detected)
        if any(w in query for w in ["analyze", "analysis", "tell me about"]):
            return QueryIntent.STOCK_ANALYSIS
        
        return QueryIntent.GENERAL
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords"""
        important_words = [
            "earnings", "revenue", "profit", "growth", "dividend",
            "pe", "eps", "market cap", "volume", "price",
            "buy", "sell", "hold", "bullish", "bearish"
        ]
        
        found = [w for w in important_words if w in query]
        return found
    
    def _detect_time_period(self, query: str) -> Optional[str]:
        """Detect time period in query"""
        if "today" in query or "now" in query:
            return "1d"
        if "week" in query:
            return "1wk"
        if "month" in query:
            return "1mo"
        if "quarter" in query:
            return "3mo"
        if "year" in query:
            return "1y"
        return None
    
    def _collect_data(self, context: QueryContext) -> str:
        """Collect relevant data based on query context"""
        data_parts = []
        
        ticker = context.tickers[0] if context.tickers else None
        
        if ticker:
            print(f"[DEBUG] Fetching data for: {ticker}")
        
        # Stock price data
        if ticker and context.intent in [QueryIntent.STOCK_PRICE, QueryIntent.STOCK_ANALYSIS, 
                                          QueryIntent.TREND, QueryIntent.GENERAL]:
            print(f"[DEBUG] Calling scraper.get_stock({ticker})")
            quote = self.scraper.get_stock(ticker)
            if quote:
                print(f"[DEBUG] Got quote: {quote.symbol} - ${quote.price:.2f}")
                data_parts.append(quote.to_text())
                self._context_data["quote"] = quote
            else:
                print(f"[DEBUG] WARNING: No quote data returned for {ticker}")
        
        # News data
        if context.intent in [QueryIntent.NEWS, QueryIntent.NEWS_ANALYSIS, 
                               QueryIntent.PREDICTION, QueryIntent.SENTIMENT]:
            if ticker:
                news = self.news_collector.get_stock_news(ticker, limit=5)
            else:
                news = self.news_collector.get_latest_news(limit=5)
            
            if news:
                data_parts.append("\n--- RECENT NEWS ---")
                for article in news[:3]:
                    data_parts.append(article.to_text())
                self._context_data["news"] = news
        
        # Financial statements
        if ticker and context.intent in [QueryIntent.FINANCIALS, QueryIntent.STOCK_ANALYSIS]:
            summary = self.statements_scraper.get_financial_summary(ticker)
            if summary:
                data_parts.append(summary)
        
        # Historical for trends
        if ticker and context.intent == QueryIntent.TREND:
            history = self.scraper.get_historical_data(ticker, period="1mo")
            if history:
                prices = [h.close for h in history]
                trend = self.trend_analyzer.analyze_price_trend(prices)
                data_parts.append(self.trend_analyzer.get_trend_summary(ticker, trend))
        
        # Index fresh data into RAG (only the current ticker's data)
        if data_parts and ticker:
            combined = "\n".join(data_parts)
            #Use ticker-specific ID to prevent mixing
            self.retriever.index_stock_data(f"{ticker}_current", combined)
            
            print(f"[DEBUG] Returning {len(combined)} characters of data for {ticker}")
            # Directly return the freshly collected data instead of retrieving from RAG
            # This ensures we ONLY show the current ticker's data
            return combined
        
        if not data_parts:
            print(f"[DEBUG] WARNING: No data collected for ticker={ticker}, intent={context.intent}")
        
        return "\n".join(data_parts) if data_parts else "No relevant data found."
    
    def _generate_response(self, context: QueryContext, retrieved_data: str) -> ChatResponse:
        """Generate LLM response"""
        # Set context
        self.prompt_builder.set_context(retrieved_data, [])
        
        # Build prompt based on intent
        include_prediction = context.intent in [QueryIntent.PREDICTION, QueryIntent.NEWS_ANALYSIS]
        
        prompt = self.prompt_builder.build_prompt(
            user_query=context.original_query,
            include_history=True,
            include_prediction=include_prediction
        )
        
        system_prompt = self.prompt_builder.build_system_prompt(include_prediction)
        
        # Generate response
        result = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt
        )
        
        # Calculate confidence
        confidence = self._estimate_confidence(context, retrieved_data)
        
        # Format response
        formatted_text = self.formatter.format_response(
            text=result.text,
            intent=context.intent,
            ticker=context.tickers[0] if context.tickers else None,
            include_disclaimer=include_prediction
        )
        
        return ChatResponse(
            text=formatted_text,
            sources=[{"type": "rag", "retrieved": True}],
            confidence=confidence,
            intent=context.intent,
            generation_params=self.llm_client.get_generation_params()
        )
    
    def _estimate_confidence(self, context: QueryContext, data: str) -> float:
        """Estimate response confidence"""
        confidence = 0.5  # Base
        
        # Data availability
        if len(data) > 500:
            confidence += 0.2
        elif len(data) > 100:
            confidence += 0.1
        
        # Ticker detected
        if context.tickers:
            confidence += 0.1
        
        # Clear intent
        if context.intent != QueryIntent.GENERAL:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _handle_help(self) -> ChatResponse:
        """Handle help command"""
        help_text = """
# RAG Financial Chatbot - Help

## What I Can Do:
- **Stock Prices**: "What is AAPL's current price?"
- **Stock Analysis**: "Analyze Tesla stock"
- **Financial Statements**: "Show me Apple's revenue and profit margins"
- **News**: "What's the latest news on Microsoft?"
- **Sentiment Analysis**: "What's the sentiment for NVDA?"
- **Market Predictions**: "Predict where Amazon stock will go"
- **Trend Analysis**: "Show me the trend for Google"

## Tips:
- Use ticker symbols like $AAPL or just AAPL
- I remember context - follow up questions work!
- Ask me to explain my reasoning

## Settings:
- "set temperature 0.7" - Adjust response creativity
- "set top_k 50" - Adjust retrieval diversity
"""
        return ChatResponse(
            text=help_text,
            sources=[],
            confidence=1.0,
            intent=QueryIntent.HELP,
            generation_params=self.llm_client.get_generation_params()
        )
    
    def _handle_settings(self, query: str) -> ChatResponse:
        """Handle settings commands"""
        # Parse setting
        temp_match = re.search(r'temperature\s+([0-9.]+)', query)
        topk_match = re.search(r'top_k\s+(\d+)', query)
        topp_match = re.search(r'top_p\s+([0-9.]+)', query)
        
        changes = []
        
        if temp_match:
            temp = float(temp_match.group(1))
            self.llm_client.update_params(temperature=temp)
            changes.append(f"temperature={temp}")
        
        if topk_match:
            topk = int(topk_match.group(1))
            self.llm_client.update_params(top_k=topk)
            changes.append(f"top_k={topk}")
        
        if topp_match:
            topp = float(topp_match.group(1))
            self.llm_client.update_params(top_p=topp)
            changes.append(f"top_p={topp}")
        
        if changes:
            text = f"✓ Settings updated: {', '.join(changes)}"
        else:
            params = self.llm_client.get_generation_params()
            text = f"Current settings:\n" + "\n".join(f"  {k}: {v}" for k, v in params.items())
        
        return ChatResponse(
            text=text,
            sources=[],
            confidence=1.0,
            intent=QueryIntent.SETTINGS,
            generation_params=self.llm_client.get_generation_params()
        )
    
    def clear_history(self):
        """Clear conversation history"""
        self.prompt_builder.clear_history()
        self._last_ticker = None
        self._context_data.clear()
