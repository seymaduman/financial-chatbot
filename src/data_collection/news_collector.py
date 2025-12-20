"""
Financial News Collector - Scrapes news from Yahoo Finance and financial media
Normalizes and stores news with sentiment tagging for RAG knowledge base
"""
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import os
import hashlib

import requests
from bs4 import BeautifulSoup

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import get_config


@dataclass
class NewsArticle:
    """News article data structure"""
    id: str
    title: str
    summary: str
    source: str
    url: str
    published_at: str
    tickers: List[str]
    sentiment: str  # positive, negative, neutral
    sentiment_score: float  # -1.0 to 1.0
    category: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_text(self) -> str:
        """Convert to text for embedding"""
        tickers_str = ", ".join(self.tickers) if self.tickers else "General Market"
        return f"""
Title: {self.title}
Source: {self.source}
Published: {self.published_at}
Related Stocks: {tickers_str}
Sentiment: {self.sentiment.upper()} (Score: {self.sentiment_score:+.2f})
Category: {self.category}

Summary: {self.summary}
"""


class NewsCollector:
    """
    Financial news collector and aggregator
    Scrapes news from Yahoo Finance and other sources
    """
    
    YAHOO_NEWS_URL = "https://finance.yahoo.com/news"
    YAHOO_STOCK_NEWS_URL = "https://finance.yahoo.com/quote/{symbol}/news"
    
    # Simple sentiment keywords for basic analysis
    POSITIVE_WORDS = {
        "surge", "soar", "gain", "jump", "rally", "rise", "grow", "growth",
        "profit", "beat", "exceed", "record", "high", "boom", "strong",
        "upgrade", "bullish", "positive", "success", "breakthrough", "innovation",
        "expansion", "dividend", "buyback", "acquisition", "partnership"
    }
    
    NEGATIVE_WORDS = {
        "fall", "drop", "decline", "plunge", "crash", "loss", "miss", "cut",
        "layoff", "bankruptcy", "default", "lawsuit", "investigation", "fraud",
        "downgrade", "bearish", "negative", "warning", "risk", "concern",
        "recession", "inflation", "sell-off", "selloff", "weak", "slump"
    }
    
    def __init__(self):
        self.config = get_config()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.config.scraping.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        self._cache: Dict[str, List[NewsArticle]] = {}
        self._cache_times: Dict[str, datetime] = {}
        
        # Ensure cache directory exists
        os.makedirs(self.config.scraping.cache_directory, exist_ok=True)
    
    def _generate_article_id(self, url: str, title: str) -> str:
        """Generate unique ID for an article"""
        content = f"{url}_{title}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self._cache_times:
            return False
        cache_age = datetime.now() - self._cache_times[key]
        return cache_age < timedelta(hours=self.config.scraping.cache_ttl_hours)
    
    def _analyze_sentiment(self, text: str) -> tuple:
        """
        Analyze sentiment of text using keyword-based approach
        Returns (sentiment_label, sentiment_score)
        """
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        positive_count = len(words & self.POSITIVE_WORDS)
        negative_count = len(words & self.NEGATIVE_WORDS)
        
        total = positive_count + negative_count
        if total == 0:
            return "neutral", 0.0
        
        score = (positive_count - negative_count) / total
        
        if score > 0.2:
            return "positive", min(score, 1.0)
        elif score < -0.2:
            return "negative", max(score, -1.0)
        else:
            return "neutral", score
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock ticker symbols from text"""
        # Common pattern: $AAPL or (AAPL) or just capital letters in context
        tickers = set()
        
        # Match $TICKER pattern
        dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', text)
        tickers.update(dollar_tickers)
        
        # Match (TICKER) pattern
        paren_tickers = re.findall(r'\(([A-Z]{1,5})\)', text)
        tickers.update(paren_tickers)
        
        # Match common stock tickers mentioned
        common_tickers = ["AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "META", "TSLA", 
                         "NVDA", "AMD", "INTC", "NFLX", "DIS", "JPM", "BAC", "GS"]
        for ticker in common_tickers:
            if ticker in text:
                tickers.add(ticker)
        
        return list(tickers)
    
    def get_latest_news(self, limit: int = 20) -> List[NewsArticle]:
        """
        Get latest financial news from Yahoo Finance
        
        Args:
            limit: Maximum number of articles to return
            
        Returns:
            List of NewsArticle objects
        """
        cache_key = "latest_news"
        if self._is_cache_valid(cache_key):
            cached = self._cache.get(cache_key, [])
            return cached[:limit]
        
        try:
            response = self.session.get(
                self.YAHOO_NEWS_URL,
                timeout=self.config.scraping.request_timeout
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "lxml")
            articles = self._parse_news_page(soup)
            
            self._cache[cache_key] = articles
            self._cache_times[cache_key] = datetime.now()
            
            return articles[:limit]
            
        except Exception as e:
            print(f"Error fetching latest news: {e}")
            return []
    
    def get_stock_news(self, symbol: str, limit: int = 10) -> List[NewsArticle]:
        """
        Get news for a specific stock
        
        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of articles
            
        Returns:
            List of NewsArticle objects
        """
        cache_key = f"stock_news_{symbol}"
        if self._is_cache_valid(cache_key):
            cached = self._cache.get(cache_key, [])
            return cached[:limit]
        
        try:
            url = self.YAHOO_STOCK_NEWS_URL.format(symbol=symbol)
            response = self.session.get(url, timeout=self.config.scraping.request_timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "lxml")
            articles = self._parse_news_page(soup, default_ticker=symbol)
            
            self._cache[cache_key] = articles
            self._cache_times[cache_key] = datetime.now()
            
            return articles[:limit]
            
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return []
    
    def _parse_news_page(self, soup: BeautifulSoup, default_ticker: str = None) -> List[NewsArticle]:
        """Parse news articles from a Yahoo Finance page"""
        articles = []
        
        # Find news items - Yahoo Finance uses various structures
        news_items = soup.find_all("li", {"class": re.compile(r"stream-item|js-stream-content")})
        
        if not news_items:
            # Alternative selector
            news_items = soup.find_all("div", {"class": re.compile(r"Ov\(h\)|news-stream")})
        
        if not news_items:
            # Try finding article links directly
            news_items = soup.find_all("a", href=re.compile(r"/news/"))
        
        for item in news_items:
            try:
                article = self._parse_news_item(item, default_ticker)
                if article:
                    articles.append(article)
            except Exception as e:
                continue
        
        # Also try to extract from embedded JSON
        script_articles = self._extract_from_scripts(soup, default_ticker)
        articles.extend(script_articles)
        
        # Deduplicate by ID
        seen_ids = set()
        unique_articles = []
        for article in articles:
            if article.id not in seen_ids:
                seen_ids.add(article.id)
                unique_articles.append(article)
        
        return unique_articles
    
    def _parse_news_item(self, item, default_ticker: str = None) -> Optional[NewsArticle]:
        """Parse a single news item"""
        # Find title
        title_elem = item.find("h3") or item.find("a")
        if not title_elem:
            return None
        
        title = title_elem.text.strip()
        if not title or len(title) < 10:
            return None
        
        # Find URL
        url = ""
        link_elem = item.find("a", href=True)
        if link_elem:
            href = link_elem.get("href", "")
            if href.startswith("/"):
                url = f"https://finance.yahoo.com{href}"
            elif href.startswith("http"):
                url = href
        
        if not url:
            return None
        
        # Find summary/description
        summary = ""
        summary_elem = item.find("p")
        if summary_elem:
            summary = summary_elem.text.strip()
        else:
            summary = title
        
        # Find source
        source = "Yahoo Finance"
        source_elem = item.find("span", {"class": re.compile(r"source|provider")})
        if source_elem:
            source = source_elem.text.strip()
        
        # Find time
        time_str = datetime.now().isoformat()
        time_elem = item.find("time") or item.find("span", {"class": re.compile(r"time|ago")})
        if time_elem:
            if time_elem.get("datetime"):
                time_str = time_elem.get("datetime")
            else:
                time_str = self._parse_relative_time(time_elem.text.strip())
        
        # Extract tickers
        full_text = f"{title} {summary}"
        tickers = self._extract_tickers(full_text)
        if default_ticker and default_ticker not in tickers:
            tickers.insert(0, default_ticker)
        
        # Analyze sentiment
        sentiment_label, sentiment_score = self._analyze_sentiment(full_text)
        
        # Categorize
        category = self._categorize_news(full_text)
        
        return NewsArticle(
            id=self._generate_article_id(url, title),
            title=title,
            summary=summary,
            source=source,
            url=url,
            published_at=time_str,
            tickers=tickers,
            sentiment=sentiment_label,
            sentiment_score=sentiment_score,
            category=category
        )
    
    def _extract_from_scripts(self, soup: BeautifulSoup, default_ticker: str = None) -> List[NewsArticle]:
        """Extract news from embedded JSON scripts"""
        articles = []
        
        scripts = soup.find_all("script")
        for script in scripts:
            if not script.string:
                continue
            
            # Look for news data in scripts
            if "stream_items" in script.string or "newsByTicker" in script.string:
                try:
                    # Try to extract JSON
                    match = re.search(r'\[.*?"title".*?\]', script.string, re.DOTALL)
                    if match:
                        data = json.loads(match.group(0))
                        for item in data:
                            if isinstance(item, dict) and "title" in item:
                                article = self._dict_to_article(item, default_ticker)
                                if article:
                                    articles.append(article)
                except:
                    continue
        
        return articles
    
    def _dict_to_article(self, data: dict, default_ticker: str = None) -> Optional[NewsArticle]:
        """Convert dictionary to NewsArticle"""
        title = data.get("title", "")
        if not title:
            return None
        
        summary = data.get("summary", "") or data.get("description", "") or title
        url = data.get("url", "") or data.get("link", "")
        source = data.get("source", "Yahoo Finance") or data.get("provider", "Yahoo Finance")
        
        if isinstance(source, dict):
            source = source.get("name", "Yahoo Finance")
        
        time_str = data.get("published_at", "") or data.get("pubDate", "") or datetime.now().isoformat()
        
        full_text = f"{title} {summary}"
        tickers = self._extract_tickers(full_text)
        if default_ticker and default_ticker not in tickers:
            tickers.insert(0, default_ticker)
        
        sentiment_label, sentiment_score = self._analyze_sentiment(full_text)
        category = self._categorize_news(full_text)
        
        return NewsArticle(
            id=self._generate_article_id(url, title),
            title=title,
            summary=summary,
            source=source,
            url=url or "https://finance.yahoo.com/news",
            published_at=time_str,
            tickers=tickers,
            sentiment=sentiment_label,
            sentiment_score=sentiment_score,
            category=category
        )
    
    def _parse_relative_time(self, text: str) -> str:
        """Parse relative time string to ISO format"""
        text = text.lower().strip()
        now = datetime.now()
        
        if "hour" in text:
            hours = int(re.search(r'(\d+)', text).group(1)) if re.search(r'(\d+)', text) else 1
            return (now - timedelta(hours=hours)).isoformat()
        elif "minute" in text:
            minutes = int(re.search(r'(\d+)', text).group(1)) if re.search(r'(\d+)', text) else 1
            return (now - timedelta(minutes=minutes)).isoformat()
        elif "day" in text:
            days = int(re.search(r'(\d+)', text).group(1)) if re.search(r'(\d+)', text) else 1
            return (now - timedelta(days=days)).isoformat()
        elif "yesterday" in text:
            return (now - timedelta(days=1)).isoformat()
        else:
            return now.isoformat()
    
    def _categorize_news(self, text: str) -> str:
        """Categorize news article"""
        text_lower = text.lower()
        
        categories = {
            "earnings": ["earnings", "revenue", "profit", "quarterly", "annual report", "eps"],
            "merger_acquisition": ["merger", "acquisition", "acquire", "takeover", "buyout", "deal"],
            "market_movement": ["market", "index", "dow", "nasdaq", "s&p", "rally", "crash"],
            "regulation": ["sec", "regulation", "lawsuit", "investigation", "compliance", "antitrust"],
            "product": ["product", "launch", "release", "innovation", "technology", "patent"],
            "leadership": ["ceo", "cfo", "executive", "resign", "appoint", "leadership"],
            "economy": ["fed", "interest rate", "inflation", "gdp", "unemployment", "economy"]
        }
        
        for category, keywords in categories.items():
            if any(kw in text_lower for kw in keywords):
                return category
        
        return "general"
    
    def get_aggregated_sentiment(self, symbol: str = None) -> Dict:
        """
        Get aggregated sentiment for a stock or the general market
        
        Returns:
            Dictionary with sentiment metrics
        """
        if symbol:
            articles = self.get_stock_news(symbol, limit=20)
        else:
            articles = self.get_latest_news(limit=30)
        
        if not articles:
            return {
                "symbol": symbol or "market",
                "article_count": 0,
                "avg_sentiment": 0.0,
                "sentiment_label": "neutral",
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0
            }
        
        scores = [a.sentiment_score for a in articles]
        avg_score = sum(scores) / len(scores)
        
        positive_count = sum(1 for a in articles if a.sentiment == "positive")
        negative_count = sum(1 for a in articles if a.sentiment == "negative")
        neutral_count = sum(1 for a in articles if a.sentiment == "neutral")
        
        if avg_score > 0.1:
            label = "positive"
        elif avg_score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "symbol": symbol or "market",
            "article_count": len(articles),
            "avg_sentiment": avg_score,
            "sentiment_label": label,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count
        }


if __name__ == "__main__":
    # Test the news collector
    collector = NewsCollector()
    
    print("Testing News Collector...")
    print("-" * 50)
    
    # Test latest news
    print("\nLatest Financial News:")
    news = collector.get_latest_news(limit=5)
    for article in news:
        print(f"\n{article.title}")
        print(f"  Source: {article.source} | Sentiment: {article.sentiment}")
        print(f"  Tickers: {', '.join(article.tickers) if article.tickers else 'None'}")
    
    print("\n" + "-" * 50)
    
    # Test stock-specific news
    print("\nAAPL News:")
    aapl_news = collector.get_stock_news("AAPL", limit=3)
    for article in aapl_news:
        print(f"\n{article.title}")
        print(f"  Sentiment: {article.sentiment} ({article.sentiment_score:+.2f})")
    
    print("\n" + "-" * 50)
    
    # Test aggregated sentiment
    print("\nAggregated Sentiment for AAPL:")
    sentiment = collector.get_aggregated_sentiment("AAPL")
    print(f"  Articles analyzed: {sentiment['article_count']}")
    print(f"  Average sentiment: {sentiment['avg_sentiment']:+.2f}")
    print(f"  Label: {sentiment['sentiment_label']}")
