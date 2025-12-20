"""
Yahoo Finance Scraper - Enhanced with yfinance for real-time data
Supports stocks, crypto, and indices dynamically
"""
import json
import re
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import os

import yfinance as yf
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import get_config


@dataclass
class StockQuote:
    """Stock quote data structure"""
    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    volume: int
    avg_volume: int
    market_cap: str
    pe_ratio: Optional[float]
    eps: Optional[float]
    dividend_yield: Optional[float]
    fifty_two_week_high: Optional[float]
    fifty_two_week_low: Optional[float]
    timestamp: str
    asset_type: str = "stock"  # stock, crypto, index
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_text(self) -> str:
        """Convert to text for embedding"""
        # Format P/E ratio
        pe_str = f"{self.pe_ratio:.2f}" if self.pe_ratio is not None else "N/A"
        
        # Format EPS
        eps_str = f"${self.eps:.2f}" if self.eps is not None else "N/A"
        
        # Format dividend yield
        div_str = f"{self.dividend_yield:.2f}%" if self.dividend_yield is not None else "N/A"
        
        # Format 52-week range
        if self.fifty_two_week_low is not None and self.fifty_two_week_high is not None:
            range_str = f"${self.fifty_two_week_low:.2f} - ${self.fifty_two_week_high:.2f}"
        else:
            range_str = "N/A"
        
        asset_label = self.asset_type.upper()
        
        return f"""
{asset_label}: {self.symbol} ({self.name})
Current Price: ${self.price:.2f}
Change: {self.change:+.2f} ({self.change_percent:+.2f}%)
Volume: {self.volume:,} (Avg: {self.avg_volume:,})
Market Cap: {self.market_cap}
P/E Ratio: {pe_str}
EPS: {eps_str}
Dividend Yield: {div_str}
52-Week Range: {range_str}
Last Updated: {self.timestamp}
"""


@dataclass
class HistoricalPrice:
    """Historical price data point"""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: float


class YahooScraper:
    """
    Yahoo Finance scraper using yfinance library
    Supports stocks, crypto (BTC-USD), and indices (^IXIC)
    """
    
    def __init__(self):
        self.config = get_config()
        self._cache: Dict[str, Any] = {}
        self._cache_times: Dict[str, datetime] = {}
        
        # Ensure cache directory exists
        os.makedirs(self.config.scraping.cache_directory, exist_ok=True)
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self._cache_times:
            return False
        cache_age = datetime.now() - self._cache_times[key]
        return cache_age < timedelta(hours=self.config.scraping.cache_ttl_hours)
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get data from cache if valid"""
        if self._is_cache_valid(key):
            return self._cache.get(key)
        return None
    
    def _set_cache(self, key: str, data: Any):
        """Set data in cache"""
        self._cache[key] = data
        self._cache_times[key] = datetime.now()
    
    def _detect_asset_type(self, symbol: str) -> str:
        """Detect if symbol is stock, crypto, or index"""
        symbol_upper = symbol.upper()
        
        if symbol_upper.startswith("^"):
            return "index"
        elif "-USD" in symbol_upper or "BTC" in symbol_upper or "ETH" in symbol_upper:
            return "crypto"
        else:
            return "stock"
    
    def get_stock(self, symbol: str) -> Optional[StockQuote]:
        """
        Get current quote data using yfinance (works for stocks, crypto, indices)
        
        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'BTC-USD', '^IXIC')
            
        Returns:
            StockQuote object or None if failed
        """
        cache_key = f"quote_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            # Use yfinance to get data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Detect asset type
            asset_type = self._detect_asset_type(symbol)
            
            # Get current price
            current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose', 0)
            
            # Get previous close for change calculation
            prev_close = info.get('previousClose', current_price)
            
            # Calculate change
            change = current_price - prev_close
            change_percent = (change / prev_close * 100) if prev_close != 0 else 0
            
            # Get volume
            volume = info.get('volume') or info.get('regularMarketVolume', 0)
            avg_volume = info.get('averageVolume') or info.get('averageDailyVolume10Day', 0)
            
            # Format market cap
            market_cap = info.get('marketCap')
            if market_cap:
                if market_cap >= 1e12:
                    market_cap_str = f"${market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    market_cap_str = f"${market_cap/1e9:.2f}B"
                elif market_cap >= 1e6:
                    market_cap_str = f"${market_cap/1e6:.2f}M"
                else:
                    market_cap_str = f"${market_cap:,.0f}"
            else:
                market_cap_str = "N/A"
            
            # Get name
            name = info.get('longName') or info.get('shortName') or symbol
            
            # Get optional fields
            pe_ratio = info.get('trailingPE') or info.get('forwardPE')
            eps = info.get('trailingEps') or info.get('epsTrailingTwelveMonths')
            dividend_yield = info.get('dividendYield')
            if dividend_yield:
                dividend_yield = dividend_yield * 100  # Convert to percentage
            
            fifty_two_week_high = info.get('fiftyTwoWeekHigh')
            fifty_two_week_low = info.get('fiftyTwoWeekLow')
            
            quote = StockQuote(
                symbol=symbol,
                name=name,
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=volume if volume else 0,
                avg_volume=avg_volume if avg_volume else 0,
                market_cap=market_cap_str,
                pe_ratio=pe_ratio,
                eps=eps,
                dividend_yield=dividend_yield,
                fifty_two_week_high=fifty_two_week_high,
                fifty_two_week_low=fifty_two_week_low,
                timestamp=datetime.now().isoformat(),
                asset_type=asset_type
            )
            
            self._set_cache(cache_key, quote)
            return quote
            
        except Exception as e:
            print(f"Error fetching {symbol} with yfinance: {e}")
            return None
    
    def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1mo",
        interval: str = "1d"
    ) -> List[HistoricalPrice]:
        """
        Get historical price data using yfinance
        
        Args:
            symbol: Ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            interval: Data interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)
            
        Returns:
            List of HistoricalPrice objects
        """
        cache_key = f"history_{symbol}_{period}_{interval}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return []
            
            historical = []
            for date, row in df.iterrows():
                try:
                    historical.append(HistoricalPrice(
                        date=date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date),
                        open=float(row['Open']) if pd.notna(row['Open']) else 0,
                        high=float(row['High']) if pd.notna(row['High']) else 0,
                        low=float(row['Low']) if pd.notna(row['Low']) else 0,
                        close=float(row['Close']) if pd.notna(row['Close']) else 0,
                        volume=int(row['Volume']) if pd.notna(row['Volume']) else 0,
                        adj_close=float(row['Close']) if pd.notna(row['Close']) else 0
                    ))
                except Exception as e:
                    print(f"Error parsing historical data row: {e}")
                    continue
            
            self._set_cache(cache_key, historical)
            return historical
            
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return []
    
    def get_multiple_stocks(self, symbols: List[str]) -> Dict[str, Optional[StockQuote]]:
        """
        Get quotes for multiple symbols
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            Dictionary mapping symbol to StockQuote
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_stock(symbol)
            time.sleep(0.1)  # Small delay to avoid rate limiting
        return results
    
    def search_stocks(self, query: str) -> List[Dict]:
        """
        Search for stocks by name or symbol using Yahoo Finance API
        """
        try:
            url = "https://query2.finance.yahoo.com/v1/finance/search"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            params = {
                'q': query,
                'quotesCount': 5,
                'newsCount': 0,
                'enableFuzzyQuery': 'true',
                'quotesQueryId': 'tss_match_phrase_query'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=5)
            data = response.json()
            
            results = []
            if 'quotes' in data:
                for quote in data['quotes']:
                    # Filter for relevant asset types
                    if quote.get('quoteType') in ['EQUITY', 'INDEX', 'CRYPTOCURRENCY', 'ETF', 'MUTUALFUND']:
                        results.append({
                            "symbol": quote.get('symbol'),
                            "name": quote.get('shortname') or quote.get('longname'),
                            "type": quote.get('quoteType'),
                            "exchange": quote.get('exchange'),
                            "score": quote.get('score', 0)
                        })
            
            return results
            
        except Exception as e:
            print(f"Error searching stocks for '{query}': {e}")
            return []

    def get_ticker_from_name(self, query: str) -> Optional[str]:
        """
        Best-effort attempt to get a ticker from a company name/query
        """
        results = self.search_stocks(query)
        if results:
            # Return the top result's symbol
            return results[0]['symbol']
        return None
    
    def clear_cache(self):
        """Clear all cached data"""
        self._cache.clear()
        self._cache_times.clear()


if __name__ == "__main__":
    # Test the scraper
    scraper = YahooScraper()
    
    print("Testing Yahoo Finance Scraper with yfinance...")
    print("-" * 60)
    
    # Test stock
    print("\n1. Testing Stock (AAPL):")
    quote = scraper.get_stock("AAPL")
    if quote:
        print(quote.to_text())
    
    # Test crypto
    print("\n" + "=" * 60)
    print("\n2. Testing Crypto (BTC-USD):")
    btc = scraper.get_stock("BTC-USD")
    if btc:
        print(btc.to_text())
    
    # Test index
    print("\n" + "=" * 60)
    print("\n3. Testing Index (^IXIC - NASDAQ):")
    nasdaq = scraper.get_stock("^IXIC")
    if nasdaq:
        print(nasdaq.to_text())
    
    # Test historical data
    print("\n" + "=" * 60)
    print("\n4. Testing Historical Data (TSLA, 5 days):")
    history = scraper.get_historical_data("TSLA", period="5d")
    if history:
        for h in history[-5:]:
            print(f"  {h.date}: Close=${h.close:.2f}, Volume={h.volume:,}")
