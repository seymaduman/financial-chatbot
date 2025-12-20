"""
Response Formatter - Formats chatbot responses with structure and citations
"""
import os
import re
from typing import Dict, List, Optional
from enum import Enum

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class ResponseFormatter:
    """Formats responses with proper structure, sections, and disclaimers"""
    
    DISCLAIMER = "\n\n⚠️ **Disclaimer**: This is NOT financial advice. Please consult a qualified financial professional before making investment decisions."
    
    def __init__(self):
        pass
    
    def format_response(
        self,
        text: str,
        intent: "Enum",
        ticker: str = None,
        include_disclaimer: bool = False,
        sources: List[Dict] = None
    ) -> str:
        """
        Format the response with proper structure
        
        Args:
            text: Raw response text
            intent: Query intent type
            ticker: Related ticker symbol
            include_disclaimer: Whether to add disclaimer
            sources: Source citations
            
        Returns:
            Formatted response text
        """
        # Clean up the text
        formatted = self._clean_text(text)
        
        # Add structure if needed
        formatted = self._add_structure(formatted, intent)
        
        # Add ticker header if relevant
        if ticker:
            formatted = self._add_ticker_context(formatted, ticker)
        
        # Add sources if provided
        if sources:
            formatted = self._add_sources(formatted, sources)
        
        # Add disclaimer for predictions
        if include_disclaimer:
            formatted += self.DISCLAIMER
        
        return formatted
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize response text"""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix markdown formatting issues
        text = re.sub(r'\*{3,}', '**', text)
        
        # Ensure proper spacing around headers
        text = re.sub(r'(#{1,3})([^\s#])', r'\1 \2', text)
        
        return text.strip()
    
    def _add_structure(self, text: str, intent: "Enum") -> str:
        """Add section structure based on intent"""
        intent_name = intent.value if hasattr(intent, 'value') else str(intent)
        
        # Check if already has headers
        if re.search(r'^#+\s', text, re.MULTILINE):
            return text
        
        # Add headers based on content
        if "predict" in intent_name or "📈" in text or "📉" in text:
            # Prediction response - already formatted
            return text
        
        return text
    
    def _add_ticker_context(self, text: str, ticker: str) -> str:
        """Add ticker context indicator"""
        # Don't add if already mentioned
        if ticker.upper() in text.upper()[:100]:
            return text
        
        return f"**[{ticker}]**\n\n{text}"
    
    def _add_sources(self, text: str, sources: List[Dict]) -> str:
        """Add source citations"""
        if not sources:
            return text
        
        source_section = "\n\n---\n**Sources:**\n"
        for i, source in enumerate(sources, 1):
            source_type = source.get("type", "unknown")
            source_id = source.get("source_id", "")
            source_section += f"{i}. [{source_type}] {source_id}\n"
        
        return text + source_section
    
    def format_stock_quote(self, quote: Dict) -> str:
        """Format stock quote data"""
        lines = [
            f"## {quote.get('symbol', 'N/A')} - {quote.get('name', 'Unknown')}",
            "",
            f"**Price:** ${quote.get('price', 0):.2f}",
            f"**Change:** {quote.get('change', 0):+.2f} ({quote.get('change_percent', 0):+.2f}%)",
            f"**Volume:** {quote.get('volume', 0):,}",
            f"**Market Cap:** {quote.get('market_cap', 'N/A')}",
            ""
        ]
        
        if quote.get('pe_ratio'):
            lines.append(f"**P/E Ratio:** {quote['pe_ratio']:.2f}")
        if quote.get('eps'):
            lines.append(f"**EPS:** ${quote['eps']:.2f}")
        if quote.get('dividend_yield'):
            lines.append(f"**Dividend Yield:** {quote['dividend_yield']:.2f}%")
        
        return "\n".join(lines)
    
    def format_news_list(self, articles: List[Dict]) -> str:
        """Format list of news articles"""
        if not articles:
            return "No recent news available."
        
        lines = ["## Latest News", ""]
        
        for article in articles[:5]:
            sentiment = article.get('sentiment', 'neutral')
            icon = "🟢" if sentiment == "positive" else "🔴" if sentiment == "negative" else "⚪"
            
            title = article.get('title', 'No title')
            source = article.get('source', 'Unknown')
            
            lines.append(f"{icon} **{title}**")
            lines.append(f"   *{source}*")
            lines.append("")
        
        return "\n".join(lines)
    
    def format_prediction(self, prediction: Dict) -> str:
        """Format market prediction"""
        direction = prediction.get('direction', 'NEUTRAL')
        confidence = prediction.get('confidence', 0)
        summary = prediction.get('summary', '')
        
        lines = [
            "## Market Direction Prediction",
            "",
            f"**Direction:** {direction}",
            f"**Confidence:** {confidence:.0f}%",
            "",
            f"**Summary:** {summary}",
            ""
        ]
        
        if prediction.get('reasoning'):
            lines.append("### Reasoning:")
            for step in prediction['reasoning']:
                lines.append(f"- {step}")
        
        lines.append("")
        lines.append(self.DISCLAIMER)
        
        return "\n".join(lines)
    
    def format_error(self, error_message: str) -> str:
        """Format error message"""
        return f"❌ **Error:** {error_message}\n\nPlease try again or rephrase your question."
    
    def format_loading(self, message: str = "Analyzing...") -> str:
        """Format loading message"""
        return f"⏳ {message}"


if __name__ == "__main__":
    print("Testing Response Formatter...")
    print("-" * 50)
    
    formatter = ResponseFormatter()
    
    # Test quote formatting
    quote = {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "price": 180.50,
        "change": 2.35,
        "change_percent": 1.32,
        "volume": 52000000,
        "market_cap": "$2.8T",
        "pe_ratio": 28.5,
        "eps": 6.33
    }
    
    print("\nFormatted Quote:")
    print(formatter.format_stock_quote(quote))
    
    # Test news formatting
    articles = [
        {"title": "Apple beats earnings", "source": "Reuters", "sentiment": "positive"},
        {"title": "iPhone sales decline", "source": "Bloomberg", "sentiment": "negative"}
    ]
    
    print("\n" + "-" * 50)
    print("Formatted News:")
    print(formatter.format_news_list(articles))
