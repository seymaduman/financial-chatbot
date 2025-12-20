"""
Trend Analyzer - Pattern recognition for price and sentiment trends
"""
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import get_config


@dataclass
class TrendResult:
    """Trend analysis result"""
    trend: str  # bullish, bearish, sideways
    strength: float  # 0-1
    support_level: Optional[float]
    resistance_level: Optional[float]
    description: str
    signals: List[str]

    def to_dict(self) -> Dict:
        return {
            "trend": self.trend,
            "strength": self.strength,
            "support": self.support_level,
            "resistance": self.resistance_level,
            "description": self.description,
            "signals": self.signals
        }


class TrendAnalyzer:
    """Analyzes price trends and patterns"""
    
    def __init__(self):
        self.config = get_config()
    
    def analyze_price_trend(self, prices: List[float], dates: List[str] = None) -> TrendResult:
        """Analyze price trend from historical data"""
        if not prices or len(prices) < 3:
            return TrendResult(
                trend="unknown", strength=0.0, support_level=None,
                resistance_level=None, description="Insufficient data",
                signals=[]
            )
        
        signals = []
        
        # Calculate trend direction
        start_price = sum(prices[:3]) / 3
        end_price = sum(prices[-3:]) / 3
        change_pct = (end_price - start_price) / start_price * 100
        
        # Moving averages
        short_ma = sum(prices[-5:]) / min(5, len(prices))
        long_ma = sum(prices[-20:]) / min(20, len(prices))
        
        if short_ma > long_ma:
            signals.append("Short MA above Long MA (bullish)")
        else:
            signals.append("Short MA below Long MA (bearish)")
        
        # Support and resistance
        support = min(prices[-20:]) if len(prices) >= 20 else min(prices)
        resistance = max(prices[-20:]) if len(prices) >= 20 else max(prices)
        
        current_price = prices[-1]
        near_support = (current_price - support) / current_price < 0.05
        near_resistance = (resistance - current_price) / current_price < 0.05
        
        if near_support:
            signals.append("Price near support level")
        if near_resistance:
            signals.append("Price near resistance level")
        
        # Determine trend
        if change_pct > 5:
            trend = "bullish"
            strength = min(1.0, change_pct / 20)
            description = f"Uptrend detected ({change_pct:+.1f}% change)"
        elif change_pct < -5:
            trend = "bearish"
            strength = min(1.0, abs(change_pct) / 20)
            description = f"Downtrend detected ({change_pct:+.1f}% change)"
        else:
            trend = "sideways"
            strength = 0.3
            description = "Consolidation/sideways movement"
        
        # Volatility check
        if len(prices) >= 5:
            avg_price = sum(prices) / len(prices)
            volatility = (max(prices) - min(prices)) / avg_price
            if volatility > 0.15:
                signals.append(f"High volatility ({volatility:.1%})")
        
        return TrendResult(
            trend=trend,
            strength=strength,
            support_level=support,
            resistance_level=resistance,
            description=description,
            signals=signals
        )

    def analyze_sentiment_trend(self, sentiment_scores: List[float]) -> TrendResult:
        """Analyze sentiment trend over time"""
        if not sentiment_scores or len(sentiment_scores) < 3:
            return TrendResult(
                trend="unknown", strength=0.0, support_level=None,
                resistance_level=None, description="Insufficient sentiment data",
                signals=[]
            )
        
        signals = []
        
        # Compare recent vs older sentiment
        mid = len(sentiment_scores) // 2
        old_avg = sum(sentiment_scores[:mid]) / mid
        new_avg = sum(sentiment_scores[mid:]) / (len(sentiment_scores) - mid)
        
        change = new_avg - old_avg
        current_avg = sum(sentiment_scores[-3:]) / min(3, len(sentiment_scores))
        
        if current_avg > 0.3:
            signals.append("Recent sentiment strongly positive")
        elif current_avg < -0.3:
            signals.append("Recent sentiment strongly negative")
        
        if change > 0.2:
            trend = "improving"
            description = "Sentiment improving over time"
        elif change < -0.2:
            trend = "declining"
            description = "Sentiment declining over time"
        else:
            trend = "stable"
            description = "Sentiment relatively stable"
        
        strength = min(1.0, abs(change) * 2)
        
        return TrendResult(
            trend=trend,
            strength=strength,
            support_level=None,
            resistance_level=None,
            description=description,
            signals=signals
        )

    def get_trend_summary(self, symbol: str, price_trend: TrendResult, 
                          sentiment_trend: TrendResult = None) -> str:
        """Generate natural language trend summary"""
        lines = [f"=== TREND ANALYSIS: {symbol} ===", ""]
        
        # Price trend
        icon = "📈" if price_trend.trend == "bullish" else "📉" if price_trend.trend == "bearish" else "➡️"
        lines.append(f"{icon} Price Trend: {price_trend.trend.upper()}")
        lines.append(f"   Strength: {price_trend.strength:.0%}")
        lines.append(f"   {price_trend.description}")
        
        if price_trend.support_level:
            lines.append(f"   Support: ${price_trend.support_level:.2f}")
        if price_trend.resistance_level:
            lines.append(f"   Resistance: ${price_trend.resistance_level:.2f}")
        
        if price_trend.signals:
            lines.append("   Signals:")
            for signal in price_trend.signals:
                lines.append(f"   • {signal}")
        
        if sentiment_trend:
            lines.append("")
            lines.append(f"💭 Sentiment Trend: {sentiment_trend.trend.upper()}")
            lines.append(f"   {sentiment_trend.description}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    print("Testing Trend Analyzer...")
    print("-" * 50)
    
    analyzer = TrendAnalyzer()
    
    # Test price trend
    prices = [150.0, 152.0, 148.0, 155.0, 158.0, 162.0, 160.0, 165.0, 170.0, 175.0]
    price_trend = analyzer.analyze_price_trend(prices)
    
    print("\nPrice Trend Analysis:")
    print(f"  Trend: {price_trend.trend}")
    print(f"  Strength: {price_trend.strength:.2f}")
    print(f"  Support: ${price_trend.support_level:.2f}")
    print(f"  Resistance: ${price_trend.resistance_level:.2f}")
    print(f"  Signals: {price_trend.signals}")
    
    # Test sentiment trend
    sentiments = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    sentiment_trend = analyzer.analyze_sentiment_trend(sentiments)
    
    print("\nSentiment Trend:")
    print(f"  Trend: {sentiment_trend.trend}")
    print(f"  {sentiment_trend.description}")
    
    # Test summary
    print("\n" + analyzer.get_trend_summary("AAPL", price_trend, sentiment_trend))
