"""
Market Predictor - News impact analysis and market direction prediction
"""
import os
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import get_config
from src.analysis.sentiment import SentimentAnalyzer


class MarketDirection(Enum):
    UP = "📈 UP"
    DOWN = "📉 DOWN"
    NEUTRAL = "➖ NEUTRAL"


@dataclass
class MarketPrediction:
    direction: MarketDirection
    confidence: float
    summary: str
    sentiment_score: float
    reasoning_steps: List[str]
    
    def to_text(self) -> str:
        lines = [
            "=== MARKET PREDICTION ===",
            f"Direction: {self.direction.value}",
            f"Confidence: {self.confidence:.0f}%",
            f"Summary: {self.summary}",
            "",
            "Reasoning:"
        ]
        for i, step in enumerate(self.reasoning_steps, 1):
            lines.append(f"{i}. {step}")
        lines.append("\n⚠️ This is NOT financial advice.")
        return "\n".join(lines)


class MarketPredictor:
    """Predicts market direction from news"""
    
    def __init__(self):
        self.config = get_config()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def predict(self, news_text: str, ticker: str = None) -> MarketPrediction:
        reasoning = []
        sentiment = self.sentiment_analyzer.analyze(news_text)
        reasoning.append(f"Sentiment: {sentiment.label} (score: {sentiment.score:+.2f})")
        
        # Detect events
        events = self._detect_events(news_text.lower())
        if events:
            reasoning.append(f"Events detected: {', '.join(events)}")
        
        # Determine direction
        if sentiment.score > 0.2:
            direction = MarketDirection.UP
            reasoning.append("Positive sentiment suggests upward movement")
        elif sentiment.score < -0.2:
            direction = MarketDirection.DOWN
            reasoning.append("Negative sentiment suggests downward movement")
        else:
            direction = MarketDirection.NEUTRAL
            reasoning.append("Mixed/neutral signals")
        
        # Calculate confidence
        confidence = min(100, 40 + abs(sentiment.score) * 50 + len(events) * 10)
        
        summary = f"{'Bullish' if direction == MarketDirection.UP else 'Bearish' if direction == MarketDirection.DOWN else 'Neutral'} outlook based on news analysis"
        
        return MarketPrediction(
            direction=direction,
            confidence=confidence,
            summary=summary,
            sentiment_score=sentiment.score,
            reasoning_steps=reasoning
        )
    
    def _detect_events(self, text: str) -> List[str]:
        events = []
        if re.search(r'beat|exceed.*earnings', text):
            events.append("earnings_beat")
        if re.search(r'miss.*earnings', text):
            events.append("earnings_miss")
        if re.search(r'layoff|job cuts', text):
            events.append("layoffs")
        if re.search(r'acquisition|acquire', text):
            events.append("acquisition")
        return events
