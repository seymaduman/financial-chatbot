"""
Analysis Module - Sentiment scoring, market prediction, and trend analysis
"""
from .sentiment import SentimentAnalyzer
from .predictor import MarketPredictor
from .trends import TrendAnalyzer

__all__ = ["SentimentAnalyzer", "MarketPredictor", "TrendAnalyzer"]
