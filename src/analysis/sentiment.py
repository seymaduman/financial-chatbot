"""
Sentiment Analyzer - Advanced sentiment scoring for financial text
Correlates sentiment with price movements
"""
import os
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import get_config


@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    text: str
    score: float  # -1.0 to 1.0
    label: str  # positive, negative, neutral
    confidence: float  # 0.0 to 1.0
    positive_keywords: List[str]
    negative_keywords: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "score": self.score,
            "label": self.label,
            "confidence": self.confidence,
            "positive_keywords": self.positive_keywords,
            "negative_keywords": self.negative_keywords
        }


@dataclass  
class AggregatedSentiment:
    """Aggregated sentiment across multiple texts"""
    total_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    average_score: float
    weighted_score: float
    label: str
    trend: str  # improving, declining, stable


class SentimentAnalyzer:
    """
    Financial sentiment analyzer with domain-specific lexicons
    Calculates sentiment scores and correlates with market movements
    """
    
    # Financial domain positive words with weights
    POSITIVE_LEXICON = {
        # Strong positive (weight 2)
        "surge": 2.0, "soar": 2.0, "skyrocket": 2.0, "boom": 2.0, "breakthrough": 2.0,
        "record": 2.0, "exceptional": 2.0, "outstanding": 2.0, "stellar": 2.0,
        
        # Moderate positive (weight 1.5)
        "beat": 1.5, "exceed": 1.5, "outperform": 1.5, "rally": 1.5, "upgrade": 1.5,
        "bullish": 1.5, "optimistic": 1.5, "confident": 1.5, "strong": 1.5,
        
        # Standard positive (weight 1)
        "gain": 1.0, "rise": 1.0, "grow": 1.0, "growth": 1.0, "profit": 1.0,
        "increase": 1.0, "improve": 1.0, "positive": 1.0, "success": 1.0,
        "innovation": 1.0, "expansion": 1.0, "dividend": 1.0, "buyback": 1.0,
        "acquisition": 1.0, "partnership": 1.0, "opportunity": 1.0, "momentum": 1.0,
        "recovery": 1.0, "rebound": 1.0, "upside": 1.0, "support": 1.0
    }
    
    # Financial domain negative words with weights
    NEGATIVE_LEXICON = {
        # Strong negative (weight 2)
        "crash": 2.0, "collapse": 2.0, "plunge": 2.0, "bankruptcy": 2.0, "fraud": 2.0,
        "scandal": 2.0, "catastrophic": 2.0, "devastating": 2.0, "crisis": 2.0,
        
        # Moderate negative (weight 1.5)
        "miss": 1.5, "downgrade": 1.5, "bearish": 1.5, "selloff": 1.5, "sell-off": 1.5,
        "layoff": 1.5, "layoffs": 1.5, "investigation": 1.5, "lawsuit": 1.5,
        
        # Standard negative (weight 1)
        "fall": 1.0, "drop": 1.0, "decline": 1.0, "loss": 1.0, "cut": 1.0,
        "weak": 1.0, "slump": 1.0, "concern": 1.0, "risk": 1.0, "warning": 1.0,
        "recession": 1.0, "inflation": 1.0, "default": 1.0, "debt": 1.0,
        "negative": 1.0, "uncertainty": 1.0, "volatile": 1.0, "pressure": 1.0,
        "disappointing": 1.0, "underperform": 1.0, "downside": 1.0
    }
    
    # Negation words that flip sentiment
    NEGATION_WORDS = {
        "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
        "hardly", "barely", "scarcely", "doesn't", "don't", "didn't",
        "won't", "wouldn't", "couldn't", "shouldn't", "isn't", "aren't",
        "wasn't", "weren't", "hasn't", "haven't", "hadn't", "cannot"
    }
    
    # Intensifier words that amplify sentiment
    INTENSIFIERS = {
        "very": 1.5, "extremely": 2.0, "highly": 1.5, "significantly": 1.5,
        "substantially": 1.5, "dramatically": 2.0, "sharply": 1.5,
        "considerably": 1.5, "remarkably": 1.5, "exceptionally": 2.0
    }
    
    def __init__(self):
        self.config = get_config()
        self._use_textblob = False
        self._textblob = None
        
        # Try to import TextBlob for additional analysis
        try:
            from textblob import TextBlob
            self._textblob = TextBlob
            self._use_textblob = True
        except ImportError:
            pass
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a text
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentResult with score and details
        """
        if not text:
            return SentimentResult(
                text=text,
                score=0.0,
                label="neutral",
                confidence=0.0,
                positive_keywords=[],
                negative_keywords=[]
            )
        
        # Tokenize and clean
        words = self._tokenize(text)
        
        # Calculate lexicon-based sentiment
        pos_score, neg_score, pos_words, neg_words = self._lexicon_score(words)
        
        # Combine with TextBlob if available
        if self._use_textblob:
            tb_score = self._textblob_score(text)
            # Weighted combination (lexicon is more domain-specific)
            final_score = 0.7 * (pos_score - neg_score) + 0.3 * tb_score
        else:
            final_score = pos_score - neg_score
        
        # Normalize to [-1, 1]
        final_score = max(-1.0, min(1.0, final_score))
        
        # Determine label
        if final_score > 0.1:
            label = "positive"
        elif final_score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        # Calculate confidence based on keyword matches
        total_keywords = len(pos_words) + len(neg_words)
        confidence = min(1.0, total_keywords / 5.0)  # Max confidence at 5+ keywords
        
        return SentimentResult(
            text=text,
            score=final_score,
            label=label,
            confidence=confidence,
            positive_keywords=pos_words,
            negative_keywords=neg_words
        )
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment for multiple texts"""
        return [self.analyze(text) for text in texts]
    
    def aggregate_sentiment(
        self, 
        results: List[SentimentResult],
        weights: List[float] = None
    ) -> AggregatedSentiment:
        """
        Aggregate multiple sentiment results
        
        Args:
            results: List of SentimentResult objects
            weights: Optional weights for each result (e.g., by recency)
            
        Returns:
            AggregatedSentiment with overall metrics
        """
        if not results:
            return AggregatedSentiment(
                total_count=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                average_score=0.0,
                weighted_score=0.0,
                label="neutral",
                trend="stable"
            )
        
        # Count by label
        positive_count = sum(1 for r in results if r.label == "positive")
        negative_count = sum(1 for r in results if r.label == "negative")
        neutral_count = sum(1 for r in results if r.label == "neutral")
        
        # Calculate average score
        average_score = sum(r.score for r in results) / len(results)
        
        # Calculate weighted score
        if weights:
            if len(weights) != len(results):
                weights = [1.0] * len(results)
            total_weight = sum(weights)
            weighted_score = sum(r.score * w for r, w in zip(results, weights)) / total_weight
        else:
            weighted_score = average_score
        
        # Determine overall label
        if weighted_score > 0.1:
            label = "positive"
        elif weighted_score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        # Calculate trend (compare first half vs second half)
        trend = "stable"
        if len(results) >= 4:
            mid = len(results) // 2
            first_half_avg = sum(r.score for r in results[:mid]) / mid
            second_half_avg = sum(r.score for r in results[mid:]) / (len(results) - mid)
            
            diff = second_half_avg - first_half_avg
            if diff > 0.15:
                trend = "improving"
            elif diff < -0.15:
                trend = "declining"
        
        return AggregatedSentiment(
            total_count=len(results),
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            average_score=average_score,
            weighted_score=weighted_score,
            label=label,
            trend=trend
        )
    
    def correlate_with_price(
        self,
        sentiment_scores: List[float],
        price_changes: List[float]
    ) -> Dict:
        """
        Calculate correlation between sentiment and price changes
        
        Args:
            sentiment_scores: List of sentiment scores
            price_changes: Corresponding price changes (percentage)
            
        Returns:
            Dictionary with correlation metrics
        """
        if len(sentiment_scores) != len(price_changes) or len(sentiment_scores) < 3:
            return {
                "correlation": 0.0,
                "sample_size": len(sentiment_scores),
                "significant": False,
                "interpretation": "Insufficient data"
            }
        
        # Calculate Pearson correlation
        n = len(sentiment_scores)
        mean_sent = sum(sentiment_scores) / n
        mean_price = sum(price_changes) / n
        
        numerator = sum((s - mean_sent) * (p - mean_price) for s, p in zip(sentiment_scores, price_changes))
        
        sent_var = sum((s - mean_sent) ** 2 for s in sentiment_scores)
        price_var = sum((p - mean_price) ** 2 for p in price_changes)
        
        denominator = (sent_var * price_var) ** 0.5
        
        if denominator == 0:
            correlation = 0.0
        else:
            correlation = numerator / denominator
        
        # Interpret correlation
        abs_corr = abs(correlation)
        if abs_corr > 0.7:
            interpretation = "Strong correlation"
        elif abs_corr > 0.4:
            interpretation = "Moderate correlation"
        elif abs_corr > 0.2:
            interpretation = "Weak correlation"
        else:
            interpretation = "No significant correlation"
        
        return {
            "correlation": correlation,
            "sample_size": n,
            "significant": abs_corr > 0.3,
            "interpretation": interpretation,
            "direction": "positive" if correlation > 0 else "negative"
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        text = text.lower()
        # Keep hyphens in words
        words = re.findall(r'\b[\w-]+\b', text)
        return words
    
    def _lexicon_score(self, words: List[str]) -> Tuple[float, float, List[str], List[str]]:
        """Calculate sentiment using lexicon"""
        pos_score = 0.0
        neg_score = 0.0
        pos_words = []
        neg_words = []
        
        negation_window = 0  # Track negation effect
        intensifier = 1.0  # Track current intensifier
        
        for i, word in enumerate(words):
            # Check for negation
            if word in self.NEGATION_WORDS:
                negation_window = 3  # Affect next 3 words
                continue
            
            # Check for intensifier
            if word in self.INTENSIFIERS:
                intensifier = self.INTENSIFIERS[word]
                continue
            
            # Check positive lexicon
            if word in self.POSITIVE_LEXICON:
                weight = self.POSITIVE_LEXICON[word] * intensifier
                if negation_window > 0:
                    # Negated positive becomes negative
                    neg_score += weight * 0.7
                    neg_words.append(f"not {word}")
                else:
                    pos_score += weight
                    pos_words.append(word)
            
            # Check negative lexicon
            elif word in self.NEGATIVE_LEXICON:
                weight = self.NEGATIVE_LEXICON[word] * intensifier
                if negation_window > 0:
                    # Negated negative becomes positive
                    pos_score += weight * 0.5
                    pos_words.append(f"not {word}")
                else:
                    neg_score += weight
                    neg_words.append(word)
            
            # Reset intensifier after use
            intensifier = 1.0
            
            # Decrement negation window
            if negation_window > 0:
                negation_window -= 1
        
        # Normalize scores
        total = pos_score + neg_score
        if total > 0:
            pos_score = pos_score / (total + 1)
            neg_score = neg_score / (total + 1)
        
        return pos_score, neg_score, pos_words, neg_words
    
    def _textblob_score(self, text: str) -> float:
        """Get TextBlob sentiment score"""
        if not self._use_textblob:
            return 0.0
        
        try:
            blob = self._textblob(text)
            return blob.sentiment.polarity
        except:
            return 0.0


if __name__ == "__main__":
    # Test the sentiment analyzer
    print("Testing Sentiment Analyzer...")
    print("-" * 50)
    
    analyzer = SentimentAnalyzer()
    
    # Test individual texts
    test_texts = [
        "Apple stock surged 10% after exceptional earnings beat analyst expectations",
        "Tesla shares crashed amid concerns about declining demand and rising competition",
        "Microsoft reported steady revenue growth in cloud services",
        "The market showed mixed signals today with volatile trading",
        "Amazon announced massive layoffs amid economic uncertainty",
        "The company did not meet revenue expectations despite strong growth"
    ]
    
    results = []
    for text in test_texts:
        result = analyzer.analyze(text)
        results.append(result)
        print(f"\nText: {text[:60]}...")
        print(f"  Score: {result.score:+.3f} | Label: {result.label}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Keywords (+): {result.positive_keywords}")
        print(f"  Keywords (-): {result.negative_keywords}")
    
    print("\n" + "-" * 50)
    
    # Test aggregation
    print("\nAggregated Sentiment:")
    agg = analyzer.aggregate_sentiment(results)
    print(f"  Total: {agg.total_count}")
    print(f"  Positive: {agg.positive_count} | Negative: {agg.negative_count} | Neutral: {agg.neutral_count}")
    print(f"  Average Score: {agg.average_score:+.3f}")
    print(f"  Overall Label: {agg.label}")
    print(f"  Trend: {agg.trend}")
    
    print("\n" + "-" * 50)
    
    # Test correlation
    print("\nSentiment-Price Correlation Test:")
    sentiment_scores = [0.5, 0.3, -0.2, 0.6, -0.4]
    price_changes = [2.1, 1.5, -1.0, 3.2, -2.5]
    
    corr = analyzer.correlate_with_price(sentiment_scores, price_changes)
    print(f"  Correlation: {corr['correlation']:.3f}")
    print(f"  Interpretation: {corr['interpretation']}")
