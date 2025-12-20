"""
Tests for RAG Financial Chatbot
"""
import os
import sys
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import get_config


class TestConfig(unittest.TestCase):
    """Test configuration"""
    
    def test_config_exists(self):
        """Test that config can be loaded"""
        config = get_config()
        self.assertIsNotNone(config)
    
    def test_ollama_config(self):
        """Test Ollama configuration"""
        config = get_config()
        self.assertEqual(config.ollama.model, "gpt-oss:120b-cloud")
        self.assertIsInstance(config.ollama.temperature, float)
        self.assertIsInstance(config.ollama.top_k, int)
    
    def test_update_params(self):
        """Test parameter updates"""
        config = get_config()
        original_temp = config.ollama.temperature
        
        config.update_generation_params(temperature=0.5)
        self.assertEqual(config.ollama.temperature, 0.5)
        
        # Reset
        config.update_generation_params(temperature=original_temp)


class TestSentimentAnalyzer(unittest.TestCase):
    """Test sentiment analysis"""
    
    def setUp(self):
        from src.analysis.sentiment import SentimentAnalyzer
        self.analyzer = SentimentAnalyzer()
    
    def test_positive_sentiment(self):
        """Test positive text detection"""
        result = self.analyzer.analyze("Stock surged 10% on strong earnings beat")
        self.assertEqual(result.label, "positive")
        self.assertGreater(result.score, 0)
    
    def test_negative_sentiment(self):
        """Test negative text detection"""
        result = self.analyzer.analyze("Company announces layoffs amid declining revenue")
        self.assertEqual(result.label, "negative")
        self.assertLess(result.score, 0)
    
    def test_neutral_sentiment(self):
        """Test neutral text detection"""
        result = self.analyzer.analyze("Company reports quarterly results")
        self.assertEqual(result.label, "neutral")


class TestMarketPredictor(unittest.TestCase):
    """Test market prediction"""
    
    def setUp(self):
        from src.analysis.predictor import MarketPredictor, MarketDirection
        self.predictor = MarketPredictor()
        self.MarketDirection = MarketDirection
    
    def test_bullish_prediction(self):
        """Test bullish news detection"""
        news = "Apple beats earnings expectations with record revenue growth"
        result = self.predictor.predict(news, ticker="AAPL")
        self.assertEqual(result.direction, self.MarketDirection.UP)
    
    def test_bearish_prediction(self):
        """Test bearish news detection"""
        news = "Company faces bankruptcy after failed merger"
        result = self.predictor.predict(news)
        self.assertEqual(result.direction, self.MarketDirection.DOWN)
    
    def test_confidence_score(self):
        """Test confidence scoring"""
        news = "Major acquisition announced with strong growth outlook"
        result = self.predictor.predict(news)
        self.assertGreater(result.confidence, 0)
        self.assertLessEqual(result.confidence, 100)


class TestEmbeddings(unittest.TestCase):
    """Test embedding service"""
    
    def setUp(self):
        from src.rag.embeddings import EmbeddingService
        self.service = EmbeddingService()
    
    def test_text_chunking(self):
        """Test text chunking"""
        text = "This is a test. " * 50  # Long text
        chunks = self.service.chunk_text(text, "test_id", "test")
        self.assertGreater(len(chunks), 0)
    
    def test_embedding_generation(self):
        """Test embedding generation"""
        embedding = self.service.embed_text("Apple stock price")
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)
    
    def test_similarity(self):
        """Test similarity calculation"""
        emb1 = self.service.embed_text("Apple stock price went up")
        emb2 = self.service.embed_text("AAPL shares increased")
        emb3 = self.service.embed_text("Weather is sunny today")
        
        sim_same = self.service.similarity(emb1, emb2)
        sim_diff = self.service.similarity(emb1, emb3)
        
        # Similar texts should have higher similarity
        self.assertGreater(sim_same, sim_diff)


class TestPromptBuilder(unittest.TestCase):
    """Test prompt building"""
    
    def setUp(self):
        from src.llm.prompt_builder import PromptBuilder
        self.builder = PromptBuilder()
    
    def test_system_prompt(self):
        """Test system prompt generation"""
        prompt = self.builder.build_system_prompt()
        self.assertIn("Financial", prompt)
        self.assertIn("RAG", prompt)
    
    def test_context_injection(self):
        """Test context injection"""
        self.builder.set_context("Apple is trading at $180")
        prompt = self.builder.build_prompt("What is Apple's price?")
        self.assertIn("$180", prompt)
    
    def test_history_management(self):
        """Test conversation history"""
        self.builder.add_user_message("Hello")
        self.builder.add_assistant_message("Hi there")
        
        history = self.builder.get_history()
        self.assertEqual(len(history), 2)


if __name__ == "__main__":
    unittest.main()
