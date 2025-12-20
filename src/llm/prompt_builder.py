"""
Prompt Builder - Context-aware prompt construction for financial analysis
Manages system prompts, context injection, and conversation history
"""
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import get_config


@dataclass
class Message:
    """Chat message"""
    role: str  # system, user, assistant
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {"role": self.role, "content": self.content}


class PromptBuilder:
    """
    Builds context-aware prompts for financial analysis
    Manages conversation history and RAG context injection
    """
    
    # System prompt for financial analysis
    SYSTEM_PROMPT = """You are an Expert Financial Analyst AI powered by Retrieval-Augmented Generation (RAG).

Your role is to provide accurate, data-grounded financial analysis based strictly on the retrieved information.

CORE PRINCIPLES:
1. BASE ALL ANSWERS ON RETRIEVED DATA - Never fabricate financial numbers or statistics
2. CITE SOURCES - Reference the retrieved documents in your answers
3. BE TRANSPARENT - Clearly distinguish between facts, analysis, and uncertainty
4. PROVIDE CONFIDENCE - When making predictions, include confidence levels (0-100%)

⚠️ **CRITICAL - CONTEXT PRIORITIZATION:**
- **ALWAYS prioritize the CURRENT RETRIEVED CONTEXT over any previous conversation history**
- If the new context is about a DIFFERENT STOCK/ASSET, treat it as a FRESH START
- DO NOT mix data from previous stocks with the current query
- DO NOT use old numbers or metrics from previous conversations
- Each new stock query should be answered ONLY with data from the CURRENT retrieved context
- If asked about Stock A, then Stock B - only use Stock B's data for Stock B questions

RESPONSE STRUCTURE:
- Use clear section headings when appropriate
- Use bullet points for lists
- Highlight key metrics and numbers
- Include relevant context from retrieved data

WHEN DATA IS MISSING:
- Explicitly state what information is unavailable
- Provide best-effort analysis with available data
- Never guess or hallucinate financial figures

DISCLAIMER:
Always remind users that this is NOT financial advice. Investment decisions should be made with professional guidance.

You have access to:
- Real-time and historical stock price data
- Financial news with sentiment analysis
- Company financial statements (income, balance sheet, cash flow)
- Market trend analysis

Current generation parameters are configured for this request. Use them appropriately for your response style."""


    PREDICTION_PROMPT = """
MARKET DIRECTION PREDICTION FRAMEWORK:

When predicting market direction based on news or events, you MUST:

1. ANALYZE the news content:
   - Sector relevance
   - Company exposure
   - Macro vs micro impact
   - Earnings impact
   - Regulatory risk

2. DETECT sentiment signals:
   - Positive indicators
   - Negative indicators  
   - Neutral factors

3. PREDICT direction with reasoning:
   📈 UP - If positive factors outweigh negative
   📉 DOWN - If negative factors outweigh positive
   ➖ NEUTRAL - If balanced or unclear

4. ASSIGN confidence score (0-100%):
   - 80-100%: Very high confidence (strong, clear signals)
   - 60-79%: High confidence (clear direction with some uncertainty)
   - 40-59%: Moderate confidence (mixed signals)
   - 20-39%: Low confidence (unclear, conflicting information)
   - 0-19%: Very low confidence (insufficient data)

5. EXPLAIN your reasoning step-by-step

⚠️ ALWAYS END WITH: "This is NOT financial advice. Please consult a financial professional before making investment decisions."
"""

    def __init__(self, max_history: int = 10):
        self.config = get_config()
        self.max_history = max_history
        self.conversation_history: List[Message] = []
        self._context_documents: List[str] = []
    
    def set_context(self, context_text: str, sources: List[Dict] = None):
        """
        Set the retrieved context for the next prompt
        
        Args:
            context_text: Retrieved document text
            sources: Source metadata for citations
        """
        self._context_documents = []
        
        if context_text:
            self._context_documents.append(context_text)
        
        if sources:
            source_info = "\n\nSOURCES:\n"
            for i, source in enumerate(sources, 1):
                source_info += f"{i}. {source.get('source_type', 'unknown')}: {source.get('source_id', 'unknown')}\n"
            self._context_documents.append(source_info)
    
    def build_system_prompt(self, include_prediction: bool = False) -> str:
        """
        Build the system prompt with optional prediction framework
        
        Args:
            include_prediction: Include market prediction instructions
            
        Returns:
            Complete system prompt
        """
        prompt = self.SYSTEM_PROMPT
        
        # Add generation parameters info
        params = self.config.ollama
        prompt += f"""

Current Parameters:
- Temperature: {params.temperature} (controls response randomness)
- Top K: {params.top_k} (retrieval diversity)
- Top P: {params.top_p} (nucleus sampling threshold)
- Max Tokens: {params.max_tokens} (response length limit)
"""
        
        if include_prediction:
            prompt += "\n" + self.PREDICTION_PROMPT
        
        return prompt
    
    def build_prompt(
        self,
        user_query: str,
        context_text: str = None,
        include_history: bool = True,
        include_prediction: bool = False
    ) -> str:
        """
        Build a complete prompt with context and history
        
        Args:
            user_query: User's question
            context_text: Retrieved context (overrides set_context)
            include_history: Include conversation history
            include_prediction: Include prediction framework
            
        Returns:
            Complete prompt string
        """
        parts = []
        
        # Add context
        context = context_text or "\n".join(self._context_documents)
        if context:
            parts.append("=== RETRIEVED CONTEXT ===")
            parts.append(context)
            parts.append("=== END CONTEXT ===\n")
        
        # Add conversation history
        if include_history and self.conversation_history:
            parts.append("=== CONVERSATION HISTORY ===")
            for msg in self.conversation_history[-self.max_history:]:
                role = msg.role.upper()
                parts.append(f"[{role}]: {msg.content}")
            parts.append("=== END HISTORY ===\n")
        
        # Add current query
        parts.append(f"USER QUERY: {user_query}")
        
        # Add instructions
        parts.append("\nProvide a comprehensive, data-grounded response based on the retrieved context.")
        
        if include_prediction:
            parts.append("Include market direction prediction with confidence score.")
        
        return "\n".join(parts)
    
    def build_messages(
        self,
        user_query: str,
        context_text: str = None,
        include_history: bool = True,
        include_prediction: bool = False
    ) -> List[Dict]:
        """
        Build messages array for chat completion
        
        Args:
            user_query: User's question
            context_text: Retrieved context
            include_history: Include conversation history  
            include_prediction: Include prediction framework
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # System message
        messages.append({
            "role": "system",
            "content": self.build_system_prompt(include_prediction)
        })
        
        # Add conversation history
        if include_history:
            for msg in self.conversation_history[-self.max_history:]:
                messages.append(msg.to_dict())
        
        # Build user message with context
        user_content = []
        
        context = context_text or "\n".join(self._context_documents)
        if context:
            user_content.append("RETRIEVED CONTEXT:")
            user_content.append(context)
            user_content.append("")
        
        user_content.append(f"MY QUESTION: {user_query}")
        
        messages.append({
            "role": "user",
            "content": "\n".join(user_content)
        })
        
        return messages
    
    def add_user_message(self, content: str):
        """Add user message to history"""
        self.conversation_history.append(Message(
            role="user",
            content=content
        ))
        self._trim_history()
    
    def add_assistant_message(self, content: str):
        """Add assistant message to history"""
        self.conversation_history.append(Message(
            role="assistant",
            content=content
        ))
        self._trim_history()
    
    def _trim_history(self):
        """Keep history within max_history limit"""
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        self._context_documents.clear()
    
    def clear_context(self):
        """Clear current context"""
        self._context_documents.clear()
    
    def get_history(self) -> List[Dict]:
        """Get conversation history as list of dicts"""
        return [msg.to_dict() for msg in self.conversation_history]
    
    def build_stock_query_prompt(self, symbol: str, query: str = None) -> str:
        """
        Build a prompt specifically for stock queries
        
        Args:
            symbol: Stock ticker symbol
            query: Optional specific question
            
        Returns:
            Prompt string
        """
        if query:
            return f"Regarding {symbol} stock: {query}"
        else:
            return f"Provide a comprehensive analysis of {symbol} stock including current price, recent performance, key metrics, and any relevant news."
    
    def build_news_analysis_prompt(self, news_text: str, ticker: str = None) -> str:
        """
        Build a prompt for news impact analysis
        
        Args:
            news_text: News article or headline
            ticker: Optional related ticker
            
        Returns:
            Prompt string with prediction request
        """
        prompt_parts = [
            "Analyze the following news and predict its market impact:",
            "",
            f"NEWS: {news_text}",
            ""
        ]
        
        if ticker:
            prompt_parts.append(f"Related Stock: {ticker}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "Please provide:",
            "1. Summary of the news",
            "2. Sector and company impact analysis",
            "3. Market direction prediction (📈 UP / 📉 DOWN / ➖ NEUTRAL)",
            "4. Confidence score (0-100%)",
            "5. Key reasoning points"
        ])
        
        return "\n".join(prompt_parts)
    
    def build_financial_analysis_prompt(self, symbol: str, analysis_type: str = "comprehensive") -> str:
        """
        Build a prompt for financial statement analysis
        
        Args:
            symbol: Stock ticker symbol
            analysis_type: Type of analysis (comprehensive, profitability, debt, growth)
            
        Returns:
            Prompt string
        """
        analysis_types = {
            "comprehensive": "Provide a comprehensive financial health analysis including profitability, liquidity, debt levels, and cash flow.",
            "profitability": "Analyze the profitability metrics including gross margin, operating margin, and net profit margin. Compare with industry standards.",
            "debt": "Analyze the debt structure including debt-to-equity ratio, interest coverage, and overall leverage risk.",
            "growth": "Analyze growth metrics including revenue growth, earnings growth, and expansion indicators."
        }
        
        instruction = analysis_types.get(analysis_type, analysis_types["comprehensive"])
        
        return f"For {symbol}: {instruction}\n\nBase your analysis strictly on the financial statement data provided in the context."


if __name__ == "__main__":
    # Test the prompt builder
    print("Testing Prompt Builder...")
    print("-" * 50)
    
    builder = PromptBuilder()
    
    # Test context setting
    builder.set_context(
        "Apple Inc. (AAPL) is trading at $180. Revenue last quarter was $95B.",
        [{"source_type": "stock", "source_id": "AAPL_quote"}]
    )
    
    # Test prompt building
    prompt = builder.build_prompt("What is Apple's current price?")
    print("Basic Prompt:")
    print(prompt[:500])
    
    print("\n" + "-" * 50)
    
    # Test messages building
    messages = builder.build_messages("What is Apple's current price?")
    print("\nMessages format:")
    for msg in messages:
        print(f"  [{msg['role']}]: {msg['content'][:100]}...")
    
    print("\n" + "-" * 50)
    
    # Test specialized prompts
    print("\nStock Query Prompt:")
    print(builder.build_stock_query_prompt("TSLA", "What is the P/E ratio?"))
    
    print("\n" + "-" * 50)
    
    print("\nNews Analysis Prompt:")
    print(builder.build_news_analysis_prompt(
        "Apple announces record iPhone sales in Q4",
        "AAPL"
    ))
    
    print("\n" + "-" * 50)
    
    # Test system prompt
    print("\nSystem Prompt (first 500 chars):")
    print(builder.build_system_prompt()[:500])
