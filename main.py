"""
RAG Financial Chatbot - Main Entry Point
Interactive CLI for financial analysis and market intelligence
"""
import os
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from config import get_config, Config
from src.chatbot.controller import ChatController


def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║          RAG Financial Intelligence Chatbot                  ║
║      Powered by Ollama LLM with Real-Time Market Data        ║
╚══════════════════════════════════════════════════════════════╝

Type 'help' for available commands, 'quit' to exit.
    """
    print(banner)


def print_status(config: Config, controller: ChatController):
    """Print system status"""
    print("\n📊 System Status:")
    print(f"   Model: {config.ollama.model}")
    print(f"   Temperature: {config.ollama.temperature}")
    print(f"   Top K: {config.ollama.top_k}")
    print(f"   Top P: {config.ollama.top_p}")
    
    # Check Ollama connection
    if controller.llm_client.health_check():
        print("   Ollama: ✓ Connected")
    else:
        print("   Ollama: ✗ Not available")
    print()


def run_interactive():
    """Run interactive chat loop"""
    config = get_config()
    controller = ChatController()
    
    print_banner()
    print_status(config, controller)
    
    print("Ready! Ask me about stocks, news, or market trends.\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You > ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! Happy investing! 📈")
                break
            
            if user_input.lower() == 'clear':
                controller.clear_history()
                print("✓ Conversation history cleared.\n")
                continue
            
            if user_input.lower() == 'status':
                print_status(config, controller)
                continue
            
            # Process query
            print("\n⏳ Analyzing...\n")
            response = controller.process_query(user_input)
            
            # Print response
            print("─" * 60)
            print(response.text)
            print("─" * 60)
            
            # Print metadata
            print(f"\n[Intent: {response.intent.value} | Confidence: {response.confidence:.0%}]\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Happy investing! 📈")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again.\n")


def run_single_query(query: str):
    """Run a single query and exit"""
    controller = ChatController()
    
    print(f"\nQuery: {query}\n")
    print("⏳ Analyzing...\n")
    
    response = controller.process_query(query)
    
    print("─" * 60)
    print(response.text)
    print("─" * 60)
    print(f"\n[Intent: {response.intent.value} | Confidence: {response.confidence:.0%}]")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RAG Financial Chatbot - Market Intelligence System"
    )
    
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Single query mode - ask a question and exit"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        help="Set generation temperature (0.0-1.0)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        help="Set top_k parameter for retrieval"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        help="Set top_p parameter for generation"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Set maximum tokens for response"
    )
    
    args = parser.parse_args()
    
    # Apply config overrides
    config = get_config()
    
    if args.temperature is not None:
        config.ollama.temperature = args.temperature
    if args.top_k is not None:
        config.ollama.top_k = args.top_k
    if args.top_p is not None:
        config.ollama.top_p = args.top_p
    if args.max_tokens is not None:
        config.ollama.max_tokens = args.max_tokens
    
    # Run appropriate mode
    if args.query:
        run_single_query(args.query)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
