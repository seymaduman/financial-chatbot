"""
RAG Financial Chatbot - Financial Intelligence Dashboard
Professional layout with sidebar navigation and quick analysis panel
"""
import streamlit as st
import sys
import os
from datetime import datetime
from typing import List, Dict
import plotly.graph_objects as go
import re
import uuid

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from config import get_config
from config import get_config
from src.chatbot.controller import ChatController
from src.data.supported_tickers import SUPPORTED_TICKERS, get_ticker_label


# Page configuration
st.set_page_config(
    page_title="RAG Financial Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional design
st.markdown("""
<style>
    :root {
        --primary-blue: #1E40AF;
        --light-blue: #3B82F6;
        --white: #FFFFFF;
        --light-gray: #F3F4F6;
        --medium-gray: #E5E7EB;
        --dark-gray: #6B7280;
        --text-dark: #1F2937;
    }
    
    .stApp {
        background-color: var(--light-gray);
    }
    
    /* Fixed Header */
    .main-header {
        background: white;
        padding: 1.5rem 2rem;
        border-bottom: 2px solid var(--medium-gray);
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: var(--primary-blue);
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: white;
        border-right: 2px solid var(--medium-gray);
    }
    
    /* Chat messages */
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, var(--primary-blue), var(--light-blue));
        color: white;
        margin-left: 2rem;
    }
    
    .stChatMessage[data-testid="assistant-message"] {
        background-color: white;
        border: 2px solid var(--medium-gray);
        margin-right: 2rem;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, var(--primary-blue), var(--light-blue));
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(30, 64, 175, 0.3);
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 64, 175, 0.4);
    }
    
    /* Quick Analysis Panel */
    .quick-panel {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid var(--medium-gray);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .quick-panel h3 {
        color: var(--primary-blue);
        font-weight: 700;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    
    if "chat_counter" not in st.session_state:
        st.session_state.chat_counter = 0
    
    if "current_chat_id" not in st.session_state:
        # Create first chat
        create_first_chat()
    
    if "previous_ticker" not in st.session_state:
        st.session_state.previous_ticker = None
    
    if "show_chart" not in st.session_state:
        st.session_state.show_chart = False
    
    if "config" not in st.session_state:
        st.session_state.config = get_config()
    
    if "delete_confirm_id" not in st.session_state:
        st.session_state.delete_confirm_id = None

    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = None


def create_first_chat():
    """Create the first chat session"""
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat_id = chat_id
    st.session_state.chats[chat_id] = {
        "title": "Chat",  # Will be numbered based on position
        "messages": [],
        "created_at": datetime.now().isoformat(),
        "controller": ChatController()
    }


def create_new_chat():
    """Create a new chat session"""
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat_id = chat_id
    st.session_state.chats[chat_id] = {
        "title": "Chat",  # Will be numbered based on position
        "messages": [],
        "created_at": datetime.now().isoformat(),
        "controller": ChatController()
    }
    st.session_state.previous_ticker = None
    st.session_state.show_chart = False
    st.rerun()


def switch_chat(chat_id: str):
    """Switch to a different chat"""
    st.session_state.current_chat_id = chat_id
    st.session_state.previous_ticker = None
    st.session_state.show_chart = False
    st.rerun()


def get_current_chat():
    """Get the current chat session"""
    return st.session_state.chats[st.session_state.current_chat_id]


def extract_ticker_from_query(query: str, controller=None) -> str:
    """Extract ticker symbol from query using ChatController logic"""
    if controller:
        # Use the robust extraction from controller (includes regex and dynamic search)
        tickers = controller._extract_tickers(query)
        if tickers:
            return tickers[0]
        return None
    
    # Fallback for legacy calls
    ticker_patterns = [
        r'\$([A-Z\-\^][A-Z0-9\-\.]{0,10})\b',
        r'\b([A-Z]{3,5})\b',
        r'\b([A-Z]{2,4})-USD\b',
        r'\^([A-Z]{2,10})\b'
    ]
    
    query_upper = query.upper()
    for pattern in ticker_patterns:
        match = re.search(pattern, query_upper)
        if match:
            return match.group(0).replace('$', '')
    
    return None


def send_message(query: str):
    """Process and send a message"""
    if not query.strip():
        return
    
    current_chat = get_current_chat()
    
    # Determine ticker: Explicit selection takes precedence, otherwise extract from query
    selected_ticker = st.session_state.get("selected_ticker")
    extracted = extract_ticker_from_query(query, current_chat["controller"])
    
    # Logic: 
    # 1. If user selected a ticker in dropdown, use that.
    # 2. If user mentions a DIFFERENT ticker in query, maybe warn or switch? 
    #    For now, let's assume dropdown is the source of truth if set.
    
    current_ticker = selected_ticker if selected_ticker else extracted
    
    if current_ticker and current_ticker != st.session_state.previous_ticker:
        if st.session_state.previous_ticker:
            st.info(f"🔄 Context switched to {current_ticker}. Previous context cleared.")
            current_chat["messages"] = []
            current_chat["controller"].clear_history()
            st.session_state.show_chart = False
        st.session_state.previous_ticker = current_ticker
    
    # Create simple override context if ticker is set
    override_context = {"ticker": current_ticker} if current_ticker else None
    
    # Add user message
    current_chat["messages"].append({"role": "user", "content": query})
    
    # Process with chatbot
    try:
        with st.spinner("🔄 Analyzing your query..."):
            # Pass the explicit ticker context to the controller
            response = current_chat["controller"].process_query(query, override_context=override_context)
        
        # Add assistant response
        current_chat["messages"].append({
            "role": "assistant",
            "content": response.text,
            "metadata": {
                "intent": response.intent.value,
                "confidence": response.confidence
            }
        })
        
        if "trend" in query.lower() or "chart" in query.lower():
            st.session_state.show_chart = True
            
    except Exception as e:
        error_message = f"""
        ❌ **Error Processing Request**
        
        I encountered an issue: {str(e)}
        
        Please verify the stock symbol and try again.
        """
        
        current_chat["messages"].append({
            "role": "assistant",
            "content": error_message,
            "metadata": {"intent": "error", "confidence": 0.0}
        })


def display_message(role: str, content: str, metadata: Dict = None):
    """Display a chat message"""
    with st.chat_message(role):
        st.markdown(content)
        if metadata and role == "assistant":
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"**Intent:** {metadata.get('intent', 'N/A')}")
            with col2:
                confidence = metadata.get("confidence", 0)
                st.caption(f"**Confidence:** {confidence:.0%}")


def main():
    """Main application"""
    
    # Initialize session state
    init_session_state()
    
    current_chat = get_current_chat()
    
    # Fixed Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 Financial Chat</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            create_new_chat()
        
        st.divider()

        # Ticker Selection
        st.subheader("🔍 Asset Selection")
        
        # Format options for display
        ticker_options = ["None (Auto-detect)"] + [t["label"] for t in SUPPORTED_TICKERS]
        
        # Find current index
        current_index = 0
        if st.session_state.get("selected_ticker"):
            label = get_ticker_label(st.session_state.selected_ticker)
            if label in ticker_options:
                current_index = ticker_options.index(label)
        
        selected_label = st.selectbox(
            "Select Stock / Crypto",
            options=ticker_options,
            index=current_index,
            help="Select an asset to focus the analysis on based on your needs."
        )
        
        # Update session state based on selection
        if selected_label != "None (Auto-detect)":
            # Find value from label
            for t in SUPPORTED_TICKERS:
                if t["label"] == selected_label:
                    st.session_state.selected_ticker = t["value"]
                    break
        else:
            st.session_state.selected_ticker = None
            
        if st.session_state.selected_ticker:
            st.caption(f"✅ Active Context: **{st.session_state.selected_ticker}**")
        
        st.divider()
        
        # Chat History
        st.subheader("💬 Chat History")
        
        # Sort chats by creation time (newest first)
        sorted_chats = sorted(
            st.session_state.chats.items(),
            key=lambda x: x[1]["created_at"],
            reverse=True
        )
        
        # Smart indexing: number based on current position
        for index, (chat_id, chat_data) in enumerate(sorted_chats, 1):
            is_active = chat_id == st.session_state.current_chat_id
            show_delete_confirm = st.session_state.delete_confirm_id == chat_id
            
            # Three-column layout: Chat button, : button, Delete button (conditional)
            # Three-column layout: Chat button, : button, Delete button (conditional)
            if show_delete_confirm:
                # [0.7, 0.15, 0.15] ratio as requested
                col_chat, col_colon, col_delete = st.columns([0.7, 0.15, 0.15])
            else:
                col_chat, col_colon = st.columns([0.85, 0.15])
            
            with col_chat:
                # Smart position-based numbering
                button_label = f"{'📌' if is_active else '💬'} Chat {index}"
                if st.button(button_label, key=f"chat_{chat_id}", use_container_width=True):
                    if not is_active:
                        switch_chat(chat_id)
            
            with col_colon:
                # Colon button to show delete confirmation
                if st.button("︙", key=f"menu_{chat_id}", help="Options", use_container_width=True):
                    st.session_state.delete_confirm_id = chat_id
                    st.rerun()
            
            # Show 'Delete' button only if colon was clicked for this chat
            if show_delete_confirm:
                with col_delete:
                    # Use icon '🗑️' and primary type
                    if st.button("🗑️", key=f"confirm_del_{chat_id}", help="Permanently delete", type="primary", use_container_width=True):
                        # Remove chat from state
                        del st.session_state.chats[chat_id]
                        st.session_state.delete_confirm_id = None
                        
                        # If deleting the active chat, handle gracefully
                        if chat_id == st.session_state.current_chat_id:
                            if len(st.session_state.chats) > 0:
                                # Switch to most recent remaining chat
                                remaining_chats = sorted(
                                    st.session_state.chats.items(),
                                    key=lambda x: x[1]["created_at"],
                                    reverse=True
                                )
                                st.session_state.current_chat_id = remaining_chats[0][0]
                            else:
                                # No chats left, create a new blank one
                                create_new_chat()
                        
                        st.rerun()
        
        st.divider()
        
        # Model Parameters in Expander
        with st.expander("⚙️ Model Parameters", expanded=False):
            temperature = st.slider(
                "🌡️ Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.config.ollama.temperature,
                step=0.1,
                help="Controls randomness"
            )
            
            top_k = st.slider(
                "🔢 Top K",
                min_value=10,
                max_value=100,
                value=st.session_state.config.ollama.top_k,
                step=5,
                help="Number of documents to retrieve"
            )
            
            top_p = st.slider(
                "📊 Top P",
                min_value=0.1,
                max_value=1.0,
                value=st.session_state.config.ollama.top_p,
                step=0.05,
                help="Nucleus sampling threshold"
            )
            
            if st.button("✅ Apply Settings", use_container_width=True):
                current_chat["controller"].llm_client.update_params(
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                st.success("✅ Settings updated!")
        
        # System Status in Expander
        with st.expander("📊 System Status", expanded=False):
            is_connected = current_chat["controller"].llm_client.health_check()
            
            if is_connected:
                st.success("🟢 Ollama Connected")
            else:
                st.error("🔴 Ollama Disconnected")
                st.caption("Start with: `ollama serve`")
            
            st.metric("Model", st.session_state.config.ollama.model)
            st.metric("Messages", len(current_chat["messages"]))
    
    # Main layout: Chat (75%) and Quick Actions (25%)
    col_chat, col_quick = st.columns([3, 1])
    
    # Chat Area
    with col_chat:
        # Display chat messages
        chat_container = st.container(height=500)
        with chat_container:
            for message in current_chat["messages"]:
                display_message(
                    role=message["role"],
                    content=message["content"],
                    metadata=message.get("metadata")
                )
        
        # Chart display if applicable
        if st.session_state.show_chart:
            with st.expander("📈 Stock Trend Visualization", expanded=False):
                sample_prices = [150, 152, 148, 155, 158, 162, 160, 165, 170, 175]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=sample_prices,
                    mode='lines+markers',
                    line=dict(color='#1E40AF', width=3),
                    marker=dict(size=8, color='#3B82F6')
                ))
                fig.update_layout(
                    title="Stock Price Trend",
                    yaxis_title="Price (USD)",
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Chat input at bottom
        user_input = st.chat_input("Type your question here... (e.g., What is AAPL's current price?)")
        
        if user_input:
            send_message(user_input)
            st.rerun()
    
    # Quick Analysis Panel
    with col_quick:
        st.markdown("""
        <div class="quick-panel">
            <h3>⚡ Quick Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick action buttons
        quick_actions = [
            ("📊 Apple (AAPL) Deep Dive", "Provide a comprehensive deep dive analysis of Apple (AAPL) including current price, financial health, recent news, and market sentiment"),
            ("😊 Tesla (TSLA) Sentiment", "Analyze Tesla (TSLA) market sentiment based on recent news and social trends"),
            ("💰 NVIDIA (NVDA) Financials", "Show me NVIDIA (NVDA) financial statements including revenue, profit margins, and key metrics"),
            ("📈 BTC/USD Market Trend", "Analyze Bitcoin (BTC-USD) market trend and price direction"),
            ("📰 Market News Summary", "Provide a summary of the latest market news and major events")
        ]
        
        for label, query in quick_actions:
            if st.button(label, key=f"quick_{label}", use_container_width=True):
                send_message(query)
                st.rerun()
        
        st.divider()
        
        # Info box
        st.info("💡 Click any button above to quickly analyze stocks and market trends.")
    
    # Footer
    st.divider()
    st.caption("⚠️ **Disclaimer:** This is an educational tool. NOT financial advice.")
    st.caption("Powered by **Ollama LLM** • **RAG Pipeline** • **Yahoo Finance**")


if __name__ == "__main__":
    main()
