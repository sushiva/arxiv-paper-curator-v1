"""
Streamlit frontend for arXiv Paper Curator RAG system.

This app provides a clean interface to:
- Search papers and financial documents using hybrid search (BM25 + vector similarity)
- Ask questions and get AI-generated answers with citations
- Browse paper and financial document metadata
"""

import streamlit as st
import requests
import os
from typing import Dict, Any, List
from datetime import datetime

# Configuration
API_URL = os.getenv("API_URL", "https://arxiv-paper-curator-v1-production.up.railway.app")

# Page config
st.set_page_config(
    page_title="Research & Financial Document Curator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapsed for screenshot-friendly layout
)

# Custom CSS - Ultra-minimal for screenshots
st.markdown("""
<style>
    /* Ultra-compact layout for fitting query + results in one view */
    .main .block-container {
        padding-top: 0.3rem;
        padding-bottom: 0.3rem;
    }
    .main-header {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
    }
    /* Reduce spacing in tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    /* Reduce spacing after radio buttons */
    .stRadio {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def call_rag_api(
    query: str,
    top_k: int = 3,
    document_type: str = "arxiv",
    ticker: str = None,
    filing_types: List[str] = None,
    model: str = None
) -> Dict[str, Any]:
    """Call the RAG API endpoint."""
    try:
        payload = {
            "query": query,
            "top_k": top_k,
            "document_type": document_type
        }

        # Add model if provided (from session state)
        if model:
            payload["model"] = model

        # Add financial-specific parameters if applicable
        if document_type == "financial":
            if ticker:
                payload["ticker"] = ticker
            if filing_types:
                payload["filing_types"] = filing_types

        response = requests.post(
            f"{API_URL}/api/v1/ask",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def check_api_health() -> bool:
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_URL}/api/v1/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    """Main Streamlit app."""

    # Header (compact for screenshots)
    st.markdown('<div class="main-header">📚 Research & Financial Document Curator</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # API Health Check
        is_healthy = check_api_health()
        if is_healthy:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Offline")
            st.info(f"API URL: {API_URL}")

        st.divider()

        # System Info
        st.subheader("🤖 AI System")
        st.info("""
        **Powered by 4-Tier LLM Fallback:**
        - Google Gemini (Primary)
        - Claude 3.5 Haiku
        - OpenAI GPT-4o-mini

        *Automatic failover for 99.9% uptime*
        """)

        st.divider()

        # Search Settings
        st.subheader("Search Settings")
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=3)

        st.divider()

        # About
        st.subheader("About")
        st.markdown("""
        This RAG system searches:
        - **arXiv Papers**: AI research
        - **Financial Docs**: SEC filings

        Technology:
        - **Hybrid Search**: BM25 + Vector
        - **Embeddings**: Jina AI (1024d)
        - **LLM**: GPT-4o-mini / Ollama
        """)

        st.divider()

        # Stats
        st.subheader("📊 System Stats")
        st.metric("arXiv Papers", "100")
        st.metric("Financial Docs", "6")
        st.metric("Search Mode", "Hybrid")

    # Main content
    tab1, tab2 = st.tabs(["🔍 Ask Questions", "📖 About"])

    with tab1:
        # Document Type Selector (compact spacing)
        st.markdown("##### 📁 Select Document Type")
        document_type = st.radio(
            "Choose the type of documents to search:",
            options=["📚 arXiv Papers", "💼 Financial Documents"],
            horizontal=True,
            label_visibility="collapsed"
        )

        # Map display names to API values
        doc_type_map = {
            "📚 arXiv Papers": "arxiv",
            "💼 Financial Documents": "financial"
        }
        selected_doc_type = doc_type_map[document_type]

        # Financial-specific filters
        ticker = None
        filing_types = None
        if selected_doc_type == "financial":
            col1, col2 = st.columns(2)
            with col1:
                ticker = st.text_input(
                    "Ticker Symbol (optional)",
                    placeholder="e.g., AAPL, MSFT",
                    help="Filter by company ticker symbol"
                )
            with col2:
                filing_types = st.multiselect(
                    "Filing Types (optional)",
                    options=["10-K", "10-Q"],
                    default=["10-K"],
                    help="Select filing types to search"
                )

        # Query input
        query_placeholder = (
            "e.g., What are Apple's main risk factors?"
            if selected_doc_type == "financial"
            else "e.g., What papers discuss machine learning for healthcare?"
        )

        query = st.text_input(
            "Enter your question:",
            placeholder=query_placeholder
        )

        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)

        if clear_button:
            st.rerun()

        # Process query
        if search_button and query:
            spinner_text = (
                "Searching financial documents and generating answer..."
                if selected_doc_type == "financial"
                else "Searching papers and generating answer..."
            )

            with st.spinner(spinner_text):
                result = call_rag_api(
                    query,
                    top_k,
                    document_type=selected_doc_type,
                    ticker=ticker if ticker else None,
                    filing_types=filing_types if filing_types else None
                )

                if result:
                    # Display answer with inline metrics
                    col_heading, col_metrics = st.columns([1, 2])
                    with col_heading:
                        st.subheader("📝 Answer")
                    with col_metrics:
                        st.caption(f"📊 Chunks: {result.get('chunks_used', 0)} | Mode: {result.get('search_mode', 'N/A')} | Sources: {len(result.get('sources', []))}")

                    st.markdown(result.get("answer", "No answer generated"))

                    # Display sources
                    if result.get("sources"):
                        st.subheader("📚 Sources")
                        for idx, source in enumerate(result.get("sources", []), 1):
                            with st.container():
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>Source {idx}:</strong>
                                    <a href="{source}" target="_blank">{source}</a>
                                </div>
                                """, unsafe_allow_html=True)

        elif search_button and not query:
            st.warning("⚠️ Please enter a question")

    with tab2:
        st.header("About Research & Financial Document Curator")

        st.markdown("""
        ### 🎯 What is this?

        An AI-powered system for searching and understanding both:
        - **Research papers** from arXiv (Computer Science, AI)
        - **Financial documents** from SEC EDGAR (10-K, 10-Q filings)

        It combines modern search techniques with large language models to provide accurate, cited answers to
        your research and financial questions.

        ### 🔧 How it works

        1. **Hybrid Search**: Combines keyword-based (BM25) and semantic (vector) search for better accuracy
        2. **Smart Chunking**: Documents are split into meaningful sections for precise retrieval
        3. **AI Synthesis**: LLM generates comprehensive answers from retrieved content
        4. **Source Citations**: Every answer includes links to the source documents

        ### 🏗️ Architecture

        - **Backend**: FastAPI + Python
        - **Vector DB**: OpenSearch (dual-index: arXiv + Financial)
        - **SQL DB**: PostgreSQL
        - **Embeddings**: Jina AI (jina-embeddings-v3, 1024 dimensions)
        - **LLM**: OpenAI GPT-4o-mini / Ollama
        - **Frontend**: Streamlit
        - **Deployment**: Railway.app

        ### 📊 Current Dataset

        **arXiv Papers:**
        - **100 papers** from arXiv (cs.AI category)
        - Indexed with title + abstract

        **Financial Documents:**
        - **6 companies** indexed (AAPL, MSFT, GOOGL, TSLA, NVDA, etc.)
        - SEC 10-K and 10-Q filings
        - Full-text search and semantic search available

        ### 🚀 Future Enhancements

        - Automated daily paper ingestion (Airflow)
        - Full paper content indexing
        - More companies and financial documents
        - Advanced filtering (by date, author, category, fiscal period)
        - Document recommendation system
        """)

        st.divider()

        st.markdown("""
        ### 🔗 Links

        - [GitHub Repository](https://github.com/sudhirshivaram/arxiv-paper-curator-v1)
        - [API Documentation](https://arxiv-paper-curator-v1-production.up.railway.app/docs)
        - [Railway Deployment](https://railway.app)
        """)


if __name__ == "__main__":
    main()
