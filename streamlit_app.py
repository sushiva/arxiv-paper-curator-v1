"""
Streamlit frontend for arXiv Paper Curator RAG system.

This app provides a clean interface to:
- Search papers using hybrid search (BM25 + vector similarity)
- Ask questions and get AI-generated answers with citations
- Browse paper metadata
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
    page_title="arXiv Paper Curator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .paper-card {
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def call_rag_api(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Call the RAG API endpoint."""
    try:
        response = requests.post(
            f"{API_URL}/api/v1/ask",
            json={"query": query, "top_k": top_k},
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

    # Header
    st.markdown('<div class="main-header">📚 arXiv Paper Curator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Research Paper Search & Question Answering</div>', unsafe_allow_html=True)

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

        # Search Settings
        st.subheader("Search Settings")
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=3)

        st.divider()

        # About
        st.subheader("About")
        st.markdown("""
        This RAG system uses:
        - **Hybrid Search**: BM25 + Vector Similarity
        - **Embeddings**: Jina AI (1024 dimensions)
        - **LLM**: OpenAI GPT-4o-mini
        - **Database**: PostgreSQL + OpenSearch
        """)

        st.divider()

        # Stats
        st.subheader("📊 System Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Papers Indexed", "100")
        with col2:
            st.metric("Search Mode", "Hybrid")

    # Main content
    tab1, tab2 = st.tabs(["🔍 Ask Questions", "📖 About"])

    with tab1:
        st.header("Ask a Question About Research Papers")

        # Example questions
        with st.expander("💡 Example Questions & Tips"):
            st.info("💡 **Tip**: Ask about specific research topics, not general definitions. This system searches 100 specialized AI research papers.")

            st.markdown("""
            **Good questions (specific to research)**:
            - What papers discuss reinforcement learning methods?
            - What are the latest advances in transformer architectures?
            - Tell me about recent work on multimodal learning
            - What research has been done on visual reasoning?
            - Explain recent advances in time series forecasting

            **Questions that won't work well**:
            - What is machine learning? ❌ (too general, no definitions in research papers)
            - Explain neural networks ❌ (textbook question, not research-specific)

            **Why?** The database contains recent research papers (not textbooks), so ask about specific research topics!
            """)

        # Query input
        query = st.text_area(
            "Enter your question:",
            placeholder="e.g., What papers discuss machine learning for healthcare?",
            height=100
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
            with st.spinner("Searching papers and generating answer..."):
                result = call_rag_api(query, top_k)

                if result:
                    # Display answer
                    st.subheader("📝 Answer")
                    st.markdown(result.get("answer", "No answer generated"))

                    # Display metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Chunks Used", result.get("chunks_used", 0))
                    with col2:
                        st.metric("Search Mode", result.get("search_mode", "N/A"))
                    with col3:
                        st.metric("Sources", len(result.get("sources", [])))

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
        st.header("About arXiv Paper Curator")

        st.markdown("""
        ### 🎯 What is this?

        arXiv Paper Curator is an AI-powered system for searching and understanding research papers from arXiv.
        It combines modern search techniques with large language models to provide accurate, cited answers to
        your research questions.

        ### 🔧 How it works

        1. **Hybrid Search**: Combines keyword-based (BM25) and semantic (vector) search for better accuracy
        2. **Smart Chunking**: Papers are split into meaningful sections for precise retrieval
        3. **AI Synthesis**: LLM generates comprehensive answers from retrieved content
        4. **Source Citations**: Every answer includes links to the source papers

        ### 🏗️ Architecture

        - **Backend**: FastAPI + Python
        - **Vector DB**: OpenSearch
        - **SQL DB**: PostgreSQL
        - **Embeddings**: Jina AI (jina-embeddings-v3)
        - **LLM**: OpenAI GPT-4o-mini
        - **Frontend**: Streamlit
        - **Deployment**: Railway.app

        ### 📊 Current Dataset

        - **100 papers** from arXiv (cs.AI category)
        - Papers indexed with title + abstract
        - Full-text search and semantic search available

        ### 🚀 Future Enhancements

        - Automated daily paper ingestion (Airflow)
        - Full paper content indexing
        - Advanced filtering (by date, author, category)
        - Paper recommendation system
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
