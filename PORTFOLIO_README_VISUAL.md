# Sudhir Shivaram

<div align="center">

### ML/AI Engineer | Full-Stack Java Engineer | Cloud & MLOps Practitioner

**12+ years building production systems** | **200+ documents indexed** | **99.9% uptime RAG systems**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/sudhirshivaram)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sushiva)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:Shivaram.Sudhir@gmail.com)

</div>

---

## 👨‍💻 About Me

Senior Software Engineer specializing in **production-grade RAG systems**, **LLM integration**, and **cloud-deployed AI solutions**. I bridge traditional software engineering (12+ years Java/Spring Boot) with cutting-edge AI/ML to deliver scalable, cost-optimized systems.

**Current Impact:**
- 🚀 Deployed RAG systems processing **200+ documents** with **2-3 second response times**
- 💰 Reduced LLM costs to **~$12/month** with 4-tier automatic fallback achieving **99.9% uptime**
- 📈 Improved retrieval relevance by **40%** through hybrid search (BM25 + vector embeddings)
- 🏢 Architected dual-index systems handling **arXiv papers** + **SEC financial filings**

---

## 🛠️ Tech Stack

### AI/ML & Data Science
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)

**Vector Databases:** FAISS • Pinecone • Weaviate • ChromaDB • OpenSearch
**LLMs:** OpenAI GPT-4 • Google Gemini • Anthropic Claude • Ollama

### Backend & Cloud
![Java](https://img.shields.io/badge/Java-ED8B00?style=for-the-badge&logo=openjdk&logoColor=white)
![Spring Boot](https://img.shields.io/badge/Spring_Boot-6DB33F?style=for-the-badge&logo=spring-boot&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)

![AWS](https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-0078D4?style=for-the-badge&logo=microsoft-azure&logoColor=white)
![GCP](https://img.shields.io/badge/GCP-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)

### Frontend & Deployment
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge)

---

## 🚀 Featured Projects

### 1️⃣ arXiv Paper Curator + Financial Documents RAG

<div align="center">

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-4CAF50?style=for-the-badge)](https://arxiv-paper-curator-v1-demo.streamlit.app/)
[![API Docs](https://img.shields.io/badge/📚_API_Docs-2196F3?style=for-the-badge)](https://arxiv-paper-curator-v1-production.up.railway.app/docs)
[![GitHub](https://img.shields.io/badge/💻_Source_Code-181717?style=for-the-badge&logo=github)](https://github.com/sudhirshivaram/arxiv-paper-curator-v1)

![arXiv RAG Demo](./screenshots/rag-demo-2.png)

</div>

**Production RAG system** deployed on Railway handling dual document types (research papers + SEC filings) with intelligent 4-tier LLM fallback strategy.

**🎯 Key Achievements:**
- ⚡ **2-3 second query responses** with hybrid search (BM25 + Jina embeddings 1024d)
- 💰 **$12/month operational cost** with 4-tier fallback (Gemini → Claude → OpenAI)
- 🎯 **99.9% uptime** through automatic provider failover
- 📊 **40% relevance improvement** via Reciprocal Rank Fusion
- 🏢 **200+ documents indexed**: 100 arXiv papers + 100 SEC filings (10-K/10-Q)

**Tech Stack:** `Python` `FastAPI` `OpenSearch` `PostgreSQL` `Railway` `Streamlit` `Google Gemini` `Anthropic Claude` `OpenAI` `Jina AI Embeddings`

**Features:**
- Dual-index architecture with document type routing
- Ticker-based filtering for financial queries (AAPL, MSFT, GOOGL, TSLA, NVDA, etc.)
- Conversational AI with source citations
- Real-time health monitoring and fallback transparency

---

### 2️⃣ PowerGrid AI Tutor — RAG for Electrical Engineering

<div align="center">

[![Live Demo](https://img.shields.io/badge/🌐_HuggingFace_Space-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/sudhirshivaram/powergrid-ai-tutor)

</div>

**Production RAG system** deployed on HuggingFace Spaces combining **hybrid search** (BM25 + FAISS) with **Cohere reranking** for electrical engineering education.

**🎯 Key Achievements:**
- 📚 **50+ research papers indexed** with full-text search
- ⚡ **Sub-3-second response times** with query expansion
- 🎯 **35%+ improvement in retrieval coverage** through query transformation
- 🔍 **Hybrid search**: BM25 keyword matching + FAISS vector similarity

**Tech Stack:** `Python` `LangChain` `FAISS` `Cohere` `HuggingFace Spaces` `Gradio`

---

### 3️⃣ Energy Consumption Forecasting — Time Series ML

**XGBoost-based forecasting model** achieving exceptional accuracy on household energy data with SHAP interpretability.

**🎯 Key Achievements:**
- 🎯 **99.82% R² accuracy** and **0.42% MAPE** on test data
- 🔧 **15+ engineered features**: lag variables, rolling statistics, cyclical encodings
- 📊 **SHAP analysis** for model interpretability and feature importance
- ⚡ **Production-ready pipeline** with hyperparameter tuning

**Tech Stack:** `Python` `XGBoost` `scikit-learn` `pandas` `SHAP`

---

## 🎓 Certifications

![AWS](https://img.shields.io/badge/AWS_Cloud_Practitioner_(CLF--002)-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Azure](https://img.shields.io/badge/Azure_Fundamentals_(AZ--900)-0078D4?style=for-the-badge&logo=microsoft-azure&logoColor=white)
![UT Austin](https://img.shields.io/badge/UT_Austin_AI/ML_Certificate-BF5700?style=for-the-badge)

---

## 💼 Professional Experience

**Senior Software Engineer** @ Computershare *(June 2025 – Present)*
- Optimizing enterprise reporting systems with cloud-native architecture
- Migrating legacy applications to Azure Cloud infrastructure
- Building microservices with Spring Boot and Docker/Kubernetes

**12+ Years Full-Stack Java Development**
- Enterprise applications with Spring Boot, React, Angular
- Cloud deployments on AWS, Azure, GCP
- CI/CD pipelines with Jenkins, GitHub Actions, Azure DevOps

---

## 📊 GitHub Stats

<div align="center">

![GitHub Stats](https://github-readme-stats.vercel.app/api?username=sushiva&show_icons=true&theme=radical)

![Top Languages](https://github-readme-stats.vercel.app/api/top-langs/?username=sushiva&layout=compact&theme=radical)

</div>

---

## 📫 Let's Connect!

I'm always interested in discussing **RAG systems**, **LLM optimization**, **vector search**, and **production ML deployments**.

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/sudhirshivaram)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sushiva)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:Shivaram.Sudhir@gmail.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-4CAF50?style=for-the-badge)](https://sushiva.github.io)

</div>

---

<div align="center">

**"Building production AI systems that are reliable, cost-effective, and actually useful."**

⭐ Star my repositories if you find them helpful!

</div>
