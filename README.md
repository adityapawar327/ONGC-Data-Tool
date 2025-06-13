# 🚀 ONGC Data Toolkit

An enterprise-grade data management and analysis platform built with Python and Streamlit, specifically designed for processing geoscientific and survey datasets. This toolkit leverages advanced AI/ML capabilities, robust data processing pipelines, and modern software architecture patterns to deliver a comprehensive solution for data management challenges.

## 🌟 Core Technologies

- **Frontend**: Streamlit, Custom CSS
- **Backend**: Python 3.8+, SQLAlchemy
- **Database**: PostgreSQL
- **AI/ML Stack**: 
  - RAG (Retrieval Augmented Generation) Architecture
  - LangChain for LLM Orchestration
  - DeepSeek-Coder-1.3B (8B parameters) for Intelligence
  - FAISS Vector Store for Semantic Search
  - Ollama for Local LLM Deployment
- **Data Processing**: Pandas, NumPy
- **Search Engine**: Custom Fuzzy Search with Levenshtein Distance
- **Document Processing**: Python-DOCX, PDF Generation

## 🎯 Key Features

### 📊 Enterprise Data Management

* **Intelligent Data Import**
  * PostgreSQL Integration with Transaction Support
  * Smart Schema Detection and Mapping
  * Automated Data Type Inference
  * Duplicate Detection and Resolution

* **Advanced Data Cleaning Pipeline**
  * Automated Data Quality Assessment
  * Multi-stage Cleaning Operations
  * Custom Validation Rules Engine
  * Batch Processing Support

* **Format Standardization**
  * Cross-format Schema Matching
  * Intelligent Data Normalization
  * Custom Template Support
  * Automated Data Transformation

### 🔍 Analysis & Intelligence

* **Enterprise Search**
  * Vector-based Semantic Search
  * Fuzzy Matching Algorithm
  * Multi-field Search Support
  * Configurable Relevance Scoring

* **Data Analysis Tools**
  * Real-time Data Visualization
  * Pattern Detection Engine
  * Statistical Analysis
  * Anomaly Detection

* **AI-Powered Features**
  * Context-Aware RAG System
  * Natural Language Query Processing
  * Automated Data Quality Reports
  * Intelligent Data Classification

### 🛠️ Technical Architecture

* **Modular Design**
  * Component-based Architecture
  * Clean Code Principles
  * Extensive Error Handling
  * Comprehensive Logging

* **Security**
  * Local LLM Deployment
  * Zero Data Leakage
  * Secure Database Operations
  * Input Validation

* **Performance**
  * Optimized Data Processing
  * Efficient Memory Management
  * Batch Operation Support
  * Caching Mechanisms

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/adityapawar327/ONGC-Data-Tool.git
cd ongc_app

# Create and activate virtual environment (Windows)
python -m venv venv
.\\venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Configure PostgreSQL
# Update DATABASE_URL in app.py with your credentials

# Run the application
streamlit run app.py
```

## 📦 Project Structure

```
ongc_app/
├── app.py              # Main application entry point
├── ai.py              # RAG system and LLM integration
├── analyze.py         # Data analysis engine
├── cleaning.py        # Data cleaning pipeline
├── compare.py         # File comparison logic
├── convert.py         # Format conversion utilities
├── data_type_wise.py  # Data classification system
├── label.py           # Document generation
├── link.py           # Data linking engine
├── searching.py      # Search implementation
└── requirements.txt  # Project dependencies
```

## 💡 Key Components

### AI Engine (`ai.py`)
- Implements enterprise-grade RAG architecture
- Integrates DeepSeek-Coder-1.3B LLM
- Custom prompt engineering for domain-specific tasks
- FAISS vector store for efficient similarity search

### Data Processing (`cleaning.py`, `analyze.py`)
- Advanced data cleaning pipelines
- Statistical analysis tools
- Pattern detection algorithms
- Data quality assessment

### Search Engine (`searching.py`)
- Custom fuzzy search implementation
- Levenshtein distance calculations
- Multi-field search capability
- Relevance scoring system

## 🔒 Security & Performance

- **Zero-Trust Architecture**
  * All AI operations run locally
  * No data leaves your system
  * Secure database operations

- **Performance Optimizations**
  * Efficient memory management
  * Batch processing capabilities
  * Caching mechanisms
  * Parallel processing support

## 🤝 Contributing

We follow a standardized commit convention. See [COMMIT_CONVENTION.md](COMMIT_CONVENTION.md) for guidelines.

## 📫 Contact

For questions or feedback, please contact the project maintainer.

## 📄 License

This project is intended for internal ONGC use. For licensing or external use, please contact the project maintainer.

---
**Keywords**: Python, Data Engineering, Machine Learning, RAG, LangChain, FAISS, Vector Search, ETL, Data Processing, Enterprise Software, Data Analysis, Natural Language Processing, PostgreSQL, Streamlit, Software Architecture, Data Pipeline, Data Quality, Business Intelligence
