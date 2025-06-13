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

## 🔧 Installation Guide

### Prerequisites

1. **Python Setup**
   ```bash
   # Install Python 3.8 or later from https://www.python.org/downloads/
   # Verify installation
   python --version
   ```

2. **PostgreSQL Installation**
   - Download and install PostgreSQL 13+ from https://www.postgresql.org/download/
   - During installation, note down your password
   - Create a new database:
     ```sql
     CREATE DATABASE ongcdata;
     ```

3. **Ollama Setup for DeepSeek**
   ```bash
   # Install Ollama from https://ollama.ai/download
   
   # Pull the latest DeepSeek Coder model (8B parameters)
   ollama pull deepseek-coder

   # Verify installation
   ollama list
   ```

### Project Setup

1. **Clone & Navigate**
   ```bash
   git clone https://github.com/adityapawar327/ONGC-Data-Tool.git
   cd ongc_app
   ```

2. **Virtual Environment**
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate (Windows)
   .\venv\Scripts\activate

   # Activate (Linux/Mac)
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   # Update pip
   python -m pip install --upgrade pip

   # Install required packages
   pip install -r requirements.txt
   ```

4. **Configure Database**
   - Open `app.py`
   - Update the DATABASE_URL:
     ```python
     DATABASE_URL = "postgresql://username:password@localhost:5432/ongcdata"
     ```
   - Replace username, password with your PostgreSQL credentials

5. **Environment Setup**
   ```bash
   # Create .env file
   echo DATABASE_URL=postgresql://username:password@localhost:5432/ongcdata > .env
   echo OLLAMA_BASE_URL=http://localhost:11434 >> .env
   ```

6. **Run Application**
   ```bash
   # Start Ollama in background
   ollama serve

   # In a new terminal, run the application
   streamlit run app.py
   ```

### Verification Steps

1. **Check Database Connection**
   - Application will show successful database connection on startup
   - Test data upload functionality

2. **Verify AI Features**
   - Go to "AI Assistant" section
   - Type a test query
   - System should respond using local DeepSeek model

3. **Test Data Processing**
   - Upload a sample Excel/CSV file
   - Verify data cleaning features
   - Test search functionality

### Troubleshooting

1. **Database Issues**
   - Verify PostgreSQL service is running
   - Check database credentials
   - Ensure database exists

2. **AI Model Issues**
   - Verify Ollama is running: `curl http://localhost:11434/api/tags`
   - Check DeepSeek model: `ollama list`
   - Try restarting Ollama service

3. **Application Errors**
   - Check Python version compatibility
   - Verify all dependencies are installed
   - Check console for error messages

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
