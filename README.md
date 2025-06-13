# ONGC Data Toolkit

A powerful Streamlit-based application for managing, cleaning, standardizing, and analyzing geoscientific and survey datasets, tailored for ONGC workflows. The toolkit provides an intuitive interface with categorized tools and detailed guidance for each operation.

## 🎯 Key Features by Category

### 📂 Data Management

* **Upload & Map:** Import Excel/CSV files to PostgreSQL with smart column mapping and duplicate detection
* **Clean & Edit:** Comprehensive data cleaning tools with batch operations and format standardization
* **Standardize Files:** Intelligent schema matching and format standardization across multiple files

### 📊 Analysis Tools

* **Compare Files:** Visual file comparison with cell-by-cell and column-wise analysis
* **Summary of Data Entry:** Advanced data analysis with pattern detection and quality checks
* **Data Type Wise:** Smart classification and processing by data types using pattern matching
* **Create Labels:** Generate professional physical media labels with ONGC branding and barcodes

### 🤖 AI & Search

* **Enterprise-Grade RAG System:** 
  * State-of-the-art DeepSeek-Coder-1.3B LLM (8B parameters) running securely via Ollama
  * Zero-trust architecture with 100% local execution - no data leaves your system
  * Meta's FAISS vector database for high-performance semantic search (>1M QPS)
  * Domain-adapted for geoscientific and petroleum engineering contexts
  * Real-time context-aware responses with sub-second latency
  * Optimized prompt engineering for technical data analysis
  * Built-in hallucination detection and fact checking
* **AI-Powered Data Assistant:** 
  * Natural language understanding for complex data queries
  * Context-aware responses with dynamic knowledge integration
  * Advanced code generation for automated data processing
  * Multi-turn dialogue support with memory retention
  * Custom data visualization suggestions
* **Enterprise Search:** 
  * Advanced fuzzy matching with Levenshtein distance
  * Semantic search powered by FAISS embeddings
  * Multi-language support with cross-lingual capabilities
  * Configurable relevance scoring and ranking

## 💫 Enhanced Features

* **Smart Column Mapping:** Automatic detection and mapping of similar columns
* **Batch Processing:** Handle multiple files efficiently
* **Visual Highlighting:** Clear visualization of data differences and patterns
* **Format Standardization:** Consistent data formatting across all files
* **Error Detection:** Built-in validation and error checking
* **Custom Templates:** Flexible templates for labels and reports
* **Interactive UI:** Categorized sidebar with detailed tool descriptions

## 🛠️ Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/adityapawar327/ONGC-Data-Tool.git
    cd ongc_app
    ```

2.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3.  **Set up PostgreSQL:**
    * Ensure PostgreSQL is installed and running
    * Update the `DATABASE_URL` in `app.py` with your credentials

4.  **Run the app:**
    ```sh
    streamlit run app.py
    ```

## 📂 Project Structure

* `app.py` - Main application with UI components and navigation
* `compare.py` - File comparison logic and visualization
* `cleaning.py` - Data cleaning and standardization utilities
* `link.py` - File linking and standardization functions
* `analyze.py` - Data analysis and summary generation
* `data_type_wise.py` - Data type classification and processing
* `ai.py` - Enterprise RAG (Retrieval Augmented Generation) system integrating DeepSeek LLM, FAISS vector store, and advanced NLP with custom prompt engineering
* `searching.py` - Advanced search with fuzzy matching
* `label.py` - Label generation and customization
* `requirements.txt` - Project dependencies

## 🎯 Usage Guide

1.  **Navigation:**

    * Use the categorized sidebar to select your desired tool
    * Read the detailed description for each tool before use

2.  **Data Management:**

    * Upload files in Excel/CSV format
    * Follow step-by-step instructions for each operation
    * Preview changes before applying them

3.  **Analysis:**

    * Compare files with visual difference highlighting
    * Generate summaries and insights
    * Create customized labels with barcodes

4.  **AI & Search:**

    * Ask questions about your data in natural language
    * Use fuzzy search to find information quickly

## 📋 Technical Requirements

### System Requirements
* Python 3.8+ with pip
* PostgreSQL database
* Ollama for local LLM deployment
* 16GB+ RAM recommended
* 50GB+ storage space for models and indices

### Core Dependencies
* **AI Components:**
  * `langchain` and related packages - For LLM integration and RAG pipeline
  * `faiss-cpu` - For vector similarity search
  * `langchain-ollama` - For local LLM deployment
  
* **Data Processing:**
  * `pandas` - Data manipulation and analysis
  * `numpy` - Numerical computing
  
* **Web Framework:**
  * `streamlit` - Modern web interface
  * `psycopg2` - PostgreSQL adapter
  
Full list of dependencies available in `requirements.txt`

## 📜 License

This project is intended for internal ONGC use. For licensing or external use, please contact the project maintainer.

## 🤝 Contributing

### Commit Convention

We follow a standardized commit message format to maintain a clear and meaningful git history. Please read our [Commit Convention Guide](COMMIT_CONVENTION.md) before contributing.

Example commit messages:
```
feat: add user authentication system
fix: resolve data loading issue in dashboard
docs: update installation instructions
```

See `COMMIT_CONVENTION.md` for detailed guidelines and examples.
