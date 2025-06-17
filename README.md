# ONGC Data Toolkit

<div align="center">
  <img src="public/ongc (1).jpg" alt="ONGC Logo" width="200"/>
  
  <h2>Enterprise Data Management & Analysis Platform</h2>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+"/>
    <img src="https://img.shields.io/badge/Streamlit-1.24+-FF4B4B.svg" alt="Streamlit"/>
    <img src="https://img.shields.io/badge/PostgreSQL-13+-336791.svg" alt="PostgreSQL"/>
    <img src="https://img.shields.io/badge/DeepSeek_R1-8B-2E8B57.svg" alt="DeepSeek-R1"/>
  </p>
</div>

## Table of Contents
- [Overview](#overview)
- [Core Technologies](#core-technologies)
- [Key Features](#key-features)
- [Installation Guide](#installation-guide)
- [Project Structure](#project-structure)
- [Module Descriptions](#module-descriptions)
- [Security & Performance](#security--performance)
- [Contributing](#contributing)

## Overview

The ONGC Data Toolkit is an enterprise-grade data management and analysis platform designed for processing geoscientific and survey datasets. This comprehensive solution leverages advanced AI/ML capabilities, robust data processing pipelines, and modern software architecture patterns to address complex data management challenges in the energy sector.

## Core Technologies

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Frontend</b></td>
      <td align="center"><b>Backend</b></td>
      <td align="center"><b>Database</b></td>
      <td align="center"><b>AI/ML</b></td>
    </tr>
    <tr>
      <td>Streamlit, Custom CSS</td>
      <td>Python 3.8+, SQLAlchemy</td>
      <td>PostgreSQL</td>
      <td>DeepSeek-R1, LangChain</td>
    </tr>
  </table>
</div>

### AI/ML Stack
- **RAG Architecture** - Retrieval Augmented Generation
- **LangChain** - LLM Orchestration
- **DeepSeek-R1** - 8B Parameter Model
- **FAISS** - Vector Store for Semantic Search
- **Ollama** - Local LLM Deployment

### Data Processing
- **Pandas** - Data Manipulation
- **NumPy** - Numerical Computing
- **Python-DOCX** - Document Processing
- **Fuzzy Search** - Custom Implementation

## Key Features

### Enterprise Data Management

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

### Analysis & Intelligence

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

## Installation Guide

<div align="center">
  <img src="https://img.shields.io/badge/Setup-Guide-00A67E.svg" alt="Setup Guide"/>
</div>

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
   
   # Pull the latest DeepSeek-R1 model (8B parameters)
   ollama pull deepseek-r1

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

## Project Structure

<div align="center">
  <img src="https://img.shields.io/badge/Project-Structure-00A67E.svg" alt="Project Structure"/>
</div>

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
├── public/           # Static assets directory
├── .devcontainer/    # Development container configuration
├── temp/            # Temporary files directory
└── requirements.txt  # Project dependencies
```

## Module Descriptions

### AI Module (`ai.py`)
The AI engine implements a sophisticated RAG (Retrieval Augmented Generation) system that powers the intelligent features of the toolkit:
- Context-aware natural language processing
- Document chunking and vectorization
- Semantic search capabilities
- Custom prompt engineering for domain-specific tasks
- Integration with DeepSeek-R1 model for advanced reasoning

### Analysis Module (`analyze.py`)
Comprehensive data analysis engine that provides:
- Statistical analysis of datasets
- Pattern detection and trend analysis
- Data quality assessment
- Automated report generation
- Visualization capabilities for insights

### Cleaning Module (`cleaning.py`)
Advanced data cleaning pipeline that handles:
- Automated data quality checks
- Missing value treatment
- Data type standardization
- Text normalization
- Duplicate detection and removal
- Custom validation rules

### Compare Module (`compare.py`)
File comparison utility that enables:
- Cell-by-cell comparison of datasets
- Visual highlighting of differences
- Structure comparison
- Version tracking
- Change detection

### Convert Module (`convert.py`)
File format conversion utility that supports:
- CSV to Excel conversion
- Excel to CSV conversion
- DOCX to PDF conversion
- Format preservation
- Batch processing capabilities

### Data Type Module (`data_type_wise.py`)
Specialized data classification system that provides:
- Automatic data type detection
- Schema validation
- Data categorization
- Format standardization
- Metadata extraction

### Label Module (`label.py`)
Document generation system for creating:
- Physical media labels
- Standardized documentation
- Custom templates
- Batch label generation
- Format-specific outputs

### Link Module (`link.py`)
Data linking engine that enables:
- Cross-dataset relationships
- Schema matching
- Data normalization
- Relationship mapping
- Link validation

### Search Module (`searching.py`)
Advanced search implementation featuring:
- Fuzzy matching algorithms
- Multi-field search
- Relevance scoring
- Pattern matching
- Search result highlighting

### Supporting Directories

#### Public Directory
Contains static assets:
- Application logos
- UI elements
- Documentation resources
- Template files

#### DevContainer Directory
Development environment configuration:
- Container setup
- Development tools
- Environment variables
- Build configurations

#### Temp Directory
Temporary file storage:
- Processing cache
- Temporary outputs
- Session data
- Working files

## Security & Performance

<div align="center">
  <img src="https://img.shields.io/badge/Security-Performance-00A67E.svg" alt="Security & Performance"/>
</div>

### Zero-Trust Architecture
* All AI operations run locally
* No data leaves your system
* Secure database operations

### Performance Optimizations
* Efficient memory management
* Batch processing capabilities
* Caching mechanisms
* Parallel processing support

## Contributing

<div align="center">
  <img src="https://img.shields.io/badge/Contributing-Guide-00A67E.svg" alt="Contributing Guide"/>
</div>

We follow a standardized commit convention. See [COMMIT_CONVENTION.md](COMMIT_CONVENTION.md) for guidelines.

---

<div align="center">
  <p>© 2024 ONGC Data Toolkit. All rights reserved.</p>
</div>
