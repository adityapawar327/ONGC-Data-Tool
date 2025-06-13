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

* **AI Assistant:** Natural language interaction for data analysis and insights
* **Fuzzy Search:** Advanced search capabilities with fuzzy matching across files

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
* `ai.py` - AI assistant integration and natural language processing
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

## 📋 Requirements

* Python 3.8+
* PostgreSQL database
* Required Python packages (listed in `requirements.txt`)

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
