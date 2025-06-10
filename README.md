# ONGC Data Toolkit

A powerful Streamlit-based application for managing, cleaning, standardizing, and analyzing geoscientific and survey datasets, tailored for ONGC workflows. The toolkit streamlines the process of uploading, mapping, cleaning, standardizing, and analyzing Excel/CSV files, with advanced fuzzy matching and AI-powered features.

---

## 🚀 Features

* **Upload & Map:** Import Excel/CSV files and map columns to your PostgreSQL database schema.
* **Compare Files:** Visually compare multiple files and highlight differences.
* **Clean & Edit:** Clean, edit, and download datasets with user-friendly tools.
* **Standardize Files:** Standardize column names and formats using fuzzy logic and synonym mapping.
* **Summary of Data Entry:** Analyze and summarize data entries, including media ID ranges and area-wise summaries.
* **Data Type Wise Processing:** Classify and process data by type (Acquisition, Processing, Interpretation) using fuzzy logic.
* **AI Assistant:** Get AI-powered assistance for data-related queries.
* **Fuzzy Search:** Quickly search and locate data using fuzzy matching.

---

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
    Ensure you have a running PostgreSQL instance.
    Update the `DATABASE_URL` in `app.py` accordingly.
4.  **Run the app:**
    ```sh
    streamlit run app.py
    ```

---

## 📂 Project Structure

* `app.py` – Main Streamlit application and navigation.
* `compare.py` – File comparison logic.
* `cleaning.py` – Data cleaning utilities.
* `link.py` – Standardization logic.
* `analyze.py` – Data analysis and summary.
* `data_type_wise.py` – Data type classification and processing.
* `ai.py` – AI assistant integration.
* `searching.py` – Fuzzy search utilities.
* `requirements.txt` – Python dependencies.

---

## ⚡ Usage

* Launch the app and use the sidebar to navigate between functionalities.
* Upload your Excel/CSV files as prompted.
* Follow on-screen instructions for mapping, cleaning, standardizing, and analyzing your data.

---

## 📋 Requirements

* Python 3.8+
* All dependencies listed in `requirements.txt`

---

## 📜 License

This project is intended for internal ONGC use. For licensing or external use, please contact the project maintainer.
