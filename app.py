import streamlit as st

# Streamlit app configuration
st.set_page_config(
    page_title="ONGC Data Toolkit",
    page_icon="./public/ongc (1).jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
from io import BytesIO
from sqlalchemy import create_engine, inspect, text
from compare import read_file, highlight_differences
from cleaning import clean_data_ui
import link  
from analyze import analyze_app
from ai import ai_chat
from searching import fuzzy_search_ui
from data_type_wise import data_type_wise_app  # Import the new module
from label import label_app

# PostgreSQL connection string
DATABASE_URL = "postgresql://postgres:test@localhost:5432/ongctest4"
engine = create_engine(DATABASE_URL)

with st.sidebar:
    st.markdown("## 🔧 Navigation")
    menu = st.radio(
        "Choose Functionality",
        [
            "📅 Upload & Map",
            "📊 Compare Files",
            "🧹 Clean & Edit",
            "🔗 Standardize Files",
            "📊 Summary of Data Entry",
            "📈 Data Type Wise", 
            "🤖 AI Assistant",
            "🔍 Search",
            "📑 Create Labels"
        ]
    )

if menu == "📅 Upload & Map":

    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.stop()

    if not tables:
        st.error("No tables found in your database.")
        st.stop()

    selected_table = st.selectbox("Choose a Table", tables, index=tables.index('data') if 'data' in tables else 0)

    try:
        table_columns = [col['name'] for col in inspector.get_columns(selected_table)]
        st.write(f"Columns in '{selected_table}':", table_columns)
    except Exception as e:
        st.error(f"Failed to retrieve columns for '{selected_table}': {e}")
        st.stop()

    uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"], key="upload1")

    if uploaded_file:
        df = read_file(uploaded_file)
        if df is not None:
            st.subheader("Preview of Uploaded Data")
            st.dataframe(df, use_container_width=True)

            st.subheader("Map Columns")
            mapping = {}
            for col in table_columns:
                selected_excel_col = st.selectbox(
                    f"Map PostgreSQL column '{col}' to Excel/CSV column:",
                    [None] + list(df.columns),
                    key=col
                )
                mapping[col] = selected_excel_col

            if st.button("🚀 Migrate Data"):
                mapped_data = pd.DataFrame()

                for table_col, excel_col in mapping.items():
                    if excel_col is None:
                        mapped_data[table_col] = [None] * len(df)
                    else:
                        mapped_data[table_col] = df[excel_col]

                st.subheader("Mapped Data Preview")
                st.dataframe(mapped_data, use_container_width=True)

                try:
                    with engine.begin() as connection:
                        existing_data = pd.read_sql_table(selected_table, con=connection)
                        # Filter new data to avoid duplicates
                        new_data = mapped_data[
                            ~mapped_data.apply(tuple, axis=1).isin(existing_data.apply(tuple, axis=1))
                        ]

                        if new_data.empty:
                            st.info("No new data to insert. All records already exist.")
                        else:
                            for _, row in new_data.iterrows():
                                cols_with_data = new_data.columns[~row.isna()]
                                col_names = ", ".join(cols_with_data)
                                col_params = ", ".join(f":{c}" for c in cols_with_data)

                                query = text(f"INSERT INTO {selected_table} ({col_names}) VALUES ({col_params})")
                                params = {c: row[c] for c in cols_with_data}
                                connection.execute(query, params)

                            st.success(f"✅ Successfully inserted {len(new_data)} new rows into '{selected_table}'!")
                except Exception as e:
                    st.error(f"❌ Failed to insert data: {e}")

elif menu == "📊 Compare Files":
    st.header("📊 Compare Multiple Excel/CSV Files")

    uploaded_files = st.file_uploader(
        "Upload two or more Excel/CSV files", 
        type=["xlsx", "xls", "csv"], 
        accept_multiple_files=True, 
        key="upload2"
    )

    if uploaded_files and len(uploaded_files) >= 2:
        dfs = []
        for file in uploaded_files:
            df = read_file(file)
            if df is not None:
                dfs.append(df.fillna(""))  # fill NaNs for consistent comparison

        if dfs:
            styled_dfs = highlight_differences(dfs)
            for i, styled_df in enumerate(styled_dfs):
                st.subheader(f"📄 File {i+1}: {uploaded_files[i].name}")
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.info("Please upload at least two files to compare.")

elif menu == "🧹 Clean & Edit":
    st.header("🧹 Clean & Edit Excel/CSV Data")
    clean_file = st.file_uploader("Upload file to clean", type=["csv", "xlsx", "xls"], key="cleaner")

    if clean_file:
        df_clean = read_file(clean_file)
        if df_clean is not None:
            df_clean = clean_data_ui(df_clean)
            st.subheader("Cleaned Data Preview")
            st.dataframe(df_clean, use_container_width=True)

            excel_buffer = BytesIO()
            df_clean.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)

            st.download_button(
                label="📥 Download Cleaned Excel",
                data=excel_buffer,
                file_name="cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

elif menu == "🔗 Standardize Files":
    link.standardize_datasets()

elif menu == "📊 Summary of Data Entry":
    analyze_app()

elif menu == "📈 Data Type Wise":
    # New functionality for Data Type Wise processing
    data_type_wise_app()

elif menu == "🤖 AI Assistant":
    ai_chat()
    
elif menu == "🔍 Search":
    fuzzy_search_ui()

elif menu == "📑 Create Labels":
    # Launch the label creation functionality
    label_app()