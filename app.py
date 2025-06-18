import streamlit as st

# Streamlit app configuration
st.set_page_config(
    page_title="ONGC Data Toolkit",
    page_icon="./public/ongc (1).jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for category selection
category_style = """
<style>
[data-testid="stSelectbox"] > div:first-of-type {
    cursor: pointer !important;
}
[data-testid="stSelectbox"] div[role="button"] {
    cursor: pointer !important;
}
[data-testid="stSelectbox"] div[role="option"] {
    cursor: pointer !important;
}
</style>
"""
st.markdown(category_style, unsafe_allow_html=True)

# Hide footer and deploy button while keeping the menu
hide_streamlit_style = """
<style>
footer {visibility: hidden;}
div[data-testid="stToolbar"] {visibility: hidden;}
.st-emotion-cache-h5rgaw.ea3mdgi1 {visibility: hidden;}

/* Fix cursor for dropdown menus */
div[role="listbox"] {cursor: pointer !important;}
div[role="combobox"] {cursor: pointer !important;}
div[data-baseweb="select"] {cursor: pointer !important;}
div[data-baseweb="select"] * {cursor: pointer !important;}
.stSelectbox, .stSelectbox *, div[class*="Select"] {cursor: pointer !important;}
div[class*="Select__control"] {cursor: pointer !important;}
div[class*="Select__input"] {cursor: pointer !important;}
div[class*="Select__option"] {cursor: pointer !important;}
div[class*="Select__value-container"] {cursor: pointer !important;}
div[class*="Select__indicators"] {cursor: pointer !important;}

/* Ensure text input stays as text cursor */
.stSelectbox input {cursor: text !important;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

try:
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
    from convert import convert_app  # Import the new conversion module
except Exception as e:
    st.error(f"Failed to import required modules: {str(e)}")
    st.stop()

# PostgreSQL connection string
DEFAULT_DATABASE_URL = "postgresql://postgres:test@localhost:5432/ongctest4"

# Initialize session state for database URL
if 'database_url' not in st.session_state:
    st.session_state.database_url = DEFAULT_DATABASE_URL

# Function to test database connection
def test_database_connection(url):
    try:
        test_engine = create_engine(url)
        with test_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, None
    except Exception as e:
        return False, str(e)

# Database connection setup
def setup_database_connection():
    # Try the current database URL
    connection_success, error_msg = test_database_connection(st.session_state.database_url)
    
    if not connection_success:
        st.markdown("### 🔧 Database Connection")
        st.warning("⚠️ Database connection failed. Please enter your PostgreSQL connection details.")
        
        # Database connection form
        with st.form("database_connection"):
            st.markdown("**PostgreSQL Connection Details:**")
            
            # Connection parameters
            host = st.text_input("Host", value="localhost", help="Database host (e.g., localhost)")
            port = st.text_input("Port", value="5432", help="Database port (default: 5432)")
            database = st.text_input("Database Name", value="ongctest4", help="Database name")
            username = st.text_input("Username", value="postgres", help="Database username")
            password = st.text_input("Password", type="password", help="Database password")
            
            # Or direct URL input
            st.markdown("**Or enter full connection URL:**")
            custom_url = st.text_input(
                "Database URL", 
                value=st.session_state.database_url,
                help="Format: postgresql://username:password@host:port/database"
            )
            
            # Test connection button
            if st.form_submit_button("🔗 Test Connection"):
                # Use custom URL if provided, otherwise build from components
                if custom_url and custom_url != st.session_state.database_url:
                    test_url = custom_url
                else:
                    test_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
                
                success, error = test_database_connection(test_url)
                
                if success:
                    st.session_state.database_url = test_url
                    st.success("✅ Database connection successful!")
                    st.rerun()
                else:
                    st.error(f"❌ Connection failed: {error}")
        
        # Show current status
        st.info(f"**Current URL:** {st.session_state.database_url}")
        return None
    else:
        return create_engine(st.session_state.database_url)

# Initialize database engine
engine = setup_database_connection()

with st.sidebar:
    # Add ONGC logo and title to sidebar
    _, center_col, _ = st.columns(3)
    with center_col:
        try:
            st.image("public/ongc (1).jpg", width=100)
        except:
            # Fallback if image is not found
            st.markdown("### ONGC")
    
    st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <span style='font-weight: bold; font-size: 24px;'>ONGC Data Toolkit</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Main category selection
    category = st.selectbox(
        "Select Category",
        ["Data Management", "Analysis Tools", "AI & Search"],
        format_func=lambda x: {
            "Data Management": "📂 Data Management",
            "Analysis Tools": "📊 Analysis Tools",
            "AI & Search": "🤖 AI & Search"
        }[x],
        key="category"
    )
    
    # Category descriptions
    category_descriptions = {
        "Data Management": "Tools for importing, cleaning, and standardizing your data files.",
        "Analysis Tools": "Analyze, compare, and generate insights from your data.",
        "AI & Search": "AI-powered tools for data analysis and advanced search capabilities."
    }
    
    # Show category description
    st.markdown(f"**Description:**\n{category_descriptions[category]}")
    st.markdown("---")

    # Tool descriptions dictionary with detailed information
    tool_descriptions = {
        "📅 Upload & Map": """Database Import and Mapping Tool

How to use:
1. Connect to PostgreSQL database
2. Choose target table
3. Upload Excel/CSV file
4. Map columns automatically or manually
5. Preview mapped data
6. Import to database

Features:
- Smart column mapping
- Data validation
- Duplicate detection
- Error checking""",

        "🧹 Clean & Edit": """Data Cleaning Tool

How to use:
1. Upload Excel/CSV file
2. Choose cleaning options:
   - Remove duplicates
   - Fix missing values
   - Format data
   - Clean text
3. Preview changes
4. Download cleaned file

Features:
- Batch cleaning
- Format fixing
- Text normalization
- Column management""",

        "🔗 Standardize Files": """File Standardization Tool

How to use:
1. Upload files
2. Choose standard format
3. Select options:
   - Column names
   - Data formats
   - Value rules
4. Download standardized files

Features:
- Schema matching
- Format detection
- Batch processing""",

        "📊 Compare Files": """File Comparison Tool

How to use:
1. Upload 2+ files
2. Choose comparison type:
   - Cell-by-cell
   - Column-wise
   - Structure only
3. View differences

Features:
- Visual highlighting
- Format checking
- Difference export""",

        "📊 Summary of Data Entry": """Data Analysis Tool

How to use:
1. Upload dataset
2. Select parameters:
   - Time period
   - Categories
   - Metrics
3. View summary

Features:
- Pattern detection
- Quality checks
- Trend analysis""",

        "📈 Data Type Wise": """Data Classification Tool

How to use:
1. Upload dataset
2. Choose criteria:
   - Data types
   - Categories
   - Time ranges
3. View analysis

Features:
- Auto-detection
- Pattern matching
- Distribution views""",

        "📑 Create Labels": """Physical Media Label Generator

How to use:
1. Prepare input file with required columns:
   - Media ID and Area/Block
   - Data Type and Format
   - Year and Shot Points

2. Configure label settings:
   - Choose ONGC standard/custom layout
   - Set barcode preferences
   - Adjust font sizes and fields

3. Generate and print:
   - Preview generated labels
   - Download as document
   - Print on label sheets

Label Contents:
- ONGC Logo and branding
- Media ID in large text
- Area and data type details
- File format and date
- Shot point ranges
- Optional barcode

Features:
- Multiple label templates
- Barcode generation
- Batch processing
- Print-ready output""",

        "🤖 AI Assistant": """AI-Powered Analysis Assistant

How to use:
1. Upload your data
2. Ask questions about:
   - Data patterns
   - Anomalies
   - Trends
   - Relationships
3. Get AI insights
4. Export findings

Features:
- Natural language queries
- Pattern recognition
- Smart recommendations
- Data visualization""",

        "🔍 Search": """Advanced Data Search Tool

How to use:
1. Upload data files
2. Configure search:
   - Choose search fields
   - Set match criteria
   - Define filters
3. Enter search terms
4. Review results

Features:
- Fuzzy matching
- Multi-file search
- Advanced filters
- Export results""",

        "🔄 Convert Files": """File Format Conversion Tool

How to use:
1. Choose conversion type:
   - CSV to Excel
   - Excel to CSV
   - DOCX to PDF
2. Upload source file
3. Click convert
4. Download converted file

Features:
- Multiple format support
- Batch conversion
- Preserves formatting
- Easy download"""
    }

    # Sub-options based on selected category with descriptions
    if category == "Data Management":
        selected_menu = st.radio(
            "Data Management Options",
            ["📅 Upload & Map", "🧹 Clean & Edit", "🔗 Standardize Files", "🔄 Convert Files"],
            key="data_mgmt_options",
            label_visibility="collapsed"
        )
    elif category == "Analysis Tools":
        selected_menu = st.radio(
            "Analysis Options",
            ["📊 Compare Files", "📊 Summary of Data Entry", "📈 Data Type Wise", "📑 Create Labels"],
            key="analysis_options",
            label_visibility="collapsed"
        )
    else:  # AI & Search
        selected_menu = st.radio(
            "AI & Search Options",
            ["🤖 AI Assistant", "🔍 Search"],
            key="ai_search_options",
            label_visibility="collapsed"
        )
    
    # Show tool description
    st.markdown("**Tool Info:**")
    st.info(tool_descriptions[selected_menu])

if selected_menu == "📅 Upload & Map":
    st.header("📅 PostgreSQL Database Manager")
    st.caption("Map and upload your Excel/CSV data to PostgreSQL database")

    # Check if database connection is available
    if engine is None:
        st.error("❌ Database connection is not available. Please configure the database connection above first.")
        st.info("💡 Use the database connection form above to connect to your PostgreSQL database.")
        st.stop()

    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.info("💡 Please check your database connection settings above.")
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

elif selected_menu == "📊 Compare Files":
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

elif selected_menu == "🧹 Clean & Edit":
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

elif selected_menu == "🔗 Standardize Files":
    link.standardize_datasets()

elif selected_menu == "📊 Summary of Data Entry":
    analyze_app()

elif selected_menu == "📈 Data Type Wise":
    # New functionality for Data Type Wise processing
    data_type_wise_app()

elif selected_menu == "🤖 AI Assistant":
    ai_chat()
    
elif selected_menu == "🔍 Search":
    fuzzy_search_ui()

elif selected_menu == "📑 Create Labels":
    # Launch the label creation functionality
    label_app()

elif selected_menu == "🔄 Convert Files":
    convert_app()