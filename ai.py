import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from datetime import datetime
import re
import subprocess

# Initialize models
@st.cache_resource
def load_language_models():
    embeddings = OllamaEmbeddings(model="deepseek-r1:latest")
    # Adjusted parameters for better context retention
    model = OllamaLLM(
        model="deepseek-r1:latest", 
        temperature=0.1,
        num_ctx=8192,  # Increased context window
        repeat_penalty=1.1,
        top_k=40,
        top_p=0.9
    )
def load_models():
    embeddings = OllamaEmbeddings(model="deepseek-r1:latest")
    model = OllamaLLM(model="deepseek-r1:latest", temperature=0.0)
    return embeddings, model

embeddings, model = load_models()

# Enhanced prompt templates with better context preservation
default_template = """
You are a professional data analyst working with tabular data from Excel or CSV files.

DATASET CONTEXT:
{context}

USER QUESTION: {question}

ANALYSIS GUIDELINES:
1. Use ONLY the data provided in the DATASET CONTEXT above
2. If information isn't available in the context, clearly state "Based on the provided data, I cannot determine..."
3. For calculations:
   - Show your work step by step
   - Cite specific column names and values
   - Double-check arithmetic
4. For data quality issues:
   - Point to specific examples from the data
   - Quantify the scope of issues
5. Always reference the actual column names and data structure shown in the context

RESPONSE FORMAT:
- Start with a direct answer to the question
- Follow with supporting details from the data
- End with actionable recommendations if applicable

Your analysis:
"""

summary_template = """
You are a data quality specialist creating a comprehensive analysis report.

DATASET INFORMATION:
{context}

TASK: Create a professional DATA QUALITY REPORT with these sections:

## DATASET OVERVIEW
- Dataset dimensions (rows × columns)
- Column inventory with data types
- Primary purpose/content of the dataset

## DATA QUALITY ASSESSMENT
### Missing Values
- Columns with missing data and percentages
- Impact assessment of missing values

### Data Consistency Issues
- Type mismatches or format inconsistencies
- Duplicate records analysis
- Range/boundary violations

### Data Integrity Concerns
- Outliers and anomalies
- Logical inconsistencies
- Referential integrity issues

## ACTIONABLE RECOMMENDATIONS
### Immediate Fixes (High Priority)
- Critical issues requiring immediate attention
- Specific Excel steps to resolve each issue

### Data Enhancement (Medium Priority)
- Improvements to data structure and quality
- Standardization recommendations

### Prevention Measures (Ongoing)
- Data validation rules to implement
- Quality control procedures

Provide specific, actionable guidance that an Excel user can immediately implement such providing with the VBA in excel and Excel formulas which can help .
"""

defects_template = """
You are an Excel data cleaning specialist. Analyze the following defect report and create a comprehensive repair plan.

DEFECT ANALYSIS:
{context}

Create a DETAILED DATA REPAIR GUIDE with these sections:

## EXECUTIVE SUMMARY
- Total defects identified
- Critical vs. non-critical issues
- Estimated effort required

## CRITICAL DEFECTS (Fix Immediately)
For each critical defect:
### Issue: [Defect Name]
- **Problem**: Detailed explanation
- **Impact**: How it affects data reliability
- **Excel Solution**:
  * Manual steps using Excel interface
  * Formula-based approach: `=FORMULA_HERE`
  * Alternative Power Query/VBA solution if needed
- **Validation**: How to verify the fix worked

## MODERATE DEFECTS (Schedule for Resolution)
[Similar format for medium-priority issues]

## MINOR DEFECTS (Address When Time Permits)
[Simplified fixes for low-priority issues]

## PREVENTION STRATEGY
- Data validation rules to implement
- Excel templates to standardize data entry
- Quality check procedures

Ensure all solutions are specific to Excel and include exact menu paths, formula syntax, and step-by-step instructions.
"""

# Improved function to convert dataframe to text chunks with better context preservation
def dataframe_to_text_chunks(df, chunk_size=2000, chunk_overlap=300, max_rows_per_chunk=75):
    """Convert dataframe to text chunks with enhanced context preservation."""
    return df_to_documents(df)

# Convert dataframe to documents
def df_to_documents(df):
    docs = []
    total_rows = len(df)  # Define total_rows here
    
    # Enhanced schema information with sample data
    schema_info = f"DATASET SCHEMA AND STRUCTURE:\n"
    schema_info += f"Total Dimensions: {total_rows} rows × {len(df.columns)} columns\n\n"
    schema_info += "COLUMN DETAILS:\n"
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()
        null_pct = (null_count / total_rows) * 100 if total_rows > 0 else 0
        
        # Get sample values (non-null)
        sample_values = df[col].dropna().astype(str).head(3).tolist()
        sample_str = f"Examples: {', '.join(sample_values)}" if sample_values else "No valid examples"
        
        schema_info += f"• {col}:\n"
        schema_info += f"  - Type: {dtype}\n"
        schema_info += f"  - Unique values: {unique_count:,}\n"
        schema_info += f"  - Missing: {null_count} ({null_pct:.1f}%)\n"
        schema_info += f"  - {sample_str}\n\n"
    
    schema_doc = Document(
        page_content=schema_info, 
        metadata={"chunk_type": "schema", "priority": "high"}
    )
    docs.append(schema_doc)
    
    # Enhanced statistics with context
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        stats_info = "NUMERIC COLUMNS STATISTICAL ANALYSIS:\n\n"
        for col in numeric_cols:
            try:
                stats = df[col].describe()
                stats_info += f"Column: {col}\n"
                stats_info += f"• Count: {int(stats['count']):,} valid values\n"
                stats_info += f"• Range: {stats['min']:.2f} to {stats['max']:.2f}\n"
                stats_info += f"• Average: {stats['mean']:.2f}\n"
                stats_info += f"• Median: {stats['50%']:.2f}\n"
                stats_info += f"• Standard Deviation: {stats['std']:.2f}\n"
                
                # Check for potential issues
                neg_count = (df[col] < 0).sum()
                zero_count = (df[col] == 0).sum()
                if neg_count > 0:
                    stats_info += f"• Negative values: {neg_count}\n"
                if zero_count > 0:
                    stats_info += f"• Zero values: {zero_count}\n"
                
                stats_info += "\n"
            except Exception as e:
                stats_info += f"Column: {col} - Statistics unavailable: {str(e)}\n\n"
        
        stats_doc = Document(
            page_content=stats_info, 
            metadata={"chunk_type": "statistics", "priority": "medium"}
        )
        docs.append(stats_doc)
    
    # Process data in chunks with better overlap handling
    max_rows_per_chunk = 75  # Define this variable
    num_row_chunks = (total_rows + max_rows_per_chunk - 1) // max_rows_per_chunk
    
    for i in range(num_row_chunks):
        start_idx = i * max_rows_per_chunk
        end_idx = min(start_idx + max_rows_per_chunk, total_rows)
        
        # Add overlap from previous chunk if not the first chunk
        if i > 0:
            overlap_start = max(0, start_idx - 5)  # 5 rows overlap
            subset_df = df.iloc[overlap_start:end_idx]
            rows_info = f"DATA CHUNK [{overlap_start+1} to {end_idx}] (includes 5-row overlap):\n"
        else:
            subset_df = df.iloc[start_idx:end_idx]
            rows_info = f"DATA CHUNK [{start_idx+1} to {end_idx}]:\n"
        
        # Convert to CSV with better formatting
        csv_buffer = StringIO()
        subset_df.to_csv(csv_buffer, index=True)  # Include row indices for reference
        csv_text = csv_buffer.getvalue()
        
        chunk_content = rows_info + csv_text
        
        chunk_doc = Document(
            page_content=chunk_content,
            metadata={
                "chunk_type": "data_rows",
                "row_start": start_idx,
                "row_end": end_idx,
                "num_rows": end_idx - start_idx,
                "total_rows": total_rows,
                "priority": "high" if i < 2 else "medium"  # Prioritize first chunks
            }
        )
        docs.append(chunk_doc)
    
    return docs

# Enhanced retrieval with better context management
def retrieve_docs(db, query, k=6):
    """Enhanced document retrieval with better context awareness."""
    query_lower = query.lower()
    
    # Always get schema information
    schema_docs = []
    try:
        schema_results = db.similarity_search(query, k=2, filter={"chunk_type": "schema"})
        schema_docs.extend(schema_results)
    except:
        pass
    
    # Determine query type and adjust retrieval strategy
    stats_keywords = ["average", "mean", "maximum", "minimum", "count", "statistics", "sum", "total"]
    structure_keywords = ["column", "field", "schema", "structure", "type", "datatype"]
    
    if any(keyword in query_lower for keyword in stats_keywords):
        # Statistics-focused query
        try:
            stats_docs = db.similarity_search(query, k=2, filter={"chunk_type": "statistics"})
            data_docs = db.similarity_search(query, k=k-len(schema_docs)-len(stats_docs))
            return schema_docs + stats_docs + data_docs
        except:
            pass
    elif any(keyword in query_lower for keyword in structure_keywords):
        # Structure-focused query
        data_docs = db.similarity_search(query, k=k-len(schema_docs))
        return schema_docs + data_docs
    
    # General query - get mixed results with priority weighting
    try:
        # Get high-priority chunks first
        high_priority_docs = db.similarity_search(query, k=k//2, filter={"priority": "high"})
        remaining_k = k - len(high_priority_docs)
        if remaining_k > 0:
            other_docs = db.similarity_search(query, k=remaining_k)
            # Remove duplicates
            seen_content = {doc.page_content for doc in high_priority_docs}
            other_docs = [doc for doc in other_docs if doc.page_content not in seen_content]
            return high_priority_docs + other_docs[:remaining_k]
        return high_priority_docs
    except:
        # Fallback to simple similarity search
        return db.similarity_search(query, k=k)

# Significantly improved query function with better context handling
def question_df(question, documents, prompt_template=None, input_vars=None):
    """Enhanced query function with improved context preservation and error handling."""
    try:
        # Build comprehensive context from documents
        context_parts = []
        
        # Organize documents by type for better context flow
        schema_docs = [doc for doc in documents if doc.metadata.get("chunk_type") == "schema"]
        stats_docs = [doc for doc in documents if doc.metadata.get("chunk_type") == "statistics"]
        data_docs = [doc for doc in documents if doc.metadata.get("chunk_type") == "data_rows"]
        
        # Always include schema for context
        for doc in schema_docs:
            context_parts.append("=== DATASET STRUCTURE ===")
            context_parts.append(doc.page_content)
            context_parts.append("")
        
        # Include statistics if relevant or available
        if stats_docs:
            context_parts.append("=== STATISTICAL SUMMARY ===")
            for doc in stats_docs:
                context_parts.append(doc.page_content)
            context_parts.append("")
        
        # Add data samples with clear separation
        if data_docs:
            context_parts.append("=== DATA SAMPLES ===")
            for i, doc in enumerate(data_docs):
                if i > 0:
                    context_parts.append("--- Next Data Section ---")
                context_parts.append(doc.page_content)
                context_parts.append("")
        
        # Combine all context with clear structure
        full_context = "\n".join(context_parts)
        
        # Use appropriate prompt template
        if prompt_template and input_vars:
            # For custom templates, ensure context is properly included
            if "context" not in input_vars:
                input_vars["context"] = full_context
            elif "{context}" in prompt_template:
                # Replace or supplement existing context
                input_vars["context"] = full_context
            
            prompt = ChatPromptTemplate.from_template(prompt_template)
        else:
            # Default template with enhanced context
            prompt = ChatPromptTemplate.from_template(default_template)
            input_vars = {"question": question, "context": full_context}
        
        # Create and execute the chain
        chain = prompt | model
        
        # Enhanced retry logic with exponential backoff
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries + 1):
            try:
                # Add small delay to prevent overwhelming the model
                if attempt > 0:
                    time.sleep(base_delay * (2 ** (attempt - 1)))
                
                response = chain.invoke(input_vars)
                
                # Validate response quality
                if len(response.strip()) < 10:
                    raise ValueError("Response too short, likely incomplete")
                
                return response
                
            except Exception as e:
                error_msg = str(e).lower()
                if attempt < max_retries:
                    if any(keyword in error_msg for keyword in ["timeout", "connection", "rate"]):
                        continue  # Retry for network issues
                    elif "context" in error_msg or "token" in error_msg:
                        # Try with reduced context
                        if len(full_context) > 4000:
                            # Truncate context and try again
                            truncated_context = full_context[:4000] + "\n... [Context truncated due to length limits]"
                            input_vars["context"] = truncated_context
                            continue
                
                # If all retries failed, provide helpful error message
                if "context" in error_msg or "token" in error_msg:
                    return ("I apologize, but the dataset is too large for me to process in a single response. "
                           "Please try asking about a specific aspect of your data (e.g., 'analyze column X' or "
                           "'show me the first 10 rows') or consider breaking your question into smaller parts.")
                else:
                    return f"I encountered an error while analyzing your data: {str(e)}. Please try rephrasing your question or check if Ollama is running properly."
                
    except Exception as e:
        return f"Error processing your query: {str(e)}. Please ensure your question is clear and try again."

# Rest of your code remains the same...
# [Include all the other functions like create_vector_store_from_df, get_df_summary, 
# detect_defects_and_format_issues, and ai_chat function here]

# Create vector store from dataframe chunks with improved metadata
def create_vector_store_from_df(df):
    """Create a FAISS vector store from dataframe with improved chunking."""
    chunked_docs = dataframe_to_text_chunks(df)
    vector_store = FAISS.from_documents(chunked_docs, embeddings)
    return vector_store

# Enhanced retrieval function with improved ranking
def retrieve_docs(db, query, k=5):
    """Retrieve relevant documents with smarter filtering based on the query."""
    # First, determine if the question is about:
    # 1. Schema/structure (columns, types)
    # 2. Statistics (min, max, average)
    # 3. Specific data rows or filtering
    
    query_lower = query.lower()
    
    # Adjust k based on query type
    k_final = k
    filter_metadata = {}
    
    # Check if the query is about schema or structure
    if any(term in query_lower for term in ["column", "field", "schema", "structure", "data type", "datatype"]):
        # Prioritize schema chunks
        filter_metadata = {"chunk_type": "schema"}
        k_final = min(k, 2)  # Don't need too many schema chunks
    
    # Check if the query is about statistics
    elif any(term in query_lower for term in ["average", "mean", "maximum", "minimum", "count", "statistic"]):
        # Prioritize statistics chunks
        filter_metadata = {"chunk_type": "statistics"}
        k_final = min(k, 3)  # Get a few statistics chunks
    
    # For specific row queries, we'll use the default behavior
    
    # First try with metadata filter if applicable
    if filter_metadata:
        try:
            filtered_docs = db.similarity_search(query, k=k_final, filter=filter_metadata)
            if filtered_docs:
                # If we got results with filter, add some general results too
                general_docs = db.similarity_search(query, k=k-len(filtered_docs))
                return filtered_docs + general_docs
        except:
            # If filtering fails, fall back to regular search
            pass
    
    # Regular similarity search
    return db.similarity_search(query, k=k)

# Improved dataframe summary with more comprehensive checks
def get_df_summary(df):
    """Generate a comprehensive summary of the dataframe."""
    summary_parts = []
    
    # Basic dimensions
    rows, cols = df.shape
    summary_parts.append(f"DATASET DIMENSIONS: {rows} rows × {cols} columns")
    
    # Column names and types
    summary_parts.append("\nCOLUMN INFORMATION:")
    # Data chunks
    for i in range(0, len(df), 50):
        chunk = df.iloc[i:i+50].to_csv(index=True)
        docs.append(Document(page_content=f"Rows {i+1}-{min(i+50, len(df))}:\n{chunk}"))
    
    return docs

# Query function
def query_data(question, vector_store):
    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = ChatPromptTemplate.from_template("""
    Based on this dataset information:
    {context}
    
    Question: {question}
    
    Provide a clear, specific answer using only the data shown above.
    """)
    
    chain = prompt | model
    return chain.invoke({"context": context, "question": question})

# Professional Data Summary Generator
def generate_professional_summary(df):
    """Generate a comprehensive professional data summary"""
    
    # Basic Dataset Information
    summary = {
        'dataset_info': {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        'column_analysis': {},
        'data_types': df.dtypes.value_counts().to_dict(),
        'completeness_score': ((df.count().sum() / (len(df) * len(df.columns))) * 100)
    }
    
    # Detailed column analysis
    for col in df.columns:
        col_info = {
            'data_type': str(df[col].dtype),
            'non_null_count': df[col].count(),
            'null_count': df[col].isnull().sum(),
            'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
            'unique_values': df[col].nunique(),
            'uniqueness_ratio': (df[col].nunique() / len(df)) * 100
        }
        
        # Type-specific analysis
        if df[col].dtype in ['int64', 'float64']:
            col_info.update({
                'min_value': df[col].min(),
                'max_value': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std_deviation': df[col].std(),
                'outliers_count': len(df[(np.abs(df[col] - df[col].mean()) > 3 * df[col].std())]) if df[col].std() > 0 else 0
            })
        elif df[col].dtype == 'object':
            col_info.update({
                'avg_length': df[col].astype(str).str.len().mean(),
                'max_length': df[col].astype(str).str.len().max(),
                'contains_email': df[col].astype(str).str.contains(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', na=False).sum(),
                'contains_phone': df[col].astype(str).str.contains(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', na=False).sum(),
                'contains_url': df[col].astype(str).str.contains(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', na=False).sum()
            })
        
        summary['column_analysis'][col] = col_info
    
    return summary

# Comprehensive Data Quality Assessment
def comprehensive_quality_check(df):
    """Perform enterprise-grade data quality assessment"""
    
    quality_issues = {
        'critical': [],
        'major': [],
        'minor': [],
        'recommendations': []
    }
    
    # 1. Missing Data Analysis
    missing_data = df.isnull().sum()
    for col, count in missing_data[missing_data > 0].items():
        percentage = (count / len(df)) * 100
        issue = {
            'column': col,
            'missing_count': count,
            'percentage': percentage,
            'excel_solution': f"=COUNTA({col}:{col})-ROWS({col}:{col}) to count missing | Use Find & Replace (Ctrl+H) to replace blanks with appropriate values"
        }
        
        if percentage > 50:
            issue['severity'] = 'Critical'
            quality_issues['critical'].append(issue)
        elif percentage > 20:
            issue['severity'] = 'Major'
            quality_issues['major'].append(issue)
        else:
            issue['severity'] = 'Minor'
            quality_issues['minor'].append(issue)
    
    # 2. Duplicate Records
    duplicates = df.duplicated()
    if duplicates.sum() > 0:
        quality_issues['major'].append({
            'issue': 'Duplicate Records',
            'count': duplicates.sum(),
            'percentage': (duplicates.sum() / len(df)) * 100,
            'excel_solution': "Data > Remove Duplicates or use COUNTIFS() to identify duplicates"
        })
    
    # 3. Data Type Inconsistencies
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check for mixed data types
            sample_values = df[col].dropna().astype(str).head(100)
            numeric_pattern = sample_values.str.match(r'^-?\d+\.?\d*$')
            date_pattern = sample_values.str.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}')
            
            if numeric_pattern.sum() > len(sample_values) * 0.8:
                quality_issues['minor'].append({
                    'issue': f'Column "{col}" contains mostly numeric data but stored as text',
                    'excel_solution': f'Select column > Data > Text to Columns > Choose appropriate format or use VALUE() function'
                })
            
            if date_pattern.sum() > len(sample_values) * 0.8:
                quality_issues['minor'].append({
                    'issue': f'Column "{col}" contains date data but not properly formatted',
                    'excel_solution': f'Select column > Format Cells > Date or use DATEVALUE() function'
                })
    
    # 4. Outlier Detection for Numeric Columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].std() > 0:
            outliers = df[(np.abs(df[col] - df[col].mean()) > 3 * df[col].std())]
            if len(outliers) > 0:
                quality_issues['minor'].append({
                    'issue': f'Column "{col}" has {len(outliers)} potential outliers',
                    'excel_solution': f'Use conditional formatting to highlight outliers or =ABS({col}-AVERAGE({col}))>3*STDEV({col})'
                })
    
    # 5. Inconsistent Formatting
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().astype(str)
            if len(sample) > 0:
                # Check for leading/trailing spaces
                spaces = sample.str.strip() != sample
                if spaces.sum() > 0:
                    quality_issues['minor'].append({
                        'issue': f'Column "{col}" has {spaces.sum()} entries with leading/trailing spaces',
                        'excel_solution': f'Use TRIM() function or Find & Replace to remove extra spaces'
                    })
                
                # Check for case inconsistencies
                if sample.nunique() != sample.str.lower().nunique():
                    quality_issues['minor'].append({
                        'issue': f'Column "{col}" has inconsistent capitalization',
                        'excel_solution': f'Use PROPER(), UPPER(), or LOWER() functions for consistent formatting'
                    })
    
    # 6. Data Completeness Score
    completeness = (df.count().sum() / (len(df) * len(df.columns))) * 100
    if completeness < 95:
        quality_issues['recommendations'].append({
            'recommendation': f'Overall data completeness is {completeness:.1f}%. Target should be >95%',
            'action': 'Review data collection processes and implement validation rules'
        })
    
    # 7. Column Name Quality
    problematic_columns = []
    for col in df.columns:
        if ' ' in col or col != col.strip():
            problematic_columns.append(col)
    
    if problematic_columns:
        quality_issues['minor'].append({
            'issue': f'Column names have spaces or formatting issues: {problematic_columns}',
            'excel_solution': 'Rename columns to use underscores instead of spaces for better compatibility'
        })
    
    return quality_issues

# Main app
def ai_chat():
    
    st.title("🤖 Excel & CSV Data Assistant")
    st.caption("Powered by Ollama and deepseek-r1:latest • Your local AI data analyst")
    
    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_history" not in st.session_state:
        st.session_state.show_history = False
    if "faq_response" not in st.session_state:
        st.session_state.faq_response = None
    if "dataframes" not in st.session_state:
        st.session_state.dataframes = []
    if "file_names" not in st.session_state:
        st.session_state.file_names = []
    
    # Session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "df" not in st.session_state:
        st.session_state.df = None
    
    # Place this check at the top of your AI assistant page or before model usage
    if not check_ollama_model("deepseek-r1"):
        show_ollama_setup("deepseek-r1")
        st.stop()
    
    # File upload
    st.markdown("### 📤 Data Import")
    uploaded_file = st.file_uploader(
        "Upload your dataset for analysis", 
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV, Excel (.xlsx, .xls)"
    )
    
    if uploaded_file:
        # Load data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            
            # Data Preview
            with st.expander("📋 Dataset Overview", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("📊 Total Records", f"{len(df):,}")
                col2.metric("📋 Columns", f"{len(df.columns):,}")
                col3.metric("❌ Missing Values", f"{df.isnull().sum().sum():,}")
                col4.metric("💾 Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                
                st.dataframe(df.head(10), use_container_width=True)
            
            # Create vector store
            if st.session_state.vector_store is None:
                with st.spinner("🔄 Building search index..."):
                    docs = df_to_documents(df)
                    st.session_state.vector_store = FAISS.from_documents(docs, embeddings)
                    st.success("✅ Analysis engine ready!")
            
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
    
    # Analysis Interface
    if st.session_state.vector_store and st.session_state.df is not None:
        df = st.session_state.df
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Generate Professional Summary", use_container_width=True):
                with st.spinner("Generating comprehensive summary..."):
                    summary = generate_professional_summary(df)
                    
                    st.markdown("## 📈 Executive Data Summary Report")
                    st.markdown(f"**Generated:** {summary['dataset_info']['analysis_timestamp']}")
                    st.markdown("---")
                    
                    # Dataset Overview
                    st.markdown("### 🎯 Dataset Overview")
                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                    with metrics_col1:
                        st.metric("Total Records", f"{summary['dataset_info']['total_records']:,}")
                    with metrics_col2:
                        st.metric("Columns", f"{summary['dataset_info']['total_columns']:,}")
                    with metrics_col3:
                        st.metric("Data Completeness", f"{summary['completeness_score']:.1f}%")
                    with metrics_col4:
                        st.metric("Memory Footprint", summary['dataset_info']['memory_usage'])
                    
                    # Data Type Distribution
                    st.markdown("### 📊 Data Type Distribution")
                    dtype_df = pd.DataFrame(list(summary['data_types'].items()), columns=['Data Type', 'Count'])
                    st.bar_chart(dtype_df.set_index('Data Type'))
                    
                    # Column Analysis
                    st.markdown("### 🔍 Detailed Column Analysis")
                    
                    analysis_data = []
                    for col, info in summary['column_analysis'].items():
                        analysis_data.append({
                            'Column': col,
                            'Data Type': info['data_type'],
                            'Non-Null Count': f"{info['non_null_count']:,}",
                            'Missing %': f"{info['null_percentage']:.1f}%",
                            'Unique Values': f"{info['unique_values']:,}",
                            'Uniqueness %': f"{info['uniqueness_ratio']:.1f}%"
                        })
                    
                    analysis_df = pd.DataFrame(analysis_data)
                    st.dataframe(analysis_df, use_container_width=True, hide_index=True)
                    
                    # Statistical Summary for Numeric Columns
                    numeric_cols = [col for col, info in summary['column_analysis'].items() 
                                  if 'mean' in info]
                    
                    if numeric_cols:
                        st.markdown("### 📈 Statistical Summary (Numeric Columns)")
                        stats_data = []
                        for col in numeric_cols:
                            info = summary['column_analysis'][col]
                            stats_data.append({
                                'Column': col,
                                'Mean': f"{info['mean']:.2f}" if not pd.isna(info['mean']) else 'N/A',
                                'Median': f"{info['median']:.2f}" if not pd.isna(info['median']) else 'N/A',
                                'Std Dev': f"{info['std_deviation']:.2f}" if not pd.isna(info['std_deviation']) else 'N/A',
                                'Min': f"{info['min_value']:.2f}" if not pd.isna(info['min_value']) else 'N/A',
                                'Max': f"{info['max_value']:.2f}" if not pd.isna(info['max_value']) else 'N/A',
                                'Outliers': info['outliers_count']
                            })
                        
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            if st.button("🔍 Comprehensive Quality Assessment", use_container_width=True):
                with st.spinner("Performing comprehensive quality analysis..."):
                    quality_report = comprehensive_quality_check(df)
                    
                    st.markdown("## 🛡️ Data Quality Assessment Report")
                    st.markdown(f"**Assessment Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    st.markdown("---")
                    
                    # Quality Score
                    total_issues = len(quality_report['critical']) + len(quality_report['major']) + len(quality_report['minor'])
                    if total_issues == 0:
                        quality_score = 100
                    else:
                        # Weight issues by severity
                        weighted_issues = (len(quality_report['critical']) * 3 + 
                                         len(quality_report['major']) * 2 + 
                                         len(quality_report['minor']) * 1)
                        quality_score = max(0, 100 - (weighted_issues * 2))
                    
                    # Quality Score Display
                    col_score1, col_score2, col_score3, col_score4 = st.columns(4)
                    with col_score1:
                        st.metric("🎯 Quality Score", f"{quality_score:.0f}/100")
                    with col_score2:
                        st.metric("🚨 Critical Issues", len(quality_report['critical']))
                    with col_score3:
                        st.metric("⚠️ Major Issues", len(quality_report['major']))
                    with col_score4:
                        st.metric("ℹ️ Minor Issues", len(quality_report['minor']))
                    
                    # Critical Issues
                    if quality_report['critical']:
                        st.markdown("### 🚨 Critical Issues (Immediate Action Required)")
                        for i, issue in enumerate(quality_report['critical'], 1):
                            issue_title = issue.get('issue', f'Missing data in {issue.get("column", "Unknown Column")}')
                            with st.expander(f"Critical Issue #{i}: {issue_title}", expanded=True):
                                if 'column' in issue:
                                    st.error(f"**Column:** {issue['column']}")
                                    st.error(f"**Missing Values:** {issue['missing_count']:,} ({issue['percentage']:.1f}%)")
                                else:
                                    st.error(f"**Issue:** {issue['issue']}")
                                    if 'count' in issue:
                                        st.error(f"**Count:** {issue['count']:,} ({issue.get('percentage', 0):.1f}%)")
                                
                                st.markdown("**💡 Excel Solution:**")
                                st.code(issue['excel_solution'], language='text')
                    
                    # Major Issues
                    if quality_report['major']:
                        st.markdown("### ⚠️ Major Issues (High Priority)")
                        for i, issue in enumerate(quality_report['major'], 1):
                            issue_title = issue.get('issue', f'Missing data in {issue.get("column", "Unknown Column")}')
                            with st.expander(f"Major Issue #{i}: {issue_title}", expanded=False):
                                if 'column' in issue:
                                    st.warning(f"**Column:** {issue['column']}")
                                    st.warning(f"**Missing Values:** {issue['missing_count']:,} ({issue['percentage']:.1f}%)")
                                else:
                                    st.warning(f"**Issue:** {issue['issue']}")
                                    if 'count' in issue:
                                        st.warning(f"**Count:** {issue['count']:,} ({issue.get('percentage', 0):.1f}%)")
                                
                                st.markdown("**💡 Excel Solution:**")
                                st.code(issue['excel_solution'], language='text')
                    
                    # Minor Issues
                    if quality_report['minor']:
                        st.markdown("### ℹ️ Minor Issues (Medium Priority)")
                        for i, issue in enumerate(quality_report['minor'], 1):
                            issue_title = issue.get('issue', f'Missing data in {issue.get("column", "Unknown Column")}')
                            with st.expander(f"Minor Issue #{i}: {issue_title}", expanded=False):
                                if 'column' in issue:
                                    st.info(f"**Column:** {issue['column']}")
                                    st.info(f"**Missing Values:** {issue['missing_count']:,} ({issue['percentage']:.1f}%)")
                                else:
                                    st.info(f"**Issue:** {issue['issue']}")
                                
                                st.markdown("**💡 Excel Solution:**")
                                st.code(issue['excel_solution'], language='text')
                    
                    # Recommendations
                    if quality_report['recommendations']:
                        st.markdown("### 🎯 Strategic Recommendations")
                        for i, rec in enumerate(quality_report['recommendations'], 1):
                            st.success(f"**{i}.** {rec['recommendation']}")
                            st.markdown(f"**Action:** {rec['action']}")
                    
                    # Export Options
                    st.markdown("### 📤 Export Quality Report")
                    if st.button("Generate Detailed Report"):
                        # Create detailed report text
                        report_text = f"""
DATA QUALITY ASSESSMENT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: {uploaded_file.name}

EXECUTIVE SUMMARY
Quality Score: {quality_score}/100
Total Issues Found: {total_issues}
- Critical: {len(quality_report['critical'])}
- Major: {len(quality_report['major'])}
- Minor: {len(quality_report['minor'])}

DETAILED FINDINGS
{'-' * 50}
"""
                        # Add all issues to report
                        for severity, issues in quality_report.items():
                            if severity != 'recommendations' and issues:
                                report_text += f"\n{severity.upper()} ISSUES:\n"
                                for issue in issues:
                                    if 'column' in issue:
                                        report_text += f"- Column '{issue['column']}': {issue['missing_count']} missing values ({issue['percentage']:.1f}%)\n"
                                        report_text += f"  Solution: {issue['excel_solution']}\n"
                                    else:
                                        report_text += f"- {issue['issue']}\n"
                                        report_text += f"  Solution: {issue['excel_solution']}\n"
                        
                        st.download_button(
                            label="📄 Download Quality Report",
                            data=report_text,
                            file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
        
        # Custom Query Interface
        st.markdown("### 💬 AI Data Assistant")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            question = st.text_input(
                "Ask specific questions about your data:",
                placeholder="e.g., What are the key patterns in this dataset?",
                help="Ask detailed questions about your data structure, patterns, or specific analyses"
            )
        
        with col2:
            if st.button("🤖 Analyze", use_container_width=True) and question:
                with st.spinner("AI analyzing your data..."):
                    try:
                        response = query_data(question, st.session_state.vector_store)
                        st.markdown("#### 🎯 Analysis Results")
                        st.write(response)
                    except Exception as e:
                        st.error(f"Analysis error: {e}")
        
        # Quick Analysis Buttons
        st.markdown("### ⚡ Quick Analysis")
        quick_cols = st.columns(4)
        
        quick_questions = [
            ("📊 Statistical Overview", "Provide statistical overview of all numeric columns"),
            ("🔍 Data Patterns", "Identify key patterns and relationships in the data"),
            ("❌ Missing Data Analysis", "Analyze missing data patterns and suggest treatment"),
            ("📈 Column Insights", "Provide insights about each column's characteristics")
        ]
        
        for i, (label, query) in enumerate(quick_questions):
            with quick_cols[i]:
                if st.button(label, use_container_width=True):
                    with st.spinner("Analyzing..."):
                        try:
                            response = query_data(query, st.session_state.vector_store)
                            st.markdown(f"#### {label}")
                            st.write(response)
                        except Exception as e:
                            st.error(f"Error: {e}")
    
    else:
        st.markdown("### 📊 AI Data Assistant")
        st.markdown("Upload a dataset to get started with data analysis and insights.")
        

def check_ollama_model(model_name="deepseek-r1"):
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model_name in result.stdout:
            return True
        return False
    except Exception:
        return False

def show_ollama_setup(model_name="deepseek-r1"):
    st.error(f"Failed to connect to Ollama or the '{model_name}' model is not available.")
    st.markdown(
        f"""
        **To use this feature, you need:**
        1. [Ollama installed and running](https://ollama.com/download)
        2. The `{model_name}` model pulled

        **To pull the model, run this in your terminal:**
        ```bash
        ollama pull {model_name}
        ```
        """
    )
    if st.button(f"Copy 'ollama pull {model_name}' to clipboard"):
        st.code(f"ollama pull {model_name}", language="bash")

if __name__ == "__main__":
    ai_chat()