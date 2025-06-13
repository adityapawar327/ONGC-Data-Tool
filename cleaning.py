import streamlit as st
import pandas as pd
import numpy as np
import re

def clean_data_ui(df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("### 🧼 Data Cleaning & Manipulation Options")

    # Step 0: Select column to filter rows for targeted cleaning
    st.markdown("#### Select a column to filter rows for cleaning (optional):")
    identifier_col = st.selectbox("Choose a column to filter rows by its values (or leave blank)", options=[None] + list(df.columns))

    # Filter rows based on selected values in the identifier column
    if identifier_col:
        unique_values = df[identifier_col].dropna().unique().tolist()
        selected_values = st.multiselect(
            f"Select values from '{identifier_col}' to apply cleaning (leave empty to select all rows)",
            unique_values
        )
        if selected_values:
            working_df = df[df[identifier_col].isin(selected_values)].copy()
        else:
            working_df = df.copy()
    else:
        working_df = df.copy()

    # Data Profiling
    with st.expander("📊 Data Profile"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("Basic Information:")
            st.write(f"- Total Rows: {len(working_df)}")
            st.write(f"- Total Columns: {len(working_df.columns)}")
            st.write(f"- Memory Usage: {working_df.memory_usage().sum() / 1024 / 1024:.2f} MB")

        with col2:
            st.write("Data Types:")
            for dtype in working_df.dtypes.value_counts().items():
                st.write(f"- {dtype[0]}: {dtype[1]} columns")

        if st.checkbox("Show detailed statistics"):
            # Numeric columns statistics
            numeric_cols = working_df.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                st.write("### Numeric Columns Statistics")
                stats_df = working_df[numeric_cols].describe()
                st.dataframe(stats_df, use_container_width=True)

            # Categorical columns statistics
            cat_cols = working_df.select_dtypes(exclude=[np.number]).columns
            if not cat_cols.empty:
                st.write("### Categorical Columns Statistics")
                for col in cat_cols:
                    st.markdown(f"**{col}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        value_counts = working_df[col].value_counts()
                        st.write(f"- Unique Values: {working_df[col].nunique()}")
                        st.write(f"- Null Values: {working_df[col].isnull().sum()}")
                    with col2:
                        if len(value_counts) > 0:
                            st.write("Top 10 Values:")
                            st.dataframe(value_counts.head(10))
                    st.markdown("---")

        # Missing values visualization
        if st.checkbox("Show missing values analysis"):
            missing_data = (working_df.isnull().sum() / len(working_df) * 100).round(2)
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if not missing_data.empty:
                st.write("### Missing Values Analysis")
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing %': missing_data.values
                })
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("No missing values found in the dataset!")

    # NEW: Handle Merged Cells
    with st.expander("0. Handle Merged Cells (Fill Down Based on Serial Number)"):
        st.markdown("**Fill down values for merged cells based on serial number grouping**")
        
        # Select serial number column
        serial_col = st.selectbox("Select Serial Number column", options=[None] + list(working_df.columns), key="serial_col")
        
        if serial_col:
            # Select columns to fill down
            filldown_cols = st.multiselect(
                "Select columns to fill down for same serial numbers",
                [col for col in working_df.columns if col != serial_col],
                key="filldown_cols"
            )
            
            if st.button("🔄 Fill Down Merged Cells") and filldown_cols:
                # Process each selected column
                for col in filldown_cols:
                    # Group by serial number and forward fill within each group
                    working_df[col] = working_df.groupby(serial_col)[col].transform(lambda x: x.fillna(method='ffill'))
                    
                    # If there are still NaN values, try backward fill within the same group
                    working_df[col] = working_df.groupby(serial_col)[col].transform(lambda x: x.fillna(method='bfill'))
                
                st.success(f"✅ Filled down values for columns: {', '.join(filldown_cols)} based on '{serial_col}'")
                
                # Show preview of changes
                st.markdown("**Preview of filled data:**")
                preview_df = working_df[[serial_col] + filldown_cols].head(10)
                st.dataframe(preview_df)
            
            # Option to fill with specific value for remaining nulls
            if st.checkbox("Fill remaining null values with custom text"):
                custom_fill = st.text_input("Enter text for remaining null values:", value="Not Available")
                target_cols = st.multiselect(
                    "Select columns to fill remaining nulls",
                    filldown_cols if 'filldown_cols' in locals() else working_df.columns.tolist(),
                    key="custom_fill_cols"
                )
                
                if st.button("Fill Remaining Nulls") and target_cols and custom_fill:
                    for col in target_cols:
                        working_df[col] = working_df[col].fillna(custom_fill)
                    st.success(f"✅ Filled remaining null values with '{custom_fill}' in columns: {', '.join(target_cols)}")

    # 1. Remove Duplicates
    with st.expander("1. Remove Duplicates"):
        if st.checkbox("Remove duplicate rows (on selected rows)"):
            before_count = len(working_df)
            working_df = working_df.drop_duplicates()
            st.success(f"✅ Duplicates removed. Rows before: {before_count}, after: {len(working_df)}")

    # 2. Handle Missing Values
    with st.expander("2. Handle Missing Values"):
        na_action = st.radio("Choose method for missing values", ["None", "Drop rows", "Fill with value", "Fill numeric with mean"])
        if na_action == "Drop rows":
            before_count = len(working_df)
            working_df = working_df.dropna()
            st.success(f"✅ Dropped rows with missing values. Rows before: {before_count}, after: {len(working_df)}")
        elif na_action == "Fill with value":
            fill_value = st.text_input("Value to fill missing cells with:")
            if fill_value:
                working_df = working_df.fillna(fill_value)
                st.success(f"✅ Filled missing cells with '{fill_value}'")
        elif na_action == "Fill numeric with mean":
            numeric_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                for col in numeric_cols:
                    mean_val = working_df[col].mean()
                    working_df[col] = working_df[col].fillna(mean_val)
                st.success(f"✅ Filled missing numeric values with column means")
            else:
                st.info("No numeric columns available for this operation.")

    # 3. Strip Whitespace and Normalize Spaces
    with st.expander("3. Strip Whitespace and Normalize Spaces"):
        if st.checkbox("Strip leading/trailing whitespace from string columns"):
            working_df = working_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            st.success("✅ Whitespace stripped from string columns")
        if st.checkbox("Replace multiple spaces inside strings with single space"):
            def fix_spaces(x):
                if isinstance(x, str):
                    return re.sub(r'\s+', ' ', x)
                else:
                    return x
            working_df = working_df.applymap(fix_spaces)
            st.success("✅ Normalized multiple spaces to single spaces")

    # NEW: Trim String Length
    with st.expander("4. Trim String Length"):
        string_cols = working_df.select_dtypes(include=['object']).columns.tolist()
        if string_cols:
            trim_cols = st.multiselect(
                "Select columns to trim",
                string_cols,
                key="trim_cols"
            )
            max_length = st.number_input("Maximum length", min_value=1, value=255)
            if st.button("Trim Selected Columns") and trim_cols:
                for col in trim_cols:
                    working_df[col] = working_df[col].astype(str).str.slice(0, max_length)
                st.success(f"✅ Trimmed {len(trim_cols)} columns to maximum length of {max_length}")
        else:
            st.info("No string columns available for trimming")

    # 4. Remove Special Characters
    with st.expander("5. Remove Special Characters"):
        special_cols = working_df.select_dtypes(include='object').columns.tolist()
        remove_chars_cols = st.multiselect("Select columns to remove special characters from", special_cols)
        if remove_chars_cols:
            def remove_special(x):
                if isinstance(x, str):
                    return re.sub(r'[^A-Za-z0-9\s]', '', x)
                else:
                    return x
            for col in remove_chars_cols:
                working_df[col] = working_df[col].apply(remove_special)
            st.success("✅ Removed special characters from selected columns")

    # 5. Normalize Case
    with st.expander("6. Normalize Case"):
        for col in working_df.select_dtypes(include='object').columns:
            case_option = st.selectbox(f"Change case for column '{col}'", ["None", "lower", "upper", "title"], key=f"case_{col}")
            if case_option == "lower":
                working_df[col] = working_df[col].str.lower()
            elif case_option == "upper":
                working_df[col] = working_df[col].str.upper()
            elif case_option == "title":
                working_df[col] = working_df[col].str.title()

    # 6. Rename Columns
    with st.expander("7. Rename Columns"):
        new_col_names = {}
        for col in working_df.columns:
            new_name = st.text_input(f"Rename column '{col}' to:", value=col, key=f"rename_{col}")
            new_col_names[col] = new_name
        working_df.rename(columns=new_col_names, inplace=True)

    # 7. Change Data Types
    with st.expander("8. Change Data Types"):
        for col in working_df.columns:
            dtype_option = st.selectbox(f"Change data type for column '{col}'", ["No change", "int", "float", "str"], key=f"dtype_{col}")
            try:
                if dtype_option == "int":
                    working_df[col] = pd.to_numeric(working_df[col], errors='coerce').fillna(0).astype(int)
                elif dtype_option == "float":
                    working_df[col] = pd.to_numeric(working_df[col], errors='coerce').astype(float)
                elif dtype_option == "str":
                    working_df[col] = working_df[col].astype(str)
            except Exception as e:
                st.warning(f"Couldn't convert '{col}' to {dtype_option}: {e}")

    # 8. Add New Column
    with st.expander("9. Add New Column"):
        new_col_name = st.text_input("New column name:")
        new_col_default = st.text_input("Default value for new column (optional):")
        if st.button("➕ Add Column") and new_col_name:
            if new_col_name in working_df.columns:
                st.warning("Column name already exists!")
            else:
                if new_col_default:
                    working_df[new_col_name] = new_col_default
                else:
                    working_df[new_col_name] = np.nan
                st.success(f"✅ Added new column '{new_col_name}'")

    # 9. Delete Columns
    with st.expander("10. Delete Columns"):
        del_cols = st.multiselect("Select columns to delete", working_df.columns.tolist())
        if st.button("🗑️ Delete Selected Columns"):
            if del_cols:
                working_df.drop(columns=del_cols, inplace=True)
                st.success(f"✅ Deleted columns: {', '.join(del_cols)}")
            else:
                st.info("No columns selected for deletion")

    # 10. Sort Data
    with st.expander("11. Sort Data"):
        sort_cols = st.multiselect("Select columns to sort by", working_df.columns.tolist())
        ascending = st.radio("Sort order", options=["Ascending", "Descending"], index=0)
        if st.button("Sort Data"):
            if sort_cols:
                working_df = working_df.sort_values(by=sort_cols, ascending=(ascending == "Ascending"))
                st.success(f"✅ Sorted data by columns: {', '.join(sort_cols)}")
            else:
                st.info("No columns selected to sort by")

    # 11. Data Validation
    with st.expander("11. Data Validation"):
        col_to_validate = st.selectbox("Select column to validate", working_df.columns.tolist(), key="validate_col")
        if col_to_validate:
            # Check column type
            is_numeric = np.issubdtype(working_df[col_to_validate].dtype, np.number)
            is_datetime = pd.api.types.is_datetime64_any_dtype(working_df[col_to_validate])
            
            validation_types = []
            if is_numeric:
                validation_types.append("Numeric Range")
            elif is_datetime:
                validation_types.append("Date Format")
            else:
                validation_types.extend(["Email", "Phone Number", "Custom Regex"])
            
            validation_type = st.selectbox(
                "Choose validation type",
                validation_types,
                key="validation_type"
            )
            
            if validation_type == "Email":
                # Convert to string first
                str_series = working_df[col_to_validate].astype(str)
                mask = str_series.str.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', na=False)
            elif validation_type == "Phone Number":
                str_series = working_df[col_to_validate].astype(str)
                mask = str_series.str.match(r'^\+?1?\d{9,15}$', na=False)
            elif validation_type == "Date Format":
                date_format = st.text_input("Enter date format (e.g., YYYY-MM-DD)", value="YYYY-MM-DD")
                try:
                    mask = pd.to_datetime(working_df[col_to_validate], format=date_format.replace('YYYY', '%Y').replace('MM', '%m').replace('DD', '%d'), errors='coerce').notna()
                except:
                    st.error("Invalid date format")
                    mask = pd.Series(False, index=working_df.index)
            elif validation_type == "Numeric Range":
                min_val = st.number_input("Minimum value", value=working_df[col_to_validate].min())
                max_val = st.number_input("Maximum value", value=working_df[col_to_validate].max())
                mask = working_df[col_to_validate].between(min_val, max_val)
            elif validation_type == "Custom Regex":  # Custom Regex
                regex_pattern = st.text_input("Enter regular expression pattern")
                try:
                    str_series = working_df[col_to_validate].astype(str)
                    mask = str_series.str.match(regex_pattern, na=False)
                except re.error:
                    st.error("Invalid regular expression")
                    mask = pd.Series(False, index=working_df.index)
            
            if 'mask' in locals():  # Only proceed if mask was created
                invalid_count = (~mask).sum()
                if invalid_count > 0:
                    st.warning(f"Found {invalid_count} invalid values in column '{col_to_validate}'")
                    show_invalid = st.checkbox("Show invalid rows")
                    if show_invalid:
                        st.dataframe(working_df[~mask], use_container_width=True)
                    
                    handle_invalid = st.radio(
                        "How to handle invalid values?",
                        ["Keep as is", "Replace with NaN", "Drop rows"],
                        key="handle_invalid"
                    )
                    
                    if st.button("Apply Validation Fix"):
                        if handle_invalid == "Replace with NaN":
                            working_df.loc[~mask, col_to_validate] = np.nan
                            st.success(f"✅ Replaced {invalid_count} invalid values with NaN")
                        elif handle_invalid == "Drop rows":
                            working_df = working_df[mask]
                            st.success(f"✅ Dropped {invalid_count} rows with invalid values")
                else:
                    st.success(f"✅ All values in column '{col_to_validate}' are valid!")

    # 12. Filter Rows (Advanced) (previous 11)
    with st.expander("12. Filter Rows (Advanced)"):
        filter_col = st.selectbox("Select column to filter rows", working_df.columns.tolist(), key="filter_col")
        if filter_col:
            dtype = working_df[filter_col].dtype
            if np.issubdtype(dtype, np.number):
                filter_op = st.selectbox("Select filter operator", ["=", ">", "<", ">=", "<=", "!="], key="filter_op_num")
                filter_val = st.number_input("Enter value to filter by", key="filter_val_num")
                if st.button("Apply Numeric Filter"):
                    expr = f"`{filter_col}` {filter_op} @filter_val"
                    working_df = working_df.query(expr)
                    st.success(f"✅ Applied filter: {filter_col} {filter_op} {filter_val}")
            else:
                filter_text = st.text_input("Enter substring to filter by", key="filter_val_str")
                case_sensitive = st.checkbox("Case sensitive", key="case_sensitive")
                use_regex = st.checkbox("Use regular expressions", key="use_regex")
                if st.button("Apply Text Filter"):
                    try:
                        if use_regex:
                            working_df = working_df[working_df[filter_col].str.contains(
                                filter_text, 
                                case=case_sensitive, 
                                regex=True, 
                                na=False
                            )]
                        else:
                            working_df = working_df[working_df[filter_col].str.contains(
                                re.escape(filter_text), 
                                case=case_sensitive, 
                                regex=True, 
                                na=False
                            )]
                        st.success(f"✅ Applied text filter on column '{filter_col}' containing '{filter_text}'")
                    except re.error:
                        st.error("Invalid regular expression pattern")

    # Data Transformation
    with st.expander("11. Data Transformation"):
        st.markdown("### Transform data using custom operations")

        # Column selection
        transform_col = st.selectbox("Select column to transform", working_df.columns, key="transform_col")
        
        if transform_col:
            operation = st.selectbox(
                "Choose transformation",
                [
                    "Math Operation",
                    "Text Operation",
                    "Binning",
                    "String Extraction",
                    "Custom Formula"
                ],
                key="transform_op"
            )

            new_col_name = st.text_input(
                "New column name (leave empty to overwrite)",
                value=f"{transform_col}_transformed",
                key="new_col_name"
            )

            if operation == "Math Operation":
                math_op = st.selectbox(
                    "Select math operation",
                    ["Add", "Subtract", "Multiply", "Divide", "Power", "Log", "Absolute"],
                    key="math_op"
                )
                
                if math_op in ["Add", "Subtract", "Multiply", "Divide", "Power"]:
                    value = st.number_input("Enter value", value=1.0, key="math_value")
                
                if st.button("Apply Math Transform"):
                    try:
                        result = working_df[transform_col].copy()
                        if math_op == "Add":
                            result = result + value
                        elif math_op == "Subtract":
                            result = result - value
                        elif math_op == "Multiply":
                            result = result * value
                        elif math_op == "Divide":
                            result = result / value
                        elif math_op == "Power":
                            result = result ** value
                        elif math_op == "Log":
                            result = np.log(result)
                        elif math_op == "Absolute":
                            result = np.abs(result)
                        
                        col_name = new_col_name if new_col_name else transform_col
                        working_df[col_name] = result
                        st.success(f"✅ Applied {math_op} transformation")
                    except Exception as e:
                        st.error(f"Error applying transformation: {str(e)}")

            elif operation == "Text Operation":
                text_op = st.selectbox(
                    "Select text operation",
                    ["Extract First N Chars", "Extract Last N Chars", "Split and Get Part", "Replace Text"],
                    key="text_op"
                )
                
                if text_op in ["Extract First N Chars", "Extract Last N Chars"]:
                    n_chars = st.number_input("Number of characters", min_value=1, value=1)
                elif text_op == "Split and Get Part":
                    split_char = st.text_input("Split character/string")
                    part_index = st.number_input("Part index (0-based)", min_value=0, value=0)
                elif text_op == "Replace Text":
                    find_text = st.text_input("Text to find")
                    replace_text = st.text_input("Replace with")
                
                if st.button("Apply Text Transform"):
                    try:
                        if text_op == "Extract First N Chars":
                            result = working_df[transform_col].str[:n_chars]
                        elif text_op == "Extract Last N Chars":
                            result = working_df[transform_col].str[-n_chars:]
                        elif text_op == "Split and Get Part":
                            result = working_df[transform_col].str.split(split_char).str[part_index]
                        elif text_op == "Replace Text":
                            result = working_df[transform_col].str.replace(find_text, replace_text)
                        
                        col_name = new_col_name if new_col_name else transform_col
                        working_df[col_name] = result
                        st.success(f"✅ Applied {text_op} transformation")
                    except Exception as e:
                        st.error(f"Error applying transformation: {str(e)}")

            elif operation == "Binning":
                n_bins = st.number_input("Number of bins", min_value=2, value=4)
                bin_labels = st.text_input("Bin labels (comma-separated, leave empty for numeric)", value="")
                
                if st.button("Apply Binning"):
                    try:
                        # Create bins using pandas cut
                        if bin_labels:
                            labels = [label.strip() for label in bin_labels.split(",")]
                            if len(labels) != n_bins:
                                st.error(f"Number of labels ({len(labels)}) must match number of bins ({n_bins})")
                            else:
                                result = pd.cut(working_df[transform_col], bins=n_bins, labels=labels)
                        else:
                            result = pd.cut(working_df[transform_col], bins=n_bins)
                        
                        col_name = new_col_name if new_col_name else transform_col
                        working_df[col_name] = result
                        st.success("✅ Applied binning transformation")
                    except Exception as e:
                        st.error(f"Error applying binning: {str(e)}")

            elif operation == "String Extraction":
                regex_pattern = st.text_input("Regular expression pattern")
                st.markdown("Example patterns:")
                st.markdown("- Email: `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}`")
                st.markdown("- Phone: `\d{3}[-.]?\d{3}[-.]?\d{4}`")
                st.markdown("- Date: `\d{4}-\d{2}-\d{2}`")
                
                if st.button("Apply Regex Extraction"):
                    try:
                        result = working_df[transform_col].str.extract(f'({regex_pattern})', expand=False)
                        col_name = new_col_name if new_col_name else transform_col
                        working_df[col_name] = result
                        st.success("✅ Applied regex extraction")
                    except Exception as e:
                        st.error(f"Error applying regex extraction: {str(e)}")

            elif operation == "Custom Formula":
                formula = st.text_input("Enter custom formula using 'x' as the column value")
                st.markdown("Examples:")
                st.markdown("- Double the value: `x * 2`")
                st.markdown("- Convert Celsius to Fahrenheit: `(x * 9/5) + 32`")
                
                if st.button("Apply Custom Formula"):
                    try:
                        # Create a safe namespace with only math functions
                        safe_dict = {
                            "x": working_df[transform_col],
                            "np": np,
                            "abs": abs,
                            "round": round
                        }
                        
                        result = eval(formula, {"__builtins__": {}}, safe_dict)
                        col_name = new_col_name if new_col_name else transform_col
                        working_df[col_name] = result
                        st.success("✅ Applied custom formula")
                    except Exception as e:
                        st.error(f"Error applying custom formula: {str(e)}")

    # 12. Format Standardization (existing)
    with st.expander("12. Format Standardization"):
        st.markdown("### Standardize data formats across columns")
        
        # Date standardization
        st.subheader("Date Standardization")
        date_cols = st.multiselect(
            "Select date columns to standardize",
            working_df.columns,
            key="date_std_cols"
        )
        if date_cols:
            target_date_format = st.selectbox(
                "Select target date format",
                ["YYYY-MM-DD", "DD-MM-YYYY", "MM/DD/YYYY", "YYYY/MM/DD", "Custom"],
                key="date_format"
            )
            
            if target_date_format == "Custom":
                target_date_format = st.text_input("Enter custom date format")
                
            if st.button("Standardize Dates"):
                for col in date_cols:
                    try:
                        # First try to parse dates with various formats
                        working_df[col] = pd.to_datetime(working_df[col], errors='coerce')
                        # Then convert to desired format
                        format_str = target_date_format.replace("YYYY", "%Y").replace("MM", "%m").replace("DD", "%d")
                        working_df[col] = working_df[col].dt.strftime(format_str)
                        st.success(f"✅ Standardized dates in column '{col}'")
                    except Exception as e:
                        st.error(f"Error standardizing dates in column '{col}': {str(e)}")

        # Number standardization
        st.subheader("Number Standardization")
        num_cols = working_df.select_dtypes(include=[np.number]).columns
        selected_num_cols = st.multiselect(
            "Select numeric columns to standardize",
            num_cols,
            key="num_std_cols"
        )
        
        if selected_num_cols:
            decimal_places = st.number_input("Decimal places", min_value=0, value=2)
            thousands_sep = st.checkbox("Add thousands separator")
            
            if st.button("Standardize Numbers"):
                for col in selected_num_cols:
                    try:
                        working_df[col] = working_df[col].round(decimal_places)
                        if thousands_sep:
                            working_df[col] = working_df[col].apply(lambda x: f"{x:,}")
                        st.success(f"✅ Standardized numbers in column '{col}'")
                    except Exception as e:
                        st.error(f"Error standardizing numbers in column '{col}': {str(e)}")

        # Text standardization
        st.subheader("Text Standardization")
        text_cols = working_df.select_dtypes(include=['object']).columns
        selected_text_cols = st.multiselect(
            "Select text columns to standardize",
            text_cols,
            key="text_std_cols"
        )
        
        if selected_text_cols:
            text_case = st.selectbox(
                "Standardize text case",
                ["No change", "UPPER", "lower", "Title Case", "Sentence case"],
                key="text_case"
            )
            
            remove_extra_spaces = st.checkbox("Remove extra spaces")
            standardize_punctuation = st.checkbox("Standardize punctuation")
            
            if st.button("Standardize Text"):
                for col in selected_text_cols:
                    try:
                        # Handle text case
                        if text_case == "UPPER":
                            working_df[col] = working_df[col].str.upper()
                        elif text_case == "lower":
                            working_df[col] = working_df[col].str.lower()
                        elif text_case == "Title Case":
                            working_df[col] = working_df[col].str.title()
                        elif text_case == "Sentence case":
                            working_df[col] = working_df[col].str.capitalize()
                        
                        # Handle spaces
                        if remove_extra_spaces:
                            working_df[col] = working_df[col].str.replace(r'\s+', ' ', regex=True)
                            working_df[col] = working_df[col].str.strip()
                        
                        # Handle punctuation
                        if standardize_punctuation:
                            # Replace multiple periods/dots with single one
                            working_df[col] = working_df[col].str.replace(r'\.+', '.', regex=True)
                            # Replace multiple exclamation marks
                            working_df[col] = working_df[col].str.replace(r'!+', '!', regex=True)
                            # Replace multiple question marks
                            working_df[col] = working_df[col].str.replace(r'\?+', '?', regex=True)
                            # Add space after punctuation if missing
                            working_df[col] = working_df[col].str.replace(r'([.,!?])([A-Za-z])', r'\1 \2', regex=True)
                        
                        st.success(f"✅ Standardized text in column '{col}'")
                    except Exception as e:
                        st.error(f"Error standardizing text in column '{col}': {str(e)}")

    # 13. Bulk Operations
    with st.expander("13. Bulk Operations"):
        st.markdown("### Apply operations to multiple columns at once")
        
        selected_cols = st.multiselect("Select columns for bulk operations", working_df.columns)
        if selected_cols:
            operation = st.selectbox(
                "Choose operation",
                [
                    "Strip whitespace",
                    "Convert to uppercase",
                    "Convert to lowercase",
                    "Remove special characters",
                    "Fill missing values",
                    "Round numbers",
                    "Format dates"
                ]
            )

            if operation == "Strip whitespace":
                if st.button("Apply Strip Whitespace"):
                    for col in selected_cols:
                        if working_df[col].dtype == 'object':
                            working_df[col] = working_df[col].astype(str).str.strip()
                    st.success(f"✅ Stripped whitespace from {len(selected_cols)} columns")

            elif operation == "Convert to uppercase":
                if st.button("Convert to Uppercase"):
                    for col in selected_cols:
                        if working_df[col].dtype == 'object':
                            working_df[col] = working_df[col].astype(str).str.upper()
                    st.success(f"✅ Converted {len(selected_cols)} columns to uppercase")

            elif operation == "Convert to lowercase":
                if st.button("Convert to Lowercase"):
                    for col in selected_cols:
                        if working_df[col].dtype == 'object':
                            working_df[col] = working_df[col].astype(str).str.lower()
                    st.success(f"✅ Converted {len(selected_cols)} columns to lowercase")

            elif operation == "Remove special characters":
                if st.button("Remove Special Characters"):
                    for col in selected_cols:
                        if working_df[col].dtype == 'object':
                            working_df[col] = working_df[col].astype(str).apply(lambda x: re.sub(r'[^A-Za-z0-9\s]', '', x))
                    st.success(f"✅ Removed special characters from {len(selected_cols)} columns")

            elif operation == "Fill missing values":
                fill_value = st.text_input("Enter value to fill missing data with")
                if st.button("Fill Missing Values") and fill_value:
                    for col in selected_cols:
                        working_df[col] = working_df[col].fillna(fill_value)
                    st.success(f"✅ Filled missing values in {len(selected_cols)} columns")

            elif operation == "Round numbers":
                decimals = st.number_input("Number of decimal places", min_value=0, value=2)
                if st.button("Round Numbers"):
                    for col in selected_cols:
                        if np.issubdtype(working_df[col].dtype, np.number):
                            working_df[col] = working_df[col].round(decimals)
                    st.success(f"✅ Rounded numeric values in selected columns to {decimals} decimal places")

            elif operation == "Format dates":
                date_format = st.text_input("Enter date format (e.g., %Y-%m-%d)")
                if st.button("Format Dates") and date_format:
                    for col in selected_cols:
                        try:
                            working_df[col] = pd.to_datetime(working_df[col]).dt.strftime(date_format)
                            st.success(f"✅ Formatted dates in column '{col}'")
                        except Exception as e:
                            st.error(f"Error formatting dates in column '{col}': {str(e)}")

    # Update original dataframe with changes for selected rows only
    if identifier_col and selected_values:
        df.loc[df[identifier_col].isin(selected_values), working_df.columns] = working_df
    else:
        df = working_df

    return df