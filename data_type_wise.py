import streamlit as st
import pandas as pd
from typing import List, Dict, Tuple, Optional
from fuzzywuzzy import fuzz, process
import re


def find_fuzzy_column(df: pd.DataFrame, target_columns: List[str], threshold: int = 80) -> Optional[str]:
    available_columns = list(df.columns)
    
    for target in target_columns:
        # First try exact match (case insensitive)
        for col in available_columns:
            if col.lower() == target.lower():
                return col
        
        # Then try fuzzy match
        match = process.extractOne(target, available_columns, scorer=fuzz.partial_ratio)
        if match and match[1] >= threshold:
            return match[0]
    
    return None


def validate_file_schema(df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    """
    Validate if the uploaded file contains required columns using fuzzy matching.
    
    Returns:
        - bool: Whether validation passed
        - List[str]: Available year columns
        - List[str]: Error messages
    """
    errors = []
    year_columns = []
    
    # Check for required Area column with fuzzy matching
    area_variations = ['Area', 'Survey Area', 'Survey_Area', 'area', 'AREA', 'Block', 'Field', 'Location']
    area_column = find_fuzzy_column(df, area_variations, threshold=70)
    
    if not area_column:
        errors.append(f"Missing required area column. Expected variations: {', '.join(area_variations)}")
    
    # Check for year-related columns with fuzzy matching
    year_column_variations = {
        'Acquisition Year': ['Acquisition Year', 'Acq Year', 'Acquisition_Year', 'AcquisitionYear', 'Survey Year'],
        'Processing Year': ['Processing Year', 'Proc Year', 'Processing_Year', 'ProcessingYear', 'Process Year'],
        'Interpretation Year': ['Interpretation Year', 'Interp Year', 'Interpretation_Year', 'InterpretationYear', 'Analysis Year']
    }
    
    for standard_name, variations in year_column_variations.items():
        found_column = find_fuzzy_column(df, variations, threshold=75)
        if found_column:
            year_columns.append(found_column)
    
    if not year_columns:
        all_year_variations = []
        for variations in year_column_variations.values():
            all_year_variations.extend(variations)
        errors.append(f"No year-related columns found. Expected variations: {', '.join(all_year_variations)}")
    
    # Check for Media_ID column with fuzzy matching
    media_id_variations = ['Media_ID', 'MediaID', 'Media ID', 'Tape ID', 'TapeID', 'Tape_ID', 'ID', 'Serial', 'Number']
    media_id_column = find_fuzzy_column(df, media_id_variations, threshold=70)
    
    if not media_id_column:
        errors.append(f"Missing required media ID column. Expected variations: {', '.join(media_id_variations)}")
    
    validation_passed = len(errors) == 0
    return validation_passed, year_columns, errors


def categorize_area(area_value: str) -> str:
    """
    Categorize the Area value into one of three categories using fuzzy logic.
    """
    if pd.isna(area_value):
        return "Unknown"
    
    area_lower = str(area_value).lower()
    
    # Define keyword categories with fuzzy matching
    acquisition_keywords = [
        'acquisition', 'acquire', 'seismic', 'survey', 'field', 'recording', 
        'shooting', 'data collection', 'exploration', 'geophysical'
    ]
    
    processing_keywords = [
        'processing', 'process', 'stack', 'migration', 'filtering', 'enhancement',
        'velocity', 'imaging', 'prestack', 'poststack', 'demultiple', 'deconvolution'
    ]
    
    interpretation_keywords = [
        'interpretation', 'interpret', 'analysis', 'attribute', 'structural',
        'stratigraphic', 'reservoir', 'geology', 'horizon', 'fault', 'mapping'
    ]
    
    # Calculate fuzzy match scores for each category
    acquisition_scores = [fuzz.partial_ratio(keyword, area_lower) for keyword in acquisition_keywords]
    processing_scores = [fuzz.partial_ratio(keyword, area_lower) for keyword in processing_keywords]
    interpretation_scores = [fuzz.partial_ratio(keyword, area_lower) for keyword in interpretation_keywords]
    
    max_acquisition = max(acquisition_scores) if acquisition_scores else 0
    max_processing = max(processing_scores) if processing_scores else 0
    max_interpretation = max(interpretation_scores) if interpretation_scores else 0
    
    # Set threshold for fuzzy matching
    threshold = 60
    
    # Determine category based on highest score above threshold
    if max_acquisition >= threshold and max_acquisition >= max_processing and max_acquisition >= max_interpretation:
        return "Acquisition Data"
    elif max_processing >= threshold and max_processing >= max_interpretation:
        return "Processing Data"
    elif max_interpretation >= threshold:
        return "Interpretation Data"
    else:
        # Fallback logic for unclear cases
        if any(word in area_lower for word in ['data', 'broadband', '3d', '2d']):
            return "Processing Data"
        elif any(word in area_lower for word in ['area', 'block', 'field', 'basin']):
            return "Acquisition Data"
        else:
            return "Acquisition Data"  # Default fallback


def extract_media_info(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], int]:
    """
    Extract Media_ID information: start tape, end tape, and media count using fuzzy column matching.
    Also checks for DVD data.
    """
    # Find Media_ID column using fuzzy matching
    media_id_variations = ['Media_ID', 'MediaID', 'Media ID', 'Tape ID', 'TapeID', 'Tape_ID', 'ID', 'Serial', 'Number']
    media_id_column = find_fuzzy_column(df, media_id_variations, threshold=70)
    
    if not media_id_column:
        return None, None, 0
    
    # Check for DVD data
    media_ids_str = df[media_id_column].astype(str).str.lower()
    has_dvd = media_ids_str.str.contains('dvd', na=False).any()
    if has_dvd:
        st.warning("⚠️ DVD data detected in the file. DVDs will be excluded from processing.")
        # Filter out DVD entries
        df = df[~media_ids_str.str.contains('dvd', na=False)]
    
    # Filter out empty/null values
    media_ids = df[media_id_column].dropna()
    media_ids = media_ids[media_ids != '']
    
    if len(media_ids) == 0:
        return None, None, 0
    
    # Convert to string and sort to get proper start/end
    media_ids = media_ids.astype(str).sort_values()
    
    start_tape = media_ids.iloc[0]
    end_tape = media_ids.iloc[-1]
    media_count = len(media_ids)
    
    return start_tape, end_tape, media_count




def determine_data_type(filename: str) -> List[str]:
    """
    Determine data type based on file name using fuzzy matching.
    Returns a list of data types that the file likely belongs to.
    """
    filename_lower = filename.lower()
    
    # Use fuzz.partial_ratio for more flexible matching
    processing_score = max(
        fuzz.partial_ratio("processing", filename_lower),
        fuzz.partial_ratio("processed", filename_lower),
        fuzz.partial_ratio("proc", filename_lower)
    )
    
    raw_score = max(
        fuzz.partial_ratio("raw", filename_lower),
        fuzz.partial_ratio("acquisition", filename_lower),
        fuzz.partial_ratio("acquired", filename_lower)
    )
    
    # Default threshold for fuzzy matching
    threshold = 60
    
    if processing_score >= threshold:
        return ["Processing Data"]
    elif raw_score >= threshold:
        return ["Acquisition Data"]
    else:
        return ["Interpretation Data"]

def clean_area_name(filename: str) -> str:
    """Clean up filename to create a meaningful area name by removing common processing keywords."""
    # List of keywords to remove (case insensitive)
    keywords_to_remove = [
        'raw data', 'raw', 'processing', 'processed', 'data', 
        'interpretation', 'interpreted', 'final', 'complete',
        'seismic', 'merged', 'updated', 'revision'
    ]
    
    # Remove file extension
    area_name = filename.rsplit('.', 1)[0]
    
    # Convert to lowercase for comparison
    name_lower = area_name.lower()
    
    # Remove each keyword if it exists as a whole word
    for keyword in keywords_to_remove:
        # Use fuzzy matching to find and remove variations of keywords
        if fuzz.partial_ratio(keyword, name_lower) > 80:
            # Create a pattern that matches the keyword with word boundaries
            pattern = re.compile(f'\\b{re.escape(keyword)}\\b', re.IGNORECASE)
            area_name = pattern.sub('', area_name)
    
    # Clean up any remaining underscores, multiple spaces, or dashes
    area_name = re.sub(r'[_-]+', ' ', area_name)
    area_name = re.sub(r'\s+', ' ', area_name)
    
    # Remove any leading/trailing spaces or special characters
    area_name = area_name.strip(' -_')
    
    return area_name

def process_file(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Process a single file based on user-selected data types.
    """      # Show preview of the uploaded data with option to view full dataset
    with st.expander(f"Preview of {filename}"):
        show_all = st.checkbox("Show all rows", key=f"show_all_{filename}")
        if show_all:
            st.dataframe(df, use_container_width=True)
        else:
            st.dataframe(df.head(), use_container_width=True)
            st.info(f"Showing first 5 rows out of {len(df)} total rows")
        st.text(f"Columns: {', '.join(df.columns)}")
    
    area_variations = ['Area', 'Survey Area', 'Survey_Area', 'area', 'AREA', 'Block', 'Field', 'Location']
    area_column = find_fuzzy_column(df, area_variations, threshold=70)
    
    # If no area column found, create a single area from cleaned filename
    if not area_column:
        area_name = clean_area_name(filename)
        st.warning(f"No area column found in {filename}. Using cleaned filename as area name: {area_name}")
        df['_area'] = area_name
        area_column = '_area'

    # Check for Media_ID column
    media_id_variations = ['Media_ID', 'MediaID', 'Media ID', 'Tape ID', 'TapeID', 'Tape_ID', 'ID', 'Serial', 'Number']
    media_id_column = find_fuzzy_column(df, media_id_variations, threshold=70)
    
    if not media_id_column:
        st.error(f"❌ No Media ID column found in {filename}. This file will be marked but not included in the final table.")
        st.warning("Expected Media ID column variations: " + ", ".join(media_id_variations))
        # Return empty DataFrame to indicate this file should not be processed
        return pd.DataFrame()

    # Determine initial data type from filename
    suggested_types = determine_data_type(filename)
      # Create checkboxes for data type selection with unique keys
    st.write("Please verify or modify the data type for this file:")
    checkbox_key_prefix = f"checkbox_{filename.replace(' ', '_').replace('.', '_')}"
    acquisition_selected = st.checkbox("Acquisition Data", 
                                     value="Acquisition Data" in suggested_types,
                                     key=f"{checkbox_key_prefix}_acquisition")
    processing_selected = st.checkbox("Processing Data", 
                                    value="Processing Data" in suggested_types,
                                    key=f"{checkbox_key_prefix}_processing")
    interpretation_selected = st.checkbox("Interpretation Data", 
                                        value="Interpretation Data" in suggested_types,
                                        key=f"{checkbox_key_prefix}_interpretation")

    unique_areas = df[area_column].dropna().unique()
    processed_rows = []
    
    for i, area in enumerate(unique_areas, 1):
        area_df = df[df[area_column] == area]
        start_tape, end_tape, media_count = extract_media_info(area_df)
        
        # Initialize variables
        acquisition_tapes = processing_tapes = interpretation_tapes = 0
        acquisition_start = acquisition_end = processing_start = processing_end = interpretation_start = interpretation_end = ''
          # Assign data based on selected data types
        if acquisition_selected:
            acquisition_tapes = media_count
            acquisition_start = start_tape
            acquisition_end = end_tape
        
        if processing_selected:
            processing_tapes = media_count
            processing_start = start_tape
            processing_end = end_tape
        
        if interpretation_selected:
            interpretation_tapes = media_count
            interpretation_start = start_tape
            interpretation_end = end_tape
            
        # If no type is selected, use the suggested type from filename
        if not any([acquisition_selected, processing_selected, interpretation_selected]):
            suggested_types = determine_data_type(filename)
            if "Acquisition Data" in suggested_types:
                acquisition_tapes = media_count
                acquisition_start = start_tape
                acquisition_end = end_tape
            elif "Processing Data" in suggested_types:
                processing_tapes = media_count
                processing_start = start_tape
                processing_end = end_tape
            else:
                interpretation_tapes = media_count
                interpretation_start = start_tape
                interpretation_end = end_tape        # Calculate the true total of all tapes for this area
        total_tapes = (
            (acquisition_tapes if acquisition_tapes else 0) + 
            (processing_tapes if processing_tapes else 0) + 
            (interpretation_tapes if interpretation_tapes else 0)
        )
        
        # If no category was assigned, use the total media count
        if total_tapes == 0:
            total_tapes = media_count
        
        row = {
            'Sr. No.': i,
            'Area': area,
            'Acquisition Data: No. of Tapes': acquisition_tapes if acquisition_tapes > 0 else '',
            'Acquisition Data: Start Tape': acquisition_start if acquisition_tapes > 0 else '',
            'Acquisition Data: End Tape': acquisition_end if acquisition_tapes > 0 else '',
            'Acquisition Data: SAPD Entry': '',
            'Processing Data: No. of Tapes': processing_tapes if processing_tapes > 0 else '',
            'Processing Data: Start Tape': processing_start if processing_tapes > 0 else '',
            'Processing Data: End Tape': processing_end if processing_tapes > 0 else '',
            'Processing Data: SAPD Entry': '',
            'Interpretation Data: No. of Tapes': interpretation_tapes if interpretation_tapes > 0 else '',
            'Interpretation Data: Start Tape': interpretation_start if interpretation_tapes > 0 else '',
            'Interpretation Data: End Tape': interpretation_end if interpretation_tapes > 0 else '',
            'Interpretation Data: SAPD Entry': '',
            'Total Tapes': total_tapes if total_tapes > 0 else ''
        }
        processed_rows.append(row)
    
    # Create and format output DataFrame
    output_df = pd.DataFrame(processed_rows)
    
    # Ensure all required columns are present and in correct order
    required_columns = [
        'Sr. No.', 'Area',
        'Acquisition Data: No. of Tapes', 'Acquisition Data: Start Tape', 'Acquisition Data: End Tape', 'Acquisition Data: SAPD Entry',
        'Processing Data: No. of Tapes', 'Processing Data: Start Tape', 'Processing Data: End Tape', 'Processing Data: SAPD Entry',
        'Interpretation Data: No. of Tapes', 'Interpretation Data: Start Tape', 'Interpretation Data: End Tape', 'Interpretation Data: SAPD Entry',
        'Total Tapes'
    ]
    
    for col in required_columns:
        if col not in output_df.columns:
            output_df[col] = ''
    
    output_df = output_df[required_columns]
    
    return output_df


def convert_df_to_csv(df: pd.DataFrame) -> str:
    """Convert DataFrame to CSV string."""
    return df.to_csv(index=False)


def data_type_wise_app():
    """
    Main function for the Data Type Wise processing functionality.
    """
    st.header("📊 Data Type Wise Processing")
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Choose Excel or CSV files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload one or more Excel or CSV files containing 'Area' and 'Media_ID' columns.",
        key="data_type_wise_uploader"
    )
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} file(s)")
        
        # Initialize list to store all processed DataFrames
        all_processed_dfs = []
        files_without_media_id = []
        
        # Process each uploaded file
        for i, uploaded_file in enumerate(uploaded_files):
            st.subheader(f"Processing: {uploaded_file.name}")
            
            try:
                # Read the file using the existing read_file function
                from compare import read_file
                df = read_file(uploaded_file)
                  
                if df is not None:
                    # Process the file
                    processed_df = process_file(df, uploaded_file.name)
                    
                    if not processed_df.empty:
                        st.success(f"Successfully processed {uploaded_file.name}")
                        all_processed_dfs.append(processed_df)
                    else:
                        st.warning(f"File {uploaded_file.name} was skipped due to missing Media ID column.")
                        files_without_media_id.append(uploaded_file.name)
                else:
                    st.error(f"Failed to read {uploaded_file.name}. Please check the file format.")
            
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        # Show summary of files without Media ID
        if files_without_media_id:
            st.subheader("⚠️ Files Skipped (No Media ID Column)")
            for filename in files_without_media_id:
                st.write(f"• {filename}")
        
        # Merge all processed DataFrames if any were successfully processed
        if all_processed_dfs:
            st.subheader("📋 Merged Output from All Files")
            
            # Concatenate all processed DataFrames
            merged_df = pd.concat(all_processed_dfs, ignore_index=True)
            
            # Reset Sr. No. to be sequential across all files
            merged_df['Sr. No.'] = range(1, len(merged_df) + 1)
            
            # Create an editable dataframe
            st.write("📝 Edit the data below if needed (double-click cells to edit):")
            edited_df = st.data_editor(
                merged_df,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "Sr. No.": st.column_config.NumberColumn(
                        "Sr. No.",
                        help="Serial Number",
                        default=len(merged_df) + 1,
                    ),
                    "Area": st.column_config.TextColumn(
                        "Area",
                        help="Area/Location name",
                        validate="^[A-Za-z0-9\\s\\-_]+$",
                    ),
                    "Acquisition Data: SAPD Entry": st.column_config.CheckboxColumn(
                        "Acquisition SAPD",
                        help="Check if SAPD entry is done",
                        default=False
                    ),
                    "Processing Data: SAPD Entry": st.column_config.CheckboxColumn(
                        "Processing SAPD",
                        help="Check if SAPD entry is done",
                        default=False
                    ),
                    "Interpretation Data: SAPD Entry": st.column_config.CheckboxColumn(
                        "Interpretation SAPD",
                        help="Check if SAPD entry is done",
                        default=False
                    ),
                    "Total Tapes": st.column_config.NumberColumn(
                        "Total Tapes",
                        help="Total number of tapes for this area",
                        format="%d",
                        required=True,
                    ),
                }
            )
            
            # Calculate dynamic totals from the edited dataframe (excluding the TOTAL row if it exists)
            def safe_convert_to_int(series):
                # First replace empty strings with NaN
                temp = pd.to_numeric(series.replace('', pd.NA), errors='coerce')
                # Then fill NaN with 0 and convert to int
                return temp.fillna(0).astype(int)
            
            # Filter out the TOTAL row if it exists
            data_rows = edited_df[edited_df['Area'] != 'TOTAL'].copy()
            
            acq_total = safe_convert_to_int(data_rows['Acquisition Data: No. of Tapes']).sum()
            proc_total = safe_convert_to_int(data_rows['Processing Data: No. of Tapes']).sum()
            interp_total = safe_convert_to_int(data_rows['Interpretation Data: No. of Tapes']).sum()
            total_datasets = acq_total + proc_total + interp_total
            
            # Display dynamic totals
            st.subheader("📊 Dynamic Summary Totals")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Acquisition Tapes", acq_total)
            with col2:
                st.metric("Processing Tapes", proc_total)
            with col3:
                st.metric("Interpretation Tapes", interp_total)
            with col4:
                st.metric("Total Tapes", total_datasets)
            
            # Add or update summary row
            summary_row = pd.DataFrame([{
                'Sr. No.': '',
                'Area': 'TOTAL',
                'Acquisition Data: No. of Tapes': acq_total if acq_total > 0 else '',
                'Acquisition Data: Start Tape': '',
                'Acquisition Data: End Tape': '',
                'Acquisition Data: SAPD Entry': '',
                'Processing Data: No. of Tapes': proc_total if proc_total > 0 else '',
                'Processing Data: Start Tape': '',
                'Processing Data: End Tape': '',
                'Processing Data: SAPD Entry': '',
                'Interpretation Data: No. of Tapes': interp_total if interp_total > 0 else '',
                'Interpretation Data: Start Tape': '',
                'Interpretation Data: End Tape': '',
                'Interpretation Data: SAPD Entry': '',
                'Total Tapes': total_datasets
            }])
            
            # Combine data rows with summary row for display and download
            final_df = pd.concat([data_rows, summary_row], ignore_index=True)
            
            # Show the final table with totals included
            st.subheader("📋 Final Output with Totals")
            st.dataframe(final_df, use_container_width=True)
            
            # Download button for final data
            csv_data = convert_df_to_csv(final_df)
            st.download_button(
                label="📥 Download Final Data",
                data=csv_data,
                file_name="final_data_type_wise_processed.csv",
                mime="text/csv",
                key="download_final_data"
            )
        else:
            st.warning("No files were successfully processed. All files may be missing required Media ID columns.")
    else:
        st.info("Please upload one or more files to begin processing.")