import streamlit as st
import pandas as pd
from io import BytesIO
from difflib import SequenceMatcher
import re
from compare import read_file  # Use your existing read_file function

def standardize_datasets():
    st.header("🎯 Dataset Standardization Tool")
    
    # Master Schema
    MASTER_SCHEMA = [
        "Sr. No.", "Media_ID", "Area Name / Block", "Dataset_Type", "File_Format", 
        "Vendor_Media_ID", "Receiver_Lines", "Receiver_Point_Range", "No_Files", 
        "Sequence_No", "Sequence_Range", "Swath", "FSP", "LSP", "FFID", "LFID", 
        "Sail_Shot_No", "File_From", "File_To", "Command", "Remarks"
    ]
    
    # Enhanced fuzzy mapping rules with synonyms and abbreviations
    FUZZY_MAPPINGS = {
        "Sr. No.": ["sr no", "sr", "serial", "serial no", "serial number", "s no", "sno", "sl no", "sl", "number", "#"],
        "Media_ID": ["media id", "media", "mediaid", "id", "media_id", "mid", "m_id"],
        "Area Name / Block": ["area", "block", "area block", "survey", "area name", "block name", "location", "region", "zone"],
        "Dataset_Type": ["dataset type", "data type", "type", "dataset", "data_type", "dtype", "format type"],
        "File_Format": ["file format", "format", "extension", "file type", "filetype", "file_format", "ext"],
        "Vendor_Media_ID": ["vendor media id", "vendor id", "vendor", "vendorid", "vendor_id", "v_id", "vmi"],
        "Receiver_Lines": ["receiver lines", "receiver", "lines", "rec lines", "rcv lines", "receiver_lines", "rec_lines"],
        "Receiver_Point_Range": ["receiver point range", "point range", "receiver range", "range", "points", "rec range"],
        "No_Files": ["no files", "number of files", "file count", "files", "count", "no of files", "num files"],
        "Sequence_No": ["sequence no", "seq no", "sequence", "seq", "sequence number", "s_no", "seqno"],
        "Sequence_Range": ["sequence range", "seq range", "range", "seq_range", "sequence_range"],
        "Swath": ["swath", "swathe", "path", "track", "line"],
        "FSP": ["fsp", "first shot point", "first sp", "start point", "starting point"],
        "LSP": ["lsp", "last shot point", "last sp", "end point", "ending point"],
        "FFID": ["ffid", "first field file id", "first file id", "start file", "first file"],
        "LFID": ["lfid", "last field file id", "last file id", "end file", "last file"],
        "Sail_Shot_No": ["sail shot no", "shot no", "shot", "sail shot", "shot number", "shotno"],
        "File_From": ["file from", "from file", "start file", "from", "file_from"],
        "File_To": ["file to", "to file", "end file", "to", "file_to"],
        "Command": ["cmd", "command", "job id", "job_id", "jobid", "command/job id", "command / job id", "job", "task"],
        "Remarks": ["remarks", "comment", "comments", "note", "notes", "description", "desc", "remark"]
    }
    
    st.code(", ".join(MASTER_SCHEMA))
    
    uploaded_files = st.file_uploader(
        "Upload Excel/CSV files to standardize", 
        type=["xlsx", "xls", "csv"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        
        def normalize_text(text):
            """Advanced text normalization for better matching"""
            if pd.isna(text) or text is None:
                return ""
            
            text = str(text).lower().strip()
            
            # Replace common separators with spaces
            text = re.sub(r'[_\-/\\|]', ' ', text)
            
            # Remove special characters except spaces
            text = re.sub(r'[^\w\s]', '', text)
            
            # Handle multiple spaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Common abbreviations and expansions
            abbreviations = {
                'no': 'number',
                'pt': 'point',
                'pts': 'points',
                'seq': 'sequence',
                'rec': 'receiver',
                'rcv': 'receiver',
                'id': 'identification',
                'sp': 'shot point',
                'fid': 'file identification'
            }
            
            words = text.split()
            normalized_words = []
            for word in words:
                normalized_words.append(abbreviations.get(word, word))
            
            return ' '.join(normalized_words)
        
        def calculate_advanced_similarity(str1, str2):
            """Advanced similarity calculation with multiple techniques"""
            norm_str1 = normalize_text(str1)
            norm_str2 = normalize_text(str2)
            
            # Basic sequence matching
            basic_score = SequenceMatcher(None, norm_str1, norm_str2).ratio()
            
            # Token-based matching (individual words)
            words1 = set(norm_str1.split())
            words2 = set(norm_str2.split())
            
            if words1 and words2:
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                token_score = len(intersection) / len(union) if union else 0
            else:
                token_score = 0
            
            # Substring matching
            substring_score = 0
            if norm_str1 in norm_str2 or norm_str2 in norm_str1:
                substring_score = 0.8
            
            # Combined score with weights
            final_score = (basic_score * 0.4) + (token_score * 0.4) + (substring_score * 0.2)
            
            return final_score
        
        def find_best_column_match(target_col, available_cols):
            """Enhanced column matching with fuzzy logic"""
            target_norm = normalize_text(target_col)
            
            # First check direct fuzzy mappings
            if target_col in FUZZY_MAPPINGS:
                for col in available_cols:
                    col_norm = normalize_text(col)
                    if col_norm in [normalize_text(variant) for variant in FUZZY_MAPPINGS[target_col]]:
                        return col, 1.0, "direct_match"
            
            # Advanced fuzzy matching
            best_match = None
            best_score = 0
            match_type = "fuzzy_match"
            
            for col in available_cols:
                score = calculate_advanced_similarity(target_col, col)
                
                # Boost score if key words match
                col_words = set(normalize_text(col).split())
                target_words = set(target_norm.split())
                
                # Special boost for exact word matches
                exact_word_matches = col_words.intersection(target_words)
                if exact_word_matches:
                    word_boost = len(exact_word_matches) / max(len(col_words), len(target_words)) * 0.3
                    score += word_boost
                
                # Penalize length differences
                len_diff = abs(len(col) - len(target_col)) / max(len(col), len(target_col))
                score -= len_diff * 0.1
                
                if score > best_score and score >= 0.4:  # Lower threshold for better matching
                    best_score = score
                    best_match = col
            
            return best_match, best_score, match_type
        
        def standardize_file(df, filename):
            """Standardize one file with enhanced matching"""
            result_df = pd.DataFrame()
            mappings = {}
            scores = {}
            
            available_cols = list(df.columns)
            used_cols = []
            
            for master_col in MASTER_SCHEMA:
                match, score, match_type = find_best_column_match(master_col, 
                    [c for c in available_cols if c not in used_cols])
                
                if match:
                    result_df[master_col] = df[match]
                    mappings[master_col] = match
                    scores[master_col] = (score, match_type)
                    used_cols.append(match)
                else:
                    result_df[master_col] = None
                    mappings[master_col] = "MISSING"
                    scores[master_col] = (0, "no_match")
            
            return result_df, mappings, scores
        
        # Process files
        results = []
        all_mappings = {}
        all_scores = {}
        
        for file in uploaded_files:
            df = read_file(file)
            if df is not None:
                std_df, mappings, scores = standardize_file(df, file.name)
                results.append((file.name, std_df))
                all_mappings[file.name] = mappings
                all_scores[file.name] = scores
        
        # Show enhanced mapping summary
        st.markdown("### 📋 Column Mappings with Confidence Scores")
        for filename, mappings in all_mappings.items():
            with st.expander(f"📁 {filename}"):
                scores = all_scores[filename]
                
                for master_col, original_col in mappings.items():
                    score, match_type = scores[master_col]
                    
                    if original_col == "MISSING":
                        st.write(f"❌ **{master_col}** → Missing (added as empty)")
                    else:
                        confidence = f"{score:.2%}"
                        if score >= 0.8:
                            icon = "🟢"
                        elif score >= 0.6:
                            icon = "🟡"
                        else:
                            icon = "🟠"
                        
                        st.write(f"{icon} **{master_col}** → {original_col} (confidence: {confidence})")
                
                # Show match statistics
                matched = sum(1 for col, orig in mappings.items() if orig != "MISSING")
                total = len(mappings)
                avg_confidence = sum(scores[col][0] for col in mappings if mappings[col] != "MISSING") / max(matched, 1)
                
                st.metric("Match Rate", f"{matched}/{total} ({matched/total:.1%})")
                if matched > 0:
                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # Show standardized data
        st.markdown("### 📊 Standardized Data")
        for filename, std_df in results:
            st.subheader(f"📄 {filename}")
            st.dataframe(std_df)
        
        # Download buttons
        st.markdown("### 📥 Downloads")
        for filename, std_df in results:
            buffer = BytesIO()
            std_df.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)
            
            st.download_button(
                label=f"Download {filename}",
                data=buffer,
                file_name=f"standardized_{filename.split('.')[0]}.xlsx",
                mime="application/vnd.ms-excel"
            )
    
    else:
        st.info("Upload files to get started")

standardize_datasets()