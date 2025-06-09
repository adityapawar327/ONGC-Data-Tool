import pandas as pd
from fuzzywuzzy import fuzz
from io import BytesIO
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter
import streamlit as st
import re


class FuzzySearcher:
    def __init__(self, similarity_threshold=75):
        self.similarity_threshold = similarity_threshold


        # Define synonyms for normalization
        self.synonyms_map = {
            "input": ["input", "i/o", "io", "ip", "input/output"],
            "output": ["output", "out", "o/p", "op", "output/input"],
            "error": ["error", "err", "fault", "fail"],
            "user": ["user", "usr", "client", "operator"],
        }

        self.reverse_synonyms = {}
        for key, syns in self.synonyms_map.items():
            for syn in syns:
                self.reverse_synonyms[syn] = key

    def normalize_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        words = text.split()
        normalized_words = []
        for w in words:
            normalized_words.append(self.reverse_synonyms.get(w, w))
        return " ".join(normalized_words)

    def search_dataframe(self, df: pd.DataFrame, query: str, columns=None):
        if df.empty or not query:
            return {"matches": [], "columns": list(df.columns), "count": 0, "cells": []}

        norm_query = self.normalize_text(query)

        matches = []
        cell_refs = []

        search_columns = df.columns if columns is None else [col for col in columns if col in df.columns]

        for row_idx, row in df.iterrows():
            for col_idx, col_name in enumerate(search_columns):
                value = row[col_name]
                if pd.isnull(value):
                    continue

                try:
                    cell_text = str(value)
                    norm_cell_text = self.normalize_text(cell_text)

                    similarity = fuzz.token_set_ratio(norm_query, norm_cell_text)
                    partial_match = (norm_query in norm_cell_text) or (norm_cell_text in norm_query)

                    if similarity >= self.similarity_threshold or partial_match:
                        col_index = df.columns.get_loc(col_name)
                        matches.append((row_idx, col_index, similarity))
                        cell_refs.append(f"{get_column_letter(col_index + 1)}{row_idx + 2}")
                except Exception:
                    continue

        return {
            "matches": matches,
            "columns": list(df.columns),
            "count": len(matches),
            "cells": cell_refs,
        }


def highlight_cells(df: pd.DataFrame, matches: list):
    matched_cells = set((m[0], m[1]) for m in matches)

    def highlight_cell(val, row, col):
        if (row, col) in matched_cells:
            return "background-color: yellow"
        return ""

    styled_df = df.style.applymap(lambda v: "", subset=df.columns).apply(
        lambda x: [highlight_cell(val, x.name, idx) for idx, val in enumerate(x)],
        axis=1
    )
    return styled_df


def highlight_single_cell(df: pd.DataFrame, match):
    # match = (row_idx, col_idx, similarity)
    row_idx, col_idx, _ = match
    def highlight_cell(val, row, col):
        if (row, col) == (row_idx, col_idx):
            return "background-color: yellow"
        return ""

    styled_df = df.style.applymap(lambda v: "", subset=df.columns).apply(
        lambda x: [highlight_cell(val, x.name, idx) for idx, val in enumerate(x)],
        axis=1
    )
    return styled_df


def create_excel_with_highlights(df: pd.DataFrame, matches: list, filename="highlighted.xlsx"):
    wb = Workbook()
    ws = wb.active

    yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

    ws.append(list(df.columns))

    for i, row in df.iterrows():
        for j, val in enumerate(row):
            cell_val = "" if pd.isnull(val) else val
            ws.cell(row=i + 2, column=j + 1).value = cell_val

    for match in matches:
        row_idx, col_idx, _ = match
        cell = ws.cell(row=row_idx + 2, column=col_idx + 1)
        cell.fill = yellow_fill

    excel_buffer = BytesIO()
    wb.save(excel_buffer)
    excel_buffer.seek(0)

    return excel_buffer


def create_excel_single_cell(df: pd.DataFrame, match, filename="highlighted_single.xlsx"):
    return create_excel_with_highlights(df, [match], filename=filename)


def read_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file '{uploaded_file.name}': {e}")
        return None


def fuzzy_search_ui():
    st.header("🔍 Multi-file Fuzzy Search Excel/CSV Data")

    uploaded_files = st.file_uploader(
        "Upload one or more Excel/CSV files",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        key="multi_search"
    )
    search_term = st.text_input("Enter search term")

    if uploaded_files and search_term:
        searcher = FuzzySearcher(similarity_threshold=75)
        all_results = []
        total_matches = 0
        file_dfs = {}
        dropdown_options = []

        for uploaded_file in uploaded_files:
            df = read_file(uploaded_file)
            if df is None:
                continue

            file_dfs[uploaded_file.name] = df
            results = searcher.search_dataframe(df, search_term)
            total_matches += results["count"]

            for idx, cell in enumerate(results["cells"]):
                similarity = results["matches"][idx][2] if idx < len(results["matches"]) else None
                dropdown_options.append({
                    "label": f"{uploaded_file.name} - {cell} (Sim: {similarity})",
                    "file": uploaded_file.name,
                    "cell": cell,
                    "match": results["matches"][idx],
                    "results": results,
                })

            all_results.append({"filename": uploaded_file.name, "df": df, "results": results})

        st.write(f"### Total matches found across files: {total_matches}")

        if total_matches > 0:
            option = st.radio(
                "Highlight options:",
                ("Highlight all matching cells", "Highlight a single selected match")
            )

            if option == "Highlight all matching cells":
                for file_result in all_results:
                    st.write(f"#### File: {file_result['filename']}")
                    df = file_result["df"]
                    matches = file_result["results"]["matches"]
                    if matches:
                        styled_df = highlight_cells(df, matches)
                        st.dataframe(styled_df, use_container_width=True)

                        excel_bytes = create_excel_with_highlights(df, matches)
                        st.download_button(
                            label=f"📥 Download Highlighted Excel for {file_result['filename']}",
                            data=excel_bytes,
                            file_name=f"highlighted_{file_result['filename']}",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.write("No matches in this file.")

            else:  # Highlight single selected match
                selected_match_label = st.selectbox(
                    "Select a matched cell to highlight",
                    options=[opt["label"] for opt in dropdown_options]
                )

                selected_match = next((opt for opt in dropdown_options if opt["label"] == selected_match_label), None)
                if selected_match:
                    file_name = selected_match["file"]
                    df = file_dfs[file_name]
                    match = selected_match["match"]
                    cell_address = selected_match["cell"]

                    styled_df = highlight_single_cell(df, match)
                    st.write(f"#### Preview highlighting cell {cell_address} in file: {file_name}")
                    st.dataframe(styled_df, use_container_width=True)

                    excel_bytes = create_excel_single_cell(df, match)
                    st.download_button(
                        label=f"📥 Download Excel with single highlight ({cell_address})",
                        data=excel_bytes,
                        file_name=f"highlighted_single_{file_name}",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                    st.markdown(f"➡️ **Navigate to cell {cell_address} in Excel** by clicking that cell after opening the downloaded file.")


        else:
            st.info("No matches found in uploaded files.")


if __name__ == "__main__":
    fuzzy_search_ui()
