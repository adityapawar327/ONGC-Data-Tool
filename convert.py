import streamlit as st
import pandas as pd
from io import BytesIO
import docx2pdf
import os
from pathlib import Path

def convert_csv_to_excel(csv_file):
    """Convert CSV file to Excel format."""
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Create Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        
        output.seek(0)
        return output, None
    except Exception as e:
        return None, str(e)

def convert_excel_to_csv(excel_file):
    """Convert Excel file to CSV format."""
    try:
        # Read Excel file
        df = pd.read_excel(excel_file)
        
        # Create CSV file in memory
        output = BytesIO()
        df.to_csv(output, index=False)
        
        output.seek(0)
        return output, None
    except Exception as e:
        return None, str(e)

def convert_docx_to_pdf(docx_file):
    """Convert DOCX file to PDF format."""
    try:
        # Create a temporary directory if it doesn't exist
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Save the uploaded file temporarily
        temp_docx = temp_dir / "temp.docx"
        temp_pdf = temp_dir / "temp.pdf"
        
        with open(temp_docx, "wb") as f:
            f.write(docx_file.getvalue())
        
        # Convert DOCX to PDF
        docx2pdf.convert(str(temp_docx), str(temp_pdf))
        
        # Read the PDF file
        with open(temp_pdf, "rb") as f:
            pdf_data = f.read()
        
        # Clean up temporary files
        temp_docx.unlink()
        temp_pdf.unlink()
        
        # Create BytesIO object with PDF data
        output = BytesIO(pdf_data)
        output.seek(0)
        
        return output, None
    except Exception as e:
        return None, str(e)

def convert_app():
    """Main conversion application interface."""
    st.title("🔄 File Format Converter")
    st.markdown("""
    Convert your files between different formats:
    - CSV ↔ Excel
    - DOCX → PDF
    """)
    
    # Create tabs for different conversion types
    tab1, tab2, tab3 = st.tabs(["CSV to Excel", "Excel to CSV", "DOCX to PDF"])
    
    with tab1:
        st.subheader("CSV to Excel Converter")
        csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="csv_upload")
        
        if csv_file:
            if st.button("Convert to Excel", key="csv_to_excel"):
                with st.spinner("Converting..."):
                    output, error = convert_csv_to_excel(csv_file)
                    
                    if error:
                        st.error(f"Conversion failed: {error}")
                    else:
                        st.success("Conversion successful!")
                        st.download_button(
                            label="📥 Download Excel file",
                            data=output,
                            file_name=f"{Path(csv_file.name).stem}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
    
    with tab2:
        st.subheader("Excel to CSV Converter")
        excel_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"], key="excel_upload")
        
        if excel_file:
            if st.button("Convert to CSV", key="excel_to_csv"):
                with st.spinner("Converting..."):
                    output, error = convert_excel_to_csv(excel_file)
                    
                    if error:
                        st.error(f"Conversion failed: {error}")
                    else:
                        st.success("Conversion successful!")
                        st.download_button(
                            label="📥 Download CSV file",
                            data=output,
                            file_name=f"{Path(excel_file.name).stem}.csv",
                            mime="text/csv"
                        )
    
    with tab3:
        st.subheader("DOCX to PDF Converter")
        docx_file = st.file_uploader("Upload DOCX file", type=["docx"], key="docx_upload")
        
        if docx_file:
            if st.button("Convert to PDF", key="docx_to_pdf"):
                with st.spinner("Converting..."):
                    output, error = convert_docx_to_pdf(docx_file)
                    
                    if error:
                        st.error(f"Conversion failed: {error}")
                    else:
                        st.success("Conversion successful!")
                        st.download_button(
                            label="📥 Download PDF file",
                            data=output,
                            file_name=f"{Path(docx_file.name).stem}.pdf",
                            mime="application/pdf"
                        )

if __name__ == "__main__":
    convert_app() 