import streamlit as st
import pandas as pd
from io import BytesIO
import os
from pathlib import Path
import tempfile
import platform
from docx2pdf import convert as docx2pdf_convert
from weasyprint import HTML
from markdown2pdf import convert as md2pdf_convert

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
    """Convert DOCX file to PDF format using platform-independent approach."""
    try:
        from docx import Document
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_docx = Path(temp_dir) / f"{docx_file.name}"
            temp_pdf = Path(temp_dir) / f"{Path(docx_file.name).stem}.pdf"
            
            # Save uploaded file
            with open(temp_docx, "wb") as f:
                f.write(docx_file.getvalue())
            
            # Read the DOCX content
            doc = Document(temp_docx)
            
            # Create PDF
            c = canvas.Canvas(str(temp_pdf), pagesize=letter)
            
            # Write content to PDF
            y = 750  # Starting y position
            for paragraph in doc.paragraphs:
                if y < 50:  # Check if we need a new page
                    c.showPage()
                    y = 750
                c.drawString(50, y, paragraph.text)
                y -= 15
            
            c.save()
            
            # Read the generated PDF
            with open(temp_pdf, "rb") as pdf_file:
                pdf_data = pdf_file.read()
            
            # Create BytesIO object with PDF data
            output = BytesIO(pdf_data)
            output.seek(0)
            
            return output, None
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary file paths
            temp_docx = Path(temp_dir) / f"{docx_file.name}"
            temp_pdf = Path(temp_dir) / f"{Path(docx_file.name).stem}.pdf"
            
            # Save the uploaded file temporarily
            with open(temp_docx, "wb") as f:
                f.write(docx_file.getvalue())
            
            try:
                # First attempt: Try using docx2pdf
                docx2pdf.convert(str(temp_docx), str(temp_pdf))
                
                # Read the PDF file
                with open(temp_pdf, "rb") as f:
                    pdf_data = f.read()
                    
            except Exception as word_error:
                # If docx2pdf fails, try using alternative method with win32com
                try:
                    import win32com.client
                    word = win32com.client.Dispatch('Word.Application')
                    doc = word.Documents.Open(str(temp_docx))
                    doc.SaveAs(str(temp_pdf), FileFormat=17)  # 17 represents PDF                    doc.Close()
                    word.Quit()
                    
                    # Read the PDF file
                    with open(temp_pdf, "rb") as f:
                        pdf_data = f.read()
                        
                except Exception as com_error:
                    # If both methods fail, inform the user
                    return None, f"Conversion failed: Both conversion methods failed.\nMethod 1: {str(word_error)}\nMethod 2: {str(com_error)}"
                finally:
                    # Cleanup COM
                    try:
                        pythoncom.CoUninitialize()
                    except:
                        pass
            
            # Create BytesIO object with PDF data
            output = BytesIO(pdf_data)
            output.seek(0)
            
            return output, None
            
    except Exception as e:
        return None, f"Conversion failed: {str(e)}"

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