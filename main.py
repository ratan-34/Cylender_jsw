import streamlit as st
import pandas as pd
import io
import json
from tabulate import tabulate
from scripts.extract_parameters import analyze_engineering_drawing, convert_pdf_to_image_bytes

def main():
    st.set_page_config(layout="wide", page_title="Engineering Drawing Parameter Extractor")

    st.title("⚙️ Engineering Drawing Parameter Extractor")
    st.markdown("""
        Upload your engineering diagrams (PDFs or images) to automatically extract key cylinder parameters.
        The application uses an AI model to analyze the drawings and provide structured data.
    """)

    uploaded_files = st.file_uploader(
        "Upload PDF or Image Files",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Drag and drop your engineering drawings here. Multiple files can be uploaded."
    )

    all_extracted_data = []

    if uploaded_files:
        st.subheader("Processing Files...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, uploaded_file in enumerate(uploaded_files):
            file_extension = uploaded_file.name.split('.')[-1].lower()
            image_data = None

            status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})...")

            if file_extension == "pdf":
                pdf_bytes = uploaded_file.read()
                image_data = convert_pdf_to_image_bytes(pdf_bytes)
                if not image_data:
                    st.error(f"Failed to convert PDF {uploaded_file.name} to image. Skipping.")
                    all_extracted_data.append({"filename": uploaded_file.name, "data": {"error": "PDF conversion failed"}})
                    continue
            elif file_extension in ["png", "jpg", "jpeg"]:
                image_data = uploaded_file.read()
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}. Skipping.")
                all_extracted_data.append({"filename": uploaded_file.name, "data": {"error": "Unsupported file type"}})
                continue

            if image_data:
                with st.spinner(f"Conducting detailed cylinder analysis for {uploaded_file.name}..."):
                    extracted_parameters = analyze_engineering_drawing(image_data, uploaded_file.name)
                    all_extracted_data.append({"filename": uploaded_file.name, "data": extracted_parameters})
            else:
                all_extracted_data.append({"filename": uploaded_file.name, "data": {"error": "No image data to analyze"}})
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("All files processed!")
        st.success("Extraction complete!")

        if all_extracted_data:
            st.subheader("📋 Extracted Parameter Tables")

            # Display each file's results as a markdown-style table
            flattened_data = []
            for item in all_extracted_data:
                filename = item.get("filename", "unknown file")
                data = item.get("data", {})

                st.markdown(f"### 📄 File: `{filename}`")

                if "error" in data:
                    st.error(f"❌ Error: {data['error']}")
                    continue

                # Prepare table
                table_data = [[k.replace("_", " ").title(), v] for k, v in data.items()]
                table_md = tabulate(table_data, headers=["Parameter", "Value"], tablefmt="github")
                st.code(table_md, language="markdown")

                # For CSV export
                row = {"filename": filename}
                row.update(data)
                flattened_data.append(row)

            # CSV download option
            csv_buffer = io.StringIO()
            pd.DataFrame(flattened_data).to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download All Extracted Data as CSV",
                data=csv_buffer.getvalue(),
                file_name="extracted_engineering_parameters.csv",
                mime="text/csv",
                help="Download all extracted values in spreadsheet format"
            )
        else:
            st.info("No data extracted. Please upload valid PDF or image files.")

    st.markdown("---")
    st.markdown("Developed by Vercel AI")

if __name__ == "__main__":
    main()
