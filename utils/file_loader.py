import fitz  # PyMuPDF
import pandas as pd
import json
import xml.etree.ElementTree as ET
import io
from io import BytesIO

def load_files(files):
    all_texts = []

    for file in files:
        filename = file.name

        if filename.endswith(".pdf"):
            file_bytes = file.read() 
            try:
                pdf_stream = BytesIO(file_bytes)
                doc = fitz.open(stream=pdf_stream, filetype="pdf")
                text = "\n".join([page.get_text() for page in doc])
            except Exception as e:
                text = f"[Error reading PDF: {str(e)}]"
            all_texts.append({
                "filename": filename,
                "content": text
            })


        elif filename.endswith(".txt"):
            content = file.read().decode()
            all_texts.append({
                "filename": filename,
                "content": content
            })

        elif filename.endswith(".json"):
            data = json.load(file)
            content = json.dumps(data)
            all_texts.append({
                "filename": filename,
                "content": content
            })

        elif filename.endswith(".xml"):
            xml_bytes = file.read()
            try:
                tree = ET.parse(io.BytesIO(xml_bytes))
                root = tree.getroot()
                content = ET.tostring(root, encoding='unicode')
            except Exception as e:
                content = f"[Failed to parse XML file: {filename}] Error: {str(e)}"
            all_texts.append({
                "filename": filename,
                "content": content
            })

        elif filename.endswith(".xlsx"):
            try:
                file.seek(0)
                df = pd.read_excel(file, engine='openpyxl')
                row_texts = []
                for _, row in df.iterrows():
                    row_text = "\n".join([f"{col.strip()}: {str(row[col]).strip()}" for col in df.columns])
                    row_texts.append(row_text)
                content = "\n\n".join(row_texts)
            except Exception as e:
                content = f"[Failed to parse Excel file: {filename}] Error: {str(e)}"
            all_texts.append({
                "filename": filename,
                "content": content
            })

    return all_texts
