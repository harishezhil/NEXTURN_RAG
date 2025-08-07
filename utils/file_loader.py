"""
HOW TO LOAD FILES?
-----------------

load_files(files):
    Reads a list of uploaded files (PDFs, TXTs, JSONs, XMLs, and Excel spreadsheets),
    and extracts their content into a unified list of dictionaries — because every file
    wants to be understood.

    Supports:
    ---------
    - .pdf: Extracts text from each page using PyMuPDF
    - .txt: Reads plain text like a diary entry
    - .json: Converts to a JSON string (we don't judge)
    - .xml: Parses using ElementTree (XML is weird, we know)
    - .xlsx: Flattens each row into readable key-value lines

    Returns:
    --------
    List[dict]: Each dict contains:
        - 'filename': Original file name
        - 'content' : Extracted or parsed text (or error message)

    Notes:
    ------
    - Handles encoding errors gracefully.
    - Error messages are preserved in content for transparency.
    - Great for feeding a RAG pipeline or an LLM that loves reading random files.
"""


import fitz  
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
                "content": text,
                "type": "pdf"
            })


        elif filename.endswith(".txt"):
            content = file.read().decode()
            all_texts.append({
                "filename": filename,
                "content": content,
                "type": "txt"
            })
        
        elif filename.endswith(".json"):
            try:
                data = json.load(file)
                if isinstance(data, list):
                    for item in data:
                        chunk = json.dumps(item, indent=2)
                        all_texts.append({
                            "filename": filename,
                            "content": chunk,
                            "type": "json"
                        })
                elif isinstance(data, dict):
                    for key, value in data.items():
                        chunk = json.dumps({key: value}, indent=2)
                        all_texts.append({
                            "filename": filename,
                            "content": chunk,
                            "type": "json"
                        })
                else:
                    content = json.dumps(data, indent=2)
                    all_texts.append({
                        "filename": filename,
                        "content": content,
                        "type": "json"
                    })
            except Exception as e:
                all_texts.append({
                    "filename": filename,
                    "content": f"[Error parsing JSON: {str(e)}]",
                    "type": "json"
                })

        elif filename.endswith(".xml"):
            try:
                xml_bytes = file.read()
                tree = ET.parse(io.BytesIO(xml_bytes))
                root = tree.getroot()
                for child in root:
                    content = ET.tostring(child, encoding='unicode')
                    all_texts.append({
                        "filename": filename,
                        "content": content,
                        "type": "xml"
                    })
            except Exception as e:
                all_texts.append({
                    "filename": filename,
                    "content": f"[Error parsing XML: {str(e)}]",
                    "type": "xml"
                })

        elif filename.endswith(".xlsx"):
            try:
                file.seek(0)
                df = pd.read_excel(file, engine='openpyxl')
                for _, row in df.iterrows():
                    row_text = "\n".join([f"{col.strip()}: {str(row[col]).strip()}" for col in df.columns])
                    all_texts.append({
                        "filename": filename,
                        "content": row_text,
                        "type": "xlsx"
                    })
            except Exception as e:
                all_texts.append({
                    "filename": filename,
                    "content": f"[Failed to parse Excel file: {filename}] Error: {str(e)}",
                    "type": "xlsx"
                })


    return all_texts
