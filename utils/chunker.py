import re

def chunk_sections(documents):
    chunks = []

    for doc in documents:
        # Handle both dicts and raw strings
        if isinstance(doc, dict):
            content = doc.get("content", "")
            filename = doc.get("filename", "Unknown")
        else:
            content = doc
            filename = "Unknown"

        # Split content into sections using double newlines or other markers
        # split_sections = re.split(r'\n\s*\n+', content)

        split_sections = re.split(r'\n(?=(?:#{1,6} |\d+\.\s+|Section\s+\d+))', content)


        for section in split_sections:
            cleaned = section.strip()
            if cleaned:
                chunks.append({
                    "content": cleaned,
                    "filename": filename
                })

    return chunks



