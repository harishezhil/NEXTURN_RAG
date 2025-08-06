"""
WHAT'S THIS FILE FOR?
-------------------

chunk_sections(documents):
    Breaks documents into readable chunks based on headings, numbered lists, or section markers.
    Works with both raw strings and dictionaries (because sometimes data is messy).

    - Uses regex magic to split on newlines before headers like '#', '1. ', or 'Section 3'.
    - Strips whitespace.
    - Keeps track of where each chunk came from (filename included).

    Returns a list of tidy content chunks, each wrapped with filename info for future detective work.
"""


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

        split_sections = re.split(r'\n(?=(?:#{1,6} |\d+\.\s+|Section\s+\d+))', content)


        for section in split_sections:
            cleaned = section.strip()
            if cleaned:
                chunks.append({
                    "content": cleaned,
                    "filename": filename
                })

    return chunks



