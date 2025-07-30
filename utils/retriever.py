import os
import requests
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "gemma2-9b-it"  

def generate_response(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload
    )

    # DEBUG logging
    try:
        data = response.json()
        if "choices" not in data:
            print("Unexpected API Response:\n", data)
            return f"Error: Unexpected response from Groq API: {data.get('error', {}).get('message', 'No choices returned.')}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(" Exception while parsing Groq response:")
        print("Status Code:", response.status_code)
        print("Response Text:", response.text)
        return " Error: Could not parse response from Groq API."
