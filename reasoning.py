import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

data = {
    "contents": [
        {
            "parts": [
                {
                    "text": "This is an example of a text prompt."
                }
            ]
        }
    ]
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, json=data)

if response.ok:
    result = response.json()
    output_text = result["candidates"][0]["content"]["parts"][0]["text"]
    print(output_text)
else:
    print("Fehler:", response.status_code, response.text)
