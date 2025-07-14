import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

IMAGE_OBJECTS = "[{\"object\":\"building\",\"coordinates\":[0,0,320,320]},{\"object\":\"car_blue\",\"coordinates\":[20,250,140,300]},{\"object\":\"car_white\",\"coordinates\":[160,260,300,310]},{\"object\":\"tree\",\"coordinates\":[50,50,90,180]},{\"object\":\"tree\",\"coordinates\":[120,40,160,170]},{\"object\":\"windows\",\"coordinates\":[30,30,100,100]},{\"object\":\"windows\",\"coordinates\":[110,30,180,100]},{\"object\":\"door\",\"coordinates\":[140,200,200,280]},{\"object\":\"canopy\",\"coordinates\":[130,190,210,220]}]"

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

print("################################" + "STARTING REASONING OPERATION OF " + str(DATASET_NUMBER) + "################################")

request_firstlevel = {
    "contents": [
        {
            "parts": [
                {
                    "text": f"Use the following json information to the dataset with the number {DATASET_NUMBER} to solve the following problem: To detect the object {VOCAB_SECONDLVL}, the hierarchically higher-level objects must be found. Hierarchically higher means a completely enclosing or containing object in order to later crop the image to this object to make it easier to find the end object. Please respond to this prompt with only your finished analysis, which only includes the word of the Vocab-FirstLevel. Here is the image content with coordinates for you to analyze :: {IMAGE_OBJECTS}"
                }
            ]
        }
    ]
}

request_groundtruth = {
    "contents": [
        {
            "parts": [
                {
                    "text": f"Use the following json information to the dataset with the number {DATASET_NUMBER} to solve the following problem: To evaluate the detectection of the object {VOCAB_SECONDLVL}, we need search terms for the ADE20K Dataset. Search terms are for the word {VOCAB_SECONDLVL} should be more than one to find all the objects in the Annotations.json and should be in the form of:: [..,..,..]. When you are ready with your analysis just return the exact format with the search terms, never return anything else."
                }
            ]
        }
    ]
}

headers = {
    "Content-Type": "application/json"
}

response_firstlevel = requests.post(url, headers=headers, json=request_firstlevel)
print("Successfully sent reasoning request to Gemini API for vocab first level.")

if response_firstlevel.ok:
    print("Successfully received reasoning response from Gemini API.")
    result = response_firstlevel.json()
    output_text = result["candidates"][0]["content"]["parts"][0]["text"]
    VOCAB_FRSTLVL = output_text.strip()
else:
    print("Fehler:", response_firstlevel.status_code, response_firstlevel.text)

response_groundtruth = requests.post(url, headers=headers, json=request_groundtruth)
print("Successfully sent reasoning request to Gemini API for vocab ground truth.")

if response_groundtruth.ok:
    print("Successfully received reasoning response from Gemini API.")
    result = response_groundtruth.json()
    output_text = result["candidates"][0]["content"]["parts"][0]["text"]
    VOCAB_GROUNDTRUTH = output_text.strip()
    print("################################" + "REASONING FOR " + str(DATASET_NUMBER) + " ENDED. VOCAB FIRST LEVEL WILL BE " + str(VOCAB_FRSTLVL.upper()) + " AND VOCAB GROUND TRUTH WILL BE " + str(VOCAB_GROUNDTRUTH.upper()) + "################################")

else:
    print("Fehler:", response_groundtruth.status_code, response_groundtruth.text)
