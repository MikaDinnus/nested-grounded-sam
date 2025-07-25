import os
import requests
from dotenv import load_dotenv
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import json


######### SETUP #########

load_dotenv()

CURRENT_DATASET = f"building_facade/ADE_train_0000{DATASET_NUMBER}.jpg"

print("################################" + "STARTING REASONING OPERATION OF " + str(DATASET_NUMBER) + "################################")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CVAPI_API_KEY = os.getenv("CVAPI_API_KEY")

cvapi_url = "https://bsc-midi-cvapi.cognitiveservices.azure.com/"
gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"


######### AZURE COMPUTER VISION API #########

client = ComputerVisionClient(cvapi_url, CognitiveServicesCredentials(CVAPI_API_KEY))

features = ["description"]

print("Successfully send reasoning request to Azure Computer Vision API.")

with open(CURRENT_DATASET, "rb") as image_stream:
    result_cv = client.analyze_image_in_stream(image_stream, visual_features=features)

result_cv_dict = result_cv.as_dict()

print("Successfully received reasoning response from Azure Computer Vision API.")
IMAGE_DESCRIPTION = result_cv_dict



######### REASONING GEMINI LLM API #########

request_firstlevel = {
    "contents": [
        {
            "parts": [
                {
                    "text": f"Use the following json information to the dataset with the number {DATASET_NUMBER} to solve the following problem: To detect the object {VOCAB_SECONDLVL}, the hierarchically higher-level objects must be found. Hierarchically higher means a completely enclosing or containing object in order to later crop the image to this object to make it easier to find the end object. Please respond to this prompt with only your finished analysis, which only includes the word of the Vocab-FirstLevel. Here is the image content description for you to analyze :: {IMAGE_DESCRIPTION}"
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
                    "text": f"Use the following json information to the dataset with the number {DATASET_NUMBER} to solve the following problem: To evaluate the detection of the object {VOCAB_SECONDLVL}, we need search terms for the ADE20K Dataset. Search terms are for the word {VOCAB_SECONDLVL} should be more than one to find all the objects in the Annotations.json and should be in the form of:: [..,..,..]. When you are ready with your analysis just return the exact format with the search terms, never return anything else."
                }
            ]
        }
    ]
}

headers = {
    "Content-Type": "application/json"
}

response_firstlevel = requests.post(gemini_url, headers=headers, json=request_firstlevel)
print("Successfully sent reasoning request to Gemini API for vocab first level.")

if response_firstlevel.ok:
    print("Successfully received reasoning response from Gemini API.")
    result = response_firstlevel.json()
    output_text = result["candidates"][0]["content"]["parts"][0]["text"]
    VOCAB_FRSTLVL = output_text.strip()
else:
    print("Fehler:", response_firstlevel.status_code, response_firstlevel.text)

response_groundtruth = requests.post(gemini_url, headers=headers, json=request_groundtruth)
print("Successfully sent reasoning request to Gemini API for vocab ground truth.")

if response_groundtruth.ok:
    print("Successfully received reasoning response from Gemini API.")
    result = response_groundtruth.json()
    output_text = result["candidates"][0]["content"]["parts"][0]["text"]
    VOCAB_GROUNDTRUTH = output_text.strip()
    print("################################" + "REASONING FOR " + str(DATASET_NUMBER) + " ENDED. VOCAB FIRST LEVEL WILL BE " + str(VOCAB_FRSTLVL.upper()) + " AND VOCAB GROUND TRUTH WILL BE " + str(VOCAB_GROUNDTRUTH.upper()) + "################################")

else:
    print("Fehler:", response_groundtruth.status_code, response_groundtruth.text)
