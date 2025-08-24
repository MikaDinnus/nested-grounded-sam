import os
import requests
from dotenv import load_dotenv
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import json
import ast

def parse_vocab_groundtruth(vocab_input):
    if isinstance(vocab_input, list):
        return [str(item).strip().lower() for item in vocab_input if str(item).strip()]
    
    if isinstance(vocab_input, str):
        vocab_input = vocab_input.strip()
        
        if ((vocab_input.startswith('"') and vocab_input.endswith('"')) or 
            (vocab_input.startswith("'") and vocab_input.endswith("'"))):
            vocab_input = vocab_input[1:-1]
        
        try:
            parsed = json.loads(vocab_input)
            if isinstance(parsed, list):
                return [str(item).strip().lower() for item in parsed if str(item).strip()]
        except:
            pass
        
        try:
            parsed = ast.literal_eval(vocab_input)
            if isinstance(parsed, list):
                return [str(item).strip().lower() for item in parsed if str(item).strip()]
        except:
            pass
        
        if ',' in vocab_input:
            vocab_input = vocab_input.strip('[](){}')
            vocab_input = vocab_input.replace('"', '').replace("'", '')
            items = [item.strip().lower() for item in vocab_input.split(',') if item.strip()]
            return items
        else:
            vocab_input = vocab_input.strip('[](){}"\'')
            if vocab_input:
                return [vocab_input.lower()]
    
    return []



######### SETUP #########

load_dotenv()

CURRENT_DATASET = f"building_facade/ADE_train_{DATASET_NUMBER}.jpg"

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
                    "text": f"Use the following json information to the dataset with the number {DATASET_NUMBER} to solve the following problem: To detect the object {VOCAB_SECONDLVL}, the hierarchically higher-level objects must be found. Hierarchically higher means a completely enclosing or containing object in order to later crop the image to this object to make it easier to find the end object. Please respond to this prompt with only your finished analysis, which only includes the word of the Vocab-FirstLevel. Be as precise and concise as possible when choosing. Here is the image content description for you to analyze :: {IMAGE_DESCRIPTION}"
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
    if 'VOCAB_GROUNDTRUTH' in locals() or 'VOCAB_GROUNDTRUTH' in globals():
        print(f"DEBUG reasoning.py: VOCAB_GROUNDTRUTH roh: {VOCAB_GROUNDTRUTH}")
        print(f"DEBUG reasoning.py: Type vor Parsing: {type(VOCAB_GROUNDTRUTH)}")
        
        VOCAB_GROUNDTRUTH = parse_vocab_groundtruth(VOCAB_GROUNDTRUTH)
        
        print(f"DEBUG reasoning.py: VOCAB_GROUNDTRUTH nach Parsing: {VOCAB_GROUNDTRUTH}")
        print(f"DEBUG reasoning.py: Type nach Parsing: {type(VOCAB_GROUNDTRUTH)}")
        print(f"DEBUG reasoning.py: Anzahl Items: {len(VOCAB_GROUNDTRUTH)}")
        
        # Validierung
        if not VOCAB_GROUNDTRUTH:
            print("WARNUNG: VOCAB_GROUNDTRUTH ist leer nach dem Parsing!")
        
        for i, item in enumerate(VOCAB_GROUNDTRUTH):
            print(f"  Item {i}: '{item}' (type: {type(item)})")
    
    vocab_groundtruth_display = str([item.upper() for item in VOCAB_GROUNDTRUTH]) if isinstance(VOCAB_GROUNDTRUTH, list) else str(VOCAB_GROUNDTRUTH).upper()
    
    print("################################" + "REASONING FOR " + str(DATASET_NUMBER) + " ENDED. VOCAB FIRST LEVEL WILL BE " + str(VOCAB_FRSTLVL.upper()) + " AND VOCAB GROUND TRUTH WILL BE " + vocab_groundtruth_display + "################################")

else:
    print("Fehler:", response_groundtruth.status_code, response_groundtruth.text)