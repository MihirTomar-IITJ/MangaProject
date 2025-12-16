# test_hf_api.py
import requests
import os
import time
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# --- 1. SETUP API HEADERS AND STANDARD SDXL ENDPOINT ---
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# --- 2. DEFINE A SIMPLE TEST PROMPT ---
test_prompt = "A cute cat wearing sunglasses, manga style, monochrome"
negative_prompt = "color, photo, realistic, blurry, ugly"

# --- 3. HELPER FUNCTION (Same as before, check for 503) ---
def query_hf_api(payload):
    print("Sending test request to Hugging Face API (SDXL)...")
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 503:
        print("Model is loading, waiting 30 seconds...")
        time.sleep(30)
        return query_hf_api(payload)
    elif response.status_code != 200:
        try:
            error_details = response.json()
            print(f"API Error Details: {error_details}")
        except Exception:
            print(f"API Error Raw Response: {response.text}")
        raise Exception(f"API Error: Status Code {response.status_code}")

    # Expecting raw image bytes for this model
    if 'image' in response.headers.get('content-type', ''):
        return response.content
    else:
        raise Exception(f"Unexpected content type: {response.headers.get('content-type')}. Response: {response.text}")

# --- 4. RUN THE TEST ---
try:
    payload = {
        "inputs": test_prompt,
        "parameters": {
            "negative_prompt": negative_prompt,
            "num_inference_steps": 25
        },
         "options": {
            "wait_for_model": True
        }
    }

    image_bytes = query_hf_api(payload)

    # Save the test image
    image = Image.open(BytesIO(image_bytes))
    image.save("test_image_hf.png")
    print("--- Success! ---")
    print("Test image saved successfully as 'test_image_hf.png'.")
    print("Your API key and basic request structure are working.")

except Exception as e:
    print(f"--- Test Failed ---")
    print(f"Error during test: {e}")
    print("There might be an issue with your API key, token permissions, or network.")