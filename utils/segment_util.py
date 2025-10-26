import requests
import os
from dotenv import load_dotenv

load_dotenv()



HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/khoongwei/brain-tumor-segmentation"
API_TOKEN = os.getenv("HUGGING_FACE_API_KEY") 
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

def segment_tumor_image(image_path):
    with open(image_path, "rb") as f:
        response = requests.post(HUGGINGFACE_API_URL, headers=HEADERS, files={"file": f})

    if response.status_code == 200:
        output_path = "segmented_output.jpg"
        with open(output_path, "wb") as out:
            out.write(response.content)
        return output_path
    else:
        raise Exception(f"Segmentation failed: {response.status_code} - {response.text}")
