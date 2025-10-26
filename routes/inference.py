from utils.image_processing import process_image
from flask import Blueprint, request, jsonify
import os
import onnxruntime as ort
import numpy as np
import uuid
import logging
from werkzeug.utils import secure_filename
import google.generativeai as genai
from PIL import Image
import json
import re

def clean_ai_json_response(response: str):
    try:
        # Remove ```json, ```, and any wrapping quotes or escape characters
        cleaned = re.sub(r'```json|```', '', response).strip()
        
        # Unescape any escape sequences like \n, \", etc.
        cleaned = bytes(cleaned, "utf-8").decode("unicode_escape")
        
        # Load and return as JSON
        return json.loads(cleaned)
    except Exception as e:
        print("Error:", e)
        return None

# Configure Gemini
genai.configure(api_key="AIzaSyCRlze7lEnO4tH6Yj-jRZdQZAroC96BZmQ")

# Flask Blueprint
interface_bp = Blueprint('interface', __name__)

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = os.path.join("models", "brain_tumor_classifier.onnx")
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Load ONNX model
try:
    ort_session = ort.InferenceSession(MODEL_PATH)
    input_details = ort_session.get_inputs()[0]
    logger.info(f"Loaded ONNX model. Input shape: {input_details.shape}")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise RuntimeError("Model initialization failed")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@interface_bp.route("/predict", methods=["POST"])
def predict_image():
    try:
        logger.info(f"Incoming request headers: {request.headers}")
        logger.info(f"Request files: {request.files}")

        if 'file' not in request.files:
            logger.error("No 'file' key in request.files")
            return jsonify({"success": False, "message": "No file part in request"}), 400

        file = request.files['file']

        if file.filename == '':
            logger.error("Empty filename submitted")
            return jsonify({"success": False, "message": "No selected file"}), 400

        if not allowed_file(file.filename):
            logger.error(f"Invalid file extension: {file.filename}")
            return jsonify({"success": False, "message": "Invalid file type"}), 400

        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        try:
            file.save(filepath)
            logger.info(f"File saved temporarily to {filepath}")

            # Preprocess image
            img = process_image(filepath)
            logger.info(f"Processed image shape: {img.shape}")

            if list(img.shape[1:]) != list(input_details.shape[1:]):
                logger.error(f"Shape mismatch. Got {img.shape[1:]}, needs {input_details.shape[1:]}")
                return jsonify({"success": False, "message": "Image processing error"}), 400

            if img.dtype != np.float32:
                img = img.astype(np.float32)

            # Run ONNX prediction
            prediction = ort_session.run(None, {input_details.name: img})[0]
            class_index = np.argmax(prediction[0])
            confidence = float(prediction[0][class_index])
            label = ['glioma', 'meningioma', 'notumor', 'pituitary'][class_index]

            # Generate message with Gemini
            try:
                gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                gemini_response = gemini_model.generate_content([
                    {
                   "text": (
                        f"The MRI scan is classified as '{label}' with a confidence of {confidence:.2f}. "
                        "Based on this label, return a JSON object in the following format:\n\n"
                        "{\n"
                        '  "header": "Short, patient-friendly title",\n'
                        '  "lists": [\n'
                        '    "Simple explanation of the tumor",\n'
                        '    "Key symptoms",\n'
                        '    "Common causes",\n'
                        '    "Treatment options",\n'
                        '    "Important notes for patients"\n'
                        "  ]\n"
                        "}\n\n"
                        "BUT — if the label is 'notumor', return only:\n"
                        '{\n  "header": "There is no brain tumor detected."\n}\n\n'
                        "Keep the language simple. Do not include any explanation or text outside the JSON object."
                    )
                    },
                    Image.open(filepath)
                ])
                gemini_message = clean_ai_json_response(gemini_response.text)
            except Exception as g_error:
                logger.error(f"Gemini response failed: {str(g_error)}")
                gemini_message = "Unable to retrieve detailed explanation at the moment."
            
            return jsonify({
                "success": True,
                "data": {
                    "result": label,
                    "confidence": confidence,
                    "message": gemini_message
                }
            })

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            return jsonify({"success": False, "message": "Processing error"}), 500

        finally:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.info(f"Removed temporary file: {filepath}")
            except Exception as e:
                logger.error(f"File cleanup failed: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": "Internal server error"}), 500
