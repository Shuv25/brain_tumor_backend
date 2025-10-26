from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import onnx
import onnxruntime as ort

load_dotenv()

from routes.inference import interface_bp
from routes.report import report_bp
from routes.chatbot import chatbot_bp
from routes.cha_with_pdf import chat_with_pdf_bp

app = Flask(__name__)

CORS(app)

# Register blueprints for other API routes
app.register_blueprint(interface_bp, url_prefix="/api/interface")
app.register_blueprint(report_bp, url_prefix="/api/report")
app.register_blueprint(chatbot_bp, url_prefix="/api")
app.register_blueprint(chat_with_pdf_bp, url_prefix="/api/chat-pdf")

# Model directory and path
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_classifier.onnx")  # Updated to ONNX model

os.makedirs(MODEL_DIR, exist_ok=True)

# Check if model exists in the local directory
if not os.path.exists(MODEL_PATH):
    raise ValueError(f"Model file '{MODEL_PATH}' not found!")

# Load the ONNX model
onnx_model = onnx.load(MODEL_PATH)
ort_session = ort.InferenceSession(MODEL_PATH)

# @app.route("/prince", methods=["GET"])
# def prince():
#     return jsonify({"message": "hello"}), 200

@app.route("/")
def home():
    return {"message": "Brain Tumor Backend API is running."}

if __name__ == '__main__':
    app.run(debug=True)
