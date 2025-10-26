from flask import Blueprint, jsonify, request
from utils.chatbot_utils import process_text

chatbot_bp = Blueprint('chatbot', __name__)


@chatbot_bp.route("/chatbot", methods=["POST"])
def handle_chat():
    try:
        print("handle Chat bot is running")
        if 'text' in request.json:
            text_query = request.json['text']
            if text_query.strip() == "":
                return jsonify({
                    "success": False, 
                    "message": "Text query is empty"
                }), 400

            response = process_text(text_query)
            
            return jsonify({
                "success": True, 
                "message": "Processed text query successfully", 
                "response": response
            }), 200
        
    except Exception as e:
        return jsonify({
            "success": False, 
            "message": str(e)
        }), 500
