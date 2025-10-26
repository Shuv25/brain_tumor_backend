from flask import Blueprint, jsonify, request, current_app
from utils.chat_with_pdf_util import process_pdf, answer_query
import uuid, os
import logging

chat_with_pdf_bp = Blueprint('chat_with_pdf', __name__)
pdf_contexts = {}

@chat_with_pdf_bp.route("/upload", methods=["POST"])
def upload_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "message": "No file was uploaded"}), 400
        
        file = request.files['file']
        if file.filename == "":
            return jsonify({"success": False, "message": "No file was chosen"}), 400

        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        upload_folder = "uploads"
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)

        context = process_pdf(filepath)
        os.remove(filepath)
        
        session_id = str(uuid.uuid4())
        pdf_contexts[session_id] = context
        current_app.logger.info(f"Session {session_id} created. Total sessions: {len(pdf_contexts)}")

        return jsonify({
            "success": True,
            "session_id": session_id,
            "message": "PDF uploaded and processed."
        })

    except Exception as e:
        current_app.logger.error(f"Upload error: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@chat_with_pdf_bp.route("/ask/<session_id>", methods=["POST"])
def ask_question(session_id):
    try:
        data = request.get_json()
        # session_id = data.get("session_id") if data else None
        text_query = data.get("text", "").strip() if data else ""
        
        current_app.logger.info(f"Ask request - Session ID: {session_id}, Text: {text_query}")
        current_app.logger.info(f"Existing sessions: {list(pdf_contexts.keys())}")

        if not session_id or session_id not in pdf_contexts:
            return jsonify({"success": False, "message": "Invalid or missing session_id"}), 400
        if not text_query:
            return jsonify({"success": False, "message": "Text query is empty"}), 400

        context = pdf_contexts[session_id]
        answer = answer_query(context, text_query)

        return jsonify({"success": True, "answer": answer}), 200

    except Exception as e:
        current_app.logger.error(f"Ask error: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500