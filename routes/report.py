from flask import Blueprint, request, jsonify, send_file
from utils.pdf_generator import generate_report
import os
import uuid

report_bp = Blueprint("report", __name__)

@report_bp.route("/generate", methods=["POST"])
def generate_report_route():
    try:
        data = request.json
        image_path = data.get("image_path")
        tumor_type = data.get("tumor_type")
        confidence = data.get("confidence")

        if not all([image_path, tumor_type, confidence is not None]):
            return jsonify({"success": False, "message": "Missing data for report"}), 400

        filename = f"report_{uuid.uuid4().hex}.pdf"
        output_path = os.path.join("reports", filename)
        os.makedirs("reports", exist_ok=True)

        generate_report(image_path, tumor_type, float(confidence), output_path)

        return send_file(output_path, as_attachment=True)

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
