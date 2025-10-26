# from fpdf import FPDF
# from PIL import Image
# import os

# TUMOR_NAME_MAP = {
#     "notumor": "No Tumor",
#     "glioma": "Glioma Tumor",
#     "meningioma": "Meningioma Tumor",
#     "pituitary": "Pituitary Tumor"
# }

# def format_tumor_name(raw_name):
#     return TUMOR_NAME_MAP.get(raw_name.lower(), raw_name.capitalize())

# def generate_report(image_path, tumor_type, confidence, output_path):
#     pdf = FPDF()
#     pdf.add_page()

#     pdf.set_font("Arial", size=16)
#     pdf.cell(200, 10, txt="Brain Tumor Classification Report", ln=True, align="C")

#     formatted_tumor = format_tumor_name(tumor_type)


#     pdf.set_font("Arial", size=12)
#     pdf.ln(10)
#     pdf.cell(200, 10, txt=f"Tumor Type: {formatted_tumor}", ln=True)
#     pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}%", ln=True)

#     pdf.ln(10)

#     if tumor_type.lower() == "notumor":
#         pdf.multi_cell(0, 10, txt="No signs of tumor detected.\nMaintain a healthy lifestyle and go for regular check-ups.")
#     else:
#         pdf.multi_cell(0, 10, txt="Tumor detected.\nPlease consult a neurologist for further evaluation and treatment options.")

#     pdf.ln(10)
#     pdf.cell(200, 10, txt="Scanned Image:", ln=True)

#     try:
#         with Image.open(image_path) as img:
#             rgb_img = img.convert('RGB') 
#             temp_path = "temp_img_for_report.jpg"
#             rgb_img.save(temp_path, format="JPEG")
#             pdf.image(temp_path, x=10, y=pdf.get_y(), w=100)
#             os.remove(temp_path)
#     except Exception as e:
#         pdf.cell(200, 10, txt=f"(Image could not be embedded: {str(e)})", ln=True)

#     pdf.output(output_path)


from fpdf import FPDF
from PIL import Image
import os
from utils.segment_util import segment_tumor_image 

TUMOR_NAME_MAP = {
    "notumor": "No Tumor",
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "pituitary": "Pituitary Tumor"
}

def format_tumor_name(raw_name):
    return TUMOR_NAME_MAP.get(raw_name.lower(), raw_name.capitalize())

def generate_report(image_path, tumor_type, confidence, output_path):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Brain Tumor Classification Report", ln=True, align="C")

    formatted_tumor = format_tumor_name(tumor_type)

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Tumor Type: {formatted_tumor}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}%", ln=True)

    pdf.ln(10)
    if tumor_type.lower() == "notumor":
        pdf.multi_cell(0, 10, txt="No signs of tumor detected.\nMaintain a healthy lifestyle and go for regular check-ups.")
    else:
        pdf.multi_cell(0, 10, txt="Tumor detected.\nPlease consult a neurologist for further evaluation and treatment options.")

    pdf.ln(10)
    pdf.cell(200, 10, txt="Segmented Tumor Image:", ln=True)

    try:
        segmented_path = segment_tumor_image(image_path)

        with Image.open(segmented_path) as img:
            rgb_img = img.convert('RGB')
            temp_path = "temp_img_for_report.jpg"
            rgb_img.save(temp_path, format="JPEG")
            pdf.image(temp_path, x=10, y=pdf.get_y(), w=100)
            os.remove(temp_path)
    except Exception as e:
        pdf.cell(200, 10, txt=f"(Image could not be embedded: {str(e)})", ln=True)

    pdf.output(output_path)
