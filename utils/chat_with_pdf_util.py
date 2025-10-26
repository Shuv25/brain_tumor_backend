import io
import base64
import fitz
from PIL import Image
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import json

MAX_TOKENS_LIMIT = 131072  

def estimate_tokens_from_words(word_count):
    return int(word_count * 1.333)

def estimate_token_size(context_images, user_query):
    text_tokens = estimate_tokens_from_words(len(user_query.split()))  
    image_tokens = sum([len(img_b64) // 4 for img_b64 in context_images])  

    return text_tokens + image_tokens

def process_pdf(filepath):
    pdf_document = fitz.open(filepath)
    base64_images = []

    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_images.append(base64_str)

    return base64_images

def answer_query(context_images, user_query):

    token_count = estimate_token_size(context_images, user_query)

    if token_count > MAX_TOKENS_LIMIT:
        return "The PDF is too large to process within the token limit."

    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    system_msg = SystemMessage(
        content=(
            "If the user greets you (e.g., 'hi', 'hello'), respond formally and politely, "
            "and always reply using the strict JSON format described below.\n\n"
            "You are a helpful, precise assistant trained to analyze medical documents, MRI scans, and provide expert-level answers to user queries.\n\n"
            "When answering any query, including greetings, ALWAYS return your answer in this strict JSON format:\n\n"
            "{\n"
            '  "message": "Concise 20-30 word summary of the answer.",\n'
            '  "description": "Brief and focused explanation (1-2 short paragraphs, directly addressing the query).",\n'
            '  "lists": ["Optional: include only if needed to break down information", "Each item should be clear and short"]\n'
            "}\n\n"
            "Important formatting rules:\n"
            "- DO NOT include any text outside this JSON structure.\n"
            "- ONLY include the 'lists' array if it adds value to the explanation.\n"
            "- DO NOT answer anything beyond what was asked.\n"
            "- Use plain language suitable for patients or general users.\n"
            "- Be accurate, to the point, and professional.\n"
        )
    )




    content = [{"type": "text", "text": user_query}]

    for img_b64 in context_images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_b64}"
            }
        })

    message = HumanMessage(content=content)

    response = llm.invoke([system_msg, message])
    parsed_answer = json.loads(response.content)

    return parsed_answer
