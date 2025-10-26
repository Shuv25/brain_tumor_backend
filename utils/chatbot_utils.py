import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import requests

load_dotenv()

# Environment variables validation
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not all([PINECONE_API_KEY, GROQ_API_KEY]):
    raise EnvironmentError("Missing required environment variables")

# Initialize models and services
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("brain-tumor")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def process_text(user_query):
    """Process user query through vector search and LLM response generation."""
    try:
        print("Processing user query:", user_query)
        
        # Validate input
        if not isinstance(user_query, str) or not user_query.strip():
            return "Invalid query: Please provide a non-empty text input"

        # Generate embedding
        try:
            embedded_query = embedding_model.encode(user_query).tolist()
        except Exception as e:
            print(f"Embedding generation failed: {str(e)}")
            return "Error processing your query. Please try again."

        # Pinecone vector search
        try:
            results = index.query(
                vector=embedded_query,
                top_k=3,
                include_metadata=True
            )
            # print("Pinecone results:", results)
        except Exception as e:
            print(f"Pinecone query failed: {str(e)}")
            return "Error accessing medical knowledge base. Please try again later."

        # Process results
        if not results or 'matches' not in results or not results['matches']:
            return "No relevant information found in our knowledge base."

        context_parts = [match['metadata']['texts'] for match in results["matches"]]
        context = "\n\n".join(context_parts)

        print("context" , context)
        print("Generated context:", context[:500] + "...")  # Truncate for logging

        # Groq API call
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        groq_payload = {
                        "messages": [
                            {
                                "role": "system",
                                "content": f"""You are a medical assistant. Only reply to:\n- Brain tumor-related questions\n- Greetings
                                \n- Today's date or time\n- Medical helpline info\n\nIf asked anything else, say: 'I can 
                                only help with brain tumor related information.'\nAlways respond in very short bullet points. 
                                Use only the provided context. If unsure, say 'Not sure'.\n\nContext: {context}"""
                            },
                            {"role": "user", "content": user_query}
                        ],
                        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                        "temperature": 0.3
                    }
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=groq_payload,
                timeout=15  
            )
            response.raise_for_status() 
            
            response_data = response.json()
            if not response_data.get('choices'):
                return "Received unexpected response format from AI service"
            
            return response_data['choices'][0]['message']['content']

        except requests.exceptions.RequestException as e:
            print(f"Groq API request failed: {str(e)}")
            return "Error connecting to AI service. Please try again later."
        except KeyError as e:
            print(f"Unexpected response format: {str(e)}")
            return "Error processing AI response."

    except Exception as e:
        print(f"Unexpected error in process_text: {str(e)}")
        return "An unexpected error occurred. Please try again later."