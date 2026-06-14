import os
from dotenv import load_dotenv
from groq import Groq
from utils.logger import logger

load_dotenv()

def generate_ai_explanation(prediction: str, probability: float, top_features: list) -> str:
    try:
        logger.info("Groq explanation request started.")
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Format the features for the prompt
        features_str = "\n".join([f"{f['feature']}: {f['importance']}" for f in top_features])
        
        prompt = f"""
The following SHAP features contributed most strongly to a solar flare prediction:

{features_str}

Prediction: {prediction}
Probability: {probability}

Generate a concise scientific explanation describing why these telemetry patterns may indicate possible solar flare activity.
Keep the explanation short and professional. Maximum 3-4 lines. Do not hallucinate.
"""
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional space weather scientist analyzing solar telemetry data."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=150,
        )
        
        explanation = completion.choices[0].message.content.strip()
        logger.info("Groq explanation success.")
        return explanation

    except Exception as e:
        logger.error(f"Groq explanation failure: {e}")
        return "AI explanation service temporarily unavailable."
