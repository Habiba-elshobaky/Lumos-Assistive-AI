import google.generativeai as genai
import PIL.Image
import os
from keys import GEMINI_API_KEY

# --- INITIALIZATION ---
genai.configure(api_key=GEMINI_API_KEY)

def get_working_model():
    
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for name in available_models:
            if "1.5-flash" in name: return genai.GenerativeModel(name)
        return genai.GenerativeModel(available_models[0]) if available_models else None
    except: return None

gemini_model = get_working_model()

def describe_scene(image_path, user_query="Hello"):
   
    if not gemini_model: return "I'm sorry, my brain is offline. Check your API key."

    try:
        if not os.path.exists(image_path): return "I can't see anything right now."
        
        img = PIL.Image.open(image_path)
        
        
        prompt = (
            f"You are 'Luma', a helpful and friendly AI assistant for a blind person. "
            f"The user just said: '{user_query}'. "
            "1. Be conversational and warm. "
            "2. If they ask to find an object, tell them exactly where it is (left, right, center). "
            "3. If they ask for a description, be vivid but concise (under 20 words). "
            "4. Always prioritize safety hazards if you see any."
        )
        
        response = gemini_model.generate_content([prompt, img])
        return response.text.strip()
    except Exception as e:
        return f"I'm having trouble processing that: {str(e)[:40]}"