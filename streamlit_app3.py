import streamlit as st
import requests
from PIL import Image
import os

# --- Backend API for GPU processing ---
API_URL = "https://3fff-172-83-13-4.ngrok-free.app/analyze/"  # Replace with real IP

# --- OpenRouter API Config ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")  # Store in secrets or env
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# --- DeepSeek Prompt Function ---
def get_critique_from_openrouter(features, caption, score):
    prompt = f"""
You are a professional photography critic.
Here is a photo description: "{caption}"
Aesthetic score: {score}/10
Technical features: {features}

Please provide a critique of the image, pointing out strengths and weaknesses.
Include suggestions to improve the photo. Format clearly.
"""
    payload = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10000,
        "temperature": 0.7,
        "top_p": 0.9
    }

    res = requests.post(OPENROUTER_API_URL, headers=HEADERS, json=payload)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

# --- Streamlit UI ---
st.title("📷 AI Photo Critique")

uploaded = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 🔧 CHANGED: Use uploaded.read() instead of .getvalue()
    image_bytes = uploaded.read()  # 🔧 Needed for proper UploadFile compatibility

    with st.spinner("Analyzing with CLIP & BLIP..."):
        try:
            # ✅ Send raw bytes in file upload format
            files = {"file": (uploaded.name, image_bytes, uploaded.type)}  # ✅ ADDED
            res = requests.post(API_URL, files=files)
            res.raise_for_status()  # ✅ ADDED: Ensure we catch bad responses

            data = res.json()
            caption = data["caption"]
            score = data["score"]
            features = data["features"]

            st.markdown("### 📝 Caption")
            st.write(caption)
            st.markdown("### 🌟 Aesthetic Score")
            st.write(score)
            st.markdown("### 🔍 Visual Features")
            st.json(features)

            with st.spinner("Getting AI Critique..."):
                critique = get_critique_from_openrouter(features, caption, score)

            st.markdown("### 🤖 Critique")
            st.markdown(critique)

        except requests.RequestException as e:
            st.error(f"Request failed: {e}")  # ✅ Better error handling
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")  # ✅ Broader catch
