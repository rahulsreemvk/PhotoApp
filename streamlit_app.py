import streamlit as st
import requests
from PIL import Image
import os

# --- Backend API for GPU processing ---
API_URL = "https://5e0f-172-83-13-4.ngrok-free.app/:8000/analyze/"  # Replace with real IP

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
        "model": "deepseek-ai/deepseek-coder:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800,
        "temperature": 0.7
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

    # Send to backend
    with st.spinner("Analyzing with CLIP & BLIP..."):
        files = {"file": uploaded.getvalue()}
        res = requests.post(API_URL, files=files)

    if res.status_code == 200:
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

        # Critique from DeepSeek
        with st.spinner("Getting AI Critique..."):
            critique = get_critique_from_openrouter(features, caption, score)

        st.markdown("### 🤖 Critique")
        st.markdown(critique)
    else:
        st.error(f"Error: {res.json().get('error')}")
