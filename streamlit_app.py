import streamlit as st
import requests
from PIL import Image
import os

# --- Backend API for GPU processing ---
API_URL = "https://b2e8-172-83-13-4.ngrok-free.app/analyze/"  # Replace with real IP

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
You are a professional photography critic analyzing a photograph of {caption}.
You will see the image along with its extracted technical features and an aesthetic score of {score:.2f}.

Critically assess the photo in terms of:
- Composition (rule of thirds, symmetry, center framing, negative space)
- Lighting, exposure, sharpness, clarity, clutter
- Whether the visual choices serve the subject and artistic intent

Do not blindly apply the rule of thirds — consider the context and possible creative choices.
Begin with a 1-10 rating, and follow with a short but insightful critique that acknowledges what works and what could be improved.

Technical features:
{features}
"""
#     prompt = f"""
# You are a professional photography critic.
# Here is a photo description: "{caption}"
# Aesthetic score: {score}/10
# Technical features: {features}

# Please provide a critique of the image, pointing out strengths and weaknesses.
# Include suggestions to improve the photo. Format clearly.
# """
    payload = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10000,
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
    with st.spinner("Analyzing..."):
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
        # st.markdown("### 🔍 Visual Features")
        # st.json(features)

        # Critique from DeepSeek
        with st.spinner("Getting AI Critique..."):
            critique = get_critique_from_openrouter(features, caption, score)

        st.markdown("### 🤖 Critique")
        st.markdown(critique)
    else:
        st.error(f"Error: {res.json().get('error')}")
