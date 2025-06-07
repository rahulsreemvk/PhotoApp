# streamlit_app.py
import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image
# import face_recognition
import tempfile
import os

# --- API CONFIG ---
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# --- FEATURE EXTRACTION ---
def extract_visual_features(image: Image.Image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    resized = cv2.resize(image_cv, (224, 224))
    h, w = resized.shape[:2]
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    contrast = np.std(gray)

    (B, G, R) = cv2.split(resized.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    colorfulness = np.sqrt(np.mean(rg**2) + np.mean(yb**2))

    chans = cv2.split(resized)
    hist_features = {}
    for chan, name in zip(chans, ['b', 'g', 'r']):
        hist = cv2.calcHist([chan], [0], None, [8], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features[f"{name}_histogram"] = hist.tolist()

    center_offset = 1.0
    try:
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliencyMap = saliency.computeSaliency(resized)
        if success:
            saliencyMap = (saliencyMap * 255).astype("uint8")
            moments = cv2.moments(saliencyMap)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                center_offset = np.linalg.norm([cx - w/2, cy - h/2]) / (w/2)
    except:
        pass

    aspect_ratio = round(image.width / image.height, 2)
    face_count = 0
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_img = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)
        face_count = len(faces)
    except:
        face_count = 0


    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0)
    lighting_std = np.std(resized)
    lighting_mean = np.mean(resized)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    blur_score = np.mean(cv2.absdiff(gray, blur))

    features = {
        "sharpness": round(sharpness, 2),
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "colorfulness": round(colorfulness, 2),
        "edge_density": round(edge_density, 4),
        "lighting_mean": round(lighting_mean, 2),
        "lighting_std": round(lighting_std, 2),
        "center_offset": round(center_offset, 2),
        "aspect_ratio": aspect_ratio,
        "face_count": face_count
    }
    features.update(hist_features)
    return features

# --- API CALL ---
def query_deepseek(prompt):
    payload = {
        "model": "deepseek-ai/deepseek-coder:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.7,
        "top_p": 0.9
    }
    res = requests.post(API_URL, headers=HEADERS, json=payload)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

# --- Streamlit App ---
st.set_page_config(page_title="Photo Critique AI", layout="wide")
st.title("📷 AI Photo Critique & Chat")

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.features = None
    st.session_state.critique = None

# --- Upload Image ---
uploaded = st.file_uploader("Upload a photo for critique", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    features = extract_visual_features(image)
    st.session_state.features = features

    prompt = f"""
You are an expert photography critic. Based on these image characteristics:
{features}\n\nGive a critique (score out of 10) and suggestions.
"""
    critique = query_deepseek(prompt)
    st.session_state.critique = critique
    st.markdown("### 📝 Critique")
    st.markdown(critique)
    st.session_state.history.append(("Critique", critique))

# --- Chat Interface ---
if st.session_state.critique:
    st.markdown("---")
    st.markdown("### 💬 Ask Questions About the Photo")
    user_q = st.text_input("Ask a question or upload a new photo")
    if user_q:
        context_prompt = f"The previous critique was: {st.session_state.critique}\nUser question: {user_q}"
        answer = query_deepseek(context_prompt)
        st.markdown("**Answer:**")
        st.write(answer)
        st.session_state.history.append(("Q: " + user_q, answer))

# --- History ---
if st.session_state.history:
    with st.expander("📜 View Session History"):
        for q, a in st.session_state.history:
            st.markdown(f"**{q}**\n{a}")
