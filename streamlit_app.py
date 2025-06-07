# streamlit_app.py
import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
import requests
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
import tempfile
import os
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models


# --- API CONFIG ---
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# --- AESTHETIC MODEL SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"

class AestheticPredictor(nn.Module):
    def __init__(self, input_dim=512):
        super(AestheticPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.model(x)

# Load aesthetic predictor weights (you can host on public URL or relative path if deployed)
@st.cache_resource
def load_aesthetic_model():
    mlp = AestheticPredictor(input_dim=512).to(device)
    model_path = "aesthetic_mlp.pt"  # Ensure it's present in the deployed root or use huggingface hub
    mlp.load_state_dict(torch.load(model_path, map_location=device))
    mlp.eval()
    return mlp

# Load CLIP model for embeddings
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

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
    
    mp_face_detection = mp.solutions.face_detection

    def count_faces(image_cv):
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
            results = fd.process(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            return len(results.detections) if results.detections else 0

    try:
        face_count = count_faces(image_cv)
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

# --- Aesthetic Scoring ---
def score_image(image: Image.Image, mlp, clip_model, processor):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
        score = mlp(image_features).item()
    return round(score, 2)

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
st.title("📷 AI Photo Critique")

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.features = None
    st.session_state.critique = None
    st.session_state.score = None

mlp_model = load_aesthetic_model()
clip_model, clip_processor = load_clip_model()

# --- Upload Image ---
uploaded = st.file_uploader("Upload a photo for critique", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    features = extract_visual_features(image)
    st.session_state.features = features

    score = score_image(image, mlp_model, clip_model, clip_processor)
    st.session_state.score = score
    st.markdown(f"### 🌟 Aesthetic Score: {score}/10")

    prompt = f"""
You are an expert photography critic. Based on these image characteristics:
{features}

Aesthetic score (out of 10): {score}

Provide a detailed critique and improvement suggestions.
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
