import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://<your-paperspace-ip>:8000/analyze/"  # Use public IP

st.title("📷 AI Photo Critique")

uploaded = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Send to Paperspace backend
    with st.spinner("Analyzing..."):
        files = {"file": uploaded.getvalue()}
        res = requests.post(API_URL, files=files)
    
    if res.status_code == 200:
        data = res.json()
        st.markdown("### 📝 Caption")
        st.write(data["caption"])
        st.markdown("### 🌟 Aesthetic Score")
        st.write(data["score"])
        st.markdown("### 🔍 Visual Features")
        st.json(data["features"])
    else:
        st.error(f"Error: {res.json().get('error')}")
