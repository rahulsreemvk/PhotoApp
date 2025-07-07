import streamlit as st
import requests
from PIL import Image
import os
from PIL import ImageEnhance
from PIL import ImageOps
import re
import time

# --- Backend API for GPU processing ---
API_URL = "https://80737aeae8c6.ngrok-free.app/analyze/"  # Replace with real IP

# --- OpenRouter API Config ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")  # Store in secrets or env
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# --- Critique Prompt ---
def get_critique_from_openrouter(features, caption, score):
    prompt = f"""
You are a professional photography critic analyzing a photograph of {caption}.
You will see the image along with its extracted technical features and an aesthetic score of {score:.2f}.

Critically assess the photo in terms of:
- Composition (rule of thirds, symmetry, center framing, negative space)
- Lighting, exposure, sharpness, clarity, clutter
- Whether the visual choices serve the subject and artistic intent

Begin with a 1-10 rating, and follow with a short but insightful critique that acknowledges what works and what could be improved.

Technical features:
{features}
"""
    if time.time() - st.session_state.last_openrouter_call < 5:
        st.warning("⏳ Please wait a few seconds before retrying.")
        st.stop()

    payload = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10000,
        "temperature": 0.7
    }
    # Implement retry logic for rate limiting
    retries = 3
    backoff = 5  # seconds
    for attempt in range(retries):
        try:
            res = requests.post(OPENROUTER_API_URL, headers=HEADERS, json=payload)
            res.raise_for_status()
            st.session_state.last_openrouter_call = time.time()
            return res.json()["choices"][0]["message"]["content"]

        except requests.exceptions.HTTPError as e:
            if res.status_code == 429:
                if attempt < retries - 1:
                    wait_time = backoff * (attempt + 1)
                    st.warning(f"Too many requests. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    st.error("⚠️ Too many requests. Please wait and try again.")
                    st.stop()
            else:
                raise e
    # res = requests.post(OPENROUTER_API_URL, headers=HEADERS, json=payload)
    # res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

# --- Follow-up QA Prompt ---
def ask_about_photo(user_question, features, critique, score, context):
    visual_description = "\n".join([f"{k.replace('_',' ').capitalize()}: {v}" for k, v in features.items()])
    prompt = f"""
You are a professional photo editor and photography critic. You previously analyzed a photo and gave it a score of {score:.2f}/10.

Context of the photo: "{context}"

You said:
{critique}

Here are the extracted visual characteristics:
{visual_description}

Now, based on the above information, answer this follow-up question from the user:
"{user_question}"

If the question asks for improvements, give specific suggestions (e.g., "increase brightness to ~130", or "use warmer color temperature", or "target edge density around 0.3").
You can also suggest LUTs, color grading styles, or artistic edits.

Be clear, concise, and avoid repeating the full critique unless necessary.
"""
    if time.time() - st.session_state.last_openrouter_call < 5:
        st.warning("⏳ Please wait a few seconds before retrying.")
        st.stop()
        
    payload = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10000,
        "temperature": 0.7
    }
    # Implement retry logic for rate limiting
    retries = 3
    backoff = 5  # seconds
    for attempt in range(retries):
        try:
            res = requests.post(OPENROUTER_API_URL, headers=HEADERS, json=payload)
            res.raise_for_status()
            st.session_state.last_openrouter_call = time.time()
            return res.json()["choices"][0]["message"]["content"]

        except requests.exceptions.HTTPError as e:
            if res.status_code == 429:
                if attempt < retries - 1:
                    wait_time = backoff * (attempt + 1)
                    st.warning(f"Too many requests. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    st.error("⚠️ Too many requests. Please wait and try again.")
                    st.stop()
            else:
                raise e
    # res = requests.post(OPENROUTER_API_URL, headers=HEADERS, json=payload)
    # res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

def ask_for_edit_json(critique_text):
    prompt = f"""
Based on the following critique, return a JSON object with precise values for image editing. Use the format below, and do not include any commentary.

### CRITIQUE:
{critique_text}

### RETURN ONLY JSON:
{{
  "crop": {{ "rule_of_thirds": true }},
  "brightness": 1.1,
  "contrast": 1.2,
  "sharpness": 1.3,
  "saturation": 1.4,
  "color_balance": {{ "red": 1.0, "green": 0.95, "blue": 1.05 }},
  "white_balance": 5600,
  "color_grading": "golden_hour",
  "blur": {{ "background": 12, "subject": 0 }},
  "vignette": 0.3,
  "subject_position": "center"
}}
"""

    payload = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10000,
        "temperature": 0.5
    }

    res = requests.post(OPENROUTER_API_URL, headers=HEADERS, json=payload)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

def extract_value(text, pattern, default=1.0):
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return default

def apply_suggestions_to_image(image, suggestion_text):
    import numpy as np
    import cv2
    from PIL import ImageEnhance

    def extract_value(text, pattern, default=1.0, min_val=0.5, max_val=2.0):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            return max(min(val, max_val), min_val)  # clamp
        return default

    # Extract & Clamp
    brightness = extract_value(suggestion_text, r'brightness.*?(\d+(\.\d+)?)', 1.0, 0.5, 2.0)
    contrast   = extract_value(suggestion_text, r'contrast.*?(\d+(\.\d+)?)', 1.0, 0.5, 2.5)
    sharpness  = extract_value(suggestion_text, r'sharpness.*?(\d+(\.\d+)?)', 1.0, 0.5, 3.0)
    saturation = extract_value(suggestion_text, r'saturation.*?(\d+(\.\d+)?)', 1.0, 0.5, 2.0)

    # Basic Enhancements
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Sharpness(image).enhance(sharpness)

    # Optional: Saturation (via HSV)
    img_np = np.array(image)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= saturation
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    img_np = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    image = Image.fromarray(img_np)

    # Vignette
    image = apply_vignette(image)

    return image

def apply_edits_from_json(image, json_params):
    import numpy as np
    import cv2
    import json

    if isinstance(json_params, str):
        try:
            json_params = json.loads(json_params)
        except:
            st.error("❌ Failed to parse JSON.")
            return image

    # Crop (Rule of Thirds – Simple Center Crop)
    if json_params.get("crop", {}).get("rule_of_thirds"):
        w, h = image.size
        crop_size = int(min(w, h) * 0.8)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        image = image.crop((left, top, left + crop_size, top + crop_size))

    # Enhancements
    enhancers = {
        "brightness": ImageEnhance.Brightness,
        "contrast": ImageEnhance.Contrast,
        "sharpness": ImageEnhance.Sharpness
    }
    for key, enhancer in enhancers.items():
        image = enhancer(image).enhance(json_params.get(key, 1.0))

    # Saturation
    img_np = np.array(image)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= json_params.get("saturation", 1.0)
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    img_np = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    image = Image.fromarray(img_np)

    # Color Balance
    balance = json_params.get("color_balance", {})
    r, g, b = balance.get("red", 1.0), balance.get("green", 1.0), balance.get("blue", 1.0)
    img_np = np.array(image).astype(np.float32)
    img_np[..., 0] *= r
    img_np[..., 1] *= g
    img_np[..., 2] *= b
    img_np = np.clip(img_np, 0, 255)
    image = Image.fromarray(img_np.astype(np.uint8))

    # White Balance – placeholder logic (use LUT in advanced)
    kelvin = json_params.get("white_balance", 5500)
    image = ImageEnhance.Color(image).enhance(1.0 if 5000 <= kelvin <= 6500 else 0.95)

    # Vignette
    image = apply_vignette(image, json_params.get("vignette", 0.3))

    return image

def apply_vignette(image, strength=0.3):
    import cv2
    import numpy as np
    width, height = image.size
    kernel_x = cv2.getGaussianKernel(width, width * strength)
    kernel_y = cv2.getGaussianKernel(height, height * strength)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.max(kernel)
    vignette = np.dstack([mask] * 3)
    img = np.array(image).astype(np.float32)
    img = img * (vignette / 255)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))



# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "photo_uploaded" not in st.session_state:
    st.session_state.photo_uploaded = False

if "features" not in st.session_state:
    st.session_state.features = {}

if "caption" not in st.session_state:
    st.session_state.caption = ""

if "score" not in st.session_state:
    st.session_state.score = 0.0

if "critique" not in st.session_state:
    st.session_state.critique = ""

if "last_suggestion_text" not in st.session_state:
    st.session_state.last_suggestion_text = ""

if "last_openrouter_call" not in st.session_state:
    st.session_state.last_openrouter_call = 0


# --- Main App UI ---
st.title("📷 AI Photo Critique & Editing Assistant")

if not st.session_state.photo_uploaded:
    uploaded = st.file_uploader("Upload a photo to begin", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.session_state.original_image = image.copy()

        with st.spinner("Analyzing..."):
            files = {"file": uploaded.getvalue()}
            res = requests.post(API_URL, files=files)

        if res.status_code == 200:
            data = res.json()
            st.session_state.caption = data["caption"]
            st.session_state.score = data["score"]
            st.session_state.features = data["features"]

            with st.spinner("Getting AI critique..."):
                st.session_state.critique = get_critique_from_openrouter(
                    st.session_state.features,
                    st.session_state.caption,
                    st.session_state.score
                )

            st.session_state.chat_history.append({
                "role": "ai",
                "message": f"**📝 Caption:** {st.session_state.caption}\n\n**🌟 Score:** {st.session_state.score}\n\n**🤖 Critique:**\n{st.session_state.critique}"
            })

            st.session_state.photo_uploaded = True
            st.rerun()
        else:
            st.error("Analysis failed. Please try again.")
else:
    # Show Chat History
    st.markdown("### 💬 Chat with AI Editor")
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f"**🧑 You:** {chat['message']}")
        else:
            st.markdown(f"**🤖 AI:** {chat['message']}")

    # Input Box for Follow-up
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask about editing, enhancement, parameters, LUTs...", key="chat_input")
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        with st.spinner("Thinking..."):
            ai_reply = ask_about_photo(
                user_input,
                st.session_state.features,
                st.session_state.critique,
                st.session_state.score,
                st.session_state.caption
            )
        st.session_state.chat_history.append({"role": "ai", "message": ai_reply})
        st.session_state.last_suggestion_text = ai_reply  # Save for potential editing
        st.rerun()

    if st.session_state.last_suggestion_text:
        if st.button("🔧 Apply AI Edits Based on Critique"):
            with st.spinner("Parsing AI edits..."):
                json_edit_text = ask_for_edit_json(st.session_state.last_suggestion_text)
                edited_image = apply_edits_from_json(st.session_state.original_image.copy(), json_edit_text)
                st.image(edited_image, caption="Edited Image (AI Suggestions)", use_container_width=True)


    st.markdown("---")
    st.subheader("📤 Upload a new or edited photo to restart")

    new_photo = st.file_uploader("Upload another version or a new image", type=["jpg", "jpeg", "png"], key="next_upload")

    if new_photo:
        image = Image.open(new_photo).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.session_state.original_image = image.copy()

        with st.spinner("Analyzing..."):
            files = {"file": new_photo.getvalue()}
            res = requests.post(API_URL, files=files)

        if res.status_code == 200:
            data = res.json()
            caption = data["caption"]
            score = data["score"]
            features = data["features"]

            with st.spinner("Getting AI critique..."):
                critique = get_critique_from_openrouter(features, caption, score)

            st.session_state.chat_history.append({
                "role": "ai",
                "message": f"**📷 New Image Analyzed**\n\n**📝 Caption:** {caption}\n\n**🌟 Score:** {score}\n\n**🤖 Critique:**\n{critique}"
            })

            # Store latest for future follow-ups
            st.session_state.caption = caption
            st.session_state.score = score
            st.session_state.features = features
            st.session_state.critique = critique

            st.rerun()
        else:
            st.error("Failed to analyze the new image.")

    # Add manual reset button
    if st.button("🔄 Reset App"):
        st.session_state.clear()
        st.rerun()

