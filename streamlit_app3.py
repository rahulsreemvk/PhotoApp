import streamlit as st
import requests
from PIL import Image
import os

# --- Backend and OpenRouter Config ---
API_URL = "https://73c6-172-83-13-4.ngrok-free.app/analyze/"  # Replace with current backend
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# --- Step 1: Initial AI Critique ---
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
    payload = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10000,
        "temperature": 0.7
    }
    res = requests.post(OPENROUTER_API_URL, headers=HEADERS, json=payload)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

# --- Step 2: Handle Follow-up Questions ---
def ask_about_photo(user_question, features, critique, score, context):
    visual_description = "\n".join([
        f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in features.items()
    ])
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
    payload = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1200,
        "temperature": 0.7
    }
    res = requests.post(OPENROUTER_API_URL, headers=HEADERS, json=payload)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

# --- Step 3: Session Setup ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "photo_uploaded" not in st.session_state:
    st.session_state.photo_uploaded = False

if "caption" not in st.session_state:
    st.session_state.caption = ""

if "score" not in st.session_state:
    st.session_state.score = 0.0

if "features" not in st.session_state:
    st.session_state.features = {}

if "critique" not in st.session_state:
    st.session_state.critique = ""

# --- Step 4: UI Structure ---
st.title("📷 AI Photo Critique + Editing Assistant")

if not st.session_state.photo_uploaded:
    uploaded = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing photo..."):
            files = {"file": uploaded.getvalue()}
            res = requests.post(API_URL, files=files)

        if res.status_code == 200:
            data = res.json()
            st.session_state.caption = data["caption"]
            st.session_state.score = data["score"]
            st.session_state.features = data["features"]

            with st.spinner("Getting expert critique..."):
                st.session_state.critique = get_critique_from_openrouter(
                    st.session_state.features,
                    st.session_state.caption,
                    st.session_state.score
                )

            st.session_state.chat_history.append({
                "role": "system",
                "message": f"📝 **Caption**: {st.session_state.caption}\n\n🌟 **Aesthetic Score**: {st.session_state.score}\n\n📋 **Critique**:\n{st.session_state.critique}"
            })

            st.session_state.photo_uploaded = True
        else:
            st.error("❌ Error analyzing the photo.")
else:
    # Display previous chat
    st.markdown("### 🧠 AI Assistant Chat")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"👤 **You:** {msg['message']}")
        else:
            st.markdown(f"🤖 **AI:** {msg['message']}")

    # Follow-up input
    user_input = st.text_input("Ask something about the photo, editing tips, or enhancement methods:")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        with st.spinner("AI thinking..."):
            ai_response = ask_about_photo(
                user_input,
                st.session_state.features,
                st.session_state.critique,
                st.session_state.score,
                st.session_state.caption
            )
        st.session_state.chat_history.append({"role": "system", "message": ai_response})
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("📤 Want to start over? Upload a new or edited photo:")

    new_upload = st.file_uploader("Upload new image", type=["jpg", "jpeg", "png"], key="new_photo")
    if new_upload:
        st.session_state.clear()
        st.experimental_rerun()
