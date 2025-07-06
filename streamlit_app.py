import streamlit as st
import requests
from PIL import Image
import os

# --- Backend API for GPU processing ---
API_URL = "https://73c6-172-83-13-4.ngrok-free.app/analyze/"  # Replace with real IP

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
    payload = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10000,
        "temperature": 0.7
    }
    res = requests.post(OPENROUTER_API_URL, headers=HEADERS, json=payload)
    res.raise_for_status()
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
    payload = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10000,
        "temperature": 0.7
    }
    res = requests.post(OPENROUTER_API_URL, headers=HEADERS, json=payload)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

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

# --- Main App UI ---
st.title("📷 AI Photo Critique & Editing Assistant")

if not st.session_state.photo_uploaded:
    uploaded = st.file_uploader("Upload a photo to begin", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

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
        st.rerun()

    st.markdown("---")
    st.subheader("📤 Upload a new or edited photo to restart")
    new_photo = st.file_uploader("New image", type=["jpg", "jpeg", "png"], key="reset_upload")
    if new_photo:
        st.session_state.clear()
        st.rerun()
