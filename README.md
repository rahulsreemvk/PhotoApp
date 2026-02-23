# AI Context-Aware Photo Editor

An AI-powered photo editing system that analyzes image aesthetics, context, depth, and segmentation to generate professional crop and color-grade recommendations.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gdp-dashboard-template.streamlit.app/)

## 🚀 Overview

This project combines:

- CLIP + BLIP feature extraction
- Custom Aesthetic MLP model
- Depth estimation
- Subject segmentation (SAM)
- DeepSeek LLM (via OpenRouter) for critique & structured edit suggestions
- Streamlit frontend
- FastAPI backend

The system:
1. Analyzes the uploaded image
2. Generates detailed critique
3. Produces structured editing parameters
4. Applies region-aware edits locally
5. Provides manual tweak controls

---

## 🏗 Architecture

Frontend:
- Streamlit (GitHub)
- Communicates with FastAPI backend

Backend:
- FastAPI running on Paperspace
- Exposed via ngrok
- Model inference + segmentation + editing logic

LLM:
- DeepSeek via OpenRouter
- Used for critique + structured edit parameter generation

---

### How to run it

1. Install the dependencies/requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

### In Paperspace (Backend)
   
3. Configure environment in Paperspace

   ```
   Create `.env` file based on `.env.example`
   ```

4. Run Backend
   ```
   uvicorn backend.inference_api:app --reload --host 0.0.0.0 --port 8000
   ```

5. Run Frontend
   ```
   streamlit run app/streamlit_app.py
   ```


---

## 🔬 Current Status

⚠️ Research Prototype – Not Production Ready

See `docs/roadmap.md` for upcoming improvements.

---



