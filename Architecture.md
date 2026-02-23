# System Architecture

## Components

1. Streamlit Frontend
2. FastAPI Backend
3. ML Models (CLIP, BLIP, Aesthetic MLP, Depth, SAM)
4. DeepSeek LLM (OpenRouter)

## Data Flow

User Upload
    ↓
Backend Feature Extraction
    ↓
Aesthetic Scoring
    ↓
Depth + Segmentation
    ↓
Metadata Summary
    ↓
DeepSeek (Structured Edit JSON)
    ↓
Local Edit Application
    ↓
Preview Options
    ↓
Manual User Tweaks

---

## Key Principle

LLM provides structured edit recommendations.
Actual pixel-level editing is done locally.
