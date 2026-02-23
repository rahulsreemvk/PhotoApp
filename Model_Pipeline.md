# Model Pipeline

## Step 1: Feature Extraction
- CLIP embeddings
- BLIP caption
- Aesthetic score (MLP)

## Step 2: Technical Analysis
- Depth estimation
- Subject segmentation
- Saliency detection
- Histogram analysis
- Color temperature estimation

## Step 3: LLM Recommendation
DeepSeek returns structured JSON:

{
  "crop": {...},
  "subject": {...},
  "background": {...},
  "global": {...}
}

## Step 4: Local Edit Application
- Mask-based editing
- Depth-aware blur
- Color grading
- LUT blending
- Exposure correction
