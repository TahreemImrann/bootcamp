# Dyslexia Handwriting Classifier — ViT + LoRA (Explainable)

## Overview
This project fine-tunes a Vision Transformer (ViT) to classify handwriting
samples into three categories relevant to dyslexia screening — **Corrected**,
**Normal**, and **Reversal** — and provides visual explanations of the
model's predictions using attention heatmaps. The model is fine-tuned with
LoRA (Low-Rank Adaptation) for parameter efficiency, and a Gradio web app
lets users upload a handwriting image and see the predicted class alongside
a heatmap of where the model focused.

## Dataset
- Source: "Gambo" handwriting dataset (downloaded via Google Drive).
- Structure: `Train/` and `Test/` folders, each split into the three class
  folders (`Corrected`, `Normal`, `Reversal`).
- Sampling: up to 800 images per class for training, 150 per class for
  testing, for balanced class representation.
- Split: 85% train / 15% validation (stratified), plus a separate held-out
  test set — 2040 train / 360 validation / 450 test images.

## Model & Method
- **Backbone:** `google/vit-base-patch16-224` (pretrained Vision Transformer).
- **Fine-tuning method:** LoRA via the `peft` library — only the attention
  query/value projections and the classification head are trained
  (297K trainable parameters out of 86M total, ~0.35%).
- **LoRA config:** rank `r=8`, `alpha=16`, dropout `0.1`, applied to
  `q_proj`/`v_proj`, with the classifier head fully trainable.
- **Data augmentation (train only):** random rotation, small translation/
  scale jitter, and color jitter, to improve robustness to handwriting
  variation.
- **Training:** Hugging Face `Trainer`, batch size 32, 8 epochs,
  learning rate 1e-4, best checkpoint selected by macro F1 score.

## Results
| Class      | Precision | Recall | F1-score |
|------------|-----------|--------|----------|
| Corrected  | 0.96      | 0.73   | 0.83     |
| Normal     | 0.60      | 0.93   | 0.73     |
| Reversal   | 0.76      | 0.52   | 0.62     |

**Overall accuracy: 73%** | **Macro F1: 0.73**

A confusion matrix is generated to visualize class-wise performance.

## Explainability
The model's self-attention weights (last transformer layer, CLS token's
attention to image patches) are extracted and overlaid on the original
image as a heatmap, showing which regions of the handwriting most
influenced the prediction. This uses the `eager` attention implementation
so that real attention weights are returned (the default `sdpa`
implementation does not expose them).

## Deployment
A Gradio interface (`gr.Interface`) wraps the model for interactive use:
- **Input:** an uploaded handwriting image.
- **Output:** predicted class probabilities + the attention heatmap overlay.
- Launched with a public shareable link for demoing outside the notebook.

## How to Run
1. Run all notebook cells top to bottom (`Runtime > Run all` in Colab).
2. Training will reproduce the saved checkpoint in
   `./vit-lora-dyslexia-final/`, or this can be loaded directly instead of
   retraining.
3. The final cell launches the Gradio demo with a public URL.

## Tech Stack
Python, PyTorch, Hugging Face `transformers`, `peft` (LoRA), `evaluate`,
scikit-learn, Gradio.
<img width="687" height="329" alt="image" src="https://github.com/user-attachments/assets/d0f856f2-d105-4ff6-a287-a76c6cb3198a" />
<img width="735" height="289" alt="image" src="https://github.com/user-attachments/assets/bd520d50-5750-4470-892d-0d8232e389ce" />

