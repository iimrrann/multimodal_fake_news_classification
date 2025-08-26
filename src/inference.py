# src/inference.py
import torch
from PIL import Image
from src.utils import load_models, get_device

DEVICE = get_device()

# Load all models once
MODELS = load_models()

def preprocess_text(text, tokenizer):
    """
    Convert raw text to CLIP token tensor.
    """
    return tokenizer([text], truncate=True).to(DEVICE)

def preprocess_image(image, preprocess):
    """
    Convert PIL image to CLIP tensor.
    """
    return preprocess(image).unsqueeze(0).to(DEVICE)

@torch.no_grad()
def predict(text_input, image_input, clip_tokenizer, clip_preprocess, selected_models=None, return_probs=False):
    if selected_models is None:
        selected_models = MODELS

    text_tensor = preprocess_text(text_input, clip_tokenizer)
    image_tensor = preprocess_image(image_input, clip_preprocess)

    votes = []
    probs_list = []
    for model in selected_models:
        logits = model(text_tensor, image_tensor)
        pred_class = logits.argmax(dim=-1).item()
        votes.append(pred_class)
        if return_probs:
            probs_list.append(torch.softmax(logits, dim=1))

    final_class = max(set(votes), key=votes.count)
    return (final_class, votes, probs_list) if return_probs else (final_class, votes)

# Optional: label map
LABEL_MAP = {
    0: "TRUE",
    1: "SATIRE",
    2: "FALSE CONNECTION",
    3: "IMPOSTER CONTENT",
    4: "MANIPULATED CONTENT",
    5: "MISLEADING CONTENT",
}
