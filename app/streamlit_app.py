import sys
from pathlib import Path
import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt
import clip

# ----------------------------
# Add project root to sys.path
# ----------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.utils import load_models, get_device
from src.inference import predict, LABEL_MAP

# ----------------------------
# Setup
# ----------------------------
DEVICE = get_device()
st.set_page_config(page_title="Multimodal Fake News Classifier", layout="wide")

# ----------------------------
# Hugging Face PAT (optional)
# ----------------------------
PAT = st.secrets.get("HF_PAT", None)

# ----------------------------
# Load models (cached)
# ----------------------------
@st.cache_resource
def load_hf_models(pat=None):
    return load_models(pat=pat)

MODELS = load_hf_models(pat=PAT)

# ----------------------------
# Load CLIP backbone (cached)
# ----------------------------
@st.cache_resource
def load_clip_model(model_name="ViT-B/32"):
    model, preprocess = clip.load(model_name, device=DEVICE, jit=False)
    return model, preprocess

clip_model, clip_preprocess = load_clip_model()
clip_tokenizer = clip.tokenize

# ----------------------------
# Layout: horizontal 2 columns
# ----------------------------
left_col, right_col = st.columns([1, 1.5])

# ----------------------------
# Left Column: Inputs
# ----------------------------
with left_col:
    st.header("Input")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    input_text = st.text_area("Enter News Title / Text")

    st.markdown("---")
    st.subheader("Select Models for Voting")
    model_names = list(MODELS.keys())
    selected_model_names = st.multiselect(
        "Select Models", options=model_names, default=model_names
    )
    selected_models = [MODELS[name] for name in selected_model_names]

    predict_button = st.button("Predict")

# ----------------------------
# Right Column: Predictions & Visualization
# ----------------------------
with right_col:
    st.header("Prediction Results")

    # Image preview
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        image.thumbnail((400, 400))
        st.image(image, caption="Uploaded Image", use_container_width=False)
    else:
        st.info("Upload an image to see it here.")

    if predict_button:
        if not uploaded_image or not input_text.strip():
            st.warning("Please provide both an image and text to classify.")
        elif len(selected_models) == 0:
            st.warning("Please select at least one model for prediction.")
        else:
            with st.spinner("Running inference..."):
                final_class, votes, probs_list = predict(
                    input_text,
                    image,
                    clip_tokenizer,
                    clip_preprocess,
                    selected_models=selected_models,
                    return_probs=True
                )

            # --- Voting Logic with Tie-Break ---
            if len(selected_models) > 1:
                vote_counts = {}
                for v in votes:
                    vote_counts[v] = vote_counts.get(v, 0) + 1
                max_votes = max(vote_counts.values())
                candidates = [cls for cls, cnt in vote_counts.items() if cnt == max_votes]

                if len(candidates) == 1:
                    chosen_class = candidates[0]
                else:
                    # tie → pick class with highest average confidence
                    avg_probs = torch.stack(probs_list).mean(dim=0).cpu().numpy().flatten()
                    chosen_class = max(candidates, key=lambda c: avg_probs[c])
            else:
                chosen_class = votes[0]

            # --- Display Results ---
            st.success(f"**Predicted Label:** {LABEL_MAP[chosen_class]}")

            st.markdown("**Individual Model Votes:**")
            for i, (vote, probs) in enumerate(zip(votes, probs_list)):
                st.write(f"{selected_model_names[i]} → {LABEL_MAP[vote]} "
                         f"(Confidence: {probs.max().item():.2f})")

            # --- Probability Bar Chart ---
            if probs_list:
                st.markdown("**Average Probability Across Selected Models:**")
                fig, ax = plt.subplots(figsize=(8, 4))
                probs_avg = torch.stack(probs_list).mean(dim=0).cpu().numpy().flatten()
                ax.bar([LABEL_MAP[i] for i in range(len(probs_avg))], probs_avg, color='skyblue')
                ax.set_ylabel("Average Probability")
                ax.set_ylim(0, 1)
                ax.set_xticks(range(len(probs_avg)))
                ax.set_xticklabels([LABEL_MAP[i] for i in range(len(probs_avg))], rotation=30, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

# ----------------------------
# Sidebar: Info
# ----------------------------
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.markdown("""
This app uses three different CLIP-based multimodal models hosted on Hugging Face:

1. **CLIP Early Fusion**  
2. **CLIP Multimodal Dynamic Tensor Fusion**  
3. **CLIP Dual Cross-Attention Token Fusion**  

Predictions are combined via majority vote, with tie-breaking based on highest confidence.
""")
