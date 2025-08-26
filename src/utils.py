# src/utils.py
import torch
import clip
from huggingface_hub import hf_hub_download, login
from src.model_definitions import CLIPEarlyFusionModel, CLIPMultimodalModel, CLIPDualCrossTokenModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 6

# ------------------------
# Hugging Face repos info
# ------------------------
MODEL_FILES = {
    "CLIP Early Fusion": {
        "class": CLIPEarlyFusionModel,
        "repo": "iimrrann/Early_Fuse",
        "filename": "best_clip_earlyfusion.pt",
    },
    "CLIP Multimodal Dynamic": {
        "class": CLIPMultimodalModel,
        "repo": "iimrrann/Dynamic_F",
        "filename": "best_clip_model_dynamic.pt",
    },
    "CLIP Dual Cross-Attention": {
        "class": CLIPDualCrossTokenModel,
        "repo": "iimrrann/Dual_F",
        "filename": "best_clip_dual_token_xattn.pt",
    },
}

def get_device():
    return DEVICE

def load_models(pat=None):
    """
    Load all models directly from Hugging Face Hub.
    Returns a dict {display_name: model}
    """
    # Login if private
    if pat:
        login(token=pat)

    # Load CLIP backbone once
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE, jit=False)
    clip_model = clip_model.float()

    models = {}

    for name, info in MODEL_FILES.items():
        try:
            # Download model from HF
            model_path = hf_hub_download(
                repo_id=info["repo"],
                filename=info["filename"],
                token=pat
            )

            # Instantiate model
            model_class = info["class"]
            model = model_class(clip_model=clip_model, num_classes=NUM_CLASSES)

            # Load checkpoint
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()

            models[name] = model
            print(f"[INFO] Loaded {name} from Hugging Face Hub")

        except Exception as e:
            print(f"[ERROR] Failed loading {name}: {e}")

    return models
