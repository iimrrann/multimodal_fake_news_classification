import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


# =======================================
# 1) Dual Cross-Attention Token Fusion
# =======================================
def extract_text_token_features(clip_model, text_tokens: torch.LongTensor) -> torch.Tensor:
    x = clip_model.token_embedding(text_tokens)           # [B, T, width]
    x = x + clip_model.positional_embedding
    x = x.permute(1, 0, 2)
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)
    x = clip_model.ln_final(x)
    return x  # [B, T, 512]


def extract_image_patch_tokens(clip_model, images: torch.FloatTensor) -> torch.Tensor:
    visual = clip_model.visual
    x = visual.conv1(images)  # [B, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    x = torch.cat([visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device), x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    if hasattr(visual, "ln_post") and visual.ln_post is not None:
        x = visual.ln_post(x)
    return x  # [B, 1+N, 768]


class DualCrossAttentionTokenFusion(nn.Module):
    def __init__(self, text_dim=512, image_dim=768, hidden_dim=384, num_heads=6, num_classes=6, dropout=0.1):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_to_image_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.image_to_text_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.text_pool = nn.Linear(hidden_dim, 1)
        self.image_pool = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, num_classes)
        )

    def forward(self, text_features, image_features):
        text_proj = self.text_proj(text_features)
        image_proj = self.image_proj(image_features)

        attended_text, _ = self.text_to_image_attn(query=text_proj, key=image_proj, value=image_proj)
        attended_image, _ = self.image_to_text_attn(query=image_proj, key=text_proj, value=text_proj)

        text_scores = self.text_pool(attended_text).squeeze(-1)
        text_weights = torch.softmax(text_scores, dim=1).unsqueeze(-1)
        text_rep = (attended_text * text_weights).sum(dim=1)

        image_scores = self.image_pool(attended_image).squeeze(-1)
        image_weights = torch.softmax(image_scores, dim=1).unsqueeze(-1)
        image_rep = (attended_image * image_weights).sum(dim=1)

        fused = torch.cat([text_rep, image_rep], dim=1)
        return self.classifier(fused)


class CLIPDualCrossTokenModel(nn.Module):
    def __init__(self, clip_model, hidden_dim=384, num_heads=6, num_classes=6):
        super().__init__()
        self.clip = clip_model
        self.fusion = DualCrossAttentionTokenFusion(
            text_dim=512, image_dim=768,
            hidden_dim=hidden_dim, num_heads=num_heads,
            num_classes=num_classes
        )

    def forward(self, text_tokens, images):
        text_token_feats = extract_text_token_features(self.clip, text_tokens)
        image_patch_feats = extract_image_patch_tokens(self.clip, images)
        return self.fusion(text_token_feats, image_patch_feats)


# =======================================
# 2) Dynamic Tensor Fusion
# =======================================
class DynamicTensorFusion(nn.Module):
    def __init__(self, text_dim=512, image_dim=512, hidden_dim=384, num_classes=6, dropout_prob=0.3):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fusion_dim = hidden_dim + 1

        self.weight_net = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, self.fusion_dim * self.fusion_dim)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.fusion_dim * self.fusion_dim),
            nn.Linear(self.fusion_dim * self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text_features, image_features):
        text_proj = self.dropout(self.text_norm(self.text_proj(text_features)))
        image_proj = self.dropout(self.image_norm(self.image_proj(image_features)))

        B = text_proj.size(0)
        text_bias = torch.cat([text_proj, torch.ones(B, 1, device=text_proj.device)], dim=1)
        image_bias = torch.cat([image_proj, torch.ones(B, 1, device=image_proj.device)], dim=1)

        weight_input = torch.cat([text_bias, image_bias], dim=1)
        weights = self.weight_net(weight_input).view(B, self.fusion_dim, self.fusion_dim)

        outer = torch.bmm(text_bias.unsqueeze(2), image_bias.unsqueeze(1))
        mask = torch.sigmoid(weights)
        weighted_fusion = outer * mask
        flat = weighted_fusion.view(B, -1)
        return self.classifier(flat)


class CLIPMultimodalModel(nn.Module):
    def __init__(self, clip_model, hidden_dim=384, num_classes=6):
        super().__init__()
        self.clip = clip_model
        self.fusion = DynamicTensorFusion(text_dim=512, image_dim=512, hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, text_tokens, images):
        text_features = self.clip.encode_text(text_tokens)
        image_features = self.clip.encode_image(images)
        return self.fusion(text_features.float(), image_features.float())


# =======================================
# 3) Early Fusion with Transformer Encoder
# =======================================
class EarlyFusionEncoder(nn.Module):
    def __init__(self, text_dim=512, image_dim=512, hidden_dim=384, num_heads=6, num_layers=2, num_classes=6):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, text_tokens, image_tokens):
        B = text_tokens.size(0)
        text_h = self.text_proj(text_tokens.float())
        image_h = self.image_proj(image_tokens.float())
        cls_tok = self.cls_token.expand(B, 1, -1).float()
        fused = torch.cat([cls_tok, image_h, text_h], dim=1)
        encoded = self.encoder(fused)
        cls_out = encoded[:, 0, :]
        return self.classifier(cls_out)


class CLIPEarlyFusionModel(nn.Module):
    def __init__(self, clip_model, hidden_dim=384, num_heads=6, num_classes=6, freeze_clip=True):
        super().__init__()
        self.clip = clip_model
        text_dim = clip_model.transformer.width          # usually 512
        image_dim = 768   #clip_model.visual.output_dim        # this will match 768 for ViT-B/32

        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

        self.fusion = EarlyFusionEncoder(
            text_dim=text_dim,
            image_dim=image_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=2,
            num_classes=num_classes
        )

    def forward(self, text_tokens, images):
        # Text token embeddings
        x = self.clip.token_embedding(text_tokens).type(self.clip.dtype)
        x = x + self.clip.positional_embedding.type(self.clip.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)
        text_tokens = x.float()

        # Vision patch tokens
        v = self.clip.visual.conv1(images.type(self.clip.dtype))
        v = v.reshape(v.shape[0], v.shape[1], -1).permute(0, 2, 1)
        v = torch.cat([self.clip.visual.class_embedding.to(v.dtype) +
                       torch.zeros(v.shape[0], 1, v.shape[-1], dtype=v.dtype, device=v.device), v], dim=1)
        v = v + self.clip.visual.positional_embedding.to(v.dtype)
        v = self.clip.visual.ln_pre(v)
        v = v.permute(1, 0, 2)
        v = self.clip.visual.transformer(v)
        v = v.permute(1, 0, 2)
        image_tokens = v.float()

        return self.fusion(text_tokens, image_tokens)
