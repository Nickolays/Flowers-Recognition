import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x)

class ViTContrastive(nn.Module):
    def __init__(self, embed_dim=768, projection_dim=128, pretrained=True):
        super().__init__()
        # Load ViT backbone and cut classification head
        self.encoder = vit_b_16(pretrained=pretrained)
        self.encoder.heads = nn.Identity()  # remove cls head

        # Projection head: MLP with normalization
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim),
            Normalize()  # final output: L2-normalized
        )

    def forward(self, x):
        features = self.encoder(x)  # shape [B, 768]
        projections = self.projection_head(features)  # shape [B, 128]
        return projections
