import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingToLatentCondition(nn.Module):
    """
    Map IJEPA embedding (B, 64, H_emb, W_emb)
    to a latent-shaped bias (B, 4, H_lat, W_lat) to add on SD latents.
    """
    def __init__(self, emb_channels=64, latent_channels=4):
        super().__init__()
        self.proj = nn.Conv2d(emb_channels, latent_channels, kernel_size=1)

    def forward(self, emb, latent_spatial):
        """
        emb: (B, C_emb, H_emb, W_emb)
        latent_spatial: (H_lat, W_lat)
        """
        H_lat, W_lat = latent_spatial
        # downsample embedding spatially to latent size
        x = F.adaptive_avg_pool2d(emb, (H_lat, W_lat))  # (B, 64, H_lat, W_lat)
        cond = self.proj(x)  # (B, 4, H_lat, W_lat)
        return cond
