import torch
import torch.nn as nn

class MIVCPooling(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, img_embeds):  # img_embeds: [B, N, D]
        x = self.mlp(img_embeds)
        return x.mean(dim=1)  # [B, D]