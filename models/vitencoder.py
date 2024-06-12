import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, inputs):
        x = self.gelu(self.fc1(inputs))
        x = self.fc2(x)
        return x


class ViTEncoder(nn.Module):
    def __init__(self, num_heads=16, head_size=4, mlp_dim=[64, 128, 32]):
        super().__init__()

        self.ln1 = nn.LayerNorm([3,32,32])

        self.resweight1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, padding=0)

        self.mhsa = nn.MultiheadAttention(
            embed_dim=8*8*3,
            num_heads=num_heads,
            batch_first=True
        )

        self.resweight2 = nn.Conv2d(in_channels=3, out_channels=mlp_dim[-1], kernel_size=1, padding=0)

        self.ln2 = nn.LayerNorm([3,32,32])

        self.mlp = MLP(*mlp_dim)

    def forward(self, inputs):
        x = self.ln1(inputs)
        x_residual = self.resweight1(x)

        x = rearrange(x, 'b c (p1 h) (p2 w) ->b (p1 p2) (c h w)', h=8, w=8)

        x, _ = self.mhsa(x, x, x)
        x = rearrange(x, 'b (p1 p2) (c h w) ->b c (p1 h) (p2 w)', c = 3, p1 = 4, h = 8)
        x = x + x_residual

        x_residual = self.resweight2(x)
        x = self.ln2(x)
        x = self.mlp(x.permute(0, 2, 3, 1))
        x = x.permute(0,3,1,2) + x_residual
        return x


