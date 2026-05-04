import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        # x: (b, t, c)
        h = self.fc(s).unsqueeze(1)  # (b, 1, 2c)
        gamma, beta = torch.chunk(h, chunks=2, dim=-1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        return (1 + gamma) * x + beta


class StyleAttention(nn.Module):
    def __init__(self, features, style_dim, num_heads, head_features):
        super().__init__()
        mid_features = head_features * num_heads
        self.num_heads = num_heads
        self.scale = head_features**-0.5
        self.norm = AdaLayerNorm(style_dim, features)
        self.to_qkv = nn.Linear(features, mid_features * 3, bias=False)
        self.to_out = nn.Linear(mid_features, features)

    def forward(self, x, s):
        x = self.norm(x, s)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.num_heads) for t in qkv)

        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class StyleTransformerBlock(nn.Module):
    def __init__(self, features, style_dim, num_heads, head_features, multiplier):
        super().__init__()
        self.attn = StyleAttention(features, style_dim, num_heads, head_features)
        self.ff = nn.Sequential(
            nn.Linear(features, features * multiplier),
            nn.GELU(),
            nn.Linear(features * multiplier, features),
        )
        self.norm = AdaLayerNorm(style_dim, features)

    def forward(self, x, s):
        x = x + self.attn(x, s)
        x = x + self.ff(self.norm(x, s))
        return x


class StyleTransformer1d(nn.Module):
    def __init__(
        self,
        channels,
        context_embedding_features,
        context_features,
        num_layers,
        num_heads,
        head_features,
        multiplier,
    ):
        super().__init__()
        self.features = channels + context_embedding_features
        self.blocks = nn.ModuleList(
            [
                StyleTransformerBlock(
                    self.features,
                    context_features,
                    num_heads,
                    head_features,
                    multiplier,
                )
                for _ in range(num_layers)
            ]
        )
        self.to_out = nn.Conv1d(self.features, channels, 1)
        self.to_mapping = nn.Sequential(
            nn.Linear(self.features, self.features),
            nn.GELU(),
            nn.Linear(self.features, self.features),
            nn.GELU(),
        )
        self.to_time = nn.Sequential(nn.Linear(1, self.features), nn.GELU())
        self.to_features = nn.Sequential(
            nn.Linear(context_features, self.features), nn.GELU()
        )

    def forward(self, x, time, embedding, features, embedding_mask_proba=0.0):
        # x: (b, 1, c), time: (b,), embedding: (b, t, c), features: (b, c)
        mapping = self.to_time(time.unsqueeze(-1)) + self.to_features(features)
        mapping = self.to_mapping(mapping).unsqueeze(1)

        x = torch.cat([x.expand(-1, embedding.size(1), -1), embedding], dim=-1)

        for block in self.blocks:
            x = x + mapping
            x = block(x, features)

        x = x.mean(dim=1, keepdim=True)
        return self.to_out(x.transpose(1, 2)).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, features, num_heads, head_features, multiplier):
        super().__init__()
        self.attn = nn.MultiheadAttention(features, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(features, features * multiplier),
            nn.GELU(),
            nn.Linear(features * multiplier, features),
        )
        self.norm1 = nn.LayerNorm(features)
        self.norm2 = nn.LayerNorm(features)

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class Transformer1d(nn.Module):
    def __init__(
        self,
        channels,
        context_embedding_features,
        num_layers,
        num_heads,
        head_features,
        multiplier,
    ):
        super().__init__()
        self.features = channels + context_embedding_features
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(self.features, num_heads, head_features, multiplier)
                for _ in range(num_layers)
            ]
        )
        self.to_out = nn.Conv1d(self.features, channels, 1)
        self.to_mapping = nn.Sequential(
            nn.Linear(self.features, self.features),
            nn.GELU(),
            nn.Linear(self.features, self.features),
            nn.GELU(),
        )
        self.to_time = nn.Sequential(nn.Linear(1, self.features), nn.GELU())

    def forward(self, x, time, embedding, features=None, embedding_mask_proba=0.0):
        mapping = self.to_time(time.unsqueeze(-1))
        mapping = self.to_mapping(mapping).unsqueeze(1)

        x = torch.cat([x.expand(-1, embedding.size(1), -1), embedding], dim=-1)

        for block in self.blocks:
            x = x + mapping
            x = block(x)

        x = x.mean(dim=1, keepdim=True)
        return self.to_out(x.transpose(1, 2)).transpose(1, 2)
