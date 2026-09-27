"""Lightweight Vision Transformer for image steganalysis (cover vs stego).

The architecture is a plain ViT with two steganalysis-specific changes:

* an optional frozen **SRM-30 residual stem** in front of the patch embedding,
  which surfaces the ~1-LSB high-frequency footprint the ViT cannot see in raw
  pixels (see README), and
* attention computed with ``F.scaled_dot_product_attention`` so the mem-efficient
  / flash kernels are used on CUDA. The parameter layout matches
  ``nn.MultiheadAttention`` exactly (``in_proj_weight``/``in_proj_bias``/
  ``out_proj.*``), so checkpoints remain interchangeable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .srm_filter import SRM_KERNELS
except ImportError:  # fallback when run as a plain script
    from srm_filter import SRM_KERNELS


class SRMStem(nn.Module):
    """Fixed SRM-30 high-pass residual front-end.

    Grayscale (N,1,H,W) -> (N,30,H,W) signed residuals. The conv weights are the
    standard 30 SRM kernels (Fridrich) and are frozen; this single change lifts a
    raw-pixel ViT from chance toward a working detector by surfacing the weak
    high-frequency footprint that content-adaptive steganography leaves behind.

    ``truncate`` clamps the residual to [-T, T]. Content edges produce residuals
    far larger than the +-1 embedding quantisation step, so clipping bounds their
    influence without touching the embedding signal — the standard SRM
    preprocessing step. ``truncate=0`` disables it (the historical behaviour).
    """

    def __init__(self, freeze: bool = True, truncate: float = 0.0):
        super().__init__()
        k = torch.from_numpy(SRM_KERNELS[:, None, :, :]).float()  # (30,1,5,5)
        self.conv = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(k)
        if freeze:
            self.conv.weight.requires_grad = False
        self.truncate = float(truncate)

    def forward(self, x):
        r = self.conv(x)
        if self.truncate > 0:
            r = torch.clamp(r, -self.truncate, self.truncate)
        return r


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int = 1, embed_dim: int = 96):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (N, C, H, W) -> (N, P, D)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """Multi-head self-attention over a single tensor, via SDPA.

    Parameter-compatible with ``nn.MultiheadAttention`` (same names and shapes)
    so state dicts interchange, but the fused kernel is faster and lets the
    smaller ``head_dim`` stay eligible for the flash path.
    """

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0, "embed_dim must be divisible by heads"
        self.embed_dim = dim
        self.num_heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))
        self.out_proj = nn.Linear(dim, dim)
        # Mirror nn.MultiheadAttention._reset_parameters
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, length, _ = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        shape = (n, length, self.num_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).reshape(n, length, self.embed_dim)
        return self.out_proj(y)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LightViT(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 8,
        in_chans: int = 1,
        embed_dim: int = 96,
        depth: int = 4,
        heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_classes: int = 1,
        use_srm: bool = False,
        truncate: float = 0.0,
        pool: str = "cls",
    ):
        super().__init__()
        assert embed_dim % heads == 0, "embed_dim must be divisible by heads"
        assert pool in ("cls", "mean"), "pool must be 'cls' or 'mean'"
        self.use_srm = use_srm
        self.pool = pool
        if use_srm:
            self.stem = SRMStem(truncate=truncate)
            in_chans = 30
        else:
            self.stem = nn.Identity()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [Block(embed_dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0] if self.pool == "cls" else x[:, 1:].mean(1)).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


VIT_KEYS = ("img_size", "patch_size", "in_chans", "embed_dim", "depth", "heads",
            "mlp_ratio", "dropout", "use_srm", "truncate", "pool")


def build_vit(cfg: dict | None = None) -> LightViT:
    cfg = cfg or {}
    return LightViT(**{k: cfg[k] for k in VIT_KEYS if k in cfg})


if __name__ == "__main__":
    for use_srm in (False, True):
        model = build_vit({"img_size": 128, "patch_size": 8, "embed_dim": 96, "depth": 4, "heads": 3, "use_srm": use_srm})
        x = torch.randn(2, 1, 128, 128)
        y = model(x)
        print(f"use_srm={use_srm}: input {tuple(x.shape)} -> logits {tuple(y.shape)}, params={count_params(model):,}")
