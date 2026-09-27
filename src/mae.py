"""Masked Autoencoder (MAE) for SRM-residual self-supervised pre-training.

Milestone 4: pre-train the lightweight ViT encoder on BOSSbase covers by masking
patches of the 30-channel SRM residual and reconstructing them, then transfer the
encoder (patch embed + transformer blocks) to the LightViT steganalysis detector
and fine-tune.

Reconstructing the residual — not raw pixels — aligns the self-supervised task with
the high-frequency signal steganalysis actually relies on (raw-pixel ViTs sit at
chance, AUC ~0.5; see README M0).
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .vit_stego import SRMStem, PatchEmbed, Block
except ImportError:  # run as a plain script
    from vit_stego import SRMStem, PatchEmbed, Block


def random_masking(x: torch.Tensor, mask_ratio: float):
    """Shuffle patches, keep (1-mask_ratio) of them, return kept tokens + bookkeeping."""
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D))
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return x_masked, mask, ids_restore, ids_keep


class MAEEncoder(nn.Module):
    """Patch embedding + transformer blocks (no cls token). Mirrors the LightViT
    encoder so its state_dict transfers directly (pos_embed covers patches only)."""

    def __init__(self, img_size: int = 128, patch_size: int = 8, embed_dim: int = 96,
                 depth: int = 4, heads: int = 3, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.patch_embed = PatchEmbed(img_size, patch_size, 30, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.blocks = nn.ModuleList(
            [Block(embed_dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

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

    def forward(self, tokens: torch.Tensor, ids_keep: torch.Tensor, ids_restore_len: int = None):
        """Process only kept patches.

        tokens: (N, P, D) — patch embeddings of the full residual map.
        ids_keep: (N, len_keep) — indices of kept patches in shuffled order.
        """
        pos = self.pos_embed.expand(tokens.shape[0], -1, -1)
        pos_keep = torch.gather(pos, 1, ids_keep.unsqueeze(-1).repeat(1, 1, tokens.shape[-1]))
        x = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).repeat(1, 1, tokens.shape[-1]))
        x = x + pos_keep
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class MAE(nn.Module):
    """SRM stem + MAEEncoder + lightweight decoder.

    forward(cover) returns (loss, pred, mask); the encoder weights are later
    transferred to LightViT via load_pretrained_encoder.
    """

    def __init__(self, img_size: int = 128, patch_size: int = 8, embed_dim: int = 96,
                 depth: int = 4, heads: int = 3, decoder_depth: int = 1,
                 mask_ratio: float = 0.75, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.mask_ratio = mask_ratio
        self.in_chans = 30

        self.stem = SRMStem()  # frozen, 1 -> 30 channels
        self.encoder = MAEEncoder(img_size, patch_size, embed_dim, depth, heads, mlp_ratio, dropout)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.decoder_blocks = nn.ModuleList(
            [Block(embed_dim, heads, mlp_ratio, dropout) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, self.in_chans * patch_size * patch_size)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def patchify(self, r: torch.Tensor) -> torch.Tensor:
        N, C, H, W = r.shape
        ps = self.patch_size
        g = H // ps
        return r.reshape(N, C, g, ps, g, ps).permute(0, 2, 4, 1, 3, 5).reshape(N, g * g, C * ps * ps)

    def forward(self, x: torch.Tensor, mask_ratio: float | None = None):
        # x: (N, 1, H, W) cover on [0, 255]
        mask_ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        D = self.encoder.patch_embed.proj.out_channels
        r = self.stem(x)                       # (N, 30, H, W)
        target = self.patchify(r)              # (N, P, 30*ps*ps)
        tokens = self.encoder.patch_embed(r)   # (N, P, D)

        x_masked, mask, ids_restore, ids_keep = random_masking(tokens, mask_ratio)
        latent = self.encoder.forward(tokens, ids_keep)  # (N, len_keep, D)

        N = x.shape[0]
        mask_tokens = self.mask_token.repeat(N, self.num_patches - latent.shape[1], 1)
        x_ = torch.cat([latent, mask_tokens], dim=1)                     # (N, P, D)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).repeat(1, 1, D))
        x_ = x_ + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x_ = blk(x_)
        pred = self.decoder_pred(self.decoder_norm(x_))                  # (N, P, 30*ps*ps)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)               # per-patch mean squared error
        loss = (loss * mask).sum() / mask.sum()
        return loss, pred, mask


def _interpolate_pos_embed(pe: torch.Tensor, target_len: int) -> torch.Tensor:
    """Bilinear resize a (1, L, D) positional embedding to (1, target_len, D)."""
    import torch.nn.functional as F
    L = pe.shape[1]
    g = int(round(L ** 0.5))
    tg = int(round(target_len ** 0.5))
    if g * g != L or tg * tg != target_len or g == tg:
        return pe[:, :target_len, :] if pe.shape[1] >= target_len else pe
    pe2 = pe.permute(0, 2, 1).reshape(1, -1, g, g)          # (1, D, g, g)
    pe2 = F.interpolate(pe2, size=(tg, tg), mode="bilinear", align_corners=False)
    return pe2.reshape(1, -1, target_len).permute(0, 2, 1)  # (1, target_len, D)


def load_pretrained_encoder(model: nn.Module, ckpt_path: str, map_location: str = "cpu") -> nn.Module:
    """Transfer MAE encoder weights (patch embed, pos embed, blocks, norm) onto LightViT.

    The LightViT cls_token/head are left to be learned during fine-tuning; the frozen
    SRM stem is identical on both sides, so it is not copied.
    """
    sd = torch.load(ckpt_path, map_location=map_location)
    if any(k.startswith("encoder.") for k in sd):
        sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}

    missing = 0
    with torch.no_grad():
        # patch embedding (Conv2d 30 -> embed_dim)
        if "patch_embed.proj.weight" in sd:
            model.patch_embed.proj.weight.copy_(sd["patch_embed.proj.weight"])
            if "patch_embed.proj.bias" in sd:
                model.patch_embed.proj.bias.copy_(sd["patch_embed.proj.bias"])
        # positional embedding -> LightViT pos_embed (slot 0 is the cls token)
        if "pos_embed" in sd:
            pe = sd["pos_embed"]
            L = model.patch_embed.num_patches
            if pe.shape[1] != L:
                pe = _interpolate_pos_embed(pe, L)
            model.pos_embed[:, 1:1 + L, :].copy_(pe.squeeze(0))
        # transformer blocks
        for i, blk in enumerate(model.blocks):
            for name, param in blk.named_parameters():
                key = f"blocks.{i}.{name}"
                if key in sd:
                    param.copy_(sd[key])
                else:
                    missing += 1
        # final layernorm
        if "norm.weight" in sd:
            model.norm.weight.copy_(sd["norm.weight"])
            if "norm.bias" in sd:
                model.norm.bias.copy_(sd["norm.bias"])

    print(f"[mae] loaded encoder from {ckpt_path} (unmatched block params: {missing})")
    return model


if __name__ == "__main__":
    m = MAE(img_size=128, patch_size=8, embed_dim=96, depth=4, heads=3,
            decoder_depth=1, mask_ratio=0.75)
    x = torch.randn(2, 1, 128, 128) * 255.0
    loss, pred, mask = m(x)
    n_enc = sum(p.numel() for p in m.encoder.parameters())
    n_dec = sum(p.numel() for p in m.parameters()) - n_enc
    print(f"input {tuple(x.shape)} -> loss {float(loss):.4f} pred {tuple(pred.shape)} mask {tuple(mask.shape)}")
    print(f"encoder params={n_enc:,}  decoder params={n_dec:,}  mask_ratio={m.mask_ratio}")