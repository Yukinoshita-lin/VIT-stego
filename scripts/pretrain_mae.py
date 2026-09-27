"""Pre-train the MAE encoder on BOSSbase covers (SRM-residual reconstruction).

Milestone 4: self-supervised pre-training. The MAE learns to reconstruct masked
patches of the 30-channel SRM high-pass residual of natural BOSSbase covers. No
labels are used — only covers. The resulting encoder (patch embed + transformer
blocks) is later transferred to LightViT and fine-tuned for cover/stego detection.

Covers are random 128x128 crops (with flip augmentation) drawn from the full
BOSSbase HD source, so each epoch sees a different view and the 2000 source
images provide ample diversity for a lightweight encoder (0.66M params).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.mae import MAE

BOSS_IMAGESET = r"F:\ood-robust-steganalysis\data\imagesets\imageset_bossbase_hd.npz"


class CoverCropDataset(Dataset):
    """Random 128x128 crops (+ flips) from the BOSSbase HD source covers."""

    def __init__(self, path: str, size: int = 128, seed: int = 0):
        self.d = np.load(path)
        # Read the stack once: indexing an NpzFile per item would re-decompress
        # the whole 524 MB member on every __getitem__.
        self.x = self.d["x"]
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.n = self.x.shape[0]

    def __len__(self) -> int:
        # One random crop per source image per epoch; the crop position and the
        # flips differ every epoch, so the same 2000 covers yield a fresh view
        # each pass. Raise --epochs rather than the length to see more crops.
        return self.n

    def __getitem__(self, i: int):
        img = self.x[i % self.n]
        h, w = img.shape
        top = self.rng.integers(0, h - self.size + 1)
        left = self.rng.integers(0, w - self.size + 1)
        crop = img[top : top + self.size, left : left + self.size]
        if self.rng.integers(0, 2):
            crop = crop[:, ::-1]  # horizontal flip
        if self.rng.integers(0, 2):
            crop = crop[::-1, :]  # vertical flip
        return torch.from_numpy(crop.astype(np.float32)).unsqueeze(0)  # (1, H, W) [0,255]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imageset", default=BOSS_IMAGESET)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--decoder-depth", type=int, default=1)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join("models", "mae_encoder_pretrained.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    ds = CoverCropDataset(args.imageset, args.size, args.seed)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    print(f"covers={len(ds)} (random crops of {args.size}x{args.size}), steps/epoch={len(loader)}")

    model = MAE(
        img_size=args.size, patch_size=args.patch, embed_dim=args.dim,
        depth=args.depth, heads=args.heads, decoder_depth=args.decoder_depth,
        mask_ratio=args.mask_ratio,
    ).to(device)
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.parameters()) - n_enc
    print(f"MAE encoder={n_enc:,} decoder={n_dec:,} mask_ratio={args.mask_ratio}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.wd)
    total_steps = args.epochs * len(loader)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * 0.5 * (1.0 + np.cos(np.pi * progress))

    t0 = time.time()
    step = 0
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x in loader:
            x = x.to(device)
            lr = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad()
            loss, _, _ = model(x)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            step += 1
        avg = running / (len(loader) * args.batch)
        if avg < best:
            best = avg
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save(model.encoder.state_dict(), args.out)
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch {epoch}/{args.epochs} | recon loss {avg:.1f} (best {best:.1f}) "
                f"| lr {lr:.2e} | {time.time()-t0:.1f}s"
            )

    print(f"done. best recon loss = {best:.1f}, encoder -> {args.out}")


if __name__ == "__main__":
    main()