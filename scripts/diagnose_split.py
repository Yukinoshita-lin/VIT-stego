"""Show what the legacy split did to the reported numbers, and why.

The split used through milestone M3 permuted the cover and stego halves
*independently*::

    cover_idx = rng.permutation(n); val_covers = cover_idx[:n_val]
    stego_idx = n + rng.permutation(n); val_stegos = stego_idx[:n_val]

A cover and its stego differ in ~40% of pixels by +-1, so image i's two halves
are near-duplicates carrying opposite labels. Under independent permutation the
twin of a validation image sits in the training set with probability
~(1 - val_frac) — here 0.73. The network learns image i as "stego" from its stego
copy and then meets image i again as a validation *cover*.

The damage is worse than a biased number: validation AUC never rises above chance
on that set, so best-epoch selection keeps an essentially untrained checkpoint.
This script trains one model on the legacy training set and tracks, every epoch,
AUC on (a) the legacy val set and (b) a balanced set of source images that never
appear in training. Same model, same weights, same crops — only the evaluation
set differs.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import CoverStegoDataset, CoverStegoView
from src.engine import evaluate, set_seed
from src.vit_stego import LightViT


def legacy_split(n: int, val_frac: float, seed: int):
    """The M0-M3 split: cover and stego halves permuted independently."""
    rng = np.random.default_rng(seed)
    n_val = int(round(n * val_frac))
    cover_idx = rng.permutation(n)
    stego_idx = n + rng.permutation(n)
    val = np.concatenate([cover_idx[:n_val], stego_idx[:n_val]])
    train = np.concatenate([cover_idx[n_val:], stego_idx[n_val:]])
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "bossbase_hill.npz"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--no-srm", action="store_true")
    ap.add_argument("--img-size", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CoverStegoDataset(args.data, img_size=args.img_size, augment=False)
    n = ds.n_per_class

    leg_train, leg_val = legacy_split(n, args.val_frac, args.seed)

    # Which source images the legacy training set has seen. Index i is the cover
    # of source i when i < n, and the stego of source i-n otherwise.
    train_ids = set(leg_train.tolist())
    train_sources = {i if i < n else i - n for i in train_ids}
    n_val_per_class = len(leg_val) // 2
    unseen_sources = [i for i in range(n) if i not in train_sources][:n_val_per_class]
    unseen = np.concatenate([unseen_sources, n + np.array(unseen_sources)])  # balanced

    twin_leak = np.mean([(i + n if i < n else i - n) in train_ids for i in leg_val])
    print(f"legacy val n={len(leg_val)} | twin-in-train fraction = {twin_leak:.3f}")
    print(f"disjoint val n={len(unseen)} (source images never seen in training, balanced)\n")

    def loader(idx, shuffle=False):
        return DataLoader(CoverStegoView(ds, idx, train=shuffle),
                          batch_size=64, shuffle=shuffle)

    set_seed(args.seed)
    model = LightViT(img_size=args.img_size, patch_size=8, in_chans=1,
                     embed_dim=96, depth=4, heads=3,
                     use_srm=not args.no_srm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    train_loader = loader(leg_train, shuffle=True)
    leg_loader, unseen_loader = loader(leg_val), loader(unseen)

    leg_curve, unseen_curve = [], []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        sched.step()
        a_leg, _, _, _ = evaluate(model, leg_loader, device)
        a_un, _, _, _ = evaluate(model, unseen_loader, device)
        leg_curve.append(a_leg)
        unseen_curve.append(a_un)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:>2} | legacy-val AUC {a_leg:.4f} | disjoint-val AUC {a_un:.4f}")

    print(f"\nbest legacy-val  AUC {max(leg_curve):.4f} at epoch {int(np.argmax(leg_curve)) + 1}")
    print(f"best disjoint-val AUC {max(unseen_curve):.4f} at epoch {int(np.argmax(unseen_curve)) + 1}")
    print(f"epoch-{args.epochs} (unselected) disjoint-val AUC {unseen_curve[-1]:.4f}")
    print("\nThe legacy val set returns ~chance for every epoch, so best-epoch selection there")
    print("keeps an untrained model — which is what the M0-M3 checkpoints were selected with.")


if __name__ == "__main__":
    main()
