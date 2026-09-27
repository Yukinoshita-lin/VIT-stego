"""Evaluate a saved checkpoint on a cover/stego dataset and report AUC / accuracy.

Reproduces the same three-way split as train.py (seed, val-frac, test-frac), so
`--split test` scores exactly the samples train.py held out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import CoverStegoDataset, CoverStegoView
from src.engine import evaluate, make_splits
from src.vit_stego import count_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", choices=["vit", "srnet"], default="vit")
    ap.add_argument("--srm", action="store_true", help="checkpoint was trained with the SRM-30 stem")
    ap.add_argument("--input", choices=["pixels", "coeff"], default="pixels",
                    help="coeff = the checkpoint was trained on quantized-DCT maps")
    ap.add_argument("--truncate", type=float, default=0.0)
    ap.add_argument("--pool", choices=["cls", "mean"], default="cls")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--tta", action="store_true", help="average scores over the 4 flip variants")
    ap.add_argument("--dump-probs", default="", help="write per-sample (probs, labels) as .npz")
    args = ap.parse_args()

    # Truncation and pooling leave no trace in the state dict (truncation has no
    # parameters; the cls token exists either way), so evaluating a checkpoint
    # with the wrong ones silently reports the wrong AUC. train.py writes a
    # sidecar JSON next to each checkpoint — trust it for anything the caller did
    # not set explicitly.
    explicit = {a[2:].replace("-", "_") for a in sys.argv[1:] if a.startswith("--")}
    sidecar = os.path.splitext(args.ckpt)[0] + ".json"
    if os.path.exists(sidecar):
        with open(sidecar, encoding="utf-8") as f:
            cfg = json.load(f)
        keys = ("model", "srm", "truncate", "pool", "img_size", "patch", "dim",
                "depth", "heads", "val_frac", "test_frac", "seed", "input")
        applied = [k for k in keys if k in cfg and k not in explicit]
        for k in applied:
            setattr(args, k, cfg[k])
        if applied:
            print(f"[evaluate] restored {', '.join(applied)} from {sidecar}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.input == "coeff" and args.srm:
        raise SystemExit("--input coeff conflicts with --srm")
    ds = CoverStegoDataset(args.data, img_size=args.img_size, augment=False, input=args.input)
    train_idx, val_idx, test_idx = make_splits(
        ds.n_per_class, args.val_frac, args.test_frac, args.seed
    )
    idx = test_idx if args.split == "test" else val_idx
    loader = DataLoader(CoverStegoView(ds, idx, train=False), batch_size=args.batch, shuffle=False)

    if args.model == "vit":
        from src.vit_stego import LightViT
        model = LightViT(
            img_size=args.img_size, patch_size=args.patch, in_chans=1,
            embed_dim=args.dim, depth=args.depth, heads=args.heads,
            use_srm=args.srm, truncate=args.truncate, pool=args.pool,
        )
    else:
        from src.srnet import SRNet
        model = SRNet(img_size=args.img_size, use_srm=args.srm, truncate=args.truncate)
    model = model.to(device)
    state = torch.load(args.ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"warning: missing={list(missing)} unexpected={list(unexpected)}")
    model.eval()

    auc, acc, probs, labels = evaluate(model, loader, device, tta=args.tta)
    print(f"params={count_params(model):,} | {args.split} n={len(labels)} "
          f"| AUC={auc:.4f} | acc={acc:.4f}{' (tta)' if args.tta else ''}")

    if args.dump_probs:
        os.makedirs(os.path.dirname(args.dump_probs) or ".", exist_ok=True)
        np.savez(args.dump_probs, probs=probs, labels=labels)
        with open(os.path.splitext(args.dump_probs)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump({"ckpt": args.ckpt, "data": args.data, "split": args.split,
                       "auc": float(auc), "acc": float(acc), "tta": args.tta}, f, indent=2)
        print(f"probs -> {args.dump_probs}")


if __name__ == "__main__":
    main()
