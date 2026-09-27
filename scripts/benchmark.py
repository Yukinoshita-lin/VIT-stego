"""Multi-seed benchmark: train + held-out test AUC with mean/std across seeds.

A single split on a 512-cover dataset puts ~+-0.05 of pure split noise on the
reported AUC, which is larger than most of the effects worth measuring. This
script re-splits and re-initialises for each seed and reports the distribution,
so a change in the mean can actually be told apart from noise.

    python scripts/benchmark.py --data data/bossbase_hill.npz --srm --truncate 4 \
        --seeds 0,1,2,3,4 --epochs 30 --tag hill_srm_trunc4

Results land in exp/<tag>.json (raw per-seed rows plus the aggregate).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.dataset import CoverStegoDataset
from src.engine import evaluate, make_loaders, set_seed, train_model
from src.mae import load_pretrained_encoder
from src.srnet import SRNet
from src.vit_stego import LightViT, count_params


def build_model(args):
    if args.model == "vit":
        return LightViT(
            img_size=args.img_size, patch_size=args.patch, in_chans=1,
            embed_dim=args.dim, depth=args.depth, heads=args.heads,
            dropout=args.dropout, use_srm=args.srm,
            truncate=args.truncate, pool=args.pool,
        )
    return SRNet(img_size=args.img_size, use_srm=args.srm, truncate=args.truncate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["vit", "srnet"], default="vit")
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pool", choices=["cls", "mean"], default="cls")
    ap.add_argument("--srm", action="store_true")
    ap.add_argument("--input", choices=["pixels", "coeff"], default="pixels",
                    help="pixels = decompressed image, coeff = quantized-DCT map (JPEG-domain npz)")
    ap.add_argument("--truncate", type=float, default=0.0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--pretrained", default="")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--quiet", action="store_true", help="per-epoch progress off")
    args = ap.parse_args()

    if args.input == "coeff" and args.srm:
        ap.error("--input coeff conflicts with --srm: the SRM residual stem is a "
                 "spatial-domain filter; JPEG quantization already high-passes the input")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CoverStegoDataset(args.data, img_size=args.img_size, augment=not args.no_augment,
                           input=args.input)

    rows = []
    t0 = time.time()
    for seed in seeds:
        set_seed(seed)
        train_loader, val_loader, test_loader = make_loaders(
            ds, args.val_frac, args.test_frac, seed, args.batch
        )
        model = build_model(args)
        if args.pretrained:
            model = load_pretrained_encoder(model, args.pretrained)
        model = model.to(device)

        result = train_model(model, train_loader, val_loader, device,
                             epochs=args.epochs, lr=args.lr,
                             weight_decay=args.weight_decay,
                             verbose=not args.quiet)
        test_auc, test_acc, _, _ = evaluate(model, test_loader, device, tta=args.tta)
        rows.append({"seed": seed, "test_auc": float(test_auc), "test_acc": float(test_acc),
                     "val_auc": result["best_auc"], "val_acc": result["best_acc"],
                     "best_epoch": result["best_epoch"]})
        print(f"[seed {seed}] test AUC {test_auc:.4f} | test acc {test_acc:.4f} "
              f"| val AUC {result['best_auc']:.4f} | best epoch {result['best_epoch']}")

    aucs = np.array([r["test_auc"] for r in rows])
    accs = np.array([r["test_acc"] for r in rows])
    # Std across seeds measures run-to-run variability (split + init), so the
    # standard error is what says whether two configs differ.
    se = aucs.std(ddof=1) / np.sqrt(len(aucs)) if len(aucs) > 1 else 0.0
    tag = args.tag or os.path.splitext(os.path.basename(args.data))[0]
    config = {k: v for k, v in vars(args).items()}
    # `--no-augment` is a store_true flag, so the recorded config would otherwise
    # carry the negation and downstream reports would render it as the opposite.
    config["augment"] = not args.no_augment
    summary = {
        "tag": tag,
        "config": config,
        "params": count_params(build_model(args)),
        "n_seeds": len(seeds),
        "test_auc_mean": float(aucs.mean()), "test_auc_std": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
        "test_auc_se": float(se),
        "test_acc_mean": float(accs.mean()),
        "test_auc_per_seed": [float(a) for a in aucs],
        "rows": rows,
        "seconds": time.time() - t0,
    }
    os.makedirs("exp", exist_ok=True)
    out = os.path.join("exp", f"{tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n== {tag} ==")
    print(f"test AUC {aucs.mean():.4f} +- {aucs.std(ddof=1) if len(aucs) > 1 else 0:.4f} "
          f"(se {se:.4f}) over {len(seeds)} seeds | test acc {accs.mean():.4f} "
          f"| params {summary['params']:,} | {summary['seconds']:.1f}s")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
