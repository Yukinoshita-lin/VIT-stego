"""Train a lightweight ViT steganalysis detector and report held-out (test) AUC.

Splits are three-way: `val` selects the best epoch, `test` is scored once
afterwards and is the number to quote. See src/engine.py for why.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    ap.add_argument("--data", default=os.path.join("data", "bossbase_hill.npz"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pool", choices=["cls", "mean"], default="cls")
    ap.add_argument("--srm", action="store_true", help="prepend fixed SRM-30 residual stem")
    ap.add_argument("--input", choices=["pixels", "coeff"], default="pixels",
                    help="pixels = decompressed image (spatial npz or JPEG-domain npz), "
                         "coeff = quantized-DCT map stored by JPEG-domain prepare_data")
    ap.add_argument("--truncate", type=float, default=0.0,
                    help="clamp SRM residuals to +-T (0 = off); needs --srm")
    ap.add_argument("--no-augment", action="store_true",
                    help="disable random crop + flip augmentation")
    ap.add_argument("--pretrained", default="", help="path to MAE encoder checkpoint to initialize from")
    ap.add_argument("--tta", action="store_true", help="average test scores over the 4 flip variants")
    ap.add_argument("--out", default=os.path.join("models", "lightvit_best.pt"))
    args = ap.parse_args()

    if args.input == "coeff" and args.srm:
        ap.error("--input coeff conflicts with --srm: the SRM residual stem is a "
                 "spatial-domain filter; JPEG quantization already high-passes the input")

    set_seed(args.seed)  # before the model is built — see engine.set_seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    ds = CoverStegoDataset(args.data, img_size=args.img_size, augment=not args.no_augment,
                           input=args.input)
    train_loader, val_loader, test_loader = make_loaders(
        ds, args.val_frac, args.test_frac, args.seed, args.batch
    )
    print(f"stored crop {ds.stored_size} -> {args.img_size}; "
          f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"test={len(test_loader.dataset)}")

    model = build_model(args)
    if args.pretrained:
        if not args.srm:
            print("warning: --pretrained expects an SRM-stem model (MAE was trained on 30-ch residuals)")
        model = load_pretrained_encoder(model, args.pretrained)
    model = model.to(device)
    print(f"model={args.model} srm={args.srm} truncate={args.truncate} pool={args.pool} "
          f"pretrained={bool(args.pretrained)} params={count_params(model):,}")

    result = train_model(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        ckpt=args.out,
    )

    test_auc, test_acc, _, _ = evaluate(model, test_loader, device, tta=args.tta)
    print(f"done. best val AUC = {result['best_auc']:.4f} (epoch {result['best_epoch']}) "
          f"| TEST AUC = {test_auc:.4f} acc = {test_acc:.4f}"
          f"{' (tta)' if args.tta else ''} | model -> {args.out}")

    summary = {
        "model": args.model, "data": args.data, "params": count_params(model),
        "srm": args.srm, "truncate": args.truncate, "pool": args.pool,
        "input": args.input,
        "img_size": args.img_size, "patch": args.patch, "dim": args.dim,
        "depth": args.depth, "heads": args.heads, "dropout": args.dropout,
        "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
        "augment": not args.no_augment, "pretrained": args.pretrained,
        "val_auc": result["best_auc"], "val_acc": result["best_acc"],
        "best_epoch": result["best_epoch"],
        "test_auc": float(test_auc), "test_acc": float(test_acc), "tta": args.tta,
        "seconds": result["seconds"],
    }
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
