"""OOD / weak-density / mismatched-embedder evaluation of a trained detector.

A detector trained on one embedding distribution (say HILL adaptive +-1 at
0.4 bpp) is scored, without retraining, on the held-out test split of several
other datasets: weaker payloads (0.2 / 0.1 bpp), mismatched embedders (STC,
LSB) and, once M8 data exists, the JPEG domain. Because every .npz is built
from the same BOSSbase covers in the same order, ``make_splits`` with the same
seed selects the *same source images* everywhere — the drop from row to row is
then attributable to the embedding change alone, not to a different split.

Model hyperparameters are restored from the checkpoint's sidecar JSON (see
evaluate.py) so the checkpoint is scored with the settings it was trained with.
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


def restore_config(args, explicit: set[str]) -> dict:
    """Fill unset hyperparameters from the checkpoint's sidecar JSON."""
    sidecar = os.path.splitext(args.ckpt)[0] + ".json"
    cfg = {}
    if os.path.exists(sidecar):
        with open(sidecar, encoding="utf-8") as f:
            cfg = json.load(f)
        keys = ("model", "srm", "truncate", "pool", "img_size", "patch", "dim",
                "depth", "heads", "val_frac", "test_frac", "seed", "input")
        applied = [k for k in keys if k in cfg and k not in explicit]
        for k in applied:
            setattr(args, k, cfg[k])
        if applied:
            print(f"[eval_ood] restored {', '.join(applied)} from {sidecar}")
    return cfg


def dataset_provenance(path: str) -> dict:
    d = np.load(path, allow_pickle=False)
    out = {}
    for k in ("method", "rate", "rate_unit", "domain", "quality"):
        if k in d.files:
            v = d[k]
            out[k] = v.item() if v.ndim == 0 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, nargs="+",
                    help="one or more .npz datasets; the first is typically the in-distribution one")
    ap.add_argument("--model", choices=["vit", "srnet"], default="vit")
    ap.add_argument("--srm", action="store_true")
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
    ap.add_argument("--split", choices=["test", "val", "all"], default="test")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--tag", default="", help="output record name -> exp/<tag>.json")
    args = ap.parse_args()

    explicit = {a[2:].replace("-", "_") for a in sys.argv[1:] if a.startswith("--")}
    sidecar_cfg = restore_config(args, explicit)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.input == "coeff" and args.srm:
        raise SystemExit("--input coeff conflicts with --srm")
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

    n_reference = None
    rows = []
    for path in args.data:
        ds = CoverStegoDataset(path, img_size=args.img_size, augment=False, input=args.input)
        if n_reference is None:
            n_reference = ds.n_per_class
        elif ds.n_per_class != n_reference:
            print(f"warning: {path} has {ds.n_per_class} covers, "
                  f"reference had {n_reference} — test indices may not pair up")

        if args.split == "all":
            idx = np.arange(2 * ds.n_per_class)
        else:
            _, val_idx, test_idx = make_splits(
                ds.n_per_class, args.val_frac, args.test_frac, args.seed
            )
            idx = test_idx if args.split == "test" else val_idx
        loader = DataLoader(CoverStegoView(ds, idx, train=False),
                            batch_size=args.batch, shuffle=False)

        auc, acc, _, _ = evaluate(model, loader, device, tta=args.tta)
        row = {"data": path, "n_eval": int(len(idx)),
               "auc": float(auc), "acc": float(acc), **dataset_provenance(path)}
        rows.append(row)
        prov = ", ".join(f"{k}={v}" for k, v in row.items()
                         if k in ("method", "rate", "rate_unit", "domain", "quality") and v is not None)
        print(f"{os.path.basename(path):36s} {prov:34s} AUC {auc:.4f} acc {acc:.4f}")

    tag = args.tag or (os.path.splitext(os.path.basename(args.ckpt))[0] + "_ood")
    os.makedirs("exp", exist_ok=True)
    out = os.path.join("exp", f"{tag}.json")
    record = {
        "tag": tag, "ckpt": args.ckpt, "split": args.split, "tta": args.tta,
        "seed": args.seed, "params": count_params(model),
        "sidecar_config": {k: v for k, v in sidecar_cfg.items()
                           if k in ("model", "srm", "truncate", "pool", "img_size",
                                    "patch", "dim", "depth", "heads", "input",
                                    "data", "epochs")},
        "rows": rows,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
