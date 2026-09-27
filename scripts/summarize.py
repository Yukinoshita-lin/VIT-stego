"""Render the exp/*.json benchmark records as a markdown table.

Keeps README numbers recomputable from the raw records instead of hand-copied:

    python scripts/summarize.py                 # every result
    python scripts/summarize.py --filter 2000   # only tags containing "2000"
    python scripts/summarize.py --sort params   # cheapest first
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(exp_dir: str):
    rows = []
    for path in sorted(glob.glob(os.path.join(exp_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if "test_auc_mean" not in d:
            continue
        cfg = d.get("config", {})
        rows.append({
            "tag": d.get("tag") or os.path.splitext(os.path.basename(path))[0],
            "auc": d["test_auc_mean"],
            "std": d.get("test_auc_std", 0.0),
            "acc": d["test_acc_mean"],
            "params": d.get("params", 0),
            "seeds": d.get("n_seeds", len(d.get("rows", []))),
            "data": os.path.basename(cfg.get("data", "")),
            "srm": cfg.get("srm", False),
            "truncate": cfg.get("truncate", 0.0),
            "pool": cfg.get("pool", "cls"),
            "model": cfg.get("model", "vit"),
            "epochs": cfg.get("epochs", 0),
            # Older records stored the store_true negation instead of `augment`;
            # defaulting to True made that column silently always read "yes".
            "augment": cfg.get("augment", not cfg.get("no_augment", False)),
            "pretrained": bool(cfg.get("pretrained", "")),
        })
    return rows


def load_ood(exp_dir: str):
    """OOD records written by scripts/eval_ood.py (schema: rows + ckpt)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(exp_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if "test_auc_mean" in d or "rows" not in d:
            continue
        for r in d["rows"]:
            prov = r.get("method", "?")
            if r.get("rate") is not None:
                prov += f" @{r['rate']:g} {r.get('rate_unit') or 'bpp'}"
            if r.get("domain"):
                prov += f" [{r['domain']}" + (f" q{r['quality']}" if r.get("quality") else "") + "]"
            rows.append({
                "tag": d.get("tag") or os.path.splitext(os.path.basename(path))[0],
                "ckpt": os.path.basename(d.get("ckpt", "")),
                "dataset": os.path.basename(r["data"]),
                "provenance": prov,
                "auc": r["auc"], "acc": r["acc"], "n_eval": r["n_eval"],
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="exp")
    ap.add_argument("--filter", default="", help="substring the tag must contain")
    ap.add_argument("--sort", choices=["auc", "params"], default="auc")
    args = ap.parse_args()

    rows = load(args.exp_dir)
    if args.filter:
        rows = [r for r in rows if args.filter in r["tag"]]
    ood = load_ood(args.exp_dir)
    if args.filter:
        ood = [r for r in ood if args.filter in r["tag"]]
    if not rows and not ood:
        print("no benchmark records found")
        return 1

    if rows:
        _print_benchmarks(rows, args.sort)
    if ood:
        _print_ood(ood)
    return 0


def _print_benchmarks(rows, sort: str) -> None:
    rows.sort(key=lambda r: (-r["auc"] if sort == "auc" else r["params"]))

    print("| run | embedding | SRM | trunc | pool | aug | epochs | params | test AUC | test acc |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        data = r["data"].replace("bossbase_", "").replace("_256.npz", "").replace(".npz", "")
        print(f"| `{r['tag']}` | {data} | {'yes' if r['srm'] else 'no'} "
              f"| {r['truncate']:g} | {r['pool']} | {'yes' if r['augment'] else 'no'} "
              f"| {r['epochs']} | {r['params']:,} | **{r['auc']:.4f}** ± {r['std']:.4f} "
              f"| {r['acc']:.4f} |")
    print(f"\n{len(rows)} runs, sorted by {'test AUC' if sort == 'auc' else 'parameter count'}")


def _print_ood(ood) -> None:
    print("\n## OOD / transfer records (scripts/eval_ood.py)\n")
    print("| record | checkpoint | dataset | embedding | AUC | acc | n |")
    print("|---|---|---|---|---|---|---|")
    for r in ood:
        print(f"| `{r['tag']}` | {r['ckpt']} | {r['dataset']} | {r['provenance']} "
              f"| **{r['auc']:.4f}** | {r['acc']:.4f} | {r['n_eval']} |")
    print(f"\n{len(ood)} OOD rows")


if __name__ == "__main__":
    sys.exit(main())
