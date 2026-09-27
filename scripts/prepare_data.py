"""Build a cover/stego .npz for the ViT steganalysis baseline.

Covers come from the BOSSbase HD imageset of the sibling ood-robust-steganalysis
project. Five embedders are available:

spatial domain (cover = the BOSSbase crop itself):
  - adaptive   HILL-driven content-adaptive +-1 (stego/hill.py),
  - stc        HILL + syndrome-trellis code, +-1 LSB matching (stego/stc.py),
  - lsb        classic LSB replacement sanity baseline;

JPEG domain (cover = the decompressed JPEG of the crop, quality --quality;
  the detector sees exactly what a deployed detector sees):
  - nsf5       non-shrinkage F5 over nonzero AC coefficients (stego/nsf5.py),
  - juniward   J-UNIWARD cost + spanned STC over all ACs (stego/juniward_jpeg.py).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from stego.hill import hill_cost
from stego.jpeg import jpeg_compress, jpeg_decompress
from stego.juniward_jpeg import embed_juniward
from stego.nsf5 import embed_nsf5
from stego.stc import apply_lsb_matching, embed_stc

BOSS_IMAGESET = r"F:\ood-robust-steganalysis\data\imagesets\imageset_bossbase_hd.npz"

# Quantized-DCT coefficient maps are stored for JPEG-domain datasets so a
# detector can work on the representation a real deployment would see (the
# JPEG file's coefficients). DC carries no payload and is zeroed; ACs are
# clipped to +-COEFF_CLIP to bound content dominance over the +-1 embedding
# signal (the coefficient-domain analogue of the SRM residual truncation).
COEFF_CLIP = 30


def coeff_map(coeffs_q: np.ndarray) -> np.ndarray:
    """(n_blocks, 8, 8) quantized coefficients -> (H, W) int16 map, DC zeroed,
    clipped to +-COEFF_CLIP. Layout: block (by, bx) at (by*8, bx*8)."""
    n = coeffs_q.shape[0]
    side = int(round(np.sqrt(n)))
    assert side * side == n
    c = (coeffs_q.reshape(side, 8, side, 8)
                  .transpose(0, 2, 1, 3)
                  .reshape(side * 8, side * 8)
                  .copy())
    c[::8, ::8] = 0  # DC of every block (ku == kv == 0)
    return np.clip(c, -COEFF_CLIP, COEFF_CLIP).astype(np.int16)


def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape
    top = (h - size) // 2
    left = (w - size) // 2
    return img[top : top + size, left : left + size]


def embed_adaptive(cover: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Content-adaptive +-1 embedding.

    Pixel-change propensity is proportional to 1/(1+rho) (HILL cost); low-cost
    (noisy/textured) pixels are modified preferentially, high-cost (smooth)
    pixels are avoided. `rate` is the target change rate in bits per pixel.
    """
    flat = cover.astype(np.int16).reshape(-1)
    n = flat.size
    rho = hill_cost(cover).reshape(-1).astype(np.float64)
    p = 1.0 / (1.0 + rho)
    p /= p.sum()
    n_change = int(round(rate * n))
    idx = rng.choice(n, size=n_change, replace=False, p=p)
    delta = rng.integers(0, 2, n_change, dtype=np.int8) * 2 - 1  # +-1
    stego = flat.copy()
    stego[idx] = np.clip(stego[idx] + delta, 0, 255)
    return stego.astype(np.uint8).reshape(cover.shape)


def embed_lsb(cover: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """LSB replacement on NATURAL covers: randomize the LSB of a `rate` fraction
    of pixels. This produces the classic histogram-equalization fingerprint and is
    the standard sanity baseline every steganalyzer must first detect.
    """
    flat = cover.reshape(-1).copy()
    n = flat.size
    n_change = int(round(rate * n))
    idx = rng.choice(n, size=n_change, replace=False)
    flat[idx] = (flat[idx] & 0xFE) | rng.integers(0, 2, n_change, dtype=np.uint8)
    return flat.reshape(cover.shape)


def embed_stc_hill(cover: np.ndarray, rate: float, rng: np.random.Generator,
                   h: int = 10, xi: float = 1e3, seed: int = 0) -> np.ndarray:
    """Proper HILL + STC embedding (Filler, size-general).

    Computes the HILL per-pixel cost, embeds ``rate`` bpp of pseudo-random payload
    with a height-h syndrome-trellis code over a scrambled pixel order (so the
    message is not front-loaded), then applies LSB-matching +/-1 modification
    (even -> +1, odd -> -1).
    """
    flat = cover.astype(np.uint8).ravel()
    n = flat.size
    rho = hill_cost(cover, xi=xi).ravel()
    m = int(round(rate * n))
    assert m <= n - h + 1, f"payload {m} bits exceeds STC capacity {n - h + 1} for h={h}"
    msg = rng.integers(0, 2, m, dtype=np.uint8)
    perm = rng.permutation(n)
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    n_active = m + h - 1
    flip_scrambled = np.zeros(n, dtype=np.uint8)
    flip_scrambled[:n_active] = embed_stc(
        (flat[perm][:n_active]) & 1, msg, rho[perm][:n_active], h=h, seed=seed
    )
    return apply_lsb_matching(flat, flip_scrambled[inv]).reshape(cover.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--n", type=int, default=512, help="number of covers to use")
    ap.add_argument("--rate", type=float, default=0.4, help="payload rate (bpp, or bpnc for nsf5)")
    ap.add_argument("--rate-unit", choices=["bpp", "bpnc"], default="bpp",
                    help="payload unit: bpp over pixels, or bpnc over nonzero ACs (nsf5 default in the literature)")
    ap.add_argument("--method", choices=["adaptive", "lsb", "stc", "nsf5", "juniward"], default="adaptive")
    ap.add_argument("--h", type=int, default=10, help="STC constraint height (stc/juniward only)")
    ap.add_argument("--quality", type=int, default=75, help="JPEG quality (nsf5/juniward only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join("data", "bossbase_hill.npz"))
    ap.add_argument("--imageset", default=BOSS_IMAGESET)
    args = ap.parse_args()

    jpeg_domain = args.method in ("nsf5", "juniward")
    if not jpeg_domain and args.quality != 75:
        print("note: --quality only applies to JPEG-domain methods (nsf5/juniward)")

    rng = np.random.default_rng(args.seed)
    d = np.load(args.imageset)
    # Decompress the image stack once. Indexing an NpzFile inside the loop
    # (`d["x"][i]`) re-reads and re-decompresses the *whole* member on every
    # iteration — with a 524 MB imageset that turns a 1-minute job into hours.
    x = d["x"]
    n_avail = x.shape[0]
    n = min(args.n, n_avail)
    print(f"source: {n_avail} images of {x.shape[1]}x{x.shape[2]}, using {n}")

    covers = np.empty((n, args.size, args.size), dtype=np.uint8)
    stegos = np.empty((n, args.size, args.size), dtype=np.uint8)
    covers_c = stegos_c = None
    if jpeg_domain:
        covers_c = np.empty((n, args.size, args.size), dtype=np.int16)
        stegos_c = np.empty((n, args.size, args.size), dtype=np.int16)
    stats = []
    for i in range(n):
        cover = center_crop(x[i], args.size)
        if jpeg_domain:
            cover_c = jpeg_compress(cover, args.quality)
            if args.method == "nsf5":
                stego_c, info = embed_nsf5(cover_c, args.rate, rng, rate_unit=args.rate_unit)
            else:
                stego_c, info = embed_juniward(cover_c, args.quality, args.rate, rng, h=args.h)
            covers[i] = jpeg_decompress(cover_c, args.quality)
            stegos[i] = jpeg_decompress(stego_c, args.quality)
            covers_c[i] = coeff_map(cover_c)
            stegos_c[i] = coeff_map(stego_c)
            stats.append(info)
        elif args.method == "lsb":
            covers[i] = cover
            stegos[i] = embed_lsb(cover, args.rate, rng)
        elif args.method == "stc":
            covers[i] = cover
            stegos[i] = embed_stc_hill(cover, args.rate, rng, h=args.h)
        else:
            covers[i] = cover
            stegos[i] = embed_adaptive(cover, args.rate, rng)
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{n}] embedded", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # `method`/`rate` ride along so the dataset can report its own provenance
    # (and so a .npz can never be silently paired with the wrong label in a table).
    meta = dict(method=np.array(args.method), rate=np.float64(args.rate),
                rate_unit=np.array(args.rate_unit), domain=np.array("jpeg" if jpeg_domain else "spatial"))
    if jpeg_domain:
        meta["quality"] = np.int64(args.quality)
        meta["coeff_clip"] = np.int64(COEFF_CLIP)
        meta["n_changed"] = np.array([s["n_changed"] for s in stats], dtype=np.int64)
        if args.method == "nsf5":
            meta["code_height"] = np.array([s["code_height"] for s in stats], dtype=np.int64)
    payload = dict(covers=covers, stegos=stegos, **meta)
    if jpeg_domain:
        payload["covers_c"] = covers_c
        payload["stegos_c"] = stegos_c
    np.savez_compressed(args.out, **payload)
    if jpeg_domain and stats:
        changed = np.array([s["n_changed"] for s in stats], dtype=np.int64)
        print(f"mean change rate: {changed.mean() / covers[0].size:.4f} of coefficients")
    print(f"saved {args.out}: {n} covers + {n} stegos @ {args.size}x{args.size}, "
          f"method={args.method}, rate={args.rate} {args.rate_unit}"
          + (f", quality={args.quality}" if jpeg_domain else ""))


if __name__ == "__main__":
    main()