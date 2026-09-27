"""nsF5 — non-shrinkage F5 steganography in the JPEG DCT domain.

Fridrich, Pevny, Kodovsky (2007). The payload is embedded into the *nonzero*
AC coefficients of the quantized DCT (DC and zero coefficients are never
touched), one parity flip per needed change, with matrix embedding so a group
of carriers carries several message bits per at-most-one change.

Two properties make this "non-shrinkage" and are load-bearing for the
round trip:

* Changes never create a new zero and never touch an existing zero. A flip
  therefore cannot move a coefficient across zero (|c| == 1 moves away from
  zero, |c| >= 2 moves toward zero), so the receiver — who only has the stego
  file — identifies the very same carrier set the embedder used, simply by
  reading the nonzero AC coefficients of the stego.
* A parity flip is exactly one change in the quantized coefficient domain
  (|delta| = 1), which is what the decompressed image the detector sees
  inherits.

Matrix embedding uses Hamming syndromes: for code height h, groups of
n = 2^h - 1 carriers carry h bits each as the syndrome H·c (column i of H is
the binary representation of i+1), fixed with at most one parity flip. h is
chosen as the largest value whose capacity fits the payload, i.e. the highest
embedding efficiency (bits per change) the carrier budget allows — the same
adaptive rule original F5 uses. At high rates this degrades gracefully to the
plain (1,1,1) code: one bit per carrier.
"""
from __future__ import annotations

import numpy as np

try:
    from .jpeg import ZIGZAG
except ImportError:  # run as a plain script: python stego/nsf5.py
    from jpeg import ZIGZAG

MAX_CODE_HEIGHT = 12


def _zigzag_flat(coeffs_q: np.ndarray) -> np.ndarray:
    """(n_blocks, 8, 8) -> (n_blocks, 64) zigzag-ordered (DC first)."""
    return coeffs_q[:, ZIGZAG[:, 0], ZIGZAG[:, 1]]


def _zigzag_unflatten(flat: np.ndarray) -> np.ndarray:
    out = np.empty((flat.shape[0], 8, 8), dtype=flat.dtype)
    out[:, ZIGZAG[:, 0], ZIGZAG[:, 1]] = flat
    return out


def _carrier_mask(coeffs_q: np.ndarray) -> np.ndarray:
    """Zigzag-ordered boolean (n_blocks, 64): nonzero AC coefficients."""
    mask = _zigzag_flat(coeffs_q) != 0
    mask[:, 0] = False  # DC first in zigzag order
    return mask


def _apply_flip(c: np.ndarray) -> np.ndarray:
    """Flip the parity of coefficient(s) c without creating a zero.

    |c| >= 2 moves toward zero (the F5-style decrement that shapes the nsF5
    histogram fingerprint); |c| == 1 must move away from zero.
    """
    return np.where(c >= 2, c - 1, np.where(c <= -2, c + 1, c + np.sign(c).astype(c.dtype)))


def plan_code_height(n_carriers: int, m: int) -> int:
    """Largest Hamming height h whose capacity floor(n_c/(2^h-1))*h >= m.

    h = 1 is the plain (1,1,1) code (one bit per carrier); it is always
    feasible when m <= n_carriers at all.
    """
    if m <= 0:
        return 1
    for h in range(MAX_CODE_HEIGHT, 0, -1):
        n = (1 << h) - 1
        if (n_carriers // n) * h >= m:
            return h
    raise ValueError(
        f"payload {m} bits exceeds nsF5 capacity {n_carriers} nonzero AC coefficients"
    )


def _syndromes(groups_lsb: np.ndarray, h: int) -> np.ndarray:
    """Syndrome of every carrier group: (n_groups, n) LSBs -> (n_groups, h) bits.

    Column i of H is the binary representation of i+1 (bit 0 = LSB), so
    syndrome = XOR over carriers of lsb_i * binary(i+1).
    """
    n = (1 << h) - 1
    idx = np.arange(1, n + 1, dtype=np.int64)
    cols = ((idx[:, None] >> np.arange(h)[None, :]) & 1).astype(np.uint8)  # (n, h)
    return (groups_lsb[:, :, None] * cols[None, :, :]).sum(axis=1) & 1


def embed_nsf5_payload(coeffs_q: np.ndarray, msg: np.ndarray) -> tuple[np.ndarray, dict]:
    """Embed ``msg`` (uint8 {0,1}) into quantized DCT coefficients.

    Returns (stego_coeffs, info) with ``info`` reporting the code height,
    carrier count and number of changed coefficients. The carrier sequence is
    the nonzero AC coefficients in JPEG order (blocks raster, zigzag within
    block); the payload is padded to whole syndrome chunks with zero bits.
    """
    coeffs_q = np.asarray(coeffs_q, dtype=np.int16)
    msg = np.asarray(msg, dtype=np.uint8).ravel()
    m = int(msg.size)
    mask = _carrier_mask(coeffs_q)
    carriers = _zigzag_flat(coeffs_q)[mask]
    n_carriers = int(carriers.size)

    h = plan_code_height(n_carriers, m)
    n = (1 << h) - 1
    num_groups = (m + h - 1) // h
    msg_padded = np.zeros(num_groups * h, dtype=np.uint8)
    msg_padded[:m] = msg

    # Syndromes of all groups at once; diff != 0 names (LSB-first binary, the
    # same convention as the H columns) the one carrier of the group whose
    # parity must flip: its column equals diff because col(j) = binary(j+1).
    groups = carriers[: num_groups * n].reshape(num_groups, n)
    syndrome = _syndromes((groups & 1).astype(np.uint8), h)
    diff = syndrome ^ msg_padded.reshape(num_groups, h)
    pos = (diff * (1 << np.arange(h))).sum(axis=1).astype(np.int64)  # (n_groups,)

    flip_group = np.nonzero(pos)[0]
    flip_idx = flip_group * n + (pos[flip_group] - 1)
    stego_carriers = carriers.copy()
    stego_carriers[flip_idx] = _apply_flip(carriers[flip_idx])

    stego_flat = _zigzag_flat(coeffs_q).copy()
    stego_flat[mask] = stego_carriers
    stego = _zigzag_unflatten(stego_flat)
    info = {"code_height": h, "n_carriers": n_carriers, "n_changed": int(flip_group.size),
            "m": m}
    return stego, info


def extract_nsf5_payload(coeffs_q: np.ndarray, m: int) -> np.ndarray:
    """Read the first ``m`` payload bits back out of stego coefficients.

    The carrier set, code height and grouping are re-derived exactly as the
    embedder chose them — from the stego's own nonzero AC coefficients and the
    payload length, both known to the receiver.
    """
    coeffs_q = np.asarray(coeffs_q, dtype=np.int16)
    m = int(m)
    if m == 0:
        return np.zeros(0, dtype=np.uint8)
    mask = _carrier_mask(coeffs_q)
    carriers = _zigzag_flat(coeffs_q)[mask]
    h = plan_code_height(carriers.size, m)
    n = (1 << h) - 1
    num_groups = (m + h - 1) // h
    if num_groups * n > carriers.size:
        raise ValueError("stego carrier set too small for the payload")
    groups = carriers[: num_groups * n].reshape(num_groups, n)
    return _syndromes((groups & 1).astype(np.uint8), h).ravel()[:m]


def embed_nsf5(coeffs_q: np.ndarray, rate: float, rng: np.random.Generator,
               rate_unit: str = "bpp") -> tuple[np.ndarray, dict]:
    """Embed a random payload at ``rate`` (dataset-facing API).

    ``rate_unit`` selects the payload convention: "bpp" — bits per *pixel*
    (m = rate * H * W, matching the spatial embedders), or "bpnc" — bits per
    *nonzero AC coefficient* (m = rate * n_carriers), the convention the nsF5
    literature quotes because the carrier count varies per image and no fixed
    bpp fits every cover.
    """
    coeffs_q = np.asarray(coeffs_q, dtype=np.int16)
    if rate_unit == "bpp":
        m = int(round(rate * coeffs_q.shape[0] * 64))
    elif rate_unit == "bpnc":
        mask = _carrier_mask(coeffs_q)
        m = int(round(rate * int(mask.sum())))
    else:
        raise ValueError(f"rate_unit must be 'bpp' or 'bpnc', got {rate_unit}")
    msg = rng.integers(0, 2, m, dtype=np.uint8)
    stego, info = embed_nsf5_payload(coeffs_q, msg)
    info.update({"rate": rate, "rate_unit": rate_unit})
    return stego, info


def _self_test():
    try:
        from .jpeg import jpeg_compress, jpeg_decompress
    except ImportError:
        from jpeg import jpeg_compress, jpeg_decompress

    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(128, 45, (128, 128)), 0, 255).astype(np.uint8)

    for quality, rate in ((85, 0.4), (75, 0.4), (85, 0.1), (95, 0.5)):
        cover_c = jpeg_compress(img, quality)
        stego_c, info = embed_nsf5(cover_c, rate, np.random.default_rng(1))

        # payload survives (JPEG coefficients are unbounded -> exact recovery)
        m = int(round(rate * cover_c.size))
        got = extract_nsf5_payload(stego_c, m)
        replay = np.random.default_rng(1)
        assert np.array_equal(replay.integers(0, 2, m, dtype=np.uint8), got), \
            "payload did not survive the round trip"

        # non-shrinkage: the carrier set is identical from the stego side
        ac = np.ones((8, 8), dtype=bool)
        ac[0, 0] = False
        cover_nz = (cover_c != 0) & ac
        stego_nz = (stego_c != 0) & ac
        assert np.array_equal(cover_nz, stego_nz), "carrier set changed (shrinkage!)"
        delta = stego_c.astype(np.int16) - cover_c.astype(np.int16)
        assert np.abs(delta).max() <= 1, "coefficient moved by more than 1"
        assert np.all(delta[~cover_nz] == 0), "zero/DC coefficient was modified"
        print(f"  Q{quality} rate={rate}: h={info['code_height']} "
              f"carriers={info['n_carriers']} changes={info['n_changed']} "
              f"({info['n_changed'] / max(1, cover_nz.sum()):.3f}/carrier), round trip OK")

    # adaptive code height: lower rate -> higher h (more bits per change)
    c = jpeg_compress(img, 85)
    _, hi = embed_nsf5(c, 0.05, np.random.default_rng(0))
    _, lo = embed_nsf5(c, 0.4, np.random.default_rng(0))
    assert hi["code_height"] >= lo["code_height"], \
        "code height should not increase with payload rate"
    print(f"  adaptive height: rate 0.05 -> h={hi['code_height']}, rate 0.4 -> h={lo['code_height']}")

    # capacity violation raises
    try:
        embed_nsf5(c, 5.0, np.random.default_rng(0))
        raise AssertionError("over-capacity embed should have failed")
    except ValueError:
        print("  over-capacity payload raises ValueError")

    # decompressed stego is a valid image that differs from the decompressed cover
    stego_c, _ = embed_nsf5(c, 0.3, np.random.default_rng(2))
    cov_px = jpeg_decompress(c, 85)
    stg_px = jpeg_decompress(stego_c, 85)
    assert stg_px.shape == cov_px.shape and stg_px.dtype == np.uint8
    n_px_changed = int((stg_px != cov_px).sum())
    print(f"  decompression: {n_px_changed}/{cov_px.size} pixels differ at 0.3 bpp")
    assert 0 < n_px_changed, "embedding had no spatial effect"
    print("nsF5 self-test PASSED")


if __name__ == "__main__":
    _self_test()
