"""J-UNIWARD — content-adaptive JPEG steganography (Holub, Denemark, Fridrich 2014).

The payload rides an STC over the LSBs of *all* AC coefficients of the
quantized DCT (63 per block, JPEG zigzag order, blocks raster — the receiver
reads the same deterministic sequence straight out of the stego coefficients,
so there is no carrier-set ambiguity). Each STC flip is a +-1 change of one
quantized coefficient.

What makes it content-adaptive is the cost. UNIWARD measures how much a
single-coefficient modification changes the wavelet magnitudes of the
*decompressed* image,

    rho_k = sum_dir sum_p ||W_dir(x + d_k)| - |W_dir(x)||_p
                        + ||W_dir(x - d_k)| - |W_dir(x)||_p

with W a 1-level Daubechies-8 DWT and d_k the spatial delta of changing
quantized coefficient k by +-1 — deterministic and block-local: the
dequantized basis image q_k * D^T E_k D. Because W is linear,
W(x +- d_k) = W(x) +- W(d_k), so the subbands of x are computed once and each
coefficient only contributes a small response window W(d_k) whose *shape* is
the same for every block. The whole cost map is therefore one batched
window-gather per coefficient position, not 2 x 63 JPEG decompressions.

The Daubechies-8 scaling filter is reconstructed by spectral factorization of
the Daubechies polynomial (no pywt needed):
    |H(w)|^2 = cos^16(w/2) * sum_{k=0}^{7} C(7+k, k) sin^{2k}(w/2),
taking the p roots at z = -1 plus the minimum-phase root of each reciprocal
pair of the remaining factor.
"""
from __future__ import annotations

from math import comb

import numpy as np

try:
    from .jpeg import ZIGZAG, dct_basis, jpeg_compress, jpeg_decompress, quality_table
    from .stc import embed_stc_spanned, extract_stc_spanned
except ImportError:  # run as a plain script: python stego/juniward_jpeg.py
    from jpeg import ZIGZAG, dct_basis, jpeg_compress, jpeg_decompress, quality_table
    from stc import embed_stc_spanned, extract_stc_spanned

P_VANISH = 8   # db8: 8 vanishing moments
PAD = 32       # symmetric padding around the decompressed cover
CANVAS = 64    # side of the response canvas (multiple of 8, wrap-free)
CANVAS_POS = 24  # pattern origin inside the canvas (multiple of 8)
SPREAD_RATE = 0.5  # target STC relative payload over the active window

_DB8_LO: np.ndarray | None = None


def _poly_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Add two highest-power-first coefficient vectors."""
    n = max(len(a), len(b))
    out = np.zeros(n)
    out[n - len(a):] += a
    out[n - len(b):] += b
    return out


def daubechies_scaling_filter(p: int) -> np.ndarray:
    """2p-tap orthonormal Daubechies-p scaling filter, L2-normalised, sum = +sqrt(2).

    Spectral factorization of the Daubechies polynomial
        |H(w)|^2 = cos^{2p}(w/2) * sum_{k=0}^{p-1} C(p-1+k, k) sin^{2k}(w/2):
    with z = e^{iw}, the squared magnitude is W(z)/(4z)^{2p-1} with
    W = (z+1)^{2p} * D, D(z) = sum_k C(p-1+k,k) 4^{p-1-k} v(z)^k z^{p-1-k},
    v = -(z-1)^2. D is self-reciprocal of degree 2p-2 (p-1 reciprocal root
    pairs); the scaling filter takes the p roots at z = -1 plus the
    minimum-phase root of each pair.
    """
    bernstein = np.array([comb(p - 1 + k, k) for k in range(p)], dtype=np.float64)

    v = np.array([-1.0, 2.0, -1.0])
    d = np.zeros(2 * p - 1)
    vk = np.array([1.0])
    for k in range(p):
        if k:
            vk = np.convolve(vk, v)
        d = _poly_add(d, np.concatenate([bernstein[k] * 4.0 ** (p - 1 - k) * vk,
                                         np.zeros(p - 1 - k)]))

    roots = np.roots(d)
    inside = roots[np.abs(roots) < 1.0]
    assert inside.size == p - 1, (
        f"expected {p - 1} minimum-phase roots of the Daubechies polynomial, got {inside.size}")

    q = np.array([1.0])
    for _ in range(p):  # p roots at z = -1
        q = np.convolve(q, [1.0, 1.0])
    q = np.convolve(q, np.poly(inside))  # degree p + (p-1) = 2p-1

    h = q[::-1].copy()  # ascending-time filter
    h /= np.linalg.norm(h)
    if h.sum() < 0:
        h = -h
    return h


def db8_scaling_filter() -> np.ndarray:
    """16-tap Daubechies-8 scaling filter, L2-normalised with sum = +sqrt(2)."""
    global _DB8_LO
    if _DB8_LO is None:
        _DB8_LO = daubechies_scaling_filter(P_VANISH)
    return _DB8_LO


def db8_filters() -> tuple[np.ndarray, np.ndarray]:
    """(scaling, wavelet) pair: g_k = (-1)^k h_{L-1-k}."""
    global _DB8_LO
    if _DB8_LO is None:
        _DB8_LO = db8_scaling_filter()
    h = _DB8_LO
    g = (-1.0) ** np.arange(h.size) * h[::-1]
    return h, g


def _circular_convolve(x: np.ndarray, kern: np.ndarray, axis: int) -> np.ndarray:
    """Periodized 1-D convolution along ``axis`` (length preserving)."""
    out = np.zeros_like(x)
    for i, k in enumerate(kern):
        if k != 0.0:
            out += k * np.roll(x, i, axis=axis)
    return out


def dwt_details_db8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-level periodized DWT detail subbands; returns (LH, HL, HH).

    Only the sum of the three |subband| fields enters the cost, so the
    horizontal/vertical naming convention is irrelevant as long as it is
    internally consistent between the image and the response canvases.
    """
    lo, hi = db8_filters()
    l_rows = _circular_convolve(x, lo, axis=1)[:, ::2]
    h_rows = _circular_convolve(x, hi, axis=1)[:, ::2]
    lh = _circular_convolve(l_rows, hi, axis=0)[::2, :]
    hl = _circular_convolve(h_rows, lo, axis=0)[::2, :]
    hh = _circular_convolve(h_rows, hi, axis=0)[::2, :]
    return lh, hl, hh


def _response_window() -> tuple[int, int, int, int]:
    """Subband window (r0, r1, c0, c1) holding the full response of one block.

    The support of W(d_k) around a block depends only on the pattern support
    (the whole 8x8 block) and the filter length, never on k, so one window
    serves every coefficient and every block.
    """
    canvas = np.zeros((CANVAS, CANVAS))
    canvas[CANVAS_POS:CANVAS_POS + 8, CANVAS_POS:CANVAS_POS + 8] = 1.0
    lh, hl, hh = dwt_details_db8(canvas)
    rows, cols = np.nonzero((lh + hl + hh) != 0)
    origin = CANVAS_POS // 2  # canvas block origin in subband coordinates
    pad = 2
    r0, r1 = rows.min() - origin - pad, rows.max() - origin + 1 + pad
    c0, c1 = cols.min() - origin - pad, cols.max() - origin + 1 + pad
    return r0, r1, c0, c1


def uniward_jpeg_cost(coeffs_q: np.ndarray, quality: int) -> np.ndarray:
    """J-UNIWARD cost of a +-1 change of every DCT coefficient.

    Args:
        coeffs_q: (n_blocks, 8, 8) int16 quantized coefficients of the JPEG
            cover — the same array the embedding will modify.
        quality:  JPEG quality the coefficients were produced with.

    Returns:
        rho: (n_blocks, 8, 8) float64. The DC entry is inf (never embedded
        into); AC entries are the costs of the +-1 changes.
    """
    n_blocks = coeffs_q.shape[0]
    side = int(round(np.sqrt(n_blocks)))
    assert side * side == n_blocks, "block grid must be square"
    img_side = side * 8

    # Decompressed JPEG cover: the spatial image the wavelet cost lives on.
    x = jpeg_decompress(coeffs_q, quality, shape=(img_side, img_side)).astype(np.float64)
    xp = np.pad(x, PAD, mode="reflect")
    subbands = list(dwt_details_db8(xp))  # SIGNED subbands: |W(x) +- R| needs signs

    r0, r1, c0, c1 = _response_window()
    win_h, win_w = r1 - r0, c1 - c0
    # Block (by, bx) origin (PAD + 8*by, PAD + 8*bx) maps to subband origin
    # (PAD/2 + 4*by, PAD/2 + 4*bx); gather the window starting there.
    base = PAD // 2 + r0
    assert base >= 0, "response window extends before the padded subbands"
    views = []
    for sb in subbands:
        sw = np.lib.stride_tricks.sliding_window_view(sb, (win_h, win_w))
        views.append(sw[base::4, base::4][:side, :side])  # (B, B, win_h, win_w) of |W(x)|

    qtab = quality_table(quality)
    db = dct_basis()
    rho = np.full((n_blocks, 8, 8), np.inf)
    grid = np.arange(n_blocks).reshape(side, side)
    for ku in range(8):
        for kv in range(8):
            if ku == 0 and kv == 0:
                continue
            pattern = qtab[ku, kv] * np.outer(db[:, ku], db[:, kv])
            canvas = np.zeros((CANVAS, CANVAS))
            canvas[CANVAS_POS:CANVAS_POS + 8, CANVAS_POS:CANVAS_POS + 8] = pattern
            resp = dwt_details_db8(canvas)
            cost = None
            for view, res in zip(views, resp):
                rw = res[CANVAS_POS // 2 + r0: CANVAS_POS // 2 + r1,
                         CANVAS_POS // 2 + c0: CANVAS_POS // 2 + c1]
                base = np.abs(view)
                d_plus = np.abs(np.abs(view + rw) - base)
                d_minus = np.abs(np.abs(view - rw) - base)
                block_cost = (d_plus + d_minus).sum(axis=(-2, -1))  # (B, B)
                cost = block_cost if cost is None else cost + block_cost
            rho[grid, ku, kv] = cost  # (side, side) block grid
    return rho


def _active_window(n: int, m: int, h: int) -> int:
    """Length of the scrambled STC window for m bits over n ACs at height h.

    Spreads the payload to a relative rate of SPREAD_RATE over the active
    window (capped by n) — the redundancy is what lets the Viterbi exploit
    the distortion function; a saturated window (n = m + h - 1) locks the
    flip rate at ~50% and makes the cost irrelevant. Deterministic in
    (n, m, h), so the extractor re-derives it.
    """
    return int(min(n, int(np.ceil(m / SPREAD_RATE)) + h - 1))


def _embed_stc_on_coeffs(ac_seq: np.ndarray, msg: np.ndarray, rho: np.ndarray,
                         h: int = 10, seed: int = 0) -> np.ndarray:
    """Spanned STC embed into a flat AC-coefficient sequence (int16, may be negative).

    Mirrors stc.embed_stc_on_cover but without the [0, 255] clip: a flip is
    applied as +1 on even coefficients, -1 on odd ones (parity is well defined
    in two's complement for negative values too).
    """
    ac_seq = np.asarray(ac_seq, dtype=np.int16).ravel()
    rho = np.asarray(rho, dtype=np.float64).ravel()
    n = ac_seq.size
    m = msg.size
    assert m <= n - h + 1, f"payload {m} bits exceeds AC capacity {n - h + 1} for h={h}"

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)

    n_active = _active_window(n, m, h)
    flip = np.zeros(n, dtype=np.uint8)
    flip[:n_active] = embed_stc_spanned(
        (ac_seq[perm][:n_active] & 1).astype(np.uint8), msg,
        rho[perm][:n_active], h=h, seed=seed,
    )
    delta = np.where((ac_seq & 1) == 0, 1, -1).astype(np.int16)
    delta[flip[inv] == 0] = 0
    return ac_seq + delta


def _extract_stc_on_coeffs(ac_seq: np.ndarray, m: int, h: int = 10, seed: int = 0) -> np.ndarray:
    ac_seq = np.asarray(ac_seq, dtype=np.int16).ravel()
    n = ac_seq.size
    assert m <= n - h + 1, f"payload {m} bits exceeds AC capacity {n - h + 1} for h={h}"
    perm = np.random.default_rng(seed).permutation(n)
    n_active = _active_window(n, m, h)
    return extract_stc_spanned((ac_seq[perm][:n_active] & 1).astype(np.uint8), m, h=h, seed=seed)


def _ac_view(coeffs_q: np.ndarray) -> np.ndarray:
    """(n_blocks, 8, 8) -> (n_blocks, 63) AC coefficients in JPEG zigzag order."""
    return coeffs_q[:, ZIGZAG[1:, 0], ZIGZAG[1:, 1]]


def _ac_scatter(coeffs_q: np.ndarray, ac_flat: np.ndarray) -> np.ndarray:
    """Inverse of _ac_view: write the AC values back, leaving DC untouched."""
    out = coeffs_q.copy()
    out[:, ZIGZAG[1:, 0], ZIGZAG[1:, 1]] = ac_flat
    return out


def embed_juniward(coeffs_q: np.ndarray, quality: int, rate: float,
                   rng: np.random.Generator, h: int = 10, seed: int = 0
                   ) -> tuple[np.ndarray, dict]:
    """Embed ``rate`` bpp of random payload with J-UNIWARD + STC.

    Returns (stego_coeffs, info) where ``info`` reports the payload size, the
    number of changed coefficients and the mean cost of the changes vs. the
    mean cost over all ACs (the STC should be picking cheap coefficients).
    """
    coeffs_q = np.asarray(coeffs_q, dtype=np.int16)
    n_pixels = coeffs_q.shape[0] * 64
    m = int(round(rate * n_pixels))
    msg = rng.integers(0, 2, m, dtype=np.uint8)

    rho = uniward_jpeg_cost(coeffs_q, quality)
    ac_rho = rho[:, ZIGZAG[1:, 0], ZIGZAG[1:, 1]].ravel()
    ac_seq = _ac_view(coeffs_q).ravel()

    stego_seq = _embed_stc_on_coeffs(ac_seq, msg, ac_rho, h=h, seed=seed)
    stego = _ac_scatter(coeffs_q, stego_seq.reshape(coeffs_q.shape[0], 63))

    changed = stego_seq != ac_seq
    info = {
        "m": m, "rate": rate, "h": h,
        "n_changed": int(changed.sum()),
        "mean_cost_changed": float(ac_rho[changed].mean()) if changed.any() else 0.0,
        "mean_cost_all": float(ac_rho.mean()),
    }
    return stego, info


def extract_juniward(coeffs_q: np.ndarray, rate: float, h: int = 10, seed: int = 0) -> np.ndarray:
    """Read the payload back out of stego coefficients (inverse of embed_juniward)."""
    coeffs_q = np.asarray(coeffs_q, dtype=np.int16)
    n_pixels = coeffs_q.shape[0] * 64
    m = int(round(rate * n_pixels))
    return _extract_stc_on_coeffs(_ac_view(coeffs_q).ravel(), m, h=h, seed=seed)


def _self_test():
    rng = np.random.default_rng(0)

    # 1) db8 filter is a valid orthonormal Daubechies-8 scaling filter.
    h, g = db8_filters()
    assert h.size == 16 and abs(np.linalg.norm(h) - 1) < 1e-12
    assert abs(h.sum() - np.sqrt(2)) < 1e-9, f"sum h = {h.sum()}, expected sqrt(2)"

    # the constructor reproduces the published db2 taps (order/sign invariant)
    db2 = daubechies_scaling_filter(2)
    db2_ref = np.array([-0.12940952255092145, 0.22414386804185735,
                        0.83651630373746899, 0.48296291314469025])
    assert np.allclose(np.sort(np.abs(db2)), np.sort(np.abs(db2_ref)), atol=1e-9), \
        f"spectral factorization disagrees with the published db2 taps: {db2}"
    assert np.allclose(np.sort(db2), np.sort(db2_ref), atol=1e-9)
    print(f"  db2 cross-check OK; db8 first taps: {h[:4]}")

    # even-shift orthogonality: <h, shift_2n h> = delta_n
    ac = np.correlate(h, h, mode="full")[15::2]  # lags 0, +2, +4, ...
    assert abs(ac[0] - 1) < 1e-12 and np.all(np.abs(ac[1:]) < 1e-9), \
        f"even-shift orthogonality violated: {ac}"
    # 8 vanishing moments of the wavelet filter
    t = np.arange(16, dtype=np.float64)
    for j in range(8):
        assert abs(float((g * t ** j).sum())) < 1e-6, f"vanishing moment {j} failed"
    print("  db8: L2-normalised, sum sqrt(2), even-shift orthogonal, 8 vanishing moments")

    # 2) cost map: DC = inf, all ACs finite and positive-ish.
    img = np.clip(rng.normal(128, 45, (64, 64)), 0, 255).astype(np.uint8)
    coeffs = jpeg_compress(img, 75)
    rho = uniward_jpeg_cost(coeffs, 75)
    assert np.all(np.isinf(rho[:, 0, 0])), "DC cost must be inf"
    assert np.all(np.isfinite(rho[:, ZIGZAG[1:, 0], ZIGZAG[1:, 1]])), "AC cost must be finite"
    assert (rho[:, ZIGZAG[1:, 0], ZIGZAG[1:, 1]] >= 0).all()
    print(f"  cost map: AC costs in [{rho[:, 1:, :][rho[:, 1:, :] > 0].min():.2e}, "
          f"{rho[:, 1:, :].max():.2e}]")

    # 3) embed/extract round trip on a 64x64 image (m = 819 <= 1008-9 ACs).
    for rate in (0.1, 0.2):
        stego, info = embed_juniward(coeffs, 75, rate, np.random.default_rng(1))
        got = extract_juniward(stego, rate)
        replay = np.random.default_rng(1)
        m = int(round(rate * 64 * 64))
        assert np.array_equal(replay.integers(0, 2, m, dtype=np.uint8), got), \
            "payload did not survive the round trip"
        delta = stego.astype(np.int16) - coeffs.astype(np.int16)
        assert np.abs(delta).max() <= 1, "coefficient moved by more than 1"
        assert np.all(delta[:, 0, 0] == 0), "DC was modified"
        print(f"  rate={rate}: m={info['m']} changed={info['n_changed']} "
              f"({info['n_changed'] / coeffs.size:.3f} per coefficient), "
              f"mean cost changed {info['mean_cost_changed']:.3e} vs all "
              f"{info['mean_cost_all']:.3e}, round trip OK")

    # 4) the STC prefers cheap coefficients: mean cost of changes < overall mean.
    stego, info = embed_juniward(coeffs, 75, 0.2, np.random.default_rng(1))
    assert info["mean_cost_changed"] < info["mean_cost_all"], \
        "STC is not selecting low-cost coefficients"
    print("  adaptivity: changed-coefficient mean cost below the AC average")
    print("J-UNIWARD self-test PASSED")


if __name__ == "__main__":
    _self_test()
