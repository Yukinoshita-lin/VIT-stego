"""JPEG coefficient-domain codec (quantized DCT only, no entropy coding).

Steganography in the JPEG domain lives entirely in the quantized DCT
coefficients, and a steganalyzer only ever sees the *decompressed* pixels —
the entropy-coded bitstream is irrelevant to both. This module therefore
implements just the part of JPEG that steganography touches:

    cover pixels -> 8x8 orthonormal DCT-II -> divide by quant table -> round
    (embed happens here, on the integer coefficients)
    coefficients -> multiply by quant table -> inverse DCT -> clip -> pixels

The orthonormal DCT-II basis equals the JPEG-spec normalization exactly (the
1/4*C(u)*C(v) factors of Annex A fold into the same scale), so the standard
Annex-K luminance table applies to these coefficients directly. Because we
never serialize a bitstream, a "cover" here is the decompressed JPEG of the
original image — precisely what a detector deployed against JPEG files sees on
the clean side.
"""
from __future__ import annotations

import numpy as np

# JPEG Annex K base quantization table, luminance (row-major 8x8).
BASE_QTABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)

# Standard JPEG zigzag scan order over an 8x8 block (row, col), DC first.
ZIGZAG = sorted(((r, c) for r in range(8) for c in range(8)),
                key=lambda rc: (rc[0] + rc[1], rc[1] if (rc[0] + rc[1]) % 2 else -rc[0]))
ZIGZAG = np.array(ZIGZAG)  # (64, 2)


def dct_basis() -> np.ndarray:
    """Orthonormal 8-point DCT-II matrix D with D @ D.T = I."""
    n = np.arange(8)
    k = n[:, None]
    d = np.cos((2 * n[None, :] + 1) * k * np.pi / 16) * np.sqrt(2 / 8)
    d[0] /= np.sqrt(2)
    return d


_D = dct_basis()


def quality_table(quality: int) -> np.ndarray:
    """Scale the Annex-K base table with the IJG quality formula."""
    if not 1 <= quality <= 100:
        raise ValueError(f"quality must be in [1, 100], got {quality}")
    scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    q = np.floor((BASE_QTABLE * scale + 50.0) / 100.0)
    return np.clip(q, 1.0, 255.0)


def _to_blocks(img: np.ndarray) -> np.ndarray:
    """(H, W) -> (H/8 * W/8, 8, 8), row-major over blocks."""
    h, w = img.shape
    if h % 8 or w % 8:
        raise ValueError(f"image size {h}x{w} must be a multiple of 8")
    return (img.reshape(h // 8, 8, w // 8, 8)
               .transpose(0, 2, 1, 3)
               .reshape(-1, 8, 8)
               .astype(np.float64))


def _from_blocks(blocks: np.ndarray, h: int, w: int) -> np.ndarray:
    return (blocks.reshape(h // 8, w // 8, 8, 8)
                  .transpose(0, 2, 1, 3)
                  .reshape(h, w))


def forward_dct(img: np.ndarray) -> np.ndarray:
    """(H, W) pixels -> (n_blocks, 8, 8) real DCT coefficients."""
    blocks = _to_blocks(img)
    return _D @ blocks @ _D.T


def inverse_dct(coeffs: np.ndarray, h: int, w: int) -> np.ndarray:
    """(n_blocks, 8, 8) coefficients -> (H, W) pixels."""
    return _from_blocks(_D.T @ coeffs @ _D, h, w)


def jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    """Quantized DCT coefficients of the JPEG-compressed image, int16.

    This is the coefficient array a JPEG decoder would read (before entropy
    coding): `jpeg_decompress(jpeg_compress(x, q), q)` reproduces the decoder
    output. Zigzag reordering is deferred to the embedders — here the layout
    stays (n_blocks, 8, 8) so spatial cost maps stay block-aligned.
    """
    img = np.asarray(img, dtype=np.float64)
    coeffs = forward_dct(img) / quality_table(quality)
    return np.rint(coeffs).astype(np.int16)


def jpeg_decompress(coeffs_q: np.ndarray, quality: int, shape: tuple | None = None) -> np.ndarray:
    """Dequantize + inverse DCT + clip: the image a JPEG decoder would show."""
    coeffs_q = np.asarray(coeffs_q, dtype=np.float64)
    if shape is None:
        side = int(round(np.sqrt(coeffs_q.shape[0] * 64)))
        shape = (side, side)
    h, w = shape
    if (h // 8) * (w // 8) != coeffs_q.shape[0]:
        raise ValueError(f"shape {shape} does not match {coeffs_q.shape[0]} blocks")
    pixels = inverse_dct(coeffs_q * quality_table(quality), h, w)
    return np.clip(np.rint(pixels), 0, 255).astype(np.uint8)


def to_zigzag(coeffs_q: np.ndarray) -> np.ndarray:
    """(n_blocks, 8, 8) -> (n_blocks, 64) in JPEG zigzag order (DC first)."""
    return coeffs_q[:, ZIGZAG[:, 0], ZIGZAG[:, 1]]


def from_zigzag(flat: np.ndarray) -> np.ndarray:
    """(n_blocks, 64) zigzag order -> (n_blocks, 8, 8)."""
    out = np.empty((flat.shape[0], 8, 8), dtype=flat.dtype)
    out[:, ZIGZAG[:, 0], ZIGZAG[:, 1]] = flat
    return out


def _self_test():
    rng = np.random.default_rng(0)

    # 1) DCT basis is orthonormal, matches scipy's DCT-II 'ortho' axis-wise.
    d = dct_basis()
    assert np.allclose(d @ d.T, np.eye(8), atol=1e-12), "DCT basis not orthonormal"
    try:
        from scipy.fft import dct, idct
        x = rng.uniform(0, 255, (32, 40, 8, 8))
        ref = dct(dct(x, axis=-2, norm="ortho"), axis=-1, norm="ortho")
        got = _D @ x @ _D.T
        assert np.allclose(ref, got, atol=1e-8), "DCT disagrees with scipy.fft ortho DCT-II"
        back = idct(idct(ref, axis=-2, norm="ortho"), axis=-1, norm="ortho")
        assert np.allclose(back, x, atol=1e-8)
        print("  DCT basis: orthonormal, == scipy.fft DCT-II ortho, invertible")
    except ImportError:
        print("  scipy unavailable — skipped cross-check")

    # 2) compress/decompress round trip: idempotent in coefficient space for
    #    every block whose reconstruction stays inside [0, 255] (decoder-side
    #    clipping perturbs re-compression exactly as it does for real JPEG),
    #    and the decompressed image is a plausible JPEG at Q85.
    img = np.clip(rng.normal(128, 40, (64, 64)), 0, 255).astype(np.uint8)
    c1 = jpeg_compress(img, 85)
    pixels = jpeg_decompress(c1, 85)
    c2 = jpeg_compress(pixels, 85)
    in_range = (pixels > 0) & (pixels < 255)
    blocks_ok = in_range.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3).reshape(64, 8, 8).all(axis=(1, 2))
    assert np.array_equal(c1[blocks_ok], c2[blocks_ok]), \
        "coefficients changed after a clip-free round trip"
    mse = float(((pixels.astype(np.float64) - img) ** 2).mean())
    psnr = 10 * np.log10(255**2 / max(mse, 1e-9))
    print(f"  round trip Q85: PSNR {psnr:.2f} dB, clip-free blocks idempotent "
          f"({int(blocks_ok.sum())}/64)")

    # 3) quality monotonicity: higher quality -> smaller quant error.
    for q_hi, q_lo in ((95, 75), (85, 50)):
        e_hi = float(np.abs(jpeg_decompress(jpeg_compress(img, q_hi), q_hi).astype(float) - img).mean())
        e_lo = float(np.abs(jpeg_decompress(jpeg_compress(img, q_lo), q_lo).astype(float) - img).mean())
        assert e_hi < e_lo, f"Q{q_hi} error {e_hi} not below Q{q_lo} error {e_lo}"
    print("  quality scaling: higher Q reconstructs better")

    # 4) zigzag round trip and DC-first property.
    flat = to_zigzag(c1)
    assert np.array_equal(from_zigzag(flat), c1)
    assert np.array_equal(flat[0, 0], c1[0, 0, 0]), "zigzag must start with DC"
    print("  zigzag: round trip OK, DC first")
    print("JPEG codec self-test PASSED")


if __name__ == "__main__":
    _self_test()
