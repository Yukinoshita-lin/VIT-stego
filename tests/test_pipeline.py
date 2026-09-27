"""Zero-dependency self-checks for the cover/stego pipeline.

Run with ``python tests/test_pipeline.py``. Exits non-zero on the first failure.

These cover the invariants that silently corrupt an experiment when broken:
a leaky split, a non-deterministic evaluation crop, an unfrozen SRM stem, a
checkpoint layout that no longer loads, and an embedder whose payload cannot be
recovered.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from scripts.prepare_data import embed_adaptive, embed_lsb, embed_stc_hill
from src.dataset import CoverStegoDataset, CoverStegoView
from src.engine import make_splits
from src.srm_filter import SRM_KERNELS
from src.vit_stego import Attention, LightViT, SRMStem, count_params

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


@test
def test_splits_are_disjoint_and_keep_pairs_together():
    n, val_frac, test_frac = 200, 0.15, 0.2
    for seed in range(4):
        train, val, test = make_splits(n, val_frac, test_frac, seed)
        sets = [set(a.tolist()) for a in (train, val, test)]
        assert not (sets[0] & sets[1]), "train/val overlap"
        assert not (sets[0] & sets[2]), "train/test overlap"
        assert not (sets[1] & sets[2]), "val/test overlap"
        assert sum(len(s) for s in sets) == 2 * n, "splits do not cover the dataset"

        for split in sets:
            covers = {i for i in split if i < n}
            stegos = {i - n for i in split if i >= n}
            assert covers == stegos, "a source image's cover and stego landed on different sides"

        # A leaky split puts a sample's twin in the training set; assert none do.
        twins = {i + n if i < n else i - n for i in train}
        assert not (twins & (sets[1] | sets[2])), "twin of a training sample is in val/test"


def _tiny_dataset(path, n=8, size=32):
    rng = np.random.default_rng(0)
    covers = rng.integers(0, 256, (n, size, size), dtype=np.uint8)
    stegos = covers.copy()
    stegos[:, 0, 0] ^= 1
    np.savez(path, covers=covers, stegos=stegos, method=np.array("adaptive"),
             rate=np.float64(0.4))


@test
def test_eval_crop_is_deterministic_and_train_crop_varies():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tiny.npz")
        _tiny_dataset(path, n=8, size=64)
        ds = CoverStegoDataset(path, img_size=32, augment=True)

        first = ds.fetch(0, train=False)[0]
        for _ in range(5):
            assert torch.equal(first, ds.fetch(0, train=False)[0]), "eval crop is not deterministic"

        views = {ds.fetch(0, train=True)[0].numpy().tobytes() for _ in range(40)}
        assert len(views) > 1, "training crop/flip produced a single view"

        ds_noaug = CoverStegoDataset(path, img_size=32, augment=False)
        assert torch.equal(ds_noaug.fetch(0, train=True)[0], first), \
            "augment=False must fall back to the deterministic crop"

        assert ds.meta.get("method") == "adaptive", "provenance metadata was not read back"
        assert abs(ds.meta.get("rate", 0) - 0.4) < 1e-9


@test
def test_view_shares_the_underlying_arrays():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tiny.npz")
        _tiny_dataset(path, n=8, size=32)
        ds = CoverStegoDataset(path, img_size=32)
        view = CoverStegoView(ds, [0, 9, 3], train=False)
        assert len(view) == 3
        x, y = view[0]
        assert x.shape == (1, 32, 32) and y.item() == 0.0
        x, y = view[1]
        assert y.item() == 1.0, "index >= n_per_class must be a stego"


@test
def test_srm_stem_is_frozen_and_truncates():
    stem = SRMStem(truncate=4.0)
    assert not stem.conv.weight.requires_grad, "SRM kernel must be frozen"
    assert SRM_KERNELS.shape == (30, 5, 5)
    assert torch.allclose(stem.conv.weight.cpu(), torch.from_numpy(SRM_KERNELS[:, None]).float())

    x = torch.rand(4, 1, 64, 64) * 255.0
    r = stem(x)
    assert r.shape == (4, 30, 64, 64)
    assert float(r.abs().max()) <= 4.0 + 1e-6, "truncation did not clamp the residual"

    open_stem = SRMStem(truncate=0.0)
    assert float(open_stem(x).abs().max()) > 4.0, "test input too tame to exercise truncation"


@test
def test_attention_matches_torch_multiheadattention_layout():
    mha = torch.nn.MultiheadAttention(96, 3, dropout=0.0, batch_first=True)
    ours = Attention(96, 3, dropout=0.0)
    # Same parameter names and shapes: an nn.MultiheadAttention state dict must
    # load into ours without keys left over.
    missing, unexpected = ours.load_state_dict(mha.state_dict(), strict=True)
    assert not missing and not unexpected

    ours.eval()
    x = torch.randn(2, 17, 96)
    with torch.no_grad():
        ref = mha(x, x, x, need_weights=False)[0]
        got = ours(x)
    assert torch.allclose(ref, got, atol=1e-5), "SDPA attention disagrees with nn.MultiheadAttention"


@test
def test_lightvit_shapes_and_param_budget():
    for use_srm, expected in ((False, 478_657), (True, 657_583)):
        m = LightViT(img_size=128, patch_size=8, embed_dim=96, depth=4, heads=3, use_srm=use_srm)
        assert count_params(m) == expected, f"use_srm={use_srm} param count drifted: {count_params(m)}"
        assert tuple(m(torch.randn(2, 1, 128, 128)).shape) == (2,)
    m = LightViT(img_size=128, patch_size=8, use_srm=True, pool="mean")
    assert tuple(m(torch.randn(2, 1, 128, 128)).shape) == (2,)


@test
def test_srnet_shares_the_truncation_knob():
    """The LightViT/SRNet comparison is only fair if both clip the residual."""
    from src.srnet import SRNet

    m = SRNet(img_size=128, use_srm=True, truncate=4.0)
    assert m.stem.truncate == 4.0, "SRNet stem ignored the truncation setting"
    assert not m.stem.conv.weight.requires_grad
    assert m.layer0[0].in_channels == 30, "SRNet first conv should see 30 SRM channels"
    assert tuple(m(torch.rand(2, 1, 128, 128) * 255.0).shape) == (2,)

    # kaiming_normal_ writes in place and ignores requires_grad=False, so a
    # constructor-time init loop over self.modules() would silently replace the
    # frozen SRM kernels with random ones — and the model would train on noise.
    assert torch.allclose(m.stem.conv.weight.cpu(), torch.from_numpy(SRM_KERNELS[:, None]).float()), \
        "SRNet re-initialised the frozen SRM kernels"

    plain = SRNet(img_size=128, use_srm=False)
    assert plain.layer0[0].in_channels == 1


@test
def test_embed_adaptive_and_lsb_change_by_at_most_one():
    rng = np.random.default_rng(0)
    cover = rng.integers(0, 256, (48, 48), dtype=np.uint8)
    for fn in (embed_adaptive, embed_lsb):
        stego = fn(cover, 0.4, np.random.default_rng(1))
        delta = stego.astype(np.int16) - cover.astype(np.int16)
        assert np.abs(delta).max() <= 1, f"{fn.__name__} moved a pixel by more than 1"
        assert (delta != 0).mean() > 0.05, f"{fn.__name__} changed almost nothing"
        assert stego.dtype == np.uint8 and stego.shape == cover.shape


@test
def test_stc_hill_payload_is_recoverable():
    """The .npz is only meaningful if the payload actually survives embedding."""
    from stego.stc import extract_stc

    rng_img = np.random.default_rng(3)
    cover = rng_img.integers(0, 256, (64, 64), dtype=np.uint8)
    rate, h, seed = 0.4, 10, 0

    stego = embed_stc_hill(cover, rate, np.random.default_rng(11), h=h, seed=seed)

    # Replay the generator to recover the message and the scrambling order that
    # embed_stc_hill used internally.
    n = cover.size
    m = int(round(rate * n))
    replay = np.random.default_rng(11)
    msg = replay.integers(0, 2, m, dtype=np.uint8)
    perm = replay.permutation(n)

    n_active = m + h - 1
    got = extract_stc((stego.ravel()[perm][:n_active]) & 1, m, h=h, seed=seed)
    n_bad = int((got != msg).sum())

    delta = stego.astype(np.int16) - cover.astype(np.int16)
    assert np.abs(delta).max() <= 1
    # Pixels saturated at 0/255 cannot absorb a +-1 flip, so a handful of the
    # payload bits can be lost; anything beyond a small fraction means the
    # embedder is broken rather than the cover being saturated.
    assert n_bad / m < 0.01, f"{n_bad}/{m} payload bits corrupted"


@test
def test_spanned_stc_round_trip_and_stride1_equivalence():
    """The spread (strided) STC must recover its payload and must degenerate
    bit-for-bit to the stride-1 schedule at the saturated length n = m + h - 1."""
    from stego.stc import embed_stc, embed_stc_spanned, extract_stc_spanned

    rng = np.random.default_rng(5)
    m, h = 12, 4
    for n in (m + h - 1, 33, 64):
        x = rng.integers(0, 2, n, dtype=np.uint8)
        msg = rng.integers(0, 2, m, dtype=np.uint8)
        rho = rng.uniform(0.5, 5.0, n)
        flip = embed_stc_spanned(x, msg, rho, h=h, seed=0)
        got = extract_stc_spanned(x ^ flip, m, h=h, seed=0)
        assert np.array_equal(got, msg), f"spanned round trip failed at n={n}"
        if n == m + h - 1:
            assert np.array_equal(flip, embed_stc(x, msg, rho, h=h, seed=0)), \
                "spanned schedule diverged from stride-1 at the saturated length"


@test
def test_jpeg_codec_round_trip_and_quality():
    """Quantized-DCT codec: ortho DCT, clip-free coefficient idempotence,
    monotone quality, zigzag round trip."""
    from stego.jpeg import from_zigzag, jpeg_compress, jpeg_decompress, to_zigzag

    rng = np.random.default_rng(7)
    img = np.clip(rng.normal(128, 45, (64, 64)), 0, 255).astype(np.uint8)
    c1 = jpeg_compress(img, 85)
    pixels = jpeg_decompress(c1, 85)
    c2 = jpeg_compress(pixels, 85)
    in_range = (pixels > 0) & (pixels < 255)
    blocks_ok = (in_range.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3)
                 .reshape(64, 8, 8).all(axis=(1, 2)))
    assert blocks_ok.any() and np.array_equal(c1[blocks_ok], c2[blocks_ok]), \
        "clip-free blocks must re-compress to identical coefficients"
    mse = float(((pixels.astype(float) - img) ** 2).mean())
    assert 10 * np.log10(255**2 / max(mse, 1e-9)) > 28, "Q85 reconstruction too lossy"
    assert np.array_equal(from_zigzag(to_zigzag(c1)), c1), "zigzag round trip failed"


@test
def test_nsf5_round_trip_and_no_shrinkage():
    """nsF5: payload recoverable from the stego's own nonzero ACs, carrier set
    unchanged (no shrinkage), changes bounded by 1, DC/zeros untouched."""
    from stego.jpeg import jpeg_compress
    from stego.nsf5 import embed_nsf5, extract_nsf5_payload

    rng = np.random.default_rng(9)
    img = np.clip(rng.normal(128, 45, (64, 64)), 0, 255).astype(np.uint8)
    cover_c = jpeg_compress(img, 85)
    rate, quality = 0.4, 85
    stego_c, info = embed_nsf5(cover_c, rate, np.random.default_rng(11), rate_unit="bpnc")

    m = int(round(rate * info["n_carriers"]))
    got = extract_nsf5_payload(stego_c, m)
    replay = np.random.default_rng(11)
    assert np.array_equal(replay.integers(0, 2, m, dtype=np.uint8), got), \
        "nsF5 payload did not survive the round trip"

    ac = np.ones((8, 8), dtype=bool)
    ac[0, 0] = False
    cover_nz = (cover_c != 0) & ac
    stego_nz = (stego_c != 0) & ac
    assert np.array_equal(cover_nz, stego_nz), "carrier set changed (shrinkage!)"
    delta = stego_c.astype(np.int16) - cover_c.astype(np.int16)
    assert np.abs(delta).max() <= 1, "coefficient moved by more than 1"
    assert np.all(delta[~cover_nz] == 0), "zero/DC coefficient was modified"
    assert info["code_height"] >= 1 and info["n_changed"] > 0


@test
def test_juniward_jpeg_round_trip_and_adaptivity():
    """J-UNIWARD: payload recoverable, +-1 coefficient changes, DC untouched,
    and the STC actually prefers cheap coefficients (the whole point of the
    UNIWARD cost — a broken cost map makes this ratio ~1)."""
    from stego.jpeg import jpeg_compress
    from stego.juniward_jpeg import embed_juniward, extract_juniward

    rng = np.random.default_rng(13)
    img = np.clip(rng.normal(128, 45, (48, 48)), 0, 255).astype(np.uint8)
    cover_c = jpeg_compress(img, 75)
    rate = 0.4
    stego_c, info = embed_juniward(cover_c, 75, rate, np.random.default_rng(15))

    got = extract_juniward(stego_c, rate)
    replay = np.random.default_rng(15)
    m = int(round(rate * 48 * 48))
    assert np.array_equal(replay.integers(0, 2, m, dtype=np.uint8), got), \
        "J-UNIWARD payload did not survive the round trip"

    delta = stego_c.astype(np.int16) - cover_c.astype(np.int16)
    assert np.abs(delta).max() <= 1, "coefficient moved by more than 1"
    assert np.all(delta[:, 0, 0] == 0), "DC was modified"
    assert info["mean_cost_changed"] < 0.8 * info["mean_cost_all"], \
        (f"STC not exploiting the cost map: changed {info['mean_cost_changed']:.1f} "
         f"vs all {info['mean_cost_all']:.1f}")


@test
def test_coeff_dataset_crops_stay_block_aligned():
    """The JPEG-domain dataset view must serve quantized-DCT maps on the raw
    integer scale, keep crops aligned to the 8x8 block grid, and flip the
    cover and its stego twin consistently."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from prepare_data import COEFF_CLIP, coeff_map

    from src.dataset import CoverStegoDataset

    rng = np.random.default_rng(17)
    n, size = 4, 64
    covers_px = rng.integers(0, 256, (n, size, size), dtype=np.uint8)
    stegos_px = covers_px.copy()

    # Build the npz the way prepare_data.py does for JPEG-domain datasets.
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "jpeg.npz")
        from stego.jpeg import jpeg_compress
        covers_c = np.stack([coeff_map(jpeg_compress(img, 75)) for img in covers_px])
        stegos_c = covers_c.copy()
        stegos_c[:, 20, 21] += 1  # one embedded coefficient (inside the centre crop)
        np.savez(path, covers=covers_px, stegos=stegos_px,
                 covers_c=covers_c.astype(np.int16), stegos_c=stegos_c.astype(np.int16),
                 method=np.array("nsf5"), rate=np.float64(0.4),
                 rate_unit=np.array("bpnc"), domain=np.array("jpeg"),
                 quality=np.int64(75), coeff_clip=np.int64(COEFF_CLIP))

        ds = CoverStegoDataset(path, img_size=32, augment=True, input="coeff")
        assert ds.stored_size == 64
        x_cover, y0 = ds.fetch(0, train=False)
        x_stego, y1 = ds.fetch(n, train=False)
        assert y0.item() == 0.0 and y1.item() == 1.0
        assert x_cover.shape == (1, 32, 32)
        assert float(x_cover.abs().max()) <= COEFF_CLIP, "coefficient map exceeded the stored clip"
        assert float(x_stego[0, 4, 5] - x_cover[0, 4, 5]) == 1.0, \
            "the +1 embedding change must survive into the served tensors"

        # deterministic eval crop, block-aligned: DC positions are always zero
        for _ in range(3):
            assert torch.equal(x_cover, ds.fetch(0, train=False)[0])
        assert float(x_cover[0, ::8, ::8].abs().max()) == 0.0, "DC not zeroed"

        # train crops land on the 8-grid before flips; a flip shifts the whole
        # grid (DC at 7-8k instead of 8k in the flipped axes) but flips cover
        # and delta together, so the embedding signal survives augmentation
        seen_flip = False
        for _ in range(10):
            x, _ = ds.fetch(0, train=True)
            grids = [float(x[0, r::8, c::8].abs().max())
                     for r, c in ((0, 0), (0, 7), (7, 0), (7, 7))]
            assert 0.0 in grids, "crop matches no block alignment (flipped or not)"
            seen_flip |= (grids[0] != 0.0)
        assert seen_flip, "no flipped view drawn in 10 draws"


def main():
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
