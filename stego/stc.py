"""stego/stc.py — Syndrome-Trellis Code (Filler et al. 2011), complete implementation.

Size-general binary STC over pixel parities:

  - The parity-check matrix H (m x n) is banded: each row i holds a height-h vector
    ``h_vec`` starting at column i, so message bit ``i`` is
    ``msg[i] = XOR_k h_vec[k] * y[i+k]`` for i = 0..m-1.
  - The payload length m is arbitrary (m <= n - h + 1), so any rate (e.g. 0.4 bpp)
    embeds into any cover size, unlike the earlier segment-based placeholder that
    required ``n == m * 2^(m-1)``.
  - A Viterbi shortest-path over 2^(h-1) states finds the minimum-distortion flip
    mask. Direction is applied separately as LSB matching (even -> +1, odd -> -1).

``embed_stc``/``extract_stc`` keep the historical stride-1 schedule (n = m + h - 1,
maximally constrained). ``embed_stc_spanned``/``extract_stc_spanned`` spread the
same banded rows over a longer cover (row i starts at floor(i*(n-h+1)/m)), which
is what real STC embedding does: the redundancy n - m is the Viterbi's freedom,
and without it the code saturates, the flip rate locks at ~50% of the active
window and the distortion function stops mattering.

Extraction simply reads the stego LSBs and XORs along each parity window.
"""
from __future__ import annotations

import numpy as np


def build_parity(h: int, seed: int = 0) -> np.ndarray:
    """First row of the parity-check matrix: h_vec[0] = h_vec[-1] = 1, interior pseudo-random."""
    rng = np.random.default_rng(seed)
    v = rng.integers(0, 2, h, dtype=np.uint8)
    v[0] = 1
    v[-1] = 1
    return v


def stc_starts(n: int, m: int, h: int) -> np.ndarray:
    """Column where each of the m syndrome rows begins, for a length-n cover.

    Row i starts at floor(i * (n - h + 1) / m), so bands are strictly spaced
    (>= 1 apart whenever n >= m + h - 1), the last band ends at column n - 1,
    and for n = m + h - 1 the schedule is exactly 0..m-1 (stride-1).
    """
    assert n >= m + h - 1, f"cover length {n} too short for {m} bits at height {h}"
    return (np.arange(m) * (n - h + 1) // m).astype(np.int64)


def _extract(bits: np.ndarray, h_vec: np.ndarray, m: int) -> np.ndarray:
    """Syndrome of the first m windows: msg[i] = XOR_k h_vec[k] * bits[i+k]."""
    idx = np.nonzero(h_vec)[0]
    windows = np.stack([bits[k : k + m] for k in idx], axis=0)  # (|idx|, m)
    return np.bitwise_xor.reduce(windows, axis=0).astype(np.uint8)


def embed_stc(x_bits: np.ndarray, msg: np.ndarray, rho: np.ndarray, h: int = 10, seed: int = 0) -> np.ndarray:
    """Viterbi STC embedding into LSB parities.

    Args:
        x_bits: (n,) uint8 {0,1} — cover LSBs, n = m + h - 1.
        msg:    (m,) uint8 {0,1} — payload.
        rho:    (n,) float (>0) — per-pixel distortion cost.
        h:      constraint height (>=2). State count = 2^(h-1).
        seed:   parity-vector seed (must match extraction).

    Returns:
        flip: (n,) uint8 {0,1} — 1 where the LSB must be flipped.
    """
    x_bits = np.asarray(x_bits, dtype=np.uint8).ravel()
    msg = np.asarray(msg, dtype=np.uint8).ravel()
    rho = np.asarray(rho, dtype=np.float64).ravel()
    n = x_bits.size
    m = msg.size
    assert h >= 2, "height h must be >= 2"
    assert n == m + h - 1, f"cover length {n} must equal m + h - 1 = {m + h - 1}"
    assert rho.size == n, "rho length must match cover length"

    h_vec = build_parity(h, seed)
    half = 1 << (h - 2)
    n_states = 1 << (h - 1)
    s_all = np.arange(n_states, dtype=np.int32)

    # ps[state] = syndrome of the h-1 state bits y_{j-h+1..j-1} with coeffs h_vec[0..h-2].
    ps = np.zeros(n_states, dtype=np.uint8)
    for t in range(h - 1):
        if h_vec[t]:
            ps ^= ((s_all >> t) & 1).astype(np.uint8)

    INF = np.inf
    dp = np.full(n_states, INF)
    dp[0] = 0.0
    trace_prev = np.zeros((n, n_states), dtype=np.int32)
    trace_dec = np.zeros((n, n_states), dtype=np.uint8)

    for j in range(n):
        ndp = np.full(n_states, INF)
        if j < h - 1:
            # initial transient: no syndrome constraint yet, two decisions per state.
            xj = int(x_bits[j])
            rj = rho[j]
            for s in range(n_states):
                base = dp[s]
                if base == INF:
                    continue
                for yj in (0, 1):
                    ns = (s >> 1) + (yj << (h - 2))
                    cost = base + (0.0 if yj == xj else rj)
                    if cost < ndp[ns]:
                        ndp[ns] = cost
                        trace_prev[j, ns] = s
                        trace_dec[j, ns] = yj
        else:
            # constrained: syndrome of window must equal msg[j-(h-1)];
            # y_j = ps[s] XOR msg bit is forced per state, and the transition is a
            # bijection, so a single vectorized assignment suffices (no min needed).
            yj = (ps.astype(np.uint8) ^ msg[j - (h - 1)]).astype(np.int32)
            ns = (s_all >> 1) + (yj << (h - 2))
            flip_bits = yj ^ x_bits[j]
            cand = dp + rho[j] * flip_bits.astype(np.float64)
            ndp[ns] = cand
            trace_prev[j, ns] = s_all
            trace_dec[j, ns] = yj.astype(np.uint8)
        dp = ndp

    # traceback from the minimum-cost final state
    state = int(np.argmin(dp))
    flip = np.zeros(n, dtype=np.uint8)
    for j in range(n - 1, -1, -1):
        flip[j] = trace_dec[j, state]
        state = int(trace_prev[j, state])
    return flip ^ x_bits  # flip mask w.r.t. cover LSBs


def extract_stc(y_bits: np.ndarray, m: int, h: int = 10, seed: int = 0) -> np.ndarray:
    """Extract m payload bits from stego LSBs (length n = m + h - 1)."""
    y_bits = np.asarray(y_bits, dtype=np.uint8).ravel()
    assert y_bits.size == m + h - 1, f"stego length {y_bits.size} != m + h - 1 = {m + h - 1}"
    return _extract(y_bits, build_parity(h, seed), m)


def embed_stc_spanned(x_bits: np.ndarray, msg: np.ndarray, rho: np.ndarray,
                      h: int = 10, seed: int = 0) -> np.ndarray:
    """Viterbi STC embedding with the syndrome rows spread over the whole cover.

    Same trellis and parity vector as :func:`embed_stc`, but row i of H starts
    at column ``stc_starts(n, m, h)[i]`` instead of i, so any n >= m + h - 1
    embeds m bits and the redundancy becomes real Viterbi freedom. At n =
    m + h - 1 this is bit-identical to :func:`embed_stc`.

    Args:
        x_bits: (n,) uint8 {0,1} — cover LSBs, n >= m + h - 1.
        msg:    (m,) uint8 {0,1} — payload.
        rho:    (n,) float (>0) — per-element distortion cost.
        h:      constraint height (>=2). State count = 2^(h-1).
        seed:   parity-vector seed (must match extraction).

    Returns:
        flip: (n,) uint8 {0,1} — 1 where the LSB must be flipped.
    """
    x_bits = np.asarray(x_bits, dtype=np.uint8).ravel()
    msg = np.asarray(msg, dtype=np.uint8).ravel()
    rho = np.asarray(rho, dtype=np.float64).ravel()
    n, m = x_bits.size, msg.size
    assert h >= 2, "height h must be >= 2"
    assert n >= m + h - 1, f"cover length {n} must be >= m + h - 1 = {m + h - 1}"
    assert rho.size == n, "rho length must match cover length"

    h_vec = build_parity(h, seed)
    n_states = 1 << (h - 1)
    s_all = np.arange(n_states, dtype=np.int32)

    # ps[state] = syndrome of the h-1 state bits with coeffs h_vec[0..h-2].
    ps = np.zeros(n_states, dtype=np.uint8)
    for t in range(h - 1):
        if h_vec[t]:
            ps ^= ((s_all >> t) & 1).astype(np.uint8)

    # A column j is constrained iff it is the last column of some band; the
    # constraint then forces y_j = ps[state] XOR msg[i] (h_vec[-1] == 1).
    starts = stc_starts(n, m, h)
    band_end = np.full(n, -1, dtype=np.int64)
    band_end[starts + h - 1] = np.arange(m)

    # Unconstrained transition, vectorized: target t receives the min of the
    # two source states 2t / 2t+1 (both emitting y_j = 0 -> t, y_j = 1 ->
    # t + half), so one fold + two scalar shifts replace the state loop.
    half = n_states >> 1
    src = 2 * np.arange(half)
    INF = np.inf
    dp = np.full(n_states, INF)
    dp[0] = 0.0
    trace_prev = np.zeros((n, n_states), dtype=np.int32)
    trace_dec = np.zeros((n, n_states), dtype=np.uint8)

    for j in range(n):
        ndp = np.empty(n_states)
        i = int(band_end[j])
        if i < 0 or j < h - 1:
            # unconstrained column: fold the two predecessor states, then add
            # the emission cost of y_j = 0 (lower half) / y_j = 1 (upper half)
            m2 = np.minimum(dp[0::2], dp[1::2])
            winner = (dp[1::2] < dp[0::2])
            prev = src + winner  # 2t or 2t+1, whichever was cheaper
            xj = int(x_bits[j])
            rj = rho[j]
            ndp[:half] = m2 + (rj if xj != 0 else 0.0)
            ndp[half:] = m2 + (rj if xj != 1 else 0.0)
            trace_prev[j, :half] = prev
            trace_prev[j, half:] = prev
            trace_dec[j, :half] = 0
            trace_dec[j, half:] = 1
        else:
            # constrained: the transition is a bijection per state
            yj = (ps.astype(np.uint8) ^ msg[i]).astype(np.int32)
            ns = (s_all >> 1) + (yj << (h - 2))
            flip_bits = yj ^ x_bits[j]
            ndp[ns] = dp + rho[j] * flip_bits.astype(np.float64)
            trace_prev[j, ns] = s_all
            trace_dec[j, ns] = yj.astype(np.uint8)
        dp = ndp

    state = int(np.argmin(dp))
    flip = np.zeros(n, dtype=np.uint8)
    for j in range(n - 1, -1, -1):
        flip[j] = trace_dec[j, state]
        state = int(trace_prev[j, state])
    return flip ^ x_bits  # flip mask w.r.t. cover LSBs


def extract_stc_spanned(y_bits: np.ndarray, m: int, h: int = 10, seed: int = 0) -> np.ndarray:
    """Extract m payload bits from a spanned stego LSB sequence (length n >= m + h - 1)."""
    y_bits = np.asarray(y_bits, dtype=np.uint8).ravel()
    starts = stc_starts(y_bits.size, m, h)
    idx = np.nonzero(build_parity(h, seed))[0]
    windows = np.stack([y_bits[starts + k] for k in idx], axis=0)  # (|idx|, m)
    return np.bitwise_xor.reduce(windows, axis=0).astype(np.uint8)


def apply_lsb_matching(cover_flat: np.ndarray, flip: np.ndarray) -> np.ndarray:
    """Apply an LSB flip mask with +/-1 direction (even -> +1, odd -> -1)."""
    cover_flat = np.asarray(cover_flat, dtype=np.uint8).ravel()
    flip = np.asarray(flip, dtype=np.uint8).ravel()
    delta = np.where((cover_flat & 1) == 0, 1, -1).astype(np.int16)
    delta[flip == 0] = 0
    return np.clip(cover_flat.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def embed_stc_on_cover(cover_flat: np.ndarray, msg: np.ndarray, rho: np.ndarray,
                       h: int = 10, seed: int = 0) -> np.ndarray:
    """One-stop size-general STC embed into a cover's LSBs.

    Scrambles the pixel order with a seed-derived permutation (so the payload is
    not front-loaded), embeds ``m`` bits over the first ``m + h - 1`` scrambled
    pixels, then un-scrambles and applies +/-1 LSB matching (even -> +1, odd -> -1).

    Args:
        cover_flat: (n,) uint8 pixel values (any cover size).
        msg:        (m,) uint8 payload, arbitrary m <= n - h + 1 (any rate).
        rho:        (n,) float per-pixel distortion cost (>0).
        h:          STC constraint height.
        seed:       parity + permutation seed (must match extract_stc_on_cover).

    Returns:
        (n,) uint8 stego pixels.
    """
    cover_flat = np.asarray(cover_flat, dtype=np.uint8).ravel()
    msg = np.asarray(msg, dtype=np.uint8).ravel()
    rho = np.maximum(np.asarray(rho, dtype=np.float64).ravel(), 1e-9)
    n = cover_flat.size
    m = msg.size
    assert rho.size == n, "rho length must match cover length"
    assert m <= n - h + 1, f"payload {m} bits exceeds STC capacity {n - h + 1} for h={h}"

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)

    n_active = m + h - 1
    flip = np.zeros(n, dtype=np.uint8)
    flip[:n_active] = embed_stc(
        (cover_flat[perm][:n_active]) & 1, msg, rho[perm][:n_active], h=h, seed=seed
    )
    return apply_lsb_matching(cover_flat, flip[inv])


def extract_stc_on_cover(stego_flat: np.ndarray, m: int, h: int = 10, seed: int = 0) -> np.ndarray:
    """Extract ``m`` payload bits from stego pixels (inverse of embed_stc_on_cover)."""
    stego_flat = np.asarray(stego_flat, dtype=np.uint8).ravel()
    n = stego_flat.size
    assert m <= n - h + 1, f"payload {m} bits exceeds STC capacity {n - h + 1} for h={h}"
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_active = m + h - 1
    return extract_stc((stego_flat[perm][:n_active]) & 1, m, h=h, seed=seed)


def _brute_force_optimum(x_bits, msg, rho, h_vec):
    """Exhaustive min-cost flip mask over 2^n patterns (tiny n only)."""
    n = x_bits.size
    best = None
    best_cost = np.inf
    for code in range(1 << n):
        flip = np.array([(code >> i) & 1 for i in range(n)], dtype=np.uint8)
        if _extract(x_bits ^ flip, h_vec, msg.size).tolist() != msg.tolist():
            continue
        cost = float((rho * flip).sum())
        if cost < best_cost:
            best_cost = cost
            best = flip
    return best, best_cost


def _self_test():
    rng = np.random.default_rng(0)

    # 1) round-trip
    print("round-trip:")
    for h in (3, 5, 8, 10):
        for m in (2, 5, 12):
            n = m + h - 1
            x = rng.integers(0, 256, n, dtype=np.uint8)
            msg = rng.integers(0, 2, m, dtype=np.uint8)
            rho = rng.uniform(1.0, 4.0, n)
            flip = embed_stc(x & 1, msg, rho, h=h, seed=0)
            stego = apply_lsb_matching(x, flip)
            got = extract_stc(stego & 1, m, h=h, seed=0)
            ok = np.array_equal(got, msg)
            print(f"  h={h:2d} m={m:3d} n={n:4d}  round-trip OK = {ok}  flips={int(flip.sum())}/{n}")
            assert ok

    # 2) optimality vs brute force (tiny cases)
    print("optimality vs brute force:")
    for h, m in ((3, 3), (3, 4), (4, 3)):
        n = m + h - 1
        for _ in range(50):
            x = rng.integers(0, 2, n, dtype=np.uint8)
            msg = rng.integers(0, 2, m, dtype=np.uint8)
            rho = rng.uniform(0.5, 5.0, n)
            h_vec = build_parity(h, 0)
            flip = embed_stc(x, msg, rho, h=h, seed=0)
            _, opt_cost = _brute_force_optimum(x, msg, rho, h_vec)
            v_cost = float((rho * flip).sum())
            assert abs(v_cost - opt_cost) < 1e-9, f"suboptimal: viterbi {v_cost} vs opt {opt_cost}"
        print(f"  h={h} m={m} n={n}  Viterbi == brute-force minimum (50 trials) OK")

    # 3) size-general wrappers (scrambled order) round-trip
    print("size-general wrapper round-trip:")
    for h in (4, 10):
        cover = rng.integers(0, 256, 96 * 96, dtype=np.uint8)
        n = cover.size
        m = int(round(0.4 * n))
        msg = rng.integers(0, 2, m, dtype=np.uint8)
        rho = rng.uniform(0.5, 5.0, n)
        stego = embed_stc_on_cover(cover, msg, rho, h=h, seed=0)
        got = extract_stc_on_cover(stego, m, h=h, seed=0)
        delta = stego.astype(np.int16) - cover.astype(np.int16)
        ok = np.array_equal(got, msg)
        assert ok
        assert np.abs(delta).max() == 1 and set(np.unique(delta)) <= {-1, 0, 1}
        print(f"  h={h:2d} n={n} m={m}  OK={ok}  change-rate={float((delta != 0).mean()):.4f}  |delta|<=1")
    # 4) spanned schedule: stride-1 equivalence, round trips, brute-force optimum
    print("spanned schedule:")
    n, m, h = 6, 2, 3
    h_vec = build_parity(h, 0)
    starts = stc_starts(n, m, h)
    assert starts.tolist() == [0, 2], f"unexpected tiny-span starts {starts}"
    ok = True
    for _ in range(200):
        x = rng.integers(0, 2, n, dtype=np.uint8)
        msg = rng.integers(0, 2, m, dtype=np.uint8)
        rho = rng.uniform(0.5, 5.0, n)
        flip = embed_stc_spanned(x, msg, rho, h=h, seed=0)
        assert np.array_equal(extract_stc_spanned(x ^ flip, m, h=h, seed=0), msg)
        _, opt_cost = _brute_force_spanned(x, msg, rho, h_vec, starts)
        v_cost = float((rho * flip).sum())
        ok &= abs(v_cost - opt_cost) < 1e-9
    assert ok, "spanned Viterbi suboptimal vs brute force"
    print(f"  tiny spanned case n={n} m={m} h={h}: 200 trials Viterbi == brute-force minimum")

    for (n, m, h) in ((14, 5, 10), (20, 12, 3), (33, 12, 4), (64, 20, 5), (65, 20, 10)):
        x = rng.integers(0, 2, n, dtype=np.uint8)
        msg = rng.integers(0, 2, m, dtype=np.uint8)
        rho = rng.uniform(0.5, 5.0, n)
        flip = embed_stc_spanned(x, msg, rho, h=h, seed=0)
        assert np.array_equal(extract_stc_spanned(x ^ flip, m, h=h, seed=0), msg)
        # stride-1 equivalence at the saturated length
        if n == m + h - 1:
            assert np.array_equal(flip, embed_stc(x, msg, rho, h=h, seed=0)), \
                "spanned != stride-1 at n = m + h - 1"
        print(f"  n={n} m={m} h={h}: round trip OK, flip rate {float(flip.mean()):.3f}")
    print("STC self-test PASSED")


def _brute_force_spanned(x_bits, msg, rho, h_vec, starts):
    """Exhaustive min-cost flip mask for tiny spanned schedules."""
    n = x_bits.size
    idx = np.nonzero(h_vec)[0]
    best, best_cost = None, np.inf
    for code in range(1 << n):
        flip = np.array([(code >> i) & 1 for i in range(n)], dtype=np.uint8)
        y = x_bits ^ flip
        synd = np.bitwise_xor.reduce(np.stack([y[starts + k] for k in idx]), axis=0)
        if synd.tolist() != msg.tolist():
            continue
        cost = float((rho * flip).sum())
        if cost < best_cost:
            best_cost, best = cost, flip
    return best, best_cost


if __name__ == "__main__":
    _self_test()