"""
src/hill.py — 空间域 HILL (High-Low-Low) 失真函数。

HILL 由 Holub 等 (2014) 提出, 用两层高通残差乘积作为嵌入代价:
  ρ(i) = 1 / ( |H_hpf(x)| * (ξ + W_hpf(1/(ξ + B_hpf(x)))) )

其中 H_hpf / W_hpf / B_hpf 分别是:
  H: 强高通 (KB residual) — 抑制纹理但保留噪声
  W: 弱高通 (均值滤波) — 捕获平滑
  B: 巴特沃斯低通 (大致保边)

简化实现 (遵循原始 HILL 思路): 用 3x3 拉普拉斯 + 15x15 均值滤波作为两层.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter, laplace


def _hpf_residual(x: np.ndarray) -> np.ndarray:
    """强高通残差: KB residual kernel (近似, 用 3x3 拉普拉斯替代)."""
    return laplace(x.astype(np.float64))


def _lpf_smooth(x: np.ndarray, size: int) -> np.ndarray:
    """低通 (均值) 滤波."""
    return uniform_filter(x.astype(np.float64), size=size, mode="reflect")


def hill_cost(cover: np.ndarray, xi: float = 1e3) -> np.ndarray:
    """空间域 HILL 失真代价 ρ (H*W,).

    步骤 (简化版):
      R_h = |H_hpf(x)|          强高通残差
      R_w = W_lpf(1/(xi + B_hpf(x)))   平滑区域的弱高通权重
      ρ = 1 / (R_h * (xi + R_w))
    """
    x = cover.astype(np.float64)
    if x.ndim == 3:
        x = x[..., 0]
    # H_hpf: KB (近似为拉普拉斯)
    R_h = np.abs(_hpf_residual(x))
    # B_hpf: 强高通 → 1/(R_h + ξ) → 低通 (15x15 均值)
    inner = 1.0 / (R_h + xi)
    R_w = _lpf_smooth(inner, size=15)
    rho = 1.0 / (R_h * (xi + R_w) + 1e-9)
    return rho.reshape(-1)


def embed_hill(cover: np.ndarray, payload: np.ndarray, h: int = 10, xi: float = 1e3,
               seed: int = 0) -> np.ndarray:
    """一站式 HILL + STC 嵌入: cover + payload → stego.

    对 cover 计算 HILL 失真代价 ρ, 用高度 h 的尺寸通用 STC (Filler) 嵌入任意长度
    payload (任意码率), 像素置乱避免负载前挂, 随后做 ±1 LSB matching (偶数 +1, 奇数 -1).
    """
    try:
        from .stc import embed_stc_on_cover
    except ImportError:
        from stc import embed_stc_on_cover
    rho = hill_cost(cover, xi=xi)
    return embed_stc_on_cover(cover, payload, rho, h=h, seed=seed).reshape(cover.shape)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cover = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    rho = hill_cost(cover)
    print(f"  HILL cost: shape={rho.shape}, range=[{rho.min():.3f}, {rho.max():.3f}], mean={rho.mean():.3f}")
