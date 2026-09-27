"""
src/distortion.py — 失真函数接口与图像级嵌入修改代价。

失真函数 ρ 定义: 修改第 i 个像素 ±1 时, 期望载体总失真 (单像素粒度).
对空间域嵌入:  ρ = Σ_dirs|residual_dir(x)|^{-1} (UNIWARD / HILL).
对 JPEG 域嵌入: ρ 在量化 DCT 系数上计算 (J-UNIWARD / J-PHHO).

设计原则:
  - DistortionFunction 是 callable (image) → rho (1D float array, 长度 = 像素数)
  - 嵌入端: 拿到 rho 后, 用 STC 在 (1±) 修改序列上做最小失真编码 (src/stc.py)
  - 提取端: 与嵌入独立, 只需读 LSB

约定:
  - 灰度图 (H, W) → flatten → rho (H*W, )
  - 修改方向 +/-1 各对应一个 cost, 简化假设: 改为 +1 的 cost = 改为 -1 的 cost (Symmetric Cost)
  - 不对称 cost (如 J-UNIWARD 中 ±1 失真不同) 在 [WATCH] 标记, 用 ρ_plus / ρ_minus 两路权重
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class DistortionFunction(Protocol):
    """失真函数接口: image (H, W) uint8 → rho (H*W,) float64."""

    name: str

    def __call__(self, image: np.ndarray) -> np.ndarray:
        ...


# ------------------ 工具: 翻转方向后图像失真增量 ------------------
def delta_to_image(x: np.ndarray, change: np.ndarray) -> np.ndarray:
    """应用 STC 输出的 ±1 修改序列到灰度图."""
    y = x.astype(np.int16).copy()
    flat = y.reshape(-1)
    flip = change.astype(np.int8)
    flat[:] = np.clip(flat + flip, 0, 255)
    return y.astype(np.uint8)


def embed_with_distortion(cover: np.ndarray, payload: np.ndarray, rho: np.ndarray,
                          h: int = 10, wet: np.ndarray | None = None) -> np.ndarray:
    """失真 + STC 一站式嵌入: cover (H,W) uint8 + payload (m,) uint8 + rho (H*W,) → stego (H,W) uint8.

    h: STC 约束高度, 越大越精确越慢 (10 是合理默认). 尺寸通用: 任意 payload 长度
       m <= n - h + 1 均可, 不再要求 n == 2^m * h.
    wet: (H*W,) bool 湿位 (不可修改); 通过把这些像素的代价置为极大实现近似
          (真正的湿纸编码需广义 STC, 本模块不实现).
    """
    try:
        from .stc import embed_stc_on_cover
    except ImportError:
        from stc import embed_stc_on_cover
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    if wet is not None:
        rho = rho.copy()
        rho[np.asarray(wet, dtype=bool).reshape(-1)] = 1e12
    return embed_stc_on_cover(cover, payload, rho, h=h, seed=0).reshape(cover.shape)
