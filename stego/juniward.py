"""
src/juniward.py — 空间域 S-UNIWARD 失真函数。

UNIWARD (Universal Wavelet Relative Distortion) 由 Guo, Ni, Shi (2014) 提出:
通过 Daubechies-8 小波在水平、垂直、对角三个方向上的高通残差,
计算每个像素的修改代价 ρ(i) = Σ_dirs |W_dir(K * x)|^{-1}, 其中 K 为小波分解核.

嵌入接口: 与 src/stc.py 串联.
  ρ = uniward_cost(cover) → stego = stc.embed_stc_on_cover(cover, payload, ρ, h)

注: 本实现为空间域 S-UNIWARD (Spatial-UNIWARD), 不涉及 JPEG 量化.
    J-UNIWARD 在 JPEG DCT 域另写 (见 juniw_jpeg.py, 待扩展).

复杂度: 512×512 单图约 1-3 秒 (取决于实现).
"""
from __future__ import annotations
import numpy as np

# 不强制依赖 PyWavelets: 提供一个纯 numpy 实现的 Daubechies-8 单步卷积.
# 如果用户已装 pywt, 速度更快, 自动切换.
try:
    import pywt
    _HAS_PYWT = True
except Exception:
    _HAS_PYWT = False


# ------------------ 滤波器核 (单步小波: 仅取 LH, HL, HH 高频子带) ------------------
# Daubechies-2 (db2, 4-tap) 正交小波; 分解用可分离的周期卷积 + 降采样.

def _db2_filters() -> tuple[np.ndarray, np.ndarray]:
    """Daubechies-2 (db2, 4-tap) 正交分解滤波器, 返回 (lo, hi)."""
    sq3 = np.sqrt(3.0)
    denom = 4.0 * np.sqrt(2.0)
    lo = np.array([1 + sq3, 3 + sq3, 3 - sq3, 1 - sq3]) / denom
    hi = np.array([1 - sq3, -(3 - sq3), 3 + sq3, -(1 + sq3)]) / denom
    return lo, hi


def _periodic_convolve1d(a: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """沿 axis 做周期卷积 (wrap), 输出与输入同尺寸, 用于可分离小波分解."""
    a = a.astype(np.float64, copy=False)
    kernel = np.asarray(kernel, dtype=np.float64)
    half = (kernel.size - 1) // 2
    out = np.zeros_like(a)
    for i, k in enumerate(kernel):
        out += k * np.roll(a, i - half, axis=axis)
    return out


def _dwt2_single(x: np.ndarray):
    """单层 2D DWT. 输入 (H, W) → 返回 (cA, (cH, cV, cD)), 与 pywt.dwt2 一致.

    可分离实现: 先行 (axis=1) 低/高通后按列降采样, 再列 (axis=0) 低/高通后按行降采样.
    """
    lo, hi = _db2_filters()
    L = _periodic_convolve1d(x, lo, axis=1)[:, ::2]   # 行低通
    H = _periodic_convolve1d(x, hi, axis=1)[:, ::2]   # 行高通
    LL = _periodic_convolve1d(L, lo, axis=0)[::2, :]
    LH = _periodic_convolve1d(L, hi, axis=0)[::2, :]  # cH = 水平边缘
    HL = _periodic_convolve1d(H, lo, axis=0)[::2, :]  # cV = 垂直边缘
    HH = _periodic_convolve1d(H, hi, axis=0)[::2, :]  # cD = 对角
    return LL, (LH, HL, HH)


def _dwt2(x: np.ndarray):
    """返回 (cA, (cH, cV, cD)) 兼容 pywt.dwt2 输出格式."""
    if _HAS_PYWT:
        return pywt.dwt2(x, "db2", mode="periodization")
    return _dwt2_single(x)


def _idwt2(cA, cH, cV, cD):
    """单层 2D 离散小波重构."""
    if _HAS_PYWT:
        return pywt.idwt2((cA, (cH, cV, cD)), "db2", mode="periodization")
    # 纯 numpy 重构: 略, 因 UNIWARD 不需要完整重构 (我们用残差)
    raise NotImplementedError("纯 numpy 路径下请安装 pywt")


def _interpolate_to_full(x: np.ndarray, target_size: tuple) -> np.ndarray:
    """将 half-size 子带 (因 DWT 降采样) 双线性上采样回原图尺寸.

    原先用 0 填充 (out[::2,::2]=x) 会在代价图上产生 2x2 棋盘伪影: 奇偶位置
    的 |W|=0 恒为常数, 使代价图失去纹理区分度. 改用双线性插值得到平滑、忠实
    反映纹理的代价图.
    """
    from scipy.ndimage import zoom
    H, W = target_size
    h, w = x.shape
    if (h, w) == (H, W):
        return x
    return zoom(x, (H / h, W / w), order=1)


# ------------------ 主函数 ------------------
def uniward_cost(cover: np.ndarray, sigma: float = 1e3) -> np.ndarray:
    """空间域 S-UNIWARD 失真代价 ρ (覆盖越小, 嵌入修改越倾向选此位置).

    实现: 在 db2 小波 1 层分解的 LH, HL, HH 三个高频子带上,
          ρ(i) = Σ_dirs |W_dir(i)|^{-1} 的归一化 (除以 σ 平滑).

    输入: cover (H, W) uint8
    输出: rho (H*W,) float64, 与 flat(cover) 顺序一致 (row-major)
    """
    x = cover.astype(np.float64)
    if x.ndim == 3:
        x = x[..., 0]
    H, W = x.shape
    cA, (cH, cV, cD) = _dwt2(x)
    # 平滑子带 (避免 /0): |W| + σ,  σ 越大失真越均匀
    cH_full = _interpolate_to_full(cH, (H, W))
    cV_full = _interpolate_to_full(cV, (H, W))
    cD_full = _interpolate_to_full(cD, (H, W))
    rho = 1.0 / (np.abs(cH_full) + sigma) + 1.0 / (np.abs(cV_full) + sigma) + 1.0 / (np.abs(cD_full) + sigma)
    return rho.reshape(-1)


def embed_uniward(cover: np.ndarray, payload: np.ndarray, h: int = 10, sigma: float = 1e3) -> np.ndarray:
    """一站式 S-UNIWARD + STC 嵌入: cover (H,W) + payload (m,) → stego (H,W).

    尺寸通用: 任意 payload 长度 m <= n - h + 1 (任意码率), 用高度 h 的 STC 通过
    像素置乱嵌入, 再做 ±1 LSB matching.
    """
    try:
        from .stc import embed_stc_on_cover
    except ImportError:
        from stc import embed_stc_on_cover
    rho = uniward_cost(cover, sigma=sigma)
    return embed_stc_on_cover(cover, payload, rho, h=h, seed=0).reshape(cover.shape)


# ------------------ 自检 ------------------
def _self_test():
    """简单验证: 在 64×64 灰度图上嵌入并提取, 检查嵌入+提取一致性."""
    rng = np.random.default_rng(0)
    cover = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    h = 4
    n = 64 * 64
    m = 1024  # 任意 payload 长度 (尺寸通用)
    payload = rng.integers(0, 2, m, dtype=np.uint8)
    stego = embed_uniward(cover, payload, h=h)
    try:
        from .stc import extract_stc_on_cover
    except ImportError:
        from stc import extract_stc_on_cover
    bits = extract_stc_on_cover(stego, m, h=h)
    n_mod = int(np.sum(stego != cover))
    print(f"  S-UNIWARD: 64x64, h={h}, m={m}, payload OK={np.array_equal(bits, payload)}, 修改 {n_mod}/{n} 像素 ({100*n_mod/n:.1f}%)")


if __name__ == "__main__":
    _self_test()
