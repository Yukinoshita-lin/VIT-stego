"""
srm_filter —— SRM (Spatial Rich Model) 高通滤波预处理层。

对图像进行标准 SRM 30 个高统核(线性残差)的高通滤波, 得到突出嵌入噪声、抑制
图像内容的残差图。作为统计特征提取(trainset / GPUtrain)的输入预处理层:

  - 每个核输出一张残差图 (带符号浮点, 值域约 -T..T)
  - 截断 + 平移缩放到 uint8 [0,255]: 增强图
  - 增强图可直接喂给现有 11 维统计特征器 (fsfeatures / featurize_gpu)

核集来源于 Ye-Net 所用标准 SRM 30 核 (Fridrich 体系), 与文献一致:
  filter_class_1(一阶差分,8) + filter_class_2(二阶,4) + filter_class_3(三阶,8)
  + filter_edge_3x3(4) + filter_edge_5x5(4) + square_3x3(1) + square_5x5(1) = 30
参考实现:
  Deep-Steganalysis/models/srm_filter_kernel.py
"""
from __future__ import annotations
import numpy as np

_K = np.float32

# ---- 原始核 (取自已验证的 Ye-Net 实现, 与 srm_filter_kernel.py 逐字一致) ----
_filter_class_1 = [
    _K([[1, 0, 0], [0, -1, 0], [0, 0, 0]]), _K([[0, 1, 0], [0, -1, 0], [0, 0, 0]]),
    _K([[0, 0, 1], [0, -1, 0], [0, 0, 0]]), _K([[0, 0, 0], [1, -1, 0], [0, 0, 0]]),
    _K([[0, 0, 0], [0, -1, 1], [0, 0, 0]]), _K([[0, 0, 0], [0, -1, 0], [1, 0, 0]]),
    _K([[0, 0, 0], [0, -1, 0], [0, 1, 0]]), _K([[0, 0, 0], [0, -1, 0], [0, 0, 1]]),
]
_filter_class_2 = [
    _K([[1, 0, 0], [0, -2, 0], [0, 0, 1]]), _K([[0, 1, 0], [0, -2, 0], [0, 1, 0]]),
    _K([[0, 0, 1], [0, -2, 0], [1, 0, 0]]), _K([[0, 0, 0], [1, -2, 1], [0, 0, 0]]),
]
_filter_class_3 = [
    _K([[-1, 0, 0, 0, 0], [0, 3, 0, 0, 0], [0, 0, -3, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]),
    _K([[0, 0, -1, 0, 0], [0, 0, 3, 0, 0], [0, 0, -3, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]),
    _K([[0, 0, 0, 0, -1], [0, 0, 0, 3, 0], [0, 0, -3, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]),
    _K([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -3, 3, -1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    _K([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, -3, 0, 0], [0, 0, 0, 3, 0], [0, 0, 0, 0, -1]]),
    _K([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -3, 0, 0], [0, 0, 3, 0, 0], [0, 0, -1, 0, 0]]),
    _K([[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, -3, 0, 0], [0, 3, 0, 0, 0], [-1, 0, 0, 0, 0]]),
    _K([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 3, -3, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
]
_filter_edge_3x3 = [
    _K([[-1, 2, -1], [2, -4, 2], [0, 0, 0]]), _K([[0, 2, -1], [0, -4, 2], [0, 2, -1]]),
    _K([[0, 0, 0], [2, -4, 2], [-1, 2, -1]]), _K([[-1, 2, 0], [2, -4, 0], [-1, 2, 0]]),
]
_filter_edge_5x5 = [
    _K([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    _K([[0, 0, -2, 2, -1], [0, 0, 8, -6, 2], [0, 0, -12, 8, -2], [0, 0, 8, -6, 2], [0, 0, -2, 2, -1]]),
    _K([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]),
    _K([[-1, 2, -2, 0, 0], [2, -6, 8, 0, 0], [-2, 8, -12, 0, 0], [2, -6, 8, 0, 0], [-1, 2, -2, 0, 0]]),
]
_square_3x3 = _K([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]])
_square_5x5 = _K([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                  [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]])

# 归一化因子 (与 all_normalized_hpf_list 一致): 类2÷2, 类3÷3, edge3x3÷4,
# edge5x5÷12, square3x3÷4, square5x5÷12; 类1 不归一。
NORM = [
    1.0] * 8 + \
    [2.0] * 4 + \
    [3.0] * 8 + \
    [4.0] * 4 + \
    [12.0] * 4 + \
    [4.0, 12.0]

_RAW_KERNELS = (_filter_class_1 + _filter_class_2 + _filter_class_3 +
                _filter_edge_3x3 + _filter_edge_5x5 + [_square_3x3, _square_5x5])

# 归一化后的 30 个核: 每个都补齐到 5x5 (3x3 居中)。
SRM_KERNELS = np.asarray([
    np.pad(k / n, ((1, 1), (1, 1))) if k.shape[0] < 5 else k / n
    for k, n in zip(_RAW_KERNELS, NORM)
], dtype=np.float32)          # (30, 5, 5)
SRM_NAMES = [f"srm{i:02d}" for i in range(30)]

# 默认残差截断阈值 (SRM 线性残差典型范围)
DEFAULT_T = 4.0


def srm_residuals_np(x_u8: np.ndarray, T: float = DEFAULT_T, pad: int = 2):
    """CPU (numpy) 版: 对 (H,W) 或 (N,H,W) uint8 灰度图计算 30 个高通残差。

    返回 (N, 30, H, W) 或 (30, H, W) float32 残差(未截断)。采用零填充(与
    torch conv padding=2 对齐), 输出与输入同尺寸。
    """
    single = x_u8.ndim == 2
    x = np.asarray(x_u8, dtype=np.float32)[None] if single else np.asarray(x_u8, dtype=np.float32)
    N, H, W = x.shape
    out = np.empty((N, SRM_KERNELS.shape[0], H, W), dtype=np.float32)
    for c in range(SRM_KERNELS.shape[0]):
        k = SRM_KERNELS[c]
        # 手动过滤: 零填充后逐位置卷积
        p = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
        acc = np.zeros((N, H, W), dtype=np.float32)
        for di in range(5):
            for dj in range(5):
                w = k[di, dj]
                if w != 0.0:
                    acc += w * p[:, di:di + H, dj:dj + W]
        out[:, c] = acc
    return out[0] if single else out


def srm_residuals_torch(x_u8, T: float = DEFAULT_T, device: str = "auto"):
    """GPU (torch) 版: 批量卷积 30 核, 返回 (N, 30, H, W) float32 残差。

    x_u8: (N, H, W) uint8 numpy 或 torch。残差未截断(值域约 -T..T ± 内容项)。
    """
    import torch
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    xt = torch.from_numpy(np.ascontiguousarray(x_u8, dtype=np.uint8)) if not torch.is_tensor(x_u8) else x_u8
    if xt.ndim == 2:
        xt = xt[None]
    xf = xt.float().unsqueeze(1).to(dev)                 # (N,1,H,W)
    w = torch.from_numpy(SRM_KERNELS[:, None, :, :]).to(dev)  # (30,1,5,5)
    r = torch.nn.functional.conv2d(xf, w, padding=2)     # (N,30,H,W)
    return r


def residuals_to_u8(res: np.ndarray, T: float = DEFAULT_T) -> np.ndarray:
    """残差[clip在 ±T] → 平移 + 缩放 → uint8 增强图。

    r∈[-T,T] → u = round((r+T)/(2T)*255), 取值 [0,255]。输入 (...,H,W)。
    """
    r = np.asarray(res)
    r = np.clip(r, -T, T)
    u = ((r + T) / (2 * T) * 255.0)
    return np.round(u).astype(np.uint8)


def get_preprocessor(T: float = DEFAULT_T):
    """返回一个可调用 pre(x_u8)->uint8增强图, 供整批/单张共用。内部按 np path。"""
    def pre(x_u8):
        res = srm_residuals_np(x_u8)
        return residuals_to_u8(res, T)
    return pre


def preprocess_for_features(x_u8, use_srm: bool = True, T: float = DEFAULT_T):
    """预处理统一入口(numpy): 返回喂给统计特征器的 uint8 图。

    use_srm=True  → 逐像素取 30 个残差的最大(绝对)响应, 合成一张 SRM 增强图,
                    突出局部最强的高通噪声方向, 抑制平坦内容。
    use_srm=False → 原图原样返回。
    """
    if not use_srm:
        return x_u8
    x = np.asarray(x_u8)
    r = srm_residuals_np(x)
    mag = np.max(np.abs(r), axis=-3)          # (N,H,W) 或 (H,W)
    return residuals_to_u8(mag, T)


def preprocess_batch_torch(x_u8, T: float = DEFAULT_T, device: str = "auto"):
    """GPU 批量合成 SRM 增强图: (N,H,W) uint8 -> (N,H,W) uint8。

    用 torch conv 并行算 30 残差, 逐像素取最大绝对响应后 clip+缩放。
    供 featurize_gpu 在 chunk 内复用同一设备张量, 避免来回拷贝。
    """
    import torch
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    xt = torch.from_numpy(np.ascontiguousarray(x_u8, dtype=np.uint8)) if not torch.is_tensor(x_u8) else x_u8
    if xt.ndim == 2:
        xt = xt[None]
    xf = xt.float().unsqueeze(1).to(dev)                 # (N,1,H,W)
    w = torch.from_numpy(SRM_KERNELS[:, None, :, :]).to(dev)
    r = torch.nn.functional.conv2d(xf, w, padding=2)     # (N,30,H,W)
    mag = r.abs().amax(dim=1).clamp(-T, T)               # (N,H,W)
    u = ((mag + T) / (2 * T) * 255.0).round()
    return u.detach().cpu().to(torch.uint8).numpy()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    im = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    r = srm_residuals_np(im)
    print("残差 shape:", r.shape, "值域: [%.3f, %.3f]" % (r.min(), r.max()))
    print("归一化核平方和(第1/9/17/25/29/30 例):",
          [round(float((SRM_KERNELS[i] ** 2).sum()), 3) for i in (0, 8, 16, 24, 28, 29)])
    print("残差→增强图 shape:", residuals_to_u8(r[0]).shape)