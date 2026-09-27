# VIT-Stego

**Lightweight Vision Transformers for image steganalysis** — a self-contained,
CPU-friendly research project that trains a compact ViT to separate cover images
from stego, building toward a deployable lightweight detector.

## Headline results

Spatial-domain steganalysis at 0.4 bpp on 2000 BOSSbase covers, held-out **test**
AUC (mean ± std over 3 seeds, 60 epochs, `scripts/benchmark.py`). Every row is
measured on the same split protocol and the same data — the ablation tables below
use the smaller 512-cover set and are not comparable row-for-row:

| Detector | Params | Embedding | test AUC | test acc |
|---|---|---|---|---|
| LightViT, raw pixels (no SRM) | 478,657 | HILL adaptive ±1 | 0.5004 ± 0.0003 | 0.5004 |
| LightViT + SRM stem | 657,583 | HILL adaptive ±1 | 0.9094 ± 0.0086 | 0.8092 |
| LightViT + SRM stem | 657,583 | LSB replacement | 0.7530 ± 0.0116 | 0.6687 |
| LightViT + SRM stem | 657,583 | HILL + STC | 0.7507 ± 0.0158 | 0.6725 |
| **LightViT-lite + SRM stem** | **240,367** | HILL adaptive ±1 | **0.9036 ± 0.0188** | 0.8171 |
| LightViT-mini + SRM stem | 95,983 | HILL adaptive ±1 | 0.8873 ± 0.0151 | 0.7983 |

Two findings drive the project:

1. **A plain ViT over raw pixels cannot see a ±1 LSB perturbation** — it sits at
   exactly chance (AUC 0.5004) even with 2000 covers and 60 epochs. A frozen
   **SRM-30 residual stem** in front of the patch embedding is what makes the
   detector work at all.
2. **The detector wants to be small.** The 240K-parameter default is 0.006 AUC
   behind the 658K-parameter model — well inside the ±0.019 seed-to-seed spread,
   i.e. indistinguishable — at 2.7× fewer parameters. A 96K-parameter model still
   reaches 0.887 (see [Lightweight trade-off](#lightweight-trade-off)).

The harder, minimum-distortion embeddings (STC, LSB replacement) are markedly
harder to detect than the naive ±1 adaptive embedder, which perturbs 40% of the
pixels and leaves the largest footprint.

## Evaluation protocol

Getting this right changed every number in this repository, so it is worth
stating precisely. A run is scored with:

- **A three-way split.** `val` picks the best epoch, `test` is scored once
  afterwards. Reporting the max-over-epochs *validation* AUC is an optimistic
  estimate — with a few hundred validation samples the selection noise alone is
  worth several AUC points.
- **Cover and stego of the same source image always land on the same side**
  (`src/engine.py:make_splits`). The images differ in ~40% of pixels by ±1 and
  carry opposite labels, so splitting them independently leaks near-duplicates
  across the boundary — see [Correction](#correction-the-earlier-split-was-leaky).
- **Mean ± std over ≥3 seeds** (`scripts/benchmark.py`). A single split on a
  512-cover dataset carries ~±0.05 of pure split noise.
- **Raw pixels on the [0, 255] scale.** The signal is a 1-LSB change; `[0,1]`
  normalisation destroys it.

### Correction: the earlier split was leaky

Through milestone M3 the cover and stego halves were permuted *independently*:

```python
cover_idx = rng.permutation(n); val_covers = cover_idx[:n_val]
stego_idx = n + rng.permutation(n); val_stegos = stego_idx[:n_val]
```

The twin of a validation image then sits in the training set with probability
`1 - val_frac` — 0.73 in the 512-cover runs. `scripts/diagnose_split.py` trains
one model on the legacy training set and tracks both evaluation sets every epoch:

```
epoch 15 | legacy-val AUC 0.6328 | disjoint-val AUC 0.8498
```

Same weights, same crops, same epoch — **0.217 AUC apart**. The leak also breaks
best-epoch selection, because the legacy validation curve stays near chance and
so keeps an essentially untrained checkpoint.

The M1 headline (`AUC 0.657`) was measured under that split. The architecture was
fine; the measurement was not. Everything below is re-measured.

## What it does

- Reuses BOSSbase covers and the HILL distortion function from the sibling
  `ood-robust-steganalysis` project.
- Generates stego three ways:
  - **HILL adaptive** — a content-adaptive ±1 embedder driven by the HILL cost
    (`--method adaptive`),
  - **HILL + STC** — Filler's syndrome-trellis code (minimum-distortion,
    size-general) over the HILL cost, applied with ±1 LSB matching
    (`--method stc`), and
  - **LSB replacement** — the classic histogram-equalization sanity baseline
    (`--method lsb`).
- Trains `LightViT`: patch embedding + transformer blocks + a classification head.
- An **SRNet baseline** (bottleneck ResNet, ~1.35M params) is included under
  `--model srnet` for like-for-like comparison.

## Results

### Ablations (512 covers, `data/bossbase_hill.npz`, 30 epochs, 3 seeds)

Each row adds one change to the row above it:

| Change | SRM | truncate | pool | augment | test AUC | test acc |
|---|---|---|---|---|---|---|
| raw pixels, no SRM | no | – | cls | yes | 0.5014 ± 0.0034 | 0.5000 |
| + SRM-30 stem | yes | 0 | cls | yes | 0.8551 ± 0.0171 | 0.7614 |
| + residual truncation T=4 | yes | 4 | cls | yes | 0.9035 ± 0.0255 | 0.7647 |
| + mean pooling | yes | 4 | mean | yes | **0.9187 ± 0.0153** | 0.8252 |
| − random crop & flips | yes | 4 | cls | no | 0.8228 ± 0.0013 | 0.7157 |

- **SRM stem: +0.354 AUC.** The one change that matters most.
- **Truncation `T=4`: +0.048.** Clamping the residual to ±4 bounds the influence of
  content edges, whose residuals are far larger than the ±1 embedding step.
  T=4 matches `srm_filter.DEFAULT_T`; T=2 and T=6 give 0.9070 and 0.9036.
- **Mean pooling over patch tokens: +0.037.** Global statistics beat a single
  CLS token when every patch carries the same weak signal.
- **Augmentation: +0.032** on 128-px crops (flips only), **+0.046** when a
  256-px stored crop can also be randomly positioned (same 30-epoch budget on the
  2000-cover set: 0.8950 ± 0.0081 with, 0.8487 ± 0.0092 without).

### Lightweight trade-off

All rows use `--srm --truncate 4 --pool mean`. The 512-cover sweep (30 epochs) and
its confirmation on 2000 covers (60 epochs):

| embed_dim | depth | heads | Params | 512 covers | 2000 covers |
|---|---|---|---|---|---|
| 32 | 2 | 2 | **95,983** | 0.9015 ± 0.0146 | 0.8873 ± 0.0151 |
| 48 | 3 | 3 | 190,303 | 0.9197 ± 0.0197 | – |
| **64** | **2** | **4** | **240,367** | **0.9208 ± 0.0119** | 0.9036 ± 0.0188 |
| 64 | 4 | 4 | 340,335 | 0.9132 ± 0.0113 | – |
| 96 | 4 | 3 | 657,583 | 0.9187 ± 0.0153 | 0.9094 ± 0.0086 |

**2.7× fewer parameters, no measurable loss.** On 512 covers the 240K model is
nominally ahead; on the larger and more reliable 2000-cover benchmark it is 0.006
behind — smaller than one standard error. Capacity plateaus from ~190K up, so the
240K configuration is the CLI default (all three entry scripts). The 96K model
stays near 0.90 AUC where the budget is tighter than the accuracy target.

The table also shows why the larger cover set matters: the 2000-cover numbers are
consistently lower and much better separated between configurations. The
512-cover test splits (204 samples) were flattering by a few AUC points.

### Effect of more data and random-crop augmentation

`data/bossbase_*_256.npz` stores 2000 covers as 256×256 crops; training draws a
random 128×128 position with flips (16 positions × 4 flips = 64 views per cover).
60 epochs, 3 seeds:

| Embedding | Dataset | SRM | augment | test AUC | test acc |
|---|---|---|---|---|---|
| HILL adaptive ±1 (0.4 bpp) | 2000 covers, 256 crop | yes | yes | **0.9094 ± 0.0086** | 0.8092 |
| HILL adaptive ±1 (0.4 bpp) | 2000 covers, 256 crop | yes | no | 0.8487 ± 0.0092 | 0.7583 |
| LSB replacement (0.4) | 2000 covers, 256 crop | yes | yes | 0.7530 ± 0.0116 | 0.6687 |
| HILL + STC (0.4 bpp) | 2000 covers, 256 crop | yes | yes | 0.7507 ± 0.0158 | 0.6725 |

> HILL + STC at 0.4 bpp modifies 19.9% of pixels with max |Δ| = 1 — the ±1
> LSB-matching profile a minimum-distortion size-general STC should produce.

### MAE self-supervised pre-training: no measurable gain

Milestone 4 pre-trains the encoder to reconstruct masked patches of the 30-channel
SRM residual of covers, then transfers it. On 512 covers it is a wash —
0.9180 ± 0.0108 vs 0.9187 ± 0.0153 without it. Reported as a negative result:
the reconstruction objective does not align closely enough with the ±1 detection
task at this scale. The machinery is kept because it is cheap to re-test on
larger cover sets (`scripts/pretrain_mae.py`).

### OOD, weak payloads and mismatched embedders (M7)

`scripts/eval_ood.py` scores a trained checkpoint — without retraining — on the
held-out test split of *other* datasets. Every .npz is built from the same
BOSSbase covers in the same order, so the same seed selects the same source
images everywhere and the drop from row to row is attributable to the embedding
change alone. The HILL-0.4-trained lite model (240K params, test AUC 0.8994
in distribution):

| Evaluation | Embedding | test AUC | test acc |
|---|---|---|---|
| in distribution | HILL adaptive @0.4 bpp | **0.8994** | 0.8037 |
| weak payload | HILL adaptive @0.2 bpp | 0.7548 | 0.6787 |
| weak payload | HILL adaptive @0.1 bpp | 0.6330 | 0.5800 |
| mismatched embedder | HILL + STC @0.4 bpp | 0.7474 | 0.6737 |
| mismatched embedder | LSB replacement @0.4 | 0.7489 | 0.6775 |
| cross-domain | nsF5, JPEG q75 @0.4 bpnc | 0.5315 | 0.5188 |

Degradation is graceful down to 0.2 bpp and the embedder mismatch costs ~0.15
AUC — but the JPEG-domain row is the important one: **the spatial detector does
not transfer to JPEG steganography at all** (0.53 ≈ chance). A deployed
detector must be trained per domain, which is what M8 sets up.

Retraining on the weak payloads instead of transferring does not help at this
data scale (2000 covers, 60 epochs, 3 seeds — all rows `--srm --truncate 4
--pool mean`): 0.2 bpp retrained reaches 0.7341 ± 0.0121 (the 0.4-model
transfers to 0.7548 — indistinguishable), and 0.1 bpp retrained stays near
chance at 0.5988 ± 0.0014. With ~1300 training pairs the bottleneck is data,
not the checkpoint's operating point.

### JPEG-domain steganography (M8)

The realistic deployment case: JPEG steganography lives in the quantized-DCT
coefficients, and a deployed detector holds the JPEG file. `prepare_data.py`
gains two embedders plus a JPEG-domain dataset format:

- **nsF5** (non-shrinkage F5, `stego/nsf5.py`) — payload into the *nonzero* AC
  coefficients with parity flips that never create a zero (|c| = 1 moves away
  from zero, otherwise toward zero), so the receiver identifies the carrier
  set from the stego alone; adaptive Hamming matrix embedding picks the
  largest code height that fits (0.4 bpnc → (7,3), 1.0 bpnc → (1,1,1)). The
  payload unit is **bpnc** (bits per nonzero AC), the convention the
  literature quotes, because a fixed bpp does not fit every cover: at q75 the
  nonzero-AC count per BOSSbase 256-crop ranges 5.5k–33k.
- **J-UNIWARD** (`stego/juniward_jpeg.py`) — STC over the LSBs of all 63 AC
  coefficients per block, cost = the UNIWARD wavelet-residual change a ±1
  coefficient change induces on the decompressed cover. The Daubechies-8
  filter is reconstructed by spectral factorization (no pywt needed), and the
  cost map is computed with a linear-response trick: the cover's subbands are
  computed once and every coefficient's ±1 response is a fixed-shape window,
  so the whole map is one batched gather per coefficient position.
- **Dataset format** — the .npz stores the decompressed JPEG cover/stego
  *and* the quantized-DCT maps (`covers_c`/`stegos_c`, DC zeroed, ACs clipped
  to ±30). Training on the coefficient maps is `--input coeff`: no SRM stem
  (JPEG quantization already high-passes the input), 120,833 parameters.

> **Correction: the earlier STC was saturated.** `embed_stc` pins syndrome row
> *i* to column *i*, forcing n = m + h − 1: the code runs at ~100% relative
> rate, the Viterbi has no freedom, the flip rate locks at ~50% of the active
> window and the distortion function cannot influence anything. This silently
> shaped the spatial HILL+STC row above (19.9% change ≈ half the active
> window). The new `embed_stc_spanned` spreads the rows via
> `start_i = floor(i·(n−h+1)/m)`; at n = m + h − 1 it is bit-identical to the
> old code, and at relative rate 0.5 the STC actually minimizes cost — the
> changed coefficients' mean J-UNIWARD cost drops to half the all-coefficient
> mean (160 vs 328).

Results (2000 covers, 60 epochs, 3 seeds):

| Input | Embedding | Params | test AUC | test acc |
|---|---|---|---|---|
| pixels (decompressed) | nsF5 0.4 bpnc q75 | 240,367 | 0.5406 ± 0.0040 | 0.5196 |
| coeff map | nsF5 0.4 bpnc q75 | 120,833 | 0.5262 ± 0.0055 | 0.5075 |
| coeff map | nsF5 1.0 bpnc q75 | 120,833 | **0.9876 ± 0.0099** | 0.9408 |
| coeff map | J-UNIWARD 0.4 bpp q75 | 120,833 | **1.0000 ± 0.0000** | 0.9962 |
| pixels | J-UNIWARD 0.4 bpp q75 | 240,367 | **1.0000 ± 0.0000** | 1.0000 |

nsF5 at 0.4 bpnc modifies on average only 1,271 of 65,536 coefficients per
image (range 145–3,743) — under this project's data budget it is undetectable
in either representation, recorded as a negative result. At 1.0 bpnc (~8.8% of
coefficients change) the coefficient-domain detector reaches 0.988, and
J-UNIWARD at 0.4 bpp (~9,900 changed coefficients, 8× more) saturates both
representations: the lightweight ViT detects JPEG-domain steganography — in
the coefficient domain, and, when the footprint is large enough, even from the
decompressed pixels.

**Cross-embedder transfer fails inside the JPEG domain.** The J-UNIWARD
coefficient model applied to both nsF5 datasets scores exactly chance (0.4997
/ 0.4996) — the two embedders flip coefficients in genuinely different ways
(STC's cost-shaped spread over all ACs vs nsF5's parity flips on nonzero ACs),
and neither transfers to the other. Detectors are embedder-specific even
within a domain, reinforcing the M7 conclusion. The one positive transfer
found: the 1.0-bpnc coefficient model applied to 0.4 bpnc scores AUC 0.7345 —
far above the 0.5262 of direct 0.4-bpnc training (accuracy 0.549 is a
threshold shift, the ranking survives): **pretraining on a strong payload and
deploying on weak payloads beats direct weak-payload training** at this data
scale.

## Quickstart

```bash
# 1. build cover/stego datasets (2000 BOSSbase covers, stored as 256x256 crops)
python scripts/prepare_data.py --n 2000 --size 256 --rate 0.4 --method adaptive --out data/bossbase_hill_256.npz
python scripts/prepare_data.py --n 2000 --size 256 --rate 0.4 --method lsb      --out data/bossbase_lsb_256.npz
python scripts/prepare_data.py --n 2000 --size 256 --rate 0.4 --method stc --h 10 --out data/bossbase_hill_stc_256.npz

# 2. train the 240K-parameter detector (random 128 crops from the 256 stored)
python scripts/train.py --data data/bossbase_hill_256.npz --srm --truncate 4 --pool mean --epochs 60 --out models/lightvit_lite_hill.pt

# 3. score a saved checkpoint on the held-out test split
python scripts/evaluate.py --data data/bossbase_hill_256.npz --ckpt models/lightvit_lite_hill.pt --srm --truncate 4 --pool mean

# 4. multi-seed benchmark (the number to quote)
python scripts/benchmark.py --data data/bossbase_hill_256.npz --srm --truncate 4 --pool mean --seeds 0,1,2 --epochs 60 --tag 2000_hill

# 5. see what the old split did
python scripts/diagnose_split.py --data data/bossbase_hill.npz --epochs 30

# 6. OOD / weak-density / mismatched-embedder evaluation of a checkpoint
python scripts/eval_ood.py --ckpt models/lightvit_lite_hill.pt --tag lite_hill_ood \
    --data data/bossbase_hill_256.npz data/bossbase_hill_256_r02.npz data/bossbase_hill_256_r01.npz \
           data/bossbase_hill_stc_256.npz data/bossbase_lsb_256.npz

# 7. JPEG-domain: build nsF5 / J-UNIWARD datasets (cover = decompressed JPEG;
#    the .npz also stores the quantized-DCT maps for --input coeff training)
python scripts/prepare_data.py --n 2000 --size 256 --rate 1.0 --rate-unit bpnc --method nsf5 --quality 75 --out data/bossbase_nsf5_256_q75_bpnc10.npz
python scripts/prepare_data.py --n 2000 --size 256 --rate 0.4 --method juniward --quality 75 --out data/bossbase_juniward_256_q75.npz

# 8. train the JPEG-domain detector on the coefficient maps (121K params, no SRM stem)
python scripts/benchmark.py --data data/bossbase_nsf5_256_q75_bpnc10.npz --input coeff --pool mean \
    --epochs 60 --seeds 0,1,2 --tag 2000_nsf5_q75_bpnc10_coeff_lite

# 9. render every record in exp/ as a table (benchmarks + OOD)
python scripts/summarize.py

# 10. self-checks (no pytest needed)
python tests/test_pipeline.py
```

## Layout

```
src/vit_stego.py         # LightViT + SRMStem (frozen SRM-30 front-end) + SDPA attention
src/srnet.py             # SRNet baseline (bottleneck ResNet, optional SRM stem)
src/mae.py               # MAE encoder + transfer into LightViT
src/srm_filter.py        # SRM 30 high-pass kernels (Fridrich), numpy + torch
src/dataset.py           # CoverStegoDataset (random-crop / flip augmentation, deterministic eval crop, pixel/coeff input)
src/engine.py            # splits, train loop, AUC evaluation — shared by every entry point
scripts/prepare_data.py  # cover crop + HILL-adaptive / LSB / HILL+STC / nsF5 / J-UNIWARD embedding
scripts/train.py         # single-run training, reports held-out test AUC
scripts/benchmark.py     # multi-seed mean ± std -> exp/<tag>.json
scripts/evaluate.py      # score a checkpoint on the val/test split
scripts/eval_ood.py      # OOD / weak-payload / mismatched-embedder transfer of a checkpoint
scripts/diagnose_split.py# quantify the legacy split's leakage
scripts/summarize.py     # exp/*.json -> markdown tables (benchmarks + OOD)
scripts/pretrain_mae.py  # MAE self-supervised pre-training on covers
tests/test_pipeline.py   # split / dataset / stem / attention / embedder / JPEG self-checks
stego/                   # distortion functions (HILL, STC, J-UNIWARD) + JPEG domain (jpeg, nsf5, juniward_jpeg)
data/                    # generated .npz (gitignored)
models/                  # checkpoints (gitignored, regenerable)
exp/                     # benchmark records (gitignored)
```

> The `models/lightvit_*.pt` files predating M5 were trained under the leaky split
> with the old defaults. Only `models/lightvit_lite_*.pt`,
> `models/srnet_hill_256.pt` and `models/mae_encoder_pretrained.pt` match the
> configuration in this README; `scripts/evaluate.py` reads the sidecar
> `<ckpt>.json` written by `train.py` so a checkpoint is scored with the settings
> it was trained with rather than the CLI defaults.

## Roadmap

- [x] M0: plain lightweight ViT over raw pixels — chance level (AUC 0.5014), confirmed
      under the corrected split.
- [x] M1: SRM-30 high-pass residual stem — the change that makes detection work.
- [x] M2: SRNet baseline for like-for-like comparison.
- [x] M3: proper size-general STC embedding (Filler), replacing the ±1 placeholder.
- [x] M4: MAE self-supervised pre-training — implemented, **no measurable gain** at
      this scale; kept for re-testing on larger cover sets.
- [x] M5: evaluation protocol — three-way split, pair-preserving, multi-seed,
      held-out test metric. (This is what re-valued every earlier result.)
- [x] M6: architecture ablation — truncation, pooling, augmentation, and the
      parameter/accuracy curve down to 96K parameters.
- [x] M7: OOD / cross-source / weak-density evaluation — transfer to 0.2/0.1 bpp
      and mismatched embedders (`scripts/eval_ood.py`), plus retrained
      weak-payload operating points.
- [x] M8: JPEG-domain embedding — nsF5 and J-UNIWARD in the quantized-DCT
      domain (`stego/jpeg.py`, `stego/nsf5.py`, `stego/juniward_jpeg.py`), with
      the coefficient-map input path (`--input coeff`) that makes them
      detectable at 121K parameters.

> Note: `stego/stc.py` is a size-general binary STC — the parity-check matrix is
> banded so any payload length m ≤ n−h+1 embeds into any cover size. It ships
> with its own round-trip and brute-force-optimality self-test
> (`python stego/stc.py`).

## License

Apache-2.0 — see [LICENSE](LICENSE).

---

# VIT-Stego（中文）

**隐写分析的轻量级 ViT** —— 一个自包含、CPU 友好的研究项目，训练紧凑的 Vision
Transformer 来区分封面图与含密图，目标是做出可部署的轻量检测器。

## 主要结果

空间域隐写分析，0.4 bpp，2000 张 BOSSbase 封面，held-out **test** AUC（3 个 seed 的
均值 ± 标准差，60 epochs，由 `scripts/benchmark.py` 产出）。每一行都在相同的划分协议与
相同数据上测得——下方消融表使用较小的 512 封面集，不可逐行比较：

| 检测器 | 参数 | 嵌入方式 | test AUC | test acc |
|---|---|---|---|---|
| LightViT，原始像素（无 SRM） | 478,657 | HILL 自适应 ±1 | 0.5004 ± 0.0003 | 0.5004 |
| LightViT + SRM stem | 657,583 | HILL 自适应 ±1 | 0.9094 ± 0.0086 | 0.8092 |
| LightViT + SRM stem | 657,583 | LSB 替换 | 0.7530 ± 0.0116 | 0.6687 |
| LightViT + SRM stem | 657,583 | HILL + STC | 0.7507 ± 0.0158 | 0.6725 |
| **LightViT-lite + SRM stem** | **240,367** | HILL 自适应 ±1 | **0.9036 ± 0.0188** | 0.8171 |
| LightViT-mini + SRM stem | 95,983 | HILL 自适应 ±1 | 0.8873 ± 0.0151 | 0.7983 |

两条主导结论：

1. **纯 ViT 直接吃原始像素看不见 ±1 的 LSB 扰动** —— 即便有 2000 张封面、训练 60 轮，
   仍恰好停在随机水平（AUC 0.5004）。前置一个**冻结的 SRM-30 残差 stem** 才是检测器
   能工作的原因。
2. **检测器越轻越好。** 24 万参数的默认配置比 66 万参数版本低 0.006 AUC——远小于
   seed 间的 ±0.019 波动，即无法区分——而参数量少 2.7 倍。9.6 万参数模型仍达 0.887
   （见[轻量化权衡](#轻量化权衡)）。

最小失真的嵌入（STC、LSB 替换）明显比朴素的 ±1 自适应嵌入更难检测——后者改动了
40% 的像素，留下的足迹最大。

## 评测协议

这一节的修正改变了本仓库此前的每一个数字，所以值得写清楚。一次运行按如下方式评分：

- **三段划分。** `val` 只用于挑选最佳 epoch，`test` 在其后评分一次。把「跨 epoch 取
  最大」的*验证集*AUC 当作上报指标是乐观估计——几百个验证样本下，仅选择噪声就值
  好几个 AUC 点。
- **同一张源图的封面与含密图必须落在同一侧**（`src/engine.py:make_splits`）。两者
  约 40% 的像素相差 ±1 却标签相反，独立划分会把近似重复样本泄漏到边界两侧——
  见[修正](#修正此前的划分存在泄漏)。
- **至少 3 个 seed 的均值 ± 标准差**（`scripts/benchmark.py`）。512 张封面的单一
  划分本身就有约 ±0.05 的噪声。
- **像素保持 [0, 255] 原始量纲。** 信号是 1 个 LSB 的变化，`[0,1]` 归一化会把它抹掉。

### 修正：此前的划分存在泄漏

直到里程碑 M3，封面与含密图两半是*各自独立*置换的：

```python
cover_idx = rng.permutation(n); val_covers = cover_idx[:n_val]
stego_idx = n + rng.permutation(n); val_stegos = stego_idx[:n_val]
```

于是一张验证图的「孪生样本」有 `1 - val_frac` 的概率落在训练集里——在 512 封面的
运行中是 0.73。`scripts/diagnose_split.py` 在旧训练集上训练一个模型，每个 epoch 同时
跟踪两个评测集：

```
epoch 15 | legacy-val AUC 0.6328 | disjoint-val AUC 0.8498
```

同样的权重、同样的裁剪、同一轮——**相差 0.217 AUC**。泄漏还会破坏最佳 epoch 选择：
旧验证曲线始终贴近随机，于是选出的几乎是未训练的 checkpoint。

M1 的头条数字（`AUC 0.657`）正是在该划分下测得的。架构没问题，是评测出了问题。
下面所有数字均已重测。

## 功能

- 复用姊妹项目 `ood-robust-steganalysis` 的 BOSSbase 封面图与 HILL 失真函数。
- 三种方式生成含密图：
  - **HILL 自适应** —— 由 HILL 代价驱动的内容自适应 ±1 嵌入器（`--method adaptive`）；
  - **HILL + STC** —— Filler 的伴随式网格码（最小失真、尺寸通用）作用于 HILL 代价，
    并做 ±1 LSB matching（`--method stc`）；
  - **LSB 替换** —— 经典的直方图均衡化 sanity 基线（`--method lsb`）。
- 训练 `LightViT`：patch 嵌入 + transformer 块 + 分类头。
- 附带 **SRNet 基线**（瓶颈 ResNet，约 1.35M 参数，`--model srnet`）用于公平对比。

## 结果

### 消融（512 封面，`data/bossbase_hill.npz`，30 epochs，3 seeds）

每一行在上一行的基础上加一项改动：

| 改动 | SRM | truncate | pool | 增强 | test AUC | test acc |
|---|---|---|---|---|---|---|
| 原始像素，无 SRM | 否 | – | cls | 是 | 0.5014 ± 0.0034 | 0.5000 |
| + SRM-30 stem | 是 | 0 | cls | 是 | 0.8551 ± 0.0171 | 0.7614 |
| + 残差截断 T=4 | 是 | 4 | cls | 是 | 0.9035 ± 0.0255 | 0.7647 |
| + 均值池化 | 是 | 4 | mean | 是 | **0.9187 ± 0.0153** | 0.8252 |
| − 随机裁剪与翻转 | 是 | 4 | cls | 否 | 0.8228 ± 0.0013 | 0.7157 |

- **SRM stem：+0.354 AUC。** 最关键的一项改动。
- **截断 `T=4`：+0.048。** 把残差限制在 ±4 内，抑制内容边缘的影响——边缘残差远大于
  ±1 的嵌入步长。T=4 与 `srm_filter.DEFAULT_T` 一致；T=2 与 T=6 分别为 0.9070、0.9036。
- **对 patch token 做均值池化：+0.037。** 当每个 patch 都携带同样微弱的信号时，
  全局统计量优于单个 CLS token。
- **数据增强：** 128 像素裁剪下（仅翻转）**+0.032**；若存储 256 像素裁剪、还可随机
  取位，则**+0.046**（同为 2000 封面、30 epochs 预算：带增强 0.8950 ± 0.0081，
  不带 0.8487 ± 0.0092）。

### 轻量化权衡

所有行均使用 `--srm --truncate 4 --pool mean`。512 封面扫描（30 epochs）及其在
2000 封面上的确认（60 epochs）：

| embed_dim | depth | heads | 参数 | 512 封面 | 2000 封面 |
|---|---|---|---|---|---|
| 32 | 2 | 2 | **95,983** | 0.9015 ± 0.0146 | 0.8873 ± 0.0151 |
| 48 | 3 | 3 | 190,303 | 0.9197 ± 0.0197 | – |
| **64** | **2** | **4** | **240,367** | **0.9208 ± 0.0119** | 0.9036 ± 0.0188 |
| 64 | 4 | 4 | 340,335 | 0.9132 ± 0.0113 | – |
| 96 | 4 | 3 | 657,583 | 0.9187 ± 0.0153 | 0.9094 ± 0.0086 |

**参数量减少 2.7 倍，无可测损失。** 在 512 封面上 24 万参数模型名义上更高；在更大更
可靠的 2000 封面基准上低 0.006——小于一个标准误。容量从约 19 万参数起进入平台期，
因此 24 万参数配置是三个入口脚本的默认值。9.6 万参数模型在预算比精度目标更紧时仍能
保持接近 0.90 的 AUC。

该表也说明了为什么更大的封面集重要：2000 封面的数字一致更低，且不同配置之间区分度
好得多。512 封面的 test 划分（204 个样本）会虚高几个 AUC 点。

### 更多数据与随机裁剪增强的效果

`data/bossbase_*_256.npz` 将 2000 张封面存为 256×256 裁剪；训练时随机取 128×128
位置并翻转（16 个位置 × 4 种翻转 = 每张封面 64 种视图）。60 epochs，3 seeds：

| 嵌入方式 | 数据集 | SRM | 增强 | test AUC | test acc |
|---|---|---|---|---|---|
| HILL 自适应 ±1（0.4 bpp） | 2000 封面，256 裁剪 | 是 | 是 | **0.9094 ± 0.0086** | 0.8092 |
| HILL 自适应 ±1（0.4 bpp） | 2000 封面，256 裁剪 | 是 | 否 | 0.8487 ± 0.0092 | 0.7583 |
| LSB 替换（0.4） | 2000 封面，256 裁剪 | 是 | 是 | 0.7530 ± 0.0116 | 0.6687 |
| HILL + STC（0.4 bpp） | 2000 封面，256 裁剪 | 是 | 是 | 0.7507 ± 0.0158 | 0.6725 |

> HILL + STC 在 0.4 bpp 下修改 19.9% 的像素、最大 |Δ|=1，正是最小失真尺寸通用 STC
> 应有的 ±1 LSB-matching 剖面。

### MAE 自监督预训练：无可测收益

里程碑 M4 预训练编码器重建封面图 30 通道 SRM 残差的掩码 patch，再迁移到检测器。
在 512 封面上结果持平——0.9180 ± 0.0108，不带预训练为 0.9187 ± 0.0153。作为负结果
如实记录：该重建目标与 ±1 检测任务在此规模下对齐得不够好。代码保留，因为在更大的
封面集上重测成本很低（`scripts/pretrain_mae.py`）。

### OOD、弱负载与不匹配嵌入器（M7）

`scripts/eval_ood.py` 对训练好的 checkpoint **不做重训**、直接在其他数据集的
held-out test 划分上评分。所有 .npz 都按相同顺序从相同 BOSSbase 封面构建，同一
seed 选出的是同一批源图像，因此行与行之间的差异只归因于嵌入方式的变化。HILL-0.4
训练的 lite 模型（24 万参数，分布内 test AUC 0.8994）：

| 评测 | 嵌入方式 | test AUC | test acc |
|---|---|---|---|
| 分布内 | HILL 自适应 @0.4 bpp | **0.8994** | 0.8037 |
| 弱负载 | HILL 自适应 @0.2 bpp | 0.7548 | 0.6787 |
| 弱负载 | HILL 自适应 @0.1 bpp | 0.6330 | 0.5800 |
| 不匹配嵌入器 | HILL + STC @0.4 bpp | 0.7474 | 0.6737 |
| 不匹配嵌入器 | LSB 替换 @0.4 | 0.7489 | 0.6775 |
| 跨域 | nsF5，JPEG q75 @0.4 bpnc | 0.5315 | 0.5188 |

退化到 0.2 bpp 仍是平缓的，嵌入器不匹配损失约 0.15 AUC——但真正重要的是 JPEG 域
那一行：**空间域检测器完全不能迁移到 JPEG 隐写**（0.53 ≈ 随机）。部署检测器必须
按域训练，这正是 M8 的出发点。

在弱负载上**重训**（而非迁移）在这个数据规模下没有收益（2000 封面、60 epochs、
3 seeds——所有行均为 `--srm --truncate 4 --pool mean`）：0.2 bpp 重训得
0.7341 ± 0.0121（0.4 模型迁移过去是 0.7548——无法区分），0.1 bpp 重训仍近随机
（0.5988 ± 0.0014）。约 1300 对训练样本下，瓶颈是数据量而不是 checkpoint 的工作点。

### JPEG 域隐写（M8）

真实部署场景：JPEG 隐写发生在量化 DCT 系数上，而部署的检测器手里只有 JPEG 文件。
`prepare_data.py` 新增两种嵌入器与 JPEG 域数据格式：

- **nsF5**（非收缩 F5，`stego/nsf5.py`）——负载嵌入**非零** AC 系数，奇偶翻转
  永不产生新零（|c| = 1 向远离零的方向移动，否则向零移动），因此接收端仅凭含密
  图即可确定载体集合；自适应 Hamming 矩阵编码选择能装下负载的最大码高
  （0.4 bpnc → (7,3)，1.0 bpnc → (1,1,1)）。负载单位是 **bpnc**（每非零 AC 的
  比特数）——文献的标准约定，因为固定 bpp 并非对每张封面都装得下：q75 下
  BOSSbase 256 裁剪的非零 AC 数在 5.5k–33k 之间波动。
- **J-UNIWARD**（`stego/juniward_jpeg.py`）——STC 作用于每块全部 63 个 AC 系数
  的 LSB，代价 = ±1 系数改动在解压图上引起的小波残差变化（UNIWARD）。db8 滤波器
  由谱分解重构（无需 pywt），代价图用线性响应技巧计算：封面子带只算一次，每个
  系数的 ±1 响应是一个形状固定的小窗口，整张代价图化为每个系数位置一次的批处理
  窗口汇聚。
- **数据格式** —— .npz 同时保存 JPEG 解压的 cover/stego 与量化 DCT 系数图
  （`covers_c`/`stegos_c`，DC 置零、AC 裁剪到 ±30）。在系数图上训练即
  `--input coeff`：无 SRM stem（JPEG 量化本身就是高通），仅 120,833 参数。

> **修正：此前的 STC 处于饱和状态。** `embed_stc` 把奇偶校验第 i 行钉在第 i 列，
> 强制 n = m + h − 1：码运行在 ~100% 相对码率上，Viterbi 没有任何自由度，翻转率
> 锁死在活动窗口的 ~50%，失真函数完全无法发挥作用。这也悄悄塑造了上方空间域
> HILL+STC 那一行（19.9% 改动 ≈ 活动窗口的一半）。新的 `embed_stc_spanned` 用
> `start_i = floor(i·(n−h+1)/m)` 摊开校验行；在 n = m + h − 1 时与旧实现逐位
> 一致，而在相对码率 0.5 下 STC 真正开始最小化代价——被改系数的平均 J-UNIWARD
> 代价降到全体均值的一半（160 vs 328）。

结果（2000 封面、60 epochs、3 seeds）：

| 输入 | 嵌入方式 | 参数 | test AUC | test acc |
|---|---|---|---|---|
| 像素（解压图） | nsF5 0.4 bpnc q75 | 240,367 | 0.5406 ± 0.0040 | 0.5196 |
| 系数图 | nsF5 0.4 bpnc q75 | 120,833 | 0.5262 ± 0.0055 | 0.5075 |
| 系数图 | nsF5 1.0 bpnc q75 | 120,833 | **0.9876 ± 0.0099** | 0.9408 |
| 系数图 | J-UNIWARD 0.4 bpp q75 | 120,833 | **1.0000 ± 0.0000** | 0.9962 |
| 像素 | J-UNIWARD 0.4 bpp q75 | 240,367 | **1.0000 ± 0.0000** | 1.0000 |

nsF5 在 0.4 bpnc 下平均每图只改动 65,536 个系数中的 1,271 个（范围 145–3,743）
——在本项目的数据预算下，两种输入表示都检测不了，作为负结果如实记录。在
1.0 bpnc（约 8.8% 系数被改）下，系数域检测器达到 0.988；而 J-UNIWARD 在
0.4 bpp（约 9,900 处改动，多 8 倍）下两种表示都直接饱和：**轻量级 ViT 能够检测
JPEG 域隐写——在系数域上，且足迹足够大时连解压像素都能检测。**

**跨嵌入器迁移在 JPEG 域内部同样失败。** J-UNIWARD 系数模型直接应用到两个 nsF5
数据集上恰为随机（0.4997 / 0.4996）——两种嵌入器改系数的方式本质不同（STC 按
代价摊开在全部 AC 上 vs nsF5 只在非零 AC 上做奇偶翻转），彼此互不迁移。即使
域内，检测器也是嵌入器特异的，进一步印证 M7 的结论。唯一发现的正向迁移：1.0-bpnc
系数模型直接应用到 0.4 bpnc 得 AUC 0.7345——远高于在 0.4 bpnc 上直接训练的
0.5262（准确率 0.549 是阈值偏移所致，排序能力保留）：在此数据规模下，
**「强负载预训练 → 弱负载部署」胜过直接在弱负载上训练**。

## 快速开始

```bash
# 1. 生成封面/含密数据集（2000 张 BOSSbase 封面，存为 256x256 裁剪）
python scripts/prepare_data.py --n 2000 --size 256 --rate 0.4 --method adaptive --out data/bossbase_hill_256.npz
python scripts/prepare_data.py --n 2000 --size 256 --rate 0.4 --method lsb      --out data/bossbase_lsb_256.npz
python scripts/prepare_data.py --n 2000 --size 256 --rate 0.4 --method stc --h 10 --out data/bossbase_hill_stc_256.npz

# 2. 训练 24 万参数检测器（从 256 存储中随机取 128 裁剪）
python scripts/train.py --data data/bossbase_hill_256.npz --srm --truncate 4 --pool mean --epochs 60 --out models/lightvit_lite_hill.pt

# 3. 在 held-out test 划分上评估已保存权重
python scripts/evaluate.py --data data/bossbase_hill_256.npz --ckpt models/lightvit_lite_hill.pt --srm --truncate 4 --pool mean

# 4. 多 seed 基准（这才是应当引用的数字）
python scripts/benchmark.py --data data/bossbase_hill_256.npz --srm --truncate 4 --pool mean --seeds 0,1,2 --epochs 60 --tag 2000_hill

# 5. 观察旧划分造成的偏差
python scripts/diagnose_split.py --data data/bossbase_hill.npz --epochs 30

# 6. checkpoint 的 OOD / 弱负载 / 不匹配嵌入器评测
python scripts/eval_ood.py --ckpt models/lightvit_lite_hill.pt --tag lite_hill_ood \
    --data data/bossbase_hill_256.npz data/bossbase_hill_256_r02.npz data/bossbase_hill_256_r01.npz \
           data/bossbase_hill_stc_256.npz data/bossbase_lsb_256.npz

# 7. JPEG 域：构建 nsF5 / J-UNIWARD 数据集（cover = JPEG 解压图；
#    npz 同时保存量化 DCT 系数图供 --input coeff 训练）
python scripts/prepare_data.py --n 2000 --size 256 --rate 1.0 --rate-unit bpnc --method nsf5 --quality 75 --out data/bossbase_nsf5_256_q75_bpnc10.npz
python scripts/prepare_data.py --n 2000 --size 256 --rate 0.4 --method juniward --quality 75 --out data/bossbase_juniward_256_q75.npz

# 8. 在系数图上训练 JPEG 域检测器（12.1 万参数，无 SRM stem）
python scripts/benchmark.py --data data/bossbase_nsf5_256_q75_bpnc10.npz --input coeff --pool mean \
    --epochs 60 --seeds 0,1,2 --tag 2000_nsf5_q75_bpnc10_coeff_lite

# 9. 把 exp/ 中的所有记录渲染成表格（基准 + OOD）
python scripts/summarize.py

# 10. 自检（无需 pytest）
python tests/test_pipeline.py
```

## 目录结构

```
src/vit_stego.py         # LightViT + SRMStem（冻结的 SRM-30 前端）+ SDPA 注意力
src/srnet.py             # SRNet 基线（瓶颈 ResNet，可选 SRM stem）
src/mae.py               # MAE 编码器 + 迁移到 LightViT
src/srm_filter.py        # SRM 30 个高通核（Fridrich），numpy + torch
src/dataset.py           # CoverStegoDataset（随机裁剪/翻转增强，验证集确定性裁剪，像素/系数双输入）
src/engine.py            # 划分、训练循环、AUC 评测——所有入口共用
scripts/prepare_data.py  # 裁剪封面 + HILL 自适应 / LSB / HILL+STC / nsF5 / J-UNIWARD 嵌入
scripts/train.py         # 单次训练，输出 held-out test AUC
scripts/benchmark.py     # 多 seed 均值 ± 标准差 -> exp/<tag>.json
scripts/evaluate.py      # 在 val/test 划分上评估权重
scripts/eval_ood.py      # checkpoint 的 OOD / 弱负载 / 不匹配嵌入器迁移评测
scripts/diagnose_split.py# 量化旧划分的泄漏
scripts/summarize.py     # exp/*.json -> markdown 表格（基准 + OOD）
scripts/pretrain_mae.py  # 在封面图上做 MAE 自监督预训练
tests/test_pipeline.py   # 划分 / 数据集 / stem / 注意力 / 嵌入器 / JPEG 自检
stego/                   # 失真函数（HILL / STC / J-UNIWARD）+ JPEG 域（jpeg / nsf5 / juniward_jpeg）
data/                    # 生成的 .npz（已 gitignore）
models/                  # 权重（已 gitignore，可重新生成）
exp/                     # 基准记录（已 gitignore）
```

## 路线图

- [x] M0：纯轻量 ViT 原始像素——随机水平（AUC 0.5014），已在修正后的划分下确认。
- [x] M1：SRM-30 高通残差 stem——让检测真正工作的那项改动。
- [x] M2：SRNet 基线，用于公平对比。
- [x] M3：正确的尺寸通用 STC 嵌入（Filler），取代 ±1 占位实现。
- [x] M4：MAE 自监督预训练——已实现，此规模下**无可测收益**；保留以便在更大封面集上重测。
- [x] M5：评测协议——三段划分、保持配对、多 seed、held-out test 指标。（正是它重估了
      此前所有结果。）
- [x] M6：架构消融——截断、池化、增强，以及下探到 9.6 万参数的参数-精度曲线。
- [x] M7：OOD / 跨源 / 弱负载评测——迁移到 0.2/0.1 bpp 与不匹配嵌入器
      （`scripts/eval_ood.py`），以及重训的弱负载工作点。
- [x] M8：JPEG 域嵌入——量化 DCT 域的 nsF5 与 J-UNIWARD
      （`stego/jpeg.py`、`stego/nsf5.py`、`stego/juniward_jpeg.py`），配合系数图
      输入路径（`--input coeff`）使其在 12.1 万参数下可检测。

> 说明：`stego/stc.py` 为尺寸通用的二值 STC——校验矩阵呈带状，故任意负载长度
> m ≤ n−h+1 都能嵌入任意尺寸载体。它自带往返与暴力最优性自测
> （`python stego/stc.py`）。

## 许可证

Apache-2.0 —— 见 [LICENSE](LICENSE)。
