"""Dataset for cover/stego pairs stored in a single .npz file."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class CoverStegoDataset(Dataset):
    """Reads a precomputed .npz with keys `covers` (N,H,W) uint8 and `stegos` (N,H,W) uint8.

    The dataset exposes covers first (label 0), then stegos (label 1), so a
    stratified split can simply partition each half independently.

    Two knobs make it usable for both training and evaluation:

    * ``img_size`` — when smaller than the stored crop, each sample is cropped
      down to it. Training draws a *random* position, evaluation takes the
      deterministic centre crop, so a held-out AUC is reproducible while the
      model still sees a different view of every cover each epoch. This is the
      cheap equivalent of having many more covers: a stored 256x256 pair yields
      16 positions x 4 flips = 64 training views.
    * ``augment`` — random horizontal/vertical flips (train only; flips are
      label-preserving for steganalysis, unlike e.g. arbitrary rotations which
      would resample the LSB plane and destroy the embedded signal).

    ``input="coeff"`` selects the JPEG-domain representation instead of the
    decompressed pixels: JPEG-domain .npz files additionally store
    ``covers_c``/``stegos_c`` (N,H,W) int16 quantized-DCT maps (DC zeroed,
    ACs clipped, see prepare_data.COEFF_CLIP). A deployed detector against
    JPEG files holds the coefficients themselves — the +-1 embedding changes
    are directly visible there, while they drown in the decompressed pixels.
    Coefficient maps are block-structured, so crops are quantised to multiples
    of 8 to keep every token aligned with one 8x8 JPEG block; flips stay
    valid because the stored (cover, delta) pair flips together.

    Use :meth:`fetch` via :class:`CoverStegoView` when the same underlying arrays
    must be shared between an augmented training view and a clean eval view.
    """

    def __init__(self, path: str, img_size: int | None = None, augment: bool = True,
                 input: str = "pixels"):
        if input not in ("pixels", "coeff"):
            raise ValueError(f"input must be 'pixels' or 'coeff', got {input}")
        d = np.load(path, allow_pickle=False)
        self.input = input
        if input == "coeff":
            if "covers_c" not in d.files:
                raise ValueError(f"{path} has no covers_c/stegos_c maps; "
                                 "regenerate it with a JPEG-domain prepare_data.py")
            self.covers = np.ascontiguousarray(d["covers_c"])
            self.stegos = np.ascontiguousarray(d["stegos_c"])
        else:
            self.covers = np.ascontiguousarray(d["covers"])
            self.stegos = np.ascontiguousarray(d["stegos"])
        if self.covers.shape != self.stegos.shape:
            raise ValueError(f"covers {self.covers.shape} != stegos {self.stegos.shape}")
        if self.covers.ndim != 3:
            raise ValueError(f"expected (N,H,W) crops, got {self.covers.shape}")

        self.n_per_class = self.covers.shape[0]
        self.stored_size = self.covers.shape[1]
        self.img_size = int(img_size) if img_size else self.stored_size
        if self.img_size > self.stored_size:
            raise ValueError(
                f"img_size {self.img_size} exceeds stored crop {self.stored_size}; "
                "regenerate the .npz with a larger --size"
            )
        if input == "coeff" and self.img_size % 8:
            raise ValueError(f"coeff input needs img_size divisible by 8, got {self.img_size}")
        self.augment = augment

        # Provenance, if prepare_data.py recorded it (method/rate/...).
        self.meta = {k: d[k].item() for k in
                     ("method", "rate", "rate_unit", "domain", "quality", "coeff_clip")
                     if k in d.files}

    def __len__(self) -> int:
        return 2 * self.n_per_class

    def fetch(self, i: int, train: bool = False):
        """Return sample ``i``; ``train`` selects random crop + flips."""
        n = self.n_per_class
        img = self.covers[i] if i < n else self.stegos[i - n]
        label = 0.0 if i < n else 1.0

        # One switch controls both random crop and flips: `--no-augment` has to
        # mean "evaluate-like sampling during training", otherwise a run billed
        # as un-augmented is still drawing a random position every epoch.
        augment = train and self.augment

        size = self.img_size
        if size < self.stored_size:
            h, w = img.shape
            if augment:
                # Coefficient maps are block-structured: crops at multiples of
                # 8 keep each token over exactly one JPEG block.
                step = 8 if self.input == "coeff" else 1
                tops = list(range(0, h - size + 1, step))
                lefts = list(range(0, w - size + 1, step))
                top = tops[int(torch.randint(0, len(tops), (1,)).item())]
                left = lefts[int(torch.randint(0, len(lefts), (1,)).item())]
            else:
                top, left = (h - size) // 2, (w - size) // 2
                if self.input == "coeff":
                    top, left = top // 8 * 8, left // 8 * 8
            img = img[top : top + size, left : left + size]

        if augment:
            if torch.rand(1).item() < 0.5:
                img = img[:, ::-1]
            if torch.rand(1).item() < 0.5:
                img = img[::-1, :]

        # Keep the raw integer scale (pixels [0,255], coefficients their
        # quantized values): the signal is a 1-LSB / 1-coefficient change that
        # vanishes under [0,1] normalization. LayerNorm inside the transformer
        # absorbs the magnitude difference downstream.
        x = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)  # (1, H, W)
        return x.float(), torch.tensor(label, dtype=torch.float32)

    def __getitem__(self, i: int):
        return self.fetch(i, train=False)


class CoverStegoView(Dataset):
    """A (dataset, indices, mode) view, so one pair of cover/stego arrays backs
    both the augmented training loader and the clean evaluation loader."""

    def __init__(self, ds: CoverStegoDataset, indices, train: bool):
        self.ds = ds
        self.indices = np.asarray(indices)
        self.train = train

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        return self.ds.fetch(int(self.indices[k]), train=self.train)
