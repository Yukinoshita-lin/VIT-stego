"""Shared split / train / evaluate engine for the steganalysis models.

Kept separate from ``scripts/train.py`` so the CLI and the multi-seed benchmark
drive exactly the same code path — a reported AUC is then a property of the
model, not of which script produced it.

Split discipline
----------------
Every experiment uses a three-way split. ``val`` picks the best epoch, ``test``
is scored once at the end and is what gets reported. Reporting the max-over-epochs
*validation* AUC (the earlier behaviour) is an optimistic estimate: with ~250
validation samples the selection noise alone is worth several AUC points.
Cover and stego of the same source image always land on the same side.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from .dataset import CoverStegoDataset, CoverStegoView


def set_seed(seed: int) -> None:
    """Seed every RNG a run touches.

    Must be called *before* the model is constructed: seeding inside the training
    loop instead would leave the weight initialisation drawn from whatever stream
    happened to be current, which makes runs incomparable even at a fixed seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def make_splits(n_per_class: int, val_frac: float = 0.15, test_frac: float = 0.2,
                seed: int = 0):
    """Return ``(train, val, test)`` index arrays over ``2*n_per_class`` samples.

    Cover and stego of source image *i* are always on the same side: splitting
    the two halves independently would place image i's cover in train and its
    stego in test, so identical scene content would appear in both and the
    held-out AUC would be confounded with content memorisation.
    """
    rng = np.random.default_rng(seed)
    n = n_per_class
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    perm = rng.permutation(n)
    test_imgs = perm[:n_test]
    val_imgs = perm[n_test:n_test + n_val]
    train_imgs = perm[n_test + n_val:]
    pack = lambda idx: np.concatenate([idx, n + idx])  # noqa: E731
    return pack(train_imgs), pack(val_imgs), pack(test_imgs)


def make_loaders(ds: CoverStegoDataset, val_frac: float = 0.15, test_frac: float = 0.2,
                 seed: int = 0, batch: int = 32, num_workers: int = 0):
    splits = make_splits(ds.n_per_class, val_frac, test_frac, seed)
    # Shuffling draws from its own generator so that the global RNG stream (which
    # dropout and the augmentation flips consume) stays reproducible.
    gen = torch.Generator().manual_seed(seed)
    loaders = []
    for idx, train in zip(splits, (True, False, False)):
        loaders.append(DataLoader(
            CoverStegoView(ds, idx, train=train),
            batch_size=batch, shuffle=train, num_workers=num_workers,
            generator=gen if train else None,
        ))
    return loaders[0], loaders[1], loaders[2]


@torch.no_grad()
def predict(model, loader, device, tta: bool = False):
    """Return (probs, labels). With ``tta`` the score is averaged over the four
    flip variants — free accuracy at eval time, no extra training."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        s = torch.sigmoid(model(x))
        if tta:
            for dims in ((3,), (2,), (2, 3)):
                s = s + torch.sigmoid(model(torch.flip(x, dims)))
            s = s / 4.0
        probs.append(s.cpu().numpy().ravel())
        labels.append(y.numpy().ravel())
    return np.concatenate(probs), np.concatenate(labels)


def evaluate(model, loader, device, tta: bool = False):
    probs, labels = predict(model, loader, device, tta=tta)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    acc = accuracy_score(labels, (probs >= 0.5).astype(int))
    return auc, acc, probs, labels


def train_model(model, train_loader, val_loader, device, epochs: int = 30,
                lr: float = 1e-3, weight_decay: float = 1e-4,
                verbose: bool = True, ckpt: str | None = None):
    """Train with AdamW + cosine schedule, keeping the best-*validation* weights.

    Validation is only used to pick the epoch; the caller is responsible for
    scoring the returned model on the test split. Seeding is the caller's job
    (see :func:`set_seed`) — it has to happen before the model is built.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc, best_acc, best_epoch = -1.0, 0.0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running, n_seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n_seen += x.size(0)
        sched.step()

        val_auc, val_acc, _, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": running / max(1, n_seen),
                        "val_auc": float(val_auc), "val_acc": float(val_acc)})
        if verbose:
            print(f"epoch {epoch}/{epochs} | loss {running / max(1, n_seen):.4f} "
                  f"| val AUC {val_auc:.4f} | val acc {val_acc:.4f} | {time.time() - t0:.1f}s")
        if val_auc > best_auc:
            best_auc, best_acc, best_epoch = float(val_auc), float(val_acc), epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    if ckpt:
        os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
        torch.save(best_state, ckpt)

    return {"best_auc": best_auc, "best_acc": best_acc, "best_epoch": best_epoch,
            "history": history, "seconds": time.time() - t0, "checkpoint": ckpt}
