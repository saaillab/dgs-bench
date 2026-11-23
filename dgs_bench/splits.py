"""Deterministic balanced-lab vs. natural-field split generator matching DGS-Bench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np


def create_dgs_splits(
    labels: Sequence[int],
    seed: int = 42,
    n_per_class: int = 400,
    train_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce exact DGS-Bench balanced lab train/test and complement field split.

    Uses np.random.default_rng(seed) in class order 0..K-1 to draw n_per_class
    balanced samples per class, then shuffles each class within the lab split with
    the same RNG to take the first train_ratio (80%) as lab_train and remainder as
    lab_test. Every remaining sample forms the natural field evaluation split.
    """
    y = np.asarray(labels, dtype=int)
    classes = np.unique(y)
    rng = np.random.default_rng(seed)
    lab_idx: List[int] = []
    for c in classes:
        cls_idx = np.where(y == c)[0]
        if len(cls_idx) < n_per_class:
            raise ValueError(f"Class {c} has only {len(cls_idx)} samples < n_per_class={n_per_class}")
        chosen = rng.choice(cls_idx, size=n_per_class, replace=False)
        lab_idx.extend(chosen.tolist())

    lab_arr = np.asarray(lab_idx, dtype=int)
    field_arr = np.setdiff1d(np.arange(len(y)), lab_arr)

    lab_y = y[lab_arr]
    tr_local: List[int] = []
    te_local: List[int] = []
    for c in classes:
        cls_local = np.where(lab_y == c)[0]
        perm = rng.permutation(cls_local)
        cut = int(len(perm) * train_ratio)
        tr_local.extend(perm[:cut].tolist())
        te_local.extend(perm[cut:].tolist())

    return lab_arr[tr_local], lab_arr[te_local], field_arr


def load_cassava_splits_json(splits_path: str | Path) -> Dict[str, object]:
    """Load fixed 10-seed Cassava split specification JSON."""
    return json.loads(Path(splits_path).read_text(encoding="utf-8"))
