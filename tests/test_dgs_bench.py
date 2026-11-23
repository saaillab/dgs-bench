"""Unit tests for Algorithm 1, theoretical mirage threshold, calibration, and splits."""

from pathlib import Path
import numpy as np
import pandas as pd

from dgs_bench import (
    apply_temperature_scaling,
    compute_brier,
    compute_dgs_report,
    compute_ece,
    create_dgs_splits,
    fit_temperature_scaling,
    saerens_em_prior_shift,
    theoretical_mirage_threshold,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_algorithm1_warning_and_mirage_detection():
    # Lab split: 5 classes x 40 = 200 balanced; moderate 55% accuracy across all classes
    rng = np.random.default_rng(7)
    n_classes = 5
    y_lab = np.repeat(np.arange(n_classes), 40)
    probs_lab = np.full((len(y_lab), n_classes), 0.08)
    for i, c in enumerate(y_lab):
        if i % 20 < 11:  # 55% correct
            probs_lab[i, c] = 0.68
        else:
            probs_lab[i, (c + 1) % n_classes] = 0.68

    # Field split: 66% majority class 3 with 85% recall on class 3, 45% on minority classes
    y_fld = np.concatenate([
        np.repeat([0, 1, 2, 4], 85),  # 340 minority
        np.full(660, 3),               # 660 majority -> 66%
    ])
    probs_fld = np.full((len(y_fld), n_classes), 0.05)
    for i, c in enumerate(y_fld):
        hit = (rng.random() < 0.85) if c == 3 else (rng.random() < 0.42)
        if hit:
            probs_fld[i, c] = 0.80
        else:
            probs_fld[i, (c + 1) % n_classes] = 0.80

    report = compute_dgs_report(
        y_lab, probs_lab, y_fld, probs_fld,
        class_names=["CBB", "CBSD", "CGM", "CMD", "Healthy"],
    )
    assert report.imbalance_warning is True
    assert report.majority_class == "CMD"
    assert report.majority_prevalence > 0.30
    assert report.dgs_acc < 0.0
    assert report.dgs_f1 > 0.0
    assert report.mirage_detected is True
    assert "DGS_F1" in report.warning_message


def test_theoretical_threshold_exact_match():
    df = pd.read_csv(REPO_ROOT / "results" / "theoretical_validation.csv")
    for _, row in df.iterrows():
        p_star = theoretical_mirage_threshold(
            float(row["lab_acc"]), float(row["cmd_recall"]), float(row["other_recall"])
        )
        assert abs(p_star - float(row["p_star"])) < 1e-6


def test_temperature_scaling_and_saerens_em():
    rng = np.random.default_rng(123)
    # Construct synthetic prior-shifted data from balanced lab source (50/50) to skewed field (80/20)
    y_lab = np.array([0] * 250 + [1] * 250)
    logits_lab = np.where(y_lab[:, None] == np.array([[0, 1]]), 3.0, -3.0) + rng.normal(0, 0.8, (500, 2))
    temp = fit_temperature_scaling(logits_lab, y_lab)
    probs_scaled = apply_temperature_scaling(logits_lab, temp)
    probs_overconf = apply_temperature_scaling(logits_lab, 0.25)
    assert compute_ece(y_lab, probs_scaled) < compute_ece(y_lab, probs_overconf)
    assert compute_brier(y_lab, probs_scaled) < compute_brier(y_lab, probs_overconf)

    # Prior shift check
    y_fld = np.array([0] * 200 + [1] * 800)
    logits_fld = np.where(y_fld[:, None] == np.array([[0, 1]]), 2.0, -2.0) + rng.normal(0, 0.6, (1000, 2))
    p_unadj = apply_temperature_scaling(logits_fld, temp)
    p_adj, est_priors, n_iter = saerens_em_prior_shift(p_unadj, [0.5, 0.5])
    assert n_iter >= 1
    assert abs(est_priors[1] - 0.80) < 0.06
    assert p_adj.shape == p_unadj.shape


def test_deterministic_split_generation():
    y = np.concatenate([np.full(500, c) for c in range(5)])
    tr1, te1, fl1 = create_dgs_splits(y, seed=42, n_per_class=400)
    tr2, te2, fl2 = create_dgs_splits(y, seed=42, n_per_class=400)
    assert len(tr1) == 1600 and len(te1) == 400 and len(fl1) == 500
    assert np.array_equal(tr1, tr2) and np.array_equal(te1, te2) and np.array_equal(fl1, fl2)
    assert len(set(tr1) & set(te1)) == 0
    assert len(set(tr1) & set(fl1)) == 0
    assert len(set(te1) & set(fl1)) == 0
