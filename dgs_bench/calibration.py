"""Calibration (ECE, Brier, Temperature Scaling) and Saerens EM prior-shift adjustment."""

from __future__ import annotations

from typing import Sequence
import numpy as np
from scipy.optimize import minimize_scalar


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def compute_ece(y_true: Sequence[int], y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Compute 15-bin Expected Calibration Error (ECE) on top-1 confidence (in [0, 1])."""
    yt = np.asarray(y_true, dtype=int)
    probs = _normalize_probs(y_prob)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == yt).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(yt)
    if n == 0:
        return float("nan")

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if not np.any(in_bin):
            continue
        acc_bin = float(accuracies[in_bin].mean())
        conf_bin = float(confidences[in_bin].mean())
        ece += (in_bin.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)


def compute_brier(y_true: Sequence[int], y_prob: np.ndarray) -> float:
    """Compute multi-class Brier score: mean over samples of sum_k (p_ik - y_ik)^2."""
    yt = np.asarray(y_true, dtype=int)
    probs = _normalize_probs(y_prob)
    n, k = probs.shape
    one_hot = np.zeros((n, k), dtype=float)
    one_hot[np.arange(n), yt] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def apply_temperature_scaling(logits_or_logprobs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply positive scalar temperature T to logits/log-probabilities via stable softmax."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scaled = np.asarray(logits_or_logprobs, dtype=float) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(scaled)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


def fit_temperature_scaling(
    logits_or_logprobs: np.ndarray,
    y_true: Sequence[int],
    bounds: tuple[float, float] = (0.05, 10.0),
) -> float:
    """Fit scalar temperature T minimizing multi-class NLL on held-out/CV lab logits."""
    yt = np.asarray(y_true, dtype=int)
    z = np.asarray(logits_or_logprobs, dtype=float)

    def nll(temp: float) -> float:
        p = apply_temperature_scaling(z, temp)
        idx = np.arange(len(yt))
        return float(-np.mean(np.log(np.clip(p[idx, yt], 1e-12, 1.0))))

    res = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(res.x)


def saerens_em_prior_shift(
    field_probs: np.ndarray,
    source_priors: Sequence[float],
    max_iter: int = 200,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Adjust predicted field posteriors for label-shift via Saerens et al. (2002) EM.

    Args:
        field_probs: Unadjusted posterior probabilities P_L(y|x) evaluated on field samples.
        source_priors: Training/source class priors pi_L (e.g., [0.2]*5 for balanced lab split).
        max_iter: Maximum EM iterations.
        tol: L1 stopping tolerance on estimated target priors pi_F.

    Returns:
        adjusted_probs: Shift-corrected posterior probabilities P_F(y|x).
        est_target_priors: Unsupervised EM estimate of target class priors pi_F.
        n_iter: Number of EM iterations executed.
    """
    probs = _normalize_probs(field_probs)
    pi_s = np.asarray(source_priors, dtype=float)
    pi_s = pi_s / pi_s.sum()
    if np.any(pi_s <= 0):
        raise ValueError("All source priors must be strictly positive.")

    pi_t = pi_s.copy()
    post = probs.copy()
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        ratios = pi_t / pi_s
        unnorm = probs * ratios[None, :]
        post = unnorm / np.maximum(unnorm.sum(axis=1, keepdims=True), 1e-12)
        new_pi_t = post.mean(axis=0)
        if np.sum(np.abs(new_pi_t - pi_t)) < tol:
            pi_t = new_pi_t
            break
        pi_t = new_pi_t

    ratios = pi_t / pi_s
    unnorm = probs * ratios[None, :]
    post = unnorm / np.maximum(unnorm.sum(axis=1, keepdims=True), 1e-12)
    return post, pi_t, n_iter
