"""DGS-Bench v1.1 Extension Experiments:
1. 10-seed Temperature Scaling + Saerens EM prior-shift adjustment across 4 backbones.
2. Grad-CAM / ViT spatial attribution energy overlays before/after GrabCut & border-only masking.
3. Multi-backbone fine-tuning extension (ResNet-50, EfficientNet-B3, MobileNetV3-S, ViT-S/16).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as tvm
import torchvision.transforms as T

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dgs_bench.calibration import (
    apply_temperature_scaling,
    compute_brier,
    compute_ece,
    fit_temperature_scaling,
    saerens_em_prior_shift,
)
from dgs_bench.metrics import compute_split_metrics
from dgs_bench.splits import create_dgs_splits

DATA_DIR = Path(r"C:\Users\Prof\AppData\Local\Temp\dgs_cassava_data")
CACHE_DIR = DATA_DIR / "features_cache"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
K = 5
CMD_IDX = 3
ARCH_NAMES = ["ResNet-50", "EfficientNet-B3", "MobileNetV3-S", "ViT-S/16"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

TRAIN_AUG = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


class TensorSliceDS(Dataset):
    """Serve preloaded (N, 3, 224, 224) uint8 images for zero-disk-I/O GPU evaluation and training."""

    def __init__(self, x_uint8: torch.Tensor, labels: torch.Tensor, indices: np.ndarray, train: bool = False):
        self.x = x_uint8
        self.y = labels
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.train = train
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int):
        idx = int(self.indices[pos])
        img = self.x[idx].float().div_(255.0)
        if self.train and torch.rand(1).item() < 0.5:
            img = torch.flip(img, dims=[2])
        img = (img - self.mean) / self.std
        return img, int(self.y[idx])


def preload_cassava_224(paths: np.ndarray) -> torch.Tensor:
    """Decode all 21,397 pre-resized 256p Cassava images once into RAM (~3.0 GB uint8)."""
    t0 = time.time()
    print(f"[preload] Decoding {len(paths)} Cassava images into RAM...", flush=True)
    n = len(paths)
    buf = torch.empty((n, 3, 224, 224), dtype=torch.uint8)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            if w != 256 and h != 256:
                if w <= h:
                    rgb = rgb.resize((256, int(round(h * 256.0 / w))), Image.Resampling.BILINEAR)
                else:
                    rgb = rgb.resize((int(round(w * 256.0 / h)), 256), Image.Resampling.BILINEAR)
                w, h = rgb.size
            left = (w - 224) // 2
            top = (h - 224) // 2
            crop = rgb.crop((left, top, left + 224, top + 224))
            arr = np.asarray(crop, dtype=np.uint8)
            buf[i] = torch.from_numpy(arr.copy()).permute(2, 0, 1)
        if (i + 1) % 7000 == 0:
            print(f"  Decoded {i+1}/{n} ({time.time()-t0:.1f}s)...", flush=True)
    print(f"  Preloaded {buf.shape} uint8 in {time.time()-t0:.1f}s ({buf.nbytes/(1024**3):.2f} GB)", flush=True)
    return buf


def build_feature_backbone(arch: str) -> nn.Module:
    if arch == "ResNet-50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
    elif arch == "EfficientNet-B3":
        m = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)
    elif arch == "MobileNetV3-S":
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier = nn.Identity()
    elif arch == "ViT-S/16":
        m = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return m.to(DEVICE).eval()


def build_classifier_backbone(arch: str) -> Tuple[nn.Module, int]:
    if arch == "ResNet-50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, K)
        return m, in_f
    if arch == "EfficientNet-B3":
        m = timm.create_model("efficientnet_b3", pretrained=True, num_classes=K)
        return m, m.classifier.in_features
    if arch == "MobileNetV3-S":
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_f = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_f, K)
        return m, in_f
    if arch == "ViT-S/16":
        m = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=K)
        return m, m.head.in_features
    raise ValueError(arch)


def extract_or_load_features(x_all: torch.Tensor, labels: np.ndarray) -> Dict[str, np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ds = TensorSliceDS(x_all, torch.as_tensor(labels, dtype=torch.long), np.arange(len(labels)), train=False)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    all_feats: Dict[str, np.ndarray] = {}

    for arch in ARCH_NAMES:
        safe = arch.replace("/", "_").replace("-", "_")
        cache_fp = CACHE_DIR / f"cassava_{safe}.npy"
        if cache_fp.exists():
            arr = np.load(cache_fp)
            if len(arr) == len(labels):
                print(f"[cache] Loaded {arch}: {arr.shape}", flush=True)
                all_feats[arch] = arr
                continue

        t0 = time.time()
        print(f"[extract] Extracting {arch} on {DEVICE}...", flush=True)
        model = build_feature_backbone(arch)
        chunks = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    f = model(x)
                chunks.append(f.float().cpu().numpy())
        arr = np.vstack(chunks)
        np.save(cache_fp, arr)
        all_feats[arch] = arr
        print(f"  {arch}: {arr.shape} in {time.time() - t0:.1f}s", flush=True)
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return all_feats


def run_experiment_a_calibration(all_feats: Dict[str, np.ndarray], labels: np.ndarray) -> None:
    print("\n=== Experiment A: 10-Seed Calibration & Saerens EM Prior Shift ===", flush=True)
    rows = []
    seeds = list(range(42, 52))
    lab_priors = np.full(K, 1.0 / K)

    for arch in ARCH_NAMES:
        X = all_feats[arch]
        for seed in seeds:
            tr_idx, te_idx, fld_idx = create_dgs_splits(labels, seed=seed, n_per_class=400)
            sc = StandardScaler().fit(X[tr_idx])
            X_tr_s = sc.transform(X[tr_idx])
            X_te_s = sc.transform(X[te_idx])
            X_fld_s = sc.transform(X[fld_idx])
            y_tr, y_te, y_fld = labels[tr_idx], labels[te_idx], labels[fld_idx]

            # Out-of-fold logits on lab_train for leakage-free temperature fitting
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            oof_logits = np.zeros((len(y_tr), K), dtype=float)
            for cv_tr, cv_va in skf.split(X_tr_s, y_tr):
                cv_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
                cv_clf.fit(X_tr_s[cv_tr], y_tr[cv_tr])
                oof_logits[cv_va] = cv_clf.decision_function(X_tr_s[cv_va])
            opt_temp = fit_temperature_scaling(oof_logits, y_tr)

            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
            clf.fit(X_tr_s, y_tr)

            logits_te = clf.decision_function(X_te_s)
            logits_fld = clf.decision_function(X_fld_s)
            probs_te_raw = clf.predict_proba(X_te_s)
            probs_fld_raw = clf.predict_proba(X_fld_s)

            probs_te_ts = apply_temperature_scaling(logits_te, opt_temp)
            probs_fld_ts = apply_temperature_scaling(logits_fld, opt_temp)

            probs_fld_em, pi_em_raw, _ = saerens_em_prior_shift(probs_fld_raw, lab_priors)
            probs_fld_ts_em, pi_em_ts, _ = saerens_em_prior_shift(probs_fld_ts, lab_priors)

            regimes = [
                ("Uncalibrated", probs_te_raw, probs_fld_raw, 1.0, float((y_fld == CMD_IDX).mean()), 0.2),
                ("TempScaled", probs_te_ts, probs_fld_ts, opt_temp, float((y_fld == CMD_IDX).mean()), 0.2),
                ("SaerensEM", probs_te_raw, probs_fld_em, 1.0, float((y_fld == CMD_IDX).mean()), float(pi_em_raw[CMD_IDX])),
                ("TempScaled+SaerensEM", probs_te_ts, probs_fld_ts_em, opt_temp, float((y_fld == CMD_IDX).mean()), float(pi_em_ts[CMD_IDX])),
            ]
            for reg_name, p_lab, p_fld, temp_used, true_cmd_p, est_cmd_p in regimes:
                lm = compute_split_metrics(y_te, p_lab, class_names=CLASS_NAMES)
                fm = compute_split_metrics(y_fld, p_fld, class_names=CLASS_NAMES)
                rows.append({
                    "seed": seed,
                    "arch": arch,
                    "regime": reg_name,
                    "temperature": round(temp_used, 4),
                    "est_cmd_prior_pct": round(est_cmd_p * 100.0, 2),
                    "true_cmd_prior_pct": round(true_cmd_p * 100.0, 2),
                    "lab_acc": lm["acc"],
                    "lab_f1": lm["f1"],
                    "lab_auprc": lm["auprc"],
                    "lab_ece": compute_ece(y_te, p_lab),
                    "lab_brier": compute_brier(y_te, p_lab),
                    "fld_acc": fm["acc"],
                    "fld_bacc": fm["bacc"],
                    "fld_f1": fm["f1"],
                    "fld_auprc": fm["auprc"],
                    "fld_auroc": fm["auroc"],
                    "fld_ece": compute_ece(y_fld, p_fld),
                    "fld_brier": compute_brier(y_fld, p_fld),
                    "dgs_acc": lm["acc"] - fm["acc"],
                    "dgs_f1": lm["f1"] - fm["f1"],
                    "dgs_auprc": lm["auprc"] - fm["auprc"],
                })

    df_per = pd.DataFrame(rows)
    df_per.to_csv(RESULTS_DIR / "v11_calibration_priorshift_perseed.csv", index=False)

    num_cols = [
        "temperature", "est_cmd_prior_pct", "lab_acc", "lab_f1", "lab_ece",
        "fld_acc", "fld_bacc", "fld_f1", "fld_auprc", "fld_ece", "fld_brier",
        "dgs_acc", "dgs_f1", "dgs_auprc",
    ]
    summary_rows = []
    for (arch, regime), grp in df_per.groupby(["arch", "regime"], sort=False):
        rec = {"arch": arch, "regime": regime, "n_seeds": len(grp)}
        for col in num_cols:
            rec[f"{col}_mean"] = round(float(grp[col].mean()), 4)
            rec[f"{col}_std"] = round(float(grp[col].std(ddof=0)), 4)
        summary_rows.append(rec)

    # Also overall 40-run mean per regime
    for regime, grp in df_per.groupby("regime", sort=False):
        rec = {"arch": "All (n=40)", "regime": regime, "n_seeds": len(grp)}
        for col in num_cols:
            rec[f"{col}_mean"] = round(float(grp[col].mean()), 4)
            rec[f"{col}_std"] = round(float(grp[col].std(ddof=0)), 4)
        summary_rows.append(rec)

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(RESULTS_DIR / "v11_calibration_priorshift_summary.csv", index=False)
    print(df_sum[df_sum["arch"] == "All (n=40)"][
        ["regime", "fld_ece_mean", "fld_brier_mean", "fld_acc_mean", "fld_f1_mean", "dgs_acc_mean", "dgs_f1_mean", "est_cmd_prior_pct_mean"]
    ].to_string(index=False), flush=True)
    plot_experiment_a_calibration(df_sum)


def plot_experiment_a_calibration(df_sum: pd.DataFrame) -> None:
    """Render 2-panel Field Calibration & Mirage Persistence chart with non-overlapping labels."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), dpi=300)
    reg_order = ["Uncalibrated", "TempScaled", "SaerensEM", "TempScaled+SaerensEM"]
    reg_short = ["Uncalibrated", "Temp. Scaled", "Saerens EM", "Temp. + EM"]
    x = np.arange(len(reg_order))
    overall = df_sum[df_sum["arch"] == "All (n=40)"].set_index("regime").loc[reg_order]

    # Panel (a): Field ECE & Brier reduction (place labels cleanly above error-bar caps)
    w = 0.34
    b1 = axes[0].bar(x - w / 2, overall["fld_ece_mean"], w, yerr=overall["fld_ece_std"], capsize=3,
                     color=OKABE_ITO[0], label="Field ECE (15-bin)")
    b2 = axes[0].bar(x + w / 2, overall["fld_brier_mean"], w, yerr=overall["fld_brier_std"], capsize=3,
                     color=OKABE_ITO[1], label="Field Multi-Class Brier")
    for bars, stds in ((b1, overall["fld_ece_std"].values), (b2, overall["fld_brier_std"].values)):
        for bar, s in zip(bars, stds):
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + float(s) + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=7.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(reg_short, fontsize=9)
    axes[0].set_ylabel("Calibration Error (Lower Is Better)")
    axes[0].set_title("(a) Field Probability Calibration (n = 40 runs)", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, 0.84)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Panel (b): DGS_Acc vs DGS_F1 vs DGS_AUPRC across regimes
    w2 = 0.24
    g1 = axes[1].bar(x - w2, overall["dgs_acc_mean"], w2, color=OKABE_ITO[3], label="DGS_Acc (pp)")
    g2 = axes[1].bar(x, overall["dgs_f1_mean"], w2, color=OKABE_ITO[2], label="DGS_F1 (pp)")
    g3 = axes[1].bar(x + w2, overall["dgs_auprc_mean"], w2, color=OKABE_ITO[4], label="DGS_AUPRC (pp)")
    axes[1].axhline(0.0, color="#1e293b", linewidth=1.0, linestyle="--")
    for bars in (g1, g2, g3):
        for bar in bars:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            off = 0.55 if h >= 0 else -0.65
            axes[1].text(bar.get_x() + bar.get_width() / 2, h + off, f"{h:+.1f}", ha="center", va=va, fontsize=7.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(reg_short, fontsize=9)
    axes[1].set_ylabel("Deployment Gap Score (Lab - Field, pp)")
    axes[1].set_title("(b) Mirage Persistence Under Calibration & Shift EM", fontsize=10, fontweight="bold")
    axes[1].set_ylim(-21.5, 19.5)
    axes[1].legend(frameon=False, fontsize=8.2, loc="upper center", ncol=3)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout(pad=1.2)
    plt.savefig(FIGURES_DIR / "fig_v11_calibration_priorshift.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def center_mask_rgb(img_rgb: np.ndarray, keep_frac: float = 0.6) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    mw, mh = int(w * (1 - keep_frac) / 2), int(h * (1 - keep_frac) / 2)
    out = np.full_like(img_rgb, 128)
    out[mh:h - mh, mw:w - mw] = img_rgb[mh:h - mh, mw:w - mw]
    return out


def border_mask_rgb(img_rgb: np.ndarray, keep_frac: float = 0.6) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    mw, mh = int(w * (1 - keep_frac) / 2), int(h * (1 - keep_frac) / 2)
    out = img_rgb.copy()
    out[mh:h - mh, mw:w - mw] = 128
    return out


def grabcut_fg_mask_and_img(img_rgb: np.ndarray, margin: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    rect = (int(w * margin), int(h * margin), int(w * (1.0 - 2.0 * margin)), int(h * (1.0 - 2.0 * margin)))
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    fg_mask = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.float32)
    gray = np.full_like(img_rgb, 128)
    masked_rgb = (img_rgb * fg_mask[:, :, None] + gray * (1.0 - fg_mask[:, :, None])).astype(np.uint8)
    return fg_mask, masked_rgb


class LinearProbeNet(nn.Module):
    """Differentiable wrapper combining ImageNet backbone with exact sklearn linear probe weights."""

    def __init__(self, arch: str, sc: StandardScaler, clf: LogisticRegression):
        super().__init__()
        self.arch = arch
        self.backbone = build_feature_backbone(arch)
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        mean_t = torch.tensor(sc.mean_, dtype=torch.float32)
        scale_t = torch.tensor(np.maximum(sc.scale_, 1e-12), dtype=torch.float32)
        w = torch.tensor(clf.coef_, dtype=torch.float32)  # (K, D)
        b = torch.tensor(clf.intercept_, dtype=torch.float32)  # (K,)
        self.register_buffer("feat_mean", mean_t)
        self.register_buffer("feat_scale", scale_t)
        self.head = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            self.head.weight.copy_(w)
            self.head.bias.copy_(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        normed = (feats - self.feat_mean) / self.feat_scale
        return self.head(normed)


def compute_gradcam_map(net: LinearProbeNet, arch: str, img_tensor: torch.Tensor, target_class: int | None = None) -> Tuple[np.ndarray, int, float]:
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    if arch == "ResNet-50":
        target_layer = net.backbone.layer4[-1]
    elif arch == "EfficientNet-B3":
        target_layer = net.backbone.conv_head
    elif arch == "ViT-S/16":
        target_layer = net.backbone.blocks[-1].norm1
    else:
        raise ValueError(arch)

    def fwd_hook(_mod, _inp, out):
        activations.append(out)
        out.register_hook(lambda g: gradients.append(g))

    h1 = target_layer.register_forward_hook(fwd_hook)
    try:
        net.zero_grad()
        x_in = img_tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)
        logits = net(x_in)
        probs = torch.softmax(logits, dim=1)
        pred_cls = int(probs.argmax(dim=1).item())
        cls_use = pred_cls if target_class is None else target_class
        conf = float(probs[0, pred_cls].item())
        logits[0, cls_use].backward()

        act = activations[0].detach()[0]
        grad = gradients[0].detach()[0]
        if arch == "ViT-S/16":
            patch_act = act[1:, :]
            patch_grad = grad[1:, :]
            weights = patch_grad.mean(dim=0, keepdim=True)
            cam = torch.relu((patch_act * weights).sum(dim=1)).reshape(14, 14).cpu().numpy()
        else:
            weights = grad.mean(dim=(1, 2), keepdim=True)
            cam = torch.relu((act * weights).sum(dim=0)).cpu().numpy()

        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
        if cam.max() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, pred_cls, conf
    finally:
        h1.remove()


def run_experiment_b_gradcam(all_feats: Dict[str, np.ndarray], paths: np.ndarray, labels: np.ndarray) -> None:
    print("\n=== Experiment B: Grad-CAM & ViT Spatial Attribution Before/After GrabCut & Border-Only ===", flush=True)
    tr_idx, _, fld_idx = create_dgs_splits(labels, seed=42, n_per_class=400)
    rng = np.random.default_rng(142)

    # Stratified 200 valid GrabCut field images (40 per class; accept only 0.15 <= fg_frac <= 0.85)
    sample_idx = []
    fg_masks_224 = []
    border_masks_224 = []
    orig_imgs_rgb = []
    gc_imgs_rgb = []
    bd_imgs_rgb = []
    fg_fracs = []
    b_mask_template = np.ones((224, 224), dtype=np.float32)
    mw = int(224 * 0.2)
    b_mask_template[mw:224 - mw, mw:224 - mw] = 0.0

    for c in range(K):
        cls_fld = rng.permutation(fld_idx[labels[fld_idx] == c])
        cls_kept = 0
        for idx in cls_fld:
            raw_rgb = np.array(Image.open(paths[idx]).convert("RGB"))
            raw_224 = np.array(Image.fromarray(raw_rgb).resize((256, 256)).crop((16, 16, 240, 240)))
            fg_m, gc_rgb = grabcut_fg_mask_and_img(raw_224, margin=0.05)
            fg_frac = float(fg_m.mean())
            if not (0.15 <= fg_frac <= 0.85):
                continue
            bd_rgb = border_mask_rgb(raw_224, keep_frac=0.6)
            sample_idx.append(int(idx))
            orig_imgs_rgb.append(raw_224)
            gc_imgs_rgb.append(gc_rgb)
            bd_imgs_rgb.append(bd_rgb)
            fg_masks_224.append(fg_m)
            border_masks_224.append(b_mask_template)
            fg_fracs.append(fg_frac)
            cls_kept += 1
            if cls_kept == 40:
                break
        assert cls_kept == 40, f"Class {c} only yielded {cls_kept}/40 valid GrabCut masks in [0.15, 0.85]"

    sample_idx = np.asarray(sample_idx, dtype=int)
    mean_fg_area_pct = round(float(np.mean(fg_fracs) * 100.0), 2)
    border_area_pct = round(float(b_mask_template.mean() * 100.0), 2)
    print(f"  Valid GrabCut sample n={len(sample_idx)}: mean_fg_area={mean_fg_area_pct:.2f}%, border_area={border_area_pct:.2f}%", flush=True)

    arches = ["ResNet-50", "EfficientNet-B3", "ViT-S/16"]
    energy_rows = []
    qualitative_samples = []

    to_tensor_norm = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    vis_targets = []  # Pick 2 interesting samples once from EfficientNet-B3 border-correct vs CMD/Healthy

    for arch in arches:
        X = all_feats[arch]
        sc = StandardScaler().fit(X[tr_idx])
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(sc.transform(X[tr_idx]), labels[tr_idx])
        probe_net = LinearProbeNet(arch, sc, clf).to(DEVICE).eval()

        cond_stats = {c: {"fg_energy": [], "border_energy": [], "correct": [], "cmd_pred": []} for c in ["Original", "GrabCut", "Border-only"]}
        per_img_store = []

        for j in range(len(sample_idx)):
            true_y = int(labels[sample_idx[j]])
            fg_m = fg_masks_224[j]
            b_m = border_masks_224[j]
            imgs_cond = [
                ("Original", orig_imgs_rgb[j]),
                ("GrabCut", gc_imgs_rgb[j]),
                ("Border-only", bd_imgs_rgb[j]),
            ]
            img_cams = {}
            for cond_name, rgb in imgs_cond:
                t_in = to_tensor_norm(Image.fromarray(rgb))
                cam, pred, conf = compute_gradcam_map(probe_net, arch, t_in)
                cam_sum = float(cam.sum()) + 1e-8
                fg_e = float((cam * fg_m).sum() / cam_sum) * 100.0
                bd_e = float((cam * b_m).sum() / cam_sum) * 100.0
                cond_stats[cond_name]["fg_energy"].append(fg_e)
                cond_stats[cond_name]["border_energy"].append(bd_e)
                cond_stats[cond_name]["correct"].append(float(pred == true_y))
                cond_stats[cond_name]["cmd_pred"].append(float(pred == CMD_IDX))
                img_cams[cond_name] = (cam, pred, conf, fg_e)
            per_img_store.append(img_cams)

        for cond_name in ["Original", "GrabCut", "Border-only"]:
            s = cond_stats[cond_name]
            energy_rows.append({
                "arch": arch,
                "condition": cond_name,
                "n": len(sample_idx),
                "acc_pct": round(float(np.mean(s["correct"]) * 100.0), 2),
                "cmd_pred_rate_pct": round(float(np.mean(s["cmd_pred"]) * 100.0), 2),
                "fg_attribution_energy_pct": round(float(np.mean(s["fg_energy"])), 2),
                "border_attribution_energy_pct": round(float(np.mean(s["border_energy"])), 2),
                "mean_fg_area_pct": mean_fg_area_pct,
                "border_area_pct": border_area_pct,
            })

        qualitative_samples.append((arch, per_img_store))
        del probe_net
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    df_gc_cam = pd.DataFrame(energy_rows)
    df_gc_cam.to_csv(RESULTS_DIR / "v11_gradcam_attribution_energy.csv", index=False)
    print(df_gc_cam.to_string(index=False), flush=True)

    # Pick one representative CMD field image with clean GrabCut foreground fraction
    chosen_j = 0
    for j in range(len(sample_idx)):
        if labels[sample_idx[j]] == CMD_IDX and 0.28 <= fg_masks_224[j].mean() <= 0.65:
            chosen_j = j
            break

    fig = plt.figure(figsize=(14.5, 6.8), dpi=300)
    gs_img = fig.add_gridspec(3, 4, left=0.04, right=0.67, top=0.90, bottom=0.05, wspace=0.08, hspace=0.28)

    row_inputs = [("Original Leaf", orig_imgs_rgb[chosen_j]), ("GrabCut Foreground", gc_imgs_rgb[chosen_j]), ("Border-Only Mask", bd_imgs_rgb[chosen_j])]
    cond_keys = ["Original", "GrabCut", "Border-only"]

    for r, (r_label, rgb) in enumerate(row_inputs):
        ax0 = fig.add_subplot(gs_img[r, 0])
        ax0.imshow(rgb)
        ax0.set_ylabel(r_label, fontsize=9, fontweight="bold")
        ax0.set_xticks([]); ax0.set_yticks([])
        if r == 0:
            ax0.set_title("Input Condition", fontsize=9.5, fontweight="bold")

        for c_idx, (arch, img_store) in enumerate(qualitative_samples):
            ax = fig.add_subplot(gs_img[r, c_idx + 1])
            cam, pred, conf, fg_e = img_store[chosen_j][cond_keys[r]]
            ax.imshow(rgb)
            ax.imshow(cam, cmap="turbo", alpha=0.48)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{arch}\nPred: {CLASS_NAMES[pred]} ({conf*100:.0f}%)", fontsize=8.8, fontweight="bold")
            else:
                ax.set_title(f"Pred: {CLASS_NAMES[pred]} ({conf*100:.0f}%)", fontsize=8.3)

    # Separated right axis with wide whitespace gap so y-label never collides with ViT-S/16 column
    ax_bar = fig.add_axes([0.75, 0.14, 0.22, 0.72])
    x = np.arange(len(arches))
    w = 0.26
    for j, cond_name in enumerate(cond_keys):
        sub = df_gc_cam[df_gc_cam["condition"] == cond_name].set_index("arch").loc[arches]
        bars = ax_bar.bar(x + (j - 1) * w, sub["fg_attribution_energy_pct"], w, color=OKABE_ITO[j], label=cond_name)
        for b in bars:
            h = b.get_height()
            ax_bar.text(b.get_x() + b.get_width() / 2, h + 1.2, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.0)
    ax_bar.axhline(mean_fg_area_pct, color="#334155", linewidth=1.2, linestyle="--", label=f"Leaf Mask Area ({mean_fg_area_pct:.1f}%)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["ResNet-50", "EffNet-B3", "ViT-S/16"], fontsize=8.5, rotation=12)
    ax_bar.set_ylabel("Foreground Grad-CAM Energy (%)", fontsize=9)
    ax_bar.set_title("Leaf-Mask Attribution Energy\n(n = 200 Stratified Field Images)", fontsize=9.5, fontweight="bold")
    ax_bar.set_ylim(0, 100)
    ax_bar.legend(frameon=False, fontsize=7.8, loc="upper right")
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    plt.savefig(FIGURES_DIR / "fig_v11_gradcam_overlays.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def fine_tune_architecture(
    arch: str,
    mode: str,
    x_all: torch.Tensor,
    labels: np.ndarray,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    fld_idx: np.ndarray,
    epochs: int = 12,
    lr_bb: float = 1e-4,
    lr_fc: float = 1e-3,
) -> Dict[str, object]:
    torch.manual_seed(42)
    model, _ = build_classifier_backbone(arch)

    def split_params():
        head_params, last_params, all_bb_params = [], [], []
        for name, p in model.named_parameters():
            is_head = (
                name.startswith("fc.")
                or name.startswith("classifier.")
                or name.startswith("head.")
            )
            if is_head:
                head_params.append(p)
            else:
                all_bb_params.append(p)
                if (
                    name.startswith("layer4.")
                    or name.startswith("conv_head.")
                    or name.startswith("bn2.")
                    or "blocks.6." in name
                    or "features.11." in name
                    or "features.12." in name
                    or "blocks.11." in name
                    or name.startswith("norm.")
                ):
                    last_params.append(p)
        return head_params, last_params, all_bb_params

    head_p, last_p, all_bb = split_params()
    if mode == "last_block":
        for p in model.parameters():
            p.requires_grad = False
        for p in last_p + head_p:
            p.requires_grad = True
        opt_groups = [{"params": last_p, "lr": lr_bb}, {"params": head_p, "lr": lr_fc}]
    elif mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        opt_groups = [{"params": all_bb, "lr": lr_bb}, {"params": head_p, "lr": lr_fc}]
    else:
        raise ValueError(mode)

    model = model.to(DEVICE)
    opt = optim.AdamW(opt_groups, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    y_t = torch.as_tensor(labels, dtype=torch.long)
    tr_loader = DataLoader(TensorSliceDS(x_all, y_t, tr_idx, train=True), batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    for _epoch in range(epochs):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

    model.eval()
    split_probs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for split_name, idx_arr in [("lab", te_idx), ("field", fld_idx)]:
        loader = DataLoader(TensorSliceDS(x_all, y_t, idx_arr, train=False), batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
        probs_list, true_list = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    out = model(x)
                probs_list.append(torch.softmax(out.float(), dim=1).cpu().numpy())
                true_list.append(y.numpy())
        split_probs[split_name] = (np.concatenate(true_list), np.vstack(probs_list))

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    y_te_t, p_te = split_probs["lab"]
    y_fl_t, p_fl = split_probs["field"]
    lm = compute_split_metrics(y_te_t, p_te, class_names=CLASS_NAMES)
    fm = compute_split_metrics(y_fl_t, p_fl, class_names=CLASS_NAMES)
    dgs_acc = lm["acc"] - fm["acc"]
    dgs_f1 = lm["f1"] - fm["f1"]
    return {
        "arch": arch,
        "mode": mode,
        "lab_acc": round(lm["acc"], 2),
        "field_acc": round(fm["acc"], 2),
        "dgs_acc": round(dgs_acc, 2),
        "lab_f1": round(lm["f1"], 2),
        "field_f1": round(fm["f1"], 2),
        "dgs_f1": round(dgs_f1, 2),
        "dgs_auprc": round(lm["auprc"] - fm["auprc"], 2),
        "field_ece": round(compute_ece(y_fl_t, p_fl), 4),
        "mirage": bool(dgs_acc < 0 and dgs_f1 > 0),
    }


def run_experiment_c_multibackbone_finetune(all_feats: Dict[str, np.ndarray], x_all: torch.Tensor, labels: np.ndarray) -> None:
    print("\n=== Experiment C: Multi-Backbone Fine-Tuning Extension ===", flush=True)
    tr_idx, te_idx, fld_idx = create_dgs_splits(labels, seed=42, n_per_class=400)
    rows = []

    for arch in ARCH_NAMES:
        X = all_feats[arch]
        sc = StandardScaler().fit(X[tr_idx])
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(sc.transform(X[tr_idx]), labels[tr_idx])
        p_te = clf.predict_proba(sc.transform(X[te_idx]))
        p_fl = clf.predict_proba(sc.transform(X[fld_idx]))
        lm = compute_split_metrics(labels[te_idx], p_te, class_names=CLASS_NAMES)
        fm = compute_split_metrics(labels[fld_idx], p_fl, class_names=CLASS_NAMES)
        rows.append({
            "arch": arch,
            "mode": "linear_probe",
            "lab_acc": round(lm["acc"], 2),
            "field_acc": round(fm["acc"], 2),
            "dgs_acc": round(lm["acc"] - fm["acc"], 2),
            "lab_f1": round(lm["f1"], 2),
            "field_f1": round(fm["f1"], 2),
            "dgs_f1": round(lm["f1"] - fm["f1"], 2),
            "dgs_auprc": round(lm["auprc"] - fm["auprc"], 2),
            "field_ece": round(compute_ece(labels[fld_idx], p_fl), 4),
            "mirage": bool((lm["acc"] - fm["acc"]) < 0 and (lm["f1"] - fm["f1"]) > 0),
        })

        for mode in ["last_block", "full"]:
            t0 = time.time()
            print(f"  Fine-tuning {arch} ({mode})...", flush=True)
            rec = fine_tune_architecture(arch, mode, x_all, labels, tr_idx, te_idx, fld_idx, epochs=12)
            print(f"    -> Lab={rec['lab_acc']:.1f}% | Fld={rec['field_acc']:.1f}% | DGS_Acc={rec['dgs_acc']:+.2f} | DGS_F1={rec['dgs_f1']:+.2f} ({time.time()-t0:.0f}s)", flush=True)
            rows.append(rec)

    df_ft = pd.DataFrame(rows)
    df_ft.to_csv(RESULTS_DIR / "v11_multibackbone_finetune.csv", index=False)
    plot_experiment_c_multibackbone_finetune(df_ft)


def plot_experiment_c_multibackbone_finetune(df_ft: pd.DataFrame) -> None:
    """Plot DGS_Acc vs DGS_F1 across Linear Probe, Last-Block FT, and Full FT with legend."""
    fig, ax = plt.subplots(figsize=(10.8, 4.7), dpi=300)
    modes = ["linear_probe", "last_block", "full"]
    mode_labels = ["Linear Probe", "Last-Block FT", "Full FT"]
    alphas = [0.45, 0.72, 1.0]
    x = np.arange(len(ARCH_NAMES))
    w = 0.13
    for i, m in enumerate(modes):
        sub = df_ft[df_ft["mode"] == m].set_index("arch").loc[ARCH_NAMES]
        ax.bar(x + (i - 1) * (2.2 * w) - w / 2, sub["dgs_acc"], w, color=OKABE_ITO[3], alpha=alphas[i])
        ax.bar(x + (i - 1) * (2.2 * w) + w / 2, sub["dgs_f1"], w, color=OKABE_ITO[2], alpha=alphas[i])

    for i, m in enumerate(modes):
        sub = df_ft[df_ft["mode"] == m].set_index("arch").loc[ARCH_NAMES]
        for k_idx, val in enumerate(sub["dgs_acc"]):
            ax.text(x[k_idx] + (i - 1) * (2.2 * w) - w / 2, val - 0.5, f"{val:+.1f}", ha="center", va="top", fontsize=6.6, rotation=90)
        for k_idx, val in enumerate(sub["dgs_f1"]):
            ax.text(x[k_idx] + (i - 1) * (2.2 * w) + w / 2, val + 0.5, f"{val:+.1f}", ha="center", va="bottom", fontsize=6.6, rotation=90)

    legend_handles = [
        Patch(facecolor=OKABE_ITO[3], alpha=alphas[0], label="DGS_Acc (Linear)"),
        Patch(facecolor=OKABE_ITO[3], alpha=alphas[1], label="DGS_Acc (Last-Block FT)"),
        Patch(facecolor=OKABE_ITO[3], alpha=alphas[2], label="DGS_Acc (Full FT)"),
        Patch(facecolor=OKABE_ITO[2], alpha=alphas[0], label="DGS_F1 (Linear)"),
        Patch(facecolor=OKABE_ITO[2], alpha=alphas[1], label="DGS_F1 (Last-Block FT)"),
        Patch(facecolor=OKABE_ITO[2], alpha=alphas[2], label="DGS_F1 (Full FT)"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False, fontsize=8.0)

    ax.axhline(0.0, color="#1e293b", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(ARCH_NAMES, fontsize=9.5)
    ax.set_ylabel("Deployment Gap Score (pp)")
    ax.set_title("Multi-Backbone Fine-Tuning Adaptation: Raw DGS_Acc Mirage Persists Across All 4 Architectures", fontsize=10, fontweight="bold")
    ax.set_ylim(-15.5, 16.5)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(pad=1.1)
    plt.savefig(FIGURES_DIR / "fig_v11_multibackbone_finetune.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def run_experiment_d_cara_cassava(x_all: torch.Tensor, labels: np.ndarray) -> None:
    print("\n=== Experiment D: CARA-FM Cassava Field Benchmark (5 Edge-to-Cloud Models) ===", flush=True)
    tr_idx, te_idx, fld_idx = create_dgs_splits(labels, seed=42, n_per_class=400)
    cara_specs = [
        ("MobileNetV3-Small", "MobileNetV3-S"),
        ("EfficientNet-B0", "EfficientNet-B0"),
        ("ResNet-18", "ResNet-18"),
        ("ViT-Small/16", "ViT-S/16"),
        ("ResNet-50", "ResNet-50"),
    ]
    y_t = torch.as_tensor(labels, dtype=torch.long)
    tr_loader = DataLoader(TensorSliceDS(x_all, y_t, tr_idx, train=True), batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    out = {}
    for display_name, arch_key in cara_specs:
        t0 = time.time()
        torch.manual_seed(42)
        if arch_key in {"MobileNetV3-S", "EfficientNet-B3", "ViT-S/16", "ResNet-50"}:
            if arch_key == "MobileNetV3-S":
                model = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
                model.classifier[3] = nn.Linear(model.classifier[3].in_features, K)
            elif arch_key == "ViT-S/16":
                model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=K)
            else:
                model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
                model.fc = nn.Linear(model.fc.in_features, K)
        elif arch_key == "EfficientNet-B0":
            model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=K)
        else:
            model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, K)

        model = model.to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        epochs = 12
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

        for _ in range(epochs):
            model.train()
            for x, y in tr_loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    loss = crit(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            sched.step()

        model.eval()
        split_eval = {}
        for sname, sidx in [("lab", te_idx), ("field", fld_idx)]:
            loader = DataLoader(TensorSliceDS(x_all, y_t, sidx, train=False), batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
            probs_l, true_l = [], []
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(DEVICE, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                        logits = model(x)
                    probs_l.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
                    true_l.append(y.numpy())
            split_eval[sname] = (np.concatenate(true_l), np.vstack(probs_l))

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        yl, pl = split_eval["lab"]
        yf, pf = split_eval["field"]
        lm = compute_split_metrics(yl, pl, class_names=CLASS_NAMES)
        fm = compute_split_metrics(yf, pf, class_names=CLASS_NAMES)
        out[display_name] = {
            "cassava_lab_acc_pct": round(lm["acc"], 2),
            "cassava_field_acc_pct": round(fm["acc"], 2),
            "cassava_field_bacc_pct": round(fm["bacc"], 2),
            "cassava_field_f1_pct": round(fm["f1"], 2),
            "cassava_dgs_acc_pp": round(lm["acc"] - fm["acc"], 2),
            "cassava_dgs_f1_pp": round(lm["f1"] - fm["f1"], 2),
            "cassava_field_ece": round(compute_ece(yf, pf), 4),
        }
        print(
            f"  CARA Cassava {display_name:18s}: Fld BAcc={fm['bacc']:.2f}%, F1={fm['f1']:.2f}%, "
            f"ECE={out[display_name]['cassava_field_ece']:.4f} ({time.time()-t0:.0f}s)",
            flush=True,
        )

    (DATA_DIR / "cara_cassava_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def load_cached_features(n_expected: int) -> Dict[str, np.ndarray]:
    all_feats: Dict[str, np.ndarray] = {}
    for arch in ARCH_NAMES:
        safe = arch.replace("/", "_").replace("-", "_")
        cache_fp = CACHE_DIR / f"cassava_{safe}.npy"
        arr = np.load(cache_fp)
        assert len(arr) == n_expected, f"Cached feature length mismatch for {arch}: {len(arr)} != {n_expected}"
        all_feats[arch] = arr
    return all_feats


def main() -> None:
    parser = argparse.ArgumentParser(description="DGS-Bench v1.1 extension suite.")
    parser.add_argument("--replot-only", action="store_true", help="Regenerate Exp A & Exp C charts from saved CSVs.")
    parser.add_argument("--gradcam-only", action="store_true", help="Re-run Experiment B Grad-CAM using cached features.")
    args = parser.parse_args()

    if args.replot_only:
        df_sum = pd.read_csv(RESULTS_DIR / "v11_calibration_priorshift_summary.csv")
        df_ft = pd.read_csv(RESULTS_DIR / "v11_multibackbone_finetune.csv")
        plot_experiment_a_calibration(df_sum)
        plot_experiment_c_multibackbone_finetune(df_ft)
        print("Replotted Exp A & Exp C charts cleanly from CSV.", flush=True)
        if not args.gradcam_only:
            return

    df = pd.read_csv(DATA_DIR / "train.csv")
    paths = np.array([str(DATA_DIR / "train_images" / iid) for iid in df["image_id"].values])
    labels = df["label"].values.astype(int)
    print(f"Loaded {len(paths)} cassava image paths. Device={DEVICE}", flush=True)

    if args.gradcam_only:
        all_feats = load_cached_features(len(labels))
        run_experiment_b_gradcam(all_feats, paths, labels)
        print("Grad-CAM Experiment B update complete.", flush=True)
        return

    x_all = preload_cassava_224(paths)
    all_feats = extract_or_load_features(x_all, labels)
    run_experiment_a_calibration(all_feats, labels)
    run_experiment_b_gradcam(all_feats, paths, labels)
    run_experiment_c_multibackbone_finetune(all_feats, x_all, labels)
    run_experiment_d_cara_cassava(x_all, labels)
    print("\nALL DGS-Bench v1.1 EXTENSION EXPERIMENTS COMPLETE.", flush=True)


if __name__ == "__main__":
    main()
