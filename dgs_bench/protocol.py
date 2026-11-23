"""Algorithm 1: Metric-Aware DGS Computation Protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence
import numpy as np

from .calibration import compute_brier, compute_ece
from .metrics import compute_split_metrics, theoretical_mirage_threshold


@dataclass
class DGSReport:
    """Structured output of Algorithm 1 (Metric-Aware DGS Computation Protocol)."""

    lab_metrics: Dict[str, object]
    field_metrics: Dict[str, object]
    dgs_acc: float
    dgs_bacc: float
    dgs_f1: float
    dgs_mcc: float
    dgs_auroc: float
    dgs_auprc: float
    lab_ece: float
    field_ece: float
    lab_brier: float
    field_brier: float
    majority_class: str
    majority_prevalence: float
    imbalance_warning: bool
    mirage_detected: bool
    theoretical_p_star_pct: float
    warning_message: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def to_markdown(self) -> str:
        warn = (
            f"WARNING: Field majority class '{self.majority_class}' prevalence is "
            f"{self.majority_prevalence * 100:.1f}% (>30.0%). Report DGS_F1 and DGS_AUPRC as primary metrics."
            if self.imbalance_warning
            else f"Balanced/moderate field majority ({self.majority_prevalence * 100:.1f}%)."
        )
        lines = [
            f"### DGS-Bench Protocol Report ({warn})",
            "",
            "| Metric | Lab (%) | Field (%) | DGS (Lab - Field, pp) |",
            "|---|---:|---:|---:|",
            f"| Accuracy | {self.lab_metrics['acc']:.2f} | {self.field_metrics['acc']:.2f} | {self.dgs_acc:+.2f} |",
            f"| Balanced Accuracy | {self.lab_metrics['bacc']:.2f} | {self.field_metrics['bacc']:.2f} | {self.dgs_bacc:+.2f} |",
            f"| Macro-F1 | {self.lab_metrics['f1']:.2f} | {self.field_metrics['f1']:.2f} | {self.dgs_f1:+.2f} |",
            f"| Macro-AUPRC | {self.lab_metrics['auprc']:.2f} | {self.field_metrics['auprc']:.2f} | {self.dgs_auprc:+.2f} |",
            f"| Macro-AUROC | {self.lab_metrics['auroc']:.2f} | {self.field_metrics['auroc']:.2f} | {self.dgs_auroc:+.2f} |",
            f"| MCC (x100) | {self.lab_metrics['mcc']:.2f} | {self.field_metrics['mcc']:.2f} | {self.dgs_mcc:+.2f} |",
            "",
            f"- Mirage detected (`DGS_Acc < 0 < DGS_F1`): **{self.mirage_detected}**",
            f"- Theoretical threshold $p^*$: **{self.theoretical_p_star_pct:.2f}%**",
            f"- Calibration: Lab ECE={self.lab_ece:.4f}, Field ECE={self.field_ece:.4f} | "
            f"Lab Brier={self.lab_brier:.4f}, Field Brier={self.field_brier:.4f}",
        ]
        return "\n".join(lines)


def compute_dgs_report(
    y_lab_true: Sequence[int],
    y_lab_prob: np.ndarray,
    y_field_true: Sequence[int],
    y_field_prob: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    majority_threshold: float = 0.30,
) -> DGSReport:
    """Execute Algorithm 1: evaluate balanced lab test vs. natural field test sets."""
    yf = np.asarray(y_field_true, dtype=int)
    n_classes = int(np.asarray(y_lab_prob).shape[1])
    names = list(class_names) if class_names is not None else [f"Class_{i}" for i in range(n_classes)]

    lab_m = compute_split_metrics(y_lab_true, y_lab_prob, class_names=names)
    fld_m = compute_split_metrics(yf, y_field_prob, class_names=names)

    counts = np.bincount(yf, minlength=n_classes)
    maj_idx = int(np.argmax(counts))
    maj_prev = float(counts[maj_idx] / max(len(yf), 1))
    maj_name = names[maj_idx]
    imbalance_warn = maj_prev > majority_threshold

    dgs_acc = float(lab_m["acc"] - fld_m["acc"])
    dgs_bacc = float(lab_m["bacc"] - fld_m["bacc"])
    dgs_f1 = float(lab_m["f1"] - fld_m["f1"])
    dgs_mcc = float(lab_m["mcc"] - fld_m["mcc"])
    dgs_auroc = float(lab_m["auroc"] - fld_m["auroc"])
    dgs_auprc = float(lab_m["auprc"] - fld_m["auprc"])

    other_recs = [
        fld_m["per_class_recall"][names[c]]
        for c in range(n_classes)
        if c != maj_idx and not np.isnan(fld_m["per_class_recall"][names[c]])
    ]
    dom_rec = float(fld_m["per_class_recall"][maj_name])
    other_mean = float(np.mean(other_recs)) if other_recs else float("nan")
    p_star = theoretical_mirage_threshold(float(lab_m["acc"]), dom_rec, other_mean)

    warning_msg = (
        f"WARNING: report DGS_F1 as primary metric (majority '{maj_name}' prevalence "
        f"{maj_prev * 100:.1f}% > {majority_threshold * 100:.0f}%)."
        if imbalance_warn
        else ""
    )

    return DGSReport(
        lab_metrics=lab_m,
        field_metrics=fld_m,
        dgs_acc=dgs_acc,
        dgs_bacc=dgs_bacc,
        dgs_f1=dgs_f1,
        dgs_mcc=dgs_mcc,
        dgs_auroc=dgs_auroc,
        dgs_auprc=dgs_auprc,
        lab_ece=compute_ece(y_lab_true, y_lab_prob),
        field_ece=compute_ece(yf, y_field_prob),
        lab_brier=compute_brier(y_lab_true, y_lab_prob),
        field_brier=compute_brier(yf, y_field_prob),
        majority_class=maj_name,
        majority_prevalence=maj_prev,
        imbalance_warning=imbalance_warn,
        mirage_detected=(dgs_acc < 0.0 and dgs_f1 > 0.0),
        theoretical_p_star_pct=p_star,
        warning_message=warning_msg,
    )
