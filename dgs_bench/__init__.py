"""DGS-Bench: Metric-aware Deployment Gap Score protocol and calibration suite."""

from .calibration import (
    compute_brier,
    compute_ece,
    fit_temperature_scaling,
    apply_temperature_scaling,
    saerens_em_prior_shift,
)
from .metrics import (
    compute_split_metrics,
    theoretical_mirage_threshold,
)
from .protocol import (
    DGSReport,
    compute_dgs_report,
)
from .splits import (
    create_dgs_splits,
    load_cassava_splits_json,
)

__all__ = [
    "DGSReport",
    "apply_temperature_scaling",
    "compute_brier",
    "compute_dgs_report",
    "compute_ece",
    "compute_split_metrics",
    "create_dgs_splits",
    "fit_temperature_scaling",
    "load_cassava_splits_json",
    "saerens_em_prior_shift",
    "theoretical_mirage_threshold",
]

__version__ = "1.1.0"
