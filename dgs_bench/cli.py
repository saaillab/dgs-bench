"""CLI entrypoint for DGS-Bench protocol verification and report generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .metrics import theoretical_mirage_threshold
from .protocol import compute_dgs_report


def verify_theoretical_results(results_dir: Path) -> bool:
    """Verify closed-form mirage threshold p* matches published Table 8 values."""
    tv_fp = results_dir / "theoretical_validation.csv"
    df = pd.read_csv(tv_fp)
    all_ok = True
    for _, row in df.iterrows():
        computed = theoretical_mirage_threshold(
            float(row["lab_acc"]), float(row["cmd_recall"]), float(row["other_recall"])
        )
        diff = abs(computed - float(row["p_star"]))
        ok = diff < 1e-6
        all_ok = all_ok and ok
        print(
            f"[verify] {row['arch']:16s} | p* expected={row['p_star']:7.2f}% "
            f"| computed={computed:7.2f}% | diff={diff:.2e} | match={ok}"
        )
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="DGS-Bench Metric-Aware Evaluation CLI")
    parser.add_argument("--demo-verify", action="store_true", help="Verify closed-form p* against saved CSVs")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory containing CSV results")
    parser.add_argument("--lab-npz", type=Path, help="Optional .npz containing y_true, y_prob for lab test split")
    parser.add_argument("--field-npz", type=Path, help="Optional .npz containing y_true, y_prob for field test split")
    parser.add_argument("--class-names", nargs="*", help="Optional ordered class names")
    parser.add_argument("--out-json", type=Path, help="Optional output JSON report path")
    args = parser.parse_args()

    if args.demo_verify:
        ok = verify_theoretical_results(args.results_dir)
        ms = pd.read_csv(args.results_dir / "multiseed_dgs.csv")
        print(
            f"[summary] n={len(ms)} runs | DGS_Acc={ms['dgs_acc'].mean():+.2f}+/-{ms['dgs_acc'].std(ddof=0):.2f} pp | "
            f"DGS_F1={ms['dgs_f1'].mean():+.2f}+/-{ms['dgs_f1'].std(ddof=0):.2f} pp | mirage={ms['mirage'].sum()}/{len(ms)}"
        )
        raise SystemExit(0 if ok else 1)

    if args.lab_npz and args.field_npz:
        lab = np.load(args.lab_npz)
        fld = np.load(args.field_npz)
        report = compute_dgs_report(
            lab["y_true"], lab["y_prob"], fld["y_true"], fld["y_prob"], class_names=args.class_names
        )
        print(report.to_markdown())
        if args.out_json:
            args.out_json.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
