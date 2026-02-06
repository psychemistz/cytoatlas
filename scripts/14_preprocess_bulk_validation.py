#!/usr/bin/env python3
"""Preprocess cross-sample correlation CSVs into JSON for visualization.

Reads correlation CSVs from results/cross_sample_validation/correlations/
and outputs visualization/data/bulk_donor_correlations.json with:
  - summary: aggregated stats per atlas/level/signature
  - donor_level: per-target rows where celltype == "all" (donor-level)
  - celltype_level: aggregated per celltype + top 20 targets per celltype
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORR_DIR = PROJECT_ROOT / "results" / "cross_sample_validation" / "correlations"
OUTPUT_PATH = PROJECT_ROOT / "visualization" / "data" / "bulk_donor_correlations.json"

# Atlas CSV files (key = atlas name in output JSON)
ATLAS_FILES = {
    "cima": "cima_correlations.csv",
    "inflammation_main": "inflammation_main_correlations.csv",
    "inflammation_val": "inflammation_val_correlations.csv",
    "inflammation_ext": "inflammation_ext_correlations.csv",
    "scatlas_normal": "scatlas_normal_correlations.csv",
    "scatlas_cancer": "scatlas_cancer_correlations.csv",
}


def round_val(v, decimals=4):
    """Round a value, handling NaN/None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), decimals)


def build_summary(summary_csv: Path) -> list:
    """Read correlation_summary.csv and return list of dicts."""
    df = pd.read_csv(summary_csv)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "atlas": row["atlas"],
            "level": row["level"],
            "signature": row["signature"],
            "n_targets": int(row["n_targets_matched"]),
            "n_significant": int(row["n_significant_p05"]),
            "n_positive": int(row["n_positive_corr"]),
            "n_strong": int(row["n_strong_corr_03"]),
            "pct_significant": round_val(row["pct_significant"], 2),
            "pct_positive": round_val(row["pct_positive"], 2),
            "median_rho": round_val(row["median_rho"]),
            "mean_rho": round_val(row["mean_rho"]),
            "std_rho": round_val(row["std_rho"]),
            "min_rho": round_val(row["min_rho"]),
            "max_rho": round_val(row["max_rho"]),
            "median_n_samples": round_val(row["median_n_samples"], 0),
        })
    return rows


def build_donor_level(atlas_key: str, csv_path: Path) -> dict:
    """Extract donor-level rows (celltype == 'all') per signature type."""
    df = pd.read_csv(csv_path)

    # Donor-level = rows where celltype is "all"
    donor_df = df[df["celltype"] == "all"].copy()

    result = {}
    for sig_type in donor_df["signature"].unique():
        sig_df = donor_df[donor_df["signature"] == sig_type]
        targets = []
        for _, row in sig_df.iterrows():
            targets.append({
                "target": row["target"],
                "gene": row["gene"],
                "level": row["level"],
                "rho": round_val(row["spearman_rho"]),
                "pval": round_val(row["spearman_pval"], 6),
                "n": int(row["n_samples"]),
                "mean_expr": round_val(row["mean_expr"]),
                "mean_activity": round_val(row["mean_activity"]),
                "significant": bool(row["spearman_pval"] < 0.05) if pd.notna(row["spearman_pval"]) else False,
            })
        result[sig_type] = targets
    return result


def build_celltype_level(atlas_key: str, csv_path: Path) -> dict:
    """Build celltype-stratified data: aggregated per celltype + top 20 targets."""
    df = pd.read_csv(csv_path)

    # Celltype-level = rows where celltype != "all"
    ct_df = df[df["celltype"] != "all"].copy()

    if ct_df.empty:
        return {}

    result = {}
    for level in ct_df["level"].unique():
        level_df = ct_df[ct_df["level"] == level]
        level_result = {}

        for sig_type in level_df["signature"].unique():
            sig_df = level_df[level_df["signature"] == sig_type]
            celltypes = sorted(sig_df["celltype"].unique().tolist())

            per_celltype = []
            for ct in celltypes:
                ct_data = sig_df[sig_df["celltype"] == ct]
                rho_vals = ct_data["spearman_rho"].dropna()
                pvals = ct_data["spearman_pval"].dropna()
                n_sig = int((pvals < 0.05).sum()) if len(pvals) > 0 else 0

                # Top 20 targets by |rho|
                top_targets = (
                    ct_data.assign(abs_rho=ct_data["spearman_rho"].abs())
                    .nlargest(20, "abs_rho")
                )
                top_list = []
                for _, row in top_targets.iterrows():
                    top_list.append({
                        "target": row["target"],
                        "gene": row["gene"],
                        "rho": round_val(row["spearman_rho"]),
                        "pval": round_val(row["spearman_pval"], 6),
                        "n": int(row["n_samples"]),
                        "significant": bool(row["spearman_pval"] < 0.05) if pd.notna(row["spearman_pval"]) else False,
                    })

                per_celltype.append({
                    "celltype": ct,
                    "median_rho": round_val(rho_vals.median()) if len(rho_vals) > 0 else None,
                    "mean_rho": round_val(rho_vals.mean()) if len(rho_vals) > 0 else None,
                    "n_targets": int(len(ct_data)),
                    "n_significant": n_sig,
                    "pct_significant": round_val(100.0 * n_sig / len(ct_data), 2) if len(ct_data) > 0 else 0,
                    "top_targets": top_list,
                })

            level_result[sig_type] = {
                "celltypes": celltypes,
                "per_celltype": per_celltype,
            }

        result[level] = level_result

    return result


def main():
    print("Building bulk_donor_correlations.json ...")

    # 1. Summary
    summary_csv = CORR_DIR / "correlation_summary.csv"
    if not summary_csv.exists():
        print(f"ERROR: {summary_csv} not found", file=sys.stderr)
        sys.exit(1)
    summary = build_summary(summary_csv)
    print(f"  Summary: {len(summary)} rows")

    # 2. Donor-level and celltype-level per atlas
    donor_level = {}
    celltype_level = {}

    for atlas_key, filename in ATLAS_FILES.items():
        csv_path = CORR_DIR / filename
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping {atlas_key}")
            continue

        print(f"  Processing {atlas_key} ...")
        donor_level[atlas_key] = build_donor_level(atlas_key, csv_path)
        celltype_level[atlas_key] = build_celltype_level(atlas_key, csv_path)

        # Print stats
        for sig, targets in donor_level[atlas_key].items():
            print(f"    donor_level/{sig}: {len(targets)} targets")
        for level, sigs in celltype_level[atlas_key].items():
            for sig, data in sigs.items():
                print(f"    celltype_level/{level}/{sig}: {len(data['celltypes'])} celltypes")

    # 3. Assemble output
    output = {
        "summary": summary,
        "donor_level": donor_level,
        "celltype_level": celltype_level,
    }

    # 4. Write JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"Size: {size_mb:.1f} MB")
    print("Done!")


if __name__ == "__main__":
    main()
