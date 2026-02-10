#!/usr/bin/env python3
"""Preprocess perturbation data (Parse 10M + Tahoe) for web visualization.

Reads analysis CSVs from results/parse10m/ and results/tahoe/, transforms them
into compact JSON files for the interactive dashboard.

Output files (visualization/data/):
  - parse10m_cytokine_heatmap.json   (90 cytokines x 18 cell types activity)
  - parse10m_ground_truth.json       (predicted vs actual scatter data)
  - parse10m_donor_variability.json  (cross-donor consistency)
  - tahoe_drug_sensitivity.json      (95 drugs x 50 cell lines)
  - tahoe_dose_response.json         (Plate 13 dose curves)
  - tahoe_pathway_activation.json    (drug -> pathway mapping)

Usage:
    python scripts/24_preprocess_perturbation_viz.py
    python scripts/24_preprocess_perturbation_viz.py --parse10m   # Parse 10M only
    python scripts/24_preprocess_perturbation_viz.py --tahoe      # Tahoe only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PARSE10M_DIR = RESULTS_DIR / "parse10m"
TAHOE_DIR = RESULTS_DIR / "tahoe"
OUTPUT_DIR = PROJECT_ROOT / "visualization" / "data"


def log(msg: str) -> None:
    """Timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def round_floats(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Round float columns in dataframe to reduce JSON size."""
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    for col in float_cols:
        df[col] = df[col].round(decimals)
    return df


def save_json(data, output_path: Path, label: str) -> None:
    """Write data to JSON and print size info."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    size_kb = output_path.stat().st_size / 1024
    log(f"  Wrote {output_path.name} ({size_kb:.1f} KB) - {label}")


# ---------------------------------------------------------------------------
# Parse 10M preprocessing
# ---------------------------------------------------------------------------
def preprocess_parse10m_cytokine_heatmap() -> bool:
    """Preprocess Parse 10M cytokine response matrix into heatmap JSON.

    Input: parse10m_cytokine_response_matrix.csv
           Expected columns: cytokine, cell_type, activity (+ optional metadata)
    Output: parse10m_cytokine_heatmap.json
            {cytokines: [...], cell_types: [...], matrix: [[...]], metadata: {...}}
    """
    log("Processing Parse 10M cytokine heatmap...")
    csv_path = PARSE10M_DIR / "parse10m_cytokine_response_matrix.csv"
    if not csv_path.exists():
        log(f"  SKIP: {csv_path} not found")
        return False

    df = pd.read_csv(csv_path)
    df = round_floats(df)
    log(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Build heatmap structure: pivot cytokines (rows) x cell types (columns)
    # Detect the activity/value column
    value_col = None
    for candidate in ["activity", "mean_activity", "activity_score", "value", "score"]:
        if candidate in df.columns:
            value_col = candidate
            break
    if value_col is None:
        # Fall back to last numeric column
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            value_col = num_cols[-1]
        else:
            log("  ERROR: No numeric value column found")
            return False

    # Detect cytokine and cell type columns
    cytokine_col = None
    for candidate in ["cytokine", "signature", "target", "protein"]:
        if candidate in df.columns:
            cytokine_col = candidate
            break
    celltype_col = None
    for candidate in ["cell_type", "celltype", "cell_type_l1", "Level1"]:
        if candidate in df.columns:
            celltype_col = candidate
            break

    if cytokine_col is None or celltype_col is None:
        log(f"  ERROR: Could not identify cytokine col ({cytokine_col}) or celltype col ({celltype_col})")
        return False

    # Pivot to matrix form
    pivot = df.pivot_table(
        index=cytokine_col, columns=celltype_col, values=value_col, aggfunc="mean"
    ).fillna(0)
    pivot = pivot.round(4)

    cytokines = pivot.index.tolist()
    cell_types = pivot.columns.tolist()
    matrix = pivot.values.tolist()

    output = {
        "cytokines": cytokines,
        "cell_types": cell_types,
        "matrix": matrix,
        "metadata": {
            "n_cytokines": len(cytokines),
            "n_cell_types": len(cell_types),
            "value_column": value_col,
            "source": "parse10m",
        },
    }

    save_json(output, OUTPUT_DIR / "parse10m_cytokine_heatmap.json",
              f"{len(cytokines)} cytokines x {len(cell_types)} cell types")
    return True


def preprocess_parse10m_ground_truth() -> bool:
    """Preprocess Parse 10M ground truth validation into scatter JSON.

    Input: parse10m_ground_truth_validation.csv
           Expected columns: target, predicted, actual (+ optional: cell_type, rho, pvalue)
    Output: parse10m_ground_truth.json
            {scatter: [...], summary: {...}}
    """
    log("Processing Parse 10M ground truth validation...")
    csv_path = PARSE10M_DIR / "parse10m_ground_truth_validation.csv"
    if not csv_path.exists():
        log(f"  SKIP: {csv_path} not found")
        return False

    df = pd.read_csv(csv_path)
    df = round_floats(df)
    log(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Detect predicted and actual columns
    pred_col = None
    for candidate in ["predicted", "predicted_activity", "pred", "activity"]:
        if candidate in df.columns:
            pred_col = candidate
            break
    actual_col = None
    for candidate in ["actual", "measured", "observed", "ground_truth", "expression"]:
        if candidate in df.columns:
            actual_col = candidate
            break

    if pred_col is None or actual_col is None:
        log(f"  ERROR: Could not identify predicted ({pred_col}) or actual ({actual_col}) columns")
        return False

    # Build scatter records
    scatter = df.to_dict(orient="records")

    # Compute summary statistics
    from scipy import stats as sp_stats
    valid = df[[pred_col, actual_col]].dropna()
    summary = {"n_total": len(df), "n_valid": len(valid)}
    if len(valid) >= 3:
        rho, pval = sp_stats.spearmanr(valid[pred_col], valid[actual_col])
        summary["spearman_rho"] = round(float(rho), 4)
        summary["spearman_pval"] = float(pval)
        r, r_pval = sp_stats.pearsonr(valid[pred_col], valid[actual_col])
        summary["pearson_r"] = round(float(r), 4)
        summary["pearson_pval"] = float(r_pval)

    # Per-target summary if target column exists
    target_col = None
    for candidate in ["target", "cytokine", "signature", "protein"]:
        if candidate in df.columns:
            target_col = candidate
            break
    per_target = []
    if target_col is not None:
        for target, grp in df.groupby(target_col):
            valid_grp = grp[[pred_col, actual_col]].dropna()
            rec = {"target": target, "n": len(valid_grp)}
            if len(valid_grp) >= 3:
                rho, pval = sp_stats.spearmanr(valid_grp[pred_col], valid_grp[actual_col])
                rec["rho"] = round(float(rho), 4)
                rec["pvalue"] = float(pval)
            per_target.append(rec)

    output = {
        "scatter": scatter,
        "summary": summary,
        "per_target": per_target,
        "columns": {"predicted": pred_col, "actual": actual_col, "target": target_col},
    }

    save_json(output, OUTPUT_DIR / "parse10m_ground_truth.json",
              f"{len(scatter)} scatter points, {len(per_target)} targets")
    return True


def preprocess_parse10m_donor_variability() -> bool:
    """Preprocess Parse 10M treatment vs control into donor variability JSON.

    Input: parse10m_treatment_vs_control.csv
           Expected columns: donor, cytokine/target, cell_type, activity_diff, pvalue, etc.
    Output: parse10m_donor_variability.json
            {records: [...], donors: [...], targets: [...], summary: {...}}
    """
    log("Processing Parse 10M donor variability...")
    csv_path = PARSE10M_DIR / "parse10m_treatment_vs_control.csv"
    if not csv_path.exists():
        log(f"  SKIP: {csv_path} not found")
        return False

    df = pd.read_csv(csv_path)
    df = round_floats(df)
    log(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Detect key columns
    donor_col = None
    for candidate in ["donor", "donor_id", "sample", "subject"]:
        if candidate in df.columns:
            donor_col = candidate
            break

    target_col = None
    for candidate in ["cytokine", "target", "signature", "protein"]:
        if candidate in df.columns:
            target_col = candidate
            break

    # Build output records
    records = df.to_dict(orient="records")

    # Summary: unique donors, targets, and cross-donor consistency
    donors = sorted(df[donor_col].unique().tolist()) if donor_col and donor_col in df.columns else []
    targets = sorted(df[target_col].unique().tolist()) if target_col and target_col in df.columns else []

    # Compute cross-donor consistency per target (mean and std of activity_diff)
    consistency = []
    diff_col = None
    for candidate in ["activity_diff", "diff", "log2fc", "mean_diff", "effect_size"]:
        if candidate in df.columns:
            diff_col = candidate
            break

    if target_col and diff_col and donor_col:
        for target, grp in df.groupby(target_col):
            per_donor = grp.groupby(donor_col)[diff_col].mean()
            consistency.append({
                "target": target,
                "n_donors": len(per_donor),
                "mean_diff": round(float(per_donor.mean()), 4),
                "std_diff": round(float(per_donor.std()), 4),
                "cv": round(float(per_donor.std() / abs(per_donor.mean())), 4)
                if abs(per_donor.mean()) > 1e-10
                else None,
            })

    output = {
        "records": records,
        "donors": donors,
        "targets": targets,
        "consistency": consistency,
        "summary": {
            "n_records": len(records),
            "n_donors": len(donors),
            "n_targets": len(targets),
        },
    }

    save_json(output, OUTPUT_DIR / "parse10m_donor_variability.json",
              f"{len(records)} records, {len(donors)} donors, {len(targets)} targets")
    return True


# ---------------------------------------------------------------------------
# Tahoe preprocessing
# ---------------------------------------------------------------------------
def preprocess_tahoe_drug_sensitivity() -> bool:
    """Preprocess Tahoe drug sensitivity matrix into JSON.

    Input: tahoe_drug_sensitivity_matrix.csv
           Expected: drugs (rows) x cell lines (columns) or long-format with drug, cell_line, value
    Output: tahoe_drug_sensitivity.json
            {drugs: [...], cell_lines: [...], matrix: [[...]], metadata: {...}}
    """
    log("Processing Tahoe drug sensitivity...")
    csv_path = TAHOE_DIR / "tahoe_drug_sensitivity_matrix.csv"
    if not csv_path.exists():
        log(f"  SKIP: {csv_path} not found")
        return False

    df = pd.read_csv(csv_path)
    df = round_floats(df)
    log(f"  Loaded {len(df)} rows x {len(df.columns)} columns")

    # Determine if matrix (wide) or long format
    drug_col = None
    for candidate in ["drug", "compound", "treatment", "drug_name"]:
        if candidate in df.columns:
            drug_col = candidate
            break

    cell_line_col = None
    for candidate in ["cell_line", "cellline", "cell_line_name", "sample"]:
        if candidate in df.columns:
            cell_line_col = candidate
            break

    if drug_col and cell_line_col:
        # Long format — pivot to matrix
        value_col = None
        for candidate in ["sensitivity", "activity", "score", "value", "ic50", "auc"]:
            if candidate in df.columns:
                value_col = candidate
                break
        if value_col is None:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            value_col = num_cols[0] if num_cols else None

        if value_col is None:
            log("  ERROR: No numeric value column found")
            return False

        pivot = df.pivot_table(
            index=drug_col, columns=cell_line_col, values=value_col, aggfunc="mean"
        ).fillna(0).round(4)
        drugs = pivot.index.tolist()
        cell_lines = pivot.columns.tolist()
        matrix = pivot.values.tolist()
    else:
        # Wide format — first column is drug names, rest are cell lines
        first_col = df.columns[0]
        drugs = df[first_col].tolist()
        cell_lines = df.columns[1:].tolist()
        matrix = df.iloc[:, 1:].fillna(0).round(4).values.tolist()

    output = {
        "drugs": drugs,
        "cell_lines": cell_lines,
        "matrix": matrix,
        "metadata": {
            "n_drugs": len(drugs),
            "n_cell_lines": len(cell_lines),
            "source": "tahoe",
        },
    }

    save_json(output, OUTPUT_DIR / "tahoe_drug_sensitivity.json",
              f"{len(drugs)} drugs x {len(cell_lines)} cell lines")
    return True


def preprocess_tahoe_dose_response() -> bool:
    """Preprocess Tahoe dose-response curves (Plate 13) into JSON.

    Input: tahoe_dose_response.csv
           Expected columns: drug, dose/concentration, response/activity, cell_line (optional)
    Output: tahoe_dose_response.json
            {curves: {drug: [{dose, response}, ...]}, drugs: [...], metadata: {...}}
    """
    log("Processing Tahoe dose-response curves...")
    csv_path = TAHOE_DIR / "tahoe_dose_response.csv"
    if not csv_path.exists():
        log(f"  SKIP: {csv_path} not found")
        return False

    df = pd.read_csv(csv_path)
    df = round_floats(df)
    log(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Detect columns
    drug_col = None
    for candidate in ["drug", "compound", "treatment", "drug_name"]:
        if candidate in df.columns:
            drug_col = candidate
            break

    dose_col = None
    for candidate in ["dose", "concentration", "conc", "dose_um", "log_dose"]:
        if candidate in df.columns:
            dose_col = candidate
            break

    response_col = None
    for candidate in ["response", "activity", "viability", "value", "score", "effect"]:
        if candidate in df.columns:
            response_col = candidate
            break

    if drug_col is None or dose_col is None or response_col is None:
        log(f"  ERROR: Missing required columns — drug={drug_col}, dose={dose_col}, response={response_col}")
        return False

    # Group by drug to build curves
    curves = {}
    for drug, grp in df.groupby(drug_col):
        drug_str = str(drug)
        # Sort by dose
        grp_sorted = grp.sort_values(dose_col)
        points = []
        for _, row in grp_sorted.iterrows():
            point = {"dose": row[dose_col], "response": row[response_col]}
            # Include cell line if present
            for cl_col in ["cell_line", "cellline", "cell_line_name"]:
                if cl_col in grp_sorted.columns:
                    point["cell_line"] = row[cl_col]
                    break
            points.append(point)
        curves[drug_str] = points

    drugs = sorted(curves.keys())

    output = {
        "curves": curves,
        "drugs": drugs,
        "metadata": {
            "n_drugs": len(drugs),
            "n_total_points": len(df),
            "dose_column": dose_col,
            "response_column": response_col,
            "source": "tahoe",
        },
    }

    save_json(output, OUTPUT_DIR / "tahoe_dose_response.json",
              f"{len(drugs)} drugs, {len(df)} data points")
    return True


def preprocess_tahoe_pathway_activation() -> bool:
    """Preprocess Tahoe cytokine pathway activation into JSON.

    Input: tahoe_cytokine_pathway_activation.csv
           Expected columns: drug, pathway/cytokine, activation_score, pvalue, etc.
    Output: tahoe_pathway_activation.json
            {records: [...], drugs: [...], pathways: [...], metadata: {...}}
    """
    log("Processing Tahoe pathway activation...")
    csv_path = TAHOE_DIR / "tahoe_cytokine_pathway_activation.csv"
    if not csv_path.exists():
        log(f"  SKIP: {csv_path} not found")
        return False

    df = pd.read_csv(csv_path)
    df = round_floats(df)
    log(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Detect columns
    drug_col = None
    for candidate in ["drug", "compound", "treatment", "drug_name"]:
        if candidate in df.columns:
            drug_col = candidate
            break

    pathway_col = None
    for candidate in ["pathway", "cytokine", "target", "signature"]:
        if candidate in df.columns:
            pathway_col = candidate
            break

    # Build output
    records = df.to_dict(orient="records")
    drugs = sorted(df[drug_col].unique().tolist()) if drug_col and drug_col in df.columns else []
    pathways = sorted(df[pathway_col].unique().tolist()) if pathway_col and pathway_col in df.columns else []

    # Build drug -> pathway mapping for quick lookup
    drug_pathway_map = {}
    if drug_col and pathway_col:
        score_col = None
        for candidate in ["activation_score", "activity", "score", "value", "effect"]:
            if candidate in df.columns:
                score_col = candidate
                break

        for drug, grp in df.groupby(drug_col):
            drug_str = str(drug)
            entries = []
            for _, row in grp.iterrows():
                entry = {"pathway": row[pathway_col]}
                if score_col:
                    entry["score"] = row[score_col]
                if "pvalue" in grp.columns:
                    entry["pvalue"] = row["pvalue"]
                if "qvalue" in grp.columns:
                    entry["qvalue"] = row["qvalue"]
                entries.append(entry)
            # Sort by absolute score descending
            if score_col:
                entries.sort(key=lambda x: abs(x.get("score", 0)), reverse=True)
            drug_pathway_map[drug_str] = entries

    output = {
        "records": records,
        "drugs": drugs,
        "pathways": pathways,
        "drug_pathway_map": drug_pathway_map,
        "metadata": {
            "n_records": len(records),
            "n_drugs": len(drugs),
            "n_pathways": len(pathways),
            "source": "tahoe",
        },
    }

    save_json(output, OUTPUT_DIR / "tahoe_pathway_activation.json",
              f"{len(records)} records, {len(drugs)} drugs, {len(pathways)} pathways")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess perturbation data (Parse 10M + Tahoe) for web visualization"
    )
    parser.add_argument(
        "--parse10m", action="store_true",
        help="Process Parse 10M data only",
    )
    parser.add_argument(
        "--tahoe", action="store_true",
        help="Process Tahoe data only",
    )
    args = parser.parse_args()

    # If neither flag is set, process all
    do_parse10m = args.parse10m or (not args.parse10m and not args.tahoe)
    do_tahoe = args.tahoe or (not args.parse10m and not args.tahoe)

    t0 = time.time()
    log("=" * 60)
    log("Preprocessing Perturbation Visualization Data")
    log("=" * 60)

    results = {}

    if do_parse10m:
        log("\n--- Parse 10M ---")
        results["parse10m_cytokine_heatmap"] = preprocess_parse10m_cytokine_heatmap()
        results["parse10m_ground_truth"] = preprocess_parse10m_ground_truth()
        results["parse10m_donor_variability"] = preprocess_parse10m_donor_variability()

    if do_tahoe:
        log("\n--- Tahoe ---")
        results["tahoe_drug_sensitivity"] = preprocess_tahoe_drug_sensitivity()
        results["tahoe_dose_response"] = preprocess_tahoe_dose_response()
        results["tahoe_pathway_activation"] = preprocess_tahoe_pathway_activation()

    # Summary
    elapsed = time.time() - t0
    n_ok = sum(1 for v in results.values() if v)
    n_skip = sum(1 for v in results.values() if not v)
    log("")
    log("=" * 60)
    log(f"Done in {elapsed:.1f}s — {n_ok} succeeded, {n_skip} skipped")
    for name, ok in results.items():
        status = "OK" if ok else "SKIPPED"
        log(f"  {name}: {status}")
    log(f"Output directory: {OUTPUT_DIR}")
    log("=" * 60)


if __name__ == "__main__":
    main()
