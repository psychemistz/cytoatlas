#!/usr/bin/env python3
"""Preprocess spatial transcriptomics (SpatialCorpus) data for web visualization.

Reads analysis CSVs from results/spatial/ and transforms them into compact JSON
files for the interactive dashboard.

Output files (visualization/data/):
  - spatial_tissue_activity.json        (tissue-level activity)
  - spatial_technology_comparison.json   (cross-technology reproducibility)
  - spatial_gene_coverage.json           (gene panel coverage per technology)
  - spatial_dataset_catalog.json         (251 dataset metadata)

Usage:
    python scripts/25_preprocess_spatial_viz.py
    python scripts/25_preprocess_spatial_viz.py --tissue     # Tissue activity only
    python scripts/25_preprocess_spatial_viz.py --technology # Technology comparison only
    python scripts/25_preprocess_spatial_viz.py --coverage   # Gene coverage only
    python scripts/25_preprocess_spatial_viz.py --catalog    # Dataset catalog only
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
SPATIAL_DIR = RESULTS_DIR / "spatial"
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
# Spatial preprocessing functions
# ---------------------------------------------------------------------------
def preprocess_spatial_tissue_activity() -> bool:
    """Preprocess spatial activity by tissue into visualization JSON.

    Input: spatial_activity_by_tissue.csv
           Expected columns: tissue, signature/cytokine, activity, technology, n_spots, etc.
    Output: spatial_tissue_activity.json
            {records: [...], tissues: [...], signatures: [...], by_tissue: {...}, metadata: {...}}
    """
    log("Processing spatial tissue activity...")
    csv_path = SPATIAL_DIR / "spatial_activity_by_tissue.csv"
    if not csv_path.exists():
        log(f"  SKIP: {csv_path} not found")
        return False

    df = pd.read_csv(csv_path)
    df = round_floats(df)
    log(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Detect key columns
    tissue_col = None
    for candidate in ["tissue", "tissue_type", "organ", "region"]:
        if candidate in df.columns:
            tissue_col = candidate
            break

    sig_col = None
    for candidate in ["signature", "cytokine", "target", "protein"]:
        if candidate in df.columns:
            sig_col = candidate
            break

    activity_col = None
    for candidate in ["activity", "mean_activity", "activity_score", "value", "score"]:
        if candidate in df.columns:
            activity_col = candidate
            break

    if tissue_col is None or activity_col is None:
        log(f"  ERROR: Missing required columns — tissue={tissue_col}, activity={activity_col}")
        return False

    records = df.to_dict(orient="records")
    tissues = sorted(df[tissue_col].unique().tolist())
    signatures = sorted(df[sig_col].unique().tolist()) if sig_col and sig_col in df.columns else []

    # Build per-tissue summary
    by_tissue = {}
    for tissue, grp in df.groupby(tissue_col):
        tissue_str = str(tissue)
        summary = {
            "n_records": len(grp),
            "mean_activity": round(float(grp[activity_col].mean()), 4),
            "std_activity": round(float(grp[activity_col].std()), 4),
        }
        # Include technology breakdown if available
        tech_col = None
        for candidate in ["technology", "tech", "platform", "method"]:
            if candidate in grp.columns:
                tech_col = candidate
                break
        if tech_col:
            summary["technologies"] = sorted(grp[tech_col].unique().tolist())
            summary["n_technologies"] = len(summary["technologies"])

        # Include n_spots if available
        for spots_col in ["n_spots", "n_cells", "spot_count"]:
            if spots_col in grp.columns:
                summary["total_spots"] = int(grp[spots_col].sum())
                break

        # Top signatures for this tissue
        if sig_col:
            top = (
                grp.groupby(sig_col)[activity_col]
                .mean()
                .sort_values(ascending=False)
                .head(20)
            )
            summary["top_signatures"] = [
                {"signature": sig, "mean_activity": round(float(val), 4)}
                for sig, val in top.items()
            ]

        by_tissue[tissue_str] = summary

    output = {
        "records": records,
        "tissues": tissues,
        "signatures": signatures,
        "by_tissue": by_tissue,
        "metadata": {
            "n_records": len(records),
            "n_tissues": len(tissues),
            "n_signatures": len(signatures),
            "source": "spatial_corpus",
        },
    }

    save_json(output, OUTPUT_DIR / "spatial_tissue_activity.json",
              f"{len(records)} records, {len(tissues)} tissues, {len(signatures)} signatures")
    return True


def preprocess_spatial_technology_comparison() -> bool:
    """Preprocess spatial technology comparison into cross-tech reproducibility JSON.

    Input: spatial_technology_comparison.csv
           Expected columns: technology_a, technology_b, signature/cytokine, rho, pvalue, etc.
    Output: spatial_technology_comparison.json
            {comparisons: [...], technologies: [...], summary: {...}, metadata: {...}}
    """
    log("Processing spatial technology comparison...")
    csv_path = SPATIAL_DIR / "spatial_technology_comparison.csv"
    if not csv_path.exists():
        log(f"  SKIP: {csv_path} not found")
        return False

    df = pd.read_csv(csv_path)
    df = round_floats(df)
    log(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Detect technology columns
    tech_a_col = None
    tech_b_col = None
    for candidate_a, candidate_b in [
        ("technology_a", "technology_b"),
        ("tech_a", "tech_b"),
        ("platform_a", "platform_b"),
        ("method_1", "method_2"),
    ]:
        if candidate_a in df.columns and candidate_b in df.columns:
            tech_a_col = candidate_a
            tech_b_col = candidate_b
            break

    # If no paired tech columns, check for single technology column (long format)
    if tech_a_col is None:
        for candidate in ["technology", "tech", "platform"]:
            if candidate in df.columns:
                tech_a_col = candidate
                break

    comparisons = df.to_dict(orient="records")

    # Collect unique technologies
    technologies = set()
    if tech_a_col and tech_a_col in df.columns:
        technologies.update(df[tech_a_col].unique().tolist())
    if tech_b_col and tech_b_col in df.columns:
        technologies.update(df[tech_b_col].unique().tolist())
    technologies = sorted(technologies)

    # Summary: mean/median rho per technology pair
    summary = {}
    rho_col = None
    for candidate in ["rho", "correlation", "spearman_rho", "r", "pearson_r"]:
        if candidate in df.columns:
            rho_col = candidate
            break

    if rho_col and tech_a_col and tech_b_col:
        for (ta, tb), grp in df.groupby([tech_a_col, tech_b_col]):
            pair_key = f"{ta} vs {tb}"
            valid_rho = grp[rho_col].dropna()
            summary[pair_key] = {
                "n_comparisons": len(grp),
                "mean_rho": round(float(valid_rho.mean()), 4) if len(valid_rho) > 0 else None,
                "median_rho": round(float(valid_rho.median()), 4) if len(valid_rho) > 0 else None,
                "std_rho": round(float(valid_rho.std()), 4) if len(valid_rho) > 1 else None,
            }
    elif rho_col and tech_a_col:
        # Single technology column — summarize per technology
        for tech, grp in df.groupby(tech_a_col):
            valid_rho = grp[rho_col].dropna()
            summary[str(tech)] = {
                "n_comparisons": len(grp),
                "mean_rho": round(float(valid_rho.mean()), 4) if len(valid_rho) > 0 else None,
                "median_rho": round(float(valid_rho.median()), 4) if len(valid_rho) > 0 else None,
            }

    output = {
        "comparisons": comparisons,
        "technologies": technologies,
        "summary": summary,
        "metadata": {
            "n_comparisons": len(comparisons),
            "n_technologies": len(technologies),
            "rho_column": rho_col,
            "source": "spatial_corpus",
        },
    }

    save_json(output, OUTPUT_DIR / "spatial_technology_comparison.json",
              f"{len(comparisons)} comparisons, {len(technologies)} technologies")
    return True


def preprocess_spatial_gene_coverage() -> bool:
    """Preprocess spatial neighborhood activity into gene coverage JSON.

    Input: spatial_neighborhood_activity.csv
           Expected columns: technology, gene/signature, coverage/detected, n_spots, etc.
    Output: spatial_gene_coverage.json
            {by_technology: {...}, technologies: [...], genes: [...], metadata: {...}}
    """
    log("Processing spatial gene coverage...")
    csv_path = SPATIAL_DIR / "spatial_neighborhood_activity.csv"
    if not csv_path.exists():
        log(f"  SKIP: {csv_path} not found")
        return False

    df = pd.read_csv(csv_path)
    df = round_floats(df)
    log(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Detect columns
    tech_col = None
    for candidate in ["technology", "tech", "platform", "method"]:
        if candidate in df.columns:
            tech_col = candidate
            break

    gene_col = None
    for candidate in ["gene", "signature", "cytokine", "target", "protein"]:
        if candidate in df.columns:
            gene_col = candidate
            break

    # Detect activity/coverage column
    activity_col = None
    for candidate in ["activity", "mean_activity", "coverage", "detection_rate", "value"]:
        if candidate in df.columns:
            activity_col = candidate
            break

    technologies = sorted(df[tech_col].unique().tolist()) if tech_col and tech_col in df.columns else []
    genes = sorted(df[gene_col].unique().tolist()) if gene_col and gene_col in df.columns else []

    # Build per-technology gene coverage
    by_technology = {}
    if tech_col and gene_col:
        for tech, grp in df.groupby(tech_col):
            tech_str = str(tech)
            n_genes = grp[gene_col].nunique()
            tech_entry = {
                "n_genes": n_genes,
                "n_records": len(grp),
                "genes": sorted(grp[gene_col].unique().tolist()),
            }

            # Activity statistics if available
            if activity_col and activity_col in grp.columns:
                valid = grp[activity_col].dropna()
                tech_entry["mean_activity"] = round(float(valid.mean()), 4) if len(valid) > 0 else None
                tech_entry["std_activity"] = round(float(valid.std()), 4) if len(valid) > 1 else None

            # Spot count if available
            for spots_col in ["n_spots", "n_cells", "spot_count"]:
                if spots_col in grp.columns:
                    tech_entry["total_spots"] = int(grp[spots_col].sum())
                    break

            # Neighborhood info if available
            for nbr_col in ["neighborhood", "niche", "region", "zone"]:
                if nbr_col in grp.columns:
                    tech_entry["neighborhoods"] = sorted(grp[nbr_col].unique().tolist())
                    tech_entry["n_neighborhoods"] = len(tech_entry["neighborhoods"])
                    break

            by_technology[tech_str] = tech_entry

    # Build gene-level overlap across technologies
    gene_tech_overlap = {}
    if tech_col and gene_col:
        for gene, grp in df.groupby(gene_col):
            techs = sorted(grp[tech_col].unique().tolist())
            gene_tech_overlap[str(gene)] = {
                "technologies": techs,
                "n_technologies": len(techs),
            }

    # Coverage matrix: gene x technology presence
    coverage_matrix = {}
    if tech_col and gene_col:
        for tech in technologies:
            tech_genes = set(df[df[tech_col] == tech][gene_col].unique())
            coverage_matrix[str(tech)] = {
                "n_genes_detected": len(tech_genes),
                "n_total_genes": len(genes),
                "coverage_pct": round(100.0 * len(tech_genes) / max(len(genes), 1), 2),
            }

    output = {
        "by_technology": by_technology,
        "gene_overlap": gene_tech_overlap,
        "coverage_matrix": coverage_matrix,
        "technologies": technologies,
        "genes": genes,
        "metadata": {
            "n_technologies": len(technologies),
            "n_genes": len(genes),
            "n_records": len(df),
            "source": "spatial_corpus",
        },
    }

    save_json(output, OUTPUT_DIR / "spatial_gene_coverage.json",
              f"{len(technologies)} technologies, {len(genes)} genes")
    return True


def preprocess_spatial_dataset_catalog() -> bool:
    """Preprocess spatial dataset catalog from all spatial CSVs.

    Aggregates metadata across the three spatial input files to produce a
    dataset-level catalog with tissue, technology, and sample counts.

    Input: spatial_activity_by_tissue.csv (primary), spatial_technology_comparison.csv,
           spatial_neighborhood_activity.csv
    Output: spatial_dataset_catalog.json
            {datasets: [...], summary: {...}, metadata: {...}}
    """
    log("Processing spatial dataset catalog...")

    datasets = []
    tissues_seen = set()
    technologies_seen = set()

    # --- Source 1: activity by tissue ---
    activity_path = SPATIAL_DIR / "spatial_activity_by_tissue.csv"
    if activity_path.exists():
        df = pd.read_csv(activity_path)
        df = round_floats(df)
        log(f"  Activity by tissue: {len(df)} rows")

        # Detect columns
        tissue_col = None
        for candidate in ["tissue", "tissue_type", "organ", "region"]:
            if candidate in df.columns:
                tissue_col = candidate
                break

        tech_col = None
        for candidate in ["technology", "tech", "platform", "method"]:
            if candidate in df.columns:
                tech_col = candidate
                break

        dataset_col = None
        for candidate in ["dataset", "dataset_id", "study", "sample_id", "sample"]:
            if candidate in df.columns:
                dataset_col = candidate
                break

        sig_col = None
        for candidate in ["signature", "cytokine", "target", "protein"]:
            if candidate in df.columns:
                sig_col = candidate
                break

        # Build dataset entries from grouping
        group_cols = [c for c in [dataset_col, tissue_col, tech_col] if c is not None]
        if group_cols:
            for group_key, grp in df.groupby(group_cols[0] if len(group_cols) == 1 else group_cols):
                entry = {"source": "activity_by_tissue", "n_records": len(grp)}
                if dataset_col and dataset_col in grp.columns:
                    entry["dataset"] = str(grp[dataset_col].iloc[0]) if dataset_col else None
                if tissue_col and tissue_col in grp.columns:
                    tissue_val = str(grp[tissue_col].iloc[0])
                    entry["tissue"] = tissue_val
                    tissues_seen.add(tissue_val)
                if tech_col and tech_col in grp.columns:
                    tech_val = str(grp[tech_col].iloc[0])
                    entry["technology"] = tech_val
                    technologies_seen.add(tech_val)
                if sig_col and sig_col in grp.columns:
                    entry["n_signatures"] = grp[sig_col].nunique()
                for spots_col in ["n_spots", "n_cells", "spot_count"]:
                    if spots_col in grp.columns:
                        entry["n_spots"] = int(grp[spots_col].sum())
                        break
                datasets.append(entry)
        else:
            # No grouping columns — treat entire file as one dataset
            datasets.append({
                "source": "activity_by_tissue",
                "n_records": len(df),
            })
    else:
        log(f"  SKIP: {activity_path} not found")

    # --- Source 2: technology comparison ---
    tech_path = SPATIAL_DIR / "spatial_technology_comparison.csv"
    if tech_path.exists():
        df_tech = pd.read_csv(tech_path)
        log(f"  Technology comparison: {len(df_tech)} rows")
        for candidate in ["technology", "tech", "platform", "technology_a", "technology_b"]:
            if candidate in df_tech.columns:
                technologies_seen.update(df_tech[candidate].unique().tolist())
    else:
        log(f"  SKIP: {tech_path} not found")

    # --- Source 3: neighborhood activity ---
    nbr_path = SPATIAL_DIR / "spatial_neighborhood_activity.csv"
    if nbr_path.exists():
        df_nbr = pd.read_csv(nbr_path)
        log(f"  Neighborhood activity: {len(df_nbr)} rows")
        for candidate in ["technology", "tech", "platform"]:
            if candidate in df_nbr.columns:
                technologies_seen.update(df_nbr[candidate].unique().tolist())
                break
    else:
        log(f"  SKIP: {nbr_path} not found")

    if not datasets and not tissues_seen and not technologies_seen:
        log("  SKIP: No spatial data files found")
        return False

    summary = {
        "n_datasets": len(datasets),
        "n_tissues": len(tissues_seen),
        "n_technologies": len(technologies_seen),
        "tissues": sorted(tissues_seen),
        "technologies": sorted(technologies_seen),
    }

    output = {
        "datasets": datasets,
        "summary": summary,
        "metadata": {
            "n_datasets": len(datasets),
            "source": "spatial_corpus",
            "input_files": {
                "activity_by_tissue": str(activity_path),
                "technology_comparison": str(tech_path),
                "neighborhood_activity": str(nbr_path),
            },
        },
    }

    save_json(output, OUTPUT_DIR / "spatial_dataset_catalog.json",
              f"{len(datasets)} datasets, {len(tissues_seen)} tissues, {len(technologies_seen)} technologies")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess spatial transcriptomics (SpatialCorpus) data for web visualization"
    )
    parser.add_argument(
        "--tissue", action="store_true",
        help="Process tissue activity only",
    )
    parser.add_argument(
        "--technology", action="store_true",
        help="Process technology comparison only",
    )
    parser.add_argument(
        "--coverage", action="store_true",
        help="Process gene coverage only",
    )
    parser.add_argument(
        "--catalog", action="store_true",
        help="Process dataset catalog only",
    )
    args = parser.parse_args()

    # If no flags set, process all
    any_flag = args.tissue or args.technology or args.coverage or args.catalog
    do_tissue = args.tissue or not any_flag
    do_technology = args.technology or not any_flag
    do_coverage = args.coverage or not any_flag
    do_catalog = args.catalog or not any_flag

    t0 = time.time()
    log("=" * 60)
    log("Preprocessing Spatial Visualization Data")
    log("=" * 60)

    results = {}

    if do_tissue:
        results["spatial_tissue_activity"] = preprocess_spatial_tissue_activity()

    if do_technology:
        results["spatial_technology_comparison"] = preprocess_spatial_technology_comparison()

    if do_coverage:
        results["spatial_gene_coverage"] = preprocess_spatial_gene_coverage()

    if do_catalog:
        results["spatial_dataset_catalog"] = preprocess_spatial_dataset_catalog()

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
