#!/usr/bin/env python3
"""Preprocess cross-sample correlation CSVs into JSON for visualization.

Reads correlation CSVs from results/cross_sample_validation/correlations/
and outputs visualization/data/bulk_donor_correlations.json with:
  - summary: aggregated stats per atlas/level/signature
  - donor_level: per-target rows where celltype == "all" (donor-level)
  - celltype_level: aggregated per celltype + top 20 targets per celltype
"""

import json
import sqlite3
import sys
import zlib
from pathlib import Path
from typing import Dict, Optional

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORR_DIR = PROJECT_ROOT / "results" / "cross_sample_validation" / "correlations"
VALIDATION_DIR = Path("/data/parks34/projects/2secactpy/results/cross_sample_validation")
OUTPUT_PATH = PROJECT_ROOT / "visualization" / "data" / "bulk_donor_correlations.json"
BULK_RNASEQ_OUTPUT_PATH = PROJECT_ROOT / "visualization" / "data" / "bulk_rnaseq_validation.json"
MAPPING_PATH = PROJECT_ROOT / "cytoatlas-api" / "static" / "data" / "signature_gene_mapping.json"

# Atlas CSV files (key = atlas name in output JSON)
ATLAS_FILES = {
    "cima": "cima_correlations.csv",
    "inflammation_main": "inflammation_main_correlations.csv",
    "inflammation_val": "inflammation_val_correlations.csv",
    "inflammation_ext": "inflammation_ext_correlations.csv",
    "scatlas_normal": "scatlas_normal_correlations.csv",
    "scatlas_cancer": "scatlas_cancer_correlations.csv",
    "gtex": "gtex_correlations.csv",
    "tcga": "tcga_correlations.csv",
}


def round_val(v, decimals=4):
    """Round a value, handling NaN/None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), decimals)


def _fisher_z_ci(rho, n, z_crit=1.96):
    """Compute 95% CI for Spearman rho using Fisher z-transform."""
    if n < 6 or rho is None or not np.isfinite(rho) or abs(rho) >= 1.0:
        return None
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)
    return [round_val(np.tanh(z - z_crit * se)), round_val(np.tanh(z + z_crit * se))]


# H5AD file patterns for donor-level aggregation (celltype=="all")
# Format: (atlas_prefix, level_name, expr_suffix)
# expr_suffix defaults to "pseudobulk" for single-cell, "expression" for bulk
ATLAS_H5AD_PATTERNS = {
    "cima": ("cima", "donor_only", "pseudobulk"),
    "inflammation_main": ("inflammation_main", "donor_only", "pseudobulk"),
    "inflammation_val": ("inflammation_val", "donor_only", "pseudobulk"),
    "inflammation_ext": ("inflammation_ext", "donor_only", "pseudobulk"),
    "scatlas_normal": ("scatlas_normal", "donor_organ", "pseudobulk"),
    "scatlas_cancer": ("scatlas_cancer", "donor_organ", "pseudobulk"),
    "gtex": ("gtex", "donor_only", "expression"),
    "tcga": ("tcga", "donor_only", "expression"),
}

SIG_TYPES = ["cytosig", "lincytosig", "secact"]

# CytoSig signature names — always keep these in SecAct dropdown even if not top-N
CYTOSIG_NAMES = {
    "Activin A", "BDNF", "BMP2", "BMP4", "BMP6", "CD40L", "CXCL12", "EGF",
    "FGF2", "GCSF", "GDF11", "GMCSF", "HGF", "IFN1", "IFNG", "IFNL",
    "IL10", "IL12", "IL13", "IL15", "IL17A", "IL1A", "IL1B", "IL2", "IL21",
    "IL22", "IL27", "IL3", "IL36", "IL4", "IL6", "LIF", "LTA", "MCSF", "NO",
    "OSM", "TGFB1", "TGFB3", "TNFA", "TRAIL", "TWEAK", "VEGFA", "WNT3A",
}
# Additional key IDs to always keep in SecAct (user-specified + common cytokines)
SECACT_MUST_KEEP = CYTOSIG_NAMES | {
    "IFNL1", "IL6", "CCL2", "CCL5", "CXCL10", "CSF1", "CSF2", "CSF3",
    "TNF", "TGFB2", "IL18", "IL33", "IL23A", "IL17F",
}
SECACT_TOP_N = None  # Include all targets (was 100)


def filter_secact_targets(targets_data: dict, top_n: int | None = SECACT_TOP_N) -> dict:
    """Keep top N SecAct targets by |rho| plus must-keep IDs.

    If top_n is None, return all targets unfiltered.
    """
    if top_n is None or len(targets_data) <= top_n:
        return targets_data

    # Sort by |rho| descending
    sorted_targets = sorted(
        targets_data.items(),
        key=lambda x: abs(x[1]["rho"]) if x[1]["rho"] is not None else 0,
        reverse=True,
    )

    # Take top N
    kept = dict(sorted_targets[:top_n])

    # Add must-keep targets that aren't already included
    for target in targets_data:
        if target not in kept and target in SECACT_MUST_KEEP:
            kept[target] = targets_data[target]

    return kept

# Celltype-level H5AD file patterns: atlas -> [(level_name, file_prefix)]
# file_prefix is used as: {prefix}_pseudobulk.h5ad, {prefix}_{sig_type}.h5ad
CELLTYPE_H5AD_PATTERNS = {
    "cima": [
        ("donor_l1", "cima_donor_l1"),
        ("donor_l2", "cima_donor_l2"),
        ("donor_l3", "cima_donor_l3"),
        ("donor_l4", "cima_donor_l4"),
    ],
    "inflammation_main": [
        ("donor_l1", "inflammation_main_donor_l1"),
        ("donor_l2", "inflammation_main_donor_l2"),
    ],
    "inflammation_val": [
        ("donor_l1", "inflammation_val_donor_l1"),
        ("donor_l2", "inflammation_val_donor_l2"),
    ],
    "inflammation_ext": [
        ("donor_l1", "inflammation_ext_donor_l1"),
        ("donor_l2", "inflammation_ext_donor_l2"),
    ],
    "scatlas_normal": [
        ("donor_organ", "scatlas_normal_donor_organ"),
        ("donor_organ_celltype1", "scatlas_normal_donor_organ_celltype1"),
        ("donor_organ_celltype2", "scatlas_normal_donor_organ_celltype2"),
    ],
    "scatlas_cancer": [
        ("donor_organ", "scatlas_cancer_donor_organ"),
        ("donor_organ_celltype1", "scatlas_cancer_donor_organ_celltype1"),
        ("donor_organ_celltype2", "scatlas_cancer_donor_organ_celltype2"),
    ],
    "gtex": [
        ("by_tissue", "gtex_by_tissue", "expression"),
    ],
    "tcga": [
        ("by_cancer", "tcga_by_cancer", "expression"),
    ],
}

# Resampled H5AD patterns: atlas -> [(level_name, resamp_expr_basename)]
RESAMPLED_H5AD_PATTERNS = {
    "cima": [
        ("l1", "cima_pseudobulk_l1_resampled.h5ad"),
        ("l2", "cima_pseudobulk_l2_resampled.h5ad"),
    ],
    "inflammation_main": [
        ("l1", "inflammation_main_pseudobulk_l1_resampled.h5ad"),
        ("l2", "inflammation_main_pseudobulk_l2_resampled.h5ad"),
    ],
    "inflammation_val": [
        ("l1", "inflammation_val_pseudobulk_l1_resampled.h5ad"),
        ("l2", "inflammation_val_pseudobulk_l2_resampled.h5ad"),
    ],
    "inflammation_ext": [
        ("l1", "inflammation_ext_pseudobulk_l1_resampled.h5ad"),
        ("l2", "inflammation_ext_pseudobulk_l2_resampled.h5ad"),
    ],
    "scatlas_normal": [
        ("celltype", "scatlas_normal_pseudobulk_celltype_resampled.h5ad"),
        ("donor_celltype", "scatlas_normal_pseudobulk_donor_celltype_resampled.h5ad"),
    ],
    "scatlas_cancer": [
        ("celltype", "scatlas_cancer_pseudobulk_celltype_resampled.h5ad"),
        ("donor_celltype", "scatlas_cancer_pseudobulk_donor_celltype_resampled.h5ad"),
    ],
}
ATLAS_VALIDATION_DIR = Path("/data/parks34/projects/2secactpy/results/atlas_validation")


def load_target_to_gene_mapping() -> Dict[str, str]:
    """Load CytoSig target name -> HGNC gene symbol mapping."""
    if not MAPPING_PATH.exists():
        return {}
    with open(MAPPING_PATH) as f:
        data = json.load(f)
    mapping = {}
    for target_name, info in data.get("cytosig_mapping", {}).items():
        mapping[target_name] = info["hgnc_symbol"]
    return mapping


def resolve_gene_name(target: str, gene_set: set, cytosig_map: Dict[str, str]) -> Optional[str]:
    """Resolve a signature target name to a gene name in the expression matrix."""
    if target in gene_set:
        return target
    if target in cytosig_map:
        mapped = cytosig_map[target]
        if mapped in gene_set:
            return mapped
    if "__" in target:
        cytokine = target.split("__")[-1]
        if cytokine in gene_set:
            return cytokine
        if cytokine in cytosig_map:
            mapped = cytosig_map[cytokine]
            if mapped in gene_set:
                return mapped
    return None


def _load_matched_lookup(atlas_key: str) -> dict:
    """Load matched correlation rows from CSV for an atlas.

    Returns dict: (target, signature) -> {rho, pval, n, lincytosig_celltype, matched_atlas_celltypes}
    """
    csv_path = CORR_DIR / ATLAS_FILES.get(atlas_key, "")
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path, low_memory=False)
    matched_df = df[df["celltype"] == "matched"]
    lookup = {}
    for _, row in matched_df.iterrows():
        key = (row["target"], row["signature"])
        lookup[key] = {
            "matched_rho": round_val(row["spearman_rho"]),
            "matched_pval": round_val(row["spearman_pval"], 6),
            "matched_n": int(row["n_samples"]),
            "lincytosig_celltype": str(row.get("lincytosig_celltype", "")),
            "matched_atlas_celltypes": str(row.get("matched_atlas_celltypes", "")),
        }
    return lookup


def build_donor_scatter() -> tuple[dict, list[dict]]:
    """Build per-donor scatter data from H5AD files.

    For each atlas and signature type, reads donor-level pseudobulk (expression)
    and activity H5AD files, then extracts per-donor (expression, activity) pairs.

    Returns:
        - dict: {atlas: {sig_type: {target: {gene, rho, pval, n, points}}}}
        - list[dict]: flat metadata rows for Parquet output
    All targets included for each signature type.
    """
    cytosig_map = load_target_to_gene_mapping()
    result = {}
    scatter_meta = []

    for atlas_key, (atlas_prefix, level_name, expr_suffix) in ATLAS_H5AD_PATTERNS.items():
        atlas_dir = VALIDATION_DIR / atlas_key
        if not atlas_dir.exists():
            print(f"    WARNING: {atlas_dir} not found, skipping scatter for {atlas_key}")
            continue

        atlas_result = {}
        matched_lookup = _load_matched_lookup(atlas_key)

        for sig_type in SIG_TYPES:
            expr_path = atlas_dir / f"{atlas_prefix}_{level_name}_{expr_suffix}.h5ad"
            act_path = atlas_dir / f"{atlas_prefix}_{level_name}_{sig_type}.h5ad"

            if not expr_path.exists() or not act_path.exists():
                print(f"    WARNING: Missing H5AD for {atlas_key}/{sig_type}, skipping")
                continue

            expr_adata = ad.read_h5ad(expr_path)
            act_adata = ad.read_h5ad(act_path)

            # Build gene name lookup - handle both symbol var_names and ENSG var_names
            gene_set = set(expr_adata.var_names)
            symbol_to_varname = {}
            if "symbol" in expr_adata.var.columns:
                # Ensembl ID index with symbol column (inflammation atlases)
                for idx, sym in zip(expr_adata.var_names, expr_adata.var["symbol"]):
                    if pd.notna(sym):
                        symbol_to_varname[str(sym)] = idx
                gene_set = set(symbol_to_varname.keys()) | gene_set

            targets_data = {}

            # Pre-compute common obs once
            common_obs = expr_adata.obs_names.intersection(act_adata.obs_names)
            if len(common_obs) < 10:
                atlas_result[sig_type] = {}
                print(f"    donor_scatter/{atlas_key}/{sig_type}: 0 targets (insufficient common obs)")
                continue

            for target in act_adata.var_names:
                gene = resolve_gene_name(target, gene_set, cytosig_map)
                if gene is None:
                    continue

                # Resolve to actual var_name (may be ENSG ID)
                expr_var = symbol_to_varname.get(gene, gene)
                if expr_var not in set(expr_adata.var_names):
                    continue

                expr_vals = expr_adata[common_obs, expr_var].X.flatten()
                act_vals = act_adata[common_obs, target].X.flatten()

                # Handle sparse matrices
                if hasattr(expr_vals, "toarray"):
                    expr_vals = expr_vals.toarray().flatten()
                if hasattr(act_vals, "toarray"):
                    act_vals = act_vals.toarray().flatten()

                # Remove NaN pairs
                mask = np.isfinite(expr_vals) & np.isfinite(act_vals)
                expr_vals = expr_vals[mask].astype(np.float64)
                act_vals = act_vals[mask].astype(np.float64)

                if len(expr_vals) < 10:
                    continue

                rho, pval = stats.spearmanr(expr_vals, act_vals)

                # Z-score normalize expression across donors
                expr_std = np.std(expr_vals)
                if expr_std > 1e-10:
                    expr_vals = (expr_vals - np.mean(expr_vals)) / expr_std

                n_total = len(expr_vals)

                # All points included — use 2-decimal precision for compactness
                points = [[round_val(float(e), 2), round_val(float(a), 2)] for e, a in zip(expr_vals, act_vals)]

                rho_ci = _fisher_z_ci(rho, n_total)

                entry = {
                    "gene": gene,
                    "rho": round_val(rho),
                    "pval": float(f"{pval:.2e}") if pval is not None and np.isfinite(pval) else None,
                    "n": n_total,
                    "rho_ci": rho_ci,
                    "points": points,
                }

                # Add matched correlation info for LinCytoSig targets
                mkey = (target, sig_type)
                if mkey in matched_lookup:
                    entry.update(matched_lookup[mkey])

                targets_data[target] = entry

                # Collect metadata for Parquet output
                scatter_meta.append({
                    "atlas": atlas_key,
                    "sig_type": sig_type,
                    "target": target,
                    "gene": gene,
                    "rho": round_val(rho),
                    "pval": float(f"{pval:.2e}") if pval is not None and np.isfinite(pval) else None,
                    "n": n_total,
                    "rho_ci_lo": rho_ci[0] if rho_ci else None,
                    "rho_ci_hi": rho_ci[1] if rho_ci else None,
                })

            atlas_result[sig_type] = targets_data
            print(f"    donor_scatter/{atlas_key}/{sig_type}: {len(targets_data)} targets")

        result[atlas_key] = atlas_result

    return result, scatter_meta


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
    """Extract donor-level rows (celltype == 'all') per signature type.

    For LinCytoSig targets, also includes matched_rho from celltype-restricted
    correlations (celltype == 'matched' rows in the CSV).
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # Donor-level = rows where celltype is "all"
    donor_df = df[df["celltype"] == "all"].copy()

    # Build lookup for matched rows: (target, level, signature) -> matched row
    matched_df = df[df["celltype"] == "matched"]
    matched_lookup = {}
    for _, mrow in matched_df.iterrows():
        key = (mrow["target"], mrow["level"], mrow["signature"])
        matched_lookup[key] = mrow

    result = {}
    for sig_type in donor_df["signature"].unique():
        sig_df = donor_df[donor_df["signature"] == sig_type]
        targets = []
        for _, row in sig_df.iterrows():
            entry = {
                "target": row["target"],
                "gene": row["gene"],
                "level": row["level"],
                "rho": round_val(row["spearman_rho"]),
                "pval": round_val(row["spearman_pval"], 6),
                "n": int(row["n_samples"]),
                "mean_expr": round_val(row["mean_expr"]),
                "mean_activity": round_val(row["mean_activity"]),
                "significant": bool(row["spearman_pval"] < 0.05) if pd.notna(row["spearman_pval"]) else False,
            }

            # Add LinCytoSig matched correlation info if available
            lcs_ct = row.get("lincytosig_celltype", "")
            if pd.notna(lcs_ct) and lcs_ct != "":
                entry["lincytosig_celltype"] = str(lcs_ct)
                mkey = (row["target"], row["level"], sig_type)
                if mkey in matched_lookup:
                    mrow = matched_lookup[mkey]
                    entry["matched_rho"] = round_val(mrow["spearman_rho"])
                    entry["matched_pval"] = round_val(mrow["spearman_pval"], 6)
                    entry["matched_n"] = int(mrow["n_samples"])
                    matched_cts = mrow.get("matched_atlas_celltypes", "")
                    if pd.notna(matched_cts) and matched_cts != "":
                        entry["matched_atlas_celltypes"] = str(matched_cts)

            targets.append(entry)
        result[sig_type] = targets
    return result


def build_celltype_level(atlas_key: str, csv_path: Path) -> dict:
    """Build celltype-stratified data: aggregated per celltype + top 20 targets."""
    df = pd.read_csv(csv_path, low_memory=False)

    # Celltype-level = rows where celltype is a real cell type name
    # Exclude special values: 'all', 'matched', 'unmappable', and NaN
    ct_df = df[
        df["celltype"].notna() & ~df["celltype"].isin(["all", "matched", "unmappable"])
    ].copy()

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


def build_celltype_scatter() -> tuple[dict, list[dict]]:
    """Build per-donor×celltype scatter data from celltype-level H5AD files.

    Returns:
        - dict: {atlas: {level: {sig_type: {target: {gene, rho, pval, n, celltypes, points}}}}}
        - list[dict]: flat metadata rows for Parquet output
    Each target's points include a celltype label for filtering in the UI.
    """
    cytosig_map = load_target_to_gene_mapping()
    result = {}
    scatter_meta = []

    for atlas_key, levels in CELLTYPE_H5AD_PATTERNS.items():
        atlas_dir = VALIDATION_DIR / atlas_key
        if not atlas_dir.exists():
            print(f"    WARNING: {atlas_dir} not found, skipping celltype scatter for {atlas_key}")
            continue

        atlas_result = {}

        for level_entry in levels:
            if len(level_entry) == 3:
                level_name, file_prefix, ct_expr_suffix = level_entry
            else:
                level_name, file_prefix = level_entry
                ct_expr_suffix = "pseudobulk"
            level_result = {}

            for sig_type in SIG_TYPES:
                expr_path = atlas_dir / f"{file_prefix}_{ct_expr_suffix}.h5ad"
                act_path = atlas_dir / f"{file_prefix}_{sig_type}.h5ad"

                if not expr_path.exists() or not act_path.exists():
                    print(f"    WARNING: Missing H5AD for {atlas_key}/{level_name}/{sig_type}, skipping")
                    continue

                expr_adata = ad.read_h5ad(expr_path)
                act_adata = ad.read_h5ad(act_path)

                # Build gene lookup
                gene_set = set(expr_adata.var_names)
                symbol_to_varname = {}
                if "symbol" in expr_adata.var.columns:
                    for idx, sym in zip(expr_adata.var_names, expr_adata.var["symbol"]):
                        if pd.notna(sym):
                            symbol_to_varname[str(sym)] = idx
                    gene_set = set(symbol_to_varname.keys()) | gene_set

                # Find celltype column
                ct_col = None
                for col in ["cell_type_l1", "cell_type_l2", "cell_type_l3", "cell_type_l4",
                            "cellType1", "cellType2", "celltype", "cell_type",
                            "Level1", "Level2", "tissue",
                            "tissue_type", "cancer_type"]:
                    if col in expr_adata.obs.columns:
                        ct_col = col
                        break

                common_obs = expr_adata.obs_names.intersection(act_adata.obs_names)
                if len(common_obs) < 10:
                    level_result[sig_type] = {}
                    continue

                celltypes_available = sorted(
                    expr_adata.obs.loc[common_obs, ct_col].dropna().unique().tolist()
                ) if ct_col else []

                targets_data = {}

                for target in act_adata.var_names:
                    gene = resolve_gene_name(target, gene_set, cytosig_map)
                    if gene is None:
                        continue
                    expr_var = symbol_to_varname.get(gene, gene)
                    if expr_var not in set(expr_adata.var_names):
                        continue

                    expr_vals = expr_adata[common_obs, expr_var].X.flatten()
                    act_vals = act_adata[common_obs, target].X.flatten()
                    if hasattr(expr_vals, "toarray"):
                        expr_vals = expr_vals.toarray().flatten()
                    if hasattr(act_vals, "toarray"):
                        act_vals = act_vals.toarray().flatten()

                    ct_labels = expr_adata.obs.loc[common_obs, ct_col].values if ct_col else None

                    mask = np.isfinite(expr_vals) & np.isfinite(act_vals)
                    expr_vals = expr_vals[mask].astype(np.float64)
                    act_vals = act_vals[mask].astype(np.float64)
                    if ct_labels is not None:
                        ct_labels = ct_labels[mask]

                    if len(expr_vals) < 10:
                        continue

                    rho, pval = stats.spearmanr(expr_vals, act_vals)
                    n_total = len(expr_vals)

                    # Z-score normalize expression
                    expr_std = np.std(expr_vals)
                    if expr_std > 1e-10:
                        expr_vals = (expr_vals - np.mean(expr_vals)) / expr_std

                    # All points included — integer-encode group labels for compactness
                    ct_to_idx = {ct: i for i, ct in enumerate(celltypes_available)}
                    points = []
                    for i in range(len(expr_vals)):
                        pt = [round_val(float(expr_vals[i]), 2), round_val(float(act_vals[i]), 2)]
                        if ct_labels is not None:
                            pt.append(ct_to_idx.get(str(ct_labels[i]), -1))
                        points.append(pt)

                    rho_ci = _fisher_z_ci(rho, n_total)

                    targets_data[target] = {
                        "gene": gene,
                        "rho": round_val(rho),
                        "pval": float(f"{pval:.2e}") if pval is not None and np.isfinite(pval) else None,
                        "n": n_total,
                        "rho_ci": rho_ci,
                        "celltypes": celltypes_available,
                        "groups": celltypes_available,
                        "points": points,
                    }

                    # Collect metadata for Parquet output
                    scatter_meta.append({
                        "atlas": atlas_key,
                        "level": level_name,
                        "sig_type": sig_type,
                        "target": target,
                        "gene": gene,
                        "rho": round_val(rho),
                        "pval": float(f"{pval:.2e}") if pval is not None and np.isfinite(pval) else None,
                        "n": n_total,
                        "rho_ci_lo": rho_ci[0] if rho_ci else None,
                        "rho_ci_hi": rho_ci[1] if rho_ci else None,
                    })

                level_result[sig_type] = targets_data
                print(f"    celltype_scatter/{atlas_key}/{level_name}/{sig_type}: {len(targets_data)} targets")

                del expr_adata, act_adata
                import gc; gc.collect()

            atlas_result[level_name] = level_result

        result[atlas_key] = atlas_result

    return result, scatter_meta


def build_resampled_scatter() -> dict:
    """Build scatter data from resampled pseudobulk activity inference results.

    Reads resampled expression H5AD + corresponding resampled activity H5AD
    (generated by scripts/16_resampled_validation.py).

    Returns dict: {atlas: {level: {sig_type: {target: {gene, rho, pval, n, rho_ci, points}}}}}
    """
    cytosig_map = load_target_to_gene_mapping()
    result = {}

    for atlas_key, levels in RESAMPLED_H5AD_PATTERNS.items():
        atlas_dir = ATLAS_VALIDATION_DIR / atlas_key / "pseudobulk"
        activity_dir = VALIDATION_DIR / atlas_key

        if not atlas_dir.exists():
            print(f"    WARNING: {atlas_dir} not found, skipping resampled scatter for {atlas_key}")
            continue

        atlas_result = {}

        for level_name, expr_filename in levels:
            level_result = {}
            expr_path = atlas_dir / expr_filename

            if not expr_path.exists():
                print(f"    WARNING: {expr_path} not found, skipping")
                continue

            for sig_type in SIG_TYPES:
                act_path = activity_dir / f"{atlas_key}_resampled_{level_name}_{sig_type}.h5ad"
                if not act_path.exists():
                    print(f"    WARNING: {act_path.name} not found, skipping (run 16_resampled_validation.py first)")
                    continue

                expr_adata = ad.read_h5ad(expr_path)
                act_adata = ad.read_h5ad(act_path)

                gene_set = set(expr_adata.var_names)
                symbol_to_varname = {}
                if "symbol" in expr_adata.var.columns:
                    for idx, sym in zip(expr_adata.var_names, expr_adata.var["symbol"]):
                        if pd.notna(sym):
                            symbol_to_varname[str(sym)] = idx
                    gene_set = set(symbol_to_varname.keys()) | gene_set

                # Find celltype and bootstrap columns
                ct_col = None
                for col in ["cell_type", "celltype"]:
                    if col in expr_adata.obs.columns:
                        ct_col = col
                        break
                boot_col = None
                for col in ["bootstrap_idx", "bootstrap"]:
                    if col in expr_adata.obs.columns:
                        boot_col = col
                        break

                common_obs = expr_adata.obs_names.intersection(act_adata.obs_names)
                if len(common_obs) < 10:
                    level_result[sig_type] = {}
                    continue

                celltypes_available = sorted(
                    expr_adata.obs.loc[common_obs, ct_col].dropna().unique().tolist()
                ) if ct_col else []

                targets_data = {}

                for target in act_adata.var_names:
                    gene = resolve_gene_name(target, gene_set, cytosig_map)
                    if gene is None:
                        continue
                    expr_var = symbol_to_varname.get(gene, gene)
                    if expr_var not in set(expr_adata.var_names):
                        continue

                    expr_vals = expr_adata[common_obs, expr_var].X.flatten()
                    act_vals = act_adata[common_obs, target].X.flatten()
                    if hasattr(expr_vals, "toarray"):
                        expr_vals = expr_vals.toarray().flatten()
                    if hasattr(act_vals, "toarray"):
                        act_vals = act_vals.toarray().flatten()

                    ct_labels = expr_adata.obs.loc[common_obs, ct_col].values if ct_col else None
                    boot_labels = expr_adata.obs.loc[common_obs, boot_col].values if boot_col else None

                    mask = np.isfinite(expr_vals) & np.isfinite(act_vals)
                    expr_vals = expr_vals[mask].astype(np.float64)
                    act_vals = act_vals[mask].astype(np.float64)
                    if ct_labels is not None:
                        ct_labels = ct_labels[mask]
                    if boot_labels is not None:
                        boot_labels = boot_labels[mask]

                    if len(expr_vals) < 10:
                        continue

                    rho, pval = stats.spearmanr(expr_vals, act_vals)
                    n_total = len(expr_vals)

                    # Bootstrap CI for rho
                    bootstrap_rhos = []
                    if boot_labels is not None:
                        for bi in np.unique(boot_labels):
                            bm = boot_labels == bi
                            if bm.sum() >= 5:
                                be = expr_vals[bm]
                                ba = act_vals[bm]
                                if np.std(be) > 1e-10 and np.std(ba) > 1e-10:
                                    br, _ = stats.spearmanr(be, ba)
                                    if np.isfinite(br):
                                        bootstrap_rhos.append(br)

                    rho_ci = None
                    if len(bootstrap_rhos) >= 10:
                        rho_ci = [
                            round_val(np.percentile(bootstrap_rhos, 2.5)),
                            round_val(np.percentile(bootstrap_rhos, 97.5)),
                        ]

                    # Z-score normalize expression
                    expr_std = np.std(expr_vals)
                    if expr_std > 1e-10:
                        expr_vals = (expr_vals - np.mean(expr_vals)) / expr_std

                    # All points included — integer-encode group labels for compactness
                    ct_to_idx = {ct: i for i, ct in enumerate(celltypes_available)}
                    points = []
                    for i in range(len(expr_vals)):
                        pt = [round_val(float(expr_vals[i]), 2), round_val(float(act_vals[i]), 2)]
                        if ct_labels is not None:
                            pt.append(ct_to_idx.get(str(ct_labels[i]), -1))
                        points.append(pt)

                    targets_data[target] = {
                        "gene": gene,
                        "rho": round_val(rho),
                        "pval": float(f"{pval:.2e}") if pval is not None and np.isfinite(pval) else None,
                        "n": n_total,
                        "rho_ci": rho_ci,
                        "celltypes": celltypes_available,
                        "groups": celltypes_available,
                        "points": points,
                    }

                level_result[sig_type] = targets_data
                print(f"    resampled_scatter/{atlas_key}/{level_name}/{sig_type}: {len(targets_data)} targets")

                del expr_adata, act_adata
                import gc; gc.collect()

            atlas_result[level_name] = level_result

        result[atlas_key] = atlas_result

    return result


def write_split_files(output: dict, bulk_rnaseq: dict | None = None):
    """Write per-atlas-sigtype split files for efficient API consumption.

    Splits the monolithic bulk_donor_correlations.json into:
      - validation/donor_scatter/{atlas}_{sigtype}.json
      - validation/celltype_scatter/{atlas}_{level}_{sigtype}.json
      - validation/bulk_donor_meta.json (summary + donor_level + celltype_level)
      - validation/bulk_rnaseq/{dataset}.json (GTEx/TCGA split)
    """
    split_dir = OUTPUT_PATH.parent / "validation"
    split_dir.mkdir(parents=True, exist_ok=True)

    # --- Donor scatter: validation/donor_scatter/{atlas}_{sigtype}.json ---
    donor_dir = split_dir / "donor_scatter"
    donor_dir.mkdir(parents=True, exist_ok=True)
    n_donor = 0
    for atlas, sigtypes in output.get("donor_scatter", {}).items():
        for sigtype, targets in sigtypes.items():
            if not targets:
                continue
            path = donor_dir / f"{atlas}_{sigtype}.json"
            with open(path, "w") as f:
                json.dump(targets, f, separators=(",", ":"))
            size_kb = path.stat().st_size / 1024
            print(f"    Split: {path.name} ({size_kb:.0f} KB, {len(targets)} targets)")
            n_donor += 1

    # --- Celltype scatter: validation/celltype_scatter/{atlas}_{level}_{sigtype}.json ---
    ct_dir = split_dir / "celltype_scatter"
    ct_dir.mkdir(parents=True, exist_ok=True)
    n_ct = 0
    for atlas, levels in output.get("celltype_scatter", {}).items():
        for level, sigtypes in levels.items():
            for sigtype, targets in sigtypes.items():
                if not targets:
                    continue
                path = ct_dir / f"{atlas}_{level}_{sigtype}.json"
                with open(path, "w") as f:
                    json.dump(targets, f, separators=(",", ":"))
                size_kb = path.stat().st_size / 1024
                print(f"    Split: {path.name} ({size_kb:.0f} KB, {len(targets)} targets)")
                n_ct += 1

    # --- Metadata without scatter points ---
    meta = {
        "summary": output.get("summary", []),
        "donor_level": output.get("donor_level", {}),
        "celltype_level": output.get("celltype_level", {}),
    }
    meta_path = split_dir / "bulk_donor_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    meta_kb = meta_path.stat().st_size / 1024
    print(f"    Split: bulk_donor_meta.json ({meta_kb:.0f} KB)")

    # --- Bulk RNA-seq split: validation/bulk_rnaseq/{dataset}.json ---
    if bulk_rnaseq:
        bulk_dir = split_dir / "bulk_rnaseq"
        bulk_dir.mkdir(parents=True, exist_ok=True)
        for dataset, data in bulk_rnaseq.items():
            path = bulk_dir / f"{dataset}.json"
            with open(path, "w") as f:
                json.dump(data, f, separators=(",", ":"))
            size_kb = path.stat().st_size / 1024
            print(f"    Split: bulk_rnaseq/{dataset}.json ({size_kb:.0f} KB)")

    print(f"\n  Split files written: {n_donor} donor, {n_ct} celltype, 1 meta"
          f"{f', {len(bulk_rnaseq)} bulk_rnaseq' if bulk_rnaseq else ''}")


SQLITE_DB_PATH = PROJECT_ROOT / "visualization" / "data" / "validation_scatter.db"

BULK_ATLASES = {"gtex", "tcga"}


def build_bulk_rnaseq_json() -> dict:
    """Build separate bulk_rnaseq_validation.json for GTEx/TCGA tab.

    Returns dict with structure:
    {
        "gtex": {
            "n_samples": int, "tissue_types": [...],
            "summary": {...},
            "donor_level": {sig_type: [targets]},
            "donor_scatter": {sig_type: {target: {gene, rho, pval, n, points}}}
        },
        "tcga": { ... }
    }
    """
    cytosig_map = load_target_to_gene_mapping()
    result = {}

    for atlas_key in ["gtex", "tcga"]:
        atlas_dir = VALIDATION_DIR / atlas_key
        csv_path = CORR_DIR / ATLAS_FILES.get(atlas_key, "")

        atlas_data = {}

        # Sample count and tissue/cancer types from pseudobulk H5AD
        pb_path = atlas_dir / f"{atlas_key}_donor_only_expression.h5ad"
        if pb_path.exists():
            pb = ad.read_h5ad(pb_path)
            atlas_data["n_samples"] = pb.shape[0]
            if atlas_key == "gtex" and "tissue_type" in pb.obs.columns:
                atlas_data["tissue_types"] = sorted(pb.obs["tissue_type"].dropna().unique().tolist())
            elif atlas_key == "tcga":
                for col in ["cancer_type", "cancer_site"]:
                    if col in pb.obs.columns:
                        atlas_data["cancer_types"] = sorted(pb.obs[col].dropna().unique().tolist())
                        break
            del pb
        else:
            atlas_data["n_samples"] = 0

        # Summary from correlation CSV
        if csv_path.exists():
            df = pd.read_csv(csv_path, low_memory=False)
            overall = df[df["celltype"] == "all"]
            sig_summary = {}
            for sig_type in overall["signature"].unique():
                sig_df = overall[overall["signature"] == sig_type]
                rho_vals = sig_df["spearman_rho"].dropna()
                pvals = sig_df["spearman_pval"].dropna()
                sig_summary[sig_type] = {
                    "n_targets": int(len(sig_df)),
                    "n_significant": int((pvals < 0.05).sum()),
                    "median_rho": round_val(rho_vals.median()),
                    "mean_rho": round_val(rho_vals.mean()),
                    "pct_positive": round_val(100.0 * (rho_vals > 0).sum() / len(rho_vals), 2) if len(rho_vals) > 0 else 0,
                }
            atlas_data["summary"] = sig_summary

            # Donor-level targets
            atlas_data["donor_level"] = build_donor_level(atlas_key, csv_path)
        else:
            atlas_data["summary"] = {}
            atlas_data["donor_level"] = {}

        # Donor scatter: built as fallback when stratified_scatter is empty (e.g. TCGA).
        # For GTEx, stratified_scatter has tissue labels and the JS prefers it.
        stratified_levels = CELLTYPE_H5AD_PATTERNS.get(atlas_key, [])
        atlas_data["donor_scatter"] = {}  # placeholder, filled below if needed

        # Build stratified scatter (by_tissue for GTEx, by_cancer for TCGA)
        # All points included (no subsampling) with integer-encoded group labels
        stratified_scatter = {}

        for level_entry in stratified_levels:
            if len(level_entry) == 3:
                level_name, file_prefix, strat_expr_suffix = level_entry
            else:
                level_name, file_prefix = level_entry
                strat_expr_suffix = "pseudobulk"
            level_scatter = {}

            for sig_type in SIG_TYPES:
                expr_path = atlas_dir / f"{file_prefix}_{strat_expr_suffix}.h5ad"
                act_path = atlas_dir / f"{file_prefix}_{sig_type}.h5ad"

                if not expr_path.exists() or not act_path.exists():
                    continue

                expr_adata = ad.read_h5ad(expr_path)
                act_adata = ad.read_h5ad(act_path)

                gene_set_s = set(expr_adata.var_names)
                sym_to_var = {}
                if "symbol" in expr_adata.var.columns:
                    for idx, sym in zip(expr_adata.var_names, expr_adata.var["symbol"]):
                        if pd.notna(sym):
                            sym_to_var[str(sym)] = idx
                    gene_set_s = set(sym_to_var.keys()) | gene_set_s

                # Find group column
                grp_col = None
                for col in ["tissue_type", "cancer_type"]:
                    if col in expr_adata.obs.columns:
                        grp_col = col
                        break

                common_obs_s = expr_adata.obs_names.intersection(act_adata.obs_names)
                if len(common_obs_s) < 10:
                    level_scatter[sig_type] = {}
                    continue

                groups_available = sorted(
                    expr_adata.obs.loc[common_obs_s, grp_col].dropna().unique().tolist()
                ) if grp_col else []

                # Build group label -> integer index mapping for compact encoding
                grp_to_idx = {g: i for i, g in enumerate(groups_available)}

                targets_data_s = {}

                for target in act_adata.var_names:
                    gene = resolve_gene_name(target, gene_set_s, cytosig_map)
                    if gene is None:
                        continue
                    expr_var_s = sym_to_var.get(gene, gene)
                    if expr_var_s not in set(expr_adata.var_names):
                        continue

                    ev = expr_adata[common_obs_s, expr_var_s].X.flatten()
                    av = act_adata[common_obs_s, target].X.flatten()
                    if hasattr(ev, "toarray"):
                        ev = ev.toarray().flatten()
                    if hasattr(av, "toarray"):
                        av = av.toarray().flatten()

                    grp_labels = expr_adata.obs.loc[common_obs_s, grp_col].values if grp_col else None

                    m = np.isfinite(ev) & np.isfinite(av)
                    ev = ev[m].astype(np.float64)
                    av = av[m].astype(np.float64)
                    if grp_labels is not None:
                        grp_labels = grp_labels[m]

                    if len(ev) < 10:
                        continue

                    rho_s, pval_s = stats.spearmanr(ev, av)
                    n_total_s = len(ev)

                    # Z-score normalize expression
                    ev_std = np.std(ev)
                    if ev_std > 1e-10:
                        ev = (ev - np.mean(ev)) / ev_std

                    # All points included — use 2-decimal precision and integer group labels
                    points_s = []
                    for i in range(len(ev)):
                        pt = [round_val(float(ev[i]), 2), round_val(float(av[i]), 2)]
                        if grp_labels is not None:
                            pt.append(grp_to_idx.get(str(grp_labels[i]), -1))
                        points_s.append(pt)

                    targets_data_s[target] = {
                        "gene": gene,
                        "rho": round_val(rho_s),
                        "pval": float(f"{pval_s:.2e}") if pval_s is not None and np.isfinite(pval_s) else None,
                        "n": n_total_s,
                        "groups": groups_available,
                        "points": points_s,
                    }

                # Filter SecAct targets (no-op when SECACT_TOP_N is None)
                if sig_type == "secact":
                    targets_data_s = filter_secact_targets(targets_data_s)

                level_scatter[sig_type] = targets_data_s
                print(f"    bulk_stratified/{atlas_key}/{level_name}/{sig_type}: {len(targets_data_s)} targets")

                del expr_adata, act_adata
                import gc; gc.collect()

            stratified_scatter[level_name] = level_scatter

        # Check if stratified_scatter produced any actual target data
        strat_has_data = any(
            len(sigs.get(st, {})) > 0
            for sigs in stratified_scatter.values()
            for st in SIG_TYPES
        )

        if strat_has_data:
            atlas_data["stratified_scatter"] = stratified_scatter
            print(f"    Using stratified_scatter for {atlas_key} (skipping donor_scatter)")
        else:
            # Stratified scatter empty — build donor_scatter as fallback (all points)
            print(f"    Stratified scatter empty for {atlas_key}, building donor_scatter fallback...")
            atlas_scatter = {}
            atlas_prefix, level_name, expr_suffix = ATLAS_H5AD_PATTERNS.get(
                atlas_key, (atlas_key, "donor_only", "expression")
            )
            for sig_type in SIG_TYPES:
                expr_path = atlas_dir / f"{atlas_prefix}_{level_name}_{expr_suffix}.h5ad"
                act_path = atlas_dir / f"{atlas_prefix}_{level_name}_{sig_type}.h5ad"
                if not expr_path.exists() or not act_path.exists():
                    continue
                expr_adata = ad.read_h5ad(expr_path)
                act_adata = ad.read_h5ad(act_path)
                gene_set_fb = set(expr_adata.var_names)
                sym_to_var_fb = {}
                if "symbol" in expr_adata.var.columns:
                    for idx, sym in zip(expr_adata.var_names, expr_adata.var["symbol"]):
                        if pd.notna(sym):
                            sym_to_var_fb[str(sym)] = idx
                    gene_set_fb = set(sym_to_var_fb.keys()) | gene_set_fb
                targets_data_fb = {}
                common_obs_fb = expr_adata.obs_names.intersection(act_adata.obs_names)
                if len(common_obs_fb) < 10:
                    atlas_scatter[sig_type] = {}
                    continue
                for target in act_adata.var_names:
                    gene = resolve_gene_name(target, gene_set_fb, cytosig_map)
                    if gene is None:
                        continue
                    expr_var_fb = sym_to_var_fb.get(gene, gene)
                    if expr_var_fb not in set(expr_adata.var_names):
                        continue
                    ev_fb = expr_adata[common_obs_fb, expr_var_fb].X.flatten()
                    av_fb = act_adata[common_obs_fb, target].X.flatten()
                    if hasattr(ev_fb, "toarray"):
                        ev_fb = ev_fb.toarray().flatten()
                    if hasattr(av_fb, "toarray"):
                        av_fb = av_fb.toarray().flatten()
                    m_fb = np.isfinite(ev_fb) & np.isfinite(av_fb)
                    ev_fb = ev_fb[m_fb].astype(np.float64)
                    av_fb = av_fb[m_fb].astype(np.float64)
                    if len(ev_fb) < 10:
                        continue
                    rho_fb, pval_fb = stats.spearmanr(ev_fb, av_fb)
                    n_fb = len(ev_fb)
                    ev_std = np.std(ev_fb)
                    if ev_std > 1e-10:
                        ev_fb = (ev_fb - np.mean(ev_fb)) / ev_std
                    points_fb = [[round_val(float(e), 2), round_val(float(a), 2)] for e, a in zip(ev_fb, av_fb)]
                    targets_data_fb[target] = {
                        "gene": gene,
                        "rho": round_val(rho_fb),
                        "pval": float(f"{pval_fb:.2e}") if pval_fb is not None and np.isfinite(pval_fb) else None,
                        "n": n_fb,
                        "points": points_fb,
                    }
                if sig_type == "secact":
                    targets_data_fb = filter_secact_targets(targets_data_fb)
                atlas_scatter[sig_type] = targets_data_fb
                print(f"    bulk_scatter/{atlas_key}/{sig_type}: {len(targets_data_fb)} targets, {n_fb} pts/target")
            atlas_data["donor_scatter"] = atlas_scatter

        result[atlas_key] = atlas_data

    return result


def write_sqlite_db(
    donor_scatter: dict,
    celltype_scatter: dict,
    bulk_rnaseq: dict | None = None,
):
    """Write all scatter data into a single SQLite DB for fast indexed lookups.

    Creates validation_scatter.db with:
      - scatter_targets: metadata per (source, atlas, level, sigtype, target)
      - scatter_points: zlib-compressed JSON point arrays keyed by target_id

    Args:
        donor_scatter: {atlas: {sig_type: {target: {gene, rho, pval, n, points, ...}}}}
        celltype_scatter: {atlas: {level: {sig_type: {target: {gene, rho, pval, n, points, ...}}}}}
        bulk_rnaseq: {dataset: {stratified_scatter: {...}, donor_scatter: {...}, ...}}
    """
    SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SQLITE_DB_PATH.exists():
        SQLITE_DB_PATH.unlink()

    conn = sqlite3.connect(str(SQLITE_DB_PATH))
    cur = conn.cursor()

    # Create tables without indexes (faster bulk insert)
    cur.executescript("""
        CREATE TABLE scatter_targets (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source      TEXT NOT NULL,
            atlas       TEXT NOT NULL,
            level       TEXT NOT NULL DEFAULT '',
            sigtype     TEXT NOT NULL,
            target      TEXT NOT NULL,
            gene        TEXT,
            rho         REAL,
            pval        REAL,
            n           INTEGER,
            rho_ci_lo   REAL,
            rho_ci_hi   REAL,
            groups_json TEXT,
            extra_json  TEXT
        );

        CREATE TABLE scatter_points (
            target_id   INTEGER PRIMARY KEY REFERENCES scatter_targets(id),
            points_blob BLOB NOT NULL
        );
    """)

    total_targets = 0
    total_points = 0

    def _insert_target(source, atlas, level, sigtype, target_name, info):
        nonlocal total_targets, total_points

        points = info.get("points", [])
        rho_ci = info.get("rho_ci")
        rho_ci_lo = rho_ci[0] if rho_ci and len(rho_ci) == 2 else None
        rho_ci_hi = rho_ci[1] if rho_ci and len(rho_ci) == 2 else None

        groups = info.get("groups") or info.get("celltypes")
        groups_json = json.dumps(groups, separators=(",", ":")) if groups else None

        # Extra fields: matched_rho, matched_pval, matched_n, etc.
        extra = {}
        for key in ("matched_rho", "matched_pval", "matched_n",
                     "lincytosig_celltype", "matched_atlas_celltypes"):
            if key in info:
                extra[key] = info[key]
        extra_json = json.dumps(extra, separators=(",", ":")) if extra else None

        cur.execute(
            "INSERT INTO scatter_targets "
            "(source, atlas, level, sigtype, target, gene, rho, pval, n, "
            " rho_ci_lo, rho_ci_hi, groups_json, extra_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (source, atlas, level, sigtype, target_name,
             info.get("gene"), info.get("rho"), info.get("pval"), info.get("n"),
             rho_ci_lo, rho_ci_hi, groups_json, extra_json),
        )
        target_id = cur.lastrowid

        blob = zlib.compress(
            json.dumps(points, separators=(",", ":")).encode(), level=6
        )
        cur.execute(
            "INSERT INTO scatter_points (target_id, points_blob) VALUES (?,?)",
            (target_id, blob),
        )

        total_targets += 1
        total_points += len(points)

    # --- Donor scatter ---
    print("  SQLite: inserting donor scatter ...")
    for atlas, sigtypes in donor_scatter.items():
        for sigtype, targets in sigtypes.items():
            for target_name, info in targets.items():
                _insert_target("donor", atlas, "", sigtype, target_name, info)
    conn.commit()
    print(f"    donor: {total_targets} targets, {total_points:,} points")

    donor_count = total_targets

    # --- Celltype scatter ---
    print("  SQLite: inserting celltype scatter ...")
    for atlas, levels in celltype_scatter.items():
        for level, sigtypes in levels.items():
            for sigtype, targets in sigtypes.items():
                for target_name, info in targets.items():
                    _insert_target("celltype", atlas, level, sigtype, target_name, info)
    conn.commit()
    ct_count = total_targets - donor_count
    print(f"    celltype: {ct_count} targets, {total_points:,} total points")

    # --- Bulk RNA-seq scatter ---
    if bulk_rnaseq:
        print("  SQLite: inserting bulk_rnaseq scatter ...")
        pre_bulk = total_targets
        for dataset, data in bulk_rnaseq.items():
            # Stratified scatter (GTEx by_tissue, TCGA by_cancer)
            for level_name, level_data in data.get("stratified_scatter", {}).items():
                for sigtype, targets in level_data.items():
                    for target_name, info in targets.items():
                        _insert_target(
                            "bulk_rnaseq", dataset, level_name, sigtype,
                            target_name, info,
                        )
            # Donor scatter fallback
            for sigtype, targets in data.get("donor_scatter", {}).items():
                if not targets:
                    continue
                _insert_target_source = "bulk_rnaseq"
                for target_name, info in targets.items():
                    _insert_target(
                        "bulk_rnaseq", dataset, "", sigtype,
                        target_name, info,
                    )
        conn.commit()
        bulk_count = total_targets - pre_bulk
        print(f"    bulk_rnaseq: {bulk_count} targets")

    # --- Create indexes AFTER inserts (much faster) ---
    print("  SQLite: creating indexes ...")
    cur.executescript("""
        CREATE UNIQUE INDEX idx_targets_unique
            ON scatter_targets(source, atlas, level, sigtype, target);
        CREATE INDEX idx_targets_lookup
            ON scatter_targets(source, atlas, level, sigtype);
    """)

    # Optimize
    print("  SQLite: ANALYZE + VACUUM ...")
    cur.execute("ANALYZE")
    conn.execute("VACUUM")
    conn.close()

    size_mb = SQLITE_DB_PATH.stat().st_size / (1024 * 1024)
    print(f"\n  SQLite DB: {SQLITE_DB_PATH}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Total: {total_targets} targets, {total_points:,} points")

    # Spot-check: verify decompression
    conn2 = sqlite3.connect(str(SQLITE_DB_PATH))
    conn2.row_factory = sqlite3.Row
    row = conn2.execute(
        "SELECT t.target, t.n, p.points_blob "
        "FROM scatter_targets t JOIN scatter_points p ON p.target_id = t.id "
        "LIMIT 1"
    ).fetchone()
    if row:
        pts = json.loads(zlib.decompress(row["points_blob"]))
        assert len(pts) == row["n"], f"Point count mismatch: {len(pts)} vs {row['n']}"
        print(f"  Spot-check OK: {row['target']} has {len(pts)} points")
    conn2.close()


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

    # 3. Build per-donor scatter data from H5AD files
    print("\n  Building donor scatter data ...")
    donor_scatter, donor_scatter_meta = build_donor_scatter()

    # 3b. Build celltype-level scatter data from donor×celltype H5AD files
    print("\n  Building celltype scatter data ...")
    celltype_scatter, celltype_scatter_meta = build_celltype_scatter()

    # 4. Assemble output (resampled_scatter removed — rho CI now in donor/celltype scatter)
    output = {
        "summary": summary,
        "donor_level": donor_level,
        "celltype_level": celltype_level,
        "donor_scatter": donor_scatter,
        "celltype_scatter": celltype_scatter,
    }

    # 5. Write JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"Size: {size_mb:.1f} MB")

    # 6. Build separate bulk RNA-seq validation JSON (GTEx + TCGA)
    print("\n  Building bulk_rnaseq_validation.json ...")
    bulk_rnaseq = build_bulk_rnaseq_json()
    if bulk_rnaseq:
        BULK_RNASEQ_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BULK_RNASEQ_OUTPUT_PATH, "w") as f:
            json.dump(bulk_rnaseq, f, separators=(",", ":"))
        size_mb2 = BULK_RNASEQ_OUTPUT_PATH.stat().st_size / (1024 * 1024)
        print(f"Output: {BULK_RNASEQ_OUTPUT_PATH}")
        print(f"Size: {size_mb2:.1f} MB")
    else:
        print("  No GTEx/TCGA data available, skipping bulk_rnaseq_validation.json")

    # 7. Write scatter metadata as Parquet for fast API queries
    parquet_dir = OUTPUT_PATH.parent / "parquet"
    parquet_dir.mkdir(exist_ok=True)

    if donor_scatter_meta:
        donor_pq_path = parquet_dir / "donor_scatter_meta.parquet"
        pd.DataFrame(donor_scatter_meta).to_parquet(donor_pq_path, index=False)
        print(f"\n  Parquet: {donor_pq_path} ({len(donor_scatter_meta)} rows)")

    if celltype_scatter_meta:
        ct_pq_path = parquet_dir / "celltype_scatter_meta.parquet"
        pd.DataFrame(celltype_scatter_meta).to_parquet(ct_pq_path, index=False)
        print(f"  Parquet: {ct_pq_path} ({len(celltype_scatter_meta)} rows)")

    # 8. Write split files for API consumption
    print("\n  Writing split files for API ...")
    write_split_files(output, bulk_rnaseq if bulk_rnaseq else None)

    # 9. Write SQLite DB for fast indexed API queries
    print("\n  Writing SQLite DB ...")
    write_sqlite_db(donor_scatter, celltype_scatter, bulk_rnaseq if bulk_rnaseq else None)

    print("\nDone!")


if __name__ == "__main__":
    main()
