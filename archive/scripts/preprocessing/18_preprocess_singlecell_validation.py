#!/usr/bin/env python3
"""Preprocess single-cell validation data into SQLite for all-cell statistics.

For each atlas and signature type, reads single-cell activity H5ADs (backed
mode) along with the original expression H5ADs, and computes:
  - Exact statistics from ALL cells (Spearman rho, Mann-Whitney, etc.)
  - 2D density bins (100x100 grid) from ALL cells
  - 50K stratified sample for WebGL scatter overlay

Output: visualization/data/singlecell_scatter.db
"""

import gc
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
ATLAS_VALIDATION_DIR = Path("/data/parks34/projects/2cytoatlas/results/atlas_validation")
OUTPUT_DB = PROJECT_ROOT / "visualization" / "data" / "singlecell_scatter.db"
MAPPING_PATH = PROJECT_ROOT / "cytoatlas-api" / "static" / "data" / "signature_gene_mapping.json"

# Atlas configs: atlas_key -> (h5ad_prefix, expression_h5ad_path)
ATLAS_CONFIGS = {
    "cima": {
        "expr_h5ad": Path("/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/"
                          "CIMA_RNA_6484974cells_36326genes_compressed.h5ad"),
        "celltype_cols": ["cell_type_l1", "cell_type_l2", "cell_type_l3", "cell_type_l4"],
        "donor_col": "donor_id",
    },
    "inflammation_main": {
        "expr_h5ad": Path("/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/"
                          "INFLAMMATION_ATLAS_main_afterQC.h5ad"),
        "celltype_cols": ["cell_type_l1", "cell_type_l2", "celltype", "cell_type"],
        "donor_col": "donor_id",
    },
    "inflammation_val": {
        "expr_h5ad": Path("/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/"
                          "INFLAMMATION_ATLAS_validation_afterQC.h5ad"),
        "celltype_cols": ["cell_type_l1", "cell_type_l2", "celltype", "cell_type"],
        "donor_col": "donor_id",
    },
    "inflammation_ext": {
        "expr_h5ad": Path("/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/"
                          "INFLAMMATION_ATLAS_external_afterQC.h5ad"),
        "celltype_cols": ["cell_type_l1", "cell_type_l2", "celltype", "cell_type"],
        "donor_col": "donor_id",
    },
    "scatlas_normal": {
        "expr_h5ad": Path("/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/"
                          "igt_s9_fine_counts.h5ad"),
        "celltype_cols": ["cellType1", "cellType2", "cell_type", "celltype"],
        "donor_col": "donor_id",
    },
    "scatlas_cancer": {
        "expr_h5ad": Path("/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/"
                          "PanCancer_igt_s9_fine_counts.h5ad"),
        "celltype_cols": ["cellType1", "cellType2", "cell_type", "celltype"],
        "donor_col": "donor_id",
    },
}

SIG_TYPES = ["cytosig", "lincytosig", "secact"]
SAMPLE_SIZE = 50_000  # Stratified sample for scatter overlay
DENSITY_BINS = 100  # 100x100 grid


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


def find_celltype_col(obs_columns, candidate_cols):
    """Find the first matching celltype column."""
    for col in candidate_cols:
        if col in obs_columns:
            return col
    return None


def find_donor_col(obs_columns, donor_col):
    """Find the donor column."""
    for col in [donor_col, "donor", "sample_id", "patient_id", "subject_id"]:
        if col in obs_columns:
            return col
    return None


def compute_density_bins(expr_vals, act_vals, n_bins=DENSITY_BINS):
    """Compute 2D density bins from all cells.

    Returns dict with axis ranges and flattened bin counts.
    """
    # Compute range
    expr_min, expr_max = float(np.min(expr_vals)), float(np.max(expr_vals))
    act_min, act_max = float(np.min(act_vals)), float(np.max(act_vals))

    # Add small margin
    expr_margin = (expr_max - expr_min) * 0.02 or 0.1
    act_margin = (act_max - act_min) * 0.02 or 0.1

    # Compute 2D histogram
    hist, xedges, yedges = np.histogram2d(
        expr_vals, act_vals,
        bins=n_bins,
        range=[[expr_min - expr_margin, expr_max + expr_margin],
               [act_min - act_margin, act_max + act_margin]],
    )

    return {
        "expr_range": [round(float(xedges[0]), 4), round(float(xedges[-1]), 4)],
        "act_range": [round(float(yedges[0]), 4), round(float(yedges[-1]), 4)],
        "n_bins": n_bins,
        "counts": hist.astype(np.int32).flatten().tolist(),
    }


def stratified_sample(expr_vals, act_vals, ct_labels, donor_labels, n=SAMPLE_SIZE):
    """Take a stratified sample proportional to cell type frequencies.

    Returns arrays of (expr, act, ct_idx, donor_idx) for sampled cells.
    """
    total = len(expr_vals)
    if total <= n:
        indices = np.arange(total)
    else:
        # Stratify by cell type
        unique_cts = np.unique(ct_labels)
        indices_list = []
        for ct in unique_cts:
            ct_mask = ct_labels == ct
            ct_count = ct_mask.sum()
            ct_n = max(1, int(round(n * ct_count / total)))
            ct_indices = np.where(ct_mask)[0]
            if len(ct_indices) > ct_n:
                ct_indices = np.random.choice(ct_indices, ct_n, replace=False)
            indices_list.append(ct_indices)
        indices = np.concatenate(indices_list)
        # Trim to exact n if over
        if len(indices) > n:
            indices = np.random.choice(indices, n, replace=False)

    return (
        expr_vals[indices],
        act_vals[indices],
        ct_labels[indices],
        donor_labels[indices] if donor_labels is not None else None,
    )


def compute_celltype_stats(expr_vals, act_vals, ct_labels):
    """Compute per-celltype statistics from all cells.

    Returns list of dicts with per-celltype breakdown.
    """
    unique_cts = np.unique(ct_labels)
    result = []
    for ct in unique_cts:
        mask = ct_labels == ct
        e = expr_vals[mask]
        a = act_vals[mask]
        n_ct = len(e)
        if n_ct < 5:
            continue

        # Expressing cells (expression > 0)
        expr_mask = e > 0
        n_expressing = int(expr_mask.sum())

        rho, pval = (np.nan, np.nan)
        if n_ct >= 10 and np.std(e) > 1e-10 and np.std(a) > 1e-10:
            rho, pval = stats.spearmanr(e, a)

        result.append({
            "celltype": str(ct),
            "n": n_ct,
            "n_expressing": n_expressing,
            "mean_expr": round(float(np.mean(e)), 4),
            "mean_act": round(float(np.mean(a)), 4),
            "rho": round(float(rho), 4) if np.isfinite(rho) else None,
            "pval": float(f"{pval:.2e}") if np.isfinite(pval) else None,
        })

    return result


def create_db():
    """Create the SQLite database with schema."""
    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_DB.exists():
        OUTPUT_DB.unlink()

    conn = sqlite3.connect(str(OUTPUT_DB))
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE sc_targets (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            atlas                   TEXT NOT NULL,
            sigtype                 TEXT NOT NULL,
            target                  TEXT NOT NULL,
            gene                    TEXT,
            n_total                 INTEGER,
            n_expressing            INTEGER,
            expressing_fraction     REAL,
            rho                     REAL,
            pval                    REAL,
            fold_change             REAL,
            mann_whitney_p          REAL,
            mean_act_expressing     REAL,
            mean_act_non_expressing REAL,
            celltype_stats_json     TEXT
        );

        CREATE TABLE sc_density (
            target_id   INTEGER PRIMARY KEY REFERENCES sc_targets(id),
            bins_blob   BLOB NOT NULL
        );

        CREATE TABLE sc_points (
            target_id   INTEGER PRIMARY KEY REFERENCES sc_targets(id),
            points_blob BLOB NOT NULL
        );
    """)

    conn.commit()
    return conn


def process_atlas_sigtype(
    conn: sqlite3.Connection,
    atlas_key: str,
    sig_type: str,
    cytosig_map: Dict[str, str],
):
    """Process one atlas + sigtype combination."""
    config = ATLAS_CONFIGS[atlas_key]
    act_h5ad_path = (
        ATLAS_VALIDATION_DIR / atlas_key / "singlecell"
        / f"{atlas_key}_singlecell_{sig_type}.h5ad"
    )
    expr_h5ad_path = config["expr_h5ad"]

    if not act_h5ad_path.exists():
        print(f"    SKIP: {act_h5ad_path.name} not found")
        return 0
    if not expr_h5ad_path.exists():
        print(f"    SKIP: Expression H5AD not found: {expr_h5ad_path}")
        return 0

    print(f"    Loading activity H5AD (backed): {act_h5ad_path.name}")
    act_adata = ad.read_h5ad(act_h5ad_path, backed="r")

    print(f"    Loading expression H5AD (backed): {expr_h5ad_path.name}")
    expr_adata = ad.read_h5ad(expr_h5ad_path, backed="r")

    # Build gene lookup
    gene_set = set(expr_adata.var_names)
    symbol_to_varname = {}
    if "symbol" in expr_adata.var.columns:
        for idx, sym in zip(expr_adata.var_names, expr_adata.var["symbol"]):
            if pd.notna(sym):
                symbol_to_varname[str(sym)] = idx
        gene_set = set(symbol_to_varname.keys()) | gene_set

    # Find common cells
    common_obs = act_adata.obs_names.intersection(expr_adata.obs_names)
    n_common = len(common_obs)
    print(f"    Common cells: {n_common:,}")
    if n_common < 100:
        print(f"    SKIP: Too few common cells")
        return 0

    # Find celltype column
    ct_col = find_celltype_col(act_adata.obs.columns, config["celltype_cols"])
    if ct_col is None:
        ct_col = find_celltype_col(expr_adata.obs.columns, config["celltype_cols"])

    # Find donor column
    donor_col = find_donor_col(act_adata.obs.columns, config["donor_col"])
    if donor_col is None:
        donor_col = find_donor_col(expr_adata.obs.columns, config["donor_col"])

    # Pre-load celltype and donor labels for common cells
    if ct_col and ct_col in act_adata.obs.columns:
        ct_labels_all = act_adata.obs.loc[common_obs, ct_col].values.astype(str)
    elif ct_col and ct_col in expr_adata.obs.columns:
        ct_labels_all = expr_adata.obs.loc[common_obs, ct_col].values.astype(str)
    else:
        ct_labels_all = np.array(["unknown"] * n_common)

    if donor_col and donor_col in act_adata.obs.columns:
        donor_labels_all = act_adata.obs.loc[common_obs, donor_col].values.astype(str)
    elif donor_col and donor_col in expr_adata.obs.columns:
        donor_labels_all = expr_adata.obs.loc[common_obs, donor_col].values.astype(str)
    else:
        donor_labels_all = None

    cur = conn.cursor()
    n_targets = 0

    for target in act_adata.var_names:
        gene = resolve_gene_name(target, gene_set, cytosig_map)
        if gene is None:
            continue
        expr_var = symbol_to_varname.get(gene, gene)
        if expr_var not in set(expr_adata.var_names):
            continue

        # Read activity values for all common cells
        act_vals = act_adata[common_obs, target].X
        if hasattr(act_vals, "toarray"):
            act_vals = act_vals.toarray()
        act_vals = act_vals.flatten().astype(np.float64)

        # Read expression values for all common cells
        expr_vals = expr_adata[common_obs, expr_var].X
        if hasattr(expr_vals, "toarray"):
            expr_vals = expr_vals.toarray()
        expr_vals = expr_vals.flatten().astype(np.float64)

        # Remove NaN
        mask = np.isfinite(expr_vals) & np.isfinite(act_vals)
        expr_vals = expr_vals[mask]
        act_vals = act_vals[mask]
        ct_labels = ct_labels_all[mask]
        donor_labels = donor_labels_all[mask] if donor_labels_all is not None else None

        n_total = len(expr_vals)
        if n_total < 100:
            continue

        # Compute exact statistics from ALL cells
        expressing_mask = expr_vals > 0
        n_expressing = int(expressing_mask.sum())
        expressing_fraction = n_expressing / n_total

        # Spearman correlation (all cells)
        rho, pval = np.nan, np.nan
        if np.std(expr_vals) > 1e-10 and np.std(act_vals) > 1e-10:
            rho, pval = stats.spearmanr(expr_vals, act_vals)

        # Mann-Whitney: activity in expressing vs non-expressing
        fold_change = None
        mann_whitney_p = None
        mean_act_expr = None
        mean_act_nonexpr = None

        if n_expressing >= 10 and (n_total - n_expressing) >= 10:
            act_expressing = act_vals[expressing_mask]
            act_non_expressing = act_vals[~expressing_mask]
            mean_act_expr = float(np.mean(act_expressing))
            mean_act_nonexpr = float(np.mean(act_non_expressing))

            if abs(mean_act_nonexpr) > 1e-10:
                fold_change = mean_act_expr / mean_act_nonexpr
            elif mean_act_expr > mean_act_nonexpr:
                fold_change = float("inf")

            try:
                _, mann_whitney_p = stats.mannwhitneyu(
                    act_expressing, act_non_expressing, alternative="two-sided"
                )
            except Exception:
                mann_whitney_p = None

        # Per-celltype stats
        celltype_stats = compute_celltype_stats(expr_vals, act_vals, ct_labels)

        # Density bins (all cells)
        density = compute_density_bins(expr_vals, act_vals)

        # Stratified sample (50K)
        s_expr, s_act, s_ct, s_donor = stratified_sample(
            expr_vals, act_vals, ct_labels, donor_labels
        )

        # Build unique label lookups for compact encoding
        unique_cts = sorted(set(ct_labels))
        ct_to_idx = {ct: i for i, ct in enumerate(unique_cts)}
        unique_donors = sorted(set(donor_labels)) if donor_labels is not None else []
        donor_to_idx = {d: i for i, d in enumerate(unique_donors)}

        # Encode sampled points compactly: [expr, act, ct_idx, donor_idx, is_expressing]
        sampled_points = {
            "celltypes": unique_cts,
            "donors": unique_donors,
            "points": [],
        }
        for i in range(len(s_expr)):
            pt = [
                round(float(s_expr[i]), 3),
                round(float(s_act[i]), 3),
                ct_to_idx.get(str(s_ct[i]), -1),
            ]
            if s_donor is not None:
                pt.append(donor_to_idx.get(str(s_donor[i]), -1))
            else:
                pt.append(-1)
            pt.append(1 if s_expr[i] > 0 else 0)  # is_expressing
            sampled_points["points"].append(pt)

        # Insert into DB
        cur.execute(
            "INSERT INTO sc_targets "
            "(atlas, sigtype, target, gene, n_total, n_expressing, "
            " expressing_fraction, rho, pval, fold_change, mann_whitney_p, "
            " mean_act_expressing, mean_act_non_expressing, celltype_stats_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                atlas_key, sig_type, target, gene, n_total, n_expressing,
                round(expressing_fraction, 6),
                round(float(rho), 6) if np.isfinite(rho) else None,
                float(f"{pval:.2e}") if np.isfinite(pval) else None,
                round(fold_change, 4) if fold_change is not None and np.isfinite(fold_change) else None,
                float(f"{mann_whitney_p:.2e}") if mann_whitney_p is not None and np.isfinite(mann_whitney_p) else None,
                round(mean_act_expr, 4) if mean_act_expr is not None else None,
                round(mean_act_nonexpr, 4) if mean_act_nonexpr is not None else None,
                json.dumps(celltype_stats, separators=(",", ":")),
            ),
        )
        target_id = cur.lastrowid

        # Insert density bins (zlib compressed)
        density_blob = zlib.compress(
            json.dumps(density, separators=(",", ":")).encode(), level=6
        )
        cur.execute(
            "INSERT INTO sc_density (target_id, bins_blob) VALUES (?,?)",
            (target_id, density_blob),
        )

        # Insert sampled points (zlib compressed)
        points_blob = zlib.compress(
            json.dumps(sampled_points, separators=(",", ":")).encode(), level=6
        )
        cur.execute(
            "INSERT INTO sc_points (target_id, points_blob) VALUES (?,?)",
            (target_id, points_blob),
        )

        n_targets += 1
        if n_targets % 50 == 0:
            print(f"      {n_targets} targets processed...")
            conn.commit()

    conn.commit()

    # Cleanup
    del act_adata, expr_adata
    gc.collect()

    return n_targets


def main():
    print("=" * 60)
    print("Single-Cell Validation Preprocessing (All Cells)")
    print("=" * 60)

    cytosig_map = load_target_to_gene_mapping()
    conn = create_db()

    total_targets = 0

    for atlas_key in ATLAS_CONFIGS:
        print(f"\n{'='*40}")
        print(f"Atlas: {atlas_key}")
        print(f"{'='*40}")

        for sig_type in SIG_TYPES:
            print(f"\n  Signature type: {sig_type}")
            n = process_atlas_sigtype(conn, atlas_key, sig_type, cytosig_map)
            print(f"    -> {n} targets written")
            total_targets += n

    # Create indexes
    print("\nCreating indexes...")
    cur = conn.cursor()
    cur.executescript("""
        CREATE UNIQUE INDEX idx_sc_targets_unique
            ON sc_targets(atlas, sigtype, target);
        CREATE INDEX idx_sc_targets_lookup
            ON sc_targets(atlas, sigtype);
    """)

    # Optimize
    print("ANALYZE + VACUUM...")
    cur.execute("ANALYZE")
    conn.execute("VACUUM")

    # Spot-check
    row = conn.execute(
        "SELECT t.target, t.n_total, t.atlas "
        "FROM sc_targets t LIMIT 1"
    ).fetchone()
    if row:
        print(f"Spot-check: {row[2]}/{row[0]} has {row[1]:,} cells")

    conn.close()

    size_mb = OUTPUT_DB.stat().st_size / (1024 * 1024)
    print(f"\nOutput: {OUTPUT_DB}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Total targets: {total_targets}")
    print("\nDone!")


if __name__ == "__main__":
    main()
