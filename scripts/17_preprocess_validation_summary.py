#!/usr/bin/env python3
"""Preprocess per-entity Spearman correlations (expression vs activity) for
the Validation Summary boxplot tab.

For each of 8 source categories, loads paired expression + activity H5AD
files, groups observations by entity (tissue, cancer type, donor, or cell
type), computes Spearman rho between expression and predicted activity for
each target within each entity, and outputs a JSON file suitable for
side-by-side CytoSig / SecAct boxplots.

Output: visualization/data/validation_corr_boxplot.json

Usage:
    python scripts/17_preprocess_validation_summary.py
"""

import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_DIR = Path("/data/parks34/projects/2cytoatlas/results/cross_sample_validation")
OUTPUT_PATH = PROJECT_ROOT / "visualization" / "data" / "validation_corr_boxplot.json"
MAPPING_PATH = PROJECT_ROOT / "cytoatlas-api" / "static" / "data" / "signature_gene_mapping.json"

MIN_N = 5          # Minimum observations per entity to compute correlation
MAX_RHOS = 50      # Subsample per-donor categories to this many rho values per target/category

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------
CATEGORIES = [
    {
        "key": "gtex_tissue",
        "label": "GTEx (per tissue)",
        "atlas_dir": "gtex",
        "expr_file": "gtex_by_tissue_expression.h5ad",
        "act_pattern": "gtex_by_tissue_{sig}.h5ad",
        "group_col": "tissue_type",
        "mode": "per_entity",
    },
    {
        "key": "tcga_cancer",
        "label": "TCGA (per cancer)",
        "atlas_dir": "tcga",
        "expr_file": "tcga_by_cancer_expression.h5ad",
        "act_pattern": "tcga_by_cancer_{sig}.h5ad",
        "group_col": "cancer_type",
        "mode": "per_entity",
    },
    {
        "key": "cima_donor",
        "label": "CIMA (per donor)",
        "atlas_dir": "cima",
        "expr_file": "cima_donor_l1_pseudobulk.h5ad",
        "act_pattern": "cima_donor_l1_{sig}.h5ad",
        "group_col": "donor",
        "mode": "per_donor",
    },
    {
        "key": "cima_celltype",
        "label": "CIMA (per celltype)",
        "atlas_dir": "cima",
        "expr_file": "cima_donor_l1_pseudobulk.h5ad",
        "act_pattern": "cima_donor_l1_{sig}.h5ad",
        "group_col": "cell_type_l1",
        "mode": "per_celltype",
    },
    {
        "key": "inflam_donor",
        "label": "Inflam (per donor)",
        "atlas_dir": "inflammation_main",
        "expr_file": "inflammation_main_donor_l1_pseudobulk.h5ad",
        "act_pattern": "inflammation_main_donor_l1_{sig}.h5ad",
        "group_col": "donor",
        "mode": "per_donor",
    },
    {
        "key": "inflam_celltype",
        "label": "Inflam (per celltype)",
        "atlas_dir": "inflammation_main",
        "expr_file": "inflammation_main_donor_l1_pseudobulk.h5ad",
        "act_pattern": "inflammation_main_donor_l1_{sig}.h5ad",
        "group_col": "Level1",
        "mode": "per_celltype",
    },
    {
        "key": "scatlas_norm_donor",
        "label": "scAtlas norm (per donor)",
        "atlas_dir": "scatlas_normal",
        "expr_file": "scatlas_normal_donor_organ_celltype1_pseudobulk.h5ad",
        "act_pattern": "scatlas_normal_donor_organ_celltype1_{sig}.h5ad",
        "group_col": "donor",
        "mode": "per_donor",
    },
    {
        "key": "scatlas_canc_donor",
        "label": "scAtlas canc (per donor)",
        "atlas_dir": "scatlas_cancer",
        "expr_file": "scatlas_cancer_donor_organ_celltype1_pseudobulk.h5ad",
        "act_pattern": "scatlas_cancer_donor_organ_celltype1_{sig}.h5ad",
        "group_col": "donor",
        "mode": "per_donor",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def safe_spearmanr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Compute Spearman rho, returning None on failure."""
    if len(x) < MIN_N:
        return None
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    try:
        rho, _ = stats.spearmanr(x, y)
        if not np.isfinite(rho):
            return None
        return float(rho)
    except Exception:
        return None


def to_dense_1d(arr) -> np.ndarray:
    """Convert sparse or matrix to dense 1-D float64 array."""
    if hasattr(arr, "toarray"):
        return arr.toarray().flatten().astype(np.float64)
    return np.asarray(arr).flatten().astype(np.float64)


def build_gene_lookup(expr_adata: ad.AnnData) -> Tuple[set, Dict[str, str]]:
    """Build gene name lookup, handling Ensembl IDs with symbol column."""
    gene_set = set(expr_adata.var_names)
    symbol_to_varname = {}
    if "symbol" in expr_adata.var.columns:
        for idx, sym in zip(expr_adata.var_names, expr_adata.var["symbol"]):
            if pd.notna(sym):
                symbol_to_varname[str(sym)] = idx
        gene_set = set(symbol_to_varname.keys()) | gene_set
    return gene_set, symbol_to_varname


# ---------------------------------------------------------------------------
# Core: process one (category, sig_type) pair
# ---------------------------------------------------------------------------
def process_one(
    cat: dict,
    sig_type: str,
    cytosig_map: Dict[str, str],
) -> Dict[str, List[float]]:
    """Load files, compute per-entity rhos, free memory, return results.

    Returns {target: [rho_values]} for all valid targets.
    """
    atlas_dir = VALIDATION_DIR / cat["atlas_dir"]
    expr_path = atlas_dir / cat["expr_file"]
    act_path = atlas_dir / cat["act_pattern"].format(sig=sig_type)

    if not expr_path.exists() or not act_path.exists():
        print(f"    WARNING: files not found, skipping")
        return {}

    # Load activity file (small: ~N_obs x 44 for cytosig)
    act_adata = ad.read_h5ad(act_path)
    act_targets = list(act_adata.var_names)

    # Load expression metadata only first to build gene lookup
    expr_adata = ad.read_h5ad(expr_path, backed="r")
    gene_set, symbol_to_varname = build_gene_lookup(expr_adata)
    expr_var_names_set = set(expr_adata.var_names)
    expr_var_names_list = list(expr_adata.var_names)

    # Determine which targets map to valid expression genes
    target_to_expr_var = {}
    for target in act_targets:
        gene = resolve_gene_name(target, gene_set, cytosig_map)
        if gene is None:
            continue
        expr_var = symbol_to_varname.get(gene, gene)
        if expr_var not in expr_var_names_set:
            continue
        target_to_expr_var[target] = expr_var

    if not target_to_expr_var:
        expr_adata.file.close()
        return {}

    # Find common observations
    common_obs = list(expr_adata.obs_names.intersection(act_adata.obs_names))
    if len(common_obs) < MIN_N:
        expr_adata.file.close()
        return {}

    # Read group column from expression obs
    group_col = cat["group_col"]
    # For backed mode, read obs column
    groups = expr_adata.obs.loc[common_obs, group_col].values

    # Now extract only the expression columns we need into a dense array
    # This avoids loading the full expression matrix
    needed_expr_vars = sorted(set(target_to_expr_var.values()))
    expr_var_to_idx = {v: i for i, v in enumerate(needed_expr_vars)}

    # Find column indices for slicing
    var_name_to_pos = {v: i for i, v in enumerate(expr_var_names_list)}
    col_indices = [var_name_to_pos[v] for v in needed_expr_vars]

    # Read expression data for common obs and needed columns
    # Use integer indexing on backed adata
    obs_name_to_pos = {n: i for i, n in enumerate(expr_adata.obs_names)}
    row_indices = [obs_name_to_pos[o] for o in common_obs]

    # Read the subset — for backed mode, slice rows then columns
    # Read in chunks to limit memory
    n_obs = len(row_indices)
    n_cols = len(col_indices)
    expr_matrix = np.empty((n_obs, n_cols), dtype=np.float64)

    CHUNK = 2000
    for start in range(0, n_obs, CHUNK):
        end = min(start + CHUNK, n_obs)
        chunk_rows = row_indices[start:end]
        # Read full row slice from backed file, then select columns
        chunk_data = expr_adata.X[chunk_rows, :]
        if hasattr(chunk_data, "toarray"):
            chunk_data = chunk_data.toarray()
        expr_matrix[start:end, :] = chunk_data[:, col_indices].astype(np.float64)

    expr_adata.file.close()
    del expr_adata
    gc.collect()

    # Also extract activity data for common obs
    act_sub = act_adata[common_obs]
    act_matrix = act_sub.X
    if hasattr(act_matrix, "toarray"):
        act_matrix = act_matrix.toarray()
    act_matrix = np.asarray(act_matrix, dtype=np.float64)
    act_var_list = list(act_adata.var_names)
    act_var_to_idx = {v: i for i, v in enumerate(act_var_list)}

    del act_adata, act_sub
    gc.collect()

    # Now compute per-entity rhos using numpy arrays only
    unique_groups = pd.unique(groups)
    result = {}

    for target, expr_var in target_to_expr_var.items():
        e_col = expr_var_to_idx[expr_var]
        a_col = act_var_to_idx[target]

        expr_col = expr_matrix[:, e_col]
        act_col = act_matrix[:, a_col]

        rhos = []
        for grp in unique_groups:
            mask = groups == grp
            n_in = mask.sum()
            if n_in < MIN_N:
                continue

            e_vals = expr_col[mask]
            a_vals = act_col[mask]

            # Remove NaN
            finite = np.isfinite(e_vals) & np.isfinite(a_vals)
            e_clean = e_vals[finite]
            a_clean = a_vals[finite]

            rho = safe_spearmanr(e_clean, a_clean)
            if rho is not None:
                rhos.append(round(rho, 4))

        if rhos:
            result[target] = rhos

    del expr_matrix, act_matrix
    gc.collect()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("Preprocessing Validation Summary Boxplot Data")
    print("=" * 60)

    cytosig_map = load_target_to_gene_mapping()
    categories_meta = [{"key": c["key"], "label": c["label"]} for c in CATEGORIES]

    cytosig_result = {"targets": [], "rhos": {}}
    secact_result = {"targets": [], "rhos": {}}

    # Process categories that share the same files together to avoid reloading.
    # Group categories by (atlas_dir, expr_file) to share loaded data.
    from collections import defaultdict
    file_groups = defaultdict(list)
    for cat in CATEGORIES:
        fk = (cat["atlas_dir"], cat["expr_file"])
        file_groups[fk].append(cat)

    for sig_type, sig_result in [("cytosig", cytosig_result), ("secact", secact_result)]:
        print(f"\n{'='*40}")
        print(f"Processing {sig_type}")
        print(f"{'='*40}")
        all_targets = set()
        cat_rhos = {}

        for file_key, cats_in_group in file_groups.items():
            atlas_dir_name, expr_file = file_key
            atlas_dir = VALIDATION_DIR / atlas_dir_name
            expr_path = atlas_dir / expr_file
            act_path = atlas_dir / cats_in_group[0]["act_pattern"].format(sig=sig_type)

            if not expr_path.exists() or not act_path.exists():
                for cat in cats_in_group:
                    print(f"  {cat['label']}: files not found, skipping")
                    cat_rhos[cat["key"]] = {}
                continue

            print(f"  Loading {atlas_dir_name}/{sig_type}...", flush=True)

            # Load activity (small)
            act_adata = ad.read_h5ad(act_path)
            act_targets = list(act_adata.var_names)

            # Load expression metadata (backed mode)
            expr_adata = ad.read_h5ad(expr_path, backed="r")
            gene_set, symbol_to_varname = build_gene_lookup(expr_adata)
            expr_var_names_list = list(expr_adata.var_names)
            expr_var_names_set = set(expr_var_names_list)

            # Map targets to expression var names
            target_to_expr_var = {}
            for target in act_targets:
                gene = resolve_gene_name(target, gene_set, cytosig_map)
                if gene is None:
                    continue
                expr_var = symbol_to_varname.get(gene, gene)
                if expr_var not in expr_var_names_set:
                    continue
                target_to_expr_var[target] = expr_var

            if not target_to_expr_var:
                for cat in cats_in_group:
                    cat_rhos[cat["key"]] = {}
                    print(f"  {cat['label']}: 0 targets resolvable")
                expr_adata.file.close()
                del act_adata, expr_adata
                gc.collect()
                continue

            # Find common observations
            common_obs = list(expr_adata.obs_names.intersection(act_adata.obs_names))
            if len(common_obs) < MIN_N:
                for cat in cats_in_group:
                    cat_rhos[cat["key"]] = {}
                    print(f"  {cat['label']}: insufficient common obs")
                expr_adata.file.close()
                del act_adata, expr_adata
                gc.collect()
                continue

            # Read group columns needed by all categories in this group
            group_cols_needed = set(c["group_col"] for c in cats_in_group)
            group_data = {}
            for gc_name in group_cols_needed:
                group_data[gc_name] = expr_adata.obs.loc[common_obs, gc_name].values

            # Extract expression columns we need
            needed_expr_vars = sorted(set(target_to_expr_var.values()))
            expr_var_to_idx = {v: i for i, v in enumerate(needed_expr_vars)}
            var_name_to_pos = {v: i for i, v in enumerate(expr_var_names_list)}
            col_indices = [var_name_to_pos[v] for v in needed_expr_vars]

            obs_name_to_pos = {n: i for i, n in enumerate(expr_adata.obs_names)}
            row_indices = [obs_name_to_pos[o] for o in common_obs]

            n_obs = len(row_indices)
            n_cols = len(col_indices)
            print(f"    Reading expression matrix: {n_obs} obs × {n_cols} genes...", flush=True)

            expr_matrix = np.empty((n_obs, n_cols), dtype=np.float64)
            CHUNK = 2000
            for start in range(0, n_obs, CHUNK):
                end = min(start + CHUNK, n_obs)
                chunk_rows = row_indices[start:end]
                chunk_data = expr_adata.X[chunk_rows, :]
                if hasattr(chunk_data, "toarray"):
                    chunk_data = chunk_data.toarray()
                expr_matrix[start:end, :] = chunk_data[:, col_indices].astype(np.float64)

            expr_adata.file.close()
            del expr_adata
            gc.collect()

            # Extract activity matrix
            act_sub = act_adata[common_obs]
            act_matrix = act_sub.X
            if hasattr(act_matrix, "toarray"):
                act_matrix = act_matrix.toarray()
            act_matrix = np.asarray(act_matrix, dtype=np.float64)
            act_var_list = list(act_adata.var_names)
            act_var_to_idx = {v: i for i, v in enumerate(act_var_list)}

            del act_adata, act_sub
            gc.collect()

            # Compute rhos for each category in this file group
            for cat in cats_in_group:
                cat_key = cat["key"]
                grp_col = cat["group_col"]
                groups = group_data[grp_col]
                unique_groups = pd.unique(groups)

                target_rhos = {}
                for target, expr_var in target_to_expr_var.items():
                    e_col = expr_var_to_idx[expr_var]
                    a_col = act_var_to_idx[target]

                    expr_col = expr_matrix[:, e_col]
                    act_col = act_matrix[:, a_col]

                    rhos = []
                    for grp in unique_groups:
                        mask = groups == grp
                        n_in = mask.sum()
                        if n_in < MIN_N:
                            continue

                        e_vals = expr_col[mask]
                        a_vals = act_col[mask]

                        finite = np.isfinite(e_vals) & np.isfinite(a_vals)
                        e_clean = e_vals[finite]
                        a_clean = a_vals[finite]

                        rho = safe_spearmanr(e_clean, a_clean)
                        if rho is not None:
                            rhos.append(round(rho, 4))

                    if rhos:
                        target_rhos[target] = rhos

                cat_rhos[cat_key] = target_rhos
                all_targets.update(target_rhos.keys())
                print(f"  {cat['label']}: {len(target_rhos)} targets")

            del expr_matrix, act_matrix, group_data
            gc.collect()

        # Build output structure
        sorted_targets = sorted(all_targets)
        sig_result["targets"] = sorted_targets

        for target in sorted_targets:
            target_rhos_by_cat = {}
            for cat in CATEGORIES:
                cat_key = cat["key"]
                rhos = cat_rhos.get(cat_key, {}).get(target, [])

                # Subsample if too many
                if len(rhos) > MAX_RHOS:
                    rng = np.random.default_rng(42)
                    rhos = sorted(rng.choice(rhos, size=MAX_RHOS, replace=False).tolist())

                target_rhos_by_cat[cat_key] = rhos

            sig_result["rhos"][target] = target_rhos_by_cat

        n_targets = len(sorted_targets)
        total_rhos = sum(
            len(v) for t in sig_result["rhos"].values() for v in t.values()
        )
        print(f"\n  {sig_type}: {n_targets} targets, {total_rhos:,} total rho values")

    # Build output
    output = {
        "categories": categories_meta,
        "cytosig": cytosig_result,
        "secact": secact_result,
    }

    # Write JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"Size: {file_size_mb:.1f} MB")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
