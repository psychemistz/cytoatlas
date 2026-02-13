#!/usr/bin/env python3
"""
Resampled Pseudobulk Validation Pipeline.

Runs activity inference (ridge regression) on bootstrap-resampled pseudobulk
expression profiles, then computes expression-activity correlations with
confidence intervals.

The resampled H5AD files contain 100 bootstrap iterations per cell type,
created by resampling 80% of cells within each cell type.

Usage:
    # All atlases
    python scripts/16_resampled_validation.py

    # Single atlas
    python scripts/16_resampled_validation.py --atlas cima

    # Force overwrite
    python scripts/16_resampled_validation.py --force

    # Specific backend
    python scripts/16_resampled_validation.py --backend numpy
"""

import argparse
import gc
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/data/parks34/projects/1ridgesig/SecActpy")

from secactpy import (
    load_cytosig, load_secact,
    ridge_batch, estimate_batch_size,
    CUPY_AVAILABLE,
)


# =============================================================================
# Paths
# =============================================================================

ATLAS_VALIDATION_DIR = Path("/data/parks34/projects/2cytoatlas/results/atlas_validation")
OUTPUT_DIR = Path("/data/parks34/projects/2cytoatlas/results/cross_sample_validation")
MAPPING_PATH = Path("/data/parks34/projects/2cytoatlas/cytoatlas-api/static/data/signature_gene_mapping.json")

# Resampled pseudobulk files per atlas (atlas_key -> [(level_name, h5ad_basename)])
RESAMPLED_FILES = {
    "cima": [
        ("l1", "cima_pseudobulk_l1_resampled.h5ad"),
        ("l2", "cima_pseudobulk_l2_resampled.h5ad"),
        ("l3", "cima_pseudobulk_l3_resampled.h5ad"),
        ("l4", "cima_pseudobulk_l4_resampled.h5ad"),
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

SIG_TYPES = ["cytosig", "lincytosig", "secact"]


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# =============================================================================
# Signature Loading
# =============================================================================

_SIGNATURES_CACHE = None


def load_signatures() -> Dict[str, pd.DataFrame]:
    global _SIGNATURES_CACHE
    if _SIGNATURES_CACHE is not None:
        return _SIGNATURES_CACHE

    log("Loading signature matrices...")
    cytosig = load_cytosig()
    secact = load_secact()

    lincytosig_path = Path(
        "/data/parks34/projects/1ridgesig/SecActpy/secactpy/data/LinCytoSig.tsv.gz"
    )
    with gzip.open(lincytosig_path, "rt") as f:
        lincytosig = pd.read_csv(f, sep="\t", index_col=0)

    _SIGNATURES_CACHE = {
        "cytosig": cytosig,
        "lincytosig": lincytosig,
        "secact": secact,
    }
    log(
        f"  CytoSig: {cytosig.shape}, LinCytoSig: {lincytosig.shape}, SecAct: {secact.shape}"
    )
    return _SIGNATURES_CACHE


def load_target_to_gene_mapping() -> Dict[str, str]:
    if not MAPPING_PATH.exists():
        return {}
    with open(MAPPING_PATH) as f:
        data = json.load(f)
    mapping = {}
    for target_name, info in data.get("cytosig_mapping", {}).items():
        mapping[target_name] = info["hgnc_symbol"]
    return mapping


def resolve_gene_name(
    target: str, gene_set: set, cytosig_map: Dict[str, str]
) -> Optional[str]:
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


# =============================================================================
# Activity Inference on Resampled Pseudobulk
# =============================================================================


def run_resampled_activity(
    expr_h5ad_path: Path,
    output_dir: Path,
    atlas: str,
    level: str,
    force: bool = False,
    lambda_: float = 5e5,
    backend: str = "auto",
) -> Dict[str, Path]:
    """Run activity inference on resampled pseudobulk expression.

    Each observation is a celltype x bootstrap iteration.
    Mean-centers genes across all observations, then runs ridge regression.
    """
    signatures = load_signatures()
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"  Loading resampled pseudobulk: {expr_h5ad_path.name}")
    expr_adata = ad.read_h5ad(expr_h5ad_path)
    log(f"    Shape: {expr_adata.shape}")

    gene_names = list(expr_adata.var_names)
    expr_matrix = expr_adata.X
    if hasattr(expr_matrix, "toarray"):
        expr_matrix = expr_matrix.toarray()
    expr_matrix = expr_matrix.astype(np.float64)

    output_paths = {}

    for sig_name, sig_matrix in signatures.items():
        act_path = output_dir / f"{atlas}_resampled_{level}_{sig_name}.h5ad"

        if act_path.exists() and not force:
            log(f"    SKIP {sig_name} (exists): {act_path.name}")
            output_paths[sig_name] = act_path
            continue

        # Find common genes
        expr_genes = set(gene_names)
        sig_genes = set(sig_matrix.index)
        common = sorted(expr_genes & sig_genes)

        if len(common) < 100:
            log(f"    {sig_name}: too few common genes ({len(common)}), skipping")
            continue

        log(f"    Computing {sig_name} ({len(common)} common genes)...")
        t0 = time.time()

        gene_idx = [gene_names.index(g) for g in common]
        X = sig_matrix.loc[common].values.copy()
        np.nan_to_num(X, copy=False, nan=0.0)

        Y = expr_matrix[:, gene_idx].T.copy()  # (genes x samples)
        np.nan_to_num(Y, copy=False, nan=0.0)
        Y -= Y.mean(axis=1, keepdims=True)

        batch_sz = estimate_batch_size(
            n_genes=len(common),
            n_features=X.shape[1],
            available_gb=32 if CUPY_AVAILABLE else 16,
        )
        result = ridge_batch(
            X, Y, lambda_=lambda_, n_rand=1000, seed=42,
            batch_size=batch_sz, backend=backend, verbose=False,
        )
        activity = result["zscore"].T.astype(np.float32)

        elapsed = time.time() - t0

        adata_act = ad.AnnData(
            X=activity,
            obs=expr_adata.obs.copy(),
            var=pd.DataFrame(index=list(sig_matrix.columns)),
        )
        adata_act.uns["common_genes"] = len(common)
        adata_act.uns["total_sig_genes"] = len(sig_genes)
        adata_act.uns["gene_coverage"] = len(common) / len(sig_genes)
        adata_act.uns["signature"] = sig_name
        adata_act.uns["source"] = expr_h5ad_path.name
        adata_act.uns["resampled"] = True

        adata_act.write_h5ad(act_path, compression="gzip")
        output_paths[sig_name] = act_path
        log(f"      {activity.shape}, {elapsed:.1f}s -> {act_path.name}")

        del activity, adata_act, X, Y, result
        gc.collect()

    del expr_adata, expr_matrix
    gc.collect()

    return output_paths


# =============================================================================
# Correlation Analysis on Resampled Data
# =============================================================================


def compute_resampled_correlations(
    atlas: str,
    level: str,
    expr_h5ad_path: Path,
    activity_dir: Path,
    output_dir: Path,
    force: bool = False,
) -> Optional[Path]:
    """Compute expression-activity correlations for resampled data.

    For each target, compute Spearman rho across all celltype x bootstrap
    observations, plus per-bootstrap rho to get confidence intervals.
    """
    csv_path = output_dir / f"{atlas}_resampled_{level}_correlations.csv"
    if csv_path.exists() and not force:
        log(f"    SKIP correlations (exists): {csv_path.name}")
        return csv_path

    cytosig_map = load_target_to_gene_mapping()

    log(f"  Loading resampled expression: {expr_h5ad_path.name}")
    expr_adata = ad.read_h5ad(expr_h5ad_path)

    # Build gene lookup
    gene_set = set(expr_adata.var_names)
    symbol_to_varname = {}
    if "symbol" in expr_adata.var.columns:
        for idx, sym in zip(expr_adata.var_names, expr_adata.var["symbol"]):
            if pd.notna(sym):
                symbol_to_varname[str(sym)] = idx
        gene_set = set(symbol_to_varname.keys()) | gene_set

    # Get celltype and bootstrap columns
    obs = expr_adata.obs
    ct_col = None
    for col in ["cell_type", "celltype", "cell_type_l1", "cell_type_l2"]:
        if col in obs.columns:
            ct_col = col
            break
    boot_col = None
    for col in ["bootstrap_idx", "bootstrap"]:
        if col in obs.columns:
            boot_col = col
            break

    if ct_col is None:
        log(f"    WARNING: No celltype column found in {expr_h5ad_path.name}")
        return None

    rows = []

    for sig_type in SIG_TYPES:
        act_path = activity_dir / f"{atlas}_resampled_{level}_{sig_type}.h5ad"
        if not act_path.exists():
            log(f"    WARNING: {act_path.name} not found, skipping")
            continue

        act_adata = ad.read_h5ad(act_path)
        common_obs = expr_adata.obs_names.intersection(act_adata.obs_names)
        if len(common_obs) < 10:
            continue

        log(f"    {sig_type}: {len(common_obs)} common obs, {act_adata.n_vars} targets")

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

            mask = np.isfinite(expr_vals) & np.isfinite(act_vals)
            expr_vals = expr_vals[mask].astype(np.float64)
            act_vals = act_vals[mask].astype(np.float64)

            if len(expr_vals) < 10:
                continue

            # Overall correlation across all celltype x bootstrap
            rho, pval = stats.spearmanr(expr_vals, act_vals)

            # Per-bootstrap correlations for CI
            bootstrap_rhos = []
            if boot_col is not None:
                obs_common = obs.loc[common_obs[mask]]
                for boot_idx in sorted(obs_common[boot_col].unique()):
                    boot_mask = obs_common[boot_col] == boot_idx
                    if boot_mask.sum() < 5:
                        continue
                    boot_expr = expr_vals[boot_mask.values]
                    boot_act = act_vals[boot_mask.values]
                    if np.std(boot_expr) > 1e-10 and np.std(boot_act) > 1e-10:
                        br, _ = stats.spearmanr(boot_expr, boot_act)
                        if np.isfinite(br):
                            bootstrap_rhos.append(br)

            rho_ci_lower = np.percentile(bootstrap_rhos, 2.5) if len(bootstrap_rhos) >= 10 else None
            rho_ci_upper = np.percentile(bootstrap_rhos, 97.5) if len(bootstrap_rhos) >= 10 else None
            rho_std = np.std(bootstrap_rhos) if len(bootstrap_rhos) >= 10 else None

            rows.append({
                "target": target,
                "gene": gene,
                "signature": sig_type,
                "atlas": atlas,
                "level": level,
                "spearman_rho": rho,
                "spearman_pval": pval,
                "n_samples": len(expr_vals),
                "mean_expr": float(np.mean(expr_vals)),
                "std_expr": float(np.std(expr_vals)),
                "mean_activity": float(np.mean(act_vals)),
                "std_activity": float(np.std(act_vals)),
                "n_bootstraps": len(bootstrap_rhos),
                "rho_ci_lower": rho_ci_lower,
                "rho_ci_upper": rho_ci_upper,
                "rho_std": rho_std,
            })

        del act_adata
        gc.collect()

    if rows:
        df = pd.DataFrame(rows)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        log(f"    Saved {len(df)} correlation rows -> {csv_path.name}")
        return csv_path
    else:
        log(f"    No correlation data generated for {atlas}/{level}")
        return None


# =============================================================================
# Main
# =============================================================================


def process_atlas(
    atlas: str,
    force: bool = False,
    backend: str = "auto",
) -> None:
    """Process all resampled levels for one atlas."""
    entries = RESAMPLED_FILES.get(atlas, [])
    if not entries:
        log(f"No resampled files defined for {atlas}")
        return

    atlas_val_dir = ATLAS_VALIDATION_DIR / atlas / "pseudobulk"
    activity_output_dir = OUTPUT_DIR / atlas
    corr_output_dir = OUTPUT_DIR / "correlations"

    for level, filename in entries:
        expr_path = atlas_val_dir / filename
        if not expr_path.exists():
            log(f"  WARNING: {expr_path} not found, skipping")
            continue

        log(f"\n--- {atlas} / {level} ---")

        # Step 1: Activity inference
        act_paths = run_resampled_activity(
            expr_h5ad_path=expr_path,
            output_dir=activity_output_dir,
            atlas=atlas,
            level=level,
            force=force,
            backend=backend,
        )

        # Step 2: Correlation analysis
        compute_resampled_correlations(
            atlas=atlas,
            level=level,
            expr_h5ad_path=expr_path,
            activity_dir=activity_output_dir,
            output_dir=corr_output_dir,
            force=force,
        )

        gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Resampled Pseudobulk Validation Pipeline"
    )
    parser.add_argument(
        "--atlas",
        nargs="+",
        default=["all"],
        help="Atlas(es) to process: cima, inflammation_main, etc., or all (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Force overwrite")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "numpy", "cupy"],
        help="Computation backend (default: auto)",
    )

    args = parser.parse_args()

    if "all" in args.atlas:
        atlases = list(RESAMPLED_FILES.keys())
    else:
        atlases = args.atlas

    log(f"Atlases: {atlases}")
    log(f"Backend: {args.backend}")
    log(f"Force: {args.force}")

    t_start = time.time()

    for atlas in atlases:
        log(f"\n{'=' * 60}")
        log(f"ATLAS: {atlas}")
        log(f"{'=' * 60}")
        process_atlas(atlas, force=args.force, backend=args.backend)

    elapsed = time.time() - t_start
    log(f"\n{'=' * 60}")
    log(f"ALL DONE ({elapsed / 60:.1f} min)")
    log(f"{'=' * 60}")

    # Verify output files
    for atlas in atlases:
        ddir = OUTPUT_DIR / atlas
        resamp_files = sorted(ddir.glob("*_resampled_*.h5ad"))
        if resamp_files:
            log(f"\n{atlas} resampled activity files:")
            for f in resamp_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                log(f"  {f.name} ({size_mb:.1f} MB)")

    corr_dir = OUTPUT_DIR / "correlations"
    resamp_csvs = sorted(corr_dir.glob("*_resampled_*_correlations.csv"))
    if resamp_csvs:
        log(f"\nResampled correlation files:")
        for f in resamp_csvs:
            size_kb = f.stat().st_size / 1024
            log(f"  {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
