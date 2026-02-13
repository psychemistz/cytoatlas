#!/usr/bin/env python3
"""
TCGA Primary Tumor Filter + Activity Inference.

Creates two new levels of TCGA data:
  Level 2: Primary tumor (sample type 01) + blood cancer (03) only
  Level 3: Per-cancer-type stratified from Level 2

Reads the existing donor_only_expression.h5ad and filters by sample type
extracted from TCGA barcodes.

Usage:
    python scripts/15b_tcga_primary_filter.py
    python scripts/15b_tcga_primary_filter.py --force
    python scripts/15b_tcga_primary_filter.py --backend numpy
"""

import argparse
import gc
import gzip
import sys
import time
from pathlib import Path
from typing import Dict

import anndata as ad
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/data/parks34/projects/1ridgesig/SecActpy")

from secactpy import load_cytosig, load_secact, ridge


# =============================================================================
# Paths
# =============================================================================

TCGA_DIR = Path('/data/parks34/projects/2cytoatlas/results/cross_sample_validation/tcga')

# Keep sample types 01 (Primary Solid Tumor) and 03 (Primary Blood Derived Cancer)
PRIMARY_SAMPLE_TYPES = {'01', '03'}


def log(msg: str) -> None:
    ts = time.strftime('%H:%M:%S')
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

    lincytosig_path = Path("/data/parks34/projects/1ridgesig/SecActpy/secactpy/data/LinCytoSig.tsv.gz")
    with gzip.open(lincytosig_path, 'rt') as f:
        lincytosig = pd.read_csv(f, sep='\t', index_col=0)

    _SIGNATURES_CACHE = {
        'cytosig': cytosig,
        'lincytosig': lincytosig,
        'secact': secact,
    }

    for name, sig in _SIGNATURES_CACHE.items():
        log(f"  {name}: {sig.shape[0]} genes x {sig.shape[1]} targets")

    return _SIGNATURES_CACHE


def extract_sample_type(barcode: str) -> str:
    """Extract 2-digit sample type from TCGA barcode.

    TCGA-OR-A5J1-01A-11R-A29S-07
                  ^^-- sample type (01 = Primary Tumor, 03 = Blood Cancer, etc.)
    """
    parts = str(barcode).split('-')
    if len(parts) >= 4:
        return parts[3][:2]
    return 'XX'


def run_activity_inference(
    expr_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    force: bool = False,
    lambda_: float = 5e5,
    backend: str = 'auto',
) -> Dict[str, Path]:
    """Run activity inference (mean-center across all samples, single ridge)."""
    signatures = load_signatures()
    gene_names = list(expr_df.columns)
    expr_matrix = expr_df.values
    output_paths = {}

    for sig_name, sig_matrix in signatures.items():
        act_path = output_dir / f"{prefix}_{sig_name}.h5ad"

        if act_path.exists() and not force:
            log(f"  SKIP {sig_name} (exists): {act_path.name}")
            output_paths[sig_name] = act_path
            continue

        log(f"  Computing {sig_name}...")

        # Find common genes
        expr_genes = set(gene_names)
        sig_genes = set(sig_matrix.index)
        common = sorted(expr_genes & sig_genes)

        if len(common) < 100:
            log(f"    Too few common genes: {len(common)}, skipping")
            continue

        gene_idx = [gene_names.index(g) for g in common]
        X = sig_matrix.loc[common].values.copy()
        np.nan_to_num(X, copy=False, nan=0.0)

        t0 = time.time()

        # Mean-center across all samples, run ridge once
        Y = expr_matrix[:, gene_idx].T.astype(np.float64)
        np.nan_to_num(Y, copy=False, nan=0.0)
        Y -= Y.mean(axis=1, keepdims=True)
        result = ridge(X, Y, lambda_=lambda_, n_rand=1000, backend=backend, verbose=False)
        activity = result['zscore'].T.astype(np.float32)

        elapsed = time.time() - t0

        adata_act = ad.AnnData(
            X=activity,
            obs=obs_df.copy(),
            var=pd.DataFrame(index=list(sig_matrix.columns)),
        )
        adata_act.uns['common_genes'] = len(common)
        adata_act.uns['total_sig_genes'] = len(sig_genes)
        adata_act.uns['gene_coverage'] = len(common) / len(sig_genes)
        adata_act.uns['signature'] = sig_name
        adata_act.uns['source'] = f"{prefix}_expression.h5ad"
        adata_act.uns['data_format'] = 'EBPlusPlus_RSEM_normalized_counts'
        adata_act.uns['transform'] = 'clip(0) -> log2(RSEM+1)'

        adata_act.write_h5ad(act_path, compression='gzip')
        output_paths[sig_name] = act_path
        log(f"    {sig_name}: {activity.shape}, {len(common)} genes, {elapsed:.1f}s -> {act_path.name}")

        del activity, adata_act, X, Y, result
        gc.collect()

    return output_paths


def run_stratified_activity_inference(
    expr_df: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    group_col: str,
    force: bool = False,
    lambda_: float = 5e5,
    backend: str = 'auto',
    min_samples: int = 30,
) -> Dict[str, Path]:
    """Run within-group stratified activity inference."""
    signatures = load_signatures()

    groups = metadata[group_col].dropna()
    groups = groups[groups != 'Unknown']
    group_counts = groups.value_counts()
    valid_groups = group_counts[group_counts >= min_samples].index.tolist()
    valid_mask = metadata[group_col].isin(valid_groups)

    log(f"\nStratified inference ({prefix}): {len(valid_groups)} groups "
        f"with >= {min_samples} samples (of {group_counts.shape[0]} total)")
    for g in sorted(valid_groups):
        log(f"  {g}: {group_counts[g]} samples")

    # Filter to valid samples
    expr_valid = expr_df.loc[valid_mask]
    meta_valid = metadata.loc[valid_mask].copy()

    # Save stratified expression H5AD
    pb_path = output_dir / f"{prefix}_expression.h5ad"
    if not pb_path.exists() or force:
        log(f"Saving stratified expression: {pb_path.name}")
        obs_df = meta_valid[[group_col, 'sample_type', 'sample_type_name']].copy()
        adata_pb = ad.AnnData(
            X=expr_valid.values.astype(np.float32),
            obs=obs_df,
            var=pd.DataFrame(index=expr_valid.columns),
        )
        adata_pb.uns['data_format'] = 'EBPlusPlus_RSEM_normalized_counts'
        adata_pb.uns['transform'] = 'clip(0) -> log2(RSEM+1)'
        adata_pb.uns['n_samples'] = expr_valid.shape[0]
        adata_pb.uns['n_groups'] = len(valid_groups)
        adata_pb.uns['group_col'] = group_col
        adata_pb.uns['primary_only'] = True
        adata_pb.write_h5ad(pb_path, compression='gzip')
        log(f"  Saved: {pb_path.name} ({adata_pb.shape})")
        del adata_pb
    else:
        log(f"SKIP expression (exists): {pb_path.name}")

    # Run activity inference per group
    gene_names = list(expr_valid.columns)
    expr_matrix = expr_valid.values
    n_samples = expr_matrix.shape[0]
    group_labels = meta_valid[group_col].values

    output_paths = {}

    for sig_name, sig_matrix in signatures.items():
        act_path = output_dir / f"{prefix}_{sig_name}.h5ad"

        if act_path.exists() and not force:
            log(f"  SKIP {sig_name} (exists): {act_path.name}")
            output_paths[sig_name] = act_path
            continue

        log(f"  Computing {sig_name} (stratified by {group_col})...")

        expr_genes = set(gene_names)
        sig_genes = set(sig_matrix.index)
        common = sorted(expr_genes & sig_genes)

        if len(common) < 100:
            log(f"    Too few common genes: {len(common)}, skipping")
            continue

        gene_idx = [gene_names.index(g) for g in common]
        X = sig_matrix.loc[common].values.copy()
        np.nan_to_num(X, copy=False, nan=0.0)

        n_targets = X.shape[1]
        activity = np.zeros((n_samples, n_targets), dtype=np.float32)

        t0 = time.time()

        for g in valid_groups:
            g_mask = (group_labels == g)
            n_g = g_mask.sum()
            if n_g < 3:
                continue

            Y_g = expr_matrix[g_mask][:, gene_idx].T.astype(np.float64)
            np.nan_to_num(Y_g, copy=False, nan=0.0)
            Y_g -= Y_g.mean(axis=1, keepdims=True)

            result_g = ridge(X, Y_g, lambda_=lambda_, n_rand=1000, backend=backend, verbose=False)
            activity[g_mask] = result_g['zscore'].T.astype(np.float32)

            del Y_g, result_g

        elapsed = time.time() - t0

        obs_df = meta_valid[[group_col, 'sample_type', 'sample_type_name']].copy()
        adata_act = ad.AnnData(
            X=activity,
            obs=obs_df,
            var=pd.DataFrame(index=list(sig_matrix.columns)),
        )
        adata_act.uns['common_genes'] = len(common)
        adata_act.uns['total_sig_genes'] = len(sig_genes)
        adata_act.uns['gene_coverage'] = len(common) / len(sig_genes)
        adata_act.uns['signature'] = sig_name
        adata_act.uns['source'] = f"{prefix}_expression.h5ad"
        adata_act.uns['data_format'] = 'EBPlusPlus_RSEM_normalized_counts'
        adata_act.uns['transform'] = 'clip(0) -> log2(RSEM+1)'
        adata_act.uns['group_col'] = group_col
        adata_act.uns['n_groups'] = len(valid_groups)
        adata_act.uns['stratified'] = True
        adata_act.uns['primary_only'] = True

        adata_act.write_h5ad(act_path, compression='gzip')
        output_paths[sig_name] = act_path
        log(f"    {sig_name}: {activity.shape}, {len(common)} genes, "
            f"{len(valid_groups)} groups, {elapsed:.1f}s -> {act_path.name}")

        del activity, adata_act, X
        gc.collect()

    return output_paths


# =============================================================================
# Main
# =============================================================================

SAMPLE_TYPE_NAMES = {
    '01': 'Primary Solid Tumor',
    '02': 'Recurrent Solid Tumor',
    '03': 'Primary Blood Derived Cancer',
    '05': 'Additional New Primary',
    '06': 'Metastatic',
    '07': 'Additional Metastatic',
    '11': 'Solid Tissue Normal',
}


def main():
    parser = argparse.ArgumentParser(
        description="TCGA Primary Tumor Filter + Activity Inference"
    )
    parser.add_argument('--force', action='store_true', help='Force overwrite existing files')
    parser.add_argument('--backend', default='auto', choices=['auto', 'numpy', 'cupy'],
                        help='Computation backend (default: auto)')
    args = parser.parse_args()

    # =========================================================================
    # Load existing donor_only expression
    # =========================================================================
    donor_expr_path = TCGA_DIR / 'tcga_donor_only_expression.h5ad'
    if not donor_expr_path.exists():
        log(f"ERROR: {donor_expr_path} not found. Run 15_bulk_validation.py first.")
        sys.exit(1)

    log(f"Loading {donor_expr_path.name}...")
    adata = ad.read_h5ad(donor_expr_path)
    log(f"  Shape: {adata.shape}")

    expr_df = pd.DataFrame(
        adata.X,
        index=adata.obs_names,
        columns=adata.var_names,
    )
    metadata_orig = adata.obs.copy()
    del adata
    gc.collect()

    # =========================================================================
    # Extract sample type from TCGA barcodes
    # =========================================================================
    log("Extracting sample types from TCGA barcodes...")
    sample_types = pd.Series(
        [extract_sample_type(b) for b in expr_df.index],
        index=expr_df.index,
        name='sample_type',
    )
    sample_type_names = sample_types.map(SAMPLE_TYPE_NAMES).fillna('Unknown')

    # Add to metadata
    metadata_orig['sample_type'] = sample_types
    metadata_orig['sample_type_name'] = sample_type_names

    log("Sample type distribution:")
    for st, count in sample_types.value_counts().sort_index().items():
        name = SAMPLE_TYPE_NAMES.get(st, 'Unknown')
        log(f"  {st} ({name}): {count}")

    # =========================================================================
    # Level 2: Primary Only (sample types 01 + 03)
    # =========================================================================
    log(f"\n{'=' * 60}")
    log("LEVEL 2: Primary Tumor + Blood Cancer Filter")
    log(f"{'=' * 60}")

    primary_mask = sample_types.isin(PRIMARY_SAMPLE_TYPES)
    expr_primary = expr_df.loc[primary_mask]
    meta_primary = metadata_orig.loc[primary_mask].copy()

    log(f"Filtered: {primary_mask.sum()} / {len(expr_df)} samples retained")
    log(f"  Primary Solid Tumor (01): {(sample_types == '01').sum()}")
    log(f"  Primary Blood Cancer (03): {(sample_types == '03').sum()}")

    # Check donor uniqueness
    donor_ids = pd.Series(
        ['-'.join(str(b).split('-')[:3]) for b in expr_primary.index],
        index=expr_primary.index,
    )
    n_donors = donor_ids.nunique()
    n_samples = len(expr_primary)
    log(f"Donors: {n_donors}, Samples: {n_samples} (ratio: {n_samples/n_donors:.2f})")

    # Save Level 2 expression
    pb_path = TCGA_DIR / 'tcga_primary_only_expression.h5ad'
    if not pb_path.exists() or args.force:
        log(f"Saving: {pb_path.name}")
        obs_df = meta_primary[['cancer_site', 'cancer_type', 'sample_type', 'sample_type_name']].copy()
        adata_pb = ad.AnnData(
            X=expr_primary.values.astype(np.float32),
            obs=obs_df,
            var=pd.DataFrame(index=expr_primary.columns),
        )
        adata_pb.uns['data_format'] = 'EBPlusPlus_RSEM_normalized_counts'
        adata_pb.uns['transform'] = 'clip(0) -> log2(RSEM+1)'
        adata_pb.uns['n_samples'] = n_samples
        adata_pb.uns['n_donors'] = n_donors
        adata_pb.uns['n_genes'] = expr_primary.shape[1]
        adata_pb.uns['primary_only'] = True
        adata_pb.uns['sample_types'] = list(PRIMARY_SAMPLE_TYPES)
        adata_pb.write_h5ad(pb_path, compression='gzip')
        log(f"  Saved: {pb_path.name} ({adata_pb.shape})")
        del adata_pb
    else:
        log(f"SKIP expression (exists): {pb_path.name}")

    # Run Level 2 activity inference
    log("\nRunning Level 2 activity inference (global mean-centering)...")
    obs_for_act = meta_primary[['cancer_site', 'cancer_type', 'sample_type', 'sample_type_name']].copy()
    run_activity_inference(
        expr_df=expr_primary,
        obs_df=obs_for_act,
        output_dir=TCGA_DIR,
        prefix='tcga_primary_only',
        force=args.force,
        backend=args.backend,
    )

    # =========================================================================
    # Level 3: Per-Cancer-Type from Level 2
    # =========================================================================
    log(f"\n{'=' * 60}")
    log("LEVEL 3: Per-Cancer-Type Stratified (from primary-only)")
    log(f"{'=' * 60}")

    run_stratified_activity_inference(
        expr_df=expr_primary,
        metadata=meta_primary,
        output_dir=TCGA_DIR,
        prefix='tcga_primary_by_cancer',
        group_col='cancer_type',
        force=args.force,
        backend=args.backend,
        min_samples=30,
    )

    # =========================================================================
    # Summary
    # =========================================================================
    log(f"\n{'=' * 60}")
    log("SUMMARY")
    log(f"{'=' * 60}")
    log(f"Level 1 (existing): tcga_donor_only_*       -> {len(expr_df)} samples")
    log(f"Level 2 (new):      tcga_primary_only_*     -> {len(expr_primary)} samples")
    log(f"Level 3 (new):      tcga_primary_by_cancer_* -> stratified from Level 2")
    log(f"\nAll files in: {TCGA_DIR}")

    import os
    for f in sorted(TCGA_DIR.glob('tcga_*.h5ad')):
        size_mb = os.path.getsize(f) / 1024 / 1024
        log(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
