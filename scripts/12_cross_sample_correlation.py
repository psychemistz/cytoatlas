#!/usr/bin/env python3
"""
Cross-Sample Correlation Validation Pipeline.

Generates donor-level pseudobulk expression and cytokine activity inference
at multiple aggregation levels for cross-sample correlation analysis.

The key question: how does sample-specific cytokine activity prediction
relate to pseudobulk gene expression of the cytokine/secreted protein?

Aggregation levels:
  - donor_only:    per donor (all cell types pooled)
  - donor_celltype: per (donor × cell type) at each annotation level
  - donor_organ:   per (donor × organ/tissue) for scAtlas
  - donor_organ_celltype: per (donor × organ × cell type) for scAtlas

For each pseudobulk, activity inference is run with 3 signature matrices:
  - CytoSig (44 cytokines)
  - LinCytoSig (178 cytokines)
  - SecAct (1,170 secreted proteins)

Usage:
    # All levels for one atlas
    python scripts/12_cross_sample_correlation.py --atlas cima

    # Specific levels
    python scripts/12_cross_sample_correlation.py --atlas cima --levels donor_only donor_l1

    # All atlases
    python scripts/12_cross_sample_correlation.py --atlas all

    # Force overwrite existing files
    python scripts/12_cross_sample_correlation.py --atlas cima --force
"""

import argparse
import gc
import gzip
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/vf/users/parks34/projects/1ridgesig/SecActpy")

from secactpy import load_cytosig, load_secact, ridge


def log(msg: str) -> None:
    """Print timestamped log message."""
    ts = time.strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


# =============================================================================
# Atlas Configuration
# =============================================================================

# Each grouping defines extra columns beyond sample_col to group by.
# None = donor-only (pool all cells per sample)
# ['col'] = donor × col
# ['col1', 'col2'] = donor × col1 × col2

ATLAS_CONFIGS = OrderedDict([
    ('cima', {
        'name': 'cima',
        'h5ad_path': '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/'
                     'CIMA_RNA_6484974cells_36326genes_compressed.h5ad',
        'sample_col': 'sample',
        'gene_col': None,  # var_names are gene symbols
        'groupings': OrderedDict([
            ('donor_only', []),
            ('donor_l1', ['cell_type_l1']),
            ('donor_l2', ['cell_type_l2']),
            ('donor_l3', ['cell_type_l3']),
            ('donor_l4', ['cell_type_l4']),
        ]),
    }),
    ('inflammation_main', {
        'name': 'inflammation_main',
        'h5ad_path': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/'
                     'INFLAMMATION_ATLAS_main_afterQC.h5ad',
        'sample_col': 'sampleID',
        'gene_col': 'symbol',  # var_names are Ensembl IDs
        'groupings': OrderedDict([
            ('donor_only', []),
            ('donor_l1', ['Level1']),
            ('donor_l2', ['Level2']),
        ]),
        'exclude_celltypes': {
            'Level1': ['Doublets', 'LowQuality_cells'],
        },
    }),
    ('inflammation_val', {
        'name': 'inflammation_val',
        'h5ad_path': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/'
                     'INFLAMMATION_ATLAS_validation_afterQC.h5ad',
        'sample_col': 'sampleID',
        'gene_col': 'symbol',
        'groupings': OrderedDict([
            ('donor_only', []),
            ('donor_l1', ['Level1pred']),
            ('donor_l2', ['Level2pred']),
        ]),
        'exclude_celltypes': {
            'Level1pred': ['Doublets', 'LowQuality_cells'],
        },
    }),
    ('inflammation_ext', {
        'name': 'inflammation_ext',
        'h5ad_path': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/'
                     'INFLAMMATION_ATLAS_external_afterQC.h5ad',
        'sample_col': 'sampleID',
        'gene_col': 'symbol',
        'groupings': OrderedDict([
            ('donor_only', []),
            ('donor_l1', ['Level1pred']),
            ('donor_l2', ['Level2pred']),
        ]),
        'exclude_celltypes': {
            'Level1pred': ['Doublets', 'LowQuality_cells'],
        },
    }),
    ('inflammation_combined', {
        'name': 'inflammation_combined',
        'h5ad_path': '/data/parks34/projects/2cytoatlas/results/inflammation_combined.h5ad',
        'sample_col': 'sampleID',
        'gene_col': 'symbol',
        'groupings': OrderedDict([
            ('donor_only', []),
            ('donor_l1', ['Level1']),
            ('donor_l2', ['Level2']),
        ]),
    }),
    ('scatlas_normal', {
        'name': 'scatlas_normal',
        'h5ad_path': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/'
                     'igt_s9_fine_counts.h5ad',
        'sample_col': 'sampleID',
        'gene_col': None,  # var_names are gene symbols
        'groupings': OrderedDict([
            ('donor_organ', ['tissue']),
            ('donor_organ_celltype1', ['tissue', 'cellType1']),
            ('donor_organ_celltype2', ['tissue', 'cellType2']),
        ]),
    }),
    ('scatlas_cancer', {
        'name': 'scatlas_cancer',
        'h5ad_path': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/'
                     'PanCancer_igt_s9_fine_counts.h5ad',
        'sample_col': 'sampleID',
        'gene_col': None,
        'groupings': OrderedDict([
            ('donor_cancertype', ['cancerType']),
            ('donor_cancertype_celltype1', ['cancerType', 'cellType1']),
            ('donor_cancertype_celltype2', ['cancerType', 'cellType2']),
        ]),
    }),
])

BASE_OUTPUT_DIR = Path('/data/parks34/projects/2cytoatlas/results/cross_sample_validation')


# =============================================================================
# Signature Loading
# =============================================================================

_SIGNATURES_CACHE = None

def load_signatures() -> Dict[str, pd.DataFrame]:
    """Load all signature matrices (cached)."""
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
    log(f"  CytoSig: {cytosig.shape}, LinCytoSig: {lincytosig.shape}, SecAct: {secact.shape}")
    return _SIGNATURES_CACHE


# =============================================================================
# Pseudobulk Generation (Multi-Level Single-Pass)
# =============================================================================

def generate_pseudobulk_multipass(
    config: Dict,
    output_dir: Path,
    levels: Optional[List[str]] = None,
    min_cells: int = 10,
    batch_size: int = 50000,
    force: bool = False,
) -> Dict[str, Tuple[np.ndarray, List[str], pd.DataFrame, Optional[List[str]]]]:
    """
    Generate donor-level pseudobulk for multiple grouping levels in a single
    pass through the H5AD file.

    Returns:
        Dict mapping level_name -> (expr_matrix, gene_names, obs_df, gene_symbols)
    """
    atlas_name = config['name']
    h5ad_path = config['h5ad_path']
    sample_col = config['sample_col']
    gene_col = config['gene_col']

    # Determine which levels to process
    all_groupings = config['groupings']
    if levels is not None:
        groupings = OrderedDict((k, v) for k, v in all_groupings.items() if k in levels)
    else:
        groupings = all_groupings

    # Check which levels need generation
    levels_to_generate = OrderedDict()
    for level_name, group_cols in groupings.items():
        pb_path = output_dir / f"{atlas_name}_{level_name}_pseudobulk.h5ad"
        if pb_path.exists() and not force:
            log(f"SKIP pseudobulk (exists): {pb_path.name}")
        else:
            levels_to_generate[level_name] = group_cols

    if not levels_to_generate:
        log("All pseudobulk files exist, loading from disk...")
        results = {}
        for level_name in groupings:
            pb_path = output_dir / f"{atlas_name}_{level_name}_pseudobulk.h5ad"
            adata_pb = ad.read_h5ad(pb_path)
            gene_names = list(adata_pb.var_names)
            gene_symbols = None
            if gene_col and gene_col in adata_pb.var.columns:
                gene_symbols = adata_pb.var[gene_col].tolist()
            results[level_name] = (adata_pb.X, gene_names, adata_pb.obs, gene_symbols)
        return results

    log("=" * 70)
    log(f"PSEUDOBULK GENERATION: {atlas_name}")
    log(f"Levels to generate: {list(levels_to_generate.keys())}")
    log("=" * 70)

    # Open H5AD
    log(f"Opening: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path, backed='r')
    n_cells, n_genes = adata.shape
    log(f"Shape: {n_cells:,} cells x {n_genes:,} genes")

    # Gene info
    gene_names = list(adata.var_names)
    var_df = adata.var.copy()
    gene_symbols = None
    if gene_col and gene_col in var_df.columns:
        gene_symbols = var_df[gene_col].tolist()
        log(f"Using var['{gene_col}'] for gene symbols ({len(gene_symbols)} genes)")

    # Load metadata columns needed across all levels
    all_cols = {sample_col}
    for group_cols in levels_to_generate.values():
        all_cols.update(group_cols)
    all_cols = sorted(all_cols)
    log(f"Loading metadata columns: {all_cols}")
    obs_df = adata.obs[list(all_cols)].copy()

    # Exclude unwanted cell types (e.g., Doublets, LowQuality_cells)
    # Mark excluded cells by setting sample_col to NaN so they become __INVALID__
    exclude_map = config.get('exclude_celltypes', {})
    if exclude_map:
        exclude_cols_needed = [c for c in exclude_map.keys() if c not in obs_df.columns]
        for ec in exclude_cols_needed:
            obs_df[ec] = adata.obs[ec].values

        exclude_mask = pd.Series(False, index=obs_df.index)
        for col, values in exclude_map.items():
            if col in obs_df.columns:
                col_mask = obs_df[col].isin(values)
                exclude_mask |= col_mask
                excluded_types = obs_df.loc[col_mask, col].value_counts()
                for ct, cnt in excluded_types.items():
                    log(f"  Excluding {cnt:,} cells with {col}='{ct}'")

        n_excluded = exclude_mask.sum()
        if n_excluded > 0:
            obs_df.loc[exclude_mask, sample_col] = np.nan
            log(f"  Total excluded: {n_excluded:,} cells ({n_excluded/len(obs_df)*100:.1f}%)")

    # Build group keys for each level
    level_group_keys = {}
    for level_name, group_cols in levels_to_generate.items():
        if not group_cols:
            # Donor-only: key is just the sample col
            keys = obs_df[sample_col].astype(str).values
        else:
            # Donor × other cols: join with '__'
            parts = [obs_df[sample_col].astype(str)]
            for col in group_cols:
                parts.append(obs_df[col].astype(str))
            keys = np.array(['__'.join(vals) for vals in zip(*[p.values for p in parts])])

        # Mark invalid (nan) entries
        has_nan = obs_df[sample_col].isna()
        for col in group_cols:
            has_nan = has_nan | obs_df[col].isna()
        keys[has_nan.values] = '__INVALID__'

        level_group_keys[level_name] = keys
        n_valid = (keys != '__INVALID__').sum()
        n_unique = len(set(keys) - {'__INVALID__'})
        log(f"  {level_name}: {n_unique:,} groups ({n_valid:,} valid cells)")

    # Pre-encode group keys to integer indices for vectorized accumulation
    level_indices = {}  # level_name -> (idx_array, unique_keys)
    accumulators = {}

    for level_name in levels_to_generate:
        keys = level_group_keys[level_name]
        unique_keys = sorted(set(keys) - {'__INVALID__'})
        key_to_idx = {k: i for i, k in enumerate(unique_keys)}
        idx_array = np.array([key_to_idx.get(k, -1) for k in keys], dtype=np.int32)
        level_indices[level_name] = (idx_array, unique_keys)

        # Pre-allocate 2D accumulators (n_groups × n_genes)
        n_groups = len(unique_keys)
        accumulators[level_name] = {
            'sum': np.zeros((n_groups, n_genes), dtype=np.float64),
            'count': np.zeros(n_groups, dtype=np.int64),
        }
        log(f"  {level_name}: allocated {n_groups} x {n_genes} accumulator "
            f"({n_groups * n_genes * 8 / 1024**3:.1f} GB)")

    # Process in batches using vectorized sparse indicator matrix multiply
    n_batches = (n_cells + batch_size - 1) // batch_size
    log(f"Processing {n_batches} batches (batch_size={batch_size:,})...")
    t0 = time.time()

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_cells)
        n_batch = end - start

        # Load raw counts (do NOT normalize per-cell; accumulate raw sums)
        X_batch = adata.X[start:end]
        if sparse.issparse(X_batch):
            X_batch = X_batch.toarray()
        X_batch = X_batch.astype(np.float64)

        # Vectorized accumulation for each level using sparse indicator × dense
        for level_name in levels_to_generate:
            batch_idx_arr = level_indices[level_name][0][start:end]
            valid = batch_idx_arr >= 0
            n_valid = valid.sum()
            if n_valid == 0:
                continue

            n_groups = len(level_indices[level_name][1])
            rows = batch_idx_arr[valid]
            cols = np.arange(n_batch)[valid]
            indicator = sparse.csr_matrix(
                (np.ones(n_valid, dtype=np.float64), (rows, cols)),
                shape=(n_groups, n_batch),
            )
            # (n_groups × n_batch) @ (n_batch × n_genes) -> (n_groups × n_genes)
            accumulators[level_name]['sum'] += indicator @ X_batch
            accumulators[level_name]['count'] += np.array(indicator.sum(axis=1), dtype=np.int64).ravel()

        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - t0
            rate = (batch_idx + 1) / elapsed
            eta = (n_batches - batch_idx - 1) / rate
            log(f"  Batch {batch_idx + 1}/{n_batches} ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

        del X_batch
        gc.collect()

    del adata
    gc.collect()

    # Build pseudobulk AnnData for each level
    results = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for level_name, group_cols in levels_to_generate.items():
        idx_array, unique_keys = level_indices[level_name]
        acc = accumulators[level_name]
        counts = acc['count']
        sums = acc['sum']

        # Filter groups by min_cells
        valid_mask = counts >= min_cells
        n_total = len(unique_keys)
        n_valid = valid_mask.sum()
        n_filtered = n_total - n_valid

        log(f"\n{level_name}: {n_valid:,} groups (>= {min_cells} cells), "
            f"{n_filtered} filtered out")

        if n_valid == 0:
            log(f"  WARNING: No valid groups for {level_name}, skipping")
            continue

        # Build pseudobulk: sum raw counts -> CPM normalize -> log1p
        valid_indices = np.where(valid_mask)[0]
        raw_sums = sums[valid_indices]  # (n_groups x n_genes), raw count sums
        valid_keys = [unique_keys[i] for i in valid_indices]
        valid_counts = counts[valid_indices]

        # CPM normalize each pseudobulk sample
        row_totals = raw_sums.sum(axis=1, keepdims=True)
        row_totals[row_totals == 0] = 1.0
        expr_matrix = (raw_sums / row_totals * 1e6)
        # log1p transform
        np.log1p(expr_matrix, out=expr_matrix)
        expr_matrix = expr_matrix.astype(np.float32)

        # Parse group keys back into obs columns
        obs_data = {}
        if not group_cols:
            # Donor-only
            obs_data['donor'] = valid_keys
        else:
            # Split key: donor__col1__col2...
            split_keys = [g.split('__') for g in valid_keys]
            obs_data['donor'] = [s[0] for s in split_keys]
            for ci, col in enumerate(group_cols):
                obs_data[col] = [s[ci + 1] for s in split_keys]

        obs_data['n_cells'] = valid_counts.tolist()
        obs_df_out = pd.DataFrame(obs_data, index=valid_keys)

        # Preserve var info
        if gene_symbols is not None:
            var_out = var_df.copy()
        else:
            var_out = pd.DataFrame(index=gene_names)

        adata_pb = ad.AnnData(X=expr_matrix, obs=obs_df_out, var=var_out)

        pb_path = output_dir / f"{atlas_name}_{level_name}_pseudobulk.h5ad"
        log(f"  Saving: {pb_path.name} ({adata_pb.shape})")
        adata_pb.write_h5ad(pb_path, compression='gzip')

        results[level_name] = (expr_matrix, gene_names, obs_df_out, gene_symbols)

    # Also load existing pseudobulk for levels we skipped
    for level_name in groupings:
        if level_name not in results:
            pb_path = output_dir / f"{atlas_name}_{level_name}_pseudobulk.h5ad"
            if pb_path.exists():
                adata_pb = ad.read_h5ad(pb_path)
                gn = list(adata_pb.var_names)
                gs = None
                if gene_col and gene_col in adata_pb.var.columns:
                    gs = adata_pb.var[gene_col].tolist()
                results[level_name] = (adata_pb.X, gn, adata_pb.obs, gs)

    return results


# =============================================================================
# Activity Inference
# =============================================================================

def _get_celltype_col(obs_df: pd.DataFrame) -> Optional[str]:
    """Detect the celltype grouping column in obs, if any.

    For donor-only levels obs has only ['donor', 'n_cells'] -> returns None.
    For celltype-stratified levels, returns the last non-donor/non-n_cells column
    (e.g. 'cell_type_l1', 'Level1', 'cellType1', 'tissue').
    """
    grouping_cols = [c for c in obs_df.columns if c not in ('donor', 'n_cells')]
    if not grouping_cols:
        return None
    # Return last grouping column (most specific celltype level)
    return grouping_cols[-1]


def run_activity_inference(
    expr_matrix: np.ndarray,
    gene_names: List[str],
    obs_df: pd.DataFrame,
    output_dir: Path,
    atlas_name: str,
    level_name: str,
    gene_symbols: Optional[List[str]] = None,
    force: bool = False,
    lambda_: float = 5e5,
    backend: str = 'auto',
) -> Dict[str, Path]:
    """
    Run CytoSig/LinCytoSig/SecAct activity inference on pseudobulk.

    For donor-only levels: mean-center across all donors, run ridge once.
    For celltype-stratified levels: split by celltype, mean-center within
    each celltype, run ridge per celltype, reassemble.

    Returns:
        Dict mapping signature_name -> output_path
    """
    signatures = load_signatures()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use gene symbols for matching if available
    matching_genes = gene_symbols if gene_symbols is not None else gene_names

    # Detect celltype column for within-celltype normalization
    ct_col = _get_celltype_col(obs_df)
    if ct_col is not None:
        log(f"  Within-celltype normalization on '{ct_col}'")

    output_paths = {}

    for sig_name, sig_matrix in signatures.items():
        act_path = output_dir / f"{atlas_name}_{level_name}_{sig_name}.h5ad"

        if act_path.exists() and not force:
            log(f"  SKIP {sig_name} (exists): {act_path.name}")
            output_paths[sig_name] = act_path
            continue

        log(f"  Computing {sig_name}...")

        # Find common genes
        expr_genes = set(matching_genes)
        sig_genes = set(sig_matrix.index)
        common = sorted(expr_genes & sig_genes)

        if len(common) < 100:
            log(f"    Too few common genes: {len(common)}, skipping")
            continue

        # Align expression to common genes
        gene_idx = [matching_genes.index(g) for g in common]
        X = sig_matrix.loc[common].values.copy()  # (genes x targets)
        np.nan_to_num(X, copy=False, nan=0.0)

        n_samples = expr_matrix.shape[0]
        n_targets = X.shape[1]
        activity = np.zeros((n_samples, n_targets), dtype=np.float32)

        t0 = time.time()

        if ct_col is None:
            # Donor-only: mean-center across all donors, run ridge once
            Y = expr_matrix[:, gene_idx].T.astype(np.float64)  # (genes x samples)
            Y -= Y.mean(axis=1, keepdims=True)
            result = ridge(X, Y, lambda_=lambda_, n_rand=1000, backend=backend, verbose=False)
            activity[:] = result['zscore'].T.astype(np.float32)
            del Y, result
        else:
            # Celltype-stratified: run ridge per celltype separately
            celltypes = obs_df[ct_col].unique()
            for ct in celltypes:
                ct_mask = (obs_df[ct_col] == ct).values
                n_ct = ct_mask.sum()
                if n_ct < 3:
                    continue

                Y_ct = expr_matrix[ct_mask][:, gene_idx].T.astype(np.float64)
                Y_ct -= Y_ct.mean(axis=1, keepdims=True)

                result_ct = ridge(X, Y_ct, lambda_=lambda_, n_rand=1000, backend=backend, verbose=False)
                activity[ct_mask] = result_ct['zscore'].T.astype(np.float32)

                del Y_ct, result_ct

        elapsed = time.time() - t0

        # Create AnnData
        adata_act = ad.AnnData(
            X=activity,
            obs=obs_df.copy(),
            var=pd.DataFrame(index=list(sig_matrix.columns)),
        )
        adata_act.uns['common_genes'] = len(common)
        adata_act.uns['total_sig_genes'] = len(sig_genes)
        adata_act.uns['gene_coverage'] = len(common) / len(sig_genes)
        adata_act.uns['signature'] = sig_name
        adata_act.uns['source_pseudobulk'] = f"{atlas_name}_{level_name}_pseudobulk.h5ad"
        if ct_col is not None:
            adata_act.uns['celltype_col'] = ct_col

        adata_act.write_h5ad(act_path, compression='gzip')
        output_paths[sig_name] = act_path
        log(f"    {sig_name}: {activity.shape}, {len(common)} genes, {elapsed:.1f}s -> {act_path.name}")

        del activity, adata_act, X
        gc.collect()

    return output_paths


# =============================================================================
# Main Pipeline
# =============================================================================

def run_atlas(
    atlas_name: str,
    levels: Optional[List[str]] = None,
    force: bool = False,
    batch_size: int = 50000,
    min_cells: int = 10,
    backend: str = 'auto',
) -> None:
    """Run full cross-sample correlation pipeline for one atlas."""
    if atlas_name not in ATLAS_CONFIGS:
        raise ValueError(f"Unknown atlas: {atlas_name}. "
                         f"Available: {list(ATLAS_CONFIGS.keys())}")

    config = ATLAS_CONFIGS[atlas_name]
    output_dir = BASE_OUTPUT_DIR / atlas_name

    log("\n" + "=" * 70)
    log(f"CROSS-SAMPLE CORRELATION PIPELINE: {atlas_name}")
    log("=" * 70)

    # Validate levels
    available_levels = list(config['groupings'].keys())
    if levels:
        invalid = set(levels) - set(available_levels)
        if invalid:
            raise ValueError(f"Invalid levels for {atlas_name}: {invalid}. "
                             f"Available: {available_levels}")
    else:
        levels = available_levels

    log(f"Levels: {levels}")
    log(f"Output: {output_dir}")

    # Step 1: Generate pseudobulk (single pass through H5AD)
    log("\n--- STEP 1: Pseudobulk Generation ---")
    pseudobulk_data = generate_pseudobulk_multipass(
        config=config,
        output_dir=output_dir,
        levels=levels,
        min_cells=min_cells,
        batch_size=batch_size,
        force=force,
    )

    # Step 2: Activity inference on each pseudobulk
    log("\n--- STEP 2: Activity Inference ---")
    for level_name in levels:
        if level_name not in pseudobulk_data:
            log(f"Skipping activity for {level_name} (no pseudobulk)")
            continue

        expr_matrix, gene_names, obs_df, gene_symbols = pseudobulk_data[level_name]
        log(f"\n{level_name}: {expr_matrix.shape[0]} samples x {expr_matrix.shape[1]} genes")

        run_activity_inference(
            expr_matrix=expr_matrix if isinstance(expr_matrix, np.ndarray) else np.asarray(expr_matrix),
            gene_names=gene_names,
            obs_df=obs_df,
            output_dir=output_dir,
            atlas_name=atlas_name,
            level_name=level_name,
            gene_symbols=gene_symbols,
            force=force,
        )

    log("\n" + "=" * 70)
    log(f"COMPLETE: {atlas_name}")
    log("=" * 70)

    # Summary
    log("\nOutput files:")
    for f in sorted(output_dir.glob("*.h5ad")):
        size_mb = f.stat().st_size / 1024**2
        log(f"  {f.name} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Sample Correlation Validation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/12_cross_sample_correlation.py --atlas cima
  python scripts/12_cross_sample_correlation.py --atlas cima --levels donor_only donor_l1
  python scripts/12_cross_sample_correlation.py --atlas all
  python scripts/12_cross_sample_correlation.py --atlas inflammation_main inflammation_val inflammation_ext
        """,
    )
    parser.add_argument(
        '--atlas', nargs='+', required=True,
        help='Atlas name(s) or "all"',
    )
    parser.add_argument(
        '--levels', nargs='+', default=None,
        help='Specific grouping levels to process (default: all for each atlas)',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force overwrite existing files',
    )
    parser.add_argument(
        '--batch-size', type=int, default=50000,
        help='Batch size for reading H5AD (default: 50000)',
    )
    parser.add_argument(
        '--min-cells', type=int, default=10,
        help='Minimum cells per group (default: 10)',
    )
    parser.add_argument(
        '--backend', default='auto',
        choices=['auto', 'numpy', 'cupy'],
        help='Ridge regression backend (default: auto)',
    )

    args = parser.parse_args()

    # Resolve atlas list
    if 'all' in args.atlas:
        atlas_list = list(ATLAS_CONFIGS.keys())
    else:
        atlas_list = args.atlas

    t_start = time.time()

    for atlas_name in atlas_list:
        run_atlas(
            atlas_name=atlas_name,
            levels=args.levels,
            force=args.force,
            batch_size=args.batch_size,
            min_cells=args.min_cells,
            backend=args.backend,
        )

    elapsed = time.time() - t_start
    log(f"\n{'=' * 70}")
    log(f"ALL DONE ({elapsed / 60:.1f} min)")
    log(f"{'=' * 70}")


if __name__ == '__main__':
    main()
