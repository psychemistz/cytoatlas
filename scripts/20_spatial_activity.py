#!/usr/bin/env python3
"""
SpatialCorpus-110M Activity Inference
======================================
Compute cytokine (CytoSig) and secreted protein (SecAct) activity signatures
from the SpatialCorpus-110M collection (251 H5AD files across multiple spatial
transcriptomics technologies).

Technology tiers:
  A (Visium): Full genome-wide panels -> run full CytoSig + SecAct
  B (Xenium/MERFISH/MERSCOPE/CosMx): Targeted gene panels -> score only
    signatures with >50% gene coverage
  C (ISS/mouse): Skip entirely

Species filter: Mouse files are excluded (human-only analysis).

Pseudobulk aggregation: file x tissue x cell_type (where annotated).

Output:
  - spatial_activity_by_technology.h5ad   (activity z-scores per file)
  - spatial_activity_by_tissue.csv        (summary by tissue)
  - spatial_technology_comparison.csv     (cross-technology comparison)

Usage:
    # All technologies
    python scripts/20_spatial_activity.py --technology all

    # Visium only (Tier A)
    python scripts/20_spatial_activity.py --technology visium

    # Targeted panels only (Tier B)
    python scripts/20_spatial_activity.py --technology targeted

    # Test mode (first 3 files per tier)
    python scripts/20_spatial_activity.py --test

    # Force numpy backend
    python scripts/20_spatial_activity.py --backend numpy
"""

import os
import sys
import gc
import warnings
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats

warnings.filterwarnings('ignore')

# Add SecActpy to path
sys.path.insert(0, '/data/parks34/projects/1ridgesig/SecActpy')
from secactpy import (
    load_cytosig, load_secact,
    ridge_batch, estimate_batch_size,
    CUPY_AVAILABLE
)

# ==============================================================================
# Configuration
# ==============================================================================

DATA_DIR = Path('/data/Jiang_Lab/Data/Seongyong/SpatialCorpus-110M')
OUTPUT_DIR = Path('/data/parks34/projects/2cytoatlas/results/spatial')

# Activity computation parameters
N_RAND = 1000
SEED = 0
LAMBDA = 5e5
MIN_GENE_COVERAGE = 0.50  # For Tier B: require 50% gene overlap
CHUNK_SIZE = 50000  # Cells per chunk for backed-mode reading

# Technology classification keywords
TIER_A_KEYWORDS = ['visium']
TIER_B_KEYWORDS = ['xenium', 'merfish', 'merscope', 'cosmx', 'nanostring']
TIER_C_KEYWORDS = ['iss']
MOUSE_KEYWORDS = ['mouse', '_mouse_']

# GPU settings
BACKEND = 'cupy' if CUPY_AVAILABLE else 'numpy'


# ==============================================================================
# Utility Functions
# ==============================================================================

def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def classify_technology(filename: str) -> str:
    """Classify a spatial H5AD file into technology tier.

    Returns:
        'A' for Visium (genome-wide), 'B' for targeted panels,
        'C' for ISS/mouse (skip), or 'skip' for mouse files.
    """
    fname_lower = filename.lower()

    # Skip mouse files
    for kw in MOUSE_KEYWORDS:
        if kw in fname_lower:
            return 'skip'

    # Tier A: Visium (genome-wide)
    for kw in TIER_A_KEYWORDS:
        if kw in fname_lower:
            return 'A'

    # Tier C: ISS (skip)
    for kw in TIER_C_KEYWORDS:
        if kw in fname_lower:
            return 'C'

    # Tier B: Targeted panels
    for kw in TIER_B_KEYWORDS:
        if kw in fname_lower:
            return 'B'

    # GSE/sfaira datasets are typically Visium or similar genome-wide
    if fname_lower.startswith('gse') or fname_lower.startswith('sfaira'):
        return 'A'

    # Default: treat as Tier A (genome-wide)
    return 'A'


def get_technology_name(filename: str) -> str:
    """Extract readable technology name from filename."""
    fname_lower = filename.lower()
    if 'visium' in fname_lower:
        return 'Visium'
    elif 'xenium' in fname_lower:
        return 'Xenium'
    elif 'merfish' in fname_lower:
        return 'MERFISH'
    elif 'merscope' in fname_lower:
        return 'MERSCOPE'
    elif 'cosmx' in fname_lower or 'nanostring' in fname_lower:
        return 'CosMx'
    elif 'iss' in fname_lower:
        return 'ISS'
    else:
        return 'Other'


def discover_files() -> Dict[str, List[Path]]:
    """Discover and classify all H5AD files in the spatial corpus.

    Returns:
        Dict mapping tier ('A', 'B', 'C', 'skip') to list of file paths.
    """
    log(f"Scanning {DATA_DIR} for H5AD files...")
    tiers = {'A': [], 'B': [], 'C': [], 'skip': []}

    if not DATA_DIR.exists():
        log(f"ERROR: Data directory not found: {DATA_DIR}")
        return tiers

    h5ad_files = sorted(DATA_DIR.glob('*.h5ad'))
    log(f"  Found {len(h5ad_files)} H5AD files")

    for f in h5ad_files:
        tier = classify_technology(f.name)
        tiers[tier].append(f)

    for tier_name, files in tiers.items():
        log(f"  Tier {tier_name}: {len(files)} files")

    return tiers


def compute_gene_panel_coverage(
    gene_names: List[str],
    sig_matrix: pd.DataFrame,
) -> Tuple[float, List[str]]:
    """Compute overlap between a gene panel and signature matrix genes.

    Args:
        gene_names: Gene names in the spatial dataset.
        sig_matrix: Signature matrix (genes x signatures).

    Returns:
        (coverage_fraction, common_genes)
    """
    panel_genes = set(g.upper() for g in gene_names)
    sig_genes = set(g.upper() for g in sig_matrix.index)
    common = sorted(panel_genes & sig_genes)
    coverage = len(common) / len(sig_genes) if len(sig_genes) > 0 else 0.0
    return coverage, common


def detect_metadata_columns(adata: ad.AnnData) -> Dict[str, Optional[str]]:
    """Detect tissue and cell type columns from obs metadata.

    Returns:
        Dict with keys 'tissue', 'cell_type', 'sample' mapping to column names.
    """
    obs_cols = list(adata.obs.columns)
    result = {'tissue': None, 'cell_type': None, 'sample': None}

    # Tissue column
    for col in ['tissue', 'tissue_type', 'organ', 'region', 'Tissue', 'tissue_id']:
        if col in obs_cols:
            result['tissue'] = col
            break

    # Cell type column
    for col in ['cell_type', 'celltype', 'cell_type_l1', 'cellType1',
                'subCluster', 'cluster', 'leiden', 'annotation', 'CellType']:
        if col in obs_cols:
            result['cell_type'] = col
            break

    # Sample column
    for col in ['sample', 'sample_id', 'donor', 'donor_id', 'patient',
                'batch', 'library_id']:
        if col in obs_cols:
            result['sample'] = col
            break

    return result


# ==============================================================================
# Pseudobulk Aggregation
# ==============================================================================

def aggregate_pseudobulk(
    adata: ad.AnnData,
    filename: str,
    meta_cols: Dict[str, Optional[str]],
    min_cells: int = 10,
    chunk_size: int = CHUNK_SIZE,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Aggregate single cells into pseudobulk profiles using chunked reading.

    Groups by file x tissue x cell_type (where annotated). Falls back to
    file-level aggregation if no tissue/cell_type columns found.

    Reads expression data in chunks to avoid loading the full dense matrix,
    which is critical for large Visium files (millions of spots x 15K+ genes).

    Args:
        adata: AnnData with expression data (can be backed).
        filename: Source filename for grouping.
        meta_cols: Dict of detected metadata column names.
        min_cells: Minimum cells per group.
        chunk_size: Number of cells to read per chunk.

    Returns:
        (expr_df, meta_df) where expr_df is genes x groups, meta_df has group info.
    """
    n_cells = adata.n_obs
    n_genes = adata.n_vars

    # Build grouping key from obs (fast even for backed mode)
    obs = adata.obs.copy()
    obs = obs.reset_index(drop=True)
    obs['_file'] = filename

    tissue_col = meta_cols.get('tissue')
    ct_col = meta_cols.get('cell_type')

    grouping_parts = ['_file']
    if tissue_col is not None and tissue_col in obs.columns:
        obs['_tissue'] = obs[tissue_col].astype(str).fillna('Unknown')
        grouping_parts.append('_tissue')
    if ct_col is not None and ct_col in obs.columns:
        obs['_celltype'] = obs[ct_col].astype(str).fillna('Unknown')
        grouping_parts.append('_celltype')

    obs['_group'] = obs[grouping_parts].astype(str).agg('__'.join, axis=1)
    group_labels = obs['_group'].values
    unique_groups = np.unique(group_labels)
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    obs['_group_idx'] = obs['_group'].map(group_to_idx)

    n_groups = len(unique_groups)
    log(f"    {n_groups} pseudobulk groups from {n_cells:,} cells")

    # Use HGNC gene symbols from 'feature_name' column if available
    # (SpatialCorpus H5AD files store Ensembl IDs in var_names but HGNC symbols
    # in var['feature_name'] — signature matrices use HGNC symbols)
    if 'feature_name' in adata.var.columns:
        gene_names = list(adata.var['feature_name'].astype(str))
    elif 'gene_name' in adata.var.columns:
        gene_names = list(adata.var['gene_name'].astype(str))
    else:
        gene_names = list(adata.var_names)

    # Initialize accumulators (sum + count per group)
    group_sums = np.zeros((n_groups, n_genes), dtype=np.float64)
    group_counts = np.zeros(n_groups, dtype=np.int64)

    # Get expression matrix handle (works for backed and in-memory)
    X = adata.X

    # Process in chunks to avoid loading full dense matrix
    n_chunks = (n_cells + chunk_size - 1) // chunk_size
    log(f"    Processing {n_cells:,} cells in {n_chunks} chunks of {chunk_size:,}...")

    for chunk_i in range(n_chunks):
        start = chunk_i * chunk_size
        end = min((chunk_i + 1) * chunk_size, n_cells)

        # Read chunk from disk (sparse or dense)
        chunk_X = X[start:end, :]
        if hasattr(chunk_X, 'toarray'):
            chunk_X = chunk_X.toarray()
        chunk_X = np.asarray(chunk_X, dtype=np.float64)

        # Get group indices for this chunk
        chunk_groups = obs['_group_idx'].iloc[start:end].values

        # Accumulate sums per group
        for local_i in range(chunk_X.shape[0]):
            g_idx = chunk_groups[local_i]
            group_sums[g_idx, :] += chunk_X[local_i, :]
            group_counts[g_idx] += 1

        del chunk_X

        if (chunk_i + 1) % 10 == 0 or chunk_i == n_chunks - 1:
            log(f"      Chunk {chunk_i + 1}/{n_chunks} ({end:,} cells)")

    # Build mean expression per group, filtering by min_cells
    meta_rows = []
    expr_list = []

    for g_idx, grp_name in enumerate(unique_groups):
        if group_counts[g_idx] < min_cells:
            continue

        mean_expr = group_sums[g_idx, :] / group_counts[g_idx]
        expr_list.append(mean_expr)

        parts = grp_name.split('__')
        row = {
            'group': grp_name,
            'file': filename,
            'n_cells': int(group_counts[g_idx]),
        }
        if len(parts) >= 2:
            row['tissue'] = parts[1]
        if len(parts) >= 3:
            row['cell_type'] = parts[2]

        meta_rows.append(row)

    del group_sums, group_counts
    gc.collect()

    if len(expr_list) == 0:
        return None, None

    expr_matrix = np.array(expr_list)  # (groups x genes)
    expr_df = pd.DataFrame(expr_matrix.T, index=gene_names,
                           columns=[r['group'] for r in meta_rows])

    # Handle duplicate gene names (multiple Ensembl IDs -> same HGNC symbol)
    if expr_df.index.duplicated().any():
        n_dups = expr_df.index.duplicated().sum()
        log(f"    Deduplicating {n_dups} duplicate gene names (averaging)")
        expr_df = expr_df.groupby(expr_df.index).mean()

    meta_df = pd.DataFrame(meta_rows).set_index('group')

    return expr_df, meta_df


# ==============================================================================
# Activity Inference
# ==============================================================================

def run_activity_for_file(
    h5ad_path: Path,
    sig_matrices: Dict[str, pd.DataFrame],
    tier: str,
    backend: str = 'auto',
) -> Optional[Dict]:
    """Run activity inference on a single spatial H5AD file.

    For Tier A: run all signature matrices.
    For Tier B: only run signatures with >50% gene coverage.

    Args:
        h5ad_path: Path to H5AD file.
        sig_matrices: Dict of signature name -> DataFrame.
        tier: 'A' or 'B'.
        backend: Computation backend.

    Returns:
        Dict with activity results and metadata, or None if skipped.
    """
    filename = h5ad_path.stem
    technology = get_technology_name(h5ad_path.name)

    log(f"\n  Processing: {h5ad_path.name} (Tier {tier}, {technology})")
    t0 = time.time()

    try:
        adata = ad.read_h5ad(h5ad_path, backed='r')
    except Exception as e:
        log(f"    ERROR reading {h5ad_path.name}: {e}")
        return None

    n_cells = adata.n_obs
    n_genes = adata.n_vars
    log(f"    Shape: {n_cells} cells x {n_genes} genes")

    if n_cells < 50:
        log(f"    SKIP: too few cells ({n_cells})")
        adata.file.close() if hasattr(adata, 'file') and adata.file is not None else None
        return None

    # Detect metadata columns
    meta_cols = detect_metadata_columns(adata)
    log(f"    Metadata: tissue={meta_cols['tissue']}, "
        f"cell_type={meta_cols['cell_type']}, sample={meta_cols['sample']}")

    # Pseudobulk aggregation using chunked reading (no to_memory() needed)
    expr_df, meta_df = aggregate_pseudobulk(adata, filename, meta_cols)

    # Close backed file handle
    try:
        if hasattr(adata, 'file') and adata.file is not None:
            adata.file.close()
    except Exception:
        pass
    del adata
    gc.collect()

    if expr_df is None:
        log(f"    SKIP: no valid pseudobulk groups")
        return None

    # Uppercase gene index for case-insensitive matching
    expr_upper = expr_df.copy()
    expr_upper.index = expr_upper.index.str.upper()
    if expr_upper.index.duplicated().any():
        expr_upper = expr_upper.groupby(expr_upper.index).mean()
    gene_names = list(expr_upper.index)

    # Run activity inference per signature matrix
    activity_results = {}

    for sig_name, sig_matrix in sig_matrices.items():
        coverage, common_genes = compute_gene_panel_coverage(gene_names, sig_matrix)

        # Tier B: require minimum gene coverage
        if tier == 'B' and coverage < MIN_GENE_COVERAGE:
            log(f"    {sig_name}: coverage {coverage:.1%} < {MIN_GENE_COVERAGE:.0%}, skipping")
            continue

        if len(common_genes) < 50:
            log(f"    {sig_name}: too few common genes ({len(common_genes)}), skipping")
            continue

        log(f"    {sig_name}: {len(common_genes)} common genes ({coverage:.1%} coverage)")

        # Align signature matrix (uppercase + dedup)
        sig_aligned = sig_matrix.copy()
        sig_aligned.index = sig_aligned.index.str.upper()
        sig_aligned = sig_aligned[~sig_aligned.index.duplicated(keep='first')]

        # Recompute common after alignment
        common_genes = sorted(set(expr_upper.index) & set(sig_aligned.index))

        # Prepare matrices
        X = sig_aligned.loc[common_genes].values.copy()
        np.nan_to_num(X, copy=False, nan=0.0)

        Y = expr_upper.loc[common_genes].values.astype(np.float64)  # (genes x samples)
        np.nan_to_num(Y, copy=False, nan=0.0)
        Y -= Y.mean(axis=1, keepdims=True)

        try:
            n_samples = Y.shape[1]
            if n_samples > 1000:
                result = ridge_batch(
                    X, Y, lambda_=LAMBDA, n_rand=N_RAND, seed=SEED,
                    batch_size=min(5000, n_samples),
                    backend=backend, verbose=False,
                )
            else:
                from secactpy import ridge
                result = ridge(X, Y, lambda_=LAMBDA, n_rand=N_RAND,
                               backend=backend, verbose=False)
            activity = result['zscore'].T.astype(np.float32)  # (samples x signatures)
        except Exception as e:
            log(f"    ERROR in ridge for {sig_name}: {e}")
            continue

        activity_results[sig_name] = {
            'activity': activity,
            'sig_columns': list(sig_matrix.columns),
            'common_genes': len(common_genes),
            'coverage': coverage,
        }

        del X, Y, result, activity
        gc.collect()

    elapsed = time.time() - t0
    log(f"    Completed in {elapsed:.1f}s")

    return {
        'filename': filename,
        'technology': technology,
        'tier': tier,
        'n_cells': n_cells,
        'n_genes': n_genes,
        'meta_df': meta_df,
        'expr_columns': list(expr_df.columns),
        'activity_results': activity_results,
    }


# ==============================================================================
# Summary and Comparison
# ==============================================================================

def build_tissue_summary(all_results: List[Dict]) -> pd.DataFrame:
    """Build summary of activity by tissue across all files.

    Returns:
        DataFrame with columns: tissue, technology, signature, mean_activity,
        std_activity, n_groups, n_cells, gene_coverage.
    """
    rows = []
    for res in all_results:
        if res is None:
            continue
        meta = res['meta_df']
        for sig_name, act_data in res['activity_results'].items():
            activity = act_data['activity']
            sig_cols = act_data['sig_columns']
            coverage = act_data['coverage']

            # Group by tissue if available
            if 'tissue' in meta.columns:
                for tissue in meta['tissue'].unique():
                    t_mask = meta['tissue'] == tissue
                    t_idx = np.where(t_mask.values)[0]
                    if len(t_idx) < 1:
                        continue

                    t_act = activity[t_idx]
                    n_cells = meta.loc[t_mask, 'n_cells'].sum()

                    for j, sig in enumerate(sig_cols):
                        vals = t_act[:, j]
                        rows.append({
                            'tissue': tissue,
                            'technology': res['technology'],
                            'signature': sig,
                            'signature_type': sig_name,
                            'mean_activity': float(np.nanmean(vals)),
                            'std_activity': float(np.nanstd(vals)),
                            'n_groups': len(t_idx),
                            'n_cells': int(n_cells),
                            'gene_coverage': coverage,
                            'file': res['filename'],
                        })
            else:
                # No tissue annotation: use file-level
                n_cells = meta['n_cells'].sum()
                for j, sig in enumerate(sig_cols):
                    vals = activity[:, j]
                    rows.append({
                        'tissue': 'Unknown',
                        'technology': res['technology'],
                        'signature': sig,
                        'signature_type': sig_name,
                        'mean_activity': float(np.nanmean(vals)),
                        'std_activity': float(np.nanstd(vals)),
                        'n_groups': activity.shape[0],
                        'n_cells': int(n_cells),
                        'gene_coverage': coverage,
                        'file': res['filename'],
                    })

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()


def build_technology_comparison(all_results: List[Dict]) -> pd.DataFrame:
    """Compare activity distributions across technologies for same tissues.

    Returns:
        DataFrame with per-technology, per-tissue, per-signature statistics.
    """
    tissue_df = build_tissue_summary(all_results)
    if tissue_df.empty:
        return pd.DataFrame()

    # Aggregate across files within same technology + tissue
    grouped = tissue_df.groupby(['tissue', 'technology', 'signature', 'signature_type']).agg(
        mean_activity=('mean_activity', 'mean'),
        std_activity=('std_activity', 'mean'),
        n_groups=('n_groups', 'sum'),
        n_cells=('n_cells', 'sum'),
        n_files=('file', 'nunique'),
        mean_coverage=('gene_coverage', 'mean'),
    ).reset_index()

    return grouped


def save_activity_h5ad(
    all_results: List[Dict],
    output_dir: Path,
) -> Optional[Path]:
    """Combine all activity results into a single H5AD file.

    Concatenates pseudobulk-level activity across all files and signatures.

    Returns:
        Path to output H5AD, or None if no data.
    """
    # Collect per-signature-type
    sig_type_data = {}  # sig_name -> list of (activity, obs_df)

    for res in all_results:
        if res is None:
            continue
        meta = res['meta_df'].copy()
        meta['technology'] = res['technology']
        meta['tier'] = res['tier']
        meta['source_n_cells'] = res['n_cells']
        meta['source_n_genes'] = res['n_genes']

        for sig_name, act_data in res['activity_results'].items():
            activity = act_data['activity']
            sig_cols = act_data['sig_columns']

            obs_df = meta.copy()
            obs_df['gene_coverage'] = act_data['coverage']
            obs_df['common_genes'] = act_data['common_genes']

            # Make obs_names unique by prepending filename
            obs_df.index = [f"{res['filename']}__{idx}" for idx in obs_df.index]

            if sig_name not in sig_type_data:
                sig_type_data[sig_name] = []

            sig_type_data[sig_name].append((activity, obs_df, sig_cols))

    if not sig_type_data:
        return None

    output_path = output_dir / 'spatial_activity_by_technology.h5ad'
    adatas = []

    for sig_name, entries in sig_type_data.items():
        act_list = [e[0] for e in entries]
        obs_list = [e[1] for e in entries]
        sig_cols = entries[0][2]  # All should have same sig columns

        combined_act = np.vstack(act_list).astype(np.float32)
        combined_obs = pd.concat(obs_list, axis=0)
        combined_obs['signature_type'] = sig_name

        adata = ad.AnnData(
            X=combined_act,
            obs=combined_obs,
            var=pd.DataFrame(index=sig_cols),
        )
        adata.uns['signature_type'] = sig_name
        adatas.append(adata)

    if len(adatas) == 1:
        combined = adatas[0]
    else:
        combined = ad.concat(adatas, join='outer', label='signature_type')

    combined.write_h5ad(output_path, compression='gzip')
    log(f"  Saved: {output_path.name} ({combined.shape})")

    del combined, adatas
    gc.collect()

    return output_path


# ==============================================================================
# Main Pipeline
# ==============================================================================

def run_pipeline(
    technology: str = 'all',
    test: bool = False,
    backend: str = 'auto',
) -> None:
    """Run the spatial activity inference pipeline.

    Args:
        technology: 'visium', 'targeted', or 'all'.
        test: If True, process only first 3 files per tier.
        backend: Computation backend ('auto', 'numpy', 'cupy').
    """
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover and classify files
    tiers = discover_files()

    # Determine which tiers to process
    if technology == 'visium':
        files_to_process = [('A', f) for f in tiers['A']]
    elif technology == 'targeted':
        files_to_process = [('B', f) for f in tiers['B']]
    else:
        files_to_process = ([('A', f) for f in tiers['A']] +
                            [('B', f) for f in tiers['B']])

    if test:
        # Limit to 3 files per tier
        tier_counts = {}
        limited = []
        for tier, f in files_to_process:
            tier_counts.setdefault(tier, 0)
            if tier_counts[tier] < 3:
                limited.append((tier, f))
                tier_counts[tier] += 1
        files_to_process = limited

    log(f"\nFiles to process: {len(files_to_process)}")
    log(f"Technology: {technology}")
    log(f"Backend: {backend if backend != 'auto' else BACKEND}")
    log(f"Test mode: {test}")

    # Load signature matrices
    log("\nLoading signature matrices...")
    cytosig = load_cytosig()
    secact = load_secact()
    log(f"  CytoSig: {cytosig.shape}")
    log(f"  SecAct: {secact.shape}")

    sig_matrices = {
        'cytosig': cytosig,
        'secact': secact,
    }

    # Process each file
    all_results = []
    for i, (tier, h5ad_path) in enumerate(files_to_process):
        log(f"\n{'=' * 60}")
        log(f"File {i+1}/{len(files_to_process)}: {h5ad_path.name}")
        log(f"{'=' * 60}")

        result = run_activity_for_file(
            h5ad_path=h5ad_path,
            sig_matrices=sig_matrices,
            tier=tier,
            backend=backend,
        )
        if result is not None:
            all_results.append(result)

        gc.collect()

    # Build outputs
    log(f"\n{'=' * 60}")
    log("Building output files...")
    log(f"{'=' * 60}")

    # 1. Activity H5AD
    h5ad_path = save_activity_h5ad(all_results, OUTPUT_DIR)

    # 2. Tissue summary CSV
    tissue_df = build_tissue_summary(all_results)
    if not tissue_df.empty:
        tissue_path = OUTPUT_DIR / 'spatial_activity_by_tissue.csv'
        tissue_df.to_csv(tissue_path, index=False)
        log(f"  Saved: {tissue_path.name} ({len(tissue_df)} rows)")

    # 3. Technology comparison CSV
    tech_df = build_technology_comparison(all_results)
    if not tech_df.empty:
        tech_path = OUTPUT_DIR / 'spatial_technology_comparison.csv'
        tech_df.to_csv(tech_path, index=False)
        log(f"  Saved: {tech_path.name} ({len(tech_df)} rows)")

    # Summary
    elapsed = time.time() - t_start
    log(f"\n{'=' * 60}")
    log(f"PIPELINE COMPLETE ({elapsed / 60:.1f} min)")
    log(f"  Files processed: {len(all_results)}/{len(files_to_process)}")
    log(f"  Output directory: {OUTPUT_DIR}")
    log(f"{'=' * 60}")

    # List output files
    out_files = sorted(OUTPUT_DIR.glob('*'))
    if out_files:
        log("\nOutput files:")
        for f in out_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            log(f"  {f.name} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="SpatialCorpus-110M Activity Inference Pipeline"
    )
    parser.add_argument(
        '--technology', default='all',
        choices=['visium', 'targeted', 'all'],
        help='Technology tier to process (default: all)',
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Test mode: process only first 3 files per tier',
    )
    parser.add_argument(
        '--backend', default='auto',
        choices=['auto', 'numpy', 'cupy'],
        help='Computation backend (default: auto)',
    )

    args = parser.parse_args()

    # Resolve backend
    backend = args.backend
    if backend == 'auto':
        backend = BACKEND
    log(f"Backend: {backend}")

    run_pipeline(
        technology=args.technology,
        test=args.test,
        backend=backend,
    )


if __name__ == '__main__':
    main()
