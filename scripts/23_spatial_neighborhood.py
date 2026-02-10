#!/usr/bin/env python3
"""
Spatial Neighborhood Activity Analysis
========================================
Analyze cytokine and secreted protein activity patterns in spatial
neighborhoods (niches) from the SpatialCorpus-110M dataset.

This script builds on the spatial activity results from script 20 to:
1. Compute activity patterns within spatial niches (local neighborhoods)
2. Identify spatially co-regulated cytokine/secreted protein programs
3. Compare activity patterns across technologies for the same tissues

Input:
    - results/spatial/spatial_activity_by_technology.h5ad  (from script 20)
    - results/spatial/spatial_activity_by_tissue.csv       (from script 20)

Output:
    - spatial_neighborhood_activity.csv        (niche-level activity patterns)
    - spatial_technology_comparison.csv        (extended cross-technology comparison)

Usage:
    # Full analysis
    python scripts/23_spatial_neighborhood.py

    # Specific atlas
    python scripts/23_spatial_neighborhood.py --atlas spatial

    # Force overwrite
    python scripts/23_spatial_neighborhood.py --force

    # Test mode (first 5 files)
    python scripts/23_spatial_neighborhood.py --test
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
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

# Add SecActpy to path
sys.path.insert(0, '/vf/users/parks34/projects/1ridgesig/SecActpy')
from secactpy import (
    load_cytosig, load_secact,
    ridge_batch, estimate_batch_size,
    CUPY_AVAILABLE
)

# ==============================================================================
# Configuration
# ==============================================================================

SPATIAL_DATA_DIR = Path('/data/Jiang_Lab/Data/Seongyong/SpatialCorpus-110M')
SPATIAL_RESULTS_DIR = Path('/vf/users/parks34/projects/2secactpy/results/spatial')
OUTPUT_DIR = Path('/vf/users/parks34/projects/2secactpy/results/spatial')

# Neighborhood analysis parameters
N_NEIGHBORS = 20         # Number of nearest neighbors for niche definition
NICHE_RADIUS_UM = 100    # Alternative: radius-based neighborhood (micrometers)
MIN_CELLS_PER_NICHE = 5  # Minimum cells to define a niche
LAMBDA = 5e5
N_RAND = 1000
CHUNK_SIZE = 50000        # Cells per chunk for backed-mode reading
MAX_CELLS_SINGLE_CELL = 500000  # Max cells for per-cell ridge (above → pseudobulk only)

# Mouse file keywords (to skip)
MOUSE_KEYWORDS = ['mouse', '_mouse_']

# GPU settings
BACKEND = 'cupy' if CUPY_AVAILABLE else 'numpy'


# ==============================================================================
# Utility Functions
# ==============================================================================

def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_mouse_file(filename: str) -> bool:
    """Check if file is a mouse dataset (to skip)."""
    fname_lower = filename.lower()
    return any(kw in fname_lower for kw in MOUSE_KEYWORDS)


def detect_spatial_coords(adata: ad.AnnData) -> Optional[str]:
    """Detect spatial coordinate representation in AnnData.

    Checks obsm keys and obs columns for spatial coordinates.

    Returns:
        Key name in obsm, or 'obs_xy' if in obs columns, or None.
    """
    # Check obsm for spatial coordinates
    spatial_keys = ['spatial', 'X_spatial', 'X_umap', 'spatial_coords']
    for key in spatial_keys:
        if key in adata.obsm:
            coords = adata.obsm[key]
            if coords.shape[1] >= 2:
                return key

    # Check obs columns
    xy_candidates = [
        ('x', 'y'),
        ('X', 'Y'),
        ('x_centroid', 'y_centroid'),
        ('x_location', 'y_location'),
        ('spatial_x', 'spatial_y'),
        ('array_row', 'array_col'),
    ]
    for xc, yc in xy_candidates:
        if xc in adata.obs.columns and yc in adata.obs.columns:
            return f'obs_{xc}_{yc}'

    return None


def get_spatial_coordinates(adata: ad.AnnData, coord_key: str) -> Optional[np.ndarray]:
    """Extract spatial coordinates from AnnData.

    Args:
        adata: AnnData object.
        coord_key: Key from detect_spatial_coords().

    Returns:
        (n_cells, 2) array of xy coordinates, or None.
    """
    if coord_key.startswith('obs_'):
        parts = coord_key.split('_')[1:]
        if len(parts) >= 2:
            xc, yc = parts[0], parts[1]
            if xc in adata.obs.columns and yc in adata.obs.columns:
                x = pd.to_numeric(adata.obs[xc], errors='coerce').values
                y = pd.to_numeric(adata.obs[yc], errors='coerce').values
                coords = np.column_stack([x, y])
                return coords
        return None
    else:
        coords = adata.obsm[coord_key]
        if hasattr(coords, 'toarray'):
            coords = coords.toarray()
        return coords[:, :2].astype(np.float64)


def detect_celltype_column(obs: pd.DataFrame) -> Optional[str]:
    """Detect cell type annotation column."""
    for col in ['cell_type', 'celltype', 'cell_type_l1', 'cellType1',
                'subCluster', 'cluster', 'leiden', 'annotation', 'CellType']:
        if col in obs.columns:
            return col
    return None


def detect_tissue_column(obs: pd.DataFrame) -> Optional[str]:
    """Detect tissue/region annotation column."""
    for col in ['tissue', 'tissue_type', 'organ', 'region', 'Tissue', 'tissue_id']:
        if col in obs.columns:
            return col
    return None


# ==============================================================================
# Neighborhood Construction
# ==============================================================================

def build_neighborhoods(
    coords: np.ndarray,
    n_neighbors: int = N_NEIGHBORS,
    radius: Optional[float] = None,
) -> np.ndarray:
    """Build spatial neighborhoods using KD-tree.

    Args:
        coords: (n_cells, 2) spatial coordinates.
        n_neighbors: Number of nearest neighbors per cell.
        radius: If provided, use radius-based neighborhoods instead.

    Returns:
        (n_cells, n_neighbors) array of neighbor indices.
        If radius-based, returns a list of variable-length arrays.
    """
    # Remove NaN coordinates
    valid_mask = np.all(np.isfinite(coords), axis=1)
    valid_coords = coords[valid_mask]

    if len(valid_coords) < n_neighbors + 1:
        return None

    tree = cKDTree(valid_coords)

    if radius is not None:
        # Radius-based neighborhoods
        neighbors = tree.query_ball_point(valid_coords, r=radius)
        return neighbors, valid_mask
    else:
        # KNN neighborhoods
        _, indices = tree.query(valid_coords, k=n_neighbors + 1)
        # Exclude self (first neighbor)
        return indices[:, 1:], valid_mask


def compute_niche_activity(
    act_matrix: np.ndarray,
    neighbor_indices: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Compute mean neighborhood activity for each cell.

    Args:
        act_matrix: (n_cells, n_signatures) activity matrix.
        neighbor_indices: (n_valid_cells, n_neighbors) neighbor indices.
        valid_mask: Boolean mask for cells with valid coordinates.

    Returns:
        (n_valid_cells, n_signatures) neighborhood mean activity.
    """
    n_valid = neighbor_indices.shape[0]
    n_sigs = act_matrix.shape[1]

    # Activity for valid cells only
    valid_act = act_matrix[valid_mask]

    niche_activity = np.zeros((n_valid, n_sigs), dtype=np.float64)

    for i in range(n_valid):
        neighbors = neighbor_indices[i]
        # Ensure neighbor indices are within bounds
        neighbors = neighbors[neighbors < len(valid_act)]
        if len(neighbors) > 0:
            niche_activity[i] = np.nanmean(valid_act[neighbors], axis=0)

    return niche_activity.astype(np.float32)


# ==============================================================================
# Spatial Co-Regulation Analysis
# ==============================================================================

def compute_spatial_coregulation(
    niche_activity: np.ndarray,
    sig_names: List[str],
    min_variance: float = 0.01,
) -> pd.DataFrame:
    """Identify spatially co-regulated signature programs.

    Computes pairwise Spearman correlation of neighborhood activity
    across all spatial locations.

    Args:
        niche_activity: (n_cells, n_signatures) niche-level activity.
        sig_names: Signature names.
        min_variance: Minimum variance threshold to include a signature.

    Returns:
        DataFrame with columns: sig1, sig2, spearman_rho, pvalue, n_cells.
    """
    n_sigs = niche_activity.shape[1]

    # Filter signatures by variance
    variances = np.nanvar(niche_activity, axis=0)
    valid_sigs = [i for i in range(n_sigs) if variances[i] > min_variance]

    if len(valid_sigs) < 2:
        return pd.DataFrame()

    rows = []
    for i in range(len(valid_sigs)):
        for j in range(i + 1, len(valid_sigs)):
            idx_i = valid_sigs[i]
            idx_j = valid_sigs[j]

            vals_i = niche_activity[:, idx_i]
            vals_j = niche_activity[:, idx_j]

            # Remove NaN
            mask = np.isfinite(vals_i) & np.isfinite(vals_j)
            if mask.sum() < 10:
                continue

            rho, pval = stats.spearmanr(vals_i[mask], vals_j[mask])

            if np.isfinite(rho):
                rows.append({
                    'sig1': sig_names[idx_i],
                    'sig2': sig_names[idx_j],
                    'spearman_rho': round(rho, 4),
                    'pvalue': pval,
                    'n_cells': int(mask.sum()),
                })

    df = pd.DataFrame(rows)

    # FDR correction
    if len(df) > 0 and 'pvalue' in df.columns:
        valid_pvals = df['pvalue'].dropna()
        if len(valid_pvals) > 0:
            _, fdr, _, _ = multipletests(valid_pvals.values, method='fdr_bh')
            df.loc[valid_pvals.index, 'fdr'] = fdr

    return df


# ==============================================================================
# Per-File Neighborhood Analysis
# ==============================================================================

def analyze_file_neighborhoods(
    h5ad_path: Path,
    sig_matrices: Dict[str, pd.DataFrame],
    n_neighbors: int = N_NEIGHBORS,
    backend: str = 'auto',
    chunk_size: int = CHUNK_SIZE,
) -> Optional[Dict]:
    """Run neighborhood analysis on a single spatial H5AD file.

    Uses backed='r' mode and chunked expression reading to avoid loading
    the full dense matrix into memory. For files exceeding MAX_CELLS_SINGLE_CELL,
    skips per-cell ridge regression (too expensive) and only extracts metadata.

    Args:
        h5ad_path: Path to H5AD file.
        sig_matrices: Dict of signature name -> DataFrame.
        n_neighbors: Number of neighbors for niche construction.
        backend: Computation backend.
        chunk_size: Cells per chunk for expression reading.

    Returns:
        Dict with neighborhood activity results, or None.
    """
    filename = h5ad_path.stem
    log(f"\n  Processing: {h5ad_path.name}")
    t0 = time.time()

    try:
        adata = ad.read_h5ad(h5ad_path, backed='r')
    except Exception as e:
        log(f"    ERROR reading {h5ad_path.name}: {e}")
        return None

    n_cells = adata.n_obs
    n_genes = adata.n_vars
    log(f"    Shape: {n_cells:,} cells x {n_genes} genes")

    if n_cells < 50:
        log(f"    SKIP: too few cells ({n_cells})")
        try:
            if hasattr(adata, 'file') and adata.file is not None:
                adata.file.close()
        except Exception:
            pass
        return None

    # Detect spatial coordinates (fast even for backed mode)
    coord_key = detect_spatial_coords(adata)
    if coord_key is None:
        log(f"    SKIP: no spatial coordinates found")
        try:
            if hasattr(adata, 'file') and adata.file is not None:
                adata.file.close()
        except Exception:
            pass
        return None

    coords = get_spatial_coordinates(adata, coord_key)
    if coords is None or len(coords) < 50:
        log(f"    SKIP: invalid spatial coordinates")
        try:
            if hasattr(adata, 'file') and adata.file is not None:
                adata.file.close()
        except Exception:
            pass
        return None

    log(f"    Spatial coords: {coord_key} ({coords.shape})")

    # Build neighborhoods from coordinates
    nbr_result = build_neighborhoods(coords, n_neighbors=n_neighbors)
    if nbr_result is None:
        log(f"    SKIP: could not build neighborhoods")
        try:
            if hasattr(adata, 'file') and adata.file is not None:
                adata.file.close()
        except Exception:
            pass
        return None

    neighbor_indices, valid_mask = nbr_result
    n_valid = valid_mask.sum()
    log(f"    Valid cells with coordinates: {n_valid}/{n_cells}")

    # Detect metadata columns (from obs, fast even for backed)
    ct_col = detect_celltype_column(adata.obs)
    tissue_col = detect_tissue_column(adata.obs)

    # Cache metadata before closing file if we skip per-cell analysis
    obs_copy = adata.obs.copy()
    gene_names = list(adata.var_names)

    file_results = {
        'filename': filename,
        'n_cells': n_cells,
        'n_valid': n_valid,
        'n_genes': n_genes,
        'niche_activities': {},
        'coregulation': {},
        'niche_stats': [],
    }

    # Check if file is too large for per-cell ridge regression
    if n_cells > MAX_CELLS_SINGLE_CELL:
        log(f"    Large file ({n_cells:,} cells > {MAX_CELLS_SINGLE_CELL:,}), "
            f"skipping per-cell ridge (use pseudobulk from script 20)")
        try:
            if hasattr(adata, 'file') and adata.file is not None:
                adata.file.close()
        except Exception:
            pass
        del adata
        gc.collect()

        elapsed = time.time() - t0
        log(f"    Completed (metadata only) in {elapsed:.1f}s")
        return file_results

    # Get expression matrix handle for chunked reading
    X = adata.X

    for sig_name, sig_matrix in sig_matrices.items():
        # Gene overlap check
        panel_genes = set(gene_names)
        sig_genes = set(sig_matrix.index)
        common = sorted(panel_genes & sig_genes)
        coverage = len(common) / len(sig_genes) if len(sig_genes) > 0 else 0.0

        if len(common) < 50:
            log(f"    {sig_name}: too few common genes ({len(common)}), skipping")
            continue

        log(f"    {sig_name}: {len(common)} common genes ({coverage:.1%} coverage)")

        # Prepare signature matrix for matched genes
        gene_idx = [gene_names.index(g) for g in common]
        X_sig = sig_matrix.loc[common].values.copy()
        np.nan_to_num(X_sig, copy=False, nan=0.0)

        sig_cols = list(sig_matrix.columns)
        n_sigs = len(sig_cols)

        # Compute per-cell activity using chunked reading
        cell_activity = np.zeros((n_cells, n_sigs), dtype=np.float32)
        n_chunks = (n_cells + chunk_size - 1) // chunk_size
        log(f"      Computing per-cell activity in {n_chunks} chunks...")

        from secactpy import ridge

        for chunk_i in range(n_chunks):
            start = chunk_i * chunk_size
            end = min((chunk_i + 1) * chunk_size, n_cells)

            # Read expression chunk from disk
            chunk_X = X[start:end, :]
            if hasattr(chunk_X, 'toarray'):
                chunk_X = chunk_X.toarray()

            # Subset to common genes and prepare for ridge
            chunk_expr = chunk_X[:, gene_idx].astype(np.float64).T  # (genes x cells_in_chunk)
            np.nan_to_num(chunk_expr, copy=False, nan=0.0)
            chunk_expr -= chunk_expr.mean(axis=1, keepdims=True)

            try:
                result_ridge = ridge(X_sig, chunk_expr, lambda_=LAMBDA,
                                     n_rand=N_RAND, backend=backend,
                                     verbose=False)
                cell_activity[start:end] = result_ridge['zscore'].T.astype(np.float32)
            except Exception as e:
                log(f"      ERROR in ridge chunk {chunk_i}: {e}")
                # Leave zeros for this chunk

            del chunk_X, chunk_expr, result_ridge
            if (chunk_i + 1) % 5 == 0 or chunk_i == n_chunks - 1:
                log(f"      Chunk {chunk_i + 1}/{n_chunks} ({end:,} cells)")

        # Compute neighborhood activity
        niche_act = compute_niche_activity(cell_activity, neighbor_indices, valid_mask)
        file_results['niche_activities'][sig_name] = {
            'niche_activity': niche_act,
            'sig_columns': sig_cols,
            'coverage': coverage,
            'common_genes': len(common),
        }

        # Compute spatial co-regulation
        coreg_df = compute_spatial_coregulation(niche_act, sig_cols)
        if not coreg_df.empty:
            file_results['coregulation'][sig_name] = coreg_df

        # Compute niche-level statistics by cell type
        if ct_col is not None:
            valid_ct = obs_copy[ct_col].values[valid_mask]
            for ct in np.unique(valid_ct):
                ct_mask = valid_ct == ct
                if ct_mask.sum() < MIN_CELLS_PER_NICHE:
                    continue

                ct_niche = niche_act[ct_mask]
                for j, sig in enumerate(sig_cols):
                    vals = ct_niche[:, j]
                    vals = vals[np.isfinite(vals)]
                    if len(vals) < 3:
                        continue

                    file_results['niche_stats'].append({
                        'file': filename,
                        'cell_type': str(ct),
                        'tissue': str(obs_copy[tissue_col].iloc[0]) if tissue_col else 'Unknown',
                        'signature': sig,
                        'signature_type': sig_name,
                        'mean_niche_activity': round(float(np.mean(vals)), 4),
                        'std_niche_activity': round(float(np.std(vals)), 4),
                        'mean_cell_activity': round(
                            float(np.mean(cell_activity[valid_mask][ct_mask, j])), 4),
                        'n_cells': int(ct_mask.sum()),
                        'gene_coverage': coverage,
                    })

        del X_sig, cell_activity, niche_act
        gc.collect()

    # Close backed file handle
    try:
        if hasattr(adata, 'file') and adata.file is not None:
            adata.file.close()
    except Exception:
        pass
    del adata
    gc.collect()

    elapsed = time.time() - t0
    log(f"    Completed in {elapsed:.1f}s")

    return file_results


# ==============================================================================
# Cross-Technology Comparison
# ==============================================================================

def compute_technology_comparison(
    all_niche_stats: pd.DataFrame,
    tissue_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare activity patterns across technologies for the same tissues.

    Args:
        all_niche_stats: Niche-level statistics across all files.
        tissue_summary: Tissue-level summary from script 20.

    Returns:
        Extended technology comparison DataFrame.
    """
    log("  Computing cross-technology comparison...")

    if all_niche_stats.empty:
        return pd.DataFrame()

    # Classify files by technology
    def classify_tech(filename):
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
        else:
            return 'Other'

    all_niche_stats = all_niche_stats.copy()
    all_niche_stats['technology'] = all_niche_stats['file'].apply(classify_tech)

    # Aggregate by technology + tissue + signature
    grouped = all_niche_stats.groupby(
        ['tissue', 'technology', 'signature', 'signature_type']
    ).agg(
        mean_niche_activity=('mean_niche_activity', 'mean'),
        std_niche_activity=('std_niche_activity', 'mean'),
        mean_cell_activity=('mean_cell_activity', 'mean'),
        n_cells=('n_cells', 'sum'),
        n_celltypes=('cell_type', 'nunique'),
        n_files=('file', 'nunique'),
        mean_coverage=('gene_coverage', 'mean'),
    ).reset_index()

    # For tissues present in multiple technologies, compute correlations
    tech_pairs = []
    tissues = grouped['tissue'].unique()

    for tissue in tissues:
        t_data = grouped[grouped['tissue'] == tissue]
        techs = t_data['technology'].unique()

        if len(techs) < 2:
            continue

        for i in range(len(techs)):
            for j in range(i + 1, len(techs)):
                t1, t2 = techs[i], techs[j]
                d1 = t_data[t_data['technology'] == t1].set_index('signature')
                d2 = t_data[t_data['technology'] == t2].set_index('signature')

                common_sigs = d1.index.intersection(d2.index)
                if len(common_sigs) < 5:
                    continue

                vals1 = d1.loc[common_sigs, 'mean_niche_activity'].values
                vals2 = d2.loc[common_sigs, 'mean_niche_activity'].values

                valid = np.isfinite(vals1) & np.isfinite(vals2)
                if valid.sum() < 5:
                    continue

                rho, pval = stats.spearmanr(vals1[valid], vals2[valid])

                tech_pairs.append({
                    'tissue': tissue,
                    'technology_1': t1,
                    'technology_2': t2,
                    'spearman_rho': round(rho, 4) if np.isfinite(rho) else None,
                    'pvalue': pval if np.isfinite(pval) else None,
                    'n_common_signatures': int(valid.sum()),
                })

    comparison_df = pd.DataFrame(tech_pairs)

    # Merge with aggregated stats
    result = grouped.copy()
    if not comparison_df.empty:
        result.attrs['tech_pair_correlations'] = comparison_df

    return result, comparison_df


# ==============================================================================
# Main Pipeline
# ==============================================================================

def run_pipeline(
    test: bool = False,
    force: bool = False,
    n_neighbors: int = N_NEIGHBORS,
    backend: str = 'auto',
) -> None:
    """Run the spatial neighborhood activity analysis pipeline.

    Args:
        test: If True, process only first 5 files.
        force: Force overwrite existing outputs.
        n_neighbors: Number of neighbors for niche construction.
        backend: Computation backend.
    """
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load signature matrices
    log("Loading signature matrices...")
    cytosig = load_cytosig()
    secact = load_secact()
    log(f"  CytoSig: {cytosig.shape}")
    log(f"  SecAct: {secact.shape}")
    sig_matrices = {'cytosig': cytosig, 'secact': secact}

    # Discover spatial files (human only, skip ISS/mouse)
    log(f"\nScanning spatial data: {SPATIAL_DATA_DIR}")
    h5ad_files = sorted(SPATIAL_DATA_DIR.glob('*.h5ad'))
    h5ad_files = [f for f in h5ad_files if not is_mouse_file(f.name)]
    log(f"  {len(h5ad_files)} human spatial H5AD files")

    if test:
        h5ad_files = h5ad_files[:5]
        log(f"  Test mode: processing first {len(h5ad_files)} files")

    # Load existing tissue summary if available
    tissue_summary_path = SPATIAL_RESULTS_DIR / 'spatial_activity_by_tissue.csv'
    if tissue_summary_path.exists():
        tissue_summary = pd.read_csv(tissue_summary_path)
        log(f"  Loaded existing tissue summary: {len(tissue_summary)} rows")
    else:
        tissue_summary = pd.DataFrame()
        log(f"  No existing tissue summary found")

    # Process each file
    all_niche_stats = []
    all_coregulation = []

    for i, h5ad_path in enumerate(h5ad_files):
        log(f"\n{'=' * 60}")
        log(f"File {i+1}/{len(h5ad_files)}: {h5ad_path.name}")
        log(f"{'=' * 60}")

        result = analyze_file_neighborhoods(
            h5ad_path=h5ad_path,
            sig_matrices=sig_matrices,
            n_neighbors=n_neighbors,
            backend=backend,
        )

        if result is not None:
            if result['niche_stats']:
                all_niche_stats.extend(result['niche_stats'])

            for sig_name, coreg_df in result.get('coregulation', {}).items():
                if not coreg_df.empty:
                    coreg_df = coreg_df.copy()
                    coreg_df['file'] = result['filename']
                    coreg_df['signature_type'] = sig_name
                    all_coregulation.append(coreg_df)

        gc.collect()

    # Build output DataFrames
    log(f"\n{'=' * 60}")
    log("Building output files...")
    log(f"{'=' * 60}")

    niche_stats_df = pd.DataFrame(all_niche_stats) if all_niche_stats else pd.DataFrame()
    coregulation_df = pd.concat(all_coregulation, ignore_index=True) if all_coregulation else pd.DataFrame()

    # Save neighborhood activity
    if not niche_stats_df.empty:
        niche_path = OUTPUT_DIR / 'spatial_neighborhood_activity.csv'
        niche_stats_df.to_csv(niche_path, index=False)
        log(f"  Saved: {niche_path.name} ({len(niche_stats_df)} rows)")

    # Save co-regulation results
    if not coregulation_df.empty:
        coreg_path = OUTPUT_DIR / 'spatial_coregulation.csv'
        coregulation_df.to_csv(coreg_path, index=False)
        log(f"  Saved: {coreg_path.name} ({len(coregulation_df)} rows)")

    # Cross-technology comparison
    if not niche_stats_df.empty:
        tech_agg, tech_pairs = compute_technology_comparison(niche_stats_df, tissue_summary)

        if not tech_agg.empty:
            tech_path = OUTPUT_DIR / 'spatial_technology_comparison.csv'
            tech_agg.to_csv(tech_path, index=False)
            log(f"  Saved: {tech_path.name} ({len(tech_agg)} rows)")

        if not tech_pairs.empty:
            pairs_path = OUTPUT_DIR / 'spatial_technology_pair_correlations.csv'
            tech_pairs.to_csv(pairs_path, index=False)
            log(f"  Saved: {pairs_path.name} ({len(tech_pairs)} rows)")

    # Summary
    elapsed = time.time() - t_start
    log(f"\n{'=' * 60}")
    log(f"PIPELINE COMPLETE ({elapsed / 60:.1f} min)")
    log(f"{'=' * 60}")

    if not niche_stats_df.empty:
        log(f"\nNeighborhood activity summary:")
        log(f"  Files processed: {niche_stats_df['file'].nunique()}")
        log(f"  Tissues: {niche_stats_df['tissue'].nunique()}")
        log(f"  Cell types: {niche_stats_df['cell_type'].nunique()}")
        log(f"  Total entries: {len(niche_stats_df)}")

    if not coregulation_df.empty:
        sig_coreg = coregulation_df[
            coregulation_df['fdr'].notna() & (coregulation_df['fdr'] < 0.05)
        ]
        log(f"\nSpatial co-regulation:")
        log(f"  Significant pairs (FDR < 0.05): {len(sig_coreg)}")
        if len(sig_coreg) > 0:
            top = sig_coreg.nlargest(5, 'spearman_rho')
            for _, row in top.iterrows():
                log(f"    {row['sig1']} <-> {row['sig2']}: "
                    f"rho={row['spearman_rho']:.3f}")

    # List output files
    out_files = sorted(OUTPUT_DIR.glob('spatial_*'))
    if out_files:
        log(f"\nOutput files:")
        for f in out_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            log(f"  {f.name} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Spatial Neighborhood Activity Analysis"
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Test mode: process only first 5 files',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force overwrite existing output files',
    )
    parser.add_argument(
        '--n-neighbors', type=int, default=N_NEIGHBORS,
        help=f'Number of spatial neighbors per cell (default: {N_NEIGHBORS})',
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
    log(f"Neighbors: {args.n_neighbors}")

    run_pipeline(
        test=args.test,
        force=args.force,
        n_neighbors=args.n_neighbors,
        backend=backend,
    )


if __name__ == '__main__':
    main()
