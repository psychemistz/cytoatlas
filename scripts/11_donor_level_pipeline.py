#!/usr/bin/env python3
"""
Donor-Level Pseudobulk and Single-Cell Activity Pipeline.

This script generates:
1. Donor-agnostic pseudobulk: Aggregate cells by (celltype × donor) to preserve donor-level variance
2. Activity inference on donor-level pseudobulk (CytoSig/LinCytoSig/SecAct)
3. Resampled donor-level pseudobulk for confidence intervals
4. Single-cell activity predictions for each cell

Usage:
    python scripts/11_donor_level_pipeline.py --atlas cima --run-singlecell
    python scripts/11_donor_level_pipeline.py --atlas inflammation_main
    python scripts/11_donor_level_pipeline.py --atlas scatlas_normal --levels celltype organ_celltype

Author: Claude Code (2026-02-04)
"""

import argparse
import gc
import gzip
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parents[1] / "cytoatlas-pipeline/src"))

from secactpy import load_cytosig, load_secact, ridge


def log(msg: str) -> None:
    """Print timestamped log message."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)


# =============================================================================
# Atlas Configuration
# =============================================================================

ATLAS_CONFIGS = {
    'cima': {
        'name': 'CIMA',
        'h5ad_path': '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad',
        'donor_col': 'sample',
        'levels': {
            'L1': 'cell_type_l1',
            'L2': 'cell_type_l2',
            'L3': 'cell_type_l3',
            'L4': 'cell_type_l4',
        },
    },
    'inflammation_main': {
        'name': 'Inflammation_Main',
        'h5ad_path': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad',
        'donor_col': 'sampleID',
        'levels': {
            'L1': 'Level1',
            'L2': 'Level2',
        },
    },
    'inflammation_val': {
        'name': 'Inflammation_Validation',
        'h5ad_path': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad',
        'donor_col': 'sampleID',
        'levels': {
            'L1': 'Level1pred',
            'L2': 'Level2pred',
        },
    },
    'inflammation_ext': {
        'name': 'Inflammation_External',
        'h5ad_path': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_external_afterQC.h5ad',
        'donor_col': 'sampleID',
        'levels': {
            'L1': 'Level1pred',
            'L2': 'Level2pred',
        },
    },
    'scatlas_normal': {
        'name': 'scAtlas_Normal',
        'h5ad_path': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad',
        'donor_col': 'donorID',
        'levels': {
            'celltype': 'cellType1',
            'organ_celltype': 'tissue+cellType1',  # Composite
        },
    },
    'scatlas_cancer': {
        'name': 'scAtlas_Cancer',
        'h5ad_path': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad',
        'donor_col': 'donorID',
        'levels': {
            'celltype': 'cellType1',
            'organ_celltype': 'tissue+cellType1',  # Composite
        },
    },
}


# =============================================================================
# Signature Loading
# =============================================================================

def load_signatures() -> Dict[str, pd.DataFrame]:
    """Load all signature matrices and fill NaN values."""
    cytosig = load_cytosig()
    secact = load_secact()

    # Load LinCytoSig
    lincytosig_path = Path("/data/parks34/projects/1ridgesig/SecActpy/secactpy/data/LinCytoSig.tsv.gz")
    with gzip.open(lincytosig_path, 'rt') as f:
        lincytosig = pd.read_csv(f, sep='\t', index_col=0)

    return {
        'cytosig': cytosig,
        'lincytosig': lincytosig,
        'secact': secact,
    }


def compute_activity(
    expr_matrix: np.ndarray,
    gene_names: List[str],
    signature: pd.DataFrame,
    lambda_: float = 5e5,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute activity scores using ridge regression.

    Args:
        expr_matrix: Expression matrix (samples × genes), log1p CPM normalized
        gene_names: List of gene names matching columns
        signature: Signature matrix (genes × targets)
        lambda_: Ridge regularization parameter

    Returns:
        Tuple of (activity_matrix, common_genes, target_names)
    """
    # Find common genes
    expr_genes = set(gene_names)
    sig_genes = set(signature.index)
    common = sorted(expr_genes & sig_genes)

    if len(common) < 100:
        raise ValueError(f"Too few common genes: {len(common)}")

    # Subset and align
    gene_idx = [gene_names.index(g) for g in common]
    Y = expr_matrix[:, gene_idx].T  # Transpose to (genes × samples)
    X = signature.loc[common].values  # (genes × targets)

    # Fill NaN in signature
    X = np.nan_to_num(X, nan=0.0)

    # Run ridge regression: X (genes × targets), Y (genes × samples)
    # Returns dict with 'zscore' (targets × samples)
    result = ridge(X, Y, lambda_=lambda_, n_rand=1000, verbose=False)
    activity = result['zscore'].T  # Transpose to (samples × targets)

    return activity, common, list(signature.columns)


# =============================================================================
# Donor-Level Pseudobulk Generation
# =============================================================================

class DonorLevelAggregator:
    """
    Generate donor-level pseudobulk and activity inference.

    For each (celltype × donor) combination, aggregate cells to create
    donor-specific expression profiles, then run activity inference.
    """

    def __init__(
        self,
        atlas_config: Dict,
        batch_size: int = 50000,
        min_cells: int = 10,
        n_bootstrap: int = 100,
        use_gpu: bool = True,
    ):
        self.config = atlas_config
        self.batch_size = batch_size
        self.min_cells = min_cells
        self.n_bootstrap = n_bootstrap
        self.use_gpu = use_gpu

        # GPU backend
        if use_gpu:
            try:
                import cupy as cp
                self.xp = cp
                n_devices = cp.cuda.runtime.getDeviceCount()
                mem = cp.cuda.runtime.memGetInfo()
                log(f"GPU: {n_devices} device(s), {mem[0]/1024**3:.1f}/{mem[1]/1024**3:.1f} GB free")
            except ImportError:
                self.xp = np
                log("CuPy not available, using NumPy")
        else:
            self.xp = np
            log("Using NumPy (CPU) backend")

    def generate_donor_pseudobulk(
        self,
        output_dir: Path,
        level_name: str,
        celltype_col: str,
    ) -> Dict[str, Path]:
        """
        Generate donor-level pseudobulk expression for a single level.

        Output files:
        - {atlas}_{level}_donor_pseudobulk.h5ad: Mean expression per (celltype, donor)
        - {atlas}_{level}_donor_pseudobulk_resampled.h5ad: Bootstrap resampled

        Returns:
            Dict mapping output type to path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        atlas_name = self.config['name'].lower().replace(' ', '_')
        h5ad_path = self.config['h5ad_path']
        donor_col = self.config['donor_col']

        log("=" * 70)
        log(f"Donor-Level Pseudobulk: {atlas_name} / {level_name}")
        log("=" * 70)

        # Handle composite columns
        if '+' in celltype_col:
            parts = celltype_col.split('+')
            composite_col = '_'.join(parts)
            is_composite = True
        else:
            composite_col = celltype_col
            is_composite = False
            parts = None

        # Open H5AD
        log(f"Opening H5AD: {h5ad_path}")
        adata = ad.read_h5ad(h5ad_path, backed='r')
        n_cells, n_genes = adata.shape
        log(f"Shape: {n_cells:,} cells × {n_genes:,} genes")

        # Get gene names and var info
        gene_names = list(adata.var_names)

        # Check if we have gene symbols in var (for Ensembl ID mapping)
        var_df = adata.var.copy() if hasattr(adata.var, 'columns') else pd.DataFrame(index=gene_names)
        gene_symbols = None
        if 'symbol' in var_df.columns:
            gene_symbols = var_df['symbol'].tolist()
            log(f"Found gene symbols in var['symbol'] - will use for activity computation")

        # Load metadata
        cols_to_load = [donor_col]
        if is_composite:
            cols_to_load.extend(parts)
        else:
            cols_to_load.append(celltype_col)

        log(f"Loading metadata: {cols_to_load}")
        obs_df = adata.obs[cols_to_load].copy()

        # Create composite column if needed
        if is_composite:
            mask_na = obs_df[parts[0]].isna() | obs_df[parts[1]].isna()
            obs_df[composite_col] = obs_df[parts[0]].astype(str) + '_' + obs_df[parts[1]].astype(str)
            obs_df.loc[mask_na, composite_col] = np.nan
            n_missing = mask_na.sum()
            if n_missing > 0:
                log(f"  {n_missing:,} cells with missing annotations excluded")

        # Handle NaN
        mask_valid = obs_df[composite_col].notna() & obs_df[donor_col].notna()
        n_invalid = (~mask_valid).sum()
        if n_invalid > 0:
            log(f"  {n_invalid:,} cells with missing celltype/donor excluded")

        # Create (celltype, donor) groups
        obs_df['_group'] = obs_df[composite_col].astype(str) + '__' + obs_df[donor_col].astype(str)
        obs_df.loc[~mask_valid, '_group'] = np.nan

        # Get unique groups
        unique_groups = sorted(obs_df['_group'].dropna().unique().tolist())
        log(f"Unique (celltype, donor) combinations: {len(unique_groups):,}")

        # Initialize accumulators
        group_sum = {g: np.zeros(n_genes, dtype=np.float64) for g in unique_groups}
        group_count = defaultdict(int)

        # Process in batches
        n_batches = (n_cells + self.batch_size - 1) // self.batch_size
        log(f"Processing {n_batches} batches...")

        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_cells)

            # Load batch
            X_batch = adata.X[start_idx:end_idx]
            if sparse.issparse(X_batch):
                X_batch = X_batch.toarray()

            # CPM normalize
            row_sums = X_batch.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            X_batch = X_batch / row_sums * 1e6
            X_batch = np.log1p(X_batch)

            # Get batch groups
            batch_groups = obs_df['_group'].iloc[start_idx:end_idx].values

            # Accumulate per group
            for i, g in enumerate(batch_groups):
                if pd.notna(g):
                    group_sum[g] += X_batch[i]
                    group_count[g] += 1

            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                log(f"  Batch {batch_idx + 1}/{n_batches}")

            del X_batch
            gc.collect()

        # Compute means and filter by min_cells
        log("Computing mean expression per group...")
        valid_groups = [g for g in unique_groups if group_count[g] >= self.min_cells]
        log(f"Groups with >= {self.min_cells} cells: {len(valid_groups):,}")

        # Build expression matrix
        expr_matrix = np.zeros((len(valid_groups), n_genes), dtype=np.float32)
        for i, g in enumerate(valid_groups):
            expr_matrix[i] = group_sum[g] / group_count[g]

        # Parse group labels back to celltype, donor
        celltypes = [g.split('__')[0] for g in valid_groups]
        donors = [g.split('__')[1] for g in valid_groups]

        # Create AnnData
        log("Creating AnnData...")
        obs_data = pd.DataFrame({
            'celltype': celltypes,
            'donor': donors,
            'n_cells': [group_count[g] for g in valid_groups],
        }, index=valid_groups)

        # Preserve var info (gene symbols, etc.) if available
        if var_df is not None and len(var_df.columns) > 0:
            var_data = var_df
        else:
            var_data = pd.DataFrame(index=gene_names)

        adata_pb = ad.AnnData(
            X=expr_matrix,
            obs=obs_data,
            var=var_data,
        )

        # Save pseudobulk
        pb_path = output_dir / f"{atlas_name}_{level_name}_donor_pseudobulk.h5ad"
        log(f"Saving: {pb_path}")
        adata_pb.write_h5ad(pb_path, compression='gzip')

        output_paths = {'pseudobulk': pb_path}

        # Generate bootstrap resampled pseudobulk
        log(f"Generating {self.n_bootstrap} bootstrap samples...")
        unique_celltypes = sorted(set(celltypes))

        resampled_data = []
        resampled_obs = []

        for ct in unique_celltypes:
            # Get donors for this celltype
            ct_mask = np.array(celltypes) == ct
            ct_indices = np.where(ct_mask)[0]

            if len(ct_indices) < 2:
                continue

            # Bootstrap resample donors
            rng = np.random.default_rng(42)
            for boot_idx in range(self.n_bootstrap):
                sampled_indices = rng.choice(ct_indices, size=len(ct_indices), replace=True)
                boot_mean = expr_matrix[sampled_indices].mean(axis=0)
                resampled_data.append(boot_mean)
                resampled_obs.append({
                    'celltype': ct,
                    'bootstrap': boot_idx,
                    'n_donors': len(ct_indices),
                })

        if resampled_data:
            resampled_matrix = np.vstack(resampled_data).astype(np.float32)
            resampled_obs_df = pd.DataFrame(resampled_obs)
            resampled_obs_df.index = [f"{r['celltype']}_boot{r['bootstrap']}" for _, r in resampled_obs_df.iterrows()]

            adata_resampled = ad.AnnData(
                X=resampled_matrix,
                obs=resampled_obs_df,
                var=var_data,
            )

            resampled_path = output_dir / f"{atlas_name}_{level_name}_donor_pseudobulk_resampled.h5ad"
            log(f"Saving: {resampled_path}")
            adata_resampled.write_h5ad(resampled_path, compression='gzip')
            output_paths['resampled'] = resampled_path

        log("=" * 70)
        log(f"Completed donor-level pseudobulk for {level_name}")
        log("=" * 70)

        return output_paths, expr_matrix, gene_names, obs_data, gene_symbols


def compute_activities_for_pseudobulk(
    expr_matrix: np.ndarray,
    gene_names: List[str],
    obs_data: pd.DataFrame,
    output_dir: Path,
    atlas_name: str,
    level_name: str,
    gene_symbols: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Compute activity scores for pseudobulk expression.

    If gene_symbols is provided (e.g., for Ensembl ID data), uses those
    for matching against signatures instead of gene_names.
    """
    signatures = load_signatures()
    output_paths = {}

    # Use gene symbols for matching if available
    if gene_symbols is not None:
        log(f"Using gene symbols for signature matching (Ensembl ID mapping)")
        matching_genes = gene_symbols
    else:
        matching_genes = gene_names

    for sig_name, sig_matrix in signatures.items():
        log(f"Computing {sig_name} activity...")
        try:
            activity, common_genes, targets = compute_activity(
                expr_matrix, matching_genes, sig_matrix
            )

            # Create activity AnnData
            adata_act = ad.AnnData(
                X=activity.astype(np.float32),
                obs=obs_data.copy(),
                var=pd.DataFrame(index=targets),
            )
            adata_act.uns['common_genes'] = len(common_genes)
            adata_act.uns['signature'] = sig_name

            # Save
            act_path = output_dir / f"{atlas_name}_{level_name}_donor_{sig_name}.h5ad"
            adata_act.write_h5ad(act_path, compression='gzip')
            output_paths[sig_name] = act_path
            log(f"  Saved: {act_path.name}")

        except Exception as e:
            log(f"  Failed: {e}")

    return output_paths


# =============================================================================
# Single-Cell Activity Prediction
# =============================================================================

def run_singlecell_activity(
    h5ad_path: str,
    output_dir: Path,
    atlas_name: str,
    signatures: List[str] = ['cytosig', 'lincytosig', 'secact'],
    batch_size: int = 10000,
    max_cells: Optional[int] = None,
) -> Dict[str, Path]:
    """
    Run single-cell activity prediction for all signatures.

    Due to computational cost, this uses a simplified ridge regression
    without permutation testing (just activity scores, no p-values).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(f"Single-Cell Activity Prediction: {atlas_name}")
    log("=" * 70)

    # Load all signatures
    sig_matrices = load_signatures()

    # Open H5AD
    log(f"Opening H5AD: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path, backed='r')
    n_cells, n_genes = adata.shape
    gene_names = list(adata.var_names)
    log(f"Shape: {n_cells:,} cells × {n_genes:,} genes")

    # Check for gene symbols (Ensembl ID mapping)
    gene_symbols = None
    if 'symbol' in adata.var.columns:
        gene_symbols = adata.var['symbol'].tolist()
        log(f"Found gene symbols in var['symbol'] - will use for activity computation")
        matching_genes = gene_symbols
    else:
        matching_genes = gene_names

    if max_cells is not None and max_cells < n_cells:
        n_cells = max_cells
        log(f"Processing first {n_cells:,} cells only")

    output_paths = {}

    for sig_name in signatures:
        if sig_name not in sig_matrices:
            log(f"Unknown signature: {sig_name}, skipping")
            continue

        log(f"\n--- {sig_name.upper()} ---")
        sig_matrix = sig_matrices[sig_name]

        # Find common genes (using symbols if available)
        common_genes = sorted(set(matching_genes) & set(sig_matrix.index))
        log(f"Common genes: {len(common_genes):,}")

        if len(common_genes) < 100:
            log(f"Too few common genes, skipping")
            continue

        gene_idx = [matching_genes.index(g) for g in common_genes]
        S = sig_matrix.loc[common_genes].values
        S = np.nan_to_num(S, nan=0.0)

        targets = list(sig_matrix.columns)
        n_targets = len(targets)

        # Process in batches
        n_batches = (n_cells + batch_size - 1) // batch_size
        log(f"Processing {n_batches} batches...")

        # Allocate output array
        activity_all = np.zeros((n_cells, n_targets), dtype=np.float32)

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_cells)

            # Load batch
            X_batch = adata.X[start_idx:end_idx]
            if sparse.issparse(X_batch):
                X_batch = X_batch.toarray()

            # CPM normalize and log1p
            row_sums = X_batch.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            X_batch = X_batch / row_sums * 1e6
            X_batch = np.log1p(X_batch)

            # Subset to common genes
            X_sub = X_batch[:, gene_idx]

            # Ridge regression: S (genes × targets), Y (genes × samples)
            Y_batch = X_sub.T  # Transpose to (genes × batch_cells)
            # Note: n_rand=1000 for permutation testing (n_rand=0 not supported on CuPy/GPU)
            result = ridge(S, Y_batch, lambda_=5e5, n_rand=1000, verbose=False)
            activity = result['zscore'].T  # Transpose to (batch_cells × targets)
            activity_all[start_idx:end_idx] = activity.astype(np.float32)

            if (batch_idx + 1) % 50 == 0 or batch_idx == n_batches - 1:
                pct = (batch_idx + 1) / n_batches * 100
                log(f"  Batch {batch_idx + 1}/{n_batches} ({pct:.1f}%)")

            del X_batch, X_sub, activity
            gc.collect()

        # Create output AnnData
        log("Creating output AnnData...")
        adata_out = ad.AnnData(
            X=activity_all,
            obs=adata.obs.iloc[:n_cells].copy(),
            var=pd.DataFrame(index=targets),
        )
        adata_out.uns['signature'] = sig_name
        adata_out.uns['common_genes'] = len(common_genes)

        # Save
        out_path = output_dir / f"{atlas_name}_singlecell_{sig_name}.h5ad"
        log(f"Saving: {out_path}")
        adata_out.write_h5ad(out_path, compression='gzip')
        output_paths[sig_name] = out_path

        del activity_all, adata_out
        gc.collect()

    return output_paths


# =============================================================================
# Main Pipeline
# =============================================================================

def run_full_pipeline(
    atlas_name: str,
    output_dir: Path,
    levels: Optional[List[str]] = None,
    run_singlecell: bool = False,
    singlecell_max_cells: Optional[int] = None,
    batch_size: int = 50000,
) -> Dict[str, Any]:
    """
    Run complete pipeline for an atlas.

    1. Generate donor-level pseudobulk for each annotation level
    2. Compute activity on donor-level pseudobulk
    3. (Optional) Compute single-cell activity predictions
    """
    if atlas_name not in ATLAS_CONFIGS:
        raise ValueError(f"Unknown atlas: {atlas_name}. Available: {list(ATLAS_CONFIGS.keys())}")

    config = ATLAS_CONFIGS[atlas_name]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_outputs = {}

    # Determine levels to process
    if levels is None:
        levels = list(config['levels'].keys())

    log("=" * 70)
    log(f"DONOR-LEVEL PIPELINE: {config['name']}")
    log("=" * 70)
    log(f"Levels: {levels}")
    log(f"Output: {output_dir}")

    # 1. Donor-level pseudobulk for each level
    aggregator = DonorLevelAggregator(atlas_config=config, batch_size=batch_size)

    for level_name in levels:
        celltype_col = config['levels'][level_name]

        pb_outputs, expr_matrix, gene_names, obs_data, gene_symbols = aggregator.generate_donor_pseudobulk(
            output_dir=output_dir / "donor_pseudobulk",
            level_name=level_name,
            celltype_col=celltype_col,
        )
        all_outputs[f'{level_name}_pseudobulk'] = pb_outputs

        # Compute activity on this pseudobulk
        act_outputs = compute_activities_for_pseudobulk(
            expr_matrix=expr_matrix,
            gene_names=gene_names,
            obs_data=obs_data,
            output_dir=output_dir / "donor_activity",
            atlas_name=config['name'].lower().replace(' ', '_'),
            level_name=level_name,
            gene_symbols=gene_symbols,
        )
        all_outputs[f'{level_name}_activity'] = act_outputs

    # 2. Single-cell activity (optional)
    if run_singlecell:
        log("\n" + "=" * 70)
        log("SINGLE-CELL ACTIVITY PREDICTION")
        log("=" * 70)

        sc_outputs = run_singlecell_activity(
            h5ad_path=config['h5ad_path'],
            output_dir=output_dir / "singlecell",
            atlas_name=config['name'].lower().replace(' ', '_'),
            max_cells=singlecell_max_cells,
        )
        all_outputs['singlecell'] = sc_outputs

    log("\n" + "=" * 70)
    log("PIPELINE COMPLETE")
    log("=" * 70)

    return all_outputs


def main():
    parser = argparse.ArgumentParser(description="Donor-Level Pseudobulk Pipeline")
    parser.add_argument('--atlas', required=True,
                        choices=list(ATLAS_CONFIGS.keys()),
                        help='Atlas to process')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('/data/parks34/projects/2cytoatlas/results/donor_level'),
                        help='Output directory')
    parser.add_argument('--levels', nargs='+',
                        help='Specific levels to process (default: all)')
    parser.add_argument('--run-singlecell', action='store_true',
                        help='Run single-cell activity prediction')
    parser.add_argument('--singlecell-max-cells', type=int,
                        help='Max cells for single-cell prediction (for testing)')
    parser.add_argument('--batch-size', type=int, default=50000,
                        help='Batch size for processing cells (default: 50000, use smaller for large datasets)')

    args = parser.parse_args()

    run_full_pipeline(
        atlas_name=args.atlas,
        output_dir=args.output_dir / args.atlas,
        levels=args.levels,
        run_singlecell=args.run_singlecell,
        singlecell_max_cells=args.singlecell_max_cells,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
