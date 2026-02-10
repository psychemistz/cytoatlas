#!/usr/bin/env python3
"""
Atlas Validation Pipeline - Unified Multi-Level Activity Inference
====================================================================

Three validation jobs in one pipeline:
[1] Atlas-level: pseudobulk (cell type aggregation) → activity inference
[2] Pseudobulk-level: resampling-based bootstrap pseudobulk → activity inference
[3] Single-cell level: direct single-cell → activity inference

All computations use batch processing, CuPy GPU acceleration, and H5AD streaming
for memory efficiency.

Output Structure:
    results/atlas_validation/
    ├── {atlas}/
    │   ├── pseudobulk/
    │   │   ├── {atlas}_pseudobulk_{level}.h5ad
    │   │   └── {atlas}_pseudobulk_{level}_resampled.h5ad
    │   ├── activity_atlas/          # [1] Atlas-level activity
    │   │   ├── {atlas}_{level}_{signature}.h5ad
    │   ├── activity_pseudobulk/     # [2] Pseudobulk-level (bootstrap)
    │   │   ├── {atlas}_{level}_{signature}_bootstrap.h5ad
    │   └── activity_singlecell/     # [3] Single-cell level
    │       └── {atlas}_{signature}_singlecell.h5ad

Usage:
    # Run all jobs for CIMA
    python 10_atlas_validation_pipeline.py --atlas cima --jobs all

    # Run only atlas-level validation
    python 10_atlas_validation_pipeline.py --atlas cima --jobs atlas

    # Run pseudobulk-level validation
    python 10_atlas_validation_pipeline.py --atlas cima --jobs pseudobulk

    # Run single-cell level validation (most expensive)
    python 10_atlas_validation_pipeline.py --atlas cima --jobs singlecell

    # Run for all atlases sequentially
    python 10_atlas_validation_pipeline.py --atlas all --jobs atlas
"""

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Literal

# Ensure line buffering for logs
sys.stdout.reconfigure(line_buffering=True)

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "cytoatlas-pipeline/src"))

import anndata as ad
import numpy as np
import pandas as pd

from cytoatlas_pipeline.batch import (
    ATLAS_REGISTRY,
    MultiLevelAggregator,
    aggregate_all_levels,
    get_atlas_config,
    SingleCellActivityInference,
    run_singlecell_activity,
)


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_ROOT = Path("/data/parks34/projects/2cytoatlas/results/atlas_validation")
SIGNATURES = ["cytosig", "secact", "lincytosig"]  # Available signatures

# Activity inference parameters
ACTIVITY_CONFIG = {
    "lambda_": 5e5,
    "n_rand": 1000,
    "seed": 42,
}


def log(msg: str):
    """Print timestamped message."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)


# =============================================================================
# [1] Atlas-Level Activity Inference
# =============================================================================

def run_atlas_level_activity(
    pseudobulk_path: Path,
    output_dir: Path,
    atlas_name: str,
    level: str,
    signatures: List[str] = SIGNATURES,
    use_gpu: bool = True,
) -> Dict[str, Path]:
    """
    Run activity inference on atlas-level pseudobulk.

    Args:
        pseudobulk_path: Path to pseudobulk H5AD
        output_dir: Output directory
        atlas_name: Atlas name
        level: Annotation level
        signatures: List of signatures to run
        use_gpu: Use GPU if available

    Returns:
        Dict mapping signatures to output paths
    """
    from secactpy import load_cytosig, load_secact
    from secactpy.batch import ridge_batch

    log(f"  Loading pseudobulk: {pseudobulk_path.name}")
    pb_adata = ad.read_h5ad(pseudobulk_path)

    # Always use X (log1p CPM), not zscore layer
    # Ridge regression internally standardizes, so pre-standardized input causes issues
    expr_matrix = pb_adata.X

    gene_names = list(pb_adata.var_names)
    cell_types = list(pb_adata.obs_names)

    output_paths = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for sig_type in signatures:
        output_path = output_dir / f"{atlas_name}_{level}_{sig_type}.h5ad"

        log(f"  Running {sig_type} activity inference...")

        # Load signature
        if sig_type == "cytosig":
            sig_df = load_cytosig()
        elif sig_type == "secact":
            sig_df = load_secact()
        elif sig_type == "lincytosig":
            import gzip
            lincytosig_path = Path("/data/parks34/projects/1ridgesig/SecActpy-dev/secactpy/data/LinCytoSig.tsv.gz")
            with gzip.open(lincytosig_path, 'rt') as f:
                sig_df = pd.read_csv(f, sep='\t', index_col=0)
        else:
            raise ValueError(f"Unknown signature type: {sig_type}")

        # Match genes
        common_genes = list(set(gene_names) & set(sig_df.index))
        gene_idx = [gene_names.index(g) for g in common_genes]

        # Subset and prepare matrices
        X = sig_df.loc[common_genes].values  # (n_genes, n_signatures)
        Y = expr_matrix[:, gene_idx].T  # (n_genes, n_celltypes)

        # Fill NaN values in signature matrix (LinCytoSig has ~13% NaN)
        X = np.nan_to_num(X, nan=0.0)

        if hasattr(Y, 'toarray'):
            Y = Y.toarray()

        # Scale Y
        Y = np.asarray(Y, dtype=np.float64)
        Y_mean = Y.mean(axis=0, keepdims=True)
        Y_std = Y.std(axis=0, ddof=1, keepdims=True)
        Y_std[Y_std < 1e-10] = 1.0
        Y_scaled = (Y - Y_mean) / Y_std

        # Run ridge regression
        result = ridge_batch(
            X, Y_scaled,
            lambda_=ACTIVITY_CONFIG["lambda_"],
            n_rand=ACTIVITY_CONFIG["n_rand"],
            seed=ACTIVITY_CONFIG["seed"],
            batch_size=1000,
            backend="cupy" if use_gpu else "numpy",
            verbose=False,
        )

        # Create output AnnData
        # result['zscore'] is (n_signatures, n_celltypes), transpose to (n_celltypes, n_signatures)
        activity_adata = ad.AnnData(
            X=result['zscore'].T.astype(np.float32),
            obs=pb_adata.obs.copy(),
            var=pd.DataFrame(index=sig_df.columns, data={'signature': sig_df.columns}),
        )
        activity_adata.layers['pvalue'] = result['pvalue'].T.astype(np.float32)
        activity_adata.layers['beta'] = result['beta'].T.astype(np.float32)
        activity_adata.uns['signature'] = sig_type
        activity_adata.uns['level'] = level
        activity_adata.uns['atlas'] = atlas_name
        activity_adata.uns['n_matched_genes'] = len(common_genes)

        activity_adata.write_h5ad(output_path, compression='gzip')
        output_paths[sig_type] = output_path

        log(f"    Saved: {output_path.name}")

        del result, activity_adata, X, Y, Y_scaled
        gc.collect()

    return output_paths


# =============================================================================
# [2] Pseudobulk-Level Activity Inference (Bootstrap)
# =============================================================================

def run_pseudobulk_level_activity(
    resampled_path: Path,
    output_dir: Path,
    atlas_name: str,
    level: str,
    signatures: List[str] = SIGNATURES,
    use_gpu: bool = True,
) -> Dict[str, Path]:
    """
    Run activity inference on resampled (bootstrap) pseudobulk.

    Args:
        resampled_path: Path to resampled pseudobulk H5AD
        output_dir: Output directory
        atlas_name: Atlas name
        level: Annotation level
        signatures: List of signatures to run
        use_gpu: Use GPU if available

    Returns:
        Dict mapping signatures to output paths
    """
    from secactpy import load_cytosig, load_secact
    from secactpy.batch import ridge_batch

    log(f"  Loading resampled pseudobulk: {resampled_path.name}")
    pb_adata = ad.read_h5ad(resampled_path)

    # Always use X (log1p CPM), not zscore layer
    expr_matrix = pb_adata.X

    gene_names = list(pb_adata.var_names)

    output_paths = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for sig_type in signatures:
        output_path = output_dir / f"{atlas_name}_{level}_{sig_type}_bootstrap.h5ad"

        log(f"  Running {sig_type} activity inference on bootstrap samples...")

        # Load signature
        if sig_type == "cytosig":
            sig_df = load_cytosig()
        elif sig_type == "secact":
            sig_df = load_secact()
        elif sig_type == "lincytosig":
            import gzip
            lincytosig_path = Path("/data/parks34/projects/1ridgesig/SecActpy-dev/secactpy/data/LinCytoSig.tsv.gz")
            with gzip.open(lincytosig_path, 'rt') as f:
                sig_df = pd.read_csv(f, sep='\t', index_col=0)
        else:
            raise ValueError(f"Unknown signature type: {sig_type}")

        # Match genes
        common_genes = list(set(gene_names) & set(sig_df.index))
        gene_idx = [gene_names.index(g) for g in common_genes]

        # Subset and prepare matrices
        X = sig_df.loc[common_genes].values
        Y = expr_matrix[:, gene_idx].T

        # Fill NaN values in signature matrix
        X = np.nan_to_num(X, nan=0.0)

        if hasattr(Y, 'toarray'):
            Y = Y.toarray()

        Y = np.asarray(Y, dtype=np.float64)
        Y_mean = Y.mean(axis=0, keepdims=True)
        Y_std = Y.std(axis=0, ddof=1, keepdims=True)
        Y_std[Y_std < 1e-10] = 1.0
        Y_scaled = (Y - Y_mean) / Y_std

        # Run ridge regression
        result = ridge_batch(
            X, Y_scaled,
            lambda_=ACTIVITY_CONFIG["lambda_"],
            n_rand=ACTIVITY_CONFIG["n_rand"],
            seed=ACTIVITY_CONFIG["seed"],
            batch_size=1000,
            backend="cupy" if use_gpu else "numpy",
            verbose=False,
        )

        # Create output AnnData
        activity_adata = ad.AnnData(
            X=result['zscore'].T.astype(np.float32),
            obs=pb_adata.obs.copy(),
            var=pd.DataFrame(index=sig_df.columns, data={'signature': sig_df.columns}),
        )
        activity_adata.layers['pvalue'] = result['pvalue'].T.astype(np.float32)
        activity_adata.uns['signature'] = sig_type
        activity_adata.uns['level'] = level
        activity_adata.uns['atlas'] = atlas_name
        activity_adata.uns['validation_type'] = 'bootstrap'
        activity_adata.uns['n_matched_genes'] = len(common_genes)

        activity_adata.write_h5ad(output_path, compression='gzip')
        output_paths[sig_type] = output_path

        log(f"    Saved: {output_path.name}")

        del result, activity_adata, X, Y, Y_scaled
        gc.collect()

    return output_paths


# =============================================================================
# [3] Single-Cell Level Activity Inference
# =============================================================================

def run_singlecell_level_activity(
    h5ad_path: Path,
    output_dir: Path,
    atlas_name: str,
    signatures: List[str] = SIGNATURES,
    batch_size: int = 10000,
    n_rand: int = 100,  # Lower for single-cell to speed up
    use_gpu: bool = True,
    max_cells: Optional[int] = None,
) -> Dict[str, Path]:
    """
    Run activity inference directly on single cells.

    Args:
        h5ad_path: Path to source H5AD file
        output_dir: Output directory
        atlas_name: Atlas name
        signatures: List of signatures to run
        batch_size: Cells per batch
        n_rand: Number of permutations (lower for speed)
        use_gpu: Use GPU if available
        max_cells: Maximum cells to process (for testing)

    Returns:
        Dict mapping signatures to output paths
    """
    output_paths = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for sig_type in signatures:
        output_path = output_dir / f"{atlas_name}_{sig_type}_singlecell.h5ad"

        log(f"  Running single-cell {sig_type} activity inference...")

        inference = SingleCellActivityInference(
            signature_type=sig_type,
            batch_size=batch_size,
            n_rand=n_rand,
            use_gpu=use_gpu,
            verbose=True,
        )

        inference.run(
            h5ad_path=h5ad_path,
            output_path=output_path,
            max_cells=max_cells,
        )

        output_paths[sig_type] = output_path

    return output_paths


# =============================================================================
# Main Pipeline
# =============================================================================

def run_atlas_validation(
    atlas_name: str,
    jobs: List[str],
    signatures: List[str] = SIGNATURES,
    use_gpu: bool = True,
    skip_existing: bool = False,
    max_cells_singlecell: Optional[int] = None,
    singlecell_batch_size: int = 10000,
    singlecell_n_rand: int = 100,
):
    """
    Run complete validation pipeline for an atlas.

    Args:
        atlas_name: Atlas name or "all" for all atlases
        jobs: List of jobs to run: ["atlas", "pseudobulk", "singlecell"]
        signatures: Signatures to run
        use_gpu: Use GPU if available
        skip_existing: Skip existing output files
        max_cells_singlecell: Max cells for single-cell (testing)
        singlecell_batch_size: Batch size for single-cell
        singlecell_n_rand: N permutations for single-cell
    """
    # Handle "all" atlases
    if atlas_name.lower() == "all":
        atlas_names = list(ATLAS_REGISTRY.keys())
    else:
        atlas_names = [atlas_name]

    for atlas in atlas_names:
        log("=" * 70)
        log(f"ATLAS VALIDATION: {atlas.upper()}")
        log("=" * 70)

        config = get_atlas_config(atlas)
        atlas_output_dir = OUTPUT_ROOT / atlas

        # Paths
        pseudobulk_dir = atlas_output_dir / "pseudobulk"
        activity_atlas_dir = atlas_output_dir / "activity_atlas"
        activity_pb_dir = atlas_output_dir / "activity_pseudobulk"
        activity_sc_dir = atlas_output_dir / "activity_singlecell"

        # [1] Atlas-level validation
        if "atlas" in jobs or "all" in jobs:
            log("\n[1] ATLAS-LEVEL VALIDATION")
            log("-" * 50)

            # First, generate pseudobulk if not exists
            pb_files = list(pseudobulk_dir.glob(f"{atlas.lower()}_pseudobulk_*.h5ad"))
            pb_files = [f for f in pb_files if 'resampled' not in f.name]

            if not pb_files or not skip_existing:
                log("Generating pseudobulk for all levels...")
                aggregate_all_levels(
                    atlas_name=atlas,
                    output_dir=pseudobulk_dir,
                    n_bootstrap=100,
                    batch_size=50000,
                    use_gpu=use_gpu,
                    skip_existing=skip_existing,
                )
                pb_files = list(pseudobulk_dir.glob(f"{atlas.lower()}_pseudobulk_*.h5ad"))
                pb_files = [f for f in pb_files if 'resampled' not in f.name]

            # Run activity inference on each level
            for pb_file in sorted(pb_files):
                level = pb_file.stem.split('_')[-1]  # Extract level from filename
                log(f"\nProcessing {level}...")

                run_atlas_level_activity(
                    pseudobulk_path=pb_file,
                    output_dir=activity_atlas_dir,
                    atlas_name=atlas,
                    level=level,
                    signatures=signatures,
                    use_gpu=use_gpu,
                )

        # [2] Pseudobulk-level validation (bootstrap)
        if "pseudobulk" in jobs or "all" in jobs:
            log("\n[2] PSEUDOBULK-LEVEL VALIDATION (Bootstrap)")
            log("-" * 50)

            # Find resampled files
            resampled_files = list(pseudobulk_dir.glob(f"{atlas.lower()}_pseudobulk_*_resampled.h5ad"))

            if not resampled_files:
                log("No resampled pseudobulk files found. Run atlas-level first.")
            else:
                for rs_file in sorted(resampled_files):
                    level = rs_file.stem.replace(f"{atlas.lower()}_pseudobulk_", "").replace("_resampled", "")
                    log(f"\nProcessing {level} bootstrap samples...")

                    run_pseudobulk_level_activity(
                        resampled_path=rs_file,
                        output_dir=activity_pb_dir,
                        atlas_name=atlas,
                        level=level,
                        signatures=signatures,
                        use_gpu=use_gpu,
                    )

        # [3] Single-cell level validation
        if "singlecell" in jobs or "all" in jobs:
            log("\n[3] SINGLE-CELL LEVEL VALIDATION")
            log("-" * 50)

            h5ad_path = Path(config.h5ad_path)

            run_singlecell_level_activity(
                h5ad_path=h5ad_path,
                output_dir=activity_sc_dir,
                atlas_name=atlas,
                signatures=signatures,
                batch_size=singlecell_batch_size,
                n_rand=singlecell_n_rand,
                use_gpu=use_gpu,
                max_cells=max_cells_singlecell,
            )

        log("\n" + "=" * 70)
        log(f"COMPLETED: {atlas.upper()}")
        log("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Atlas Validation Pipeline - Multi-Level Activity Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all validation jobs for CIMA
    python 10_atlas_validation_pipeline.py --atlas cima --jobs all

    # Run only atlas-level validation
    python 10_atlas_validation_pipeline.py --atlas cima --jobs atlas

    # Run pseudobulk-level validation (bootstrap)
    python 10_atlas_validation_pipeline.py --atlas cima --jobs pseudobulk

    # Run single-cell validation (expensive)
    python 10_atlas_validation_pipeline.py --atlas cima --jobs singlecell

    # Run for all atlases
    python 10_atlas_validation_pipeline.py --atlas all --jobs atlas

    # Quick test with limited cells
    python 10_atlas_validation_pipeline.py --atlas cima --jobs singlecell --max-cells 10000

Available atlases:
    cima, inflammation_main, inflammation_val, inflammation_ext, scatlas_normal, scatlas_cancer
        """
    )

    parser.add_argument('--atlas', type=str, default=None,
                        help='Atlas name or "all"')
    parser.add_argument('--jobs', nargs='+', default=None,
                        choices=['atlas', 'pseudobulk', 'singlecell', 'all'],
                        help='Jobs to run')
    parser.add_argument('--signatures', nargs='+', default=['cytosig', 'secact', 'lincytosig'],
                        choices=['cytosig', 'secact', 'lincytosig'],
                        help='Signatures to run (default: all three)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip existing output files')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration')
    parser.add_argument('--max-cells', type=int, default=None,
                        help='Maximum cells for single-cell validation (testing)')
    parser.add_argument('--singlecell-batch-size', type=int, default=10000,
                        help='Batch size for single-cell (default: 10000)')
    parser.add_argument('--singlecell-n-rand', type=int, default=100,
                        help='Permutations for single-cell (default: 100)')
    parser.add_argument('--list-atlases', action='store_true',
                        help='List available atlases and exit')

    args = parser.parse_args()

    if args.list_atlases:
        print("\nAvailable Atlases:")
        print("=" * 60)
        for name, config in ATLAS_REGISTRY.items():
            levels = ', '.join(config.list_levels())
            print(f"  {name}:")
            print(f"    Levels: {levels}")
            print(f"    Cells: ~{config.n_cells:,}")
        return

    # Check required arguments
    if not args.atlas or not args.jobs:
        parser.print_help()
        print("\nERROR: --atlas and --jobs are required")
        sys.exit(1)

    log("=" * 70)
    log("ATLAS VALIDATION PIPELINE")
    log("=" * 70)
    log(f"Atlas: {args.atlas}")
    log(f"Jobs: {args.jobs}")
    log(f"Signatures: {args.signatures}")
    log(f"GPU: {'Disabled' if args.no_gpu else 'Enabled'}")
    if args.max_cells:
        log(f"Max cells (single-cell): {args.max_cells:,}")
    log("=" * 70)

    start_time = time.time()

    run_atlas_validation(
        atlas_name=args.atlas,
        jobs=args.jobs,
        signatures=args.signatures,
        use_gpu=not args.no_gpu,
        skip_existing=args.skip_existing,
        max_cells_singlecell=args.max_cells,
        singlecell_batch_size=args.singlecell_batch_size,
        singlecell_n_rand=args.singlecell_n_rand,
    )

    total_time = time.time() - start_time
    log(f"\nTotal pipeline time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")


if __name__ == "__main__":
    main()
