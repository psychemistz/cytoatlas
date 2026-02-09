#!/usr/bin/env python3
"""
Atlas-Level Activity Inference for Validation
==============================================

Runs CytoSig, LinCytoSig, and SecAct activity inference on pseudobulk data
using GPU-accelerated batch processing.

Output H5AD Structure:
- X: Z-scores (samples × signatures)
- layers['beta']: Activity coefficients
- layers['se']: Standard errors
- layers['pvalue']: P-values
- uns: Signature metadata, gene overlap statistics

Usage:
    # Single pseudobulk file with all signatures
    python 09_atlas_validation_activity.py --input pseudobulk.h5ad

    # Specific signatures
    python 09_atlas_validation_activity.py --input pseudobulk.h5ad --signatures cytosig lincytosig

    # By atlas/level
    python 09_atlas_validation_activity.py --atlas cima --level L1

    # SLURM array job
    python 09_atlas_validation_activity.py --config-index 0
"""

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import anndata as ad

# Ensure line buffering for SLURM logs
sys.stdout.reconfigure(line_buffering=True)

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "cytoatlas-pipeline/src"))
sys.path.insert(0, "/vf/users/parks34/projects/1ridgesig/SecActpy")

from secactpy import (
    load_cytosig, load_secact, load_lincytosig,
    ridge_batch, CUPY_AVAILABLE
)

from cytoatlas_pipeline.batch import ATLAS_REGISTRY, get_atlas_config


# =============================================================================
# Configuration
# =============================================================================

PSEUDOBULK_ROOT = Path("/vf/users/parks34/projects/2secactpy/results/atlas_validation")
OUTPUT_ROOT = Path("/vf/users/parks34/projects/2secactpy/results/atlas_validation")

SIGNATURES = ['cytosig', 'lincytosig', 'secact']

SIGNATURE_LOADERS = {
    'cytosig': load_cytosig,
    'lincytosig': load_lincytosig,
    'secact': load_secact,
}

# Inference parameters
N_RAND = 1000
SEED = 42
BATCH_SIZE = 5000


def get_all_configs() -> List[Tuple[str, str]]:
    """Get all (atlas, level) combinations."""
    configs = []
    for atlas_name, atlas_config in ATLAS_REGISTRY.items():
        for level in atlas_config.list_levels():
            configs.append((atlas_name, level))
    return configs


# =============================================================================
# Logging
# =============================================================================

def log(msg: str):
    """Print timestamped message."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)


# =============================================================================
# Activity Inference
# =============================================================================

def run_activity_inference(
    pseudobulk_path: Path,
    output_dir: Path,
    signatures: List[str] = None,
    n_rand: int = N_RAND,
    seed: int = SEED,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, Path]:
    """
    Run activity inference on pseudobulk data.

    Args:
        pseudobulk_path: Path to pseudobulk H5AD file
        output_dir: Output directory for activity H5AD files
        signatures: List of signatures to run
        n_rand: Number of permutations
        seed: Random seed
        batch_size: Samples per batch

    Returns:
        Dict mapping signature names to output paths
    """
    if signatures is None:
        signatures = SIGNATURES

    log(f"Loading pseudobulk: {pseudobulk_path}")
    adata = ad.read_h5ad(pseudobulk_path)

    n_samples = adata.n_obs
    n_genes = adata.n_vars
    genes = list(adata.var_names)
    sample_names = list(adata.obs_names)

    log(f"  Shape: {adata.shape} (samples × genes)")
    log(f"  Backend: {'CuPy (GPU)' if CUPY_AVAILABLE else 'NumPy (CPU)'}")

    # Get basename for output naming
    basename = pseudobulk_path.stem.replace('_pseudobulk', '')

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}

    for sig_name in signatures:
        log(f"\n{'='*60}")
        log(f"Running {sig_name.upper()} inference")
        log('='*60)

        # Load signature matrix
        sig_matrix = SIGNATURE_LOADERS[sig_name]()
        log(f"  Signature shape: {sig_matrix.shape}")

        # Align genes
        common_genes = list(set(genes) & set(sig_matrix.index))
        log(f"  Common genes: {len(common_genes)} ({len(common_genes)/len(sig_matrix):.1%} of signature)")

        if len(common_genes) < 100:
            log(f"  WARNING: Too few common genes, skipping")
            continue

        # Get aligned data
        gene_idx = [genes.index(g) for g in common_genes]
        X = sig_matrix.loc[common_genes].values  # (genes × signatures)
        Y = adata.X[:, gene_idx].T  # (genes × samples)

        if hasattr(Y, 'toarray'):
            Y = Y.toarray()

        feature_names = list(sig_matrix.columns)
        n_features = len(feature_names)

        log(f"  Running ridge regression ({n_samples} samples, {n_features} signatures)")
        log(f"  Batch size: {batch_size}, n_rand: {n_rand}")

        start_time = time.time()

        # Run ridge regression
        result = ridge_batch(
            X, Y,
            batch_size=batch_size,
            n_rand=n_rand,
            seed=seed,
            verbose=True
        )

        inference_time = time.time() - start_time
        log(f"  Inference completed in {inference_time:.1f}s")

        # Create AnnData (samples × signatures)
        activity_adata = ad.AnnData(
            X=result['zscore'].T.astype(np.float32),  # (samples × signatures)
            obs=adata.obs.copy(),
            var=pd.DataFrame(index=feature_names),
        )
        activity_adata.layers['beta'] = result['beta'].T.astype(np.float32)
        activity_adata.layers['se'] = result['se'].T.astype(np.float32)
        activity_adata.layers['pvalue'] = result['pvalue'].T.astype(np.float32)

        # Metadata
        activity_adata.uns['signature'] = sig_name
        activity_adata.uns['gene_overlap'] = len(common_genes) / len(sig_matrix)
        activity_adata.uns['n_genes_used'] = len(common_genes)
        activity_adata.uns['inference_time_seconds'] = inference_time
        activity_adata.uns['pseudobulk_source'] = str(pseudobulk_path)

        # Copy atlas metadata if present
        for key in ['atlas', 'level', 'celltype_col', 'atlas_stats']:
            if key in adata.uns:
                activity_adata.uns[key] = adata.uns[key]

        # Save
        output_path = output_dir / f"{basename}_{sig_name}.h5ad"
        activity_adata.write_h5ad(output_path, compression='gzip')
        log(f"  Saved: {output_path}")

        output_paths[sig_name] = output_path

        # Clean up
        del result
        gc.collect()

    return output_paths


def run_inference_for_atlas(
    atlas_name: str,
    level: str,
    signatures: List[str] = None,
    **kwargs
) -> Dict[str, Path]:
    """
    Run activity inference for a specific atlas/level.

    Args:
        atlas_name: Atlas name
        level: Annotation level
        signatures: Signatures to run
        **kwargs: Additional arguments for run_activity_inference

    Returns:
        Dict mapping signature names to output paths
    """
    # Find pseudobulk file
    pseudobulk_dir = PSEUDOBULK_ROOT / atlas_name / "pseudobulk"
    pseudobulk_path = pseudobulk_dir / f"{atlas_name}_pseudobulk_{level.lower()}.h5ad"

    if not pseudobulk_path.exists():
        raise FileNotFoundError(f"Pseudobulk not found: {pseudobulk_path}")

    # Output directory
    output_dir = OUTPUT_ROOT / atlas_name / "activity"

    return run_activity_inference(
        pseudobulk_path=pseudobulk_path,
        output_dir=output_dir,
        signatures=signatures,
        **kwargs
    )


# =============================================================================
# Validation (Expression vs Activity Correlation)
# =============================================================================

def validate_expression_vs_activity(
    pseudobulk_path: Path,
    activity_path: Path,
    output_path: Path,
    min_samples: int = 5,
) -> pd.DataFrame:
    """
    Validate activity by correlating with z-scored target gene expression.

    Uses the atlas-level z-scored expression layer from pseudobulk.

    Returns:
        DataFrame with validation results per signature
    """
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    log("Running validation (z-scored expression vs activity correlation)...")

    # Load data
    expr_adata = ad.read_h5ad(pseudobulk_path)
    act_adata = ad.read_h5ad(activity_path)

    # Use z-scored expression layer if available
    if 'zscore' in expr_adata.layers:
        log("  Using atlas-level z-scored expression")
        expr_matrix = expr_adata.layers['zscore']
    else:
        log("  WARNING: No zscore layer, using X")
        expr_matrix = expr_adata.X

    expr_genes = list(expr_adata.var_names)
    expr_genes_upper = {g.upper(): (i, g) for i, g in enumerate(expr_genes)}

    sig_name = act_adata.uns.get('signature', 'unknown')

    # CytoSig name mapping
    CYTOSIG_TO_HGNC = {
        'TNFA': 'TNF', 'IFNA': 'IFNA1', 'IFNB': 'IFNB1', 'IFNL': 'IFNL1',
        'GMCSF': 'CSF2', 'GCSF': 'CSF3', 'MCSF': 'CSF1',
        'IL12': 'IL12A', 'Activin A': 'INHBA', 'TWEAK': 'TNFSF12',
        'CD40L': 'CD40LG', 'PDL1': 'CD274',
    }

    results = []

    for sig_idx, signature in enumerate(act_adata.var_names):
        # Determine target gene
        if sig_name == 'lincytosig' and '__' in signature:
            parts = signature.split('__')
            cytokine = parts[1] if len(parts) > 1 else signature
            gene_name = CYTOSIG_TO_HGNC.get(cytokine, cytokine)
        else:
            gene_name = CYTOSIG_TO_HGNC.get(signature, signature)
            cytokine = signature

        # Find gene in expression
        if gene_name.upper() not in expr_genes_upper:
            continue

        gene_idx, actual_gene = expr_genes_upper[gene_name.upper()]

        # Get common samples
        common_samples = [s for s in act_adata.obs_names if s in expr_adata.obs_names]
        if len(common_samples) < min_samples:
            continue

        # Get sample indices
        expr_sample_idx = [list(expr_adata.obs_names).index(s) for s in common_samples]
        act_sample_idx = [list(act_adata.obs_names).index(s) for s in common_samples]

        # Get values
        expr_vals = expr_matrix[expr_sample_idx, gene_idx]
        act_vals = act_adata.X[act_sample_idx, sig_idx]

        if hasattr(expr_vals, 'toarray'):
            expr_vals = np.asarray(expr_vals).flatten()
        if hasattr(act_vals, 'toarray'):
            act_vals = np.asarray(act_vals).flatten()

        # Remove NaN/Inf
        mask = np.isfinite(expr_vals) & np.isfinite(act_vals)
        if mask.sum() < min_samples:
            continue

        expr_vals = expr_vals[mask]
        act_vals = act_vals[mask]

        # Correlations
        r_pearson, p_pearson = stats.pearsonr(expr_vals, act_vals)
        r_spearman, p_spearman = stats.spearmanr(expr_vals, act_vals)

        results.append({
            'signature': signature,
            'gene': actual_gene,
            'cytokine': cytokine,
            'pearson_r': r_pearson,
            'pearson_p': p_pearson,
            'spearman_r': r_spearman,
            'spearman_p': p_spearman,
            'r2': r_pearson ** 2,
            'n_samples': len(expr_vals),
        })

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        # FDR correction
        _, results_df['pearson_q'], _, _ = multipletests(
            results_df['pearson_p'], method='fdr_bh'
        )
        _, results_df['spearman_q'], _, _ = multipletests(
            results_df['spearman_p'], method='fdr_bh'
        )

    log(f"  Validated {len(results_df)} signatures")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    log(f"  Saved: {output_path}")

    return results_df


# =============================================================================
# CLI
# =============================================================================

def list_configs():
    """List all available configurations."""
    configs = get_all_configs()

    print("\nAvailable Atlas/Level Configurations:")
    print("=" * 60)
    print(f"{'Index':<8} {'Atlas':<25} {'Level':<15}")
    print("-" * 60)

    for idx, (atlas, level) in enumerate(configs):
        print(f"{idx:<8} {atlas:<25} {level:<15}")

    print("-" * 60)
    print(f"Total: {len(configs)} configurations")
    print(f"\nSignatures: {', '.join(SIGNATURES)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run activity inference on pseudobulk data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # By atlas/level
    python 09_atlas_validation_activity.py --atlas cima --level L1

    # Specific input file
    python 09_atlas_validation_activity.py --input pseudobulk.h5ad

    # Specific signatures
    python 09_atlas_validation_activity.py --atlas cima --level L1 --signatures cytosig lincytosig

    # SLURM array job
    python 09_atlas_validation_activity.py --config-index 0

    # With validation
    python 09_atlas_validation_activity.py --atlas cima --level L1 --validate
        """
    )

    parser.add_argument('--atlas', type=str, help='Atlas name')
    parser.add_argument('--level', type=str, help='Annotation level')
    parser.add_argument('--input', type=str, help='Input pseudobulk H5AD file')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--signatures', nargs='+', choices=SIGNATURES,
                        help=f'Signatures to run (default: all)')
    parser.add_argument('--config-index', type=int, help='Config index for SLURM array')
    parser.add_argument('--list-configs', action='store_true', help='List configurations')
    parser.add_argument('--validate', action='store_true',
                        help='Run expression vs activity validation')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--n-rand', type=int, default=N_RAND,
                        help=f'Number of permutations (default: {N_RAND})')
    parser.add_argument('--seed', type=int, default=SEED,
                        help=f'Random seed (default: {SEED})')

    args = parser.parse_args()

    # List configs
    if args.list_configs:
        list_configs()
        return

    # Process by config index (SLURM array jobs)
    if args.config_index is not None:
        configs = get_all_configs()
        if args.config_index >= len(configs):
            print(f"ERROR: Config index {args.config_index} out of range (max: {len(configs)-1})")
            sys.exit(1)

        atlas_name, level = configs[args.config_index]
        log(f"Config {args.config_index}: {atlas_name}/{level}")

        output_paths = run_inference_for_atlas(
            atlas_name, level,
            signatures=args.signatures,
            batch_size=args.batch_size,
            n_rand=args.n_rand,
            seed=args.seed,
        )

        if args.validate:
            pseudobulk_path = PSEUDOBULK_ROOT / atlas_name / "pseudobulk" / f"{atlas_name}_pseudobulk_{level.lower()}.h5ad"
            for sig_name, activity_path in output_paths.items():
                validation_path = activity_path.parent / f"{atlas_name}_{level.lower()}_{sig_name}_validation.csv"
                validate_expression_vs_activity(pseudobulk_path, activity_path, validation_path)

        return

    # Process by atlas/level
    if args.atlas and args.level:
        output_paths = run_inference_for_atlas(
            args.atlas, args.level,
            signatures=args.signatures,
            batch_size=args.batch_size,
            n_rand=args.n_rand,
            seed=args.seed,
        )

        if args.validate:
            pseudobulk_path = PSEUDOBULK_ROOT / args.atlas / "pseudobulk" / f"{args.atlas}_pseudobulk_{args.level.lower()}.h5ad"
            for sig_name, activity_path in output_paths.items():
                validation_path = activity_path.parent / f"{args.atlas}_{args.level.lower()}_{sig_name}_validation.csv"
                validate_expression_vs_activity(pseudobulk_path, activity_path, validation_path)

        return

    # Process by input file
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"ERROR: Input file not found: {input_path}")
            sys.exit(1)

        output_dir = Path(args.output_dir) if args.output_dir else input_path.parent

        output_paths = run_activity_inference(
            pseudobulk_path=input_path,
            output_dir=output_dir,
            signatures=args.signatures,
            batch_size=args.batch_size,
            n_rand=args.n_rand,
            seed=args.seed,
        )

        if args.validate:
            for sig_name, activity_path in output_paths.items():
                validation_path = output_dir / f"{input_path.stem}_{sig_name}_validation.csv"
                validate_expression_vs_activity(input_path, activity_path, validation_path)

        return

    # No arguments - print help
    parser.print_help()
    print("\n")
    list_configs()


if __name__ == "__main__":
    main()
