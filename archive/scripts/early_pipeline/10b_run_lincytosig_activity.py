#!/usr/bin/env python3
"""
Run LinCytoSig activity inference on existing pseudobulk files.

This script is for adding LinCytoSig after pseudobulk is already generated.
"""

import argparse
import gc
import gzip
import sys
import time
from pathlib import Path

# Ensure line buffering
sys.stdout.reconfigure(line_buffering=True)

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "cytoatlas-pipeline/src"))

import anndata as ad
import numpy as np
import pandas as pd


OUTPUT_ROOT = Path("/vf/users/parks34/projects/2secactpy/results/atlas_validation")
LINCYTOSIG_PATH = Path("/vf/users/parks34/projects/1ridgesig/SecActpy-dev/secactpy/data/LinCytoSig.tsv.gz")


def log(msg: str):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)


def load_lincytosig() -> pd.DataFrame:
    """Load LinCytoSig signature matrix."""
    with gzip.open(LINCYTOSIG_PATH, 'rt') as f:
        return pd.read_csv(f, sep='\t', index_col=0)


def run_lincytosig_activity(
    pseudobulk_path: Path,
    output_dir: Path,
    atlas_name: str,
    level: str,
    use_gpu: bool = True,
) -> Path:
    """
    Run LinCytoSig activity inference on pseudobulk.

    Args:
        pseudobulk_path: Path to pseudobulk H5AD
        output_dir: Output directory
        atlas_name: Atlas name
        level: Annotation level
        use_gpu: Use GPU if available

    Returns:
        Output path
    """
    from secactpy.batch import ridge_batch

    log(f"  Loading pseudobulk: {pseudobulk_path.name}")
    pb_adata = ad.read_h5ad(pseudobulk_path)

    # Always use X (log1p CPM), not zscore layer
    expr_matrix = pb_adata.X

    gene_names = list(pb_adata.var_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{atlas_name}_{level}_lincytosig.h5ad"

    log(f"  Loading LinCytoSig...")
    sig_df = load_lincytosig()
    log(f"    Shape: {sig_df.shape}")

    # Match genes
    common_genes = list(set(gene_names) & set(sig_df.index))
    log(f"    Matched {len(common_genes)} genes")

    gene_idx = [gene_names.index(g) for g in common_genes]

    # Prepare matrices
    X = sig_df.loc[common_genes].values
    Y = expr_matrix[:, gene_idx].T

    # Fill NaN values in signature matrix (LinCytoSig has ~13% NaN)
    X = np.nan_to_num(X, nan=0.0)

    if hasattr(Y, 'toarray'):
        Y = Y.toarray()

    Y = np.asarray(Y, dtype=np.float64)
    Y_mean = Y.mean(axis=0, keepdims=True)
    Y_std = Y.std(axis=0, ddof=1, keepdims=True)
    Y_std[Y_std < 1e-10] = 1.0
    Y_scaled = (Y - Y_mean) / Y_std

    log(f"  Running ridge regression...")
    result = ridge_batch(
        X, Y_scaled,
        lambda_=5e5,
        n_rand=1000,
        seed=42,
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
    activity_adata.layers['beta'] = result['beta'].T.astype(np.float32)
    activity_adata.uns['signature'] = 'lincytosig'
    activity_adata.uns['level'] = level
    activity_adata.uns['atlas'] = atlas_name
    activity_adata.uns['n_matched_genes'] = len(common_genes)

    activity_adata.write_h5ad(output_path, compression='gzip')
    log(f"    Saved: {output_path.name}")

    del result, activity_adata
    gc.collect()

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Run LinCytoSig activity inference on existing pseudobulk"
    )
    parser.add_argument('--atlas', type=str, required=True,
                        help='Atlas name')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU')

    args = parser.parse_args()

    atlas_output_dir = OUTPUT_ROOT / args.atlas
    pseudobulk_dir = atlas_output_dir / "pseudobulk"
    activity_dir = atlas_output_dir / "activity_atlas"

    log("=" * 70)
    log(f"LINCYTOSIG ACTIVITY INFERENCE: {args.atlas.upper()}")
    log("=" * 70)

    # Find pseudobulk files
    pb_files = sorted(pseudobulk_dir.glob(f"{args.atlas.lower()}_pseudobulk_*.h5ad"))
    pb_files = [f for f in pb_files if 'resampled' not in f.name]

    if not pb_files:
        log(f"No pseudobulk files found in {pseudobulk_dir}")
        return

    log(f"Found {len(pb_files)} pseudobulk files")

    for pb_file in pb_files:
        level = pb_file.stem.split('_')[-1]
        log(f"\nProcessing {level}...")

        run_lincytosig_activity(
            pseudobulk_path=pb_file,
            output_dir=activity_dir,
            atlas_name=args.atlas,
            level=level,
            use_gpu=not args.no_gpu,
        )

    log("\n" + "=" * 70)
    log("COMPLETED")
    log("=" * 70)


if __name__ == "__main__":
    main()
