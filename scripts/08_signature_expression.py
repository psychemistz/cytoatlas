#!/usr/bin/env python3 -u
"""
Compute Mean Signature Gene Expression (CPM) per Cell Type for Validation.

This script computes the mean CPM expression of signature genes for each cell type,
which is used to validate inferred activity scores against actual expression.

For each signature (e.g., IFNG):
1. Get genes with non-zero weights in the signature matrix
2. Extract CPM expression values for those genes from the H5AD
3. Compute mean CPM per cell type

Output: JSON files with mean signature gene expression per cell type.

Usage:
    python scripts/08_signature_expression.py --atlas cima
    python scripts/08_signature_expression.py --atlas all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# Data paths from CLAUDE.md
H5AD_PATHS = {
    "cima": "/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad",
    "inflammation": "/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad",
    "scatlas": "/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad",
}

# Cell type column names by atlas
CELLTYPE_COLS = {
    "cima": "cell_type_l2",  # 27 intermediate cell types
    "inflammation": "cell_type",
    "scatlas": "cell_type_fine",
}


def log(msg: str) -> None:
    """Print with flush for real-time output."""
    print(msg, flush=True)


def load_signature_matrix(signature_type: str) -> Optional[pd.DataFrame]:
    """Load signature matrix (genes x signatures)."""
    try:
        if signature_type == "CytoSig":
            from secactpy import load_cytosig
            return load_cytosig()
        elif signature_type == "SecAct":
            from secactpy import load_secact
            return load_secact()
        else:
            log(f"  Unknown signature type: {signature_type}")
            return None
    except Exception as e:
        log(f"  Error loading {signature_type}: {e}")
    return None


def get_signature_genes(signature: str, sig_matrix: pd.DataFrame) -> List[str]:
    """Get genes with non-zero weights for a signature."""
    if signature not in sig_matrix.columns:
        return []
    weights = sig_matrix[signature]
    return weights[weights != 0].index.tolist()


def normalize_to_cpm(X: np.ndarray) -> np.ndarray:
    """Normalize expression matrix to CPM (counts per million)."""
    # Sum per cell (row)
    total_counts = X.sum(axis=1, keepdims=True)
    # Avoid division by zero
    total_counts = np.maximum(total_counts, 1)
    # CPM normalization
    cpm = (X / total_counts) * 1e6
    return cpm


def compute_signature_expression(
    atlas: str,
    signature_type: str,
    output_dir: Path,
    batch_size: int = 10000,
    max_cells_per_ct: int = 5000,
) -> Dict[str, Any]:
    """
    Compute mean signature gene expression (CPM) per cell type.

    Args:
        atlas: Atlas name (cima, inflammation, scatlas)
        signature_type: CytoSig, LinCytoSig, or SecAct
        output_dir: Output directory for JSON files
        batch_size: Number of cells to process at once
        max_cells_per_ct: Maximum cells to sample per cell type

    Returns:
        Dict with signature -> {cell_type -> {mean_cpm, n_cells, n_genes}}
    """
    import anndata as ad

    h5ad_path = H5AD_PATHS.get(atlas)
    if not h5ad_path or not Path(h5ad_path).exists():
        log(f"  H5AD not found: {h5ad_path}")
        return {}

    # Load signature matrix
    sig_matrix = load_signature_matrix(signature_type)
    if sig_matrix is None:
        return {}

    signatures = sig_matrix.columns.tolist()
    log(f"  {signature_type}: {len(signatures)} signatures")

    # Load H5AD in backed mode
    log(f"  Loading {h5ad_path}...")
    adata = ad.read_h5ad(h5ad_path, backed='r')
    log(f"  Shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    # Get cell type column
    ct_col = CELLTYPE_COLS.get(atlas, "cell_type")
    if ct_col not in adata.obs.columns:
        for col in ['cell_type', 'celltype', 'CellType', 'cell_type_fine']:
            if col in adata.obs.columns:
                ct_col = col
                break
    log(f"  Cell type column: {ct_col}")

    # Get available genes
    available_genes = set(adata.var_names)
    log(f"  Available genes: {len(available_genes):,}")

    # Find overlapping genes for each signature
    sig_gene_map = {}
    for sig in signatures:
        sig_genes = get_signature_genes(sig, sig_matrix)
        overlap = [g for g in sig_genes if g in available_genes]
        if overlap:
            sig_gene_map[sig] = overlap

    log(f"  Signatures with gene overlap: {len(sig_gene_map)}")

    # Get unique cell types
    cell_types = adata.obs[ct_col].unique().tolist()
    log(f"  Cell types: {len(cell_types)}")

    # Get all signature genes we need (union across all signatures)
    all_sig_genes = set()
    for genes in sig_gene_map.values():
        all_sig_genes.update(genes)
    all_sig_genes = sorted(all_sig_genes)
    gene_to_idx = {g: i for i, g in enumerate(all_sig_genes)}
    gene_indices = [adata.var_names.get_loc(g) for g in all_sig_genes]

    log(f"  Total signature genes to extract: {len(all_sig_genes)}")

    # Results: signature -> {cell_type -> {mean_cpm, n_cells, n_genes}}
    results = {sig: {} for sig in sig_gene_map}

    # Process each cell type
    for ct_idx, ct in enumerate(cell_types):
        log(f"  [{ct_idx+1}/{len(cell_types)}] Processing {ct}...")

        # Get cell indices for this cell type
        ct_mask = adata.obs[ct_col] == ct
        ct_indices = np.where(ct_mask)[0]
        n_cells_total = len(ct_indices)

        if n_cells_total < 10:
            log(f"    Skipping (only {n_cells_total} cells)")
            continue

        # Sample if too many cells
        if n_cells_total > max_cells_per_ct:
            np.random.seed(42)
            ct_indices = np.random.choice(ct_indices, max_cells_per_ct, replace=False)
            ct_indices = np.sort(ct_indices)
            log(f"    Sampled {max_cells_per_ct} from {n_cells_total} cells")

        n_cells = len(ct_indices)

        # Extract expression matrix for signature genes (cells x genes)
        # Use raw counts layer if available
        use_layer = 'counts' in adata.layers
        X = adata.layers['counts'] if use_layer else adata.X

        expr_chunks = []
        for i in range(0, len(ct_indices), batch_size):
            batch_indices = ct_indices[i:i+batch_size]
            chunk = X[batch_indices][:, gene_indices]
            if hasattr(chunk, 'toarray'):
                chunk = chunk.toarray()
            expr_chunks.append(chunk)

        expr_matrix = np.vstack(expr_chunks)  # cells x signature_genes

        if ct_idx == 0:
            log(f"    Using {'layers[counts]' if use_layer else '.X'}")

        # Check if data is already normalized (look at scale)
        max_val = np.max(expr_matrix)
        if max_val > 100:
            # Likely raw counts, normalize to CPM
            log(f"    Normalizing to CPM (max value: {max_val:.0f})")
            expr_cpm = normalize_to_cpm(expr_matrix)
        else:
            # Already normalized (log-normalized or similar)
            # Convert from log scale: exp(x) - 1 to get pseudo-counts, then CPM
            log(f"    Data appears log-normalized (max value: {max_val:.2f}), using as-is")
            expr_cpm = expr_matrix

        # Compute mean CPM for each signature
        for sig, sig_genes in sig_gene_map.items():
            sig_gene_idx = [gene_to_idx[g] for g in sig_genes if g in gene_to_idx]
            if not sig_gene_idx:
                continue

            # Mean CPM across signature genes for each cell, then mean across cells
            sig_expr = expr_cpm[:, sig_gene_idx]
            mean_per_cell = np.mean(sig_expr, axis=1)  # Mean across genes
            mean_cpm = float(np.mean(mean_per_cell))   # Mean across cells

            results[sig][str(ct)] = {
                "mean_cpm": mean_cpm,
                "n_cells": n_cells,
                "n_genes": len(sig_gene_idx),
            }

    # Clean up
    del adata
    import gc
    gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean signature gene expression (CPM) per cell type"
    )
    parser.add_argument(
        "--atlas",
        required=True,
        choices=["cima", "inflammation", "scatlas", "all"],
        help="Atlas to process",
    )
    parser.add_argument(
        "--signature-type",
        choices=["CytoSig", "SecAct", "all"],
        default="all",
        help="Signature type (default: all = CytoSig + SecAct)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/vf/users/parks34/projects/2secactpy/visualization/data"),
        help="Output directory",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=5000,
        help="Max cells to sample per cell type (default: 5000)",
    )

    args = parser.parse_args()

    atlases = ["cima", "inflammation", "scatlas"] if args.atlas == "all" else [args.atlas]
    sig_types = ["CytoSig", "SecAct"] if args.signature_type == "all" else [args.signature_type]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for atlas in atlases:
        log(f"\n{'='*60}")
        log(f"Processing {atlas.upper()}")
        log(f"{'='*60}")

        # Load existing data if present (to merge with new results)
        output_file = args.output_dir / f"{atlas}_signature_expression.json"
        if output_file.exists():
            log(f"  Loading existing data from {output_file}")
            with open(output_file) as f:
                all_results = json.load(f)
            log(f"  Existing signatures: {len(all_results)}")
        else:
            all_results = {}

        for sig_type in sig_types:
            log(f"\n{sig_type}:")

            results = compute_signature_expression(
                atlas=atlas,
                signature_type=sig_type,
                output_dir=args.output_dir,
                max_cells_per_ct=args.max_cells,
            )

            # Add signature type to results (merge with existing)
            for sig, ct_data in results.items():
                all_results[f"{sig_type}:{sig}"] = ct_data

        # Save combined results
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        log(f"\nSaved to {output_file}")

        # Summary
        n_sigs = len(all_results)
        n_ct = len(set(ct for data in all_results.values() for ct in data.keys()))
        log(f"Total: {n_sigs} signatures, {n_ct} cell types")

    log("\nDone!")


if __name__ == "__main__":
    main()
