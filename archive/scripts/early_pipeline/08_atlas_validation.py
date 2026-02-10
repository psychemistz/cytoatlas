#!/usr/bin/env python3
"""
Atlas-Level Validation: Pseudobulk Expression vs Activity Prediction.

Workflow:
1. Load single-cell gene expression from H5AD
2. Aggregate to pseudobulk per cell type (sum counts across cells)
3. Normalize pseudobulk (CPM + log1p)
4. Compute activity using CytoSig/SecAct ridge regression
5. Compute mean signature gene expression per cell type
6. Output: JSON with expression and activity for scatter plot validation

Usage:
    python scripts/08_atlas_validation.py --atlas cima
    python scripts/08_atlas_validation.py --atlas all
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp


# Data paths from CLAUDE.md
H5AD_PATHS = {
    "cima": "/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad",
    "inflammation": "/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad",
    "scatlas": "/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad",
}

# Cell type column names by atlas
CELLTYPE_COLS = {
    "cima": "cell_type_l2",
    "inflammation": "Level2",
    "scatlas": "cellType2",
}

# Gene symbol column (if var_names are not symbols)
GENE_SYMBOL_COLS = {
    "cima": None,  # var_names are symbols
    "inflammation": "symbol",  # Need to use var['symbol']
    "scatlas": None,  # var_names are symbols
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


def aggregate_pseudobulk_by_celltype(
    adata,
    ct_col: str,
    gene_symbol_col: Optional[str] = None,
    batch_size: int = 50000,
) -> pd.DataFrame:
    """
    Aggregate single-cell expression to pseudobulk per cell type.

    Returns:
        DataFrame with genes as rows, cell types as columns (summed counts)
    """
    import anndata as ad

    log(f"  Aggregating to pseudobulk by {ct_col}...")

    # Get cell types
    cell_types = adata.obs[ct_col].unique().tolist()
    log(f"  Cell types: {len(cell_types)}")

    # Get gene names
    if gene_symbol_col and gene_symbol_col in adata.var.columns:
        gene_names = adata.var[gene_symbol_col].tolist()
    else:
        gene_names = list(adata.var_names)

    n_genes = len(gene_names)
    n_cells = adata.shape[0]

    # Use raw counts if available
    if 'counts' in adata.layers:
        log("  Using layers['counts'] for raw counts")
        use_layer = 'counts'
    else:
        log("  Using .X")
        use_layer = None

    # Aggregate by cell type
    pseudobulk_dict = {}

    for ct in cell_types:
        ct_mask = adata.obs[ct_col] == ct
        ct_indices = np.where(ct_mask)[0]
        n_ct_cells = len(ct_indices)

        if n_ct_cells < 10:
            log(f"    Skipping {ct} (only {n_ct_cells} cells)")
            continue

        log(f"    {ct}: {n_ct_cells:,} cells")

        # Sum counts in batches to manage memory
        ct_sum = np.zeros(n_genes)

        for i in range(0, len(ct_indices), batch_size):
            batch_idx = ct_indices[i:i+batch_size]

            if use_layer:
                batch_data = adata.layers[use_layer][batch_idx, :]
            else:
                batch_data = adata.X[batch_idx, :]

            if sp.issparse(batch_data):
                batch_sum = np.asarray(batch_data.sum(axis=0)).ravel()
            else:
                batch_sum = batch_data.sum(axis=0)

            ct_sum += batch_sum

        pseudobulk_dict[ct] = ct_sum

    # Create DataFrame
    pseudobulk_df = pd.DataFrame(pseudobulk_dict, index=gene_names)
    log(f"  Pseudobulk shape: {pseudobulk_df.shape}")

    return pseudobulk_df


def normalize_pseudobulk(pseudobulk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize pseudobulk: CPM + log1p transformation.

    Args:
        pseudobulk_df: genes x cell_types (raw counts)

    Returns:
        Normalized DataFrame (genes x cell_types)
    """
    log("  Normalizing pseudobulk (CPM + log1p)...")

    # CPM normalization (per cell type / column)
    total_counts = pseudobulk_df.sum(axis=0)
    cpm = pseudobulk_df.div(total_counts, axis=1) * 1e6

    # Log1p transformation
    log_cpm = np.log1p(cpm)

    return log_cpm


def compute_activity(
    expr_df: pd.DataFrame,
    sig_matrix: pd.DataFrame,
    signature_type: str,
) -> pd.DataFrame:
    """
    Compute activity scores using ridge regression.

    Args:
        expr_df: genes x cell_types (normalized expression)
        sig_matrix: genes x signatures

    Returns:
        DataFrame: signatures x cell_types (activity z-scores)
    """
    from secactpy import ridge

    log(f"  Computing {signature_type} activity...")

    # Match genes
    common_genes = expr_df.index.intersection(sig_matrix.index)
    log(f"    Common genes: {len(common_genes)}")

    if len(common_genes) < 10:
        log(f"    Warning: Too few common genes ({len(common_genes)})")
        return pd.DataFrame()

    # Subset to common genes
    expr_matched = expr_df.loc[common_genes]
    sig_matched = sig_matrix.loc[common_genes]

    # Run ridge regression
    # X = signature matrix (genes x signatures)
    # Y = expression matrix (genes x samples/cell_types)
    result = ridge(
        X=sig_matched.values,
        Y=expr_matched.values,
        n_rand=1000,
        seed=42,
        verbose=False,
    )

    # Create DataFrame (result is signatures x samples)
    activity_df = pd.DataFrame(
        result['zscore'],
        index=sig_matrix.columns,
        columns=expr_df.columns,
    )

    log(f"    Activity shape: {activity_df.shape}")

    return activity_df


def compute_target_gene_expression(
    expr_df: pd.DataFrame,
    sig_matrix: pd.DataFrame,
    signature_type: str,
    zscore: bool = True,
) -> pd.DataFrame:
    """
    Get target gene expression per cell type for each signature.

    For CytoSig/SecAct, the signature name is often the target gene name.
    E.g., IFNG signature -> IFNG gene expression
          TNF signature -> TNF gene expression (or TNFA -> TNF)

    Args:
        expr_df: genes x cell_types (normalized expression)
        sig_matrix: genes x signatures
        signature_type: CytoSig or SecAct
        zscore: If True, z-score expression across cell types (default: True)

    Returns:
        DataFrame: signatures x cell_types (target gene expression, z-scored if enabled)
    """
    log(f"  Getting target gene expression (zscore={zscore})...")

    # Mapping from signature name to gene symbol (for cases where they differ)
    sig_to_gene = {
        # CytoSig name -> HGNC symbol
        "TNFA": "TNF",
        "IFNG": "IFNG",
        "IFN1": "IFNA1",  # Type I interferon
        "IFNL": "IFNL1",  # Type III interferon
        "TGFB1": "TGFB1",
        "TGFB3": "TGFB3",
        "IL1A": "IL1A",
        "IL1B": "IL1B",
        "IL2": "IL2",
        "IL3": "IL3",
        "IL4": "IL4",
        "IL6": "IL6",
        "IL10": "IL10",
        "IL12": "IL12A",
        "IL13": "IL13",
        "IL15": "IL15",
        "IL17A": "IL17A",
        "IL21": "IL21",
        "IL22": "IL22",
        "IL27": "IL27",
        "IL36": "IL36A",
        "CXCL12": "CXCL12",
        "CD40L": "CD40LG",
        "LTA": "LTA",
        "TRAIL": "TNFSF10",
        "TWEAK": "TNFSF12",
        "VEGFA": "VEGFA",
        "EGF": "EGF",
        "FGF2": "FGF2",
        "HGF": "HGF",
        "BDNF": "BDNF",
        "GCSF": "CSF3",
        "GMCSF": "CSF2",
        "MCSF": "CSF1",
        "OSM": "OSM",
        "LIF": "LIF",
        "NO": None,  # Not a gene
        "BMP2": "BMP2",
        "BMP4": "BMP4",
        "BMP6": "BMP6",
        "GDF11": "GDF11",
        "WNT3A": "WNT3A",
        "Activin A": "INHBA",
    }

    results = {}
    found = 0
    not_found = []

    for sig in sig_matrix.columns:
        # Get gene name for this signature
        gene = sig_to_gene.get(sig, sig)  # Default: signature name = gene name

        if gene is None:
            continue

        # Check if gene exists in expression data
        if gene in expr_df.index:
            results[sig] = expr_df.loc[gene]
            found += 1
        else:
            # Try without version suffix, uppercase, etc.
            gene_upper = gene.upper()
            matching = [g for g in expr_df.index if g.upper() == gene_upper]
            if matching:
                results[sig] = expr_df.loc[matching[0]]
                found += 1
            else:
                not_found.append(sig)

    result_df = pd.DataFrame(results).T
    log(f"    Found target genes: {found}/{len(sig_matrix.columns)}")
    if not_found and len(not_found) <= 10:
        log(f"    Not found: {not_found}")

    # Z-score expression across cell types (for each signature/gene)
    if zscore and not result_df.empty:
        from scipy import stats as scipy_stats
        result_df = result_df.apply(lambda row: scipy_stats.zscore(row, nan_policy='omit'), axis=1)
        result_df = result_df.fillna(0)
        log(f"    Z-scored expression (mean-centered, unit variance)")

    return result_df


def run_atlas_validation(
    atlas: str,
    output_dir: Path,
    signature_types: List[str] = ["CytoSig", "SecAct"],
) -> Dict[str, Any]:
    """
    Run atlas-level validation for one atlas.

    Returns dict with validation data for each signature type.
    """
    import anndata as ad

    h5ad_path = H5AD_PATHS.get(atlas)
    if not h5ad_path or not Path(h5ad_path).exists():
        log(f"  H5AD not found: {h5ad_path}")
        return {}

    ct_col = CELLTYPE_COLS.get(atlas, "cell_type")
    gene_symbol_col = GENE_SYMBOL_COLS.get(atlas)

    log(f"  Loading {h5ad_path}...")
    adata = ad.read_h5ad(h5ad_path, backed='r')
    log(f"  Shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    # Aggregate to pseudobulk
    pseudobulk_raw = aggregate_pseudobulk_by_celltype(
        adata, ct_col, gene_symbol_col
    )

    # Free memory
    del adata
    gc.collect()

    # Normalize
    pseudobulk_norm = normalize_pseudobulk(pseudobulk_raw)

    # Process each signature type
    all_results = {}

    for sig_type in signature_types:
        log(f"\n  Processing {sig_type}...")

        sig_matrix = load_signature_matrix(sig_type)
        if sig_matrix is None:
            continue

        # Compute activity
        activity_df = compute_activity(pseudobulk_norm, sig_matrix, sig_type)
        if activity_df.empty:
            continue

        # Get target gene expression (e.g., IFNG expression for IFNG activity)
        expr_df = compute_target_gene_expression(pseudobulk_norm, sig_matrix, sig_type)
        if expr_df.empty:
            continue

        # Create validation data points
        # Each point: (signature, cell_type, expression, activity)
        points = []

        for sig in activity_df.index:
            if sig not in expr_df.index:
                continue

            for ct in activity_df.columns:
                if ct not in expr_df.columns:
                    continue

                points.append({
                    "signature": sig,
                    "cell_type": ct,
                    "expression": float(expr_df.loc[sig, ct]),
                    "activity": float(activity_df.loc[sig, ct]),
                })

        # Compute overall correlation
        if points:
            expr_vals = [p["expression"] for p in points]
            act_vals = [p["activity"] for p in points]

            from scipy import stats
            r, p_value = stats.pearsonr(expr_vals, act_vals)
            rho, rho_p = stats.spearmanr(expr_vals, act_vals)

            all_results[sig_type] = {
                "points": points,
                "n_signatures": len(activity_df.index),
                "n_cell_types": len(activity_df.columns),
                "n_points": len(points),
                "pearson_r": float(r),
                "pearson_p": float(p_value),
                "spearman_rho": float(rho),
                "spearman_p": float(rho_p),
            }

            log(f"    {sig_type}: {len(points)} points, r={r:.3f}, rho={rho:.3f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Atlas-level validation: pseudobulk expression vs activity"
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
        help="Signature type (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/vf/users/parks34/projects/2cytoatlas/visualization/data"),
        help="Output directory",
    )

    args = parser.parse_args()

    atlases = ["cima", "inflammation", "scatlas"] if args.atlas == "all" else [args.atlas]
    sig_types = ["CytoSig", "SecAct"] if args.signature_type == "all" else [args.signature_type]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for atlas in atlases:
        log(f"\n{'='*60}")
        log(f"ATLAS-LEVEL VALIDATION: {atlas.upper()}")
        log(f"{'='*60}")

        results = run_atlas_validation(atlas, args.output_dir, sig_types)

        if results:
            # Save results
            output_file = args.output_dir / f"{atlas}_atlas_validation.json"
            with open(output_file, "w") as f:
                json.dump({
                    "atlas": atlas,
                    "validation_level": "atlas",
                    "description": "Target gene expression vs activity correlation per cell type (e.g., IFNG expression vs IFNG activity)",
                    **results,
                }, f, indent=2)
            log(f"\nSaved to {output_file}")

    log("\nDone!")


if __name__ == "__main__":
    main()
