#!/usr/bin/env python3
"""
Extract gene expression from sampled cells for memory efficiency.

Samples cells within each cell type to estimate mean expression.
Applies consistent log1p(CPM) normalization across all atlases.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Key genes to extract
KEY_GENES = [
    'IFNG', 'TNF', 'IL1A', 'IL1B', 'IL2', 'IL4', 'IL6', 'IL10', 'IL12A', 'IL12B',
    'IL13', 'IL17A', 'IL17F', 'IL21', 'IL22', 'IL23A', 'IL27', 'IL33',
    'TGFB1', 'TGFB2', 'TGFB3', 'CSF1', 'CSF2', 'CSF3', 'CXCL8', 'CCL2', 'CCL3',
    'CCL4', 'CCL5', 'CXCL10', 'CXCL11', 'VEGFA', 'EGF', 'FGF2', 'HGF',
    'SPP1', 'MMP9', 'MMP2', 'TIMP1', 'GZMB', 'PRF1', 'FASLG',
    'CD274', 'PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT',
]

DATA_PATHS = {
    'cima': '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad',
    'inflammation_main': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad',
    'inflammation_val': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad',
    'inflammation_ext': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_external_afterQC.h5ad',
    'scatlas_normal': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad',
}

OUTPUT_DIR = Path('/vf/users/parks34/projects/2secactpy/visualization/data')

# Sample size per cell type (reduced for memory efficiency)
SAMPLE_SIZE = 2000

# Atlas-specific cell type column preferences
# Use finer annotations where available
ATLAS_CELLTYPE_COLS = {
    'CIMA': ['cell_type_l2', 'cell_type_l1'],  # cell_type_l2 has 27 types, l1 has 6
    'Inflammation': ['Level1', 'Level2', 'cell_type'],
    'scAtlas': ['cellType1', 'cellType2', 'cell_type'],
}


def get_cell_type_column(adata, atlas_name: str = None):
    """Find the cell type column, preferring finer annotations."""
    # Use atlas-specific preference if available
    if atlas_name and atlas_name in ATLAS_CELLTYPE_COLS:
        for col in ATLAS_CELLTYPE_COLS[atlas_name]:
            if col in adata.obs.columns:
                return col

    # Fallback to general search
    for col in ['cell_type', 'cell_type_l1', 'celltype', 'CellType', 'cell_type_fine',
                'cellType1', 'cellType2', 'ann1', 'majorCluster',
                'Level1', 'Level2']:
        if col in adata.obs.columns:
            return col
    return None


def detect_normalization(X_sample):
    """
    Detect if data is already log-normalized or raw counts.

    Returns: 'log_normalized', 'cpm', or 'raw_counts'
    """
    max_val = X_sample.max()
    mean_val = X_sample.mean()

    # Log-normalized data typically has max < 10 and mean < 1
    if max_val < 15 and mean_val < 2:
        return 'log_normalized'
    # CPM has values in thousands
    elif max_val > 100:
        return 'cpm' if mean_val > 10 else 'raw_counts'
    else:
        return 'raw_counts'


def normalize_expression(X_sample, detected_norm: str):
    """
    Normalize expression to log1p(CPM) scale.

    Target scale: log1p(CPM/100) to get values roughly 0-5 range
    """
    if detected_norm == 'log_normalized':
        # Already log-normalized, return as-is
        return X_sample
    elif detected_norm == 'cpm':
        # CPM -> log1p(CPM/100) for comparable scale
        return np.log1p(X_sample / 100)
    else:
        # Raw counts -> CPM -> log1p
        # Normalize per cell to CPM first
        row_sums = X_sample.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cpm = X_sample / row_sums * 1e6
        return np.log1p(cpm / 100)


def extract_sampled_expression(h5ad_path: str, genes: list[str], atlas_name: str) -> pd.DataFrame:
    """
    Extract gene expression by sampling cells per cell type.
    Applies consistent normalization across atlases.
    """
    import gc

    logger.info(f"Loading {atlas_name} from {h5ad_path}")

    # Load in backed mode
    adata = sc.read_h5ad(h5ad_path, backed='r')

    # Find genes - handle Ensembl IDs vs gene symbols
    available_genes = set(adata.var_names)
    genes_found = [g for g in genes if g in available_genes]

    # If var_names are Ensembl IDs, check 'symbol' column
    gene_name_map = {}  # Maps query gene to actual var_name
    if len(genes_found) == 0 and 'symbol' in adata.var.columns:
        logger.info("var_names are Ensembl IDs, using 'symbol' column for gene lookup")
        symbol_to_idx = {}
        for idx, symbol in zip(adata.var_names, adata.var['symbol'].values):
            if isinstance(symbol, bytes):
                symbol = symbol.decode()
            symbol_to_idx[symbol] = idx

        genes_found = [g for g in genes if g in symbol_to_idx]
        gene_name_map = {g: symbol_to_idx[g] for g in genes_found}
        logger.info(f"Found {len(genes_found)}/{len(genes)} genes via symbol column")
    else:
        gene_name_map = {g: g for g in genes_found}
        logger.info(f"Found {len(genes_found)}/{len(genes)} genes")

    if not genes_found:
        return pd.DataFrame()

    # Get cell type column (use atlas-specific preferences)
    cell_type_col = get_cell_type_column(adata, atlas_name)
    if not cell_type_col:
        logger.warning(f"No cell type column in {atlas_name}")
        return pd.DataFrame()

    logger.info(f"Using cell type column: {cell_type_col}")

    # Get unique cell types BEFORE subsetting (obs will change after subset)
    cell_types = adata.obs[cell_type_col].unique()
    cell_type_labels = adata.obs[cell_type_col].values.copy()  # Copy labels
    logger.info(f"Found {len(cell_types)} cell types")

    # Get the actual var_names to subset (may be Ensembl IDs)
    var_names_to_subset = [gene_name_map[g] for g in genes_found]

    # Subset to only the genes we need - this creates a view, not a copy
    # This is the key optimization: subset columns first, then slice rows
    logger.info(f"Subsetting to {len(genes_found)} genes...")
    adata_genes = adata[:, var_names_to_subset]
    logger.info("Gene subset complete")

    results = []
    np.random.seed(42)

    for ct_idx, ct in enumerate(cell_types):
        if ct_idx % 5 == 0:
            logger.info(f"Processing cell type {ct_idx+1}/{len(cell_types)}: {ct}")
            gc.collect()

        # Get cells of this type using the copied labels
        mask = (cell_type_labels == ct)
        cell_indices = np.where(mask)[0]
        n_total = len(cell_indices)

        if n_total < 10:
            continue

        # Sample cells
        n_sample = min(SAMPLE_SIZE, n_total)
        sampled_indices = np.random.choice(cell_indices, size=n_sample, replace=False)
        sampled_indices = np.sort(sampled_indices)

        # Read expression for sampled cells - now only reads the genes we need
        X_sample = adata_genes.X[sampled_indices, :]
        if hasattr(X_sample, 'toarray'):
            X_sample = X_sample.toarray()

        # Detect and apply normalization on first cell type
        if ct_idx == 0:
            detected_norm = detect_normalization(X_sample)
            logger.info(f"Detected normalization: {detected_norm}")

        # Apply consistent normalization
        X_normalized = normalize_expression(X_sample, detected_norm)

        # Calculate stats using normalized values
        for j, gene in enumerate(genes_found):
            expr_raw = X_sample[:, j]  # For pct_expressed calculation
            expr_norm = X_normalized[:, j]  # For mean expression
            mean_expr = float(np.mean(expr_norm))
            pct_expr = float(np.sum(expr_raw > 0) / len(expr_raw) * 100)

            results.append({
                'gene': gene,
                'cell_type': str(ct),
                'mean_expression': round(mean_expr, 4),
                'pct_expressed': round(pct_expr, 2),
                'n_cells': int(n_total),  # Report total, not sampled
                'atlas': atlas_name,
            })

        del X_sample
        gc.collect()

    del adata_genes
    gc.collect()

    return pd.DataFrame(results)


def main():
    logger.info("Starting sampled gene expression extraction")

    all_results = []

    # CIMA
    if Path(DATA_PATHS['cima']).exists():
        df = extract_sampled_expression(DATA_PATHS['cima'], KEY_GENES, 'CIMA')
        if not df.empty:
            all_results.append(df)

    # Inflammation Atlas (main, validation, external)
    inflam_dfs = []
    for name in ['inflammation_main', 'inflammation_val', 'inflammation_ext']:
        if Path(DATA_PATHS[name]).exists():
            df = extract_sampled_expression(DATA_PATHS[name], KEY_GENES, 'Inflammation')
            if not df.empty:
                inflam_dfs.append(df)

    if inflam_dfs:
        # Combine and aggregate inflammation data
        inflam_combined = pd.concat(inflam_dfs, ignore_index=True)
        # Group by gene and cell_type, aggregate
        inflam_agg = inflam_combined.groupby(['gene', 'cell_type', 'atlas']).agg({
            'mean_expression': 'mean',
            'pct_expressed': 'mean',
            'n_cells': 'sum',
        }).reset_index()
        all_results.append(inflam_agg)

    # scAtlas Normal
    if Path(DATA_PATHS['scatlas_normal']).exists():
        df = extract_sampled_expression(DATA_PATHS['scatlas_normal'], KEY_GENES, 'scAtlas')
        if not df.empty:
            all_results.append(df)

    if not all_results:
        logger.error("No data extracted")
        return 1

    combined = pd.concat(all_results, ignore_index=True)
    logger.info(f"Total records: {len(combined)}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / 'gene_expression.json', 'w') as f:
        json.dump(combined.to_dict(orient='records'), f)

    gene_dir = OUTPUT_DIR / 'genes'
    gene_dir.mkdir(exist_ok=True)

    for gene in combined['gene'].unique():
        gene_df = combined[combined['gene'] == gene]
        with open(gene_dir / f'{gene}.json', 'w') as f:
            json.dump(gene_df.to_dict(orient='records'), f)

    with open(OUTPUT_DIR / 'gene_list.json', 'w') as f:
        json.dump(sorted(combined['gene'].unique().tolist()), f)

    logger.info("Done!")
    return 0


if __name__ == '__main__':
    exit(main())
