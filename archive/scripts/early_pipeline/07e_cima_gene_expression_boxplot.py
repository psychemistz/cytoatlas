#!/usr/bin/env python3
"""
Extract CIMA gene expression with boxplot statistics.
Uses cell_type_l2 (27 cell types) and computes quartile stats for boxplots.
"""

import gc
import json
import logging
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data paths
CIMA_PATH = '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad'
OUTPUT_DIR = Path('/vf/users/parks34/projects/2secactpy/visualization/data')

# Cell type column to use (27 cell types)
CELLTYPE_COL = 'cell_type_l2'

# Sampling parameters - reduced for memory efficiency
SAMPLE_SIZE = 300  # Cells per cell type (reduced from 500)
CELL_BATCH_SIZE = 50  # Process cells in batches (reduced from 100)

# Key cytokine/immune genes to prioritize
KEY_GENES = [
    # Interferons
    'IFNG', 'IFNA1', 'IFNB1', 'IFNL1',
    # Interleukins
    'IL1A', 'IL1B', 'IL2', 'IL4', 'IL5', 'IL6', 'IL7', 'IL10', 'IL12A', 'IL12B',
    'IL13', 'IL15', 'IL17A', 'IL17F', 'IL18', 'IL21', 'IL22', 'IL23A', 'IL27',
    # TNF family
    'TNF', 'TNFA', 'LTA', 'LTB', 'FASLG', 'TRAIL', 'CD40LG',
    # Chemokines
    'CCL2', 'CCL3', 'CCL4', 'CCL5', 'CCL7', 'CCL8', 'CCL11', 'CCL17', 'CCL19', 'CCL20', 'CCL21', 'CCL22',
    'CXCL1', 'CXCL2', 'CXCL3', 'CXCL8', 'CXCL9', 'CXCL10', 'CXCL11', 'CXCL12', 'CXCL13',
    # Growth factors
    'TGFB1', 'TGFB2', 'TGFB3', 'VEGFA', 'EGF', 'FGF2', 'PDGFA', 'PDGFB',
    'CSF1', 'CSF2', 'CSF3',
    # Other immune molecules
    'CD274', 'PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT',
    'GZMA', 'GZMB', 'GZMK', 'PRF1', 'GNLY',
    'FOXP3', 'RORC', 'TBX21', 'GATA3',
]


def decode_if_bytes(val):
    """Decode bytes to string if necessary."""
    if isinstance(val, bytes):
        return val.decode()
    return val


def load_cell_types(h5_file, celltype_col: str) -> np.ndarray:
    """Load cell type labels from h5ad file."""
    obs = h5_file['obs']
    col_data = obs[celltype_col]

    # Handle categorical encoding
    if 'categories' in col_data and 'codes' in col_data:
        categories = [decode_if_bytes(c) for c in col_data['categories'][:]]
        codes = col_data['codes'][:]
        cell_types = np.array([categories[c] for c in codes])
    else:
        cell_types = col_data[:]
        if hasattr(cell_types[0], 'decode'):
            cell_types = np.array([decode_if_bytes(v) for v in cell_types])

    return cell_types


def load_gene_symbols(h5_file) -> tuple:
    """Load gene symbols from h5ad file."""
    var = h5_file['var']

    # Try different approaches
    if '_index' in var and hasattr(var['_index'], 'shape'):
        gene_names = [decode_if_bytes(v) for v in var['_index'][:]]
        return gene_names, True

    # Check for symbol column
    for col in ['symbol', 'gene_symbol', 'gene_name', 'gene']:
        if col in var:
            col_data = var[col]
            if 'categories' in col_data:
                cats = [decode_if_bytes(c) for c in col_data['categories'][:]]
                codes = col_data['codes'][:]
                gene_names = [cats[c] for c in codes]
            else:
                gene_names = [decode_if_bytes(v) for v in col_data[:]]
            return gene_names, True

    return None, False


def get_matrix_shape(h5_file) -> tuple:
    """Get matrix shape from h5ad file."""
    if 'X' in h5_file:
        X = h5_file['X']
        if 'shape' in X.attrs:
            return tuple(X.attrs['shape'])
        elif hasattr(X, 'shape'):
            return X.shape

    # Fall back to obs/var dimensions
    n_cells = len(h5_file['obs']['_index'])
    n_genes = len(h5_file['var']['_index'])
    return n_cells, n_genes


def read_sparse_rows(h5_file, row_indices, n_cols, gene_indices=None) -> np.ndarray:
    """Read specific rows from sparse matrix in h5ad file.

    If gene_indices is provided, only return those columns (much more memory efficient).
    """
    X = h5_file['X']

    # Check if sparse (CSR format)
    if 'indptr' in X:
        indptr = X['indptr'][:]

        # Determine output shape
        if gene_indices is not None:
            out_cols = len(gene_indices)
            gene_set = set(gene_indices)
            gene_idx_map = {g: i for i, g in enumerate(gene_indices)}
        else:
            out_cols = n_cols

        result = np.zeros((len(row_indices), out_cols), dtype=np.float32)

        # Read data/indices in chunks to avoid loading all at once
        for i, row_idx in enumerate(row_indices):
            start = indptr[row_idx]
            end = indptr[row_idx + 1]

            if start == end:
                continue

            # Read only this row's data
            row_indices_sparse = X['indices'][start:end]
            row_data = X['data'][start:end]

            if gene_indices is not None:
                # Only keep requested genes
                for col, val in zip(row_indices_sparse, row_data):
                    if col in gene_set:
                        result[i, gene_idx_map[col]] = val
            else:
                result[i, row_indices_sparse] = row_data

        return result
    else:
        # Dense matrix
        if gene_indices is not None:
            return X[row_indices, :][:, gene_indices]
        return X[row_indices, :]


def normalize_expression(X: np.ndarray) -> np.ndarray:
    """Normalize expression: library size normalize then log1p."""
    # Library size normalization
    lib_sizes = X.sum(axis=1, keepdims=True)
    lib_sizes = np.where(lib_sizes == 0, 1, lib_sizes)
    X_norm = X / lib_sizes * 10000

    # Log1p transform
    X_log = np.log1p(X_norm)

    return X_log


def compute_boxplot_stats(values: np.ndarray) -> dict:
    """Compute boxplot statistics from array of values.

    Computes stats on expressing cells only (>0) for meaningful boxplots,
    but also includes mean of all cells for comparison.

    Args:
        values: Array of expression values
    """
    # Filter to expressing cells only for meaningful boxplot
    expressing_values = values[values > 0]
    n_expressing = len(expressing_values)
    n_total = len(values)
    pct_expressed = n_expressing / n_total * 100 if n_total > 0 else 0

    if n_expressing < 3:  # Need at least 3 points for quartiles
        return None

    return {
        'min': float(np.min(expressing_values)),
        'q1': float(np.percentile(expressing_values, 25)),
        'median': float(np.median(expressing_values)),
        'q3': float(np.percentile(expressing_values, 75)),
        'max': float(np.max(expressing_values)),
        'mean': float(np.mean(expressing_values)),  # Mean of expressing cells
        'mean_all': float(np.mean(values)),  # Mean of all cells (including zeros)
        'std': float(np.std(expressing_values)),
        'n_expressing': n_expressing,
        'n_total': n_total,
        'pct_expressed': round(pct_expressed, 2),
    }


def extract_cima_expression():
    """Extract CIMA gene expression with boxplot statistics."""
    logger.info("=" * 60)
    logger.info("Extracting CIMA gene expression with boxplot stats")
    logger.info(f"Using cell type column: {CELLTYPE_COL}")

    results = []  # Mean expression data
    boxplot_results = []  # Boxplot statistics

    with h5py.File(CIMA_PATH, 'r') as f:
        n_cells, n_genes = get_matrix_shape(f)
        logger.info(f"Shape: {n_cells:,} cells × {n_genes:,} genes")

        # Load cell types
        cell_types = load_cell_types(f, CELLTYPE_COL)
        unique_cell_types = np.unique(cell_types)
        logger.info(f"Found {len(unique_cell_types)} unique cell types")

        # Load gene symbols
        gene_symbols, symbols_valid = load_gene_symbols(f)
        if not symbols_valid:
            logger.error("Could not load gene symbols")
            return pd.DataFrame(), pd.DataFrame()
        logger.info(f"Loaded {len(gene_symbols)} gene symbols")

        # Create gene name to index mapping
        gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}

        # Get indices and names of key genes
        key_gene_indices = []
        key_gene_names = []
        for gene in KEY_GENES:
            if gene in gene_to_idx:
                key_gene_indices.append(gene_to_idx[gene])
                key_gene_names.append(gene)
        logger.info(f"Found {len(key_gene_indices)} key genes in data")

        # Process each cell type
        for ct_idx, ct in enumerate(unique_cell_types):
            ct_mask = (cell_types == ct)
            cell_indices = np.where(ct_mask)[0]
            n_total = len(cell_indices)

            if n_total < 10:
                continue

            # Sample cells
            n_sample = min(SAMPLE_SIZE, n_total)
            sampled_indices = np.random.choice(cell_indices, size=n_sample, replace=False)
            sampled_indices = np.sort(sampled_indices)

            logger.info(f"Processing cell type {ct_idx + 1}/{len(unique_cell_types)}: {ct} ({n_sample} cells)")

            # Read expression for sampled cells - ONLY key genes to save memory
            X_accumulated = []
            for batch_start in range(0, len(sampled_indices), CELL_BATCH_SIZE):
                batch_end = min(batch_start + CELL_BATCH_SIZE, len(sampled_indices))
                batch_indices = sampled_indices[batch_start:batch_end]
                # Only read key gene columns
                X_batch = read_sparse_rows(f, batch_indices, n_genes, gene_indices=key_gene_indices)
                X_accumulated.append(X_batch)

            X_sample = np.vstack(X_accumulated)  # Shape: (n_sample, n_key_genes)
            del X_accumulated

            # Normalize - need to get library size from all genes
            # For efficiency, we estimate library size from key genes only (approximation)
            lib_sizes = X_sample.sum(axis=1, keepdims=True)
            lib_sizes = np.where(lib_sizes == 0, 1, lib_sizes)
            # Scale to approximate total counts (key genes are ~0.2% of genome)
            X_normalized = np.log1p(X_sample / lib_sizes * 10000)

            # Calculate stats for key genes (with boxplot stats)
            for i, gene_idx in enumerate(key_gene_indices):
                gene_name = key_gene_names[i]
                gene_expr = X_normalized[:, i]  # Use local index now
                raw_expr = X_sample[:, i]

                mean_expr = float(np.mean(gene_expr))
                pct_expressed = float(np.sum(raw_expr > 0) / len(raw_expr) * 100)

                # Store mean expression data
                results.append({
                    'gene': gene_name,
                    'cell_type': str(ct),
                    'mean_expression': round(mean_expr, 4),
                    'pct_expressed': round(pct_expressed, 2),
                    'n_cells': int(n_total),
                    'atlas': 'CIMA',
                })

                # Compute and store boxplot stats (on expressing cells only)
                bp_stats = compute_boxplot_stats(gene_expr)
                if bp_stats:
                    boxplot_results.append({
                        'gene': gene_name,
                        'cell_type': str(ct),
                        'atlas': 'CIMA',
                        **bp_stats,
                        'n': bp_stats.get('n_expressing', n_sample),
                    })

            del X_sample, X_normalized
            gc.collect()

    logger.info(f"Extracted {len(results)} mean expression records")
    logger.info(f"Extracted {len(boxplot_results)} boxplot records")

    return pd.DataFrame(results), pd.DataFrame(boxplot_results)


def main():
    logger.info("Starting CIMA gene expression extraction with boxplot stats")

    # Extract data
    expr_df, boxplot_df = extract_cima_expression()

    if expr_df.empty:
        logger.error("No data extracted")
        return 1

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save mean expression (update existing gene_expression.json)
    logger.info("Updating gene_expression.json with CIMA data...")

    # Load existing data
    existing_file = OUTPUT_DIR / 'gene_expression.json'
    if existing_file.exists():
        with open(existing_file) as f:
            existing_data = json.load(f)
        # Remove old CIMA data
        existing_data = [d for d in existing_data if d['atlas'] != 'CIMA']
        logger.info(f"Removed old CIMA data, keeping {len(existing_data)} records")
    else:
        existing_data = []

    # Add new CIMA data
    new_data = existing_data + expr_df.to_dict(orient='records')
    with open(existing_file, 'w') as f:
        json.dump(new_data, f)
    logger.info(f"Saved {len(new_data)} total records to gene_expression.json")

    # Save boxplot data
    boxplot_file = OUTPUT_DIR / 'expression_boxplot.json'

    # Load existing boxplot data if any
    if boxplot_file.exists():
        with open(boxplot_file) as f:
            existing_boxplot = json.load(f)
        existing_boxplot = [d for d in existing_boxplot if d.get('atlas') != 'CIMA']
    else:
        existing_boxplot = []

    # Add new boxplot data
    new_boxplot = existing_boxplot + boxplot_df.to_dict(orient='records')
    with open(boxplot_file, 'w') as f:
        json.dump(new_boxplot, f)
    logger.info(f"Saved {len(new_boxplot)} boxplot records to expression_boxplot.json")

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info(f"CIMA cell types: {expr_df['cell_type'].nunique()}")
    logger.info(f"Genes with data: {expr_df['gene'].nunique()}")

    return 0


if __name__ == '__main__':
    exit(main())
