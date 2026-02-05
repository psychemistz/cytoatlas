#!/usr/bin/env python3
"""
Extract scAtlas gene expression with boxplot statistics.
Uses cellType1 (433 cell types) and tissue (organ) for filtering.
Computes quartile stats for boxplots on expressing cells only.
"""

import gc
import json
import logging
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data paths
SCATLAS_PATH = '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad'
OUTPUT_DIR = Path('/vf/users/parks34/projects/2secactpy/visualization/data')

# Column names
CELLTYPE_COL = 'cellType1'  # 468 cell types
ORGAN_COL = 'tissue'  # Organ/tissue column

# Sampling parameters
SAMPLE_SIZE = 300  # Cells per cell type per organ
CELL_BATCH_SIZE = 50

# Key cytokine/immune genes
KEY_GENES = [
    'IFNG', 'IFNA1', 'IFNB1', 'IFNL1',
    'IL1A', 'IL1B', 'IL2', 'IL4', 'IL5', 'IL6', 'IL7', 'IL10', 'IL12A', 'IL12B',
    'IL13', 'IL15', 'IL17A', 'IL17F', 'IL18', 'IL21', 'IL22', 'IL23A', 'IL27',
    'TNF', 'TNFA', 'LTA', 'LTB', 'FASLG', 'TRAIL', 'CD40LG',
    'CCL2', 'CCL3', 'CCL4', 'CCL5', 'CCL7', 'CCL8', 'CCL11', 'CCL17', 'CCL19', 'CCL20', 'CCL21', 'CCL22',
    'CXCL1', 'CXCL2', 'CXCL3', 'CXCL8', 'CXCL9', 'CXCL10', 'CXCL11', 'CXCL12', 'CXCL13',
    'TGFB1', 'TGFB2', 'TGFB3', 'VEGFA', 'EGF', 'FGF2', 'PDGFA', 'PDGFB',
    'CSF1', 'CSF2', 'CSF3',
    'CD274', 'PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT',
    'GZMA', 'GZMB', 'GZMK', 'PRF1', 'GNLY',
    'FOXP3', 'RORC', 'TBX21', 'GATA3',
]


def decode_if_bytes(val):
    if isinstance(val, bytes):
        return val.decode()
    return val


def load_categorical_column(h5_file, col_name: str) -> np.ndarray:
    """Load a categorical column from obs."""
    obs = h5_file['obs']
    col_data = obs[col_name]

    if 'categories' in col_data and 'codes' in col_data:
        categories = [decode_if_bytes(c) for c in col_data['categories'][:]]
        codes = col_data['codes'][:]
        return np.array([categories[c] for c in codes])
    else:
        values = col_data[:]
        if hasattr(values[0], 'decode'):
            return np.array([decode_if_bytes(v) for v in values])
        return values


def load_gene_symbols(h5_file) -> list:
    """Load gene symbols from var."""
    var = h5_file['var']

    # Try different column names for gene symbols
    for col in ['_index', 'gene_symbol', 'gene_name', 'symbol']:
        if col in var or (col == '_index' and '_index' in var.attrs):
            if col == '_index':
                index_col = var.attrs.get('_index', '_index')
                if isinstance(index_col, bytes):
                    index_col = index_col.decode()
                if index_col in var:
                    col_data = var[index_col]
                else:
                    continue
            else:
                col_data = var[col]

            if 'categories' in col_data:
                cats = [decode_if_bytes(c) for c in col_data['categories'][:]]
                codes = col_data['codes'][:]
                return [cats[c] for c in codes]
            else:
                return [decode_if_bytes(v) for v in col_data[:]]

    return None


def get_matrix_shape(h5_file) -> tuple:
    if 'X' in h5_file:
        X = h5_file['X']
        if 'shape' in X.attrs:
            return tuple(X.attrs['shape'])
        elif hasattr(X, 'shape'):
            return X.shape
    return None, None


def read_sparse_rows(h5_file, row_indices, n_cols, gene_indices=None) -> np.ndarray:
    """Read specific rows from sparse matrix."""
    X = h5_file['X']

    if 'indptr' in X:
        indptr = X['indptr'][:]

        if gene_indices is not None:
            out_cols = len(gene_indices)
            gene_set = set(gene_indices)
            gene_idx_map = {g: i for i, g in enumerate(gene_indices)}
        else:
            out_cols = n_cols

        result = np.zeros((len(row_indices), out_cols), dtype=np.float32)

        for i, row_idx in enumerate(row_indices):
            start = indptr[row_idx]
            end = indptr[row_idx + 1]
            if start == end:
                continue

            row_indices_sparse = X['indices'][start:end]
            row_data = X['data'][start:end]

            if gene_indices is not None:
                for col, val in zip(row_indices_sparse, row_data):
                    if col in gene_set:
                        result[i, gene_idx_map[col]] = val
            else:
                result[i, row_indices_sparse] = row_data

        return result
    else:
        if gene_indices is not None:
            return X[row_indices, :][:, gene_indices]
        return X[row_indices, :]


def compute_boxplot_stats(values: np.ndarray) -> dict:
    """Compute boxplot statistics on expressing cells only."""
    expressing_values = values[values > 0]
    n_expressing = len(expressing_values)
    n_total = len(values)
    pct_expressed = n_expressing / n_total * 100 if n_total > 0 else 0

    if n_expressing < 3:
        return None

    return {
        'min': float(np.min(expressing_values)),
        'q1': float(np.percentile(expressing_values, 25)),
        'median': float(np.median(expressing_values)),
        'q3': float(np.percentile(expressing_values, 75)),
        'max': float(np.max(expressing_values)),
        'mean': float(np.mean(expressing_values)),
        'mean_all': float(np.mean(values)),
        'std': float(np.std(expressing_values)),
        'n_expressing': n_expressing,
        'n_total': n_total,
        'pct_expressed': round(pct_expressed, 2),
    }


def extract_scatlas_expression():
    """Extract scAtlas gene expression with boxplot statistics."""
    logger.info("=" * 60)
    logger.info("Extracting scAtlas gene expression with boxplot stats")
    logger.info(f"Using cell type column: {CELLTYPE_COL}")
    logger.info(f"Using organ column: {ORGAN_COL}")

    results = []
    boxplot_results = []

    with h5py.File(SCATLAS_PATH, 'r') as f:
        n_cells, n_genes = get_matrix_shape(f)
        logger.info(f"Shape: {n_cells:,} cells × {n_genes:,} genes")

        # Load cell types and organs
        cell_types = load_categorical_column(f, CELLTYPE_COL)
        organs = load_categorical_column(f, ORGAN_COL)
        unique_organs = np.unique(organs)
        logger.info(f"Found {len(unique_organs)} unique organs")

        # Load gene symbols
        gene_symbols = load_gene_symbols(f)
        if gene_symbols is None:
            logger.error("Could not load gene symbols")
            return pd.DataFrame(), pd.DataFrame()
        logger.info(f"Loaded {len(gene_symbols)} gene symbols")

        # Create gene name to index mapping
        gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}

        # Get indices of key genes
        key_gene_indices = []
        key_gene_names = []
        for gene in KEY_GENES:
            if gene in gene_to_idx:
                key_gene_indices.append(gene_to_idx[gene])
                key_gene_names.append(gene)
        logger.info(f"Found {len(key_gene_indices)} key genes in data")

        if not key_gene_indices:
            logger.error("No key genes found in data")
            return pd.DataFrame(), pd.DataFrame()

        # Process each organ
        for organ_idx, organ in enumerate(unique_organs):
            organ_mask = (organs == organ)
            organ_cell_types = cell_types[organ_mask]
            unique_cts_in_organ = np.unique(organ_cell_types)

            logger.info(f"Processing organ {organ_idx + 1}/{len(unique_organs)}: {organ} ({len(unique_cts_in_organ)} cell types)")

            # Process each cell type within this organ
            for ct in unique_cts_in_organ:
                ct_mask = organ_mask & (cell_types == ct)
                cell_indices = np.where(ct_mask)[0]
                n_total = len(cell_indices)

                if n_total < 10:
                    continue

                n_sample = min(SAMPLE_SIZE, n_total)
                sampled_indices = np.random.choice(cell_indices, size=n_sample, replace=False)
                sampled_indices = np.sort(sampled_indices)

                # Read expression for key genes only
                X_accumulated = []
                for batch_start in range(0, len(sampled_indices), CELL_BATCH_SIZE):
                    batch_end = min(batch_start + CELL_BATCH_SIZE, len(sampled_indices))
                    batch_indices = sampled_indices[batch_start:batch_end]
                    X_batch = read_sparse_rows(f, batch_indices, n_genes, gene_indices=key_gene_indices)
                    X_accumulated.append(X_batch)

                X_sample = np.vstack(X_accumulated)
                del X_accumulated

                # Normalize
                lib_sizes = X_sample.sum(axis=1, keepdims=True)
                lib_sizes = np.where(lib_sizes == 0, 1, lib_sizes)
                X_normalized = np.log1p(X_sample / lib_sizes * 10000)

                # Calculate stats for key genes
                for i, gene_idx in enumerate(key_gene_indices):
                    gene_name = key_gene_names[i]
                    gene_expr = X_normalized[:, i]
                    raw_expr = X_sample[:, i]

                    mean_expr = float(np.mean(gene_expr))
                    pct_expressed = float(np.sum(raw_expr > 0) / len(raw_expr) * 100)

                    # Cell type name includes organ for scAtlas
                    cell_type_with_organ = f"{organ}: {ct}"

                    results.append({
                        'gene': gene_name,
                        'cell_type': cell_type_with_organ,
                        'organ': organ,
                        'base_cell_type': str(ct),
                        'mean_expression': round(mean_expr, 4),
                        'pct_expressed': round(pct_expressed, 2),
                        'n_cells': int(n_total),
                        'atlas': 'scAtlas_Normal',
                    })

                    bp_stats = compute_boxplot_stats(gene_expr)
                    if bp_stats:
                        boxplot_results.append({
                            'gene': gene_name,
                            'cell_type': cell_type_with_organ,
                            'organ': organ,
                            'base_cell_type': str(ct),
                            'atlas': 'scAtlas_Normal',
                            **bp_stats,
                            'n': bp_stats.get('n_expressing', n_sample),
                        })

                del X_sample, X_normalized
                gc.collect()

    logger.info(f"Extracted {len(results)} mean expression records")
    logger.info(f"Extracted {len(boxplot_results)} boxplot records")

    return pd.DataFrame(results), pd.DataFrame(boxplot_results)


def main():
    logger.info("Starting scAtlas gene expression extraction")

    expr_df, boxplot_df = extract_scatlas_expression()

    if expr_df.empty:
        logger.error("No data extracted")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Update gene_expression.json
    logger.info("Updating gene_expression.json with scAtlas data...")
    existing_file = OUTPUT_DIR / 'gene_expression.json'
    if existing_file.exists():
        with open(existing_file) as f:
            existing_data = json.load(f)
        # Remove old scAtlas data
        existing_data = [d for d in existing_data if d['atlas'] not in ('scAtlas', 'scAtlas_Normal')]
        logger.info(f"Removed old scAtlas data, keeping {len(existing_data)} records")
    else:
        existing_data = []

    new_data = existing_data + expr_df.to_dict(orient='records')
    with open(existing_file, 'w') as f:
        json.dump(new_data, f)
    logger.info(f"Saved {len(new_data)} total records to gene_expression.json")

    # Update expression_boxplot.json
    boxplot_file = OUTPUT_DIR / 'expression_boxplot.json'
    if boxplot_file.exists():
        with open(boxplot_file) as f:
            existing_boxplot = json.load(f)
        existing_boxplot = [d for d in existing_boxplot if d.get('atlas') not in ('scAtlas', 'scAtlas_Normal')]
    else:
        existing_boxplot = []

    new_boxplot = existing_boxplot + boxplot_df.to_dict(orient='records')
    with open(boxplot_file, 'w') as f:
        json.dump(new_boxplot, f)
    logger.info(f"Saved {len(new_boxplot)} boxplot records to expression_boxplot.json")

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info(f"scAtlas organs: {expr_df['organ'].nunique()}")
    logger.info(f"scAtlas cell types: {expr_df['cell_type'].nunique()}")
    logger.info(f"Genes with data: {expr_df['gene'].nunique()}")

    return 0


if __name__ == '__main__':
    exit(main())
