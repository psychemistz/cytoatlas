#!/usr/bin/env python3
"""
Simple gene expression extraction using h5py for memory efficiency.

Extracts mean gene expression for a few key genes by cell type.
"""

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Just the most important genes
KEY_GENES = [
    'IFNG', 'TNF', 'IL1B', 'IL2', 'IL4', 'IL6', 'IL10', 'IL17A',
    'TGFB1', 'CCL2', 'CCL5', 'CXCL8', 'CXCL10', 'VEGFA',
    'GZMB', 'PRF1', 'CD274', 'PDCD1', 'SPP1', 'MMP9'
]

OUTPUT_DIR = Path('/vf/users/parks34/projects/2secactpy/visualization/data')


def extract_cima_expression():
    """Extract gene expression from CIMA using h5py."""
    h5_path = '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad'

    logger.info(f"Opening {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        # Get gene names
        var_names = f['var']['_index'][:]
        if isinstance(var_names[0], bytes):
            var_names = [g.decode() for g in var_names]
        else:
            var_names = list(var_names)

        # Find gene indices
        gene_to_idx = {g: i for i, g in enumerate(var_names)}
        genes_found = [g for g in KEY_GENES if g in gene_to_idx]
        logger.info(f"Found {len(genes_found)}/{len(KEY_GENES)} genes")

        # Get cell type labels
        obs_keys = list(f['obs'].keys())
        cell_type_col = None
        for col in ['cell_type', 'cell_type_l1', 'celltype', 'CellType']:
            if col in obs_keys:
                cell_type_col = col
                break

        if not cell_type_col:
            logger.error("No cell type column found")
            return pd.DataFrame()

        cell_types = f['obs'][cell_type_col][:]
        if isinstance(cell_types[0], bytes):
            cell_types = np.array([ct.decode() for ct in cell_types])

        logger.info(f"Using cell type column: {cell_type_col}")
        unique_cts = np.unique(cell_types)
        logger.info(f"Found {len(unique_cts)} cell types")

        # Get expression data structure
        X_group = f['X']
        is_sparse = 'data' in X_group and 'indices' in X_group

        if is_sparse:
            logger.info("Sparse matrix format detected")
            X_data = X_group['data'][:]
            X_indices = X_group['indices'][:]
            X_indptr = X_group['indptr'][:]
            n_cells = len(X_indptr) - 1
        else:
            logger.info("Dense matrix format detected")
            X_dense = X_group[:]
            n_cells = X_dense.shape[0]

        logger.info(f"Total cells: {n_cells}")

        results = []

        # Process each gene
        for gene_idx, gene in enumerate(genes_found):
            if gene_idx % 5 == 0:
                logger.info(f"Processing gene {gene_idx+1}/{len(genes_found)}: {gene}")

            gene_col = gene_to_idx[gene]

            # Extract expression for this gene
            if is_sparse:
                # For CSR sparse matrix, need to extract column
                gene_expr = np.zeros(n_cells)
                for cell in range(n_cells):
                    start, end = X_indptr[cell], X_indptr[cell + 1]
                    cell_indices = X_indices[start:end]
                    cell_data = X_data[start:end]
                    match = np.where(cell_indices == gene_col)[0]
                    if len(match) > 0:
                        gene_expr[cell] = cell_data[match[0]]
            else:
                gene_expr = X_dense[:, gene_col]

            # Calculate stats per cell type
            for ct in unique_cts:
                mask = cell_types == ct
                n_cells_ct = mask.sum()

                if n_cells_ct < 10:
                    continue

                expr_ct = gene_expr[mask]
                mean_expr = float(np.mean(expr_ct))
                pct_expr = float(np.sum(expr_ct > 0) / n_cells_ct * 100)

                results.append({
                    'gene': gene,
                    'cell_type': str(ct),
                    'mean_expression': round(mean_expr, 4),
                    'pct_expressed': round(pct_expr, 2),
                    'n_cells': int(n_cells_ct),
                    'atlas': 'CIMA',
                })

    return pd.DataFrame(results)


def main():
    logger.info("Starting simple gene expression extraction")

    # Process CIMA
    df = extract_cima_expression()

    if df.empty:
        logger.error("No data extracted")
        return 1

    logger.info(f"Extracted {len(df)} records")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save full dataset
    full_path = OUTPUT_DIR / 'gene_expression.json'
    with open(full_path, 'w') as f:
        json.dump(df.to_dict(orient='records'), f)
    logger.info(f"Saved to {full_path}")

    # Save per-gene files
    gene_dir = OUTPUT_DIR / 'genes'
    gene_dir.mkdir(exist_ok=True)

    for gene in df['gene'].unique():
        gene_df = df[df['gene'] == gene]
        gene_path = gene_dir / f'{gene}.json'
        with open(gene_path, 'w') as f:
            json.dump(gene_df.to_dict(orient='records'), f)

    # Save gene list
    gene_list = sorted(df['gene'].unique().tolist())
    with open(OUTPUT_DIR / 'gene_list.json', 'w') as f:
        json.dump(gene_list, f)

    logger.info(f"Saved {len(gene_list)} gene files")
    logger.info("Done!")
    return 0


if __name__ == '__main__':
    exit(main())
