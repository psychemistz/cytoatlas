#!/usr/bin/env python3
"""
Extract gene expression from sampled cells for memory efficiency.

Samples cells within each cell type to estimate mean expression.
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
    'scatlas_normal': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad',
}

OUTPUT_DIR = Path('/vf/users/parks34/projects/2secactpy/visualization/data')

# Sample size per cell type
SAMPLE_SIZE = 5000


def get_cell_type_column(adata):
    """Find the cell type column."""
    for col in ['cell_type', 'cell_type_l1', 'celltype', 'CellType', 'cell_type_fine']:
        if col in adata.obs.columns:
            return col
    return None


def extract_sampled_expression(h5ad_path: str, genes: list[str], atlas_name: str) -> pd.DataFrame:
    """
    Extract gene expression by sampling cells per cell type.
    """
    import gc

    logger.info(f"Loading {atlas_name} from {h5ad_path}")

    # Load in backed mode
    adata = sc.read_h5ad(h5ad_path, backed='r')

    # Find genes
    available_genes = set(adata.var_names)
    genes_found = [g for g in genes if g in available_genes]
    logger.info(f"Found {len(genes_found)}/{len(genes)} genes")

    if not genes_found:
        return pd.DataFrame()

    # Get cell type column
    cell_type_col = get_cell_type_column(adata)
    if not cell_type_col:
        logger.warning(f"No cell type column in {atlas_name}")
        return pd.DataFrame()

    logger.info(f"Using cell type column: {cell_type_col}")

    # Get gene indices
    var_names = list(adata.var_names)
    gene_indices = [var_names.index(g) for g in genes_found]

    # Get unique cell types
    cell_types = adata.obs[cell_type_col].unique()
    logger.info(f"Found {len(cell_types)} cell types")

    results = []
    np.random.seed(42)

    for ct_idx, ct in enumerate(cell_types):
        if ct_idx % 5 == 0:
            logger.info(f"Processing cell type {ct_idx+1}/{len(cell_types)}: {ct}")
            gc.collect()

        # Get cells of this type
        mask = (adata.obs[cell_type_col] == ct).values
        cell_indices = np.where(mask)[0]
        n_total = len(cell_indices)

        if n_total < 10:
            continue

        # Sample cells
        n_sample = min(SAMPLE_SIZE, n_total)
        sampled_indices = np.random.choice(cell_indices, size=n_sample, replace=False)
        sampled_indices = np.sort(sampled_indices)

        # Read expression for sampled cells
        X_sample = adata.X[sampled_indices, :][:, gene_indices]
        if hasattr(X_sample, 'toarray'):
            X_sample = X_sample.toarray()

        # Calculate stats
        for j, gene in enumerate(genes_found):
            expr = X_sample[:, j]
            mean_expr = float(np.mean(expr))
            pct_expr = float(np.sum(expr > 0) / len(expr) * 100)

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

    return pd.DataFrame(results)


def main():
    logger.info("Starting sampled gene expression extraction")

    all_results = []

    # CIMA
    if Path(DATA_PATHS['cima']).exists():
        df = extract_sampled_expression(DATA_PATHS['cima'], KEY_GENES, 'CIMA')
        if not df.empty:
            all_results.append(df)

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
