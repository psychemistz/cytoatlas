#!/usr/bin/env python3
"""
Extract gene expression from Inflammation Atlas only.
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

KEY_GENES = [
    'IFNG', 'TNF', 'IL1A', 'IL1B', 'IL2', 'IL4', 'IL6', 'IL10', 'IL12A', 'IL12B',
    'IL13', 'IL17A', 'IL17F', 'IL21', 'IL22', 'IL23A', 'IL27', 'IL33',
    'TGFB1', 'TGFB2', 'TGFB3', 'CSF1', 'CSF2', 'CSF3', 'CXCL8', 'CCL2', 'CCL3',
    'CCL4', 'CCL5', 'CXCL10', 'CXCL11', 'VEGFA', 'EGF', 'FGF2', 'HGF',
    'SPP1', 'MMP9', 'MMP2', 'TIMP1', 'GZMB', 'PRF1', 'FASLG',
    'CD274', 'PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT',
]

DATA_PATHS = {
    'inflammation_main': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad',
    'inflammation_val': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad',
    'inflammation_ext': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_external_afterQC.h5ad',
}

OUTPUT_DIR = Path('/vf/users/parks34/projects/2cytoatlas/visualization/data')
SAMPLE_SIZE = 2000


def extract_sampled_expression(h5ad_path: str, genes: list[str], atlas_name: str) -> pd.DataFrame:
    """Extract gene expression by sampling cells per cell type."""
    import gc

    logger.info(f"Loading {atlas_name} from {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path, backed='r')

    # Handle Ensembl IDs - use symbol column
    gene_name_map = {}
    if 'symbol' in adata.var.columns:
        logger.info("Using 'symbol' column for gene lookup")
        symbol_to_idx = {}
        for idx, symbol in zip(adata.var_names, adata.var['symbol'].values):
            if isinstance(symbol, bytes):
                symbol = symbol.decode()
            symbol_to_idx[symbol] = idx

        genes_found = [g for g in genes if g in symbol_to_idx]
        gene_name_map = {g: symbol_to_idx[g] for g in genes_found}
        logger.info(f"Found {len(genes_found)}/{len(genes)} genes via symbol column")
    else:
        available_genes = set(adata.var_names)
        genes_found = [g for g in genes if g in available_genes]
        gene_name_map = {g: g for g in genes_found}
        logger.info(f"Found {len(genes_found)}/{len(genes)} genes")

    if not genes_found:
        return pd.DataFrame()

    # Use Level1 for cell type
    cell_type_col = 'Level1' if 'Level1' in adata.obs.columns else 'Level2'
    if cell_type_col not in adata.obs.columns:
        logger.warning(f"No cell type column in {atlas_name}")
        return pd.DataFrame()

    logger.info(f"Using cell type column: {cell_type_col}")

    cell_types = adata.obs[cell_type_col].unique()
    cell_type_labels = adata.obs[cell_type_col].values.copy()
    logger.info(f"Found {len(cell_types)} cell types")

    var_names_to_subset = [gene_name_map[g] for g in genes_found]

    logger.info(f"Subsetting to {len(genes_found)} genes...")
    adata_genes = adata[:, var_names_to_subset]
    logger.info("Gene subset complete")

    results = []
    np.random.seed(42)

    for ct_idx, ct in enumerate(cell_types):
        if ct_idx % 5 == 0:
            logger.info(f"Processing cell type {ct_idx+1}/{len(cell_types)}: {ct}")
            gc.collect()

        mask = (cell_type_labels == ct)
        cell_indices = np.where(mask)[0]
        n_total = len(cell_indices)

        if n_total < 10:
            continue

        n_sample = min(SAMPLE_SIZE, n_total)
        sampled_indices = np.random.choice(cell_indices, size=n_sample, replace=False)
        sampled_indices = np.sort(sampled_indices)

        X_sample = adata_genes.X[sampled_indices, :]
        if hasattr(X_sample, 'toarray'):
            X_sample = X_sample.toarray()

        for j, gene in enumerate(genes_found):
            expr = X_sample[:, j]
            mean_expr = float(np.mean(expr))
            pct_expr = float(np.sum(expr > 0) / len(expr) * 100)

            results.append({
                'gene': gene,
                'cell_type': str(ct),
                'mean_expression': round(mean_expr, 4),
                'pct_expressed': round(pct_expr, 2),
                'n_cells': int(n_total),
                'atlas': atlas_name,
            })

        del X_sample
        gc.collect()

    del adata_genes
    gc.collect()

    return pd.DataFrame(results)


def main():
    logger.info("Starting Inflammation Atlas gene expression extraction")

    all_dfs = []
    for name, path in DATA_PATHS.items():
        if Path(path).exists():
            df = extract_sampled_expression(path, KEY_GENES, 'Inflammation')
            if not df.empty:
                all_dfs.append(df)

    if not all_dfs:
        logger.error("No data extracted")
        return 1

    # Combine and aggregate
    combined = pd.concat(all_dfs, ignore_index=True)
    aggregated = combined.groupby(['gene', 'cell_type', 'atlas']).agg({
        'mean_expression': 'mean',
        'pct_expressed': 'mean',
        'n_cells': 'sum',
    }).reset_index()

    logger.info(f"Total records: {len(aggregated)}")

    # Save to separate file (will be merged later)
    output_file = OUTPUT_DIR / 'gene_expression_inflammation.json'
    with open(output_file, 'w') as f:
        json.dump(aggregated.to_dict(orient='records'), f)

    logger.info(f"Saved to {output_file}")
    logger.info("Done!")
    return 0


if __name__ == '__main__':
    exit(main())
