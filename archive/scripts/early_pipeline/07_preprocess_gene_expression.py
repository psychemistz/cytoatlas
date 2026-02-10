#!/usr/bin/env python3
"""
Preprocess gene expression data for the gene-centric search feature.

Extracts mean gene expression by cell type for each atlas and saves as JSON
for fast API access.

Usage:
    python scripts/07_preprocess_gene_expression.py [--genes GENE1,GENE2,...] [--top-genes N]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data paths from CLAUDE.md
DATA_PATHS = {
    'cima': '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad',
    'inflammation_main': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad',
    'inflammation_val': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad',
    'inflammation_ext': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_external_afterQC.h5ad',
    'scatlas_normal': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad',
    'scatlas_cancer': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad',
}

# Output directory
OUTPUT_DIR = Path('/vf/users/parks34/projects/2cytoatlas/visualization/data')

# Common cytokine/secreted protein genes to always include
PRIORITY_GENES = [
    # Key cytokines (CytoSig)
    'IFNG', 'TNF', 'IL1A', 'IL1B', 'IL2', 'IL4', 'IL6', 'IL10', 'IL12A', 'IL12B',
    'IL13', 'IL17A', 'IL17F', 'IL21', 'IL22', 'IL23A', 'IL27', 'IL33', 'IL35',
    'TGFB1', 'TGFB2', 'TGFB3', 'CSF1', 'CSF2', 'CSF3', 'CXCL8', 'CCL2', 'CCL3',
    'CCL4', 'CCL5', 'CXCL10', 'CXCL11', 'VEGFA', 'EGF', 'FGF2', 'HGF', 'PDGFA',
    'PDGFB', 'NGF', 'BDNF', 'LIF', 'OSM', 'IFNA1', 'IFNB1',
    # Additional secreted proteins
    'SPP1', 'MMP9', 'MMP2', 'TIMP1', 'SERPINE1', 'FN1', 'COL1A1', 'CTGF',
    'ANGPT1', 'ANGPT2', 'THBS1', 'SPARC', 'POSTN', 'TNC', 'LGALS1', 'LGALS3',
    # Immune markers
    'CD274', 'PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'CD28', 'ICOS',
    'GZMB', 'PRF1', 'FASLG', 'CD40LG', 'CD70', 'TNFSF10',
]


def get_cell_type_column(adata):
    """Find the cell type column in obs."""
    candidates = ['cell_type', 'celltype', 'cell_types', 'CellType', 'Celltype',
                  'cell_type_fine', 'celltype_fine', 'annotation', 'cluster']
    for col in candidates:
        if col in adata.obs.columns:
            return col
    # Return first column that looks like cell type
    for col in adata.obs.columns:
        if 'cell' in col.lower() or 'type' in col.lower() or 'annot' in col.lower():
            return col
    return None


def get_sample_column(adata):
    """Find the sample column in obs."""
    candidates = ['sample', 'sample_id', 'donor', 'donor_id', 'patient', 'subject']
    for col in candidates:
        if col in adata.obs.columns:
            return col
    return None


def extract_gene_expression(h5ad_path: str, genes: list[str], atlas_name: str) -> pd.DataFrame:
    """
    Extract mean gene expression by cell type from an h5ad file.

    Memory-efficient: processes one gene at a time and uses chunked cell reading.

    Args:
        h5ad_path: Path to h5ad file
        genes: List of gene names to extract
        atlas_name: Name of the atlas for labeling

    Returns:
        DataFrame with columns: gene, cell_type, mean_expression, pct_expressed, n_cells, atlas
    """
    import gc
    logger.info(f"Loading {atlas_name} from {h5ad_path}")

    # Load in backed mode for memory efficiency
    adata = sc.read_h5ad(h5ad_path, backed='r')

    # Get available genes
    if hasattr(adata, 'var_names'):
        available_genes = set(adata.var_names)
    else:
        available_genes = set(adata.var.index)

    # Filter to genes that exist
    genes_to_extract = [g for g in genes if g in available_genes]
    logger.info(f"Found {len(genes_to_extract)}/{len(genes)} genes in {atlas_name}")

    if not genes_to_extract:
        logger.warning(f"No genes found in {atlas_name}")
        return pd.DataFrame()

    # Get cell type column
    cell_type_col = get_cell_type_column(adata)
    if cell_type_col is None:
        logger.warning(f"No cell type column found in {atlas_name}")
        return pd.DataFrame()

    logger.info(f"Using cell type column: {cell_type_col}")

    # Get unique cell types
    cell_types = adata.obs[cell_type_col].unique()
    logger.info(f"Found {len(cell_types)} cell types")

    # Pre-compute cell type indices (memory efficient)
    cell_type_info = {}
    for ct in cell_types:
        mask = (adata.obs[cell_type_col] == ct).values
        n_cells = mask.sum()
        if n_cells >= 10:
            cell_type_info[str(ct)] = {
                'indices': np.where(mask)[0],
                'n_cells': int(n_cells)
            }
    logger.info(f"Processing {len(cell_type_info)} cell types with >= 10 cells")

    # Build gene index lookup
    var_names_list = list(adata.var_names)

    results = []

    # Process ONE gene at a time to minimize memory
    for gene_idx, gene in enumerate(genes_to_extract):
        if gene_idx % 5 == 0:
            logger.info(f"Processing gene {gene_idx+1}/{len(genes_to_extract)}: {gene}")
            gc.collect()

        gene_col = var_names_list.index(gene)

        # For each cell type, calculate stats for this gene
        for ct, info in cell_type_info.items():
            cell_indices = info['indices']
            n_cells = info['n_cells']

            # Process in smaller chunks
            chunk_size = 50000
            total_sum = 0.0
            total_nonzero = 0

            for start in range(0, len(cell_indices), chunk_size):
                end = min(start + chunk_size, len(cell_indices))
                chunk_idx = cell_indices[start:end]

                # Read ONLY this single gene column for these cells
                X_chunk = adata.X[chunk_idx, gene_col]

                # Handle sparse matrix
                if hasattr(X_chunk, 'toarray'):
                    X_chunk = X_chunk.toarray().flatten()
                elif hasattr(X_chunk, 'A1'):
                    X_chunk = X_chunk.A1
                else:
                    X_chunk = np.asarray(X_chunk).flatten()

                total_sum += float(X_chunk.sum())
                total_nonzero += int((X_chunk > 0).sum())

                del X_chunk

            mean_expr = total_sum / n_cells
            pct_expr = total_nonzero / n_cells * 100

            results.append({
                'gene': gene,
                'cell_type': ct,
                'mean_expression': round(mean_expr, 4),
                'pct_expressed': round(pct_expr, 2),
                'n_cells': n_cells,
                'atlas': atlas_name,
            })

    df = pd.DataFrame(results)
    logger.info(f"Extracted {len(df)} records from {atlas_name}")

    return df


def process_all_atlases(genes: list[str]) -> pd.DataFrame:
    """Process all atlases and combine results."""
    all_results = []

    # CIMA
    if Path(DATA_PATHS['cima']).exists():
        df = extract_gene_expression(DATA_PATHS['cima'], genes, 'CIMA')
        all_results.append(df)
    else:
        logger.warning(f"CIMA file not found: {DATA_PATHS['cima']}")

    # Inflammation Atlas (combine main, val, ext)
    inflam_dfs = []
    for name, path in [('inflammation_main', DATA_PATHS['inflammation_main']),
                       ('inflammation_val', DATA_PATHS['inflammation_val']),
                       ('inflammation_ext', DATA_PATHS['inflammation_ext'])]:
        if Path(path).exists():
            df = extract_gene_expression(path, genes, 'Inflammation')
            inflam_dfs.append(df)
        else:
            logger.warning(f"{name} file not found: {path}")

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
        df = extract_gene_expression(DATA_PATHS['scatlas_normal'], genes, 'scAtlas_Normal')
        all_results.append(df)
    else:
        logger.warning(f"scAtlas normal file not found: {DATA_PATHS['scatlas_normal']}")

    # scAtlas Cancer
    if Path(DATA_PATHS['scatlas_cancer']).exists():
        df = extract_gene_expression(DATA_PATHS['scatlas_cancer'], genes, 'scAtlas_Cancer')
        all_results.append(df)
    else:
        logger.warning(f"scAtlas cancer file not found: {DATA_PATHS['scatlas_cancer']}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        return combined

    return pd.DataFrame()


def save_gene_expression_data(df: pd.DataFrame, output_dir: Path):
    """Save gene expression data as JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full dataset
    full_path = output_dir / 'gene_expression.json'
    records = df.to_dict(orient='records')
    with open(full_path, 'w') as f:
        json.dump(records, f)
    logger.info(f"Saved full dataset to {full_path}")

    # Save per-gene files for faster lookup
    gene_dir = output_dir / 'genes'
    gene_dir.mkdir(exist_ok=True)

    for gene in df['gene'].unique():
        gene_df = df[df['gene'] == gene]
        gene_path = gene_dir / f'{gene}.json'
        gene_records = gene_df.to_dict(orient='records')
        with open(gene_path, 'w') as f:
            json.dump(gene_records, f)

    logger.info(f"Saved {df['gene'].nunique()} individual gene files to {gene_dir}")

    # Save gene list
    gene_list_path = output_dir / 'gene_list.json'
    gene_list = sorted(df['gene'].unique().tolist())
    with open(gene_list_path, 'w') as f:
        json.dump(gene_list, f)
    logger.info(f"Saved gene list ({len(gene_list)} genes) to {gene_list_path}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess gene expression data')
    parser.add_argument('--genes', type=str, help='Comma-separated list of genes')
    parser.add_argument('--top-genes', type=int, default=0, help='Include top N variable genes')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR), help='Output directory')
    args = parser.parse_args()

    # Determine genes to process
    genes = PRIORITY_GENES.copy()

    if args.genes:
        additional = [g.strip() for g in args.genes.split(',')]
        genes = list(set(genes + additional))

    logger.info(f"Processing {len(genes)} genes")

    # Process all atlases
    df = process_all_atlases(genes)

    if df.empty:
        logger.error("No data extracted")
        return 1

    logger.info(f"Total records: {len(df)}")
    logger.info(f"Genes: {df['gene'].nunique()}")
    logger.info(f"Cell types: {df['cell_type'].nunique()}")
    logger.info(f"Atlases: {df['atlas'].unique().tolist()}")

    # Save results
    save_gene_expression_data(df, Path(args.output_dir))

    logger.info("Done!")
    return 0


if __name__ == '__main__':
    exit(main())
