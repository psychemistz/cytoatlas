#!/usr/bin/env python3
"""
Extract ALL gene expression from sampled cells across all atlases.

Uses h5py directly for memory-efficient reading without loading full matrix.
Processes cell types one at a time to minimize memory usage.
"""

import gc
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

DATA_PATHS = {
    'cima': '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad',
    'inflammation_main': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad',
    'inflammation_val': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad',
    'inflammation_ext': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_external_afterQC.h5ad',
    'scatlas_normal': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad',
    'scatlas_cancer': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad',
}

OUTPUT_DIR = Path('/vf/users/parks34/projects/2secactpy/visualization/data')

# Sample size per cell type (reduced for memory efficiency)
SAMPLE_SIZE = 500

# Cell batch size for reading from h5ad
CELL_BATCH_SIZE = 100

# Atlas-specific cell type column preferences
ATLAS_CELLTYPE_COLS = {
    'CIMA': ['cell_type_l2', 'cell_type_l1'],
    'Inflammation': ['Level1', 'Level2', 'cell_type'],
    'scAtlas': ['cellType1', 'cellType2', 'cell_type'],
}


def decode_if_bytes(val):
    """Decode bytes to string if necessary."""
    if isinstance(val, bytes):
        return val.decode()
    return val


def get_cell_type_column(obs_keys, atlas_name: str = None):
    """Find the cell type column from obs keys."""
    if atlas_name and atlas_name in ATLAS_CELLTYPE_COLS:
        for col in ATLAS_CELLTYPE_COLS[atlas_name]:
            if col in obs_keys:
                return col

    for col in ['cell_type', 'cell_type_l1', 'celltype', 'CellType', 'cell_type_fine',
                'cellType1', 'cellType2', 'ann1', 'majorCluster', 'Level1', 'Level2']:
        if col in obs_keys:
            return col
    return None


def load_cell_types_h5py(h5_file, celltype_col: str) -> np.ndarray:
    """Load cell type labels from h5ad file using h5py."""
    obs = h5_file['obs']

    # Check if column exists
    if celltype_col not in obs:
        raise ValueError(f"Column {celltype_col} not found in obs")

    col_data = obs[celltype_col]

    # Handle different storage formats
    if isinstance(col_data, h5py.Group):
        # Categorical storage: has 'codes' and 'categories'
        if 'codes' in col_data and 'categories' in col_data:
            codes = col_data['codes'][:]
            categories = col_data['categories'][:]
            categories = [decode_if_bytes(c) for c in categories]
            cell_types = np.array([categories[c] for c in codes])
        else:
            raise ValueError(f"Categorical column {celltype_col} missing codes/categories")
    else:
        # Direct array storage
        cell_types = col_data[:]
        if hasattr(cell_types[0], 'decode'):
            cell_types = np.array([decode_if_bytes(v) for v in cell_types])

    return cell_types


def load_gene_names_h5py(h5_file) -> tuple:
    """
    Load gene names from h5ad file using h5py.
    Returns (gene_symbols, var_names_are_symbols).

    Handles different h5ad formats:
    - Format 1: var['_index'] contains gene names directly (e.g., CIMA)
    - Format 2: var.attrs['_index'] specifies column name (e.g., Inflammation Atlas)
    - Categorical columns: codes + categories structure
    """
    var = h5_file['var']

    # Try to load var_names (index)
    var_names = None

    # Format 1: Direct _index dataset
    if '_index' in var and hasattr(var['_index'], 'shape'):
        var_names = var['_index'][:]
        var_names = [decode_if_bytes(v) for v in var_names]
    # Format 2: _index attribute points to a column
    elif '_index' in var.attrs:
        index_col = var.attrs['_index']
        if isinstance(index_col, bytes):
            index_col = index_col.decode()
        if index_col in var:
            col = var[index_col]
            # Check if categorical (has codes + categories)
            if hasattr(col, 'keys') and 'codes' in col and 'categories' in col:
                codes = col['codes'][:]
                cats = [decode_if_bytes(c) for c in col['categories'][:]]
                var_names = [cats[c] for c in codes]
            else:
                var_names = [decode_if_bytes(v) for v in col[:]]

    if var_names is None:
        raise ValueError("Could not find gene index in h5ad file")

    # Check if var_names are gene symbols or Ensembl IDs
    if not var_names[0].startswith('ENSG'):
        return np.array(var_names), True

    # var_names are Ensembl IDs, try to get symbols
    if 'symbol' in var:
        sym_col = var['symbol']
        # Check if categorical
        if hasattr(sym_col, 'keys') and 'codes' in sym_col and 'categories' in sym_col:
            codes = sym_col['codes'][:]
            cats = [decode_if_bytes(c) for c in sym_col['categories'][:]]
            symbols = [cats[c] for c in codes]
        else:
            symbols = [decode_if_bytes(s) for s in sym_col[:]]
        return np.array(symbols), True

    # Fall back to Ensembl IDs
    return np.array(var_names), False


def normalize_expression(X_sample, detected_norm: str):
    """Normalize expression to log1p(CPM) scale."""
    if detected_norm == 'log_normalized':
        return X_sample
    elif detected_norm == 'cpm':
        return np.log1p(X_sample / 100)
    else:
        # Raw counts - normalize per cell
        row_sums = X_sample.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cpm = X_sample / row_sums * 1e6
        return np.log1p(cpm / 100)


def detect_normalization(X_sample):
    """Detect if data is log-normalized, CPM, or raw counts."""
    max_val = X_sample.max()
    mean_val = X_sample.mean()

    if max_val < 15 and mean_val < 2:
        return 'log_normalized'
    elif max_val > 100:
        return 'cpm' if mean_val > 10 else 'raw_counts'
    else:
        return 'raw_counts'


def get_matrix_shape(f):
    """Get shape of X matrix, handling both dense and sparse storage."""
    X = f['X']

    # If it's a Dataset with shape attribute, return directly
    if hasattr(X, 'shape') and not isinstance(X, h5py.Group):
        return X.shape

    # For sparse matrix (stored as Group), get shape from attributes or infer
    if isinstance(X, h5py.Group):
        # Check for shape attribute
        if 'shape' in X.attrs:
            shape = X.attrs['shape']
            return tuple(shape)

        # Infer from indptr and data
        if 'indptr' in X:
            n_cells = len(X['indptr'][:]) - 1
            # Get n_genes from var
            n_genes = len(f['var']['_index'][:])
            return (n_cells, n_genes)

    raise ValueError("Cannot determine matrix shape")


def read_sparse_rows(f, row_indices, n_genes):
    """Read specific rows from a sparse matrix stored in CSR format."""
    X = f['X']

    if not isinstance(X, h5py.Group):
        # Dense matrix - direct indexing
        data = X[row_indices, :]
        if hasattr(data, 'toarray'):
            data = data.toarray()
        return data

    # Sparse CSR matrix
    indptr = X['indptr'][:]
    indices = X['indices'][:]
    data = X['data'][:]

    # Build dense matrix for requested rows
    result = np.zeros((len(row_indices), n_genes), dtype=np.float32)

    for i, row_idx in enumerate(row_indices):
        start = indptr[row_idx]
        end = indptr[row_idx + 1]
        cols = indices[start:end]
        vals = data[start:end]
        result[i, cols] = vals

    return result


def extract_all_gene_expression(h5ad_path: str, atlas_name: str) -> pd.DataFrame:
    """
    Extract expression for ALL genes using h5py for memory efficiency.
    Processes one cell type at a time.
    """
    logger.info(f"Loading {atlas_name} from {h5ad_path}")

    results = []

    with h5py.File(h5ad_path, 'r') as f:
        # Get basic info - handle both dense and sparse matrices
        n_cells, n_genes = get_matrix_shape(f)
        logger.info(f"Shape: {n_cells} cells × {n_genes} genes")

        # Get cell type column
        obs_keys = list(f['obs'].keys())
        celltype_col = get_cell_type_column(obs_keys, atlas_name)
        if not celltype_col:
            logger.warning(f"No cell type column found in {atlas_name}")
            return pd.DataFrame()

        logger.info(f"Using cell type column: {celltype_col}")

        # Load cell types
        cell_types = load_cell_types_h5py(f, celltype_col)
        unique_cell_types = np.unique(cell_types)
        logger.info(f"Found {len(unique_cell_types)} unique cell types")

        # Load gene names
        gene_symbols, symbols_valid = load_gene_names_h5py(f)
        logger.info(f"Loaded {len(gene_symbols)} gene names (symbols_valid={symbols_valid})")

        # Detect normalization from a small sample
        sample_idx = np.random.choice(n_cells, size=min(100, n_cells), replace=False)
        sample_idx = np.sort(sample_idx)
        X_detect = read_sparse_rows(f, sample_idx, n_genes)[:, :100]
        detected_norm = detect_normalization(X_detect)
        logger.info(f"Detected normalization: {detected_norm}")
        del X_detect

        # Process each cell type
        np.random.seed(42)
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

            if (ct_idx + 1) % 5 == 0 or ct_idx == 0:
                logger.info(f"Processing cell type {ct_idx + 1}/{len(unique_cell_types)}: {ct} ({n_sample} cells)")

            # Read expression for sampled cells in batches to avoid memory issues
            X_accumulated = []
            for batch_start in range(0, len(sampled_indices), CELL_BATCH_SIZE):
                batch_end = min(batch_start + CELL_BATCH_SIZE, len(sampled_indices))
                batch_indices = sampled_indices[batch_start:batch_end]

                # Read batch of cells (all genes at once) - handles sparse matrices
                X_batch = read_sparse_rows(f, batch_indices, n_genes)
                X_accumulated.append(X_batch)

            X_sample = np.vstack(X_accumulated)
            del X_accumulated

            # Normalize
            X_normalized = normalize_expression(X_sample, detected_norm)

            # Calculate stats for ALL genes
            mean_expr = np.mean(X_normalized, axis=0)
            pct_expressed = np.sum(X_sample > 0, axis=0) / X_sample.shape[0] * 100

            # Store results for genes with any expression
            for gene_idx in range(n_genes):
                if pct_expressed[gene_idx] > 0 or mean_expr[gene_idx] > 0:
                    results.append({
                        'gene': gene_symbols[gene_idx],
                        'cell_type': str(ct),
                        'mean_expression': round(float(mean_expr[gene_idx]), 4),
                        'pct_expressed': round(float(pct_expressed[gene_idx]), 2),
                        'n_cells': int(n_total),
                        'atlas': atlas_name,
                    })

            del X_sample, X_normalized, mean_expr, pct_expressed
            gc.collect()

    logger.info(f"Extracted {len(results)} records from {atlas_name}")
    return pd.DataFrame(results)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract gene expression from all atlases')
    parser.add_argument('--skip-cima', action='store_true', help='Skip CIMA (already processed)')
    parser.add_argument('--skip-inflammation', action='store_true', help='Skip Inflammation Atlas')
    parser.add_argument('--skip-scatlas', action='store_true', help='Skip scAtlas')
    args = parser.parse_args()

    logger.info("Starting ALL gene expression extraction")

    all_results = []

    # CIMA
    if args.skip_cima:
        logger.info("=" * 60)
        logger.info("Skipping CIMA (--skip-cima flag)")
    elif Path(DATA_PATHS['cima']).exists():
        logger.info("=" * 60)
        logger.info("Processing CIMA...")
        df = extract_all_gene_expression(DATA_PATHS['cima'], 'CIMA')
        if not df.empty:
            all_results.append(df)
            logger.info(f"CIMA: {len(df)} records, {df['gene'].nunique()} genes")
        gc.collect()

    # Inflammation Atlas (combine all cohorts)
    if args.skip_inflammation:
        logger.info("=" * 60)
        logger.info("Skipping Inflammation Atlas (--skip-inflammation flag)")
    else:
        logger.info("=" * 60)
        logger.info("Processing Inflammation Atlas...")
        inflam_dfs = []
        for name in ['inflammation_main', 'inflammation_val', 'inflammation_ext']:
            if Path(DATA_PATHS[name]).exists():
                df = extract_all_gene_expression(DATA_PATHS[name], 'Inflammation')
                if not df.empty:
                    inflam_dfs.append(df)
                gc.collect()

        if inflam_dfs:
            inflam_combined = pd.concat(inflam_dfs, ignore_index=True)
            # Aggregate across cohorts
            inflam_agg = inflam_combined.groupby(['gene', 'cell_type', 'atlas']).agg({
                'mean_expression': 'mean',
                'pct_expressed': 'mean',
                'n_cells': 'sum',
            }).reset_index()
            all_results.append(inflam_agg)
            logger.info(f"Inflammation: {len(inflam_agg)} records, {inflam_agg['gene'].nunique()} genes")
            del inflam_combined
            gc.collect()

    # scAtlas Normal
    if args.skip_scatlas:
        logger.info("=" * 60)
        logger.info("Skipping scAtlas (--skip-scatlas flag)")
    elif Path(DATA_PATHS['scatlas_normal']).exists():
        logger.info("=" * 60)
        logger.info("Processing scAtlas Normal...")
        df = extract_all_gene_expression(DATA_PATHS['scatlas_normal'], 'scAtlas_Normal')
        if not df.empty:
            all_results.append(df)
            logger.info(f"scAtlas Normal: {len(df)} records, {df['gene'].nunique()} genes")
        gc.collect()

    # scAtlas Cancer
    if not args.skip_scatlas and Path(DATA_PATHS['scatlas_cancer']).exists():
        logger.info("=" * 60)
        logger.info("Processing scAtlas Cancer...")
        df = extract_all_gene_expression(DATA_PATHS['scatlas_cancer'], 'scAtlas_Cancer')
        if not df.empty:
            all_results.append(df)
            logger.info(f"scAtlas Cancer: {len(df)} records, {df['gene'].nunique()} genes")
        gc.collect()

    if not all_results:
        logger.error("No data extracted")
        return 1

    # Combine all results
    logger.info("=" * 60)
    logger.info("Combining results...")
    combined = pd.concat(all_results, ignore_index=True)

    logger.info(f"Total records: {len(combined)}")
    logger.info(f"Total unique genes: {combined['gene'].nunique()}")
    logger.info(f"Atlases: {combined['atlas'].unique().tolist()}")

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save combined file (may be large)
    logger.info("Saving combined gene expression file...")
    combined.to_parquet(OUTPUT_DIR / 'gene_expression_all.parquet', index=False)
    logger.info(f"Saved {OUTPUT_DIR / 'gene_expression_all.parquet'}")

    # Also save as JSON for API compatibility (filtered to genes with significant expression)
    # Keep genes with mean expression > 0.1 in at least one cell type
    gene_max_expr = combined.groupby('gene')['mean_expression'].max()
    significant_genes = gene_max_expr[gene_max_expr > 0.1].index.tolist()
    logger.info(f"Genes with significant expression (max > 0.1): {len(significant_genes)}")

    combined_significant = combined[combined['gene'].isin(significant_genes)]
    with open(OUTPUT_DIR / 'gene_expression.json', 'w') as f:
        json.dump(combined_significant.to_dict(orient='records'), f)
    logger.info(f"Saved {OUTPUT_DIR / 'gene_expression.json'} ({len(combined_significant)} records)")

    # Save per-gene files for the most expressed genes (top 10000)
    gene_dir = OUTPUT_DIR / 'genes'
    gene_dir.mkdir(exist_ok=True)

    # Get top 10000 genes by max expression
    top_genes = gene_max_expr.nlargest(10000).index.tolist()

    logger.info(f"Saving per-gene files for top {len(top_genes)} genes...")
    for gene in top_genes:
        gene_df = combined[combined['gene'] == gene]
        safe_gene_name = gene.replace('/', '_').replace('\\', '_')
        with open(gene_dir / f'{safe_gene_name}.json', 'w') as f:
            json.dump(gene_df.to_dict(orient='records'), f)

    # Save complete gene list
    with open(OUTPUT_DIR / 'gene_list.json', 'w') as f:
        json.dump(sorted(combined['gene'].unique().tolist()), f)

    # Save significant gene list
    with open(OUTPUT_DIR / 'gene_list_significant.json', 'w') as f:
        json.dump(sorted(significant_genes), f)

    logger.info(f"Saved {len(top_genes)} per-gene files to {gene_dir}")
    logger.info("=" * 60)
    logger.info("Done!")
    return 0


if __name__ == '__main__':
    exit(main())
