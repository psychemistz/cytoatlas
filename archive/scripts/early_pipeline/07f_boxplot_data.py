#!/usr/bin/env python3
"""
Generate box plot data (quartiles) for gene expression and activity data.

For each (signature/gene, cell_type, atlas) combination, computes:
- min, q1 (25%), median, q3 (75%), max
- mean, std, n_samples

This enables proper box plot visualization in the frontend.
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

# Pseudobulk activity files
ACTIVITY_FILES = {
    'CIMA_CytoSig': '/vf/users/parks34/projects/2secactpy/results/cima/CIMA_CytoSig_pseudobulk.h5ad',
    'CIMA_SecAct': '/vf/users/parks34/projects/2secactpy/results/cima/CIMA_SecAct_pseudobulk.h5ad',
    'Inflammation_main_CytoSig': '/vf/users/parks34/projects/2secactpy/results/inflammation/main_CytoSig_pseudobulk.h5ad',
    'Inflammation_main_SecAct': '/vf/users/parks34/projects/2secactpy/results/inflammation/main_SecAct_pseudobulk.h5ad',
    'Inflammation_val_CytoSig': '/vf/users/parks34/projects/2secactpy/results/inflammation/validation_CytoSig_pseudobulk.h5ad',
    'Inflammation_val_SecAct': '/vf/users/parks34/projects/2secactpy/results/inflammation/validation_SecAct_pseudobulk.h5ad',
    'Inflammation_ext_CytoSig': '/vf/users/parks34/projects/2secactpy/results/inflammation/external_CytoSig_pseudobulk.h5ad',
    'Inflammation_ext_SecAct': '/vf/users/parks34/projects/2secactpy/results/inflammation/external_SecAct_pseudobulk.h5ad',
    'scAtlas_normal_CytoSig': '/vf/users/parks34/projects/2secactpy/results/scatlas/scatlas_normal_CytoSig_pseudobulk.h5ad',
    'scAtlas_normal_SecAct': '/vf/users/parks34/projects/2secactpy/results/scatlas/scatlas_normal_SecAct_pseudobulk.h5ad',
    'scAtlas_cancer_CytoSig': '/vf/users/parks34/projects/2secactpy/results/scatlas/scatlas_cancer_CytoSig_pseudobulk.h5ad',
    'scAtlas_cancer_SecAct': '/vf/users/parks34/projects/2secactpy/results/scatlas/scatlas_cancer_SecAct_pseudobulk.h5ad',
}

# Gene expression source files
EXPRESSION_FILES = {
    'CIMA': '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad',
    'Inflammation_main': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad',
    'Inflammation_val': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad',
    'Inflammation_ext': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_external_afterQC.h5ad',
    'scAtlas_normal': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad',
    'scAtlas_cancer': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad',
}

# Cell type columns for expression
CELLTYPE_COLS = {
    'CIMA': 'cell_type_l2',
    'Inflammation_main': 'Level1',
    'Inflammation_val': 'Level1',
    'Inflammation_ext': 'Level1',
    'scAtlas_normal': 'cellType1',
    'scAtlas_cancer': 'cellType1',
}

OUTPUT_DIR = Path('/vf/users/parks34/projects/2secactpy/visualization/data')

# Sample size per cell type for expression
SAMPLE_SIZE = 500


def decode_if_bytes(val):
    """Decode bytes to string if necessary."""
    if isinstance(val, bytes):
        return val.decode()
    return val


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


def compute_quartiles(values):
    """Compute box plot statistics for an array of values.

    For n>=3: Returns full quartile statistics for boxplot
    For n=1-2: Returns data points for scatter display
    For n=0: Returns None
    """
    values = np.array(values)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return None

    # For 1-2 samples, return individual points (can't compute meaningful quartiles)
    if len(values) < 3:
        return {
            'min': float(np.min(values)),
            'q1': float(np.min(values)),  # Same as min for display
            'median': float(np.mean(values)),  # Use mean as center
            'q3': float(np.max(values)),  # Same as max for display
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)) if len(values) > 1 else 0.0,
            'n': int(len(values)),
            'points': [float(v) for v in values],  # Raw points for scatter
        }

    return {
        'min': float(np.min(values)),
        'q1': float(np.percentile(values, 25)),
        'median': float(np.median(values)),
        'q3': float(np.percentile(values, 75)),
        'max': float(np.max(values)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'n': int(len(values)),
    }


def process_activity_file(file_path: str, atlas_name: str, sig_type: str) -> list:
    """
    Process a pseudobulk activity h5ad file to extract box plot data.

    Pseudobulk files are (signatures × samples) where each sample is a (cell_type, sample_id).
    For scAtlas, cell types are shown per organ (e.g., "Macrophage (Spleen)").
    """
    logger.info(f"Processing {file_path}")

    results = []
    is_scatlas = 'scatlas' in atlas_name.lower() or 'scAtlas' in atlas_name

    with h5py.File(file_path, 'r') as f:
        # Get shape - signatures × samples
        n_sigs, n_samples = f['X'].shape
        logger.info(f"Shape: {n_sigs} signatures × {n_samples} samples")

        # Get signature names from obs (rows)
        sig_names = f['obs']['_index'][:]
        sig_names = [decode_if_bytes(s) for s in sig_names]

        # Get cell types from var (columns)
        var = f['var']
        if 'cell_type' in var:
            ct_data = var['cell_type']
            if isinstance(ct_data, h5py.Group):
                codes = ct_data['codes'][:]
                cats = ct_data['categories'][:]
                cats = [decode_if_bytes(c) for c in cats]
                cell_types = np.array([cats[c] for c in codes])
            else:
                cell_types = ct_data[:]
                cell_types = np.array([decode_if_bytes(c) for c in cell_types])
        else:
            logger.warning("No cell_type column found")
            return results

        # For scAtlas, also get tissue/organ to create "CellType (Organ)" names
        tissues = None
        if is_scatlas and 'tissue' in var:
            tissue_data = var['tissue']
            if isinstance(tissue_data, h5py.Group):
                codes = tissue_data['codes'][:]
                cats = tissue_data['categories'][:]
                cats = [decode_if_bytes(c) for c in cats]
                tissues = np.array([cats[c] for c in codes])
            else:
                tissues = tissue_data[:]
                tissues = np.array([decode_if_bytes(t) for t in tissues])

        # Create display names: for scAtlas, combine cell_type with organ
        if is_scatlas and tissues is not None:
            display_names = np.array([f"{ct} ({tissue})" for ct, tissue in zip(cell_types, tissues)])
        else:
            display_names = cell_types

        unique_display_names = np.unique(display_names)
        logger.info(f"Found {len(unique_display_names)} cell types" +
                   (" (per organ)" if is_scatlas and tissues is not None else ""))

        # Load full activity matrix (signatures × samples)
        X = f['X'][:]

        # For each signature and cell type, compute quartiles
        for sig_idx, sig_name in enumerate(sig_names):
            for ct in unique_display_names:
                ct_mask = (display_names == ct)
                values = X[sig_idx, ct_mask]

                stats = compute_quartiles(values)
                if stats is None:
                    continue

                results.append({
                    'signature': sig_name,
                    'cell_type': str(ct),
                    'atlas': atlas_name,
                    'signature_type': sig_type,
                    **stats
                })

    logger.info(f"Generated {len(results)} records")
    return results


def process_expression_file(file_path: str, atlas_name: str, genes: list) -> list:
    """
    Process gene expression file to extract box plot data for specified genes.

    Samples cells per cell type and computes quartiles of log-normalized expression.
    """
    logger.info(f"Processing expression from {file_path}")
    logger.info(f"Genes to extract: {len(genes)}")

    results = []

    with h5py.File(file_path, 'r') as f:
        n_cells, n_genes_total = get_matrix_shape(f)
        logger.info(f"Shape: {n_cells} cells × {n_genes_total} genes")

        # Get gene names
        var_names = f['var']['_index'][:]
        var_names = [decode_if_bytes(v) for v in var_names]

        # Build gene to index mapping
        gene_to_idx = {g: i for i, g in enumerate(var_names)}

        # Also check for symbol column if var_names are Ensembl IDs
        if var_names[0].startswith('ENSG') and 'symbol' in f['var']:
            symbols = f['var']['symbol'][:]
            symbols = [decode_if_bytes(s) for s in symbols]
            for i, sym in enumerate(symbols):
                if sym not in gene_to_idx:
                    gene_to_idx[sym] = i

        # Filter to genes that exist
        valid_genes = [(g, gene_to_idx[g]) for g in genes if g in gene_to_idx]
        logger.info(f"Found {len(valid_genes)} genes in this atlas")

        if not valid_genes:
            return results

        # Get cell type column
        atlas_key = atlas_name.replace('_Normal', '_normal').replace('_Cancer', '_cancer')
        if atlas_key not in CELLTYPE_COLS:
            # Try base name
            for key in CELLTYPE_COLS:
                if key in atlas_key or atlas_key in key:
                    atlas_key = key
                    break

        celltype_col = CELLTYPE_COLS.get(atlas_key)
        if not celltype_col or celltype_col not in f['obs']:
            logger.warning(f"Cell type column not found for {atlas_name}")
            return results

        # Load cell types
        ct_data = f['obs'][celltype_col]
        if isinstance(ct_data, h5py.Group):
            codes = ct_data['codes'][:]
            cats = ct_data['categories'][:]
            cats = [decode_if_bytes(c) for c in cats]
            cell_types = np.array([cats[c] for c in codes])
        else:
            cell_types = ct_data[:]
            cell_types = np.array([decode_if_bytes(c) for c in cell_types])

        unique_cell_types = np.unique(cell_types)
        logger.info(f"Found {len(unique_cell_types)} cell types")

        # Detect normalization from a small sample
        sample_idx = np.random.choice(n_cells, size=min(100, n_cells), replace=False)
        sample_idx = np.sort(sample_idx)
        X_detect = read_sparse_rows(f, sample_idx, n_genes_total)[:, :100]

        max_val = X_detect.max()
        mean_val = X_detect.mean()
        if max_val < 15 and mean_val < 2:
            is_log_normalized = True
        else:
            is_log_normalized = False
        logger.info(f"Log normalized: {is_log_normalized}")
        del X_detect

        # Sample cells per cell type and compute expression quartiles
        np.random.seed(42)
        gene_indices = [idx for _, idx in valid_genes]

        for ct in unique_cell_types:
            ct_mask = (cell_types == ct)
            cell_indices = np.where(ct_mask)[0]
            n_total = len(cell_indices)

            if n_total < 10:
                continue

            # Sample cells
            n_sample = min(SAMPLE_SIZE, n_total)
            sampled_indices = np.random.choice(cell_indices, size=n_sample, replace=False)
            sampled_indices = np.sort(sampled_indices)

            # Read expression for sampled cells and target genes
            X_full = read_sparse_rows(f, sampled_indices, n_genes_total)
            X_sample = X_full[:, gene_indices]
            del X_full

            # Normalize if needed
            if not is_log_normalized:
                row_sums = X_sample.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                cpm = X_sample / row_sums * 1e6
                X_sample = np.log1p(cpm / 100)

            # Compute quartiles for each gene
            for gene_idx, (gene_name, _) in enumerate(valid_genes):
                values = X_sample[:, gene_idx]
                stats = compute_quartiles(values)
                if stats is None:
                    continue

                # Add percent expressed
                pct_expressed = float(np.sum(values > 0) / len(values) * 100)

                results.append({
                    'gene': gene_name,
                    'cell_type': str(ct),
                    'atlas': atlas_name,
                    'pct_expressed': round(pct_expressed, 2),
                    **stats
                })

        del X_sample
        gc.collect()

    logger.info(f"Generated {len(results)} expression records")
    return results


def get_signature_genes():
    """Get list of all CytoSig and SecAct signature names."""
    from secactpy import load_cytosig, load_secact

    cytosig = load_cytosig()
    secact = load_secact()

    cytosig_sigs = list(cytosig.columns)
    secact_sigs = list(secact.columns)

    logger.info(f"CytoSig signatures: {len(cytosig_sigs)}")
    logger.info(f"SecAct signatures: {len(secact_sigs)}")

    return cytosig_sigs, secact_sigs


def main():
    logger.info("Starting box plot data generation")

    # Get signature names (these are also gene names for expression)
    cytosig_sigs, secact_sigs = get_signature_genes()
    all_genes = list(set(cytosig_sigs + secact_sigs))

    all_activity_results = []

    # Process activity files
    for file_key, file_path in ACTIVITY_FILES.items():
        if not Path(file_path).exists():
            logger.warning(f"File not found: {file_path}")
            continue

        # Parse atlas and signature type from key
        parts = file_key.split('_')
        if file_key.startswith('CIMA'):
            atlas_name = 'CIMA'
            sig_type = parts[-1]
        elif file_key.startswith('Inflammation'):
            atlas_name = 'Inflammation'
            sig_type = parts[-1]
        elif file_key.startswith('scAtlas'):
            tissue = parts[1]  # normal or cancer
            atlas_name = f'scAtlas_{tissue.capitalize()}'
            sig_type = parts[-1]
        else:
            continue

        logger.info("=" * 60)
        logger.info(f"Processing {file_key} -> {atlas_name} ({sig_type})")

        try:
            results = process_activity_file(file_path, atlas_name, sig_type)
            all_activity_results.extend(results)
        except Exception as e:
            logger.error(f"Failed to process {file_key}: {e}")
            import traceback
            traceback.print_exc()

        gc.collect()

    # Save activity results IMMEDIATELY (before expression processing which may OOM)
    logger.info("=" * 60)
    logger.info("Saving activity box plot data...")
    # Save activity results
    if all_activity_results:
        activity_df = pd.DataFrame(all_activity_results)
        logger.info(f"Total activity records: {len(activity_df)}")

        # Save combined file
        output_path = OUTPUT_DIR / 'activity_boxplot.json'
        with open(output_path, 'w') as f:
            json.dump(activity_df.to_dict(orient='records'), f)
        logger.info(f"Saved {output_path}")

        # Save per-signature files
        boxplot_dir = OUTPUT_DIR / 'boxplot'
        boxplot_dir.mkdir(exist_ok=True)

        for sig in activity_df['signature'].unique():
            sig_df = activity_df[activity_df['signature'] == sig]
            safe_name = sig.replace('/', '_').replace('\\', '_').replace(' ', '_')
            with open(boxplot_dir / f'{safe_name}_activity.json', 'w') as f:
                json.dump(sig_df.to_dict(orient='records'), f)

        logger.info(f"Saved {activity_df['signature'].nunique()} per-signature activity files")

    # Note: Expression box plot data is generated by the all-gene expression job
    # and can be derived from the gene expression data later

    logger.info("=" * 60)
    logger.info("Done!")
    return 0


if __name__ == '__main__':
    exit(main())
