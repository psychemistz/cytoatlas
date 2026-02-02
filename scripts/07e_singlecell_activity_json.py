#!/usr/bin/env python3
"""
Process single-cell activity h5ad files into JSON for web visualization.

Loads cell type information from original atlas files,
aggregates single-cell activity statistics per cell type,
and saves to JSON format.
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

# Single-cell activity files
SINGLECELL_FILES = {
    'CIMA_CytoSig': '/vf/users/parks34/projects/2secactpy/results/cima/CIMA_CytoSig_singlecell.h5ad',
    'CIMA_SecAct': '/vf/users/parks34/projects/2secactpy/results/cima/CIMA_SecAct_singlecell.h5ad',
    'Inflammation_main_CytoSig': '/vf/users/parks34/projects/2secactpy/results/inflammation/main_CytoSig_singlecell.h5ad',
    'Inflammation_main_SecAct': '/vf/users/parks34/projects/2secactpy/results/inflammation/main_SecAct_singlecell.h5ad',
    'Inflammation_val_CytoSig': '/vf/users/parks34/projects/2secactpy/results/inflammation/validation_CytoSig_singlecell.h5ad',
    'Inflammation_val_SecAct': '/vf/users/parks34/projects/2secactpy/results/inflammation/validation_SecAct_singlecell.h5ad',
    'Inflammation_ext_CytoSig': '/vf/users/parks34/projects/2secactpy/results/inflammation/external_CytoSig_singlecell.h5ad',
    'Inflammation_ext_SecAct': '/vf/users/parks34/projects/2secactpy/results/inflammation/external_SecAct_singlecell.h5ad',
    'scAtlas_normal_CytoSig': '/vf/users/parks34/projects/2secactpy/results/scatlas/scatlas_normal_CytoSig_singlecell.h5ad',
    'scAtlas_normal_SecAct': '/vf/users/parks34/projects/2secactpy/results/scatlas/scatlas_normal_SecAct_singlecell.h5ad',
    'scAtlas_cancer_CytoSig': '/vf/users/parks34/projects/2secactpy/results/scatlas/scatlas_cancer_CytoSig_singlecell.h5ad',
    'scAtlas_cancer_SecAct': '/vf/users/parks34/projects/2secactpy/results/scatlas/scatlas_cancer_SecAct_singlecell.h5ad',
}

# Original atlas files for cell type info
ATLAS_FILES = {
    'CIMA': '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad',
    'Inflammation_main': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad',
    'Inflammation_val': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad',
    'Inflammation_ext': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_external_afterQC.h5ad',
    'scAtlas_normal': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad',
    'scAtlas_cancer': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad',
}

# Cell type column preferences
CELLTYPE_COLS = {
    'CIMA': 'cell_type_l2',
    'Inflammation_main': 'Level1',
    'Inflammation_val': 'Level1',
    'Inflammation_ext': 'Level1',
    'scAtlas_normal': 'cellType1',
    'scAtlas_cancer': 'cellType1',
}

OUTPUT_DIR = Path('/vf/users/parks34/projects/2secactpy/visualization/data')

# Chunk size for processing
CHUNK_SIZE = 500000


def load_cell_types(atlas_key: str) -> pd.Series:
    """Load cell type labels from original atlas file."""
    atlas_path = ATLAS_FILES[atlas_key]
    celltype_col = CELLTYPE_COLS[atlas_key]

    logger.info(f"Loading cell types from {atlas_key}...")

    with h5py.File(atlas_path, 'r') as f:
        # Get cell indices
        obs_index = f['obs']['_index'][:]
        if hasattr(obs_index[0], 'decode'):
            obs_index = [v.decode() for v in obs_index]

        # Get cell type column - handle different storage formats
        cell_types = None

        if celltype_col in f['obs']:
            col_data = f['obs'][celltype_col]

            # Check if it's a group (categorical) or dataset (direct array)
            if isinstance(col_data, h5py.Group):
                # Categorical storage: has 'codes' and 'categories' subgroups
                if 'codes' in col_data and 'categories' in col_data:
                    codes = col_data['codes'][:]
                    categories = col_data['categories'][:]
                    if hasattr(categories[0], 'decode'):
                        categories = [c.decode() for c in categories]
                    cell_types = [categories[c] for c in codes]
                else:
                    raise ValueError(f"Categorical column {celltype_col} missing codes/categories")
            else:
                # Direct array storage
                cell_types = col_data[:]
                if hasattr(cell_types[0], 'decode'):
                    cell_types = [v.decode() for v in cell_types]

        # Try legacy __categories format
        if cell_types is None:
            cat_key = f'__categories/{celltype_col}'
            if cat_key in f['obs']:
                categories = f['obs'][cat_key][:]
                if hasattr(categories[0], 'decode'):
                    categories = [c.decode() for c in categories]
                codes = f['obs'][celltype_col][:]
                cell_types = [categories[c] for c in codes]

        if cell_types is None:
            raise ValueError(f"Cell type column {celltype_col} not found or unsupported format")

    logger.info(f"Loaded {len(cell_types)} cell type labels")
    return pd.Series(cell_types, index=obs_index)


def process_singlecell_file(sc_path: str, cell_types: pd.Series, atlas_name: str, sig_type: str) -> list:
    """
    Process a single-cell activity h5ad file and compute per-cell-type statistics.

    Returns list of records with statistics per (cell_type, signature).
    """
    logger.info(f"Processing {sc_path}")

    results = []

    with h5py.File(sc_path, 'r') as f:
        # Get signature names
        var_names = f['var']['_index'][:]
        if hasattr(var_names[0], 'decode'):
            var_names = [v.decode() for v in var_names]

        n_cells, n_sigs = f['X'].shape
        logger.info(f"Shape: {n_cells} cells × {n_sigs} signatures")

        # Get cell indices
        obs_index = f['obs']['_index'][:]
        if hasattr(obs_index[0], 'decode'):
            obs_index = [v.decode() for v in obs_index]

        # Map cell indices to cell types
        cell_type_array = cell_types.reindex(obs_index).values
        unique_cell_types = np.unique(cell_type_array[~pd.isna(cell_type_array)])
        logger.info(f"Found {len(unique_cell_types)} cell types")

        # Process in chunks to avoid memory issues
        n_chunks = (n_cells + CHUNK_SIZE - 1) // CHUNK_SIZE

        # Accumulators for each (cell_type, signature)
        # Using online algorithm for mean and variance
        ct_sig_stats = {}  # {(cell_type, sig_idx): {'n': 0, 'mean': 0, 'M2': 0, 'values': []}}

        for chunk_idx in range(n_chunks):
            start = chunk_idx * CHUNK_SIZE
            end = min((chunk_idx + 1) * CHUNK_SIZE, n_cells)

            logger.info(f"Processing chunk {chunk_idx + 1}/{n_chunks} (cells {start}-{end})")

            # Load chunk of activity matrix
            X_chunk = f['X'][start:end, :]
            ct_chunk = cell_type_array[start:end]

            # Process each cell type in this chunk
            for ct in unique_cell_types:
                ct_mask = (ct_chunk == ct)
                if not np.any(ct_mask):
                    continue

                X_ct = X_chunk[ct_mask, :]

                # Update statistics for each signature
                for sig_idx in range(n_sigs):
                    key = (ct, sig_idx)
                    values = X_ct[:, sig_idx]

                    if key not in ct_sig_stats:
                        ct_sig_stats[key] = {
                            'n': 0,
                            'mean': 0.0,
                            'M2': 0.0,
                            'min': float('inf'),
                            'max': float('-inf'),
                            'sum': 0.0,
                        }

                    stats = ct_sig_stats[key]

                    # Welford's online algorithm for mean and variance
                    for val in values:
                        stats['n'] += 1
                        delta = val - stats['mean']
                        stats['mean'] += delta / stats['n']
                        delta2 = val - stats['mean']
                        stats['M2'] += delta * delta2
                        stats['min'] = min(stats['min'], val)
                        stats['max'] = max(stats['max'], val)
                        stats['sum'] += val

            del X_chunk
            gc.collect()

        # Convert accumulated stats to results
        logger.info("Computing final statistics...")
        for (ct, sig_idx), stats in ct_sig_stats.items():
            n = stats['n']
            if n < 10:
                continue

            mean_val = stats['mean']
            variance = stats['M2'] / n if n > 1 else 0
            std_val = np.sqrt(variance)

            results.append({
                'cell_type': str(ct),
                'signature': var_names[sig_idx],
                'signature_type': sig_type,
                'atlas': atlas_name,
                'mean_activity': round(float(mean_val), 4),
                'std_activity': round(float(std_val), 4),
                'min_activity': round(float(stats['min']), 4),
                'max_activity': round(float(stats['max']), 4),
                'n_cells': int(n),
            })

    logger.info(f"Generated {len(results)} records")
    return results


def main():
    logger.info("Starting single-cell activity JSON extraction")

    all_results = []

    # Process each atlas
    for sc_key, sc_path in SINGLECELL_FILES.items():
        if not Path(sc_path).exists():
            logger.warning(f"File not found: {sc_path}")
            continue

        # Determine atlas and signature type
        parts = sc_key.split('_')
        if sc_key.startswith('CIMA'):
            atlas_key = 'CIMA'
            atlas_name = 'CIMA'
            sig_type = parts[-1]  # CytoSig or SecAct
        elif sc_key.startswith('Inflammation'):
            cohort = parts[1]  # main, val, ext
            atlas_key = f'Inflammation_{cohort}'
            atlas_name = 'Inflammation'
            sig_type = parts[-1]
        elif sc_key.startswith('scAtlas'):
            tissue = parts[1]  # normal or cancer
            atlas_key = f'scAtlas_{tissue}'
            atlas_name = f'scAtlas_{tissue.capitalize()}'
            sig_type = parts[-1]
        else:
            logger.warning(f"Unknown file pattern: {sc_key}")
            continue

        logger.info("=" * 60)
        logger.info(f"Processing {sc_key} -> {atlas_name} ({sig_type})")

        try:
            # Load cell types
            cell_types = load_cell_types(atlas_key)

            # Process single-cell file
            results = process_singlecell_file(sc_path, cell_types, atlas_name, sig_type)
            all_results.extend(results)

            del cell_types
            gc.collect()

        except Exception as e:
            logger.error(f"Failed to process {sc_key}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        logger.error("No results generated")
        return 1

    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Atlases: {df['atlas'].unique().tolist()}")
    logger.info(f"Signature types: {df['signature_type'].unique().tolist()}")

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save combined single-cell activity file
    output_path = OUTPUT_DIR / 'singlecell_activity.json'
    with open(output_path, 'w') as f:
        json.dump(df.to_dict(orient='records'), f)
    logger.info(f"Saved {output_path}")

    # Save per-atlas files
    for atlas in df['atlas'].unique():
        for sig_type in df['signature_type'].unique():
            subset = df[(df['atlas'] == atlas) & (df['signature_type'] == sig_type)]
            if len(subset) == 0:
                continue

            filename = f"{atlas.lower()}_singlecell_{sig_type.lower()}.json"
            with open(OUTPUT_DIR / filename, 'w') as f:
                json.dump(subset.to_dict(orient='records'), f)
            logger.info(f"Saved {filename} ({len(subset)} records)")

    # Aggregate Inflammation cohorts
    inflam_df = df[df['atlas'] == 'Inflammation']
    if len(inflam_df) > 0:
        for sig_type in inflam_df['signature_type'].unique():
            subset = inflam_df[inflam_df['signature_type'] == sig_type]
            # Aggregate by cell_type and signature
            agg = subset.groupby(['cell_type', 'signature', 'signature_type', 'atlas']).agg({
                'mean_activity': 'mean',
                'std_activity': 'mean',  # Average of stds (approximation)
                'min_activity': 'min',
                'max_activity': 'max',
                'n_cells': 'sum',
            }).reset_index()

            filename = f"inflammation_singlecell_{sig_type.lower()}_combined.json"
            with open(OUTPUT_DIR / filename, 'w') as f:
                json.dump(agg.to_dict(orient='records'), f)
            logger.info(f"Saved {filename} ({len(agg)} records)")

    logger.info("=" * 60)
    logger.info("Done!")
    return 0


if __name__ == '__main__':
    exit(main())
