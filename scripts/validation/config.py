"""
Validation Test Configuration
============================
Configuration for GPU-based validation across all atlases, signatures, and levels.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ==============================================================================
# Data Paths
# ==============================================================================

DATA_ROOT = Path('/data/Jiang_Lab/Data/Seongyong')

ATLAS_CONFIG = {
    'cima': {
        'name': 'CIMA',
        'h5ad_path': DATA_ROOT / 'CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad',
        'n_cells': 6_484_974,
        'cell_type_columns': {
            'L1': 'cell_type_l1',
            'L2': 'cell_type_l2',
            'L3': 'cell_type_l3',
        },
        'sample_col': 'sample',
        'gene_col': None,  # Index
    },
    'inflammation': {
        'name': 'Inflammation',
        'h5ad_path': DATA_ROOT / 'Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad',
        'n_cells': 4_918_140,
        'cell_type_columns': {
            'L1': 'Level1',
            'L2': 'Level2',
        },
        'sample_col': 'sampleID',
        'gene_col': None,
    },
    'inflammation_val': {
        'name': 'Inflammation_Validation',
        'h5ad_path': DATA_ROOT / 'Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad',
        'n_cells': 849_922,
        'cell_type_columns': {
            'L1': 'Level1',
            'L2': 'Level2',
        },
        'sample_col': 'sampleID',
        'gene_col': None,
    },
    'scatlas_normal': {
        'name': 'scAtlas_Normal',
        'h5ad_path': DATA_ROOT / 'scAtlas_2025/igt_s9_fine_counts.h5ad',
        'n_cells': 2_293_951,
        'cell_type_columns': {
            'L1': 'subCluster',
            'L2': 'cellType2',
        },
        'sample_col': 'tissue',  # Use tissue as sample grouping
        'gene_col': None,
    },
    'scatlas_cancer': {
        'name': 'scAtlas_Cancer',
        'h5ad_path': DATA_ROOT / 'scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad',
        'n_cells': 4_146_975,
        'cell_type_columns': {
            'L1': 'subCluster',
            'L2': 'cellType2',
        },
        'sample_col': 'tissue',
        'gene_col': None,
    },
}

# ==============================================================================
# Signature Configuration
# ==============================================================================

SIGNATURE_CONFIG = {
    'cytosig': {
        'name': 'CytoSig',
        'n_signatures': 44,
        'loader': 'load_cytosig',
    },
    'lincytosig': {
        'name': 'LinCytoSig',
        'n_signatures': 178,
        'loader': 'load_lincytosig',
        'cell_type_specific': True,  # Requires cell type matching
    },
    'secact': {
        'name': 'SecAct',
        'n_signatures': 1249,
        'loader': 'load_secact',
    },
}

# ==============================================================================
# Aggregation Configuration
# ==============================================================================

AGGREGATION_CONFIG = {
    'pseudobulk': {
        'name': 'Pseudobulk',
        'description': 'Standard pseudobulk aggregation (sum counts per cell_type × sample)',
        'normalize': True,
        'min_cells': 10,
        'batch_size': 5000,
        'streaming_threshold': 50000,  # Use streaming if > this many samples
    },
    'resampled': {
        'name': 'Resampled Pseudobulk',
        'description': 'Bootstrap resampling to normalize cell counts',
        'n_cells_per_group': 100,  # Resample to this many cells
        'n_replicates': 10,  # Number of bootstrap replicates
        'min_cells': 50,  # Skip groups with fewer cells
        'batch_size': 5000,
        'streaming_threshold': 50000,
    },
    'singlecell': {
        'name': 'Single-cell',
        'description': 'Per-cell activity inference with streaming output',
        'batch_size': 5000,
        'max_cells': 100000,  # Subsample for validation
        'streaming_threshold': 20000,  # Use streaming earlier for single-cell
    },
}

# ==============================================================================
# Validation Parameters
# ==============================================================================

VALIDATION_CONFIG = {
    'n_rand': 1000,
    'seed': 42,
    'min_samples': 10,  # Minimum samples for correlation
    'fdr_method': 'fdr_bh',
    'alpha': 0.05,
}

# ==============================================================================
# Output Configuration
# ==============================================================================

OUTPUT_ROOT = Path('/vf/users/parks34/projects/2secactpy/results/validation')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# SLURM Configuration
# ==============================================================================

SLURM_CONFIG = {
    'partition': 'gpu',
    'gres': 'gpu:a100:1',
    'mem': '64G',
    'time': '08:00:00',
    'cpus_per_task': 8,
}

# ==============================================================================
# Test Matrix Generation
# ==============================================================================

def generate_test_matrix() -> List[Dict]:
    """Generate all validation test combinations."""
    tests = []

    for atlas_key, atlas_cfg in ATLAS_CONFIG.items():
        for sig_key, sig_cfg in SIGNATURE_CONFIG.items():
            for agg_key, agg_cfg in AGGREGATION_CONFIG.items():
                for level_key, level_col in atlas_cfg['cell_type_columns'].items():
                    tests.append({
                        'atlas': atlas_key,
                        'atlas_name': atlas_cfg['name'],
                        'signature': sig_key,
                        'signature_name': sig_cfg['name'],
                        'aggregation': agg_key,
                        'level': level_key,
                        'cell_type_col': level_col,
                        'sample_col': atlas_cfg['sample_col'],
                        'h5ad_path': str(atlas_cfg['h5ad_path']),
                    })

    return tests


def get_output_path(atlas: str, signature: str, aggregation: str, level: str) -> Path:
    """Get output path for a validation result."""
    return OUTPUT_ROOT / atlas / signature / f"{aggregation}_{level}.csv"


def get_scatter_data_path(atlas: str, signature: str, level: str) -> Path:
    """Get output path for scatter plot data."""
    return OUTPUT_ROOT / 'scatter_data' / f"{atlas}_{signature}_{level}.json"


if __name__ == '__main__':
    # Print test matrix summary
    tests = generate_test_matrix()
    print(f"Total validation tests: {len(tests)}")
    print()

    # Group by atlas
    from collections import Counter
    atlas_counts = Counter(t['atlas'] for t in tests)
    for atlas, count in atlas_counts.items():
        print(f"  {atlas}: {count} tests")

    print()
    print("Sample tests:")
    for t in tests[:5]:
        print(f"  {t['atlas_name']} / {t['signature_name']} / {t['aggregation']} / {t['level']}")
