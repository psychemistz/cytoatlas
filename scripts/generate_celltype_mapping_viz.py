#!/usr/bin/env python3
"""
Generate cell type mapping visualization data for Cross-Atlas panel.

Creates JSON data for the Cell Type Mapping panel showing:
1. Coarse level (8 lineages) mapping across atlases
2. Fine level (~30 types) mapping with atlas-specific annotations
3. Cell counts per type per atlas
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import anndata as ad

# Import mapping module
import sys
sys.path.insert(0, str(Path(__file__).parent))
from cell_type_mapping import (
    COARSE_LINEAGES, FINE_TYPES,
    CIMA_TO_COARSE, INFLAMMATION_TO_COARSE, SCATLAS_TO_COARSE,
    CIMA_TO_FINE, INFLAMMATION_TO_FINE, SCATLAS_TO_FINE,
    get_shared_types, summarize_mapping
)

# Paths
RESULTS_DIR = Path('/data/parks34/projects/2secactpy/results')
VIZ_OUTPUT_DIR = Path('/data/parks34/projects/2secactpy/visualization/data')

# Raw data paths
CIMA_H5AD = Path('/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad')
INFLAM_H5AD = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad')
SCATLAS_H5AD = Path('/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad')


def get_cima_cell_counts() -> dict:
    """Get cell counts per cell type from CIMA pseudobulk results."""
    print("Loading CIMA cell counts...")
    # Use pseudobulk file which has cell counts
    pb_path = RESULTS_DIR / 'cima' / 'CIMA_CytoSig_pseudobulk.h5ad'
    if pb_path.exists():
        adata = ad.read_h5ad(pb_path)
        # var contains sample-celltype combos with n_cells
        counts = adata.var.groupby('cell_type')['n_cells'].sum().to_dict()
        print(f"  Found {len(counts)} cell types")
        return counts
    return {}


def get_inflammation_cell_counts() -> dict:
    """Get cell counts per cell type from Inflammation Atlas."""
    print("Loading Inflammation Atlas cell counts...")
    adata = ad.read_h5ad(INFLAM_H5AD, backed='r')
    counts = adata.obs['Level2'].value_counts().to_dict()
    print(f"  Found {len(counts)} cell types")
    return counts


def get_scatlas_cell_counts() -> dict:
    """Get cell counts per cell type from scAtlas."""
    print("Loading scAtlas cell counts...")
    adata = ad.read_h5ad(SCATLAS_H5AD, backed='r')
    # Filter to immune compartment only
    immune_mask = adata.obs['compartment'] == 'Immune'
    counts = adata.obs.loc[immune_mask, 'subCluster'].value_counts().to_dict()
    print(f"  Found {len(counts)} immune cell types")
    return counts


def build_coarse_mapping_data(cima_counts, inflam_counts, scatlas_counts) -> dict:
    """Build coarse level mapping data."""
    print("\nBuilding coarse level mapping...")

    # Aggregate counts by coarse type for each atlas
    coarse_data = []

    for lineage in COARSE_LINEAGES:
        entry = {
            'lineage': lineage,
            'cima': {'types': [], 'total_cells': 0},
            'inflammation': {'types': [], 'total_cells': 0},
            'scatlas': {'types': [], 'total_cells': 0}
        }

        # CIMA
        for orig, mapped in CIMA_TO_COARSE.items():
            if mapped == lineage:
                count = cima_counts.get(orig, 0)
                entry['cima']['types'].append({'name': orig, 'cells': count})
                entry['cima']['total_cells'] += count

        # Inflammation
        for orig, mapped in INFLAMMATION_TO_COARSE.items():
            if mapped == lineage and mapped is not None:
                count = inflam_counts.get(orig, 0)
                entry['inflammation']['types'].append({'name': orig, 'cells': count})
                entry['inflammation']['total_cells'] += count

        # scAtlas
        for orig, mapped in SCATLAS_TO_COARSE.items():
            if mapped == lineage:
                count = scatlas_counts.get(orig, 0)
                entry['scatlas']['types'].append({'name': orig, 'cells': count})
                entry['scatlas']['total_cells'] += count

        # Sort by cell count
        for atlas in ['cima', 'inflammation', 'scatlas']:
            entry[atlas]['types'].sort(key=lambda x: -x['cells'])

        coarse_data.append(entry)

    return coarse_data


def build_fine_mapping_data(cima_counts, inflam_counts, scatlas_counts) -> dict:
    """Build fine level mapping data."""
    print("\nBuilding fine level mapping...")

    fine_data = []

    # Get all fine types that exist in at least one atlas
    all_fine_types = set()
    for mapping in [CIMA_TO_FINE, INFLAMMATION_TO_FINE, SCATLAS_TO_FINE]:
        all_fine_types.update(v for v in mapping.values() if v)

    for fine_type in sorted(all_fine_types):
        entry = {
            'fine_type': fine_type,
            'cima': {'types': [], 'total_cells': 0},
            'inflammation': {'types': [], 'total_cells': 0},
            'scatlas': {'types': [], 'total_cells': 0}
        }

        # CIMA
        for orig, mapped in CIMA_TO_FINE.items():
            if mapped == fine_type:
                count = cima_counts.get(orig, 0)
                entry['cima']['types'].append({'name': orig, 'cells': count})
                entry['cima']['total_cells'] += count

        # Inflammation
        for orig, mapped in INFLAMMATION_TO_FINE.items():
            if mapped == fine_type:
                count = inflam_counts.get(orig, 0)
                entry['inflammation']['types'].append({'name': orig, 'cells': count})
                entry['inflammation']['total_cells'] += count

        # scAtlas
        for orig, mapped in SCATLAS_TO_FINE.items():
            if mapped == fine_type:
                count = scatlas_counts.get(orig, 0)
                entry['scatlas']['types'].append({'name': orig, 'cells': count})
                entry['scatlas']['total_cells'] += count

        # Sort by cell count
        for atlas in ['cima', 'inflammation', 'scatlas']:
            entry[atlas]['types'].sort(key=lambda x: -x['cells'])

        fine_data.append(entry)

    return fine_data


def build_sankey_data(coarse_data) -> dict:
    """Build Sankey diagram data for coarse level mapping."""
    print("\nBuilding Sankey diagram data...")

    nodes = []
    links = []
    node_idx = {}

    # Add atlas nodes
    atlases = ['CIMA', 'Inflammation', 'scAtlas']
    for atlas in atlases:
        node_idx[atlas] = len(nodes)
        nodes.append({'name': atlas, 'category': 'atlas'})

    # Add lineage nodes
    for lineage in COARSE_LINEAGES:
        node_idx[lineage] = len(nodes)
        nodes.append({'name': lineage, 'category': 'lineage'})

    # Add links from atlases to lineages
    for entry in coarse_data:
        lineage = entry['lineage']
        for atlas_key, atlas_name in [('cima', 'CIMA'), ('inflammation', 'Inflammation'), ('scatlas', 'scAtlas')]:
            if entry[atlas_key]['total_cells'] > 0:
                links.append({
                    'source': node_idx[atlas_name],
                    'target': node_idx[lineage],
                    'value': entry[atlas_key]['total_cells'],
                    'n_types': len(entry[atlas_key]['types'])
                })

    return {'nodes': nodes, 'links': links}


def main():
    print("=" * 60)
    print("Generating Cell Type Mapping Visualization Data")
    print("=" * 60)

    # Get cell counts from each atlas
    cima_counts = get_cima_cell_counts()
    inflam_counts = get_inflammation_cell_counts()
    scatlas_counts = get_scatlas_cell_counts()

    # Build mapping data
    coarse_data = build_coarse_mapping_data(cima_counts, inflam_counts, scatlas_counts)
    fine_data = build_fine_mapping_data(cima_counts, inflam_counts, scatlas_counts)
    sankey_data = build_sankey_data(coarse_data)

    # Get shared types
    shared_coarse = list(get_shared_types('coarse'))
    shared_fine = list(get_shared_types('fine'))

    # Build summary statistics
    summary = {
        'coarse': {
            'n_lineages': len(COARSE_LINEAGES),
            'n_shared': len(shared_coarse),
            'shared_types': shared_coarse,
            'cima_types': len([k for k, v in CIMA_TO_COARSE.items() if v]),
            'inflammation_types': len([k for k, v in INFLAMMATION_TO_COARSE.items() if v]),
            'scatlas_types': len([k for k, v in SCATLAS_TO_COARSE.items() if v])
        },
        'fine': {
            'n_types': len(set(v for v in CIMA_TO_FINE.values() if v) |
                          set(v for v in INFLAMMATION_TO_FINE.values() if v) |
                          set(v for v in SCATLAS_TO_FINE.values() if v)),
            'n_shared': len(shared_fine),
            'shared_types': shared_fine
        }
    }

    # Compile output
    output = {
        'coarse_mapping': coarse_data,
        'fine_mapping': fine_data,
        'sankey': sankey_data,
        'summary': summary,
        'lineages': COARSE_LINEAGES
    }

    # Save to JSON
    output_path = VIZ_OUTPUT_DIR / 'celltype_mapping.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Also update cross_atlas.json with mapping info
    cross_atlas_path = VIZ_OUTPUT_DIR / 'cross_atlas.json'
    if cross_atlas_path.exists():
        with open(cross_atlas_path, 'r') as f:
            cross_atlas = json.load(f)
    else:
        cross_atlas = {}

    cross_atlas['celltype_mapping'] = output
    cross_atlas['shared_cell_types'] = shared_coarse  # For backward compatibility

    with open(cross_atlas_path, 'w') as f:
        json.dump(cross_atlas, f)
    print(f"Updated: {cross_atlas_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nCoarse Level ({len(COARSE_LINEAGES)} lineages):")
    for entry in coarse_data:
        lineage = entry['lineage']
        cima_n = len(entry['cima']['types'])
        inflam_n = len(entry['inflammation']['types'])
        scatlas_n = len(entry['scatlas']['types'])
        print(f"  {lineage}: CIMA={cima_n}, Inflammation={inflam_n}, scAtlas={scatlas_n}")

    print(f"\nFine Level ({len(fine_data)} types):")
    print(f"  Shared across all 3 atlases: {len(shared_fine)}")

    print("\nDone!")


if __name__ == '__main__':
    main()
