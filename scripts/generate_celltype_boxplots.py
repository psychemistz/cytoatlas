#!/usr/bin/env python3
"""
Generate cell-type specific Age/BMI stratified boxplot data for CIMA.

This script computes median activity statistics for each combination of:
- Cell type
- Signature (CytoSig/SecAct)
- Age bin or BMI category
"""

import h5py
import json
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path


def get_sample_metadata(h5_path: str) -> pd.DataFrame:
    """Extract sample-level metadata (age, BMI) from CIMA h5ad file."""
    print("Reading sample metadata from h5ad...")

    with h5py.File(h5_path, 'r') as f:
        obs = f['obs']

        # Get sample IDs - handle categorical encoding
        sample_key = 'sample' if 'sample' in obs else 'sample_id'
        sample_data = obs[sample_key]
        if 'categories' in sample_data:
            # Categorical encoding
            codes = sample_data['codes'][:]
            categories = sample_data['categories'][:].astype(str)
            samples = categories[codes]
        else:
            samples = sample_data[:].astype(str)

        # Get age and BMI (numeric)
        age = obs['age'][:]
        bmi = obs['BMI'][:]

        # Get cell type - handle categorical encoding
        cell_type_key = 'cell_type_l4' if 'cell_type_l4' in obs else 'cell_type'
        ct_data = obs[cell_type_key]
        if 'categories' in ct_data:
            codes = ct_data['codes'][:]
            categories = ct_data['categories'][:].astype(str)
            cell_types = categories[codes]
        else:
            cell_types = ct_data[:].astype(str)

    # Create DataFrame
    df = pd.DataFrame({
        'sample': samples,
        'age': age,
        'bmi': bmi,
        'cell_type': cell_types,
    })

    # Get unique sample-level metadata
    sample_meta = df.groupby('sample').agg({
        'age': 'first',
        'bmi': 'first',
    }).reset_index()

    print(f"Found {len(sample_meta)} unique samples")
    print(f"Age range: {sample_meta['age'].min():.0f} - {sample_meta['age'].max():.0f}")
    print(f"BMI range: {sample_meta['bmi'].min():.1f} - {sample_meta['bmi'].max():.1f}")

    return sample_meta


def bin_age(age: float) -> str:
    """Bin age into decade groups."""
    if age < 30:
        return '<30'
    elif age < 40:
        return '30-39'
    elif age < 50:
        return '40-49'
    elif age < 60:
        return '50-59'
    elif age < 70:
        return '60-69'
    else:
        return '70+'


def bin_bmi(bmi: float) -> str:
    """Bin BMI into WHO categories."""
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'


def compute_boxplot_stats(values: np.ndarray) -> dict:
    """Compute boxplot statistics from values."""
    if len(values) == 0:
        return None

    values = values[~np.isnan(values)]
    if len(values) == 0:
        return None

    return {
        'min': float(np.min(values)),
        'q1': float(np.percentile(values, 25)),
        'median': float(np.median(values)),
        'q3': float(np.percentile(values, 75)),
        'max': float(np.max(values)),
        'mean': float(np.mean(values)),
        'n': int(len(values)),
    }


def generate_celltype_boxplots(
    activity_h5ad_path: str,
    sample_meta: pd.DataFrame,
    sig_type: str = 'CytoSig',
) -> tuple[list[dict], list[dict]]:
    """
    Generate cell-type specific boxplot data for age and BMI stratification.

    Returns:
        (age_boxplots, bmi_boxplots): Lists of boxplot statistics
    """
    print(f"Loading {sig_type} activity data...")
    adata = ad.read_h5ad(activity_h5ad_path)

    # Activity matrix: signatures (rows) x sample-celltype (cols)
    activity = pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var.index)
    var_meta = adata.var.copy()

    signatures = activity.index.tolist()
    cell_types = var_meta['cell_type'].unique().tolist()

    print(f"Signatures: {len(signatures)}")
    print(f"Cell types: {len(cell_types)}")

    # Add age/BMI bins to sample metadata
    sample_meta = sample_meta.copy()
    sample_meta['age_bin'] = sample_meta['age'].apply(bin_age)
    sample_meta['bmi_bin'] = sample_meta['bmi'].apply(bin_bmi)

    # Create lookup for sample metadata
    sample_to_age_bin = dict(zip(sample_meta['sample'], sample_meta['age_bin']))
    sample_to_bmi_bin = dict(zip(sample_meta['sample'], sample_meta['bmi_bin']))

    age_boxplots = []
    bmi_boxplots = []

    age_bins = ['<30', '30-39', '40-49', '50-59', '60-69', '70+']
    bmi_bins = ['Underweight', 'Normal', 'Overweight', 'Obese']

    total = len(signatures) * len(cell_types)
    processed = 0

    for sig in signatures:
        sig_values = activity.loc[sig]

        for ct in cell_types:
            # Get columns for this cell type
            ct_mask = var_meta['cell_type'] == ct
            ct_cols = var_meta.index[ct_mask]
            ct_samples = var_meta.loc[ct_cols, 'sample'].values
            ct_values = sig_values[ct_cols].values

            # Compute age-binned statistics
            for age_bin in age_bins:
                bin_mask = np.array([sample_to_age_bin.get(s) == age_bin for s in ct_samples])
                bin_values = ct_values[bin_mask]

                stats = compute_boxplot_stats(bin_values)
                if stats:
                    stats['signature'] = sig
                    stats['sig_type'] = sig_type
                    stats['cell_type'] = ct
                    stats['bin'] = age_bin
                    age_boxplots.append(stats)

            # Compute BMI-binned statistics
            for bmi_bin in bmi_bins:
                bin_mask = np.array([sample_to_bmi_bin.get(s) == bmi_bin for s in ct_samples])
                bin_values = ct_values[bin_mask]

                stats = compute_boxplot_stats(bin_values)
                if stats:
                    stats['signature'] = sig
                    stats['sig_type'] = sig_type
                    stats['cell_type'] = ct
                    stats['bin'] = bmi_bin
                    bmi_boxplots.append(stats)

            processed += 1
            if processed % 100 == 0:
                print(f"  Processed {processed}/{total} combinations...")

    print(f"Generated {len(age_boxplots)} age boxplots, {len(bmi_boxplots)} BMI boxplots")

    return age_boxplots, bmi_boxplots


def main():
    # Paths
    cima_h5_path = '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad'
    cytosig_path = '/data/parks34/projects/2secactpy/results/cima/CIMA_CytoSig_pseudobulk.h5ad'
    secact_path = '/data/parks34/projects/2secactpy/results/cima/CIMA_SecAct_pseudobulk.h5ad'
    output_path = '/vf/users/parks34/projects/2secactpy/visualization/data/age_bmi_boxplots.json'

    # Get sample metadata
    sample_meta = get_sample_metadata(cima_h5_path)

    # Load existing boxplot data
    print(f"\nLoading existing boxplot data from {output_path}...")
    with open(output_path) as f:
        boxplot_data = json.load(f)

    # Generate CytoSig cell-type specific boxplots
    print("\n=== Generating CytoSig cell-type boxplots ===")
    cytosig_age, cytosig_bmi = generate_celltype_boxplots(cytosig_path, sample_meta, 'CytoSig')

    # Generate SecAct cell-type specific boxplots
    print("\n=== Generating SecAct cell-type boxplots ===")
    secact_age, secact_bmi = generate_celltype_boxplots(secact_path, sample_meta, 'SecAct')

    # Keep sample-level data (cell_type is None or 'All')
    cima_age_sample = [r for r in boxplot_data['cima']['age'] if r.get('cell_type') in (None, 'All')]
    cima_bmi_sample = [r for r in boxplot_data['cima']['bmi'] if r.get('cell_type') in (None, 'All')]

    # Combine sample-level + CytoSig cell-type + SecAct cell-type
    boxplot_data['cima']['age'] = cima_age_sample + cytosig_age + secact_age
    boxplot_data['cima']['bmi'] = cima_bmi_sample + cytosig_bmi + secact_bmi

    # Add cell_types list
    cell_types = sorted(list(set(r['cell_type'] for r in cytosig_age if r.get('cell_type'))))
    boxplot_data['cima']['cell_types'] = cell_types

    print(f"\nCIMA age boxplots: {len(boxplot_data['cima']['age'])} total")
    print(f"  CytoSig: {len(cytosig_age)} cell-type specific")
    print(f"  SecAct: {len(secact_age)} cell-type specific")
    print(f"CIMA bmi boxplots: {len(boxplot_data['cima']['bmi'])} total")
    print(f"CIMA cell types: {len(cell_types)}")

    # Save updated data
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(boxplot_data, f)

    print("Done!")


if __name__ == '__main__':
    main()
