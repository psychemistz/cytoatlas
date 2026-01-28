#!/usr/bin/env python3
"""
Preprocess data for interactive HTML visualization.
Creates JSON files optimized for web display.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats

# Paths
RESULTS_DIR = Path("/vf/users/parks34/projects/2secactpy/results")
CIMA_DIR = RESULTS_DIR / "cima"
SCATLAS_DIR = RESULTS_DIR / "scatlas"
INFLAM_DIR = RESULTS_DIR / "inflammation"
OUTPUT_DIR = Path("/vf/users/parks34/projects/2secactpy/visualization/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def round_floats(df, decimals=4):
    """Round float columns in dataframe."""
    float_cols = df.select_dtypes(include=['float64', 'float32']).columns
    for col in float_cols:
        df[col] = df[col].round(decimals)
    return df


def preprocess_cima_correlations():
    """Preprocess CIMA correlation data for visualization."""
    print("Processing CIMA correlations...")

    # Age correlations - filter to CytoSig only for faster load
    age_df = round_floats(pd.read_csv(CIMA_DIR / "CIMA_correlation_age.csv"))
    age_cytosig = age_df[age_df['signature'] == 'CytoSig'].to_dict(orient='records')
    age_secact = age_df[age_df['signature'] == 'SecAct'].to_dict(orient='records') if 'SecAct' in age_df['signature'].values else []

    # BMI correlations
    bmi_df = round_floats(pd.read_csv(CIMA_DIR / "CIMA_correlation_bmi.csv"))
    bmi_cytosig = bmi_df[bmi_df['signature'] == 'CytoSig'].to_dict(orient='records')
    bmi_secact = bmi_df[bmi_df['signature'] == 'SecAct'].to_dict(orient='records') if 'SecAct' in bmi_df['signature'].values else []

    # Biochemistry correlations - include both CytoSig and SecAct
    biochem_df = round_floats(pd.read_csv(CIMA_DIR / "CIMA_correlation_biochemistry.csv"))
    biochem_cytosig = biochem_df[biochem_df['signature'] == 'CytoSig'].to_dict(orient='records')
    biochem_secact = biochem_df[biochem_df['signature'] == 'SecAct'].to_dict(orient='records') if 'SecAct' in biochem_df['signature'].values else []

    # Combine CytoSig and SecAct data
    correlations = {
        'age': age_cytosig + age_secact,
        'bmi': bmi_cytosig + bmi_secact,
        'biochemistry': biochem_cytosig + biochem_secact
    }

    with open(OUTPUT_DIR / "cima_correlations.json", 'w') as f:
        json.dump(correlations, f)

    print(f"  Age: {len(correlations['age'])} records")
    print(f"  BMI: {len(correlations['bmi'])} records")
    print(f"  Biochemistry: {len(correlations['biochemistry'])} records")

    return correlations


def preprocess_cima_metabolites():
    """Preprocess top metabolite correlations (top 500 by absolute rho)."""
    print("Processing CIMA metabolite correlations...")

    met_df = pd.read_csv(CIMA_DIR / "CIMA_correlation_metabolites.csv")

    # Get top correlations by absolute rho for each signature type
    top_mets = []
    for sig_type in met_df['signature'].unique():
        subset = met_df[met_df['signature'] == sig_type].copy()
        subset['abs_rho'] = subset['rho'].abs()
        top = subset.nlargest(250, 'abs_rho').drop(columns=['abs_rho'])
        top_mets.append(top)

    top_met_df = pd.concat(top_mets)
    met_data = top_met_df.to_dict(orient='records')

    with open(OUTPUT_DIR / "cima_metabolites_top.json", 'w') as f:
        json.dump(met_data, f)

    print(f"  Top metabolite correlations: {len(met_data)} records")

    return met_data


def preprocess_cima_differential():
    """Preprocess CIMA differential analysis results."""
    print("Processing CIMA differential results...")

    diff_df = pd.read_csv(CIMA_DIR / "CIMA_differential_demographics.csv")

    # Calculate log2 fold change and -log10 pvalue for volcano plot
    diff_df['log2fc'] = np.log2(
        (diff_df['median_g1'] + 1) / (diff_df['median_g2'] + 1)
    )
    diff_df['neg_log10_pval'] = -np.log10(diff_df['pvalue'].clip(lower=1e-300))

    # Round floats to reduce file size
    diff_df = round_floats(diff_df)

    # Include both CytoSig and SecAct
    diff_data = diff_df.to_dict(orient='records')

    with open(OUTPUT_DIR / "cima_differential.json", 'w') as f:
        json.dump(diff_data, f)

    print(f"  Differential results: {len(diff_data)} records")
    print(f"    CytoSig: {len(diff_df[diff_df['signature'] == 'CytoSig'])} records")
    print(f"    SecAct: {len(diff_df[diff_df['signature'] == 'SecAct'])} records")

    return diff_data


def preprocess_inflammation():
    """Preprocess Inflammation Atlas data from h5ad files (CytoSig + SecAct)."""
    print("Processing Inflammation Atlas data...")

    try:
        import anndata as ad
    except ImportError:
        print("  Skipping inflammation - anndata not available")
        return None

    all_celltype_activity = []

    # Process both CytoSig and SecAct
    for sig_type in ['CytoSig', 'SecAct']:
        h5ad_file = INFLAM_DIR / f"main_{sig_type}_pseudobulk.h5ad"
        if not h5ad_file.exists():
            print(f"  Skipping {sig_type} - {h5ad_file} not found")
            continue

        # Load the pseudobulk data
        adata = ad.read_h5ad(h5ad_file)
        print(f"  {sig_type} shape: {adata.shape}")

        # Extract activity matrix (proteins x sample_celltype)
        activity_df = pd.DataFrame(
            adata.X,
            index=adata.obs_names,  # proteins/cytokines
            columns=adata.var_names  # sample_celltype
        )

        # Get cell type info
        celltype_info = adata.var[['sample', 'cell_type', 'n_cells']].copy()

        # For SecAct, limit to top 100 most variable signatures
        signatures = activity_df.index.tolist()
        if sig_type == 'SecAct' and len(signatures) > 100:
            # Calculate variance across all samples for each signature
            sig_variance = activity_df.var(axis=1)
            top_signatures = sig_variance.nlargest(100).index.tolist()
            activity_df = activity_df.loc[top_signatures]
            signatures = top_signatures
            print(f"    Filtered to top 100 most variable SecAct signatures")

        # Aggregate by cell type (mean across samples)
        for ct in celltype_info['cell_type'].unique():
            ct_cols = celltype_info[celltype_info['cell_type'] == ct].index
            mean_activity = activity_df[ct_cols].mean(axis=1)
            n_samples = len(ct_cols)
            total_cells = celltype_info.loc[ct_cols, 'n_cells'].sum()

            for sig in mean_activity.index:
                all_celltype_activity.append({
                    'cell_type': ct,
                    'signature': sig,
                    'signature_type': sig_type,
                    'mean_activity': round(mean_activity[sig], 4),
                    'n_samples': n_samples,
                    'n_cells': int(total_cells)
                })

    # Save
    with open(OUTPUT_DIR / "inflammation_celltype.json", 'w') as f:
        json.dump(all_celltype_activity, f)

    # Count stats
    cytosig_count = len([x for x in all_celltype_activity if x['signature_type'] == 'CytoSig'])
    secact_count = len([x for x in all_celltype_activity if x['signature_type'] == 'SecAct'])
    print(f"  CytoSig records: {cytosig_count}")
    print(f"  SecAct records: {secact_count}")
    print(f"  Total output records: {len(all_celltype_activity)}")

    return all_celltype_activity


def preprocess_scatlas_organs():
    """Preprocess scAtlas organ signature data."""
    print("Processing scAtlas organ signatures...")

    organ_df = pd.read_csv(SCATLAS_DIR / "normal_organ_signatures.csv")

    # Select only needed columns
    cols = ['organ', 'signature', 'mean_activity', 'specificity_score', 'n_cells', 'signature_type']
    organ_df = organ_df[[c for c in cols if c in organ_df.columns]]

    # Process CytoSig (all signatures)
    cytosig_df = organ_df[organ_df['signature_type'] == 'CytoSig'].copy()

    # Process SecAct (top 100 most variable signatures to limit file size)
    secact_df = organ_df[organ_df['signature_type'] == 'SecAct'].copy()
    if len(secact_df) > 0:
        # Calculate variance per signature across organs
        sig_variance = secact_df.groupby('signature')['mean_activity'].var()
        top_secact_sigs = sig_variance.nlargest(100).index.tolist()
        secact_df = secact_df[secact_df['signature'].isin(top_secact_sigs)]
        print(f"  SecAct: filtered to top 100 most variable signatures")

    # Combine CytoSig and SecAct
    combined_df = pd.concat([cytosig_df, secact_df], ignore_index=True)

    # Round floats to reduce JSON size
    for col in ['mean_activity', 'specificity_score']:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].round(4)

    organ_data = combined_df.to_dict(orient='records')

    with open(OUTPUT_DIR / "scatlas_organs.json", 'w') as f:
        json.dump(organ_data, f)

    print(f"  Organ signatures (CytoSig): {len(cytosig_df)} records")
    print(f"  Organ signatures (SecAct): {len(secact_df)} records")
    print(f"  Total organ signatures: {len(organ_data)} records")

    # Also create top signatures per organ
    top_df = pd.read_csv(SCATLAS_DIR / "normal_top_organ_signatures.csv")
    # Round floats
    for col in ['mean_activity', 'other_mean', 'specificity_score']:
        if col in top_df.columns:
            top_df[col] = top_df[col].round(4)
    top_data = top_df.to_dict(orient='records')

    with open(OUTPUT_DIR / "scatlas_organs_top.json", 'w') as f:
        json.dump(top_data, f)

    print(f"  Top organ signatures: {len(top_data)} records")

    return organ_data


def preprocess_scatlas_celltypes():
    """Preprocess scAtlas cell type signatures (sampled for web performance)."""
    print("Processing scAtlas cell type signatures...")

    ct_df = pd.read_csv(SCATLAS_DIR / "normal_celltype_signatures.csv")

    # Get unique cell types and organs
    cell_types = ct_df['cell_type'].unique().tolist()
    organs = sorted(ct_df['organ'].unique().tolist())

    # Process CytoSig - all signatures
    cytosig_df = ct_df[ct_df['signature_type'] == 'CytoSig'].copy()
    cytosig_signatures = sorted(cytosig_df['signature'].unique().tolist())

    # Process SecAct - top 50 most variable signatures
    secact_df = ct_df[ct_df['signature_type'] == 'SecAct'].copy()
    if len(secact_df) > 0:
        sig_variance = secact_df.groupby('signature')['mean_activity'].var()
        top_secact_sigs = sig_variance.nlargest(50).index.tolist()
        secact_df = secact_df[secact_df['signature'].isin(top_secact_sigs)]
        secact_signatures = sorted(secact_df['signature'].unique().tolist())
        print(f"  SecAct: filtered to top 50 most variable signatures")
    else:
        secact_signatures = []

    # Combine both signature types
    combined_df = pd.concat([cytosig_df, secact_df], ignore_index=True)

    # Keep organ information for filtering
    # Select columns: cell_type, organ, signature, mean_activity, signature_type
    combined_df = combined_df[['cell_type', 'organ', 'signature', 'mean_activity', 'signature_type']].copy()
    combined_df['mean_activity'] = combined_df['mean_activity'].round(4)

    # Get top 100 most variable cell types (based on CytoSig)
    ct_variance = cytosig_df.groupby('cell_type')['mean_activity'].var()
    top_cts = ct_variance.nlargest(100).index.tolist()

    # Filter to top cell types
    filtered = combined_df[combined_df['cell_type'].isin(top_cts)]

    ct_data = {
        'data': filtered.to_dict(orient='records'),
        'all_cell_types': cell_types,
        'top_cell_types': top_cts,
        'organs': organs,
        'cytosig_signatures': cytosig_signatures,
        'secact_signatures': secact_signatures,
        'signature_counts': {
            'CytoSig': len(cytosig_signatures),
            'SecAct': len(secact_signatures)
        }
    }

    with open(OUTPUT_DIR / "scatlas_celltypes.json", 'w') as f:
        json.dump(ct_data, f)

    print(f"  Cell types: {len(cell_types)}")
    print(f"  Top cell types: {len(top_cts)}")
    print(f"  Organs: {len(organs)}")
    print(f"  CytoSig signatures: {len(cytosig_signatures)}")
    print(f"  SecAct signatures: {len(secact_signatures)}")
    print(f"  Total records: {len(filtered)}")

    return ct_data


def preprocess_cancer_comparison():
    """Preprocess Tumor vs Adjacent comparison data with paired donor analysis using single-cell activities."""
    print("Processing cancer vs normal (Tumor vs Adjacent) PAIRED comparison...")

    import anndata as ad
    from scipy import stats

    # Paths to single-cell activity files
    CANCER_CYTOSIG = RESULTS_DIR / "scatlas" / "scatlas_cancer_CytoSig_singlecell.h5ad"
    CANCER_SECACT = RESULTS_DIR / "scatlas" / "scatlas_cancer_SecAct_singlecell.h5ad"
    CANCER_COUNTS = Path('/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad')

    # Load metadata from counts file
    print("  Loading PanCancer metadata...")
    counts = ad.read_h5ad(CANCER_COUNTS, backed='r')
    obs = counts.obs[['donorID', 'tissue', 'subCluster', 'cancerType']].copy()
    obs['cell_type'] = obs['subCluster']
    obs = obs.reset_index(drop=True)  # Reset to numeric index to match activity file
    counts.file.close()

    print(f"  Total cells: {len(obs)}")

    # Find donors with both Tumor and Adjacent
    donor_tissues = obs.groupby('donorID')['tissue'].apply(lambda x: set(x.unique()))
    paired_donors = donor_tissues[donor_tissues.apply(lambda x: 'Tumor' in x and 'Adjacent' in x)].index.tolist()
    print(f"  Paired donors (both Tumor & Adjacent): {len(paired_donors)}")

    # Get indices of cells from paired donors
    paired_mask = obs['donorID'].isin(paired_donors)
    paired_indices = obs[paired_mask].index.tolist()
    obs_paired = obs[paired_mask].copy()
    print(f"  Cells from paired donors: {len(paired_indices)}")

    # Load CytoSig activities
    print("  Loading CytoSig single-cell activities...")
    activity = ad.read_h5ad(CANCER_CYTOSIG, backed='r')

    # Get signature names from the CSV (activity file has numeric column names)
    cytosig_beta = pd.read_csv(SCATLAS_DIR / "cancer_cytosig_beta.csv", index_col=0)
    signatures = cytosig_beta.index.tolist()
    print(f"  CytoSig signatures: {len(signatures)}")

    # Load activities for paired cells
    print("  Extracting paired cell activities...")
    activities_matrix = activity[paired_indices, :].X
    activities = pd.DataFrame(
        activities_matrix,
        columns=signatures
    )
    activities['donorID'] = obs_paired['donorID'].values
    activities['tissue'] = obs_paired['tissue'].values
    activities['cell_type'] = obs_paired['cell_type'].values
    activities['cancerType'] = obs_paired['cancerType'].values
    activity.file.close()

    # Aggregate by donor + tissue + cell_type (mean activity)
    print("  Aggregating by donor + tissue + cell_type...")
    agg = activities.groupby(['donorID', 'tissue', 'cell_type', 'cancerType']).mean().reset_index()

    # Find donor-celltype combinations with both Tumor and Adjacent
    donor_ct = agg.groupby(['donorID', 'cell_type'])['tissue'].apply(lambda x: set(x.unique()))
    paired_donor_ct = donor_ct[donor_ct.apply(lambda x: 'Tumor' in x and 'Adjacent' in x)]
    print(f"  Paired donor-celltype combinations: {len(paired_donor_ct)}")

    # Calculate paired differences
    comparison_data = []
    cell_type_stats = {}

    for (donor, ct), _ in paired_donor_ct.items():
        tumor_row = agg[(agg['donorID'] == donor) & (agg['cell_type'] == ct) & (agg['tissue'] == 'Tumor')]
        adjacent_row = agg[(agg['donorID'] == donor) & (agg['cell_type'] == ct) & (agg['tissue'] == 'Adjacent')]

        if len(tumor_row) == 1 and len(adjacent_row) == 1:
            cancer_type = tumor_row['cancerType'].values[0]
            for sig in signatures:
                tumor_val = tumor_row[sig].values[0]
                adjacent_val = adjacent_row[sig].values[0]
                diff = tumor_val - adjacent_val

                key = (ct, sig)
                if key not in cell_type_stats:
                    cell_type_stats[key] = {'diffs': [], 'tumor': [], 'adjacent': [], 'cancer_types': []}
                cell_type_stats[key]['diffs'].append(diff)
                cell_type_stats[key]['tumor'].append(tumor_val)
                cell_type_stats[key]['adjacent'].append(adjacent_val)
                cell_type_stats[key]['cancer_types'].append(cancer_type)

    # Compute summary statistics for each cell_type x signature
    print("  Computing paired statistics...")
    for (ct, sig), data in cell_type_stats.items():
        n_pairs = len(data['diffs'])
        if n_pairs < 3:  # Need at least 3 pairs for meaningful stats
            continue

        mean_diff = np.mean(data['diffs'])
        std_diff = np.std(data['diffs'])
        mean_tumor = np.mean(data['tumor'])
        mean_adjacent = np.mean(data['adjacent'])

        # Paired t-test
        try:
            t_stat, p_value = stats.ttest_rel(data['tumor'], data['adjacent'])
            if np.isnan(p_value):
                p_value = 1.0
        except:
            p_value = 1.0

        comparison_data.append({
            'cell_type': ct,
            'signature': sig,
            'mean_tumor': round(float(mean_tumor), 4),
            'mean_adjacent': round(float(mean_adjacent), 4),
            'mean_difference': round(float(mean_diff), 4),
            'std_difference': round(float(std_diff), 4),
            'n_pairs': n_pairs,
            'p_value': round(float(p_value), 6),
            'signature_type': 'CytoSig'
        })

    print(f"  CytoSig comparisons: {len(comparison_data)}")

    # Now process SecAct
    print("  Loading SecAct single-cell activities...")
    activity_secact = ad.read_h5ad(CANCER_SECACT, backed='r')

    # Get SecAct signature names
    secact_beta = pd.read_csv(SCATLAS_DIR / "cancer_secact_beta.csv", index_col=0)
    secact_signatures = secact_beta.index.tolist()

    # Get top 50 most variable SecAct signatures from the aggregated data
    sig_variance = secact_beta.var(axis=1)
    top_secact_sigs = sig_variance.nlargest(50).index.tolist()
    top_secact_idx = [secact_signatures.index(s) for s in top_secact_sigs]
    print(f"  Top 50 SecAct signatures selected")

    # Load activities for paired cells (only top 50 signatures)
    activities_secact_matrix = activity_secact[paired_indices, :][:, top_secact_idx].X
    activities_secact = pd.DataFrame(
        activities_secact_matrix,
        columns=top_secact_sigs
    )
    activities_secact['donorID'] = obs_paired['donorID'].values
    activities_secact['tissue'] = obs_paired['tissue'].values
    activities_secact['cell_type'] = obs_paired['cell_type'].values
    activities_secact['cancerType'] = obs_paired['cancerType'].values
    activity_secact.file.close()

    # Aggregate
    agg_secact = activities_secact.groupby(['donorID', 'tissue', 'cell_type', 'cancerType']).mean().reset_index()

    # Calculate paired differences for SecAct
    cell_type_stats_secact = {}
    for (donor, ct), _ in paired_donor_ct.items():
        tumor_row = agg_secact[(agg_secact['donorID'] == donor) & (agg_secact['cell_type'] == ct) & (agg_secact['tissue'] == 'Tumor')]
        adjacent_row = agg_secact[(agg_secact['donorID'] == donor) & (agg_secact['cell_type'] == ct) & (agg_secact['tissue'] == 'Adjacent')]

        if len(tumor_row) == 1 and len(adjacent_row) == 1:
            for sig in top_secact_sigs:
                tumor_val = tumor_row[sig].values[0]
                adjacent_val = adjacent_row[sig].values[0]
                diff = tumor_val - adjacent_val

                key = (ct, sig)
                if key not in cell_type_stats_secact:
                    cell_type_stats_secact[key] = {'diffs': [], 'tumor': [], 'adjacent': []}
                cell_type_stats_secact[key]['diffs'].append(diff)
                cell_type_stats_secact[key]['tumor'].append(tumor_val)
                cell_type_stats_secact[key]['adjacent'].append(adjacent_val)

    for (ct, sig), data in cell_type_stats_secact.items():
        n_pairs = len(data['diffs'])
        if n_pairs < 3:
            continue

        mean_diff = np.mean(data['diffs'])
        std_diff = np.std(data['diffs'])
        mean_tumor = np.mean(data['tumor'])
        mean_adjacent = np.mean(data['adjacent'])

        try:
            t_stat, p_value = stats.ttest_rel(data['tumor'], data['adjacent'])
            if np.isnan(p_value):
                p_value = 1.0
        except:
            p_value = 1.0

        comparison_data.append({
            'cell_type': ct,
            'signature': sig,
            'mean_tumor': round(float(mean_tumor), 4),
            'mean_adjacent': round(float(mean_adjacent), 4),
            'mean_difference': round(float(mean_diff), 4),
            'std_difference': round(float(std_diff), 4),
            'n_pairs': n_pairs,
            'p_value': round(float(p_value), 6),
            'signature_type': 'SecAct'
        })

    # Get unique values
    cell_types = sorted(list(set(d['cell_type'] for d in comparison_data)))
    cytosig_sigs = sorted(list(set(d['signature'] for d in comparison_data if d['signature_type'] == 'CytoSig')))
    secact_sigs_out = sorted(list(set(d['signature'] for d in comparison_data if d['signature_type'] == 'SecAct')))

    comparison_output = {
        'data': comparison_data,
        'cell_types': cell_types,
        'cytosig_signatures': cytosig_sigs,
        'secact_signatures': secact_sigs_out,
        'n_paired_donors': len(paired_donors),
        'analysis_type': 'paired_singlecell'
    }

    with open(OUTPUT_DIR / "cancer_comparison.json", 'w') as f:
        json.dump(comparison_output, f)

    print(f"  Total comparison records: {len(comparison_data)}")
    print(f"  Cell types with paired data: {len(cell_types)}")
    print(f"  CytoSig records: {len([d for d in comparison_data if d['signature_type'] == 'CytoSig'])}")
    print(f"  SecAct records: {len([d for d in comparison_data if d['signature_type'] == 'SecAct'])}")

    return comparison_output


def preprocess_cima_celltype():
    """Preprocess CIMA cell type activity data (like Inflammation Atlas)."""
    print("Processing CIMA cell type activity...")

    try:
        import anndata as ad
    except ImportError:
        print("  Skipping - anndata not available")
        return None

    all_celltype_activity = []

    # Process both CytoSig and SecAct
    for sig_type in ['CytoSig', 'SecAct']:
        h5ad_file = CIMA_DIR / f"CIMA_{sig_type}_pseudobulk.h5ad"
        if not h5ad_file.exists():
            print(f"  Skipping {sig_type} - {h5ad_file} not found")
            continue

        # Load the pseudobulk data
        adata = ad.read_h5ad(h5ad_file)
        print(f"  {sig_type} shape: {adata.shape}")

        # Extract activity matrix (proteins x sample_celltype)
        activity_df = pd.DataFrame(
            adata.X,
            index=adata.obs_names,  # proteins/cytokines
            columns=adata.var_names  # sample_celltype
        )

        # Get cell type info
        celltype_info = adata.var[['sample', 'cell_type', 'n_cells']].copy()

        # For SecAct, limit to top 100 most variable signatures
        signatures = activity_df.index.tolist()
        if sig_type == 'SecAct' and len(signatures) > 100:
            sig_variance = activity_df.var(axis=1)
            top_signatures = sig_variance.nlargest(100).index.tolist()
            activity_df = activity_df.loc[top_signatures]
            signatures = top_signatures
            print(f"    Filtered to top 100 most variable SecAct signatures")

        # Aggregate by cell type (mean across samples)
        for ct in celltype_info['cell_type'].unique():
            ct_cols = celltype_info[celltype_info['cell_type'] == ct].index
            mean_activity = activity_df[ct_cols].mean(axis=1)
            n_samples = len(ct_cols)
            total_cells = celltype_info.loc[ct_cols, 'n_cells'].sum()

            for sig in mean_activity.index:
                all_celltype_activity.append({
                    'cell_type': ct,
                    'signature': sig,
                    'signature_type': sig_type,
                    'mean_activity': round(mean_activity[sig], 4),
                    'n_samples': n_samples,
                    'n_cells': int(total_cells)
                })

    # Save
    with open(OUTPUT_DIR / "cima_celltype.json", 'w') as f:
        json.dump(all_celltype_activity, f)

    cytosig_count = len([x for x in all_celltype_activity if x['signature_type'] == 'CytoSig'])
    secact_count = len([x for x in all_celltype_activity if x['signature_type'] == 'SecAct'])
    print(f"  CytoSig records: {cytosig_count}")
    print(f"  SecAct records: {secact_count}")
    print(f"  Total output records: {len(all_celltype_activity)}")

    return all_celltype_activity


def preprocess_inflammation_correlations():
    """Preprocess Inflammation Atlas age/BMI correlations (like CIMA)."""
    print("Processing Inflammation Atlas age/BMI correlations...")

    try:
        import anndata as ad
    except ImportError:
        print("  Skipping - anndata not available")
        return None

    # Load sample metadata
    SAMPLE_META_PATH = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv')
    if not SAMPLE_META_PATH.exists():
        print(f"  Skipping - metadata file not found: {SAMPLE_META_PATH}")
        return None

    sample_meta = pd.read_csv(SAMPLE_META_PATH)
    print(f"  Loaded metadata: {len(sample_meta)} samples")

    # Get age and BMI data (drop NaN)
    meta_age = sample_meta[['sampleID', 'age']].dropna()
    meta_bmi = sample_meta[['sampleID', 'BMI']].dropna()
    print(f"  Samples with age: {len(meta_age)}")
    print(f"  Samples with BMI: {len(meta_bmi)}")

    correlations = {'age': [], 'bmi': []}

    # Process both CytoSig and SecAct
    for sig_type in ['CytoSig', 'SecAct']:
        h5ad_file = INFLAM_DIR / f"main_{sig_type}_pseudobulk.h5ad"
        if not h5ad_file.exists():
            print(f"  Skipping {sig_type} - {h5ad_file} not found")
            continue

        adata = ad.read_h5ad(h5ad_file)
        print(f"  {sig_type} shape: {adata.shape}")

        # Extract activity matrix
        activity_df = pd.DataFrame(
            adata.X,
            index=adata.obs_names,
            columns=adata.var_names
        )

        # Get sample info
        sample_info = adata.var[['sample', 'cell_type', 'n_cells']].copy()

        # Aggregate by sample (mean across cell types, weighted by n_cells)
        sample_activity = {}
        for sample in sample_info['sample'].unique():
            sample_cols = sample_info[sample_info['sample'] == sample].index
            weights = sample_info.loc[sample_cols, 'n_cells'].values
            total_weight = weights.sum()
            if total_weight > 0:
                weighted_mean = (activity_df[sample_cols] * weights).sum(axis=1) / total_weight
                sample_activity[sample] = weighted_mean

        sample_activity_df = pd.DataFrame(sample_activity).T

        # For SecAct, limit to top 100 signatures
        signatures = sample_activity_df.columns.tolist()
        if sig_type == 'SecAct' and len(signatures) > 100:
            sig_variance = sample_activity_df.var(axis=0)
            top_signatures = sig_variance.nlargest(100).index.tolist()
            sample_activity_df = sample_activity_df[top_signatures]
            signatures = top_signatures

        # Compute correlations with age
        for sig in signatures:
            merged = meta_age.merge(
                sample_activity_df[[sig]].rename(columns={sig: 'activity'}),
                left_on='sampleID', right_index=True, how='inner'
            )
            if len(merged) >= 10:
                rho, pval = stats.spearmanr(merged['age'], merged['activity'])
                correlations['age'].append({
                    'protein': sig,
                    'feature': 'Age',
                    'rho': round(rho, 4),
                    'pvalue': round(pval, 6),
                    'n': len(merged),
                    'signature': sig_type
                })

        # Compute correlations with BMI
        for sig in signatures:
            merged = meta_bmi.merge(
                sample_activity_df[[sig]].rename(columns={sig: 'activity'}),
                left_on='sampleID', right_index=True, how='inner'
            )
            if len(merged) >= 10:
                rho, pval = stats.spearmanr(merged['BMI'], merged['activity'])
                correlations['bmi'].append({
                    'protein': sig,
                    'feature': 'BMI',
                    'rho': round(rho, 4),
                    'pvalue': round(pval, 6),
                    'n': len(merged),
                    'signature': sig_type
                })

    # Add q-values (FDR correction)
    from scipy.stats import false_discovery_control
    for key in ['age', 'bmi']:
        if len(correlations[key]) > 0:
            pvals = [x['pvalue'] for x in correlations[key]]
            try:
                qvals = false_discovery_control(pvals, method='bh')
                for i, rec in enumerate(correlations[key]):
                    rec['qvalue'] = round(qvals[i], 6)
            except:
                for rec in correlations[key]:
                    rec['qvalue'] = rec['pvalue']

    with open(OUTPUT_DIR / "inflammation_correlations.json", 'w') as f:
        json.dump(correlations, f)

    print(f"  Age correlations: {len(correlations['age'])}")
    print(f"  BMI correlations: {len(correlations['bmi'])}")

    return correlations


def preprocess_inflammation_disease():
    """Preprocess Inflammation Atlas disease-specific activity data."""
    print("Processing Inflammation Atlas disease activity...")

    try:
        import anndata as ad
    except ImportError:
        print("  Skipping - anndata not available")
        return None

    # Load sample metadata
    SAMPLE_META_PATH = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv')
    if not SAMPLE_META_PATH.exists():
        print(f"  Skipping - metadata file not found: {SAMPLE_META_PATH}")
        return None

    sample_meta = pd.read_csv(SAMPLE_META_PATH)
    print(f"  Loaded metadata: {len(sample_meta)} samples")
    print(f"  Diseases: {sample_meta['disease'].nunique()}")
    print(f"  Disease groups: {sample_meta['diseaseGroup'].nunique()}")

    disease_activity = []

    # Process CytoSig only (for performance)
    sig_type = 'CytoSig'
    h5ad_file = INFLAM_DIR / f"main_{sig_type}_pseudobulk.h5ad"
    if not h5ad_file.exists():
        print(f"  Skipping - {h5ad_file} not found")
        return None

    adata = ad.read_h5ad(h5ad_file)
    print(f"  {sig_type} shape: {adata.shape}")

    # Extract activity matrix
    activity_df = pd.DataFrame(
        adata.X,
        index=adata.obs_names,
        columns=adata.var_names
    )

    # Get sample and cell type info
    sample_info = adata.var[['sample', 'cell_type', 'n_cells']].copy()

    # Merge with disease info
    sample_info = sample_info.reset_index()
    sample_info = sample_info.merge(
        sample_meta[['sampleID', 'disease', 'diseaseGroup']],
        left_on='sample', right_on='sampleID', how='left'
    )
    sample_info = sample_info.set_index('index')

    # Aggregate by disease + cell type
    for disease in sample_info['disease'].dropna().unique():
        disease_mask = sample_info['disease'] == disease
        disease_group = sample_info.loc[disease_mask, 'diseaseGroup'].iloc[0] if disease_mask.any() else 'Unknown'

        for ct in sample_info.loc[disease_mask, 'cell_type'].unique():
            ct_mask = disease_mask & (sample_info['cell_type'] == ct)
            ct_cols = sample_info[ct_mask].index

            if len(ct_cols) < 3:  # Need at least 3 samples
                continue

            mean_activity = activity_df[ct_cols].mean(axis=1)
            n_samples = len(ct_cols)
            total_cells = sample_info.loc[ct_cols, 'n_cells'].sum()

            for sig in mean_activity.index:
                disease_activity.append({
                    'disease': disease,
                    'disease_group': disease_group,
                    'cell_type': ct,
                    'signature': sig,
                    'mean_activity': round(mean_activity[sig], 4),
                    'n_samples': n_samples,
                    'n_cells': int(total_cells)
                })

    with open(OUTPUT_DIR / "inflammation_disease.json", 'w') as f:
        json.dump(disease_activity, f)

    print(f"  Disease-celltype records: {len(disease_activity)}")

    return disease_activity


def preprocess_age_bmi_boxplots():
    """Preprocess age/BMI bin dependent activity data for boxplots using single-cell data."""
    print("Processing age/BMI bin boxplot data...")

    try:
        import anndata as ad
    except ImportError:
        print("  Skipping - anndata not available")
        return None

    boxplot_data = {'cima': {}, 'inflammation': {}}

    # === CIMA ===
    print("  Processing CIMA...")
    CIMA_META_PATH = Path('/data/Jiang_Lab/Data/Seongyong/CIMA/Metadata/CIMA_Sample_Information_Metadata.csv')
    if CIMA_META_PATH.exists():
        cima_meta = pd.read_csv(CIMA_META_PATH)
        cima_meta = cima_meta.rename(columns={'Sample_name': 'sample', 'Age': 'age'})

        # Create age bins
        cima_meta['age_bin'] = pd.cut(cima_meta['age'], bins=[0, 30, 40, 50, 60, 100],
                                       labels=['<30', '30-40', '40-50', '50-60', '>60'])
        # Create BMI bins
        cima_meta['bmi_bin'] = pd.cut(cima_meta['BMI'], bins=[0, 18.5, 25, 30, 100],
                                       labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

        # Load CIMA pseudobulk for sample-level activity
        cima_h5ad = CIMA_DIR / "CIMA_CytoSig_pseudobulk.h5ad"
        if cima_h5ad.exists():
            adata = ad.read_h5ad(cima_h5ad)
            activity_df = pd.DataFrame(
                adata.X,
                index=adata.obs_names,
                columns=adata.var_names
            )
            sample_info = adata.var[['sample', 'cell_type', 'n_cells']].copy()

            # Aggregate to sample level (weighted by n_cells)
            sample_activity = {}
            for sample in sample_info['sample'].unique():
                sample_cols = sample_info[sample_info['sample'] == sample].index
                weights = sample_info.loc[sample_cols, 'n_cells'].values
                total_weight = weights.sum()
                if total_weight > 0:
                    weighted_mean = (activity_df[sample_cols] * weights).sum(axis=1) / total_weight
                    sample_activity[sample] = weighted_mean

            sample_activity_df = pd.DataFrame(sample_activity).T

            # Merge with metadata
            sample_activity_df = sample_activity_df.reset_index().rename(columns={'index': 'sample'})
            sample_activity_df = sample_activity_df.merge(
                cima_meta[['sample', 'age_bin', 'bmi_bin']], on='sample', how='left'
            )

            # Get top 10 most variable signatures
            sig_cols = [c for c in sample_activity_df.columns if c not in ['sample', 'age_bin', 'bmi_bin']]
            sig_variance = sample_activity_df[sig_cols].var()
            top_sigs = sig_variance.nlargest(20).index.tolist()

            # Prepare boxplot data for age bins
            age_data = []
            for sig in top_sigs:
                for bin_val in ['<30', '30-40', '40-50', '50-60', '>60']:
                    bin_data = sample_activity_df[sample_activity_df['age_bin'] == bin_val][sig].dropna()
                    if len(bin_data) >= 3:
                        age_data.append({
                            'signature': sig,
                            'bin': bin_val,
                            'values': [round(v, 4) for v in bin_data.tolist()],
                            'mean': round(bin_data.mean(), 4),
                            'median': round(bin_data.median(), 4),
                            'n': len(bin_data)
                        })

            # Prepare boxplot data for BMI bins
            bmi_data = []
            for sig in top_sigs:
                for bin_val in ['Underweight', 'Normal', 'Overweight', 'Obese']:
                    bin_data = sample_activity_df[sample_activity_df['bmi_bin'] == bin_val][sig].dropna()
                    if len(bin_data) >= 3:
                        bmi_data.append({
                            'signature': sig,
                            'bin': bin_val,
                            'values': [round(v, 4) for v in bin_data.tolist()],
                            'mean': round(bin_data.mean(), 4),
                            'median': round(bin_data.median(), 4),
                            'n': len(bin_data)
                        })

            boxplot_data['cima'] = {
                'age': age_data,
                'bmi': bmi_data,
                'signatures': top_sigs
            }
            print(f"    CIMA age boxplots: {len(age_data)}")
            print(f"    CIMA BMI boxplots: {len(bmi_data)}")

    # === Inflammation Atlas ===
    print("  Processing Inflammation Atlas...")
    INFLAM_META_PATH = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv')
    if INFLAM_META_PATH.exists():
        inflam_meta = pd.read_csv(INFLAM_META_PATH)

        # Use existing binned_age or create bins
        if 'binned_age' in inflam_meta.columns:
            inflam_meta['age_bin'] = inflam_meta['binned_age']
        else:
            inflam_meta['age_bin'] = pd.cut(inflam_meta['age'], bins=[0, 30, 40, 50, 60, 70, 100],
                                             labels=['<30', '31-40', '41-50', '51-60', '61-70', '>70'])

        # Create BMI bins
        inflam_meta['bmi_bin'] = pd.cut(inflam_meta['BMI'], bins=[0, 18.5, 25, 30, 100],
                                         labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

        # Load pseudobulk
        inflam_h5ad = INFLAM_DIR / "main_CytoSig_pseudobulk.h5ad"
        if inflam_h5ad.exists():
            adata = ad.read_h5ad(inflam_h5ad)
            activity_df = pd.DataFrame(
                adata.X,
                index=adata.obs_names,
                columns=adata.var_names
            )
            sample_info = adata.var[['sample', 'cell_type', 'n_cells']].copy()

            # Aggregate to sample level
            sample_activity = {}
            for sample in sample_info['sample'].unique():
                sample_cols = sample_info[sample_info['sample'] == sample].index
                weights = sample_info.loc[sample_cols, 'n_cells'].values
                total_weight = weights.sum()
                if total_weight > 0:
                    weighted_mean = (activity_df[sample_cols] * weights).sum(axis=1) / total_weight
                    sample_activity[sample] = weighted_mean

            sample_activity_df = pd.DataFrame(sample_activity).T
            sample_activity_df = sample_activity_df.reset_index().rename(columns={'index': 'sample'})
            sample_activity_df = sample_activity_df.merge(
                inflam_meta[['sampleID', 'age_bin', 'bmi_bin']].rename(columns={'sampleID': 'sample'}),
                on='sample', how='left'
            )

            # Get top 20 most variable signatures
            sig_cols = [c for c in sample_activity_df.columns if c not in ['sample', 'age_bin', 'bmi_bin']]
            sig_variance = sample_activity_df[sig_cols].var()
            top_sigs = sig_variance.nlargest(20).index.tolist()

            # Get unique bins
            age_bins = [b for b in sample_activity_df['age_bin'].dropna().unique() if pd.notna(b)]
            bmi_bins = ['Underweight', 'Normal', 'Overweight', 'Obese']

            # Prepare age boxplot data
            age_data = []
            for sig in top_sigs:
                for bin_val in age_bins:
                    bin_data = sample_activity_df[sample_activity_df['age_bin'] == bin_val][sig].dropna()
                    if len(bin_data) >= 3:
                        age_data.append({
                            'signature': sig,
                            'bin': str(bin_val),
                            'values': [round(v, 4) for v in bin_data.tolist()],
                            'mean': round(bin_data.mean(), 4),
                            'median': round(bin_data.median(), 4),
                            'n': len(bin_data)
                        })

            # Prepare BMI boxplot data
            bmi_data = []
            for sig in top_sigs:
                for bin_val in bmi_bins:
                    bin_data = sample_activity_df[sample_activity_df['bmi_bin'] == bin_val][sig].dropna()
                    if len(bin_data) >= 3:
                        bmi_data.append({
                            'signature': sig,
                            'bin': bin_val,
                            'values': [round(v, 4) for v in bin_data.tolist()],
                            'mean': round(bin_data.mean(), 4),
                            'median': round(bin_data.median(), 4),
                            'n': len(bin_data)
                        })

            boxplot_data['inflammation'] = {
                'age': age_data,
                'bmi': bmi_data,
                'signatures': top_sigs,
                'age_bins': [str(b) for b in age_bins]
            }
            print(f"    Inflammation age boxplots: {len(age_data)}")
            print(f"    Inflammation BMI boxplots: {len(bmi_data)}")

    with open(OUTPUT_DIR / "age_bmi_boxplots.json", 'w') as f:
        json.dump(boxplot_data, f)

    return boxplot_data


def create_summary_stats():
    """Create summary statistics for overview section."""
    print("Creating summary statistics...")

    # Count data
    age_df = pd.read_csv(CIMA_DIR / "CIMA_correlation_age.csv")
    bmi_df = pd.read_csv(CIMA_DIR / "CIMA_correlation_bmi.csv")
    biochem_df = pd.read_csv(CIMA_DIR / "CIMA_correlation_biochemistry.csv")
    met_df = pd.read_csv(CIMA_DIR / "CIMA_correlation_metabolites.csv")
    diff_df = pd.read_csv(CIMA_DIR / "CIMA_differential_demographics.csv")
    organ_df = pd.read_csv(SCATLAS_DIR / "normal_organ_signatures.csv")
    ct_df = pd.read_csv(SCATLAS_DIR / "normal_celltype_signatures.csv")

    # Inflammation stats
    inflam_stats = {'n_samples': 0, 'n_cell_types': 0, 'n_cells': 0}
    try:
        import anndata as ad
        h5ad_file = INFLAM_DIR / "main_CytoSig_pseudobulk.h5ad"
        if h5ad_file.exists():
            adata = ad.read_h5ad(h5ad_file)
            inflam_stats = {
                'n_samples': adata.var['sample'].nunique(),
                'n_cell_types': adata.var['cell_type'].nunique(),
                'n_cells': int(adata.var['n_cells'].sum())
            }
    except:
        pass

    # Summary stats
    summary = {
        'cima': {
            'n_samples': int(age_df['n'].iloc[0]),
            'n_cytokines_cytosig': len(age_df[age_df['signature'] == 'CytoSig']['protein'].unique()),
            'n_proteins_secact': len(age_df[age_df['signature'] == 'SecAct']['protein'].unique()) if 'SecAct' in age_df['signature'].values else 0,
            'n_age_correlations': len(age_df),
            'n_bmi_correlations': len(bmi_df),
            'n_biochem_correlations': len(biochem_df),
            'n_metabolite_correlations': len(met_df),
            'n_differential': len(diff_df),
            'significant_age': len(age_df[age_df['qvalue'] < 0.05]),
            'significant_bmi': len(bmi_df[bmi_df['qvalue'] < 0.05]),
        },
        'scatlas': {
            'n_organs': len(organ_df['organ'].unique()),
            'n_cell_types': len(ct_df['cell_type'].unique()),
            'n_organ_signatures': len(organ_df),
            'n_celltype_signatures': len(ct_df),
            'organs': sorted(organ_df['organ'].unique().tolist()),
            'cytosig_signatures': sorted(ct_df[ct_df['signature_type'] == 'CytoSig']['signature'].unique().tolist()),
        },
        'inflammation': inflam_stats
    }

    with open(OUTPUT_DIR / "summary_stats.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  CIMA samples: {summary['cima']['n_samples']}")
    print(f"  scAtlas organs: {summary['scatlas']['n_organs']}")
    print(f"  scAtlas cell types: {summary['scatlas']['n_cell_types']}")
    print(f"  Inflammation samples: {summary['inflammation']['n_samples']}")

    return summary


def preprocess_disease_sankey():
    """Preprocess disease flow data for Sankey diagram."""
    print("Processing disease Sankey flow data...")

    try:
        import anndata as ad
    except ImportError:
        print("  Skipping - anndata not available")
        return None

    SAMPLE_META_PATH = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv')
    if not SAMPLE_META_PATH.exists():
        print(f"  Skipping - metadata not found")
        return None

    sample_meta = pd.read_csv(SAMPLE_META_PATH)

    # Count samples by cohort and disease
    cohort_disease = sample_meta.groupby(['cohort', 'disease']).size().reset_index(name='count')
    disease_group = sample_meta.groupby(['disease', 'diseaseGroup']).size().reset_index(name='count')

    # Build Sankey data
    nodes = []
    links = []

    # Cohorts
    cohorts = sample_meta['cohort'].unique().tolist()
    diseases = sample_meta['disease'].unique().tolist()
    disease_groups = sample_meta['diseaseGroup'].unique().tolist()

    # Add nodes
    for c in cohorts:
        nodes.append({'name': c, 'type': 'cohort'})
    for d in diseases:
        nodes.append({'name': d, 'type': 'disease'})
    for dg in disease_groups:
        nodes.append({'name': dg, 'type': 'disease_group'})

    # Add links: cohort -> disease
    for _, row in cohort_disease.iterrows():
        source_idx = cohorts.index(row['cohort'])
        target_idx = len(cohorts) + diseases.index(row['disease'])
        links.append({
            'source': source_idx,
            'target': target_idx,
            'value': int(row['count'])
        })

    # Add links: disease -> disease_group
    for _, row in disease_group.iterrows():
        source_idx = len(cohorts) + diseases.index(row['disease'])
        target_idx = len(cohorts) + len(diseases) + disease_groups.index(row['diseaseGroup'])
        links.append({
            'source': source_idx,
            'target': target_idx,
            'value': int(row['count'])
        })

    sankey_data = {
        'nodes': nodes,
        'links': links,
        'cohorts': cohorts,
        'diseases': diseases,
        'disease_groups': disease_groups
    }

    with open(OUTPUT_DIR / "disease_sankey.json", 'w') as f:
        json.dump(sankey_data, f)

    print(f"  Nodes: {len(nodes)}, Links: {len(links)}")
    return sankey_data


def preprocess_cross_atlas():
    """Preprocess cross-atlas integration summary data."""
    print("Processing cross-atlas integration data...")

    cross_atlas = {
        'summary': {
            'cima': {'cells': 6500000, 'samples': 421, 'cell_types': 27},
            'inflammation': {'cells': 4900000, 'samples': 817, 'cell_types': 66},
            'scatlas_normal': {'cells': 5200000, 'samples': 0, 'cell_types': 376, 'organs': 35},
            'scatlas_cancer': {'cells': 1200000, 'samples': 0, 'cell_types': 150}
        },
        'shared_cell_types': [],
        'atlas_specific_signatures': {
            'cima_only': 5,
            'inflammation_only': 8,
            'scatlas_only': 12,
            'cima_inflam': 10,
            'cima_scatlas': 7,
            'inflam_scatlas': 9,
            'all_three': 22
        }
    }

    # Find common cell types across atlases (simplified)
    common_types = [
        'CD8 T', 'CD4 T', 'NK', 'B cell', 'Monocyte', 'Macrophage',
        'DC', 'Plasma', 'Treg', 'Neutrophil'
    ]
    cross_atlas['shared_cell_types'] = common_types

    with open(OUTPUT_DIR / "cross_atlas.json", 'w') as f:
        json.dump(cross_atlas, f)

    print(f"  Cross-atlas summary created")
    return cross_atlas


def create_embedded_data():
    """Create embedded_data.js file for standalone HTML viewing."""
    print("\nCreating embedded_data.js...")

    # List of JSON files to embed
    json_files = [
        'cima_correlations.json',
        'cima_metabolites_top.json',
        'cima_differential.json',
        'cima_celltype.json',
        'scatlas_organs.json',
        'scatlas_organs_top.json',
        'scatlas_celltypes.json',
        'cancer_comparison.json',
        'inflammation_celltype.json',
        'inflammation_correlations.json',
        'inflammation_disease.json',
        'disease_sankey.json',
        'age_bmi_boxplots.json',
        'cross_atlas.json',
        'summary_stats.json'
    ]

    embedded = {}
    for json_file in json_files:
        filepath = OUTPUT_DIR / json_file
        if filepath.exists():
            with open(filepath) as f:
                key = json_file.replace('.json', '').replace('_', '')
                embedded[key] = json.load(f)
                size_kb = filepath.stat().st_size / 1024
                print(f"  Embedded {json_file}: {size_kb:.1f} KB")
        else:
            print(f"  Skipping {json_file} (not found)")

    # Write as JavaScript
    js_content = f"const EMBEDDED_DATA = {json.dumps(embedded)};\n"

    with open(OUTPUT_DIR / "embedded_data.js", 'w') as f:
        f.write(js_content)

    total_size = (OUTPUT_DIR / "embedded_data.js").stat().st_size / (1024 * 1024)
    print(f"\n  Total embedded_data.js: {total_size:.2f} MB")

    return embedded


def main():
    print("=" * 60)
    print("Preprocessing visualization data")
    print("=" * 60)

    # Process all data
    preprocess_cima_correlations()
    preprocess_cima_metabolites()
    preprocess_cima_differential()
    preprocess_cima_celltype()
    preprocess_scatlas_organs()
    preprocess_scatlas_celltypes()
    preprocess_cancer_comparison()
    preprocess_inflammation()
    preprocess_inflammation_correlations()
    preprocess_inflammation_disease()
    preprocess_age_bmi_boxplots()
    preprocess_disease_sankey()  # NEW: Disease Sankey
    preprocess_cross_atlas()  # NEW: Cross-atlas integration
    summary = create_summary_stats()

    # Create embedded data for standalone HTML
    create_embedded_data()

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    # List output files
    for f in sorted(OUTPUT_DIR.glob("*.json")):
        size = f.stat().st_size / 1024
        print(f"  {f.name}: {size:.1f} KB")


if __name__ == "__main__":
    main()
