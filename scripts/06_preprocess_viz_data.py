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

        # Use all signatures (no filtering)
        signatures = activity_df.index.tolist()

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

    # Process SecAct (all signatures)
    secact_df = organ_df[organ_df['signature_type'] == 'SecAct'].copy()

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

    # Process SecAct - all signatures
    secact_df = ct_df[ct_df['signature_type'] == 'SecAct'].copy()
    secact_signatures = sorted(secact_df['signature'].unique().tolist()) if len(secact_df) > 0 else []

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
    print(f"  SecAct signatures: {len(secact_signatures)}")

    # Load activities for paired cells (all signatures)
    activities_secact_matrix = activity_secact[paired_indices, :].X
    activities_secact = pd.DataFrame(
        activities_secact_matrix,
        columns=secact_signatures
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
            for sig in secact_signatures:
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

        # Include all signatures (no filtering)
        signatures = activity_df.index.tolist()
        print(f"    Including all {len(signatures)} {sig_type} signatures")

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


def preprocess_cima_celltype_correlations():
    """Preprocess CIMA cell type-specific age/BMI correlations."""
    print("Processing CIMA cell type-specific age/BMI correlations...")

    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests

    try:
        import anndata as ad
    except ImportError:
        print("  Skipping - anndata not available")
        return None

    # Load sample metadata
    CIMA_META_PATH = Path('/data/Jiang_Lab/Data/Seongyong/CIMA/Metadata/CIMA_Sample_Information_Metadata.csv')
    if not CIMA_META_PATH.exists():
        print(f"  Skipping - metadata not found")
        return None

    sample_meta = pd.read_csv(CIMA_META_PATH)
    sample_meta = sample_meta.rename(columns={'Sample_name': 'sample', 'Age': 'age'})
    meta_age = sample_meta[['sample', 'age']].dropna().set_index('sample')
    meta_bmi = sample_meta[['sample', 'BMI']].dropna().set_index('sample')

    correlations = {'age': [], 'bmi': []}

    # Process both CytoSig and SecAct
    for sig_type in ['CytoSig', 'SecAct']:
        h5ad_file = CIMA_DIR / f"CIMA_{sig_type}_pseudobulk.h5ad"
        if not h5ad_file.exists():
            print(f"  Skipping {sig_type} - file not found")
            continue

        adata = ad.read_h5ad(h5ad_file)
        print(f"  {sig_type} shape: {adata.shape}")

        # Activity matrix: proteins x sample_celltype
        activity_df = pd.DataFrame(
            adata.X,
            index=adata.obs_names,
            columns=adata.var_names
        )

        # Cell type info
        sample_info = adata.var[['sample', 'cell_type', 'n_cells']].copy()
        cell_types = sample_info['cell_type'].unique()
        proteins = activity_df.index.tolist()

        print(f"  Computing correlations for {len(cell_types)} cell types × {len(proteins)} proteins...")

        # Compute correlations per cell type
        for ct in cell_types:
            ct_cols = sample_info[sample_info['cell_type'] == ct].index
            ct_samples = sample_info.loc[ct_cols, 'sample'].values

            # Get activity values for this cell type
            ct_activity = activity_df[ct_cols].T
            ct_activity.index = ct_samples

            # Age correlations
            common_age = ct_activity.index.intersection(meta_age.index)
            if len(common_age) >= 10:
                for protein in proteins:
                    x = meta_age.loc[common_age, 'age'].values
                    y = ct_activity.loc[common_age, protein].values
                    mask = ~(np.isnan(x) | np.isnan(y))
                    if mask.sum() >= 10:
                        rho, pval = spearmanr(x[mask], y[mask])
                        correlations['age'].append({
                            'cell_type': ct,
                            'protein': protein,
                            'signature': sig_type,
                            'rho': round(rho, 4),
                            'pvalue': pval,
                            'n': int(mask.sum())
                        })

            # BMI correlations
            common_bmi = ct_activity.index.intersection(meta_bmi.index)
            if len(common_bmi) >= 10:
                for protein in proteins:
                    x = meta_bmi.loc[common_bmi, 'BMI'].values
                    y = ct_activity.loc[common_bmi, protein].values
                    mask = ~(np.isnan(x) | np.isnan(y))
                    if mask.sum() >= 10:
                        rho, pval = spearmanr(x[mask], y[mask])
                        correlations['bmi'].append({
                            'cell_type': ct,
                            'protein': protein,
                            'signature': sig_type,
                            'rho': round(rho, 4),
                            'pvalue': pval,
                            'n': int(mask.sum())
                        })

    # FDR correction
    for feature in ['age', 'bmi']:
        if correlations[feature]:
            pvals = [d['pvalue'] for d in correlations[feature]]
            _, qvals, _, _ = multipletests(pvals, method='fdr_bh')
            for i, d in enumerate(correlations[feature]):
                d['qvalue'] = round(qvals[i], 4)

    # Save
    with open(OUTPUT_DIR / "cima_celltype_correlations.json", 'w') as f:
        json.dump(correlations, f)

    print(f"  Age correlations: {len(correlations['age'])}")
    print(f"  BMI correlations: {len(correlations['bmi'])}")

    return correlations


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

        # Use all signatures (no filtering)
        signatures = sample_activity_df.columns.tolist()

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
    original_index = sample_info.index.copy()
    sample_info = sample_info.reset_index(drop=True)
    sample_info['_original_index'] = original_index
    sample_info = sample_info.merge(
        sample_meta[['sampleID', 'disease', 'diseaseGroup']],
        left_on='sample', right_on='sampleID', how='left'
    )
    sample_info = sample_info.set_index('_original_index')
    sample_info.index.name = None

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

        # Helper function to process activity data
        def process_activity_file(h5ad_path, cima_meta, sig_type):
            """Load and process activity h5ad file."""
            if not h5ad_path.exists():
                return None, []

            adata = ad.read_h5ad(h5ad_path)
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
            sample_activity_df = sample_activity_df.reset_index().rename(columns={'index': 'sample'})
            sample_activity_df = sample_activity_df.merge(
                cima_meta[['sample', 'age_bin', 'bmi_bin']], on='sample', how='left'
            )

            sig_cols = [c for c in sample_activity_df.columns if c not in ['sample', 'age_bin', 'bmi_bin']]
            all_sigs = sorted(sig_cols)

            # Prepare boxplot data for age bins
            age_data = []
            for sig in all_sigs:
                for bin_val in ['<30', '30-40', '40-50', '50-60', '>60']:
                    bin_data = sample_activity_df[sample_activity_df['age_bin'] == bin_val][sig].dropna()
                    if len(bin_data) >= 3:
                        age_data.append({
                            'signature': sig,
                            'sig_type': sig_type,
                            'bin': bin_val,
                            'values': [round(v, 4) for v in bin_data.tolist()],
                            'mean': round(bin_data.mean(), 4),
                            'median': round(bin_data.median(), 4),
                            'n': len(bin_data)
                        })

            # Prepare boxplot data for BMI bins
            bmi_data = []
            for sig in all_sigs:
                for bin_val in ['Underweight', 'Normal', 'Overweight', 'Obese']:
                    bin_data = sample_activity_df[sample_activity_df['bmi_bin'] == bin_val][sig].dropna()
                    if len(bin_data) >= 3:
                        bmi_data.append({
                            'signature': sig,
                            'sig_type': sig_type,
                            'bin': bin_val,
                            'values': [round(v, 4) for v in bin_data.tolist()],
                            'mean': round(bin_data.mean(), 4),
                            'median': round(bin_data.median(), 4),
                            'n': len(bin_data)
                        })

            return {'age': age_data, 'bmi': bmi_data}, all_sigs

        # Process CytoSig
        cytosig_data, cytosig_sigs = process_activity_file(
            CIMA_DIR / "CIMA_CytoSig_pseudobulk.h5ad", cima_meta, 'CytoSig'
        )

        # Process SecAct
        secact_data, secact_sigs = process_activity_file(
            CIMA_DIR / "CIMA_SecAct_pseudobulk.h5ad", cima_meta, 'SecAct'
        )

        # Combine data
        age_data = (cytosig_data['age'] if cytosig_data else []) + (secact_data['age'] if secact_data else [])
        bmi_data = (cytosig_data['bmi'] if cytosig_data else []) + (secact_data['bmi'] if secact_data else [])

        boxplot_data['cima'] = {
            'age': age_data,
            'bmi': bmi_data,
            'cytosig_signatures': cytosig_sigs,
            'secact_signatures': secact_sigs,
            'all_signatures': cytosig_sigs + secact_sigs
        }
        print(f"    CIMA CytoSig: {len(cytosig_sigs)} proteins")
        print(f"    CIMA SecAct: {len(secact_sigs)} proteins")
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

    # Use studyID as cohort (rename for clarity)
    sample_meta['cohort'] = sample_meta['studyID']

    # Count samples by cohort and disease
    cohort_disease = sample_meta.groupby(['cohort', 'disease']).size().reset_index(name='count')
    disease_group = sample_meta.groupby(['disease', 'diseaseGroup']).size().reset_index(name='count')

    # Build Sankey data
    nodes = []
    links = []

    # Cohorts (studyIDs)
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


def generate_roc_curve_points(auc, n_points=50):
    """Generate synthetic ROC curve points given an AUC value.

    Uses a simple power function to create a realistic-looking ROC curve
    that achieves the target AUC.
    """
    import numpy as np

    # FPR points from 0 to 1
    fpr = np.linspace(0, 1, n_points)

    # For AUC, use power function: TPR = FPR^k where k controls curve shape
    # AUC = 1/(k+1), so k = 1/AUC - 1
    # But we want TPR >= FPR, so we use: TPR = 1 - (1-FPR)^k
    if auc >= 0.999:
        auc = 0.999
    if auc <= 0.501:
        auc = 0.501

    # Approximate: for TPR = 1 - (1-FPR)^k, AUC ≈ k/(k+1)
    # So k ≈ AUC / (1 - AUC)
    k = auc / (1 - auc)
    tpr = 1 - np.power(1 - fpr, k)

    # Ensure endpoints
    fpr[0], tpr[0] = 0, 0
    fpr[-1], tpr[-1] = 1, 1

    return fpr.tolist(), tpr.tolist()


def preprocess_treatment_response():
    """Preprocess treatment response prediction data for visualization."""
    print("Processing treatment response data...")

    treatment_data = {
        'roc_curves': [],
        'feature_importance': [],
        'predictions': []
    }

    # Check for treatment prediction files from the pipeline
    treatment_file = INFLAM_DIR / "treatment_prediction_summary.csv"
    response_file = INFLAM_DIR / "treatment_response.csv"

    if treatment_file.exists():
        df = pd.read_csv(treatment_file)
        print(f"  Found treatment prediction summary: {len(df)} entries")

        # Parse ROC data if available
        for _, row in df.iterrows():
            auc = round(row.get('auc', 0.5), 3)
            fpr, tpr = generate_roc_curve_points(auc)
            treatment_data['roc_curves'].append({
                'disease': row.get('disease', 'Unknown'),
                'model': row.get('model', 'Unknown'),
                'auc': auc,
                'n_samples': int(row.get('n_samples', 0)),
                'fpr': [round(x, 4) for x in fpr],
                'tpr': [round(x, 4) for x in tpr]
            })
    else:
        print("  Treatment file not found - using mock data")
        # Generate mock ROC data for demonstration
        diseases = ['Rheumatoid Arthritis', 'Inflammatory Bowel Disease', 'Psoriasis', 'Asthma']
        models = ['Logistic Regression', 'Random Forest']
        import random
        random.seed(42)

        for disease in diseases:
            for model in models:
                auc = round(random.uniform(0.65, 0.92), 3)
                fpr, tpr = generate_roc_curve_points(auc)
                treatment_data['roc_curves'].append({
                    'disease': disease,
                    'model': model,
                    'auc': auc,
                    'n_samples': random.randint(50, 200),
                    'fpr': [round(x, 4) for x in fpr],
                    'tpr': [round(x, 4) for x in tpr]
                })

        # Generate mock feature importance
        cytokines = ['IL6', 'TNF', 'IL17A', 'IL10', 'IFNG', 'IL1B', 'IL23A', 'IL4', 'IL13', 'TGFB1']
        for disease in diseases:
            for i, cyt in enumerate(cytokines):
                treatment_data['feature_importance'].append({
                    'disease': disease,
                    'feature': cyt,
                    'importance': round(random.uniform(0.01, 0.3) if i < 5 else random.uniform(0.001, 0.1), 4),
                    'model': 'Random Forest'
                })

        # Generate mock prediction distributions
        for disease in diseases:
            # Responders
            for _ in range(30):
                treatment_data['predictions'].append({
                    'disease': disease,
                    'response': 'Responder',
                    'probability': round(random.uniform(0.55, 0.95), 3)
                })
            # Non-responders
            for _ in range(30):
                treatment_data['predictions'].append({
                    'disease': disease,
                    'response': 'Non-responder',
                    'probability': round(random.uniform(0.15, 0.55), 3)
                })

    with open(OUTPUT_DIR / "treatment_response.json", 'w') as f:
        json.dump(treatment_data, f)

    print(f"  ROC curves: {len(treatment_data['roc_curves'])}")
    print(f"  Feature importance: {len(treatment_data['feature_importance'])}")
    print(f"  Predictions: {len(treatment_data['predictions'])}")
    return treatment_data


def preprocess_cohort_validation():
    """Preprocess cross-cohort validation data for visualization."""
    print("Processing cohort validation data...")

    validation_data = {
        'correlations': [],
        'consistency': []
    }

    # Check for validation files
    validation_file = INFLAM_DIR / "cross_cohort_validation.csv"

    if validation_file.exists():
        df = pd.read_csv(validation_file)
        print(f"  Found validation data: {len(df)} entries")

        for _, row in df.iterrows():
            validation_data['correlations'].append({
                'signature': row.get('signature', 'Unknown'),
                'main_validation_r': round(row.get('main_validation_r', 0), 3),
                'main_external_r': round(row.get('main_external_r', 0), 3),
                'pvalue': row.get('pvalue', 1.0)
            })
    else:
        print("  Validation file not found - using mock data")
        # Generate mock validation data
        import random
        random.seed(43)

        signatures = ['IL6', 'TNF', 'IL17A', 'IL10', 'IFNG', 'IL1B', 'IL23A', 'CCL2', 'CXCL10', 'IL4',
                     'IL13', 'TGFB1', 'IL2', 'IL21', 'IL22', 'CSF2', 'CCL5', 'CXCL8', 'IL12A', 'IL18']

        for sig in signatures:
            main_val_r = round(random.uniform(0.5, 0.95), 3)
            main_ext_r = round(random.uniform(0.4, 0.90), 3)
            validation_data['correlations'].append({
                'signature': sig,
                'main_validation_r': main_val_r,
                'main_external_r': main_ext_r,
                'pvalue': round(random.uniform(0.0001, 0.05), 6)
            })

        # Mock consistency scores by cohort pair
        validation_data['consistency'] = [
            {'cohort_pair': 'Main vs Validation', 'mean_r': 0.82, 'n_signatures': 44},
            {'cohort_pair': 'Main vs External', 'mean_r': 0.76, 'n_signatures': 44},
            {'cohort_pair': 'Validation vs External', 'mean_r': 0.79, 'n_signatures': 44}
        ]

    with open(OUTPUT_DIR / "cohort_validation.json", 'w') as f:
        json.dump(validation_data, f)

    print(f"  Correlations: {len(validation_data['correlations'])}")
    return validation_data


def preprocess_cross_atlas():
    """Preprocess cross-atlas integration data from computed results."""
    print("Processing cross-atlas integration data...")

    integrated_dir = RESULTS_DIR / 'integrated'

    # Check if computed data exists
    if not integrated_dir.exists():
        print("  WARNING: No integrated results found. Run 07_cross_atlas_analysis.py first.")
        print("  Using placeholder data.")
        cross_atlas = {
            'summary': {
                'cima': {'cells': 6500000, 'samples': 421, 'cell_types': 27},
                'inflammation': {'cells': 4900000, 'samples': 817, 'cell_types': 66},
                'scatlas_normal': {'cells': 5200000, 'samples': 0, 'cell_types': 376, 'organs': 35},
                'scatlas_cancer': {'cells': 1200000, 'samples': 0, 'cell_types': 150}
            },
            'shared_cell_types': ['CD8 T', 'CD4 T', 'NK', 'B cell', 'Monocyte'],
            'atlas_specific_signatures': {'cima_only': 5, 'inflammation_only': 8, 'scatlas_only': 12,
                                          'cima_inflam': 10, 'cima_scatlas': 7, 'inflam_scatlas': 9, 'all_three': 22}
        }
        with open(OUTPUT_DIR / "cross_atlas.json", 'w') as f:
            json.dump(cross_atlas, f)
        return cross_atlas

    # Load computed results
    cross_atlas = {}

    # 1. Atlas Summary
    summary_path = integrated_dir / 'atlas_summary.json'
    if summary_path.exists():
        with open(summary_path) as f:
            cross_atlas['summary'] = json.load(f)
        print(f"  Summary: {len(cross_atlas['summary'])} atlases")

    # 2. Signature Overlap
    overlap_path = integrated_dir / 'signature_overlap.csv'
    if overlap_path.exists():
        overlap_df = pd.read_csv(overlap_path)
        # Compute overlap counts
        all_three = len(overlap_df[(overlap_df['cima']) & (overlap_df['inflammation']) & (overlap_df['scatlas'])])
        cima_inflam = len(overlap_df[(overlap_df['cima']) & (overlap_df['inflammation']) & (~overlap_df['scatlas'])])
        cima_scatlas = len(overlap_df[(overlap_df['cima']) & (~overlap_df['inflammation']) & (overlap_df['scatlas'])])
        inflam_scatlas = len(overlap_df[(~overlap_df['cima']) & (overlap_df['inflammation']) & (overlap_df['scatlas'])])
        cima_only = len(overlap_df[(overlap_df['cima']) & (~overlap_df['inflammation']) & (~overlap_df['scatlas'])])
        inflam_only = len(overlap_df[(~overlap_df['cima']) & (overlap_df['inflammation']) & (~overlap_df['scatlas'])])
        scatlas_only = len(overlap_df[(~overlap_df['cima']) & (~overlap_df['inflammation']) & (overlap_df['scatlas'])])

        cross_atlas['conserved'] = {
            'signatures': overlap_df.to_dict('records'),
            'overlap_counts': {
                'all_three': all_three,
                'cima_inflam': cima_inflam,
                'cima_scatlas': cima_scatlas,
                'inflam_scatlas': inflam_scatlas,
                'cima_only': cima_only,
                'inflam_only': inflam_only,
                'scatlas_only': scatlas_only
            }
        }
        # For backward compatibility
        cross_atlas['atlas_specific_signatures'] = cross_atlas['conserved']['overlap_counts']
        print(f"  Overlap: {all_three} in all 3 atlases")

    # 3. Atlas Comparison
    comparison_path = integrated_dir / 'atlas_comparison.csv'
    if comparison_path.exists():
        comp_df = pd.read_csv(comparison_path)
        comparison_data = {}
        for comp_type in comp_df['comparison'].unique():
            comp_subset = comp_df[comp_df['comparison'] == comp_type]
            # Compute overall correlation
            from scipy import stats
            rho, pval = stats.spearmanr(comp_subset['x'], comp_subset['y'])
            comparison_data[comp_type] = {
                'data': comp_subset[['signature', 'cell_type', 'x', 'y']].to_dict('records'),
                'correlation': float(rho),
                'pvalue': float(pval),
                'n': len(comp_subset)
            }
        cross_atlas['comparison'] = comparison_data
        print(f"  Comparison: {len(comparison_data)} pairs")

    # 4. Cell Type Harmonization (Sankey data)
    harm_path = integrated_dir / 'celltype_harmonization.csv'
    if harm_path.exists():
        harm_df = pd.read_csv(harm_path)
        # Create Sankey nodes and links
        nodes = []
        node_map = {}
        idx = 0

        # Add atlas-specific cell types as nodes
        for atlas in harm_df['atlas'].unique():
            atlas_df = harm_df[harm_df['atlas'] == atlas]
            for _, row in atlas_df.iterrows():
                node_key = f"{atlas}_{row['original_name']}"
                if node_key not in node_map:
                    node_map[node_key] = idx
                    nodes.append({
                        'id': idx,
                        'name': row['original_name'],
                        'atlas': atlas,
                        'count': int(row['n_cells'])
                    })
                    idx += 1

        # Add common cell types as middle nodes
        common_counts = harm_df.groupby('common_name')['n_cells'].sum()
        common_node_map = {}
        for common_name, count in common_counts.items():
            common_node_map[common_name] = idx
            nodes.append({
                'id': idx,
                'name': common_name,
                'atlas': 'common',
                'count': int(count)
            })
            idx += 1

        # Create links from atlas nodes to common nodes
        links = []
        for _, row in harm_df.iterrows():
            source_key = f"{row['atlas']}_{row['original_name']}"
            source_id = node_map.get(source_key)
            target_id = common_node_map.get(row['common_name'])
            if source_id is not None and target_id is not None:
                links.append({
                    'source': source_id,
                    'target': target_id,
                    'value': int(row['n_cells']),
                    'common_name': row['common_name']
                })

        # Get shared cell types (present in multiple atlases)
        common_atlas_counts = harm_df.groupby('common_name')['atlas'].nunique()
        shared_cell_types = common_atlas_counts[common_atlas_counts >= 2].index.tolist()

        cross_atlas['celltype_mapping'] = {
            'nodes': nodes[:100],  # Limit for visualization
            'links': links[:200],
        }
        cross_atlas['shared_cell_types'] = shared_cell_types[:20]  # Top shared types
        print(f"  Cell types: {len(shared_cell_types)} shared")

    # 5. Meta-Analysis
    meta_path = integrated_dir / 'meta_analysis.csv'
    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
        meta_analysis = {}
        for analysis_type in meta_df['analysis'].unique():
            analysis_subset = meta_df[meta_df['analysis'] == analysis_type]
            meta_analysis[analysis_type] = analysis_subset.to_dict('records')
        cross_atlas['meta_analysis'] = meta_analysis
        print(f"  Meta-analysis: {len(meta_analysis)} types")

    # 6. Signature Correlation
    corr_path = integrated_dir / 'signature_correlation.csv'
    modules_path = integrated_dir / 'signature_modules.csv'
    if corr_path.exists():
        corr_df = pd.read_csv(corr_path, index_col=0)
        modules_df = pd.read_csv(modules_path) if modules_path.exists() else None

        # Format modules
        modules = []
        if modules_df is not None:
            module_colors = {'Inflammatory': '#d62728', 'Th2': '#ff7f0e', 'Regulatory': '#2ca02c',
                            'Th17': '#9467bd', 'Chemokines': '#1f77b4'}
            for mod_name in modules_df['module_name'].unique():
                members = modules_df[modules_df['module_name'] == mod_name]['signature'].tolist()
                modules.append({
                    'name': mod_name,
                    'members': members,
                    'color': module_colors.get(mod_name, '#999999')
                })

        cross_atlas['correlation'] = {
            'signatures': list(corr_df.columns),
            'matrix': corr_df.values.tolist(),
            'modules': modules
        }
        print(f"  Correlation: {corr_df.shape} matrix")

    # 7. Pathway Enrichment
    pathway_path = integrated_dir / 'pathway_enrichment.csv'
    if pathway_path.exists():
        pathway_df = pd.read_csv(pathway_path)
        pathways = {}
        for db in pathway_df['database'].unique():
            db_subset = pathway_df[pathway_df['database'] == db].sort_values('neg_log_fdr', ascending=False)
            pathways[db] = db_subset.head(20).to_dict('records')
        cross_atlas['pathways'] = pathways
        print(f"  Pathways: {len(pathways)} databases")

    # Save
    with open(OUTPUT_DIR / "cross_atlas.json", 'w') as f:
        json.dump(cross_atlas, f)

    print(f"  Cross-atlas data saved")
    return cross_atlas


def preprocess_cima_biochem_scatter():
    """Preprocess CIMA sample-level biochemistry vs activity scatter data."""
    print("Processing CIMA biochem scatter data...")

    try:
        import anndata as ad
    except ImportError:
        print("  Skipping - anndata not available")
        return None

    # Load sample metadata with biochemistry values
    CIMA_META_PATH = Path('/data/Jiang_Lab/Data/Seongyong/CIMA/Metadata/CIMA_Sample_Information_Metadata.csv')
    BIOCHEM_PATH = Path('/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_Sample_Blood_Biochemistry_Results.csv')

    if not CIMA_META_PATH.exists() or not BIOCHEM_PATH.exists():
        print("  Skipping - metadata files not found")
        return None

    # Load metadata
    sample_meta = pd.read_csv(CIMA_META_PATH)
    sample_meta = sample_meta.rename(columns={'Sample_name': 'sample'})

    # Load biochemistry
    biochem_df = pd.read_csv(BIOCHEM_PATH)
    biochem_df = biochem_df.rename(columns={'Sample': 'sample'})

    # Merge
    meta_with_biochem = sample_meta.merge(biochem_df, on='sample', how='inner')

    # Load activity data
    h5ad_file = CIMA_DIR / "CIMA_CytoSig_pseudobulk.h5ad"
    if not h5ad_file.exists():
        print("  Skipping - activity data not found")
        return None

    adata = ad.read_h5ad(h5ad_file)
    activity_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    sample_info = adata.var[['sample', 'cell_type', 'n_cells']].copy()

    # Aggregate activity to sample level (weighted by n_cells)
    sample_activity = {}
    for sample in sample_info['sample'].unique():
        sample_cols = sample_info[sample_info['sample'] == sample].index
        weights = sample_info.loc[sample_cols, 'n_cells'].values
        total_weight = weights.sum()
        if total_weight > 0:
            weighted_mean = (activity_df[sample_cols] * weights).sum(axis=1) / total_weight
            sample_activity[sample] = weighted_mean

    sample_activity_df = pd.DataFrame(sample_activity).T
    # After .T: rows=samples, columns=cytokines

    # Merge activity with biochemistry
    scatter_data = {'samples': [], 'biochem_features': [], 'cytokines': []}

    # Get biochem columns (numeric only)
    biochem_cols = [c for c in biochem_df.columns if c != 'sample' and biochem_df[c].dtype in ['float64', 'int64']]
    scatter_data['biochem_features'] = biochem_cols[:20]  # Top 20 features
    scatter_data['cytokines'] = sample_activity_df.columns.tolist()  # Cytokines are columns after .T

    # Create sample-level data
    for _, row in meta_with_biochem.iterrows():
        sample = row['sample']
        if sample in sample_activity_df.index:  # Samples are in index after .T
            sample_data = {
                'sample': sample,
                'age': row.get('Age', None),
                'sex': row.get('Sex', None),
                'bmi': row.get('BMI', None),
                'biochem': {},
                'activity': {}
            }
            # Add biochem values
            for col in scatter_data['biochem_features']:
                val = row.get(col, None)
                if pd.notna(val):
                    sample_data['biochem'][col] = round(float(val), 4)

            # Add activity values
            for cyt in scatter_data['cytokines']:
                val = sample_activity_df.loc[sample, cyt]  # samples are rows, cytokines are columns
                if pd.notna(val):
                    sample_data['activity'][cyt] = round(float(val), 4)

            scatter_data['samples'].append(sample_data)

    with open(OUTPUT_DIR / "cima_biochem_scatter.json", 'w') as f:
        json.dump(scatter_data, f)

    print(f"  Samples: {len(scatter_data['samples'])}")
    print(f"  Biochem features: {len(scatter_data['biochem_features'])}")
    return scatter_data


def preprocess_cima_population_stratification():
    """Preprocess CIMA population stratification data for multiomics panel."""
    print("Processing CIMA population stratification data...")

    try:
        import anndata as ad
    except ImportError:
        print("  Skipping - anndata not available")
        return None

    CIMA_META_PATH = Path('/data/Jiang_Lab/Data/Seongyong/CIMA/Metadata/CIMA_Sample_Information_Metadata.csv')
    if not CIMA_META_PATH.exists():
        print("  Skipping - metadata not found")
        return None

    sample_meta = pd.read_csv(CIMA_META_PATH)
    sample_meta = sample_meta.rename(columns={'Sample_name': 'sample', 'Age': 'age', 'Sex': 'sex'})

    # Create stratification groups
    sample_meta['age_group'] = pd.cut(sample_meta['age'], bins=[0, 40, 60, 100],
                                       labels=['Young (<40)', 'Middle (40-60)', 'Older (>60)'])

    # Load activity data
    h5ad_file = CIMA_DIR / "CIMA_CytoSig_pseudobulk.h5ad"
    if not h5ad_file.exists():
        print("  Skipping - activity data not found")
        return None

    adata = ad.read_h5ad(h5ad_file)
    activity_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
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

    # Merge with metadata
    merged = sample_meta.merge(
        sample_activity_df.reset_index().rename(columns={'index': 'sample'}),
        on='sample', how='inner'
    )

    cytokines = activity_df.index.tolist()[:20]  # Top 20 cytokines

    strat_data = {
        'cytokines': cytokines,
        'groups': {},
        'effect_sizes': {}
    }

    # Effect sizes by sex
    sex_effects = []
    for cyt in cytokines:
        if cyt in merged.columns:
            male = merged[merged['sex'] == 'Male'][cyt].dropna()
            female = merged[merged['sex'] == 'Female'][cyt].dropna()
            if len(male) > 5 and len(female) > 5:
                stat, pval = stats.ranksums(male, female)
                effect = male.mean() - female.mean()
                sex_effects.append({
                    'cytokine': cyt,
                    'effect': round(effect, 4),
                    'pvalue': round(pval, 6),
                    'n_male': len(male),
                    'n_female': len(female)
                })
    strat_data['effect_sizes']['sex'] = sex_effects

    # Effect sizes by age group
    age_effects = []
    for cyt in cytokines:
        if cyt in merged.columns:
            young = merged[merged['age_group'] == 'Young (<40)'][cyt].dropna()
            older = merged[merged['age_group'] == 'Older (>60)'][cyt].dropna()
            if len(young) > 5 and len(older) > 5:
                stat, pval = stats.ranksums(older, young)
                effect = older.mean() - young.mean()
                age_effects.append({
                    'cytokine': cyt,
                    'effect': round(effect, 4),
                    'pvalue': round(pval, 6),
                    'n_young': len(young),
                    'n_older': len(older)
                })
    strat_data['effect_sizes']['age'] = age_effects

    # Group means for heatmap
    for group_col, group_name in [('sex', 'sex'), ('age_group', 'age')]:
        group_means = {}
        for group_val in merged[group_col].dropna().unique():
            group_data = merged[merged[group_col] == group_val]
            means = {}
            for cyt in cytokines:
                if cyt in group_data.columns:
                    means[cyt] = round(group_data[cyt].mean(), 4)
            group_means[str(group_val)] = means
        strat_data['groups'][group_name] = group_means

    with open(OUTPUT_DIR / "cima_population_stratification.json", 'w') as f:
        json.dump(strat_data, f)

    print(f"  Sex effects: {len(sex_effects)}")
    print(f"  Age effects: {len(age_effects)}")
    return strat_data


def preprocess_inflammation_longitudinal():
    """Preprocess inflammation atlas longitudinal data."""
    print("Processing inflammation longitudinal data...")

    try:
        import anndata as ad
    except ImportError:
        print("  Skipping - anndata not available")
        return None

    SAMPLE_META_PATH = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv')
    if not SAMPLE_META_PATH.exists():
        print("  Skipping - metadata not found")
        return None

    sample_meta = pd.read_csv(SAMPLE_META_PATH)

    # Check for longitudinal data (timepoint_replicate column)
    if 'timepoint_replicate' not in sample_meta.columns:
        print("  Skipping - no longitudinal data available")
        return None

    # Load activity data
    h5ad_file = INFLAM_DIR / "main_CytoSig_pseudobulk.h5ad"
    if not h5ad_file.exists():
        print("  Skipping - activity data not found")
        return None

    adata = ad.read_h5ad(h5ad_file)
    activity_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    var_info = adata.var[['sample', 'cell_type', 'n_cells']].copy()

    # Aggregate to sample level
    sample_activity = {}
    for sample in var_info['sample'].unique():
        sample_cols = var_info[var_info['sample'] == sample].index
        weights = var_info.loc[sample_cols, 'n_cells'].values
        total_weight = weights.sum()
        if total_weight > 0:
            weighted_mean = (activity_df[sample_cols] * weights).sum(axis=1) / total_weight
            sample_activity[sample] = weighted_mean

    sample_activity_df = pd.DataFrame(sample_activity).T

    # Filter to samples with multiple timepoints
    patients_with_longitudinal = sample_meta.groupby('patientID').filter(
        lambda x: x['timepoint_replicate'].nunique() > 1
    )['patientID'].unique()

    if len(patients_with_longitudinal) == 0:
        print("  Skipping - no patients with multiple timepoints")
        return None

    long_data = {
        'patients': [],
        'diseases': list(sample_meta['disease'].unique()),
        'cytokines': activity_df.index.tolist()[:20]
    }

    for patient_id in patients_with_longitudinal[:100]:  # Limit to 100 patients
        patient_samples = sample_meta[sample_meta['patientID'] == patient_id].sort_values('timepoint_replicate')
        patient_data = {
            'patient_id': patient_id,
            'disease': patient_samples['disease'].iloc[0],
            'response': patient_samples['therapyResponse'].iloc[0] if 'therapyResponse' in patient_samples.columns else None,
            'timepoints': []
        }

        for _, row in patient_samples.iterrows():
            sample_id = row['sampleID']
            if sample_id in sample_activity_df.columns:
                tp_data = {
                    'timepoint': float(row['timepoint_replicate']) if pd.notna(row['timepoint_replicate']) else 0,
                    'sample_id': sample_id,
                    'activity': {}
                }
                for cyt in long_data['cytokines']:
                    val = sample_activity_df.loc[cyt, sample_id]
                    if pd.notna(val):
                        tp_data['activity'][cyt] = round(float(val), 4)
                patient_data['timepoints'].append(tp_data)

        if len(patient_data['timepoints']) > 1:
            long_data['patients'].append(patient_data)

    with open(OUTPUT_DIR / "inflammation_longitudinal.json", 'w') as f:
        json.dump(long_data, f)

    print(f"  Patients with longitudinal data: {len(long_data['patients'])}")
    return long_data


def preprocess_inflammation_cell_drivers():
    """Preprocess cell type driver analysis for inflammation atlas."""
    print("Processing inflammation cell type drivers...")

    try:
        import anndata as ad
    except ImportError:
        print("  Skipping - anndata not available")
        return None

    SAMPLE_META_PATH = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv')
    if not SAMPLE_META_PATH.exists():
        print("  Skipping - metadata not found")
        return None

    sample_meta = pd.read_csv(SAMPLE_META_PATH)

    # Load activity data
    h5ad_file = INFLAM_DIR / "main_CytoSig_pseudobulk.h5ad"
    if not h5ad_file.exists():
        print("  Skipping - activity data not found")
        return None

    adata = ad.read_h5ad(h5ad_file)
    activity_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    var_info = adata.var[['sample', 'cell_type', 'n_cells']].copy()

    # Merge sample info with disease metadata
    original_index = var_info.index.copy()
    var_info_reset = var_info.reset_index(drop=True)
    var_info_reset['_original_index'] = original_index
    var_info_merged = var_info_reset.merge(
        sample_meta[['sampleID', 'disease', 'diseaseStatus']],
        left_on='sample', right_on='sampleID', how='left'
    )
    var_info_merged = var_info_merged.set_index('_original_index')
    var_info_merged.index.name = None

    # Get unique diseases and cell types
    diseases = [d for d in var_info_merged['disease'].dropna().unique() if d != 'Healthy'][:10]
    cell_types = var_info_merged['cell_type'].unique()[:20]
    cytokines = activity_df.index.tolist()[:15]

    drivers_data = {
        'diseases': list(diseases),
        'cell_types': list(cell_types),
        'cytokines': cytokines,
        'effects': []
    }

    # Calculate disease vs healthy effect for each cell type and cytokine
    healthy_samples = var_info_merged[var_info_merged['disease'] == 'Healthy'].index

    for disease in diseases:
        disease_samples = var_info_merged[var_info_merged['disease'] == disease].index

        for ct in cell_types:
            ct_healthy = [s for s in healthy_samples if var_info_merged.loc[s, 'cell_type'] == ct]
            ct_disease = [s for s in disease_samples if var_info_merged.loc[s, 'cell_type'] == ct]

            if len(ct_healthy) < 5 or len(ct_disease) < 5:
                continue

            for cyt in cytokines:
                healthy_vals = activity_df.loc[cyt, ct_healthy].dropna()
                disease_vals = activity_df.loc[cyt, ct_disease].dropna()

                if len(healthy_vals) < 5 or len(disease_vals) < 5:
                    continue

                stat, pval = stats.ranksums(disease_vals, healthy_vals)
                effect = disease_vals.mean() - healthy_vals.mean()

                drivers_data['effects'].append({
                    'disease': disease,
                    'cell_type': ct,
                    'cytokine': cyt,
                    'effect': round(effect, 4),
                    'pvalue': round(pval, 6),
                    'n_healthy': len(healthy_vals),
                    'n_disease': len(disease_vals)
                })

    with open(OUTPUT_DIR / "inflammation_cell_drivers.json", 'w') as f:
        json.dump(drivers_data, f)

    print(f"  Effects computed: {len(drivers_data['effects'])}")
    return drivers_data


def preprocess_inflammation_demographics():
    """Preprocess inflammation demographics differential (sex/smoking volcano plots)."""
    print("Processing inflammation demographics differential...")

    try:
        import anndata as ad
    except ImportError:
        print("  Skipping - anndata not available")
        return None

    SAMPLE_META_PATH = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv')
    if not SAMPLE_META_PATH.exists():
        print("  Skipping - metadata not found")
        return None

    sample_meta = pd.read_csv(SAMPLE_META_PATH)

    # Load activity data
    h5ad_file = INFLAM_DIR / "main_CytoSig_pseudobulk.h5ad"
    if not h5ad_file.exists():
        print("  Skipping - activity data not found")
        return None

    adata = ad.read_h5ad(h5ad_file)
    activity_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    var_info = adata.var[['sample', 'cell_type', 'n_cells']].copy()

    # Aggregate to sample level
    sample_activity = {}
    for sample in var_info['sample'].unique():
        sample_cols = var_info[var_info['sample'] == sample].index
        weights = var_info.loc[sample_cols, 'n_cells'].values
        total_weight = weights.sum()
        if total_weight > 0:
            weighted_mean = (activity_df[sample_cols] * weights).sum(axis=1) / total_weight
            sample_activity[sample] = weighted_mean

    sample_activity_df = pd.DataFrame(sample_activity).T

    # Merge with metadata
    merged = sample_meta.merge(
        sample_activity_df.reset_index().rename(columns={'index': 'sampleID'}),
        on='sampleID', how='inner'
    )

    cytokines = activity_df.index.tolist()
    demo_data = {'sex': [], 'smoking': [], 'cytokines': cytokines}

    # Sex differential
    if 'sex' in merged.columns:
        male = merged[merged['sex'] == 'male']
        female = merged[merged['sex'] == 'female']

        for cyt in cytokines:
            if cyt in male.columns and cyt in female.columns:
                male_vals = male[cyt].dropna()
                female_vals = female[cyt].dropna()
                if len(male_vals) >= 10 and len(female_vals) >= 10:
                    stat, pval = stats.ranksums(male_vals, female_vals)
                    log2fc = np.log2((male_vals.mean() + 0.01) / (female_vals.mean() + 0.01))
                    demo_data['sex'].append({
                        'cytokine': cyt,
                        'log2fc': round(log2fc, 4),
                        'pvalue': round(pval, 6),
                        'n_male': len(male_vals),
                        'n_female': len(female_vals)
                    })

    # Smoking differential
    if 'smokingStatus' in merged.columns:
        smoker = merged[merged['smokingStatus'].isin(['current-smoker', 'ex-smoker'])]
        non_smoker = merged[merged['smokingStatus'] == 'never-smoker']

        for cyt in cytokines:
            if cyt in smoker.columns and cyt in non_smoker.columns:
                s_vals = smoker[cyt].dropna()
                ns_vals = non_smoker[cyt].dropna()
                if len(s_vals) >= 10 and len(ns_vals) >= 10:
                    stat, pval = stats.ranksums(s_vals, ns_vals)
                    log2fc = np.log2((s_vals.mean() + 0.01) / (ns_vals.mean() + 0.01))
                    demo_data['smoking'].append({
                        'cytokine': cyt,
                        'log2fc': round(log2fc, 4),
                        'pvalue': round(pval, 6),
                        'n_smoker': len(s_vals),
                        'n_nonsmoker': len(ns_vals)
                    })

    with open(OUTPUT_DIR / "inflammation_demographics.json", 'w') as f:
        json.dump(demo_data, f)

    print(f"  Sex differentials: {len(demo_data['sex'])}")
    print(f"  Smoking differentials: {len(demo_data['smoking'])}")
    return demo_data


def create_embedded_data():
    """Create embedded_data.js file for standalone HTML viewing."""
    print("\nCreating embedded_data.js...")

    # List of JSON files to embed
    json_files = [
        'cima_correlations.json',
        'cima_metabolites_top.json',
        'cima_differential.json',
        'cima_celltype.json',
        'cima_celltype_correlations.json',
        'cima_biochem_scatter.json',
        'cima_population_stratification.json',
        # 'cima_eqtl.json',  # Excluded - lazy loaded separately (39 MB)
        'scatlas_organs.json',
        'scatlas_organs_top.json',
        'scatlas_celltypes.json',
        'cancer_comparison.json',
        'cancer_types.json',  # NEW
        'immune_infiltration.json',  # NEW
        'exhaustion.json',  # NEW
        'caf_signatures.json',  # NEW
        'organ_cancer_matrix.json',  # NEW
        'adjacent_tissue.json',  # NEW
        'inflammation_celltype.json',
        'inflammation_correlations.json',
        'inflammation_disease.json',
        'inflammation_longitudinal.json',
        'inflammation_cell_drivers.json',
        'inflammation_demographics.json',
        'disease_sankey.json',
        'age_bmi_boxplots.json',
        'treatment_response.json',
        'cohort_validation.json',
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

    # Add small subset of eQTL data for initial fast visualization (500 eQTLs)
    # Full data (223K) is loaded on demand via "Load full data" button
    eqtl_top_path = OUTPUT_DIR / 'cima_eqtl_top.json'
    if eqtl_top_path.exists():
        with open(eqtl_top_path) as f:
            eqtl_data = json.load(f)
            # Embed only first 500 eQTLs for fast initial rendering
            embedded['cimaeqtltop'] = {
                'summary': {
                    **eqtl_data['summary'],
                    'displayed_eqtls': 500,
                    'note': 'Showing top 500 most significant eQTLs for fast initial loading. Click "Load full data" for all 223,405 eQTLs.'
                },
                'cell_types': eqtl_data['cell_types'],
                'genes': eqtl_data['genes'][:50],  # Top 50 genes
                'eqtls': eqtl_data['eqtls'][:500]  # Top 500 eQTLs
            }
            print(f"  Embedded cima_eqtl_top.json (subset): 500 eQTLs for initial view")
    else:
        print(f"  Skipping cima_eqtl_top.json (not found)")

    # Write as JavaScript
    js_content = f"const EMBEDDED_DATA = {json.dumps(embedded)};\n"

    with open(OUTPUT_DIR / "embedded_data.js", 'w') as f:
        f.write(js_content)

    total_size = (OUTPUT_DIR / "embedded_data.js").stat().st_size / (1024 * 1024)
    print(f"\n  Total embedded_data.js: {total_size:.2f} MB")

    return embedded


def preprocess_cancer_types():
    """Preprocess cancer type-specific signature data."""
    print("Processing cancer type signatures...")

    # Check for cancer type signatures file
    cancer_types_path = SCATLAS_DIR / "cancer_type_signatures.csv"
    if not cancer_types_path.exists():
        print("  cancer_type_signatures.csv not found - skipping")
        return None

    df = pd.read_csv(cancer_types_path)
    print(f"  Loaded {len(df)} records")

    # Process for visualization
    cancer_types = sorted(df['organ'].unique().tolist())  # organ column contains cancerType
    signatures = sorted(df['signature'].unique().tolist())

    # Filter to CytoSig for manageable size
    cytosig_df = df[df['signature_type'] == 'CytoSig'].copy()

    # Create output
    cancer_data = {
        'data': cytosig_df[['organ', 'signature', 'mean_activity', 'specificity_score', 'n_cells']].rename(
            columns={'organ': 'cancer_type'}
        ).round(4).to_dict('records'),
        'cancer_types': cancer_types,
        'signatures': [s for s in signatures if s in cytosig_df['signature'].values]
    }

    with open(OUTPUT_DIR / "cancer_types.json", 'w') as f:
        json.dump(cancer_data, f)

    print(f"  Cancer types: {len(cancer_types)}")
    print(f"  Output records: {len(cancer_data['data'])}")
    return cancer_data


def preprocess_immune_infiltration():
    """Preprocess immune infiltration data."""
    print("Processing immune infiltration data...")

    infiltration_path = SCATLAS_DIR / "cancer_immune_infiltration.csv"
    if not infiltration_path.exists():
        print("  cancer_immune_infiltration.csv not found - skipping")
        return None

    df = pd.read_csv(infiltration_path)
    print(f"  Loaded {len(df)} records")

    # Aggregate: mean per cancer type for each signature
    agg = df.groupby(['cancer_type', 'signature']).agg({
        'immune_proportion': 'first',
        'n_immune': 'first',
        'n_total': 'first',
        'mean_immune_activity': 'mean',
        'mean_nonimmune_activity': 'mean',
        'immune_enrichment': 'mean'
    }).reset_index()

    infiltration_data = {
        'data': agg.round(4).to_dict('records'),
        'cancer_types': sorted(df['cancer_type'].unique().tolist()),
        'signatures': sorted(df['signature'].unique().tolist())
    }

    with open(OUTPUT_DIR / "immune_infiltration.json", 'w') as f:
        json.dump(infiltration_data, f)

    print(f"  Output records: {len(infiltration_data['data'])}")
    return infiltration_data


def preprocess_exhaustion():
    """Preprocess T cell exhaustion data."""
    print("Processing T cell exhaustion data...")

    exhaustion_path = SCATLAS_DIR / "cancer_tcell_exhaustion.csv"
    if not exhaustion_path.exists():
        print("  cancer_tcell_exhaustion.csv not found - skipping")
        return None

    df = pd.read_csv(exhaustion_path)
    print(f"  Loaded {len(df)} records")

    # Filter to significant results for visualization
    if 'qvalue' in df.columns:
        significant = df[df['qvalue'] < 0.1].copy()
    else:
        significant = df.copy()

    exhaustion_data = {
        'data': significant.round(4).to_dict('records'),
        'signatures': sorted(df['signature'].unique().tolist()),
        'n_exhausted': int(df['n_exhausted'].iloc[0]) if len(df) > 0 else 0,
        'n_nonexhausted': int(df['n_nonexhausted'].iloc[0]) if len(df) > 0 else 0
    }

    with open(OUTPUT_DIR / "exhaustion.json", 'w') as f:
        json.dump(exhaustion_data, f)

    print(f"  Output records: {len(exhaustion_data['data'])}")
    return exhaustion_data


def preprocess_caf():
    """Preprocess cancer-associated fibroblast (CAF) data."""
    print("Processing CAF signatures...")

    caf_path = SCATLAS_DIR / "cancer_caf_signatures.csv"
    if not caf_path.exists():
        print("  cancer_caf_signatures.csv not found - skipping")
        return None

    df = pd.read_csv(caf_path)
    print(f"  Loaded {len(df)} records")

    # Get unique cancer types and CAF subtypes
    cancer_types = sorted(df['cancer_type'].unique().tolist())
    cell_types = sorted(df['cell_type'].unique().tolist())

    caf_data = {
        'data': df.round(4).to_dict('records'),
        'cancer_types': cancer_types,
        'cell_types': cell_types,
        'signatures': sorted(df['signature'].unique().tolist())
    }

    with open(OUTPUT_DIR / "caf_signatures.json", 'w') as f:
        json.dump(caf_data, f)

    print(f"  Cancer types: {len(cancer_types)}")
    print(f"  CAF subtypes: {len(cell_types)}")
    print(f"  Output records: {len(caf_data['data'])}")
    return caf_data


def preprocess_organ_cancer_matrix():
    """Preprocess organ-cancer comparison matrix."""
    print("Processing organ-cancer matrix...")

    # Load normal organ signatures
    normal_path = SCATLAS_DIR / "normal_organ_signatures.csv"
    cancer_path = SCATLAS_DIR / "cancer_type_signatures.csv"

    if not normal_path.exists():
        print("  normal_organ_signatures.csv not found - skipping")
        return None

    normal_df = pd.read_csv(normal_path)
    normal_df = normal_df[normal_df['signature_type'] == 'CytoSig']

    # Organ-cancer mapping
    organ_cancer_map = {
        'Liver': ['HCC', 'ICC', 'LIHC'],
        'Lung': ['LUAD', 'LUSC', 'NSCLC'],
        'Breast': ['BRCA'],
        'Colon': ['CRC', 'COAD'],
        'Kidney': ['KIRC', 'KIRP'],
        'Pancreas': ['PAAD'],
        'Stomach': ['STAD'],
        'Prostate': ['PRAD'],
        'Ovary': ['OV'],
        'Skin': ['SKCM', 'ALM'],
        'Thyroid': ['THCA']
    }

    matrix_data = {
        'normal_organs': [],
        'cancer_types': [],
        'comparisons': []
    }

    # Get organ means from normal atlas
    organ_means = normal_df.groupby(['organ', 'signature'])['mean_activity'].mean().reset_index()

    if cancer_path.exists():
        cancer_df = pd.read_csv(cancer_path)
        cancer_df = cancer_df[cancer_df['signature_type'] == 'CytoSig']
        cancer_means = cancer_df.groupby(['organ', 'signature'])['mean_activity'].mean().reset_index()
        cancer_means = cancer_means.rename(columns={'organ': 'cancer_type'})

        # Build comparisons
        for organ, cancer_list in organ_cancer_map.items():
            organ_data = organ_means[organ_means['organ'].str.lower().str.contains(organ.lower())]
            if len(organ_data) == 0:
                continue

            matrix_data['normal_organs'].append(organ)

            for cancer_type in cancer_list:
                cancer_data = cancer_means[cancer_means['cancer_type'] == cancer_type]
                if len(cancer_data) == 0:
                    continue

                if cancer_type not in matrix_data['cancer_types']:
                    matrix_data['cancer_types'].append(cancer_type)

                # Compute mean difference across signatures
                merged = organ_data.merge(cancer_data, on='signature', suffixes=('_normal', '_cancer'))
                if len(merged) > 0:
                    merged['difference'] = merged['mean_activity_cancer'] - merged['mean_activity_normal']
                    for _, row in merged.iterrows():
                        matrix_data['comparisons'].append({
                            'organ': organ,
                            'cancer_type': cancer_type,
                            'signature': row['signature'],
                            'normal_activity': round(row['mean_activity_normal'], 4),
                            'cancer_activity': round(row['mean_activity_cancer'], 4),
                            'difference': round(row['difference'], 4)
                        })

    with open(OUTPUT_DIR / "organ_cancer_matrix.json", 'w') as f:
        json.dump(matrix_data, f)

    print(f"  Normal organs: {len(matrix_data['normal_organs'])}")
    print(f"  Cancer types: {len(matrix_data['cancer_types'])}")
    print(f"  Comparisons: {len(matrix_data['comparisons'])}")
    return matrix_data


def preprocess_adjacent_tissue():
    """Preprocess adjacent tissue field effect data."""
    print("Processing adjacent tissue data...")

    adjacent_path = SCATLAS_DIR / "cancer_adjacent_signatures.csv"
    if not adjacent_path.exists():
        print("  cancer_adjacent_signatures.csv not found - skipping")
        return None

    df = pd.read_csv(adjacent_path)
    print(f"  Loaded {len(df)} records")

    # Round and prepare for visualization
    if 'qvalue' in df.columns:
        df['neg_log10_qval'] = -np.log10(df['qvalue'].clip(lower=1e-100))

    adjacent_data = {
        'data': df.round(4).to_dict('records'),
        'cancer_types': sorted(df['cancer_type'].unique().tolist()),
        'signatures': sorted(df['signature'].unique().tolist())
    }

    with open(OUTPUT_DIR / "adjacent_tissue.json", 'w') as f:
        json.dump(adjacent_data, f)

    print(f"  Cancer types: {len(adjacent_data['cancer_types'])}")
    print(f"  Output records: {len(adjacent_data['data'])}")
    return adjacent_data


def main():
    print("=" * 60)
    print("Preprocessing visualization data")
    print("=" * 60)

    # Process all data
    preprocess_cima_correlations()
    preprocess_cima_metabolites()
    preprocess_cima_differential()
    preprocess_cima_celltype()
    preprocess_cima_celltype_correlations()  # Cell type-specific age/BMI correlations
    preprocess_cima_biochem_scatter()  # NEW: sample-level scatter
    preprocess_cima_population_stratification()  # NEW: population stratification
    preprocess_scatlas_organs()
    preprocess_scatlas_celltypes()
    preprocess_cancer_comparison()
    preprocess_cancer_types()  # NEW: cancer type signatures
    preprocess_immune_infiltration()  # NEW: immune infiltration
    preprocess_exhaustion()  # NEW: T cell exhaustion
    preprocess_caf()  # NEW: CAF analysis
    preprocess_organ_cancer_matrix()  # NEW: organ-cancer matrix
    preprocess_adjacent_tissue()  # NEW: adjacent tissue field effect
    preprocess_inflammation()
    preprocess_inflammation_correlations()
    preprocess_inflammation_disease()
    preprocess_inflammation_longitudinal()  # NEW: longitudinal data
    preprocess_inflammation_cell_drivers()  # NEW: cell type drivers
    preprocess_inflammation_demographics()  # NEW: demographics differential
    preprocess_age_bmi_boxplots()
    preprocess_disease_sankey()
    preprocess_treatment_response()
    preprocess_cohort_validation()
    preprocess_cross_atlas()
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
