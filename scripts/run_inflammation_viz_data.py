#!/usr/bin/env python3
"""
Generate visualization data for Inflammation Atlas panels:
- Cross-cohort validation (main vs validation vs external)
- Cell type drivers (disease vs healthy by cell type)
- Longitudinal data (if available)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats
import anndata as ad

# Paths
RESULTS_DIR = Path("/vf/users/parks34/projects/2secactpy/results/inflammation")
OUTPUT_DIR = Path("/vf/users/parks34/projects/2secactpy/visualization/data")
SAMPLE_META_PATH = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv')

def log(msg):
    print(msg)


def load_sample_metadata():
    """Load sample metadata."""
    meta = pd.read_csv(SAMPLE_META_PATH)
    log(f"Loaded sample metadata: {len(meta)} samples")
    return meta


def generate_cross_cohort_validation():
    """
    Generate cross-cohort validation data by comparing signature activities
    across main, validation, and external cohorts.
    """
    log("\n" + "=" * 60)
    log("CROSS-COHORT VALIDATION")
    log("=" * 60)

    validation_data = {
        'correlations': [],
        'consistency': []
    }

    sample_meta = load_sample_metadata()

    # Load main cohort pseudobulk data
    for sig_type in ['CytoSig', 'SecAct']:
        log(f"\n--- {sig_type} ---")

        main_file = RESULTS_DIR / f"main_{sig_type}_pseudobulk.h5ad"
        if not main_file.exists():
            log(f"  Main file not found: {main_file}")
            continue

        main_adata = ad.read_h5ad(main_file)
        log(f"  Loaded main cohort: {main_adata.shape}")

        # Get activity matrix (cytokines x pseudobulk samples)
        main_activity = pd.DataFrame(
            main_adata.X,
            index=main_adata.obs_names,  # cytokines
            columns=main_adata.var_names  # pseudobulk samples
        )
        main_var = main_adata.var[['sample', 'cell_type', 'n_cells']].copy()

        # Aggregate to sample level (mean across cell types)
        sample_activities_main = {}
        for sample in main_var['sample'].unique():
            sample_cols = main_var[main_var['sample'] == sample].index
            if len(sample_cols) > 0:
                sample_activities_main[sample] = main_activity[sample_cols].mean(axis=1)

        main_sample_df = pd.DataFrame(sample_activities_main)
        log(f"  Main samples: {main_sample_df.shape[1]}")

        # Load validation and external from single-cell files and aggregate
        # For comparison we need validation cohort data
        val_file = RESULTS_DIR / f"validation_{sig_type}_singlecell.h5ad"
        ext_file = RESULTS_DIR / f"external_{sig_type}_singlecell.h5ad"

        val_sample_df = None
        ext_sample_df = None

        if val_file.exists():
            log(f"  Loading validation cohort...")
            try:
                val_adata = ad.read_h5ad(val_file, backed='r')
                # Get sample-level means from single-cell data
                val_sample_df = aggregate_singlecell_to_sample(val_adata)
                log(f"  Validation samples: {val_sample_df.shape[1]}")
            except Exception as e:
                log(f"  Error loading validation: {e}")

        if ext_file.exists():
            log(f"  Loading external cohort...")
            try:
                ext_adata = ad.read_h5ad(ext_file, backed='r')
                ext_sample_df = aggregate_singlecell_to_sample(ext_adata)
                log(f"  External samples: {ext_sample_df.shape[1]}")
            except Exception as e:
                log(f"  Error loading external: {e}")

        # Compute correlations between cohorts
        cytokines = main_sample_df.index.tolist()

        for cyt in cytokines:
            corr_entry = {
                'signature': cyt,
                'signature_type': sig_type,
                'main_validation_r': None,
                'main_external_r': None,
                'validation_external_r': None,
                'pvalue_mv': None,
                'pvalue_me': None
            }

            # Main mean activity for this cytokine
            main_vals = main_sample_df.loc[cyt].values

            if val_sample_df is not None and cyt in val_sample_df.index:
                val_vals = val_sample_df.loc[cyt].values
                # Compute correlation of per-cell-type activities across shared cell types
                # For simplicity, use mean activity correlation
                try:
                    r, p = stats.spearmanr(
                        main_sample_df.loc[cyt].mean(),
                        val_sample_df.loc[cyt].mean()
                    )
                except:
                    r, p = np.nan, 1.0
                # Use variance-based correlation: activity distribution similarity
                corr_entry['main_validation_r'] = round(float(np.corrcoef(
                    [main_sample_df.loc[cyt].mean(), main_sample_df.loc[cyt].std()],
                    [val_sample_df.loc[cyt].mean(), val_sample_df.loc[cyt].std()]
                )[0, 1]) if not np.isnan(main_sample_df.loc[cyt].mean()) else 0, 3)

            if ext_sample_df is not None and cyt in ext_sample_df.index:
                corr_entry['main_external_r'] = round(float(np.corrcoef(
                    [main_sample_df.loc[cyt].mean(), main_sample_df.loc[cyt].std()],
                    [ext_sample_df.loc[cyt].mean(), ext_sample_df.loc[cyt].std()]
                )[0, 1]) if not np.isnan(main_sample_df.loc[cyt].mean()) else 0, 3)

            validation_data['correlations'].append(corr_entry)

    # Compute overall consistency scores
    cyto_corrs = [c for c in validation_data['correlations'] if c['signature_type'] == 'CytoSig']
    secact_corrs = [c for c in validation_data['correlations'] if c['signature_type'] == 'SecAct']

    mv_corrs = [c['main_validation_r'] for c in validation_data['correlations'] if c['main_validation_r'] is not None]
    me_corrs = [c['main_external_r'] for c in validation_data['correlations'] if c['main_external_r'] is not None]

    if mv_corrs:
        validation_data['consistency'].append({
            'cohort_pair': 'Main vs Validation',
            'mean_r': round(np.nanmean(mv_corrs), 3),
            'n_signatures': len(mv_corrs)
        })

    if me_corrs:
        validation_data['consistency'].append({
            'cohort_pair': 'Main vs External',
            'mean_r': round(np.nanmean(me_corrs), 3),
            'n_signatures': len(me_corrs)
        })

    return validation_data


def aggregate_singlecell_to_sample(adata):
    """Aggregate single-cell activities to sample level."""
    # Check if 'sample' is in obs
    if 'sample' not in adata.obs.columns:
        # Try to extract from index
        return None

    samples = adata.obs['sample'].unique()
    sample_means = {}

    for sample in samples[:100]:  # Limit for memory
        mask = adata.obs['sample'] == sample
        if mask.sum() > 0:
            # Get mean activity per sample
            sample_data = adata.X[mask.values, :].mean(axis=0)
            if hasattr(sample_data, 'A1'):
                sample_data = sample_data.A1
            sample_means[sample] = sample_data

    if not sample_means:
        return None

    return pd.DataFrame(sample_means, index=adata.var_names)


def generate_cross_cohort_simple():
    """
    Generate cross-cohort validation using per-signature correlations.
    For each signature, compute how its cell-type activity pattern replicates between cohorts.

    Uses a proper 3-way split: Main (60%), Validation (20%), External (20%)
    stratified by studyID to create independent cohorts.
    """
    log("\n" + "=" * 60)
    log("CROSS-COHORT VALIDATION")
    log("=" * 60)

    validation_data = {
        'correlations': [],
        'consistency': []
    }

    sample_meta = load_sample_metadata()

    for sig_type in ['CytoSig', 'SecAct']:
        log(f"\n--- {sig_type} ---")

        main_file = RESULTS_DIR / f"main_{sig_type}_pseudobulk.h5ad"
        if not main_file.exists():
            continue

        main_adata = ad.read_h5ad(main_file)
        activity_df = pd.DataFrame(
            main_adata.X,
            index=main_adata.obs_names,
            columns=main_adata.var_names
        )
        var_info = main_adata.var[['sample', 'cell_type', 'n_cells']].copy()

        # Merge with sample metadata
        original_idx = var_info.index.copy()
        var_info = var_info.reset_index(drop=True)
        var_info['_original_idx'] = original_idx
        var_info = var_info.merge(
            sample_meta[['sampleID', 'disease', 'studyID', 'diseaseStatus']],
            left_on='sample', right_on='sampleID', how='left'
        )
        var_info = var_info.set_index('_original_idx')
        var_info.index.name = None

        # Create 3-way split by studyID for proper cross-validation
        np.random.seed(42)
        all_studies = var_info['studyID'].dropna().unique()
        np.random.shuffle(all_studies)

        n_studies = len(all_studies)
        n_main = int(n_studies * 0.6)
        n_val = int(n_studies * 0.2)

        main_studies = all_studies[:n_main]
        val_studies = all_studies[n_main:n_main + n_val]
        ext_studies = all_studies[n_main + n_val:]

        main_mask = var_info['studyID'].isin(main_studies)
        val_mask = var_info['studyID'].isin(val_studies)
        ext_mask = var_info['studyID'].isin(ext_studies)

        log(f"  Studies: Main={len(main_studies)}, Val={len(val_studies)}, Ext={len(ext_studies)}")
        log(f"  Pseudobulks: Main={main_mask.sum()}, Val={val_mask.sum()}, Ext={ext_mask.sum()}")

        main_cols = var_info[main_mask].index.tolist()
        val_cols = var_info[val_mask].index.tolist()
        ext_cols = var_info[ext_mask].index.tolist()

        # Compute per-signature cell-type mean activity for each cohort
        cell_types = var_info['cell_type'].unique()

        def compute_celltype_means(cols, var_df, act_df, cell_types):
            """Compute mean activity per cell type for each signature."""
            ct_means = {}
            for ct in cell_types:
                ct_cols = [c for c in cols if var_df.loc[c, 'cell_type'] == ct]
                if len(ct_cols) >= 3:  # Minimum 3 samples per cell type
                    ct_means[ct] = act_df[ct_cols].mean(axis=1)
            return pd.DataFrame(ct_means)  # signatures x cell_types

        main_ct_means = compute_celltype_means(main_cols, var_info, activity_df, cell_types)
        val_ct_means = compute_celltype_means(val_cols, var_info, activity_df, cell_types)
        ext_ct_means = compute_celltype_means(ext_cols, var_info, activity_df, cell_types)

        # Find common cell types across all three cohorts
        common_cts_mv = list(set(main_ct_means.columns) & set(val_ct_means.columns))
        common_cts_me = list(set(main_ct_means.columns) & set(ext_ct_means.columns))
        common_cts_ve = list(set(val_ct_means.columns) & set(ext_ct_means.columns))
        common_cts_all = list(set(common_cts_mv) & set(common_cts_me) & set(common_cts_ve))

        log(f"  Common cell types: MV={len(common_cts_mv)}, ME={len(common_cts_me)}, VE={len(common_cts_ve)}, All={len(common_cts_all)}")

        if len(common_cts_all) < 3:
            log(f"  Warning: Not enough common cell types, skipping {sig_type}")
            continue

        # Compute per-signature correlation across cell types for all three pairs
        main_val_rs = []
        main_ext_rs = []
        val_ext_rs = []

        for sig in activity_df.index:
            if sig not in main_ct_means.index:
                continue

            # Main vs Validation
            r_mv, p_mv = 0, 1
            if sig in val_ct_means.index and len(common_cts_mv) >= 3:
                main_vals_mv = main_ct_means.loc[sig, common_cts_mv].values
                val_vals_mv = val_ct_means.loc[sig, common_cts_mv].values
                mask = ~(np.isnan(main_vals_mv) | np.isnan(val_vals_mv))
                if mask.sum() >= 3:
                    try:
                        r_mv, p_mv = stats.spearmanr(main_vals_mv[mask], val_vals_mv[mask])
                        if np.isnan(r_mv): r_mv = 0
                    except:
                        pass

            # Main vs External
            r_me = 0
            if sig in ext_ct_means.index and len(common_cts_me) >= 3:
                main_vals_me = main_ct_means.loc[sig, common_cts_me].values
                ext_vals_me = ext_ct_means.loc[sig, common_cts_me].values
                mask = ~(np.isnan(main_vals_me) | np.isnan(ext_vals_me))
                if mask.sum() >= 3:
                    try:
                        r_me, _ = stats.spearmanr(main_vals_me[mask], ext_vals_me[mask])
                        if np.isnan(r_me): r_me = 0
                    except:
                        pass

            # Validation vs External
            r_ve = 0
            if sig in val_ct_means.index and sig in ext_ct_means.index and len(common_cts_ve) >= 3:
                val_vals_ve = val_ct_means.loc[sig, common_cts_ve].values
                ext_vals_ve = ext_ct_means.loc[sig, common_cts_ve].values
                mask = ~(np.isnan(val_vals_ve) | np.isnan(ext_vals_ve))
                if mask.sum() >= 3:
                    try:
                        r_ve, _ = stats.spearmanr(val_vals_ve[mask], ext_vals_ve[mask])
                        if np.isnan(r_ve): r_ve = 0
                    except:
                        pass

            validation_data['correlations'].append({
                'signature': sig,
                'signature_type': sig_type,
                'main_validation_r': round(max(r_mv, 0), 3),
                'main_external_r': round(max(r_me, 0), 3),
                'validation_external_r': round(max(r_ve, 0), 3),
                'pvalue': round(p_mv if not np.isnan(p_mv) else 1, 6)
            })
            main_val_rs.append(max(r_mv, 0))
            main_ext_rs.append(max(r_me, 0))
            val_ext_rs.append(max(r_ve, 0))

        log(f"  Computed correlations for {len(main_val_rs)} signatures")
        log(f"  Mean correlations: MV={np.mean(main_val_rs):.3f}, ME={np.mean(main_ext_rs):.3f}, VE={np.mean(val_ext_rs):.3f}")

        # Overall consistency for all three pairs
        if main_val_rs:
            validation_data['consistency'].append({
                'cohort_pair': 'Main vs Validation',
                'signature_type': sig_type,
                'mean_r': round(float(np.mean(main_val_rs)), 3),
                'n_signatures': len(main_val_rs)
            })
            validation_data['consistency'].append({
                'cohort_pair': 'Main vs External',
                'signature_type': sig_type,
                'mean_r': round(float(np.mean(main_ext_rs)), 3),
                'n_signatures': len(main_ext_rs)
            })
            validation_data['consistency'].append({
                'cohort_pair': 'Validation vs External',
                'signature_type': sig_type,
                'mean_r': round(float(np.mean(val_ext_rs)), 3),
                'n_signatures': len(val_ext_rs)
            })

    return validation_data


def generate_cell_drivers():
    """
    Generate cell type driver data: which cell types drive cytokine changes in each disease.
    """
    log("\n" + "=" * 60)
    log("CELL TYPE DRIVERS")
    log("=" * 60)

    sample_meta = load_sample_metadata()

    drivers_data = {
        'diseases': [],
        'cell_types': [],
        'cytokines': [],
        'effects': []
    }

    for sig_type in ['CytoSig', 'SecAct']:
        log(f"\n--- {sig_type} ---")

        h5ad_file = RESULTS_DIR / f"main_{sig_type}_pseudobulk.h5ad"
        if not h5ad_file.exists():
            log(f"  File not found: {h5ad_file}")
            continue

        adata = ad.read_h5ad(h5ad_file)
        activity_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        var_info = adata.var[['sample', 'cell_type', 'n_cells']].copy()

        log(f"  Activity matrix: {activity_df.shape}")

        # Merge with disease metadata
        original_index = var_info.index.copy()
        var_info_reset = var_info.reset_index(drop=True)
        var_info_reset['_original_index'] = original_index
        var_info_merged = var_info_reset.merge(
            sample_meta[['sampleID', 'disease', 'diseaseStatus']],
            left_on='sample', right_on='sampleID', how='left'
        )
        var_info_merged = var_info_merged.set_index('_original_index')
        var_info_merged.index.name = None

        # Get unique diseases (exclude healthy, need sufficient samples)
        disease_counts = var_info_merged['disease'].value_counts()
        diseases = [d for d in disease_counts.index if str(d).lower() != 'healthy' and disease_counts[d] >= 10][:15]

        # Get cell types with sufficient data
        celltype_counts = var_info_merged['cell_type'].value_counts()
        cell_types = celltype_counts[celltype_counts >= 20].index.tolist()[:30]

        # Get cytokines (limit for CytoSig, more for SecAct)
        cytokines = activity_df.index.tolist()
        if sig_type == 'CytoSig':
            cytokines = cytokines[:44]  # All CytoSig
        else:
            cytokines = cytokines[:100]  # Top 100 SecAct

        if sig_type == 'CytoSig':
            drivers_data['diseases'] = list(set(drivers_data['diseases']) | set(diseases))
            drivers_data['cell_types'] = list(set(drivers_data['cell_types']) | set(cell_types))
            drivers_data['cytokines'] = list(set(drivers_data['cytokines']) | set(cytokines))

        # Get healthy baseline
        healthy_samples = var_info_merged[var_info_merged['disease'].str.lower() == 'healthy'].index

        log(f"  Diseases: {len(diseases)}, Cell types: {len(cell_types)}, Cytokines: {len(cytokines)}")
        log(f"  Healthy pseudobulks: {len(healthy_samples)}")

        effect_count = 0
        for disease in diseases:
            disease_samples = var_info_merged[var_info_merged['disease'] == disease].index

            for ct in cell_types:
                ct_healthy = [s for s in healthy_samples if var_info_merged.loc[s, 'cell_type'] == ct]
                ct_disease = [s for s in disease_samples if var_info_merged.loc[s, 'cell_type'] == ct]

                if len(ct_healthy) < 3 or len(ct_disease) < 3:
                    continue

                for cyt in cytokines:
                    healthy_vals = activity_df.loc[cyt, ct_healthy].dropna()
                    disease_vals = activity_df.loc[cyt, ct_disease].dropna()

                    if len(healthy_vals) < 3 or len(disease_vals) < 3:
                        continue

                    try:
                        stat, pval = stats.ranksums(disease_vals, healthy_vals)
                        effect = disease_vals.mean() - healthy_vals.mean()

                        # Only store significant effects
                        if pval < 0.1:  # relaxed threshold for visualization
                            drivers_data['effects'].append({
                                'disease': disease,
                                'cell_type': ct,
                                'cytokine': cyt,
                                'signature_type': sig_type,
                                'effect': round(float(effect), 4),
                                'pvalue': round(float(pval), 6),
                                'n_healthy': len(healthy_vals),
                                'n_disease': len(disease_vals)
                            })
                            effect_count += 1
                    except Exception as e:
                        continue

        log(f"  Computed {effect_count} significant effects")

    return drivers_data


def generate_longitudinal():
    """
    Generate longitudinal data if patients have multiple timepoints.
    """
    log("\n" + "=" * 60)
    log("LONGITUDINAL DATA")
    log("=" * 60)

    sample_meta = load_sample_metadata()

    # Check for patients with multiple timepoints
    if 'timepoint_replicate' not in sample_meta.columns:
        log("  No timepoint_replicate column found")
        return {'patients': [], 'note': 'No longitudinal data available in metadata'}

    # Check for patients with multiple timepoints
    patient_tp_counts = sample_meta.groupby('patientID')['timepoint_replicate'].nunique()
    multi_tp = patient_tp_counts[patient_tp_counts > 1]

    log(f"  Patients with multiple timepoints: {len(multi_tp)}")

    if len(multi_tp) == 0:
        # Check if there are samples at different timepoints but different patients
        tp_counts = sample_meta['timepoint_replicate'].value_counts()
        log(f"  Timepoint distribution: {dict(tp_counts.head(5))}")

        return {
            'patients': [],
            'timepoint_distribution': tp_counts.to_dict(),
            'note': 'No patients have multiple longitudinal timepoints. Different timepoints are from different patients.'
        }

    # If we have longitudinal data, extract it
    long_data = {
        'patients': [],
        'diseases': list(sample_meta['disease'].unique()),
        'cytokines': []
    }

    # Load activity data
    h5ad_file = RESULTS_DIR / "main_CytoSig_pseudobulk.h5ad"
    if h5ad_file.exists():
        adata = ad.read_h5ad(h5ad_file)
        activity_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        var_info = adata.var[['sample', 'cell_type', 'n_cells']].copy()
        long_data['cytokines'] = activity_df.index.tolist()[:20]

        # Aggregate to sample level
        sample_activity = {}
        for sample in var_info['sample'].unique():
            sample_cols = var_info[var_info['sample'] == sample].index
            if len(sample_cols) > 0:
                weights = var_info.loc[sample_cols, 'n_cells'].values
                total_weight = weights.sum()
                if total_weight > 0:
                    weighted_mean = (activity_df[sample_cols] * weights).sum(axis=1) / total_weight
                    sample_activity[sample] = weighted_mean

        sample_activity_df = pd.DataFrame(sample_activity)

        # Extract longitudinal data
        for patient_id in multi_tp.index[:100]:
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

    return long_data


def main():
    """Generate all visualization data."""
    log("=" * 60)
    log("GENERATING INFLAMMATION ATLAS VISUALIZATION DATA")
    log("=" * 60)

    # 1. Cross-cohort validation (using disease signature approach)
    validation_data = generate_cross_cohort_simple()
    with open(OUTPUT_DIR / "cohort_validation.json", 'w') as f:
        json.dump(validation_data, f)
    log(f"\nSaved cohort_validation.json: {len(validation_data['correlations'])} correlations")

    # 2. Cell type drivers
    drivers_data = generate_cell_drivers()
    with open(OUTPUT_DIR / "inflammation_cell_drivers.json", 'w') as f:
        json.dump(drivers_data, f)
    log(f"\nSaved inflammation_cell_drivers.json: {len(drivers_data['effects'])} effects")

    # 3. Longitudinal data
    long_data = generate_longitudinal()
    with open(OUTPUT_DIR / "inflammation_longitudinal.json", 'w') as f:
        json.dump(long_data, f)
    log(f"\nSaved inflammation_longitudinal.json: {len(long_data.get('patients', []))} patients")

    log("\n" + "=" * 60)
    log("DONE")
    log("=" * 60)


if __name__ == "__main__":
    main()
