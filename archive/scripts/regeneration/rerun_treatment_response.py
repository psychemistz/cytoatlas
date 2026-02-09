#!/usr/bin/env python3
"""
Re-run treatment response prediction with proper cross-validation and ROC curve calculation.

This script loads the existing activity data and re-computes:
1. Logistic Regression and Random Forest models with stratified CV
2. Actual ROC curves (FPR/TPR) from cross-validated predictions
3. Feature importance (RF) and normalized coefficients (LR)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Paths
RESULTS_DIR = Path("results/inflammation")
OUTPUT_DIR = Path("visualization/data")
SEED = 42

def log(msg):
    print(msg)

def build_response_predictor_with_roc(
    activity_df: pd.DataFrame,
    sample_meta: pd.DataFrame,
    agg_meta_df: pd.DataFrame,
    disease: str = None,
    n_folds: int = 5
) -> dict:
    """
    Build treatment response predictor with actual ROC curves.

    Args:
        activity_df: proteins (rows) x sample_celltype (columns)
        sample_meta: sample-level metadata with therapyResponse
        agg_meta_df: column metadata with 'column', 'sample', 'cell_type'
        disease: filter to specific disease (None = all)
        n_folds: number of CV folds
    """
    log(f"Building predictor{' for ' + disease if disease else ' for All Diseases'}...")

    # Filter to samples with response data
    response_meta = sample_meta[sample_meta['therapyResponse'].isin(['R', 'NR'])].copy()

    if disease:
        response_meta = response_meta[response_meta['disease'] == disease]

    if len(response_meta) < 10:
        log(f"  Warning: Too few samples ({len(response_meta)})")
        return None

    # Map columns to samples using agg_meta_df
    col_to_sample = dict(zip(agg_meta_df['column'], agg_meta_df['sample']))

    # Aggregate activity by sample (mean across cell types)
    # activity_df has proteins as rows, sample_celltype as columns
    activity_T = activity_df.T.copy()  # Now sample_celltype as rows, proteins as columns
    activity_T['sample'] = activity_T.index.map(col_to_sample)

    # Drop rows without sample mapping
    activity_T = activity_T[activity_T['sample'].notna()]

    # Group by sample and take mean across cell types
    activity_by_sample = activity_T.groupby('sample').mean()

    # Merge with response labels
    merged = activity_by_sample.merge(
        response_meta[['sampleID', 'therapyResponse', 'disease']],
        left_index=True,
        right_on='sampleID',
        how='inner'
    )

    if len(merged) < 10:
        log(f"  Warning: Too few matched samples ({len(merged)})")
        return None

    # Prepare features and labels
    feature_cols = [c for c in activity_df.index if c in merged.columns]
    X = merged[feature_cols].values
    y = (merged['therapyResponse'] == 'R').astype(int).values

    n_pos = y.sum()
    n_neg = len(y) - n_pos

    log(f"  Features: {len(feature_cols)}, Samples: {len(y)}, R={n_pos}, NR={n_neg}")

    # Need at least 2 samples per class for CV
    if n_pos < 2 or n_neg < 2:
        log(f"  Warning: Not enough samples per class")
        return None

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation
    n_splits = min(n_folds, n_pos, n_neg)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    results = {
        'disease': disease if disease else 'All Diseases',
        'n_samples': len(y),
        'n_responders': int(n_pos),
        'n_nonresponders': int(n_neg),
        'n_features': len(feature_cols),
        'true_labels': y.tolist(),
    }

    # Logistic Regression
    try:
        lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight='balanced')
        y_pred_lr = cross_val_predict(lr, X_scaled, y, cv=cv, method='predict_proba')[:, 1]

        # Calculate AUC
        auc_lr = roc_auc_score(y, y_pred_lr)

        # Calculate actual ROC curve
        fpr_lr, tpr_lr, _ = roc_curve(y, y_pred_lr)

        results['lr_auc'] = round(auc_lr, 4)
        results['lr_fpr'] = [round(x, 4) for x in fpr_lr.tolist()]
        results['lr_tpr'] = [round(x, 4) for x in tpr_lr.tolist()]
        results['lr_probs'] = [round(x, 3) for x in y_pred_lr.tolist()]

        # Fit final model for coefficients
        lr.fit(X_scaled, y)
        coef_df = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': lr.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        results['lr_top_features'] = coef_df.head(20).to_dict('records')

        log(f"  Logistic Regression AUC: {auc_lr:.3f}")
    except Exception as e:
        log(f"  Logistic Regression failed: {e}")
        results['lr_auc'] = None

    # Random Forest
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight='balanced')
        y_pred_rf = cross_val_predict(rf, X_scaled, y, cv=cv, method='predict_proba')[:, 1]

        # Calculate AUC
        auc_rf = roc_auc_score(y, y_pred_rf)

        # Calculate actual ROC curve
        fpr_rf, tpr_rf, _ = roc_curve(y, y_pred_rf)

        results['rf_auc'] = round(auc_rf, 4)
        results['rf_fpr'] = [round(x, 4) for x in fpr_rf.tolist()]
        results['rf_tpr'] = [round(x, 4) for x in tpr_rf.tolist()]
        results['rf_probs'] = [round(x, 3) for x in y_pred_rf.tolist()]

        # Fit final model for feature importance
        rf.fit(X_scaled, y)
        imp_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        results['rf_top_features'] = imp_df.head(20).to_dict('records')

        log(f"  Random Forest AUC: {auc_rf:.3f}")
    except Exception as e:
        log(f"  Random Forest failed: {e}")
        results['rf_auc'] = None

    return results


def main():
    log("=" * 60)
    log("RE-RUNNING TREATMENT RESPONSE PREDICTION")
    log("=" * 60)

    # Load activity data
    log("\nLoading activity data...")

    import anndata as ad

    # Load CytoSig pseudobulk
    cyto_path = RESULTS_DIR / "main_CytoSig_pseudobulk.h5ad"
    if not cyto_path.exists():
        log(f"ERROR: {cyto_path} not found")
        return

    cyto_adata = ad.read_h5ad(cyto_path)
    log(f"  CytoSig: {cyto_adata.shape} (proteins x sample_celltype)")

    # Load SecAct pseudobulk
    secact_path = RESULTS_DIR / "main_SecAct_pseudobulk.h5ad"
    if not secact_path.exists():
        log(f"ERROR: {secact_path} not found")
        return

    secact_adata = ad.read_h5ad(secact_path)
    log(f"  SecAct: {secact_adata.shape} (proteins x sample_celltype)")

    # Get sample metadata from inflammation atlas
    sample_meta_path = Path("/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv")
    if not sample_meta_path.exists():
        log(f"ERROR: {sample_meta_path} not found")
        return

    sample_meta = pd.read_csv(sample_meta_path)
    log(f"  Sample metadata: {len(sample_meta)} samples")

    # Check for therapy response data
    if 'therapyResponse' not in sample_meta.columns:
        log("ERROR: No therapyResponse column in metadata")
        return

    response_counts = sample_meta['therapyResponse'].value_counts()
    log(f"  Response data: {response_counts.to_dict()}")

    all_results = {'CytoSig': [], 'SecAct': []}

    # Process each signature type
    for sig_name, adata in [('CytoSig', cyto_adata), ('SecAct', secact_adata)]:
        log(f"\n{'=' * 60}")
        log(f"PROCESSING {sig_name}")
        log(f"{'=' * 60}")

        # Get activity matrix - proteins (obs) x sample_celltype (var)
        # Note: adata is transposed (proteins as obs, sample_celltype as var)
        activity_df = pd.DataFrame(
            adata.X,
            index=adata.obs_names,  # proteins
            columns=adata.var_names  # sample_celltype
        )
        log(f"  Activity matrix: {activity_df.shape}")

        # Get aggregation metadata from var (columns = sample_celltype)
        agg_meta_df = adata.var.copy()
        agg_meta_df.index.name = 'column'
        agg_meta_df = agg_meta_df.reset_index()
        log(f"  Aggregation metadata: {len(agg_meta_df)} columns")

        # Get diseases with response data
        response_meta = sample_meta[sample_meta['therapyResponse'].isin(['R', 'NR'])]
        disease_counts = response_meta.groupby('disease').agg({
            'therapyResponse': [
                lambda x: (x == 'R').sum(),
                lambda x: (x == 'NR').sum()
            ]
        })
        disease_counts.columns = ['n_R', 'n_NR']

        # Filter diseases with enough samples
        min_per_class = 3
        eligible_diseases = disease_counts[
            (disease_counts['n_R'] >= min_per_class) &
            (disease_counts['n_NR'] >= min_per_class)
        ].index.tolist()

        log(f"Eligible diseases: {eligible_diseases}")

        # Run per-disease prediction
        for disease in eligible_diseases:
            result = build_response_predictor_with_roc(
                activity_df, sample_meta, agg_meta_df, disease=disease
            )
            if result:
                result['signature_type'] = sig_name
                all_results[sig_name].append(result)

        # Run pan-disease prediction
        log("\nRunning All Diseases combined...")
        pan_result = build_response_predictor_with_roc(
            activity_df, sample_meta, agg_meta_df, disease=None
        )
        if pan_result:
            pan_result['signature_type'] = sig_name
            all_results[sig_name].append(pan_result)

    # Save results
    log("\n" + "=" * 60)
    log("SAVING RESULTS")
    log("=" * 60)

    # Save detailed results
    details_path = RESULTS_DIR / "treatment_prediction_details.json"
    with open(details_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"Saved: {details_path}")

    # Create summary CSV
    summary_rows = []
    for sig_type, results in all_results.items():
        for r in results:
            for model in ['Logistic Regression', 'Random Forest']:
                prefix = 'lr' if model == 'Logistic Regression' else 'rf'
                auc = r.get(f'{prefix}_auc')
                if auc is not None:
                    summary_rows.append({
                        'disease': r['disease'],
                        'model': model,
                        'signature_type': sig_type,
                        'auc': auc,
                        'n_samples': r['n_samples'],
                        'n_responders': r['n_responders'],
                        'n_nonresponders': r['n_nonresponders']
                    })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "treatment_prediction_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log(f"Saved: {summary_path}")

    # Print summary
    log("\nSummary:")
    for sig_type in ['CytoSig', 'SecAct']:
        sig_rows = [r for r in summary_rows if r['signature_type'] == sig_type]
        if sig_rows:
            log(f"\n{sig_type}:")
            for r in sig_rows:
                log(f"  {r['disease']}: {r['model']} AUC={r['auc']:.3f} (n={r['n_samples']})")

    log("\nDone!")


if __name__ == "__main__":
    main()
