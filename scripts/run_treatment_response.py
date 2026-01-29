#!/usr/bin/env python3
"""
Run treatment response prediction for Inflammation Atlas.
Generates JSON data for visualization.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import anndata as ad

# Paths
RESULTS_DIR = Path("/vf/users/parks34/projects/2secactpy/results/inflammation")
OUTPUT_DIR = Path("/vf/users/parks34/projects/2secactpy/visualization/data")
SEED = 42

def log(msg):
    print(msg)


def load_sample_metadata():
    """Load sample metadata with treatment response."""
    meta_file = '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv'
    meta = pd.read_csv(meta_file)
    log(f"Loaded sample metadata: {len(meta)} samples")

    # Check therapy response
    response_meta = meta[meta['therapyResponse'].isin(['R', 'NR'])]
    log(f"Samples with response data: {len(response_meta)}")

    return meta


def build_response_predictor(
    activity_df: pd.DataFrame,
    sample_meta: pd.DataFrame,
    agg_meta_df: pd.DataFrame,
    disease: str = None,
    n_folds: int = 5
) -> dict:
    """
    Build treatment response predictor using cytokine activities.
    """
    log(f"Building predictor{' for ' + disease if disease else ' (all diseases)'}...")

    # Filter to samples with response data
    response_meta = sample_meta[sample_meta['therapyResponse'].isin(['R', 'NR'])].copy()

    if disease:
        response_meta = response_meta[response_meta['disease'] == disease]

    if len(response_meta) < 15:
        log(f"  Too few samples ({len(response_meta)})")
        return None

    # Map columns to samples
    col_to_sample = agg_meta_df['sample'].to_dict()

    # Aggregate activity by sample (mean across cell types)
    activity_T = activity_df.T.copy()
    activity_T['sample'] = activity_T.index.map(col_to_sample)
    activity_by_sample = activity_T.groupby('sample').mean()

    # Merge with response labels
    merged = activity_by_sample.merge(
        response_meta[['sampleID', 'therapyResponse', 'disease']],
        left_index=True,
        right_on='sampleID',
        how='inner'
    )

    if len(merged) < 15:
        log(f"  Too few matched samples ({len(merged)})")
        return None

    # Prepare features and labels
    feature_cols = [c for c in activity_df.index if c in merged.columns]
    X = merged[feature_cols].values
    y = (merged['therapyResponse'] == 'R').astype(int).values

    log(f"  Features: {len(feature_cols)}, Samples: {len(y)}, R={y.sum()}, NR={len(y) - y.sum()}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation (adjust folds based on minority class)
    min_class = min(y.sum(), len(y) - y.sum())
    actual_folds = min(n_folds, min_class)
    if actual_folds < 2:
        log(f"  Too few samples in minority class")
        return None

    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=SEED)

    results = {
        'disease': disease if disease else 'All Diseases',
        'n_samples': len(y),
        'n_responders': int(y.sum()),
        'n_nonresponders': int(len(y) - y.sum()),
        'n_features': len(feature_cols),
    }

    # Logistic Regression
    try:
        lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight='balanced')
        y_pred_lr = cross_val_predict(lr, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
        auc_lr = roc_auc_score(y, y_pred_lr)
        fpr_lr, tpr_lr, _ = roc_curve(y, y_pred_lr)
        results['lr_auc'] = round(auc_lr, 3)
        results['lr_fpr'] = [round(x, 4) for x in fpr_lr.tolist()]
        results['lr_tpr'] = [round(x, 4) for x in tpr_lr.tolist()]
        results['lr_probs'] = [round(x, 3) for x in y_pred_lr.tolist()]
        log(f"  Logistic Regression AUC: {auc_lr:.3f}")

        # Fit final model for coefficients
        lr.fit(X_scaled, y)
        coef_df = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': lr.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        results['lr_top_features'] = coef_df.head(15).to_dict('records')
    except Exception as e:
        log(f"  Logistic Regression failed: {e}")
        results['lr_auc'] = None

    # Random Forest
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight='balanced')
        y_pred_rf = cross_val_predict(rf, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
        auc_rf = roc_auc_score(y, y_pred_rf)
        fpr_rf, tpr_rf, _ = roc_curve(y, y_pred_rf)
        results['rf_auc'] = round(auc_rf, 3)
        results['rf_fpr'] = [round(x, 4) for x in fpr_rf.tolist()]
        results['rf_tpr'] = [round(x, 4) for x in tpr_rf.tolist()]
        results['rf_probs'] = [round(x, 3) for x in y_pred_rf.tolist()]
        log(f"  Random Forest AUC: {auc_rf:.3f}")

        # Fit final model for feature importance
        rf.fit(X_scaled, y)
        imp_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        results['rf_top_features'] = imp_df.head(15).to_dict('records')
    except Exception as e:
        log(f"  Random Forest failed: {e}")
        results['rf_auc'] = None

    # Store true labels for violin plots
    results['true_labels'] = y.tolist()

    return results


def run_treatment_response_prediction():
    """Run treatment response prediction for all available data."""
    log("\n" + "="*60)
    log("TREATMENT RESPONSE PREDICTION")
    log("="*60)

    sample_meta = load_sample_metadata()

    # Get diseases with sufficient response data
    response_meta = sample_meta[sample_meta['therapyResponse'].isin(['R', 'NR'])]
    disease_counts = response_meta.groupby('disease').agg({
        'therapyResponse': [
            lambda x: (x == 'R').sum(),
            lambda x: (x == 'NR').sum()
        ]
    })
    disease_counts.columns = ['n_R', 'n_NR']
    disease_counts['total'] = disease_counts['n_R'] + disease_counts['n_NR']

    # Need at least 10 samples with both R and NR
    eligible = disease_counts[(disease_counts['n_R'] >= 5) & (disease_counts['n_NR'] >= 5)]
    eligible_diseases = eligible.index.tolist()
    log(f"\nEligible diseases (>=5 R and >=5 NR): {eligible_diseases}")

    all_results = {'CytoSig': [], 'SecAct': []}

    for sig_type in ['CytoSig', 'SecAct']:
        log(f"\n--- {sig_type} ---")

        # Load pseudobulk activity data
        h5ad_file = RESULTS_DIR / f"main_{sig_type}_pseudobulk.h5ad"
        if not h5ad_file.exists():
            log(f"  {h5ad_file} not found, skipping")
            continue

        adata = ad.read_h5ad(h5ad_file)
        log(f"  Loaded {h5ad_file.name}: {adata.shape}")

        # Extract activity matrix
        activity_df = pd.DataFrame(
            adata.X,
            index=adata.obs_names,
            columns=adata.var_names
        )
        agg_meta_df = adata.var[['sample', 'cell_type', 'n_cells']].copy()

        # Run per-disease predictions
        for disease in eligible_diseases:
            result = build_response_predictor(
                activity_df, sample_meta, agg_meta_df, disease=disease
            )
            if result:
                result['signature_type'] = sig_type
                all_results[sig_type].append(result)

        # Run pan-disease prediction
        result = build_response_predictor(
            activity_df, sample_meta, agg_meta_df, disease=None
        )
        if result:
            result['signature_type'] = sig_type
            all_results[sig_type].append(result)

    return all_results


def generate_visualization_json(results):
    """Convert results to visualization-friendly JSON format."""
    log("\n" + "="*60)
    log("GENERATING VISUALIZATION JSON")
    log("="*60)

    viz_data = {
        'roc_curves': [],
        'feature_importance': [],
        'predictions': []
    }

    for sig_type, sig_results in results.items():
        for r in sig_results:
            disease = r['disease']

            # ROC curves
            if r.get('lr_auc'):
                viz_data['roc_curves'].append({
                    'disease': disease,
                    'model': 'Logistic Regression',
                    'signature_type': sig_type,
                    'auc': r['lr_auc'],
                    'n_samples': r['n_samples'],
                    'fpr': r['lr_fpr'],
                    'tpr': r['lr_tpr']
                })

            if r.get('rf_auc'):
                viz_data['roc_curves'].append({
                    'disease': disease,
                    'model': 'Random Forest',
                    'signature_type': sig_type,
                    'auc': r['rf_auc'],
                    'n_samples': r['n_samples'],
                    'fpr': r['rf_fpr'],
                    'tpr': r['rf_tpr']
                })

            # Feature importance (use RF importance)
            if r.get('rf_top_features'):
                for feat in r['rf_top_features']:
                    viz_data['feature_importance'].append({
                        'disease': disease,
                        'signature_type': sig_type,
                        'feature': feat['feature'],
                        'importance': round(feat['importance'], 4),
                        'model': 'Random Forest'
                    })

            # Predictions for violin plots
            if r.get('rf_probs') and r.get('true_labels'):
                for prob, label in zip(r['rf_probs'], r['true_labels']):
                    viz_data['predictions'].append({
                        'disease': disease,
                        'signature_type': sig_type,
                        'response': 'Responder' if label == 1 else 'Non-responder',
                        'probability': prob
                    })

    log(f"  ROC curves: {len(viz_data['roc_curves'])}")
    log(f"  Feature importance: {len(viz_data['feature_importance'])}")
    log(f"  Predictions: {len(viz_data['predictions'])}")

    return viz_data


def main():
    # Run predictions
    results = run_treatment_response_prediction()

    # Generate visualization JSON
    viz_data = generate_visualization_json(results)

    # Save
    output_file = OUTPUT_DIR / "treatment_response.json"
    with open(output_file, 'w') as f:
        json.dump(viz_data, f)
    log(f"\nSaved to {output_file}")

    # Also save detailed results
    details_file = RESULTS_DIR / "treatment_prediction_details.json"
    with open(details_file, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"Saved details to {details_file}")

    # Create summary CSV for compatibility
    summary_rows = []
    for sig_type, sig_results in results.items():
        for r in sig_results:
            if r.get('lr_auc'):
                summary_rows.append({
                    'disease': r['disease'],
                    'model': 'Logistic Regression',
                    'signature_type': sig_type,
                    'auc': r['lr_auc'],
                    'n_samples': r['n_samples'],
                    'n_responders': r['n_responders'],
                    'n_nonresponders': r['n_nonresponders']
                })
            if r.get('rf_auc'):
                summary_rows.append({
                    'disease': r['disease'],
                    'model': 'Random Forest',
                    'signature_type': sig_type,
                    'auc': r['rf_auc'],
                    'n_samples': r['n_samples'],
                    'n_responders': r['n_responders'],
                    'n_nonresponders': r['n_nonresponders']
                })

    summary_df = pd.DataFrame(summary_rows)
    summary_file = RESULTS_DIR / "treatment_prediction_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    log(f"Saved summary to {summary_file}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
