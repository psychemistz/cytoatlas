#!/usr/bin/env python3
"""
Preprocess cancer type signatures for visualization.
"""

import json
import pandas as pd
from pathlib import Path

# Paths
RESULTS_DIR = Path("/data/parks34/projects/2secactpy/results/scatlas")
OUTPUT_DIR = Path("/vf/users/parks34/projects/2secactpy/visualization/data")

def main():
    print("Loading cancer type signatures...")
    df = pd.read_csv(RESULTS_DIR / "cancer_type_signatures.csv")

    print(f"  Shape: {df.shape}")
    print(f"  Cancer types: {sorted(df['organ'].unique())}")

    # Rename organ to cancer_type for clarity
    df = df.rename(columns={'organ': 'cancer_type'})

    # Cancer type labels
    cancer_labels = {
        'BRCA': 'Breast Cancer',
        'CRC': 'Colorectal Cancer',
        'ESCA': 'Esophageal Cancer',
        'HCC': 'Hepatocellular Carcinoma',
        'HNSC': 'Head & Neck Squamous',
        'ICC': 'Intrahepatic Cholangiocarcinoma',
        'KIRC': 'Kidney Renal Clear Cell',
        'LUAD': 'Lung Adenocarcinoma',
        'LYM': 'Lymphoma',
        'PAAD': 'Pancreatic Adenocarcinoma',
        'STAD': 'Stomach Adenocarcinoma',
        'THCA': 'Thyroid Carcinoma',
        'cSCC': 'Cutaneous Squamous Cell'
    }

    # Get unique values
    cancer_types = sorted(df['cancer_type'].unique())
    cytosig_sigs = sorted(df[df['signature_type'] == 'CytoSig']['signature'].unique())
    secact_sigs = sorted(df[df['signature_type'] == 'SecAct']['signature'].unique())

    print(f"  CytoSig signatures: {len(cytosig_sigs)}")
    print(f"  SecAct signatures: {len(secact_sigs)}")

    # Prepare data records
    records = []
    for _, row in df.iterrows():
        records.append({
            'cancer_type': row['cancer_type'],
            'signature': row['signature'],
            'mean_activity': round(row['mean_activity'], 4) if pd.notna(row['mean_activity']) else 0,
            'other_mean': round(row['other_mean'], 4) if pd.notna(row['other_mean']) else 0,
            'log2fc': round(row['log2fc'], 4) if pd.notna(row['log2fc']) else None,
            'specificity_score': round(row['specificity_score'], 4) if pd.notna(row['specificity_score']) else 0,
            'n_cells': int(row['n_cells']) if pd.notna(row['n_cells']) else 0,
            'signature_type': row['signature_type']
        })

    # Create output structure
    output = {
        'data': records,
        'cancer_types': cancer_types,
        'cancer_labels': cancer_labels,
        'cytosig_signatures': cytosig_sigs,
        'secact_signatures': secact_sigs[:100],  # Limit SecAct for performance
        'total_secact': len(secact_sigs)
    }

    # Save JSON
    output_path = OUTPUT_DIR / "cancer_types.json"
    with open(output_path, 'w') as f:
        json.dump(output, f)

    print(f"Saved: {output_path}")
    print(f"  Records: {len(records)}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()
