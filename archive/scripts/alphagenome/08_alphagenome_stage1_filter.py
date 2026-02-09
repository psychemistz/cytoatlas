#!/usr/bin/env python3
"""
AlphaGenome Stage 1: Filter eQTLs to Cytokine Gene Sets
=========================================================
Filter 223,405 CIMA cis-eQTLs (71,530 at FDR<0.05) to those where the target gene
is in CytoSig or SecAct signature matrices.

Input:
- CIMA_Lead_cis-xQTL.csv: 223,405 lead eQTL records
- CytoSig: 4,881 genes x 43 cytokines
- SecAct: 7,919 genes x 1,169 proteins

Output:
- results/alphagenome/stage1_cytokine_eqtls.csv
- results/alphagenome/stage1_summary.json
"""

import os
import sys
import json
import time
from pathlib import Path

import pandas as pd
import numpy as np

# Add SecActpy to path
sys.path.insert(0, '/vf/users/parks34/projects/1ridgesig/SecActpy')
from secactpy import load_cytosig, load_secact

# ==============================================================================
# Configuration
# ==============================================================================

EQTL_PATH = Path('/data/Jiang_Lab/Data/Seongyong/CIMA/xQTL/CIMA_Lead_cis-xQTL.csv')
OUTPUT_DIR = Path('/vf/users/parks34/projects/2secactpy/results/alphagenome')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# FDR threshold for significant eQTLs
FDR_THRESHOLD = 0.05


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=" * 60)
    log("ALPHAGENOME STAGE 1: FILTER eQTLs TO CYTOKINE GENES")
    log("=" * 60)

    # Load signature matrices
    log("Loading signature matrices...")
    cytosig = load_cytosig()
    secact = load_secact()
    log(f"  CytoSig: {cytosig.shape[0]} genes x {cytosig.shape[1]} cytokines")
    log(f"  SecAct: {secact.shape[0]} genes x {secact.shape[1]} proteins")

    # Get unique gene sets
    cytosig_genes = set(cytosig.index.str.upper())
    secact_genes = set(secact.index.str.upper())
    all_signature_genes = cytosig_genes | secact_genes
    log(f"  Union of signature genes: {len(all_signature_genes)}")

    # Load eQTL data
    log(f"\nLoading eQTL data: {EQTL_PATH}")
    eqtl_df = pd.read_csv(EQTL_PATH, index_col=0)
    log(f"  Total records: {len(eqtl_df):,}")
    log(f"  Columns: {list(eqtl_df.columns)}")

    # Filter to significant eQTLs
    log(f"\nFiltering to FDR < {FDR_THRESHOLD}...")
    sig_eqtl = eqtl_df[eqtl_df['study_wise_qval'] < FDR_THRESHOLD].copy()
    log(f"  Significant eQTLs: {len(sig_eqtl):,}")

    # Show cell type distribution
    log("\n  eQTLs by cell type:")
    ct_counts = sig_eqtl['celltype'].value_counts()
    for ct in ct_counts.head(10).index:
        log(f"    {ct}: {ct_counts[ct]:,}")
    if len(ct_counts) > 10:
        log(f"    ... and {len(ct_counts) - 10} more cell types")

    # Filter to cytokine-related genes
    log("\nFiltering to genes in signature matrices...")
    sig_eqtl['gene_upper'] = sig_eqtl['phenotype_id'].str.upper()
    cytokine_eqtl = sig_eqtl[sig_eqtl['gene_upper'].isin(all_signature_genes)].copy()
    log(f"  Cytokine-related eQTLs: {len(cytokine_eqtl):,}")

    # Annotate which signature(s) each gene belongs to
    cytokine_eqtl['in_cytosig'] = cytokine_eqtl['gene_upper'].isin(cytosig_genes)
    cytokine_eqtl['in_secact'] = cytokine_eqtl['gene_upper'].isin(secact_genes)

    # Show gene distribution
    unique_genes = cytokine_eqtl['phenotype_id'].nunique()
    unique_variants = cytokine_eqtl['variant_id'].nunique()
    log(f"\n  Unique genes: {unique_genes}")
    log(f"  Unique variants: {unique_variants}")
    log(f"  In CytoSig only: {(cytokine_eqtl['in_cytosig'] & ~cytokine_eqtl['in_secact']).sum()}")
    log(f"  In SecAct only: {(~cytokine_eqtl['in_cytosig'] & cytokine_eqtl['in_secact']).sum()}")
    log(f"  In both: {(cytokine_eqtl['in_cytosig'] & cytokine_eqtl['in_secact']).sum()}")

    # Show top genes by eQTL count
    log("\n  Top genes by eQTL count:")
    gene_counts = cytokine_eqtl['phenotype_id'].value_counts()
    for gene in gene_counts.head(15).index:
        log(f"    {gene}: {gene_counts[gene]}")

    # Clean up and save
    cytokine_eqtl = cytokine_eqtl.drop(columns=['gene_upper'])

    output_csv = OUTPUT_DIR / 'stage1_cytokine_eqtls.csv'
    cytokine_eqtl.to_csv(output_csv, index=False)
    log(f"\nSaved: {output_csv}")

    # Create summary JSON
    summary = {
        'stage': 1,
        'description': 'Filter eQTLs to cytokine/secreted protein genes',
        'input': {
            'eqtl_file': str(EQTL_PATH),
            'total_eqtls': len(eqtl_df),
            'fdr_threshold': FDR_THRESHOLD,
            'significant_eqtls': len(sig_eqtl),
        },
        'signature_genes': {
            'cytosig_genes': len(cytosig_genes),
            'secact_genes': len(secact_genes),
            'union_genes': len(all_signature_genes),
        },
        'output': {
            'cytokine_eqtls': len(cytokine_eqtl),
            'unique_genes': unique_genes,
            'unique_variants': unique_variants,
            'in_cytosig': int(cytokine_eqtl['in_cytosig'].sum()),
            'in_secact': int(cytokine_eqtl['in_secact'].sum()),
        },
        'cell_types': ct_counts.to_dict(),
        'top_genes': gene_counts.head(20).to_dict(),
    }

    summary_path = OUTPUT_DIR / 'stage1_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log(f"Saved: {summary_path}")

    log("\nStage 1 complete!")
    log(f"  Output: {len(cytokine_eqtl):,} cytokine-related eQTLs")

    return cytokine_eqtl


if __name__ == '__main__':
    main()
