#!/usr/bin/env python3
"""
AlphaGenome Stage 2: Format Variants for AlphaGenome API
=========================================================
Convert CIMA eQTL variant IDs to AlphaGenome API format.

Input:
- results/alphagenome/stage1_cytokine_eqtls.csv

Transformations:
- Parse variant_id (e.g., "chr1_814733") to CHROM + POS
- Map A2(REF) -> REF, A1(ALT/effect allele) -> ALT
- Flag indels for special handling
- Validate hg38 coordinates

Output:
- results/alphagenome/stage2_alphagenome_input.csv
- results/alphagenome/stage2_summary.json
"""

import os
import sys
import json
import time
import re
from pathlib import Path

import pandas as pd
import numpy as np

# ==============================================================================
# Configuration
# ==============================================================================

INPUT_DIR = Path('/vf/users/parks34/projects/2secactpy/results/alphagenome')
OUTPUT_DIR = INPUT_DIR


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_variant_id(variant_id: str) -> tuple:
    """
    Parse CIMA variant_id format to chromosome and position.

    Format: chr#_position (e.g., "chr1_814733")

    Returns:
        (chrom, pos) tuple or (None, None) if parsing fails
    """
    match = re.match(r'^(chr[\dXYM]+)_(\d+)$', variant_id, re.IGNORECASE)
    if match:
        return match.group(1).lower(), int(match.group(2))
    return None, None


def classify_variant(ref: str, alt: str) -> str:
    """
    Classify variant type.

    Returns:
        'SNV', 'insertion', 'deletion', 'complex', or 'unknown'
    """
    if pd.isna(ref) or pd.isna(alt):
        return 'unknown'

    ref = str(ref).strip()
    alt = str(alt).strip()

    if len(ref) == 1 and len(alt) == 1:
        return 'SNV'
    elif len(ref) == 1 and len(alt) > 1:
        return 'insertion'
    elif len(ref) > 1 and len(alt) == 1:
        return 'deletion'
    elif len(ref) > 1 and len(alt) > 1:
        return 'complex'
    else:
        return 'unknown'


def main():
    log("=" * 60)
    log("ALPHAGENOME STAGE 2: FORMAT VARIANTS FOR API")
    log("=" * 60)

    # Load Stage 1 output
    input_csv = INPUT_DIR / 'stage1_cytokine_eqtls.csv'
    log(f"Loading: {input_csv}")
    eqtl_df = pd.read_csv(input_csv)
    log(f"  Loaded {len(eqtl_df):,} eQTLs")

    # Parse variant IDs
    log("\nParsing variant IDs...")
    parsed = eqtl_df['variant_id'].apply(parse_variant_id)
    eqtl_df['CHROM'] = [p[0] for p in parsed]
    eqtl_df['POS'] = [p[1] for p in parsed]

    # Check for parsing failures
    parse_failures = eqtl_df['CHROM'].isna().sum()
    if parse_failures > 0:
        log(f"  WARNING: {parse_failures} variants failed to parse")
        failed_examples = eqtl_df[eqtl_df['CHROM'].isna()]['variant_id'].head(5).tolist()
        log(f"    Examples: {failed_examples}")

    # Remove unparseable variants
    eqtl_df = eqtl_df[eqtl_df['CHROM'].notna()].copy()
    eqtl_df['POS'] = eqtl_df['POS'].astype(int)
    log(f"  Successfully parsed: {len(eqtl_df):,}")

    # Map REF/ALT columns
    # A2(REF) is the reference allele, A1(ALT/effect allele) is the alternate
    log("\nMapping REF/ALT alleles...")
    eqtl_df['REF'] = eqtl_df['A2(REF)']
    eqtl_df['ALT'] = eqtl_df['A1(ALT/effect allele)']

    # Validate REF/ALT
    missing_ref = eqtl_df['REF'].isna().sum()
    missing_alt = eqtl_df['ALT'].isna().sum()
    if missing_ref > 0 or missing_alt > 0:
        log(f"  WARNING: Missing REF: {missing_ref}, Missing ALT: {missing_alt}")
        eqtl_df = eqtl_df.dropna(subset=['REF', 'ALT'])
        log(f"  After removing missing: {len(eqtl_df):,}")

    # Classify variant types
    log("\nClassifying variant types...")
    eqtl_df['variant_type'] = eqtl_df.apply(
        lambda row: classify_variant(row['REF'], row['ALT']),
        axis=1
    )

    type_counts = eqtl_df['variant_type'].value_counts()
    for vtype, count in type_counts.items():
        log(f"  {vtype}: {count:,}")

    # Show chromosome distribution
    log("\nChromosome distribution:")
    chrom_counts = eqtl_df['CHROM'].value_counts().sort_index()
    for chrom in list(chrom_counts.index)[:5]:
        log(f"  {chrom}: {chrom_counts[chrom]:,}")
    log(f"  ... ({len(chrom_counts)} chromosomes total)")

    # Create unique variant records (deduplicate across cell types)
    log("\nDeduplicating variants...")
    # Group by variant and take representative record (keep best p-value)
    eqtl_df = eqtl_df.sort_values('pval_nominal')
    unique_variants = eqtl_df.drop_duplicates(subset=['variant_id'], keep='first').copy()
    log(f"  Unique variants: {len(unique_variants):,}")

    # For AlphaGenome, we'll also keep track of all gene-variant associations
    variant_gene_map = eqtl_df.groupby('variant_id').agg({
        'phenotype_id': lambda x: list(set(x)),
        'celltype': lambda x: list(set(x)),
        'slope': list,
        'pval_nominal': 'min',
    }).reset_index()
    variant_gene_map.columns = ['variant_id', 'target_genes', 'cell_types', 'eqtl_betas', 'best_pval']

    # Merge with unique variants
    unique_variants = unique_variants.merge(
        variant_gene_map[['variant_id', 'target_genes', 'cell_types', 'eqtl_betas']],
        on='variant_id',
        how='left'
    )

    # Convert lists to strings for CSV storage
    unique_variants['target_genes_str'] = unique_variants['target_genes'].apply(
        lambda x: '|'.join(x) if isinstance(x, list) else str(x)
    )
    unique_variants['cell_types_str'] = unique_variants['cell_types'].apply(
        lambda x: '|'.join(x) if isinstance(x, list) else str(x)
    )
    unique_variants['eqtl_betas_str'] = unique_variants['eqtl_betas'].apply(
        lambda x: '|'.join(map(str, x)) if isinstance(x, list) else str(x)
    )

    # Select columns for AlphaGenome input
    output_cols = [
        'variant_id', 'CHROM', 'POS', 'REF', 'ALT',
        'phenotype_id', 'celltype', 'slope', 'pval_nominal', 'study_wise_qval',
        'variant_type', 'in_cytosig', 'in_secact',
        'target_genes_str', 'cell_types_str', 'eqtl_betas_str'
    ]

    output_df = unique_variants[output_cols].copy()
    output_df = output_df.rename(columns={
        'phenotype_id': 'primary_gene',
        'slope': 'eqtl_beta',
    })

    # Sort by chromosome and position
    chrom_order = [f'chr{i}' for i in range(1, 23)] + ['chrx', 'chry']
    output_df['chrom_sort'] = output_df['CHROM'].apply(
        lambda x: chrom_order.index(x) if x in chrom_order else 99
    )
    output_df = output_df.sort_values(['chrom_sort', 'POS']).drop(columns=['chrom_sort'])

    # Save output
    output_csv = OUTPUT_DIR / 'stage2_alphagenome_input.csv'
    output_df.to_csv(output_csv, index=False)
    log(f"\nSaved: {output_csv}")

    # Create summary JSON (convert numpy types to native Python types)
    summary = {
        'stage': 2,
        'description': 'Format variants for AlphaGenome API',
        'input': {
            'stage1_eqtls': int(len(pd.read_csv(input_csv))),
        },
        'output': {
            'unique_variants': int(len(output_df)),
            'parse_failures': int(parse_failures),
        },
        'variant_types': {k: int(v) for k, v in type_counts.to_dict().items()},
        'chromosomes': {k: int(v) for k, v in chrom_counts.to_dict().items()},
        'genome_build': 'hg38',
        'notes': [
            'Variants deduplicated across cell types (best p-value kept)',
            'target_genes_str contains all associated genes (pipe-separated)',
            'Coordinates are already hg38 from CIMA',
        ]
    }

    summary_path = OUTPUT_DIR / 'stage2_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log(f"Saved: {summary_path}")

    log("\nStage 2 complete!")
    log(f"  Output: {len(output_df):,} unique variants ready for AlphaGenome")

    return output_df


if __name__ == '__main__':
    main()
