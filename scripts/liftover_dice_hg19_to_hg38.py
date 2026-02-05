#!/usr/bin/env python3
"""
Liftover DICE eQTL coordinates from hg19 to hg38.

DICE database provides immune cell-type-specific eQTLs in hg19 coordinates.
This script converts them to hg38 for matching with CIMA eQTLs.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from pyliftover import LiftOver

# ==============================================================================
# Configuration
# ==============================================================================

DICE_DIR = Path('/vf/users/parks34/projects/2secactpy/results/alphagenome/dice_data')
OUTPUT_DIR = DICE_DIR / 'hg38'

# Cell types to process
CELL_TYPES = [
    'B_CELL_NAIVE',
    'CD4_NAIVE',
    'CD4_STIM',
    'CD8_NAIVE',
    'CD8_STIM',
    'NK',
    'TFH',
    'TH1',
    'TH2',
    'TH17',
    'TREG_MEM',
    'TREG_NAIVE',
]


def liftover_dice_vcf(
    input_vcf: Path,
    output_tsv: Path,
    lo: LiftOver
) -> Tuple[int, int, int]:
    """
    Liftover a DICE VCF file from hg19 to hg38.

    Returns:
        (total, lifted, unmapped) counts
    """
    # Read VCF (skip header lines starting with ##)
    df = pd.read_csv(
        input_vcf,
        sep='\t',
        comment='#',
        names=['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO'],
        dtype={'CHROM': str, 'POS': int}
    )

    total = len(df)

    # Liftover each position
    hg38_chroms = []
    hg38_positions = []

    for _, row in df.iterrows():
        chrom = row['CHROM']
        pos = row['POS']

        # pyliftover expects 0-based coordinates
        result = lo.convert_coordinate(chrom, pos - 1)

        if result and len(result) > 0:
            # Take first (best) match, convert back to 1-based
            new_chrom, new_pos, strand, score = result[0]
            hg38_chroms.append(new_chrom)
            hg38_positions.append(new_pos + 1)
        else:
            hg38_chroms.append(None)
            hg38_positions.append(None)

    # Add new columns
    df['CHROM_hg38'] = hg38_chroms
    df['POS_hg38'] = hg38_positions

    # Filter to successfully lifted variants
    lifted_df = df[df['POS_hg38'].notna()].copy()
    lifted_df['POS_hg38'] = lifted_df['POS_hg38'].astype(int)

    # Save
    lifted_df.to_csv(output_tsv, sep='\t', index=False)

    lifted = len(lifted_df)
    unmapped = total - lifted

    return total, lifted, unmapped


def main():
    print("=" * 60)
    print("DICE Liftover: hg19 -> hg38")
    print("=" * 60)

    # Initialize liftover
    print("\nInitializing LiftOver (downloading chain file if needed)...")
    lo = LiftOver('hg19', 'hg38')
    print("  LiftOver ready")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Process each cell type
    results = []

    for cell_type in CELL_TYPES:
        input_vcf = DICE_DIR / f'{cell_type}.vcf'

        if not input_vcf.exists():
            print(f"\n[SKIP] {cell_type}: file not found")
            continue

        output_tsv = OUTPUT_DIR / f'{cell_type}_hg38.tsv'

        print(f"\n[{cell_type}]")
        print(f"  Input: {input_vcf.name}")

        total, lifted, unmapped = liftover_dice_vcf(input_vcf, output_tsv, lo)

        print(f"  Total: {total:,}")
        print(f"  Lifted: {lifted:,} ({lifted/total*100:.1f}%)")
        print(f"  Unmapped: {unmapped:,} ({unmapped/total*100:.1f}%)")
        print(f"  Output: {output_tsv.name}")

        results.append({
            'cell_type': cell_type,
            'total': total,
            'lifted': lifted,
            'unmapped': unmapped,
            'lift_rate': lifted / total
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    results_df = pd.DataFrame(results)
    print(f"\nTotal variants processed: {results_df['total'].sum():,}")
    print(f"Total lifted: {results_df['lifted'].sum():,}")
    print(f"Overall lift rate: {results_df['lifted'].sum() / results_df['total'].sum() * 100:.1f}%")

    # Save summary
    summary_path = OUTPUT_DIR / 'liftover_summary.csv'
    results_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
