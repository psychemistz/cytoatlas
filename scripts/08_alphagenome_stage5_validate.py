#!/usr/bin/env python3
"""
AlphaGenome Stage 5: Validate Against Multiple eQTL Sources
=============================================================
Cross-validate prioritized variants against multiple eQTL datasets:

1. GTEx v8/v10 whole blood bulk eQTLs
2. GTEx v8 neutrophil ieQTLs (cell type interaction)
3. DICE immune cell type-specific eQTLs

Analysis:
- Match variants by position between CIMA and each validation source
- Compute concordance metrics for each source
- Generate comprehensive validation report

Input:
- results/alphagenome/stage4_prioritized.csv
- GTEx eQTL data (bulk + ieQTL)
- DICE cell-type eQTL VCFs

Output:
- results/alphagenome/stage5_gtex_matched.csv
- results/alphagenome/stage5_validation_metrics.json
- results/alphagenome/stage5_report.md
"""

import os
import sys
import json
import time
import gzip
import argparse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ==============================================================================
# Configuration
# ==============================================================================

INPUT_DIR = Path('/vf/users/parks34/projects/2secactpy/results/alphagenome')
OUTPUT_DIR = INPUT_DIR

def parse_args():
    parser = argparse.ArgumentParser(description='Stage 5: Validate against GTEx')
    parser.add_argument('--input', type=str, default=None,
                        help='Input CSV file (default: stage4_prioritized.csv)')
    parser.add_argument('--test', action='store_true',
                        help='Use stage4_prioritized_test.csv for testing')
    parser.add_argument('--output-suffix', type=str, default='',
                        help='Suffix for output files (e.g., "_test")')
    return parser.parse_args()

# GTEx data paths - check both possible locations
GTEX_DIR = Path('/data/parks34/projects/2secactpy/results/alphagenome/gtex_data')
if not GTEX_DIR.exists():
    GTEX_DIR = INPUT_DIR / 'gtex_data'

# Supported GTEx file names (in order of preference)
GTEX_FILES = [
    # V10/V11 parquet format (preferred)
    'GTEx_Analysis_v10_eQTL_updated/Whole_Blood.v10.eQTLs.signif_pairs.parquet',
    'GTEx_Analysis_v11_eQTL_updated/Whole_Blood.v11.eQTLs.signif_pairs.parquet',
    # V8 text format
    'Whole_Blood.v8.signif_variant_gene_pairs.txt.gz',  # Significant only (~50 MB)
    'Whole_Blood.nominal.allpairs.txt.gz',               # All associations (large)
    'Whole-Blood.nominal.allpairs.txt.gz',               # Alternative naming
]

# GTEx ieQTL file (cell type interaction QTLs)
GTEX_IEQTL_FILE = 'GTEx_Analysis_v8_ieQTL/Whole_Blood.Neutrophils.ieQTL.eigenMT.annotated.txt.gz'

# DICE data directory (original hg19 and lifted hg38)
DICE_DIR = Path('/data/parks34/projects/2secactpy/results/alphagenome/dice_data')
DICE_HG38_DIR = DICE_DIR / 'hg38'  # Lifted to hg38 coordinates

# DICE cell type mapping: CIMA cell type prefix -> DICE file name (hg38 TSV format)
DICE_CELL_MAPPING = {
    'CD4': ['CD4_NAIVE_hg38.tsv', 'CD4_STIM_hg38.tsv', 'TH1_hg38.tsv', 'TH2_hg38.tsv', 'TH17_hg38.tsv', 'TFH_hg38.tsv', 'TREG_MEM_hg38.tsv', 'TREG_NAIVE_hg38.tsv'],
    'CD8': ['CD8_NAIVE_hg38.tsv', 'CD8_STIM_hg38.tsv'],
    'Treg': ['TREG_MEM_hg38.tsv', 'TREG_NAIVE_hg38.tsv'],
    'Th17': ['TH17_hg38.tsv'],
    'Th1': ['TH1_hg38.tsv'],
    'Th2': ['TH2_hg38.tsv'],
    'Tfh': ['TFH_hg38.tsv'],
    'NK': ['NK_hg38.tsv'],
    'B': ['B_CELL_NAIVE_hg38.tsv'],
    'Mono': ['MONOCYTES_CLASSICAL_hg38.tsv', 'MONOCYTES_NONCLASSICAL_hg38.tsv'],
    'MAIT': ['CD8_NAIVE_hg38.tsv'],  # MAIT cells - closest match is CD8
}

# Matching parameters
POSITION_TOLERANCE = 0  # Exact position match required

# P-value threshold for filtering all-pairs file
PVAL_THRESHOLD = 1e-5  # More lenient than FDR to capture more matches


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_gtex_file() -> Tuple[Path, bool, bool]:
    """
    Find available GTEx file.

    Returns:
        (path, is_allpairs, is_parquet) - path to file, whether it's all-pairs format, and whether it's parquet
    """
    GTEX_DIR.mkdir(parents=True, exist_ok=True)

    for filename in GTEX_FILES:
        local_path = GTEX_DIR / filename
        if local_path.exists():
            is_allpairs = 'allpairs' in filename.lower()
            is_parquet = filename.endswith('.parquet')
            log(f"Found GTEx file: {local_path}")
            if is_parquet:
                log(f"  (parquet format - V10/V11)")
            if is_allpairs:
                log(f"  (all-pairs format - will filter by p-value < {PVAL_THRESHOLD})")
            return local_path, is_allpairs, is_parquet

    return None, False, False


def download_gtex_eqtls() -> Tuple[Path, bool, bool]:
    """Find or download GTEx whole blood eQTLs."""
    gtex_path, is_allpairs, is_parquet = find_gtex_file()

    if gtex_path is not None:
        return gtex_path, is_allpairs, is_parquet

    log("GTEx data not found. Please download one of these files:")
    for f in GTEX_FILES:
        log(f"  - {f}")
    log(f"Place in: {GTEX_DIR}/")
    raise FileNotFoundError("GTEx data not found")


def load_gtex_eqtls(gtex_path: Path, target_genes: set, is_allpairs: bool = False, is_parquet: bool = False) -> pd.DataFrame:
    """
    Load GTEx eQTLs, filtering to target genes.

    GTEx signif_variant_gene_pairs format (V8 txt.gz):
    - variant_id: chr1_123456_A_G_b38
    - gene_id: ENSG00000123456.1
    - tss_distance: 12345
    - pval_nominal: 1e-10
    - slope: 0.5
    - slope_se: 0.1

    GTEx V10/V11 parquet format:
    - variant_id: chr1_123456_A_G_b38
    - gene_id: ENSG00000123456.1
    - tss_distance, af, ma_samples, ma_count
    - pval_nominal, slope, slope_se
    - pval_nominal_threshold, min_pval_nominal, pval_beta
    """
    log(f"Loading GTEx eQTLs: {gtex_path}")

    if is_parquet:
        # Load parquet file directly
        log("  Reading parquet format...")
        gtex_df = pd.read_parquet(gtex_path)
        log(f"  Loaded {len(gtex_df):,} GTEx eQTLs")
        return gtex_df

    # Handle gzip text format
    if is_allpairs:
        log(f"  Filtering to p-value < {PVAL_THRESHOLD}")

    # Read in chunks to handle large file
    chunks = []
    chunk_size = 500000
    total_rows = 0

    with gzip.open(gtex_path, 'rt') as f:
        reader = pd.read_csv(f, sep='\t', chunksize=chunk_size)

        for i, chunk in enumerate(reader):
            total_rows += len(chunk)

            # For all-pairs file, filter by p-value to reduce size
            if is_allpairs and 'pval_nominal' in chunk.columns:
                chunk = chunk[chunk['pval_nominal'] < PVAL_THRESHOLD]

            if len(chunk) > 0:
                chunks.append(chunk)

            if (i + 1) % 10 == 0:
                log(f"  Processed {total_rows:,} rows, kept {sum(len(c) for c in chunks):,}")

    if not chunks:
        log("  WARNING: No GTEx records passed filtering")
        return pd.DataFrame()

    gtex_df = pd.concat(chunks, ignore_index=True)
    log(f"  Total: {total_rows:,} rows, kept {len(gtex_df):,} GTEx eQTLs")

    return gtex_df


def load_gtex_ieqtls() -> Optional[pd.DataFrame]:
    """
    Load GTEx neutrophil ieQTLs from whole blood.

    ieQTL format columns:
    - variant_id: chr_pos_ref_alt_b38
    - gene_id, gene_name: target gene
    - b_g: main genotype effect
    - b_gi: genotype x cell type interaction effect (ieQTL effect)
    - pval_gi: interaction p-value
    - pval_adj_bh: BH-adjusted p-value
    """
    ieqtl_path = GTEX_DIR / GTEX_IEQTL_FILE
    if not ieqtl_path.exists():
        log(f"  GTEx ieQTL file not found: {ieqtl_path}")
        return None

    log(f"Loading GTEx neutrophil ieQTLs: {ieqtl_path}")

    ieqtl_df = pd.read_csv(ieqtl_path, sep='\t', compression='gzip')
    log(f"  Loaded {len(ieqtl_df):,} ieQTLs")

    # Parse variant coordinates
    parsed = ieqtl_df['variant_id'].str.extract(r'^(chr\d+)_(\d+)_([ACGT]+)_([ACGT]+)_b38$')
    ieqtl_df['chrom'] = parsed[0]
    ieqtl_df['pos'] = pd.to_numeric(parsed[1], errors='coerce')
    ieqtl_df['ref'] = parsed[2]
    ieqtl_df['alt'] = parsed[3]

    # Filter to valid coordinates
    ieqtl_df = ieqtl_df[ieqtl_df['chrom'].notna()].copy()
    log(f"  Valid coordinates: {len(ieqtl_df):,}")

    return ieqtl_df


def load_dice_eqtls(cell_types: List[str] = None) -> Optional[pd.DataFrame]:
    """
    Load DICE immune cell eQTLs (hg38 lifted coordinates).

    Reads from hg38-lifted TSV files that contain:
    - Original hg19 coordinates (CHROM, POS)
    - Lifted hg38 coordinates (CHROM_hg38, POS_hg38)
    - INFO fields parsed from original VCF

    Args:
        cell_types: List of cell type prefixes to load (e.g., ['CD4', 'CD8'])
                   If None, loads all available DICE files.

    Returns:
        DataFrame with columns: chrom, pos, ref, alt, gene_symbol, beta, pval, dice_celltype
    """
    # Use hg38 directory if available, fall back to original
    dice_dir = DICE_HG38_DIR if DICE_HG38_DIR.exists() else DICE_DIR
    use_hg38 = DICE_HG38_DIR.exists()

    if not dice_dir.exists():
        log(f"  DICE directory not found: {dice_dir}")
        return None

    log(f"  Using DICE data from: {dice_dir} ({'hg38' if use_hg38 else 'hg19'})")

    # Determine which files to load
    if cell_types:
        files_to_load = set()
        for ct in cell_types:
            # Find matching DICE cell types
            for prefix, dice_files in DICE_CELL_MAPPING.items():
                if ct.upper().startswith(prefix.upper()):
                    files_to_load.update(dice_files)
        files_to_load = list(files_to_load)
    else:
        if use_hg38:
            files_to_load = [f.name for f in dice_dir.glob('*_hg38.tsv')]
        else:
            files_to_load = [f.name for f in dice_dir.glob('*.vcf')]

    if not files_to_load:
        log("  No DICE files to load")
        return None

    log(f"Loading DICE eQTLs from {len(files_to_load)} cell types...")

    all_records = []

    for data_file in files_to_load:
        file_path = dice_dir / data_file
        if not file_path.exists() or file_path.stat().st_size == 0:
            continue

        # Extract cell type name from filename
        if use_hg38:
            dice_celltype = data_file.replace('_hg38.tsv', '')
        else:
            dice_celltype = data_file.replace('.vcf', '')

        log(f"  Loading {data_file}...")

        if use_hg38:
            # Load hg38 TSV format
            df = pd.read_csv(file_path, sep='\t')

            # Parse INFO field for each row
            records = []
            for _, row in df.iterrows():
                # Parse INFO field
                info_dict = {}
                if pd.notna(row.get('INFO', '')):
                    for item in str(row['INFO']).split(';'):
                        if '=' in item:
                            key, val = item.split('=', 1)
                            info_dict[key] = val

                # Use hg38 coordinates
                if pd.notna(row.get('CHROM_hg38')) and pd.notna(row.get('POS_hg38')):
                    records.append({
                        'chrom': row['CHROM_hg38'],
                        'pos': int(row['POS_hg38']),
                        'rsid': row.get('ID', ''),
                        'ref': row.get('REF', ''),
                        'alt': row.get('ALT', ''),
                        'gene_symbol': info_dict.get('GeneSymbol', ''),
                        'gene_id': info_dict.get('Gene', ''),
                        'beta': float(info_dict.get('Beta', 0)),
                        'pval': float(info_dict.get('Pvalue', 1)),
                        'dice_celltype': dice_celltype,
                    })
        else:
            # Original VCF format (hg19)
            records = []
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) < 8:
                        continue

                    chrom, pos, rsid, ref, alt, qual, filt, info = parts[:8]

                    # Parse INFO field
                    info_dict = {}
                    for item in info.split(';'):
                        if '=' in item:
                            key, val = item.split('=', 1)
                            info_dict[key] = val

                    records.append({
                        'chrom': chrom,
                        'pos': int(pos),
                        'rsid': rsid,
                        'ref': ref,
                        'alt': alt,
                        'gene_symbol': info_dict.get('GeneSymbol', ''),
                        'gene_id': info_dict.get('Gene', ''),
                        'beta': float(info_dict.get('Beta', 0)),
                        'pval': float(info_dict.get('Pvalue', 1)),
                        'dice_celltype': dice_celltype,
                    })

        if records:
            all_records.extend(records)
            log(f"    {len(records):,} eQTLs")

    if not all_records:
        log("  No DICE records loaded")
        return None

    dice_df = pd.DataFrame(all_records)
    log(f"  Total DICE eQTLs: {len(dice_df):,}")

    return dice_df


def parse_gtex_variant(variant_id: str) -> Tuple[str, int, str, str]:
    """
    Parse GTEx variant ID format.

    Format: chr#_position_ref_alt_b38
    Example: chr1_123456_A_G_b38

    Returns:
        (chrom, pos, ref, alt) or (None, None, None, None) if parsing fails
    """
    parts = variant_id.split('_')
    if len(parts) >= 4:
        chrom = parts[0]
        try:
            pos = int(parts[1])
            ref = parts[2]
            alt = parts[3]
            return chrom, pos, ref, alt
        except ValueError:
            pass
    return None, None, None, None


def create_gene_mapping() -> Dict[str, str]:
    """
    Create ENSG to gene symbol mapping.

    This is a placeholder - in production, load from Ensembl BioMart.
    """
    # Common cytokine genes (partial list for demonstration)
    mapping = {
        # Interleukins
        'ENSG00000136244': 'IL6',
        'ENSG00000169429': 'IL8',  # CXCL8
        'ENSG00000113520': 'IL4',
        'ENSG00000113525': 'IL5',
        'ENSG00000096968': 'IL17A',
        'ENSG00000100385': 'IL2',
        'ENSG00000169245': 'CXCL10',
        # Interferons
        'ENSG00000111537': 'IFNG',
        'ENSG00000107201': 'DDX58',
        # TNF family
        'ENSG00000232810': 'TNF',
        'ENSG00000120217': 'TNFSF14',
        # Chemokines
        'ENSG00000189377': 'CXCL2',
        'ENSG00000163739': 'CXCL1',
        # Growth factors
        'ENSG00000153317': 'CSF3',
        'ENSG00000164220': 'CSF2',
    }
    return mapping


def match_variants(
    cima_df: pd.DataFrame,
    gtex_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Match variants between CIMA and GTEx by genomic position.

    Returns matched variants with both eQTL effect sizes.
    """
    log("Matching variants between CIMA and GTEx...")

    # Handle empty GTEx dataframe
    if len(gtex_df) == 0:
        log("  WARNING: Empty GTEx data, no matches possible")
        return pd.DataFrame()

    # Parse GTEx variant IDs if needed (skip if already parsed)
    gtex_df = gtex_df.copy()
    if 'chrom' not in gtex_df.columns:
        gtex_parsed = gtex_df['variant_id'].apply(parse_gtex_variant)
        gtex_df['chrom'] = [p[0] for p in gtex_parsed]
        gtex_df['pos'] = [p[1] for p in gtex_parsed]
        gtex_df['ref'] = [p[2] for p in gtex_parsed]
        gtex_df['alt'] = [p[3] for p in gtex_parsed]

    # Remove unparseable
    gtex_df = gtex_df[gtex_df['chrom'].notna()].copy()
    log(f"  GTEx variants with valid coordinates: {len(gtex_df):,}")

    if len(gtex_df) == 0:
        log("  WARNING: No valid GTEx coordinates after parsing")
        return pd.DataFrame()

    # Create position keys
    cima_df = cima_df.copy()
    cima_df['pos_key'] = cima_df['CHROM'] + '_' + cima_df['POS'].astype(str)
    gtex_df['pos_key'] = gtex_df['chrom'] + '_' + gtex_df['pos'].astype(str)

    # Find position matches
    common_positions = set(cima_df['pos_key']) & set(gtex_df['pos_key'])
    log(f"  Common positions: {len(common_positions)}")

    if len(common_positions) == 0:
        log("  WARNING: No matching positions found")
        return pd.DataFrame()

    # Filter to common positions
    cima_matched = cima_df[cima_df['pos_key'].isin(common_positions)].copy()
    gtex_matched = gtex_df[gtex_df['pos_key'].isin(common_positions)].copy()

    # Prepare gtex subset with renamed columns
    gtex_subset = gtex_matched[['pos_key', 'gene_id', 'slope', 'slope_se', 'pval_nominal',
                                 'ref', 'alt']].copy()
    gtex_subset = gtex_subset.rename(columns={
        'gene_id': 'gtex_gene_id',
        'slope': 'gtex_slope',
        'slope_se': 'gtex_slope_se',
        'pval_nominal': 'gtex_pval',
        'ref': 'gtex_ref',
        'alt': 'gtex_alt',
    })

    # Merge on position key
    merged = cima_matched.merge(
        gtex_subset,
        on='pos_key',
        how='inner'
    )

    log(f"  Matched variant-gene pairs: {len(merged)}")

    # Check allele concordance
    # Note: May need to flip effect direction if alleles are swapped
    merged['alleles_match'] = (
        (merged['REF'] == merged['gtex_ref']) & (merged['ALT'] == merged['gtex_alt'])
    ) | (
        (merged['REF'] == merged['gtex_alt']) & (merged['ALT'] == merged['gtex_ref'])
    )

    merged['alleles_flipped'] = (
        (merged['REF'] == merged['gtex_alt']) & (merged['ALT'] == merged['gtex_ref'])
    )

    log(f"  Exact allele matches: {merged['alleles_match'].sum()}")
    log(f"  Flipped alleles: {merged['alleles_flipped'].sum()}")

    return merged


def match_variants_ieqtl(
    cima_df: pd.DataFrame,
    ieqtl_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Match variants between CIMA and GTEx ieQTLs.
    """
    log("Matching variants against GTEx ieQTLs...")

    if ieqtl_df is None or len(ieqtl_df) == 0:
        log("  No ieQTL data available")
        return pd.DataFrame()

    cima_df = cima_df.copy()
    cima_df['pos_key'] = cima_df['CHROM'] + '_' + cima_df['POS'].astype(str)

    ieqtl_df = ieqtl_df.copy()
    ieqtl_df['pos_key'] = ieqtl_df['chrom'] + '_' + ieqtl_df['pos'].astype(str)

    common_positions = set(cima_df['pos_key']) & set(ieqtl_df['pos_key'])
    log(f"  Common positions: {len(common_positions)}")

    if len(common_positions) == 0:
        return pd.DataFrame()

    cima_matched = cima_df[cima_df['pos_key'].isin(common_positions)].copy()
    ieqtl_matched = ieqtl_df[ieqtl_df['pos_key'].isin(common_positions)].copy()

    # Prepare ieqtl subset
    ieqtl_subset = ieqtl_matched[['pos_key', 'gene_id', 'gene_name', 'b_g', 'b_gi',
                                   'pval_gi', 'pval_adj_bh', 'ref', 'alt']].copy()
    ieqtl_subset = ieqtl_subset.rename(columns={
        'gene_id': 'ieqtl_gene_id',
        'gene_name': 'ieqtl_gene_name',
        'b_g': 'ieqtl_beta_genotype',
        'b_gi': 'ieqtl_beta_interaction',
        'pval_gi': 'ieqtl_pval',
        'pval_adj_bh': 'ieqtl_pval_adj',
        'ref': 'ieqtl_ref',
        'alt': 'ieqtl_alt',
    })

    merged = cima_matched.merge(ieqtl_subset, on='pos_key', how='inner')
    log(f"  Matched variant-gene pairs: {len(merged)}")

    # Check allele concordance
    merged['alleles_match'] = (
        (merged['REF'] == merged['ieqtl_ref']) & (merged['ALT'] == merged['ieqtl_alt'])
    ) | (
        (merged['REF'] == merged['ieqtl_alt']) & (merged['ALT'] == merged['ieqtl_ref'])
    )
    merged['alleles_flipped'] = (
        (merged['REF'] == merged['ieqtl_alt']) & (merged['ALT'] == merged['ieqtl_ref'])
    )

    return merged


def match_variants_dice(
    cima_df: pd.DataFrame,
    dice_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Match variants between CIMA and DICE eQTLs.
    """
    log("Matching variants against DICE eQTLs...")

    if dice_df is None or len(dice_df) == 0:
        log("  No DICE data available")
        return pd.DataFrame()

    cima_df = cima_df.copy()
    cima_df['pos_key'] = cima_df['CHROM'] + '_' + cima_df['POS'].astype(str)

    dice_df = dice_df.copy()
    dice_df['pos_key'] = dice_df['chrom'] + '_' + dice_df['pos'].astype(str)

    common_positions = set(cima_df['pos_key']) & set(dice_df['pos_key'])
    log(f"  Common positions: {len(common_positions)}")

    if len(common_positions) == 0:
        return pd.DataFrame()

    cima_matched = cima_df[cima_df['pos_key'].isin(common_positions)].copy()
    dice_matched = dice_df[dice_df['pos_key'].isin(common_positions)].copy()

    # Prepare dice subset
    dice_subset = dice_matched[['pos_key', 'gene_symbol', 'beta', 'pval',
                                 'dice_celltype', 'ref', 'alt']].copy()
    dice_subset = dice_subset.rename(columns={
        'gene_symbol': 'dice_gene',
        'beta': 'dice_beta',
        'pval': 'dice_pval',
        'ref': 'dice_ref',
        'alt': 'dice_alt',
    })

    merged = cima_matched.merge(dice_subset, on='pos_key', how='inner')
    log(f"  Matched variant-gene pairs: {len(merged)}")

    # Check allele concordance
    merged['alleles_match'] = (
        (merged['REF'] == merged['dice_ref']) & (merged['ALT'] == merged['dice_alt'])
    ) | (
        (merged['REF'] == merged['dice_alt']) & (merged['ALT'] == merged['dice_ref'])
    )
    merged['alleles_flipped'] = (
        (merged['REF'] == merged['dice_alt']) & (merged['ALT'] == merged['dice_ref'])
    )

    return merged


def compute_concordance(matched_df: pd.DataFrame, cima_col: str = 'eqtl_beta',
                        val_col: str = 'gtex_slope', source_name: str = 'GTEx') -> Dict[str, float]:
    """
    Compute effect direction concordance between CIMA and a validation source.

    Args:
        matched_df: DataFrame with matched variants
        cima_col: Column name for CIMA effect size
        val_col: Column name for validation source effect size
        source_name: Name of validation source for logging
    """
    if len(matched_df) == 0:
        return {'concordance': 0.0, 'n_matched': 0, 'source': source_name}

    # Get effect sizes
    cima_beta = matched_df[cima_col].values
    val_beta = matched_df[val_col].values

    # Flip effect if alleles are swapped
    val_beta_adj = val_beta.copy()
    if 'alleles_flipped' in matched_df.columns:
        val_beta_adj[matched_df['alleles_flipped'].values] *= -1

    # Direction concordance
    same_direction = np.sign(cima_beta) == np.sign(val_beta_adj)
    concordance = np.mean(same_direction)

    # Effect size correlation
    if len(cima_beta) > 2:
        r, p = stats.pearsonr(cima_beta, val_beta_adj)
        rho, rho_p = stats.spearmanr(cima_beta, val_beta_adj)
    else:
        r, p = np.nan, np.nan
        rho, rho_p = np.nan, np.nan

    return {
        'source': source_name,
        'concordance': float(concordance),
        'n_matched': len(matched_df),
        'pearson_r': float(r) if not np.isnan(r) else None,
        'pearson_p': float(p) if not np.isnan(p) else None,
        'spearman_rho': float(rho) if not np.isnan(rho) else None,
        'spearman_p': float(rho_p) if not np.isnan(rho_p) else None,
    }


def generate_report(
    cima_df: pd.DataFrame,
    all_metrics: Dict[str, Dict],
    matched_counts: Dict[str, int]
) -> str:
    """Generate markdown validation report for multiple sources."""

    report = f"""# AlphaGenome eQTL Multi-Source Validation Report

## Overview

This report validates CIMA immune cell eQTLs against multiple independent datasets:

1. **GTEx Bulk eQTLs** - Whole blood bulk tissue eQTLs (GTEx v8/v10)
2. **GTEx ieQTLs** - Neutrophil cell type interaction QTLs (GTEx v8)
3. **DICE eQTLs** - Immune cell type-specific eQTLs from sorted cells

## Input Summary

| Metric | Value |
|--------|-------|
| CIMA prioritized variants | {len(cima_df):,} |

## Validation Results by Source

"""

    # Add results for each source
    for source, metrics in all_metrics.items():
        n_matched = matched_counts.get(source, 0)
        match_rate = n_matched / len(cima_df) * 100 if len(cima_df) > 0 else 0

        concordance = metrics.get('concordance', 0) * 100
        pearson_r = metrics.get('pearson_r')
        spearman_rho = metrics.get('spearman_rho')

        report += f"""### {source}

| Metric | Value |
|--------|-------|
| Matched variants | {n_matched:,} |
| Match rate | {match_rate:.1f}% |
| Direction concordance | {concordance:.1f}% |
| Pearson r | {f'{pearson_r:.3f}' if pearson_r else 'N/A'} |
| Spearman ρ | {f'{spearman_rho:.3f}' if spearman_rho else 'N/A'} |

"""

    # Overall interpretation
    report += """## Interpretation

"""
    # Calculate average concordance across sources with matches
    valid_metrics = [m for m in all_metrics.values() if m.get('n_matched', 0) > 0]
    if valid_metrics:
        avg_concordance = np.mean([m['concordance'] for m in valid_metrics])
        if avg_concordance > 0.7:
            report += "**Strong overall concordance** across validation sources.\n"
            report += "The prioritized eQTL variants show consistent effects across datasets.\n"
        elif avg_concordance > 0.5:
            report += "**Moderate concordance** observed.\n"
            report += "Cell type-specific effects may explain some discordance between bulk and cell-sorted data.\n"
        else:
            report += "**Variable concordance** across sources.\n"
            report += "This may reflect genuine tissue/cell-specific regulation or sample differences.\n"

        # Highlight best performing source
        best_source = max(valid_metrics, key=lambda m: m['concordance'])
        report += f"\nBest concordance: **{best_source['source']}** ({best_source['concordance']*100:.1f}%)\n"
    else:
        report += "No variants matched any validation source.\n"

    report += """
## Data Sources

### GTEx Bulk eQTLs
- Source: GTEx Portal (v8 or v10)
- Tissue: Whole Blood
- Type: Standard cis-eQTLs from bulk tissue

### GTEx ieQTLs
- Source: GTEx v8 Cell Type Interaction QTLs
- Tissue: Whole Blood, Neutrophil enrichment
- Type: Genotype × cell type interaction effects
- Reference: Kim-Hellmuth et al., Science 2020

### DICE eQTLs
- Source: DICE Database (dice-database.org)
- Cell Types: Sorted immune cells (CD4, CD8, NK, B, Monocytes, etc.)
- Type: Cell type-specific cis-eQTLs from FACS-sorted cells

## Notes

- GTEx bulk represents whole blood mixture, while CIMA and DICE have cell-type resolution
- ieQTLs capture effects that vary with neutrophil proportion
- DICE data is from healthy donors, CIMA includes various disease states
- Allele matching was performed to ensure proper effect direction comparison

## Files Generated

- `stage5_gtex_matched.csv`: Matched variants with GTEx bulk effect sizes
- `stage5_ieqtl_matched.csv`: Matched variants with ieQTL effects
- `stage5_dice_matched.csv`: Matched variants with DICE cell-type effects
- `stage5_validation_metrics.json`: Detailed validation statistics
- `stage5_report.md`: This report
"""

    return report


def main():
    args = parse_args()

    log("=" * 60)
    log("ALPHAGENOME STAGE 5: MULTI-SOURCE VALIDATION")
    log("=" * 60)

    # Determine input file
    if args.input:
        input_csv = Path(args.input)
    elif args.test:
        input_csv = INPUT_DIR / 'stage4_prioritized_test.csv'
    else:
        input_csv = INPUT_DIR / 'stage4_prioritized.csv'

    # Output suffix for test mode
    suffix = args.output_suffix or ('_test' if args.test else '')

    log(f"Loading: {input_csv}")

    if not input_csv.exists():
        log(f"ERROR: Input file not found: {input_csv}")
        log("Please run Stage 4 first")
        return

    cima_df = pd.read_csv(input_csv)
    log(f"  Loaded {len(cima_df):,} prioritized variants")

    # Get target genes and cell types for filtering
    target_genes = set()
    cell_types = set()
    if 'primary_gene' in cima_df.columns:
        target_genes.update(cima_df['primary_gene'].dropna().str.upper())
    if 'target_genes_str' in cima_df.columns:
        for genes_str in cima_df['target_genes_str'].dropna():
            target_genes.update(g.upper() for g in genes_str.split('|'))
    if 'celltype' in cima_df.columns:
        cell_types.update(cima_df['celltype'].dropna().unique())
    if 'cell_types_str' in cima_df.columns:
        for ct_str in cima_df['cell_types_str'].dropna():
            cell_types.update(ct_str.split('|'))

    log(f"  Target genes: {len(target_genes)}")
    log(f"  Cell types: {len(cell_types)}")

    # Storage for all validation results
    all_metrics = {}
    matched_counts = {}
    all_matched_dfs = {}

    # =====================================================================
    # 1. GTEx Bulk eQTLs
    # =====================================================================
    log("\n" + "-" * 40)
    log("VALIDATION SOURCE 1: GTEx Bulk eQTLs")
    log("-" * 40)

    try:
        gtex_path, is_allpairs, is_parquet = download_gtex_eqtls()
        gtex_df = load_gtex_eqtls(gtex_path, target_genes, is_allpairs=is_allpairs, is_parquet=is_parquet)
        gtex_matched = match_variants(cima_df, gtex_df)

        if len(gtex_matched) > 0:
            metrics = compute_concordance(gtex_matched, 'eqtl_beta', 'gtex_slope', 'GTEx Bulk')
            all_metrics['GTEx Bulk'] = metrics
            matched_counts['GTEx Bulk'] = len(gtex_matched)
            all_matched_dfs['gtex'] = gtex_matched
            log(f"  Concordance: {metrics['concordance']*100:.1f}%")
        else:
            log("  No matches found")
            all_metrics['GTEx Bulk'] = {'concordance': 0.0, 'n_matched': 0, 'source': 'GTEx Bulk'}
            matched_counts['GTEx Bulk'] = 0

    except Exception as e:
        log(f"  ERROR: {e}")
        all_metrics['GTEx Bulk'] = {'concordance': 0.0, 'n_matched': 0, 'source': 'GTEx Bulk'}
        matched_counts['GTEx Bulk'] = 0

    # =====================================================================
    # 2. GTEx ieQTLs (Neutrophil interaction)
    # =====================================================================
    log("\n" + "-" * 40)
    log("VALIDATION SOURCE 2: GTEx Neutrophil ieQTLs")
    log("-" * 40)

    try:
        ieqtl_df = load_gtex_ieqtls()
        if ieqtl_df is not None:
            ieqtl_matched = match_variants_ieqtl(cima_df, ieqtl_df)

            if len(ieqtl_matched) > 0:
                # Use interaction effect (b_gi) for concordance
                metrics = compute_concordance(ieqtl_matched, 'eqtl_beta', 'ieqtl_beta_interaction', 'GTEx ieQTL')
                all_metrics['GTEx ieQTL'] = metrics
                matched_counts['GTEx ieQTL'] = len(ieqtl_matched)
                all_matched_dfs['ieqtl'] = ieqtl_matched
                log(f"  Concordance: {metrics['concordance']*100:.1f}%")
            else:
                log("  No matches found")
                all_metrics['GTEx ieQTL'] = {'concordance': 0.0, 'n_matched': 0, 'source': 'GTEx ieQTL'}
                matched_counts['GTEx ieQTL'] = 0
        else:
            all_metrics['GTEx ieQTL'] = {'concordance': 0.0, 'n_matched': 0, 'source': 'GTEx ieQTL'}
            matched_counts['GTEx ieQTL'] = 0

    except Exception as e:
        log(f"  ERROR: {e}")
        all_metrics['GTEx ieQTL'] = {'concordance': 0.0, 'n_matched': 0, 'source': 'GTEx ieQTL'}
        matched_counts['GTEx ieQTL'] = 0

    # =====================================================================
    # 3. DICE Cell-Type Specific eQTLs
    # =====================================================================
    log("\n" + "-" * 40)
    log("VALIDATION SOURCE 3: DICE Cell-Type eQTLs")
    log("-" * 40)

    try:
        # Extract cell type prefixes from CIMA cell types
        cell_type_prefixes = list(cell_types) if cell_types else None
        dice_df = load_dice_eqtls(cell_type_prefixes)

        if dice_df is not None:
            dice_matched = match_variants_dice(cima_df, dice_df)

            if len(dice_matched) > 0:
                metrics = compute_concordance(dice_matched, 'eqtl_beta', 'dice_beta', 'DICE')
                all_metrics['DICE'] = metrics
                matched_counts['DICE'] = len(dice_matched)
                all_matched_dfs['dice'] = dice_matched
                log(f"  Concordance: {metrics['concordance']*100:.1f}%")

                # Show cell type breakdown
                if 'dice_celltype' in dice_matched.columns:
                    ct_counts = dice_matched['dice_celltype'].value_counts()
                    log("  Matches by DICE cell type:")
                    for ct, count in ct_counts.head(5).items():
                        log(f"    {ct}: {count}")
            else:
                log("  No matches found")
                all_metrics['DICE'] = {'concordance': 0.0, 'n_matched': 0, 'source': 'DICE'}
                matched_counts['DICE'] = 0
        else:
            all_metrics['DICE'] = {'concordance': 0.0, 'n_matched': 0, 'source': 'DICE'}
            matched_counts['DICE'] = 0

    except Exception as e:
        log(f"  ERROR: {e}")
        all_metrics['DICE'] = {'concordance': 0.0, 'n_matched': 0, 'source': 'DICE'}
        matched_counts['DICE'] = 0

    # =====================================================================
    # Save Results
    # =====================================================================
    log("\n" + "=" * 60)
    log("SAVING RESULTS")
    log("=" * 60)

    # Save matched variants for each source
    for source_key, matched_df in all_matched_dfs.items():
        output_csv = OUTPUT_DIR / f'stage5_{source_key}_matched{suffix}.csv'
        matched_df.to_csv(output_csv, index=False)
        log(f"Saved: {output_csv}")

    # Save combined metrics
    metrics_json = OUTPUT_DIR / f'stage5_validation_metrics{suffix}.json'
    full_metrics = {
        'stage': 5,
        'description': 'Multi-source eQTL validation',
        'input': {
            'prioritized_variants': len(cima_df),
            'target_genes': len(target_genes),
            'cell_types': list(cell_types),
        },
        'output': {
            'sources_validated': len([m for m in all_metrics.values() if m.get('n_matched', 0) > 0]),
            'total_matched': sum(matched_counts.values()),
        },
        'by_source': all_metrics,
        'matched_counts': matched_counts,
    }

    with open(metrics_json, 'w') as f:
        json.dump(full_metrics, f, indent=2)
    log(f"Saved: {metrics_json}")

    # Generate report
    report = generate_report(cima_df, all_metrics, matched_counts)
    report_path = OUTPUT_DIR / f'stage5_report{suffix}.md'
    with open(report_path, 'w') as f:
        f.write(report)
    log(f"Saved: {report_path}")

    # Summary
    log("\n" + "=" * 60)
    log("VALIDATION SUMMARY")
    log("=" * 60)
    for source, metrics in all_metrics.items():
        n_matched = matched_counts.get(source, 0)
        concordance = metrics.get('concordance', 0) * 100
        log(f"  {source}: {n_matched} matches, {concordance:.1f}% concordance")

    log("\nStage 5 complete!")


if __name__ == '__main__':
    main()
