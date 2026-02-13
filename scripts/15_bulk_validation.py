#!/usr/bin/env python3
"""
Bulk RNA-seq Validation Pipeline (TCGA + GTEx).

Loads bulk expression data from GTEx v11 (parquet, ~19.8K samples) and
TCGA PanCancer (TSV, ~11K samples), maps gene IDs to HGNC symbols,
transforms with log2(x+1), runs ridge regression activity inference,
and saves H5AD files.

Falls back to TOIL-recomputed unified dataset if primary files are absent.

This validates that single-cell-derived cytokine signatures generalize
to independent bulk RNA-seq datasets.

Usage:
    # Both datasets
    python scripts/15_bulk_validation.py --dataset all

    # GTEx only
    python scripts/15_bulk_validation.py --dataset gtex

    # TCGA only
    python scripts/15_bulk_validation.py --dataset tcga

    # Force overwrite
    python scripts/15_bulk_validation.py --dataset all --force

    # Choose backend (auto detects GPU)
    python scripts/15_bulk_validation.py --dataset all --backend auto
"""

import argparse
import gc
import gzip
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/data/parks34/projects/1ridgesig/SecActpy")

from secactpy import load_cytosig, load_secact, ridge


# =============================================================================
# Paths
# =============================================================================

BULK_DATA_DIR = Path('/data/parks34/projects/2cytoatlas/data/bulk')
BASE_OUTPUT_DIR = Path('/data/parks34/projects/2cytoatlas/results/cross_sample_validation')

# Primary data files (preferred)
GTEX_V11_TPM = BULK_DATA_DIR / 'GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet'
TCGA_PANCAN = BULK_DATA_DIR / 'EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv'

# Fallback data files
TOIL_TPM = BULK_DATA_DIR / 'TcgaTargetGtex_rsem_gene_tpm.gz'

# Metadata / mapping
GTEX_V11_SAMPLE_ATTR = BULK_DATA_DIR / 'GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt'
PHENOTYPE_FILE = BULK_DATA_DIR / 'TcgaTargetGTEX_phenotype.txt.gz'
PROBEMAP = BULK_DATA_DIR / 'gencode.v23.annotation.gene.probemap'


def log(msg: str) -> None:
    """Print timestamped log message."""
    ts = time.strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


# =============================================================================
# Signature Loading (same as script 12)
# =============================================================================

_SIGNATURES_CACHE = None

def load_signatures() -> Dict[str, pd.DataFrame]:
    """Load all signature matrices (cached)."""
    global _SIGNATURES_CACHE
    if _SIGNATURES_CACHE is not None:
        return _SIGNATURES_CACHE

    log("Loading signature matrices...")
    cytosig = load_cytosig()
    secact = load_secact()

    lincytosig_path = Path("/data/parks34/projects/1ridgesig/SecActpy/secactpy/data/LinCytoSig.tsv.gz")
    with gzip.open(lincytosig_path, 'rt') as f:
        lincytosig = pd.read_csv(f, sep='\t', index_col=0)

    _SIGNATURES_CACHE = {
        'cytosig': cytosig,
        'lincytosig': lincytosig,
        'secact': secact,
    }
    log(f"  CytoSig: {cytosig.shape}, LinCytoSig: {lincytosig.shape}, SecAct: {secact.shape}")
    return _SIGNATURES_CACHE


# =============================================================================
# Gene ID Mapping
# =============================================================================

def load_probemap(probemap_path: Path) -> Dict[str, str]:
    """
    Load GENCODE v23 probemap: ENSG.version -> HGNC symbol.

    The probemap file has columns: id, gene, chrom, chromStart, chromEnd, strand
    where 'id' is ENSG.version and 'gene' is HGNC symbol.
    """
    log(f"Loading gene probemap: {probemap_path.name}")
    df = pd.read_csv(probemap_path, sep='\t')
    mapping = {}
    for _, row in df.iterrows():
        ensg_id = str(row['id'])
        gene_symbol = str(row['gene'])
        if gene_symbol and gene_symbol != 'nan':
            mapping[ensg_id] = gene_symbol
            # Also map without version suffix
            ensg_base = ensg_id.split('.')[0]
            if ensg_base not in mapping:
                mapping[ensg_base] = gene_symbol
    log(f"  {len(mapping)} gene ID mappings loaded")
    return mapping


# =============================================================================
# Data Loading
# =============================================================================

def _map_ensg_to_symbols(
    gene_ids: List[str],
    probemap: Dict[str, str],
    descriptions: Optional[pd.Series] = None,
) -> Tuple[List[str], List[int]]:
    """Map ENSG IDs to HGNC symbols using probemap and optional Description column.

    Returns all valid mappings including duplicates. Callers should use
    groupby(level=0).mean() to average duplicate gene symbols.
    """
    gene_symbols = []
    valid_rows = []

    for i, gid in enumerate(gene_ids):
        symbol = probemap.get(gid)
        if not symbol:
            symbol = probemap.get(gid.split('.')[0])
        if not symbol and descriptions is not None:
            desc = str(descriptions.iloc[i])
            if desc and desc != 'nan':
                symbol = desc
        if symbol and symbol != 'nan':
            gene_symbols.append(symbol)
            valid_rows.append(i)

    return gene_symbols, valid_rows


def load_gtex_v11(probemap: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load GTEx v11 TPM from parquet (19,788 samples, 74K genes).

    The parquet file has:
      - Index 'Name': versioned ENSG IDs (ENSG00000223972.6)
      - Column 'Description': gene symbols (DDX11L1)
      - Remaining columns: sample TPM values (float32)
    """
    log("Loading GTEx v11 TPM (parquet)...")
    t0 = time.time()

    tpm = pd.read_parquet(GTEX_V11_TPM)
    log(f"  Raw matrix: {tpm.shape} ({time.time() - t0:.1f}s)")

    # Extract Description column for gene symbol mapping
    descriptions = tpm['Description'] if 'Description' in tpm.columns else None
    tpm = tpm.drop(columns=['Description'], errors='ignore')

    # Map gene IDs to HGNC symbols (use Description first, probemap as backup)
    log("  Mapping gene IDs to symbols...")
    gene_symbols, valid_rows = _map_ensg_to_symbols(
        tpm.index.tolist(), probemap, descriptions
    )

    tpm = tpm.iloc[valid_rows]
    tpm.index = gene_symbols
    n_before = tpm.shape[0]
    n_dup = n_before - tpm.index.nunique()
    tpm = tpm.groupby(level=0).mean()
    log(f"  After gene mapping: {tpm.shape[0]} unique genes, {tpm.shape[1]} samples"
        f" ({n_dup} duplicates averaged)")

    # log2(TPM + 1)
    log("  Applying log2(TPM + 1) transform...")
    tpm = np.log2(tpm + 1)

    expr_df = tpm.T  # (samples x genes)

    # Build metadata from GTEx v11 sample attributes (full coverage)
    metadata = pd.DataFrame(index=expr_df.index)
    metadata['dataset'] = 'GTEX'

    if GTEX_V11_SAMPLE_ATTR.exists():
        log("  Loading GTEx v11 sample attributes...")
        attrs = pd.read_csv(GTEX_V11_SAMPLE_ATTR, sep='\t', low_memory=False,
                            usecols=['SAMPID', 'SMTS', 'SMTSD'])
        attrs = attrs.set_index('SAMPID')
        common = expr_df.index.intersection(attrs.index)
        if len(common) > 0:
            metadata.loc[common, 'tissue_type'] = attrs.loc[common, 'SMTS'].values
            metadata.loc[common, 'tissue_detail'] = attrs.loc[common, 'SMTSD'].values
            log(f"  Annotations matched for {len(common)}/{len(expr_df)} samples")
    elif PHENOTYPE_FILE.exists():
        log("  Falling back to TOIL phenotype metadata...")
        pheno = pd.read_csv(PHENOTYPE_FILE, sep='\t', compression='gzip', encoding='latin-1')
        pheno = pheno.set_index('sample')
        common = expr_df.index.intersection(pheno.index)
        if len(common) > 0:
            pheno_sub = pheno.loc[common].copy()
            if '_primary_site' in pheno_sub.columns:
                metadata.loc[common, 'tissue_type'] = pheno_sub['_primary_site'].values
            if 'detailed_category' in pheno_sub.columns:
                metadata.loc[common, 'tissue_detail'] = pheno_sub['detailed_category'].values
            log(f"  Phenotype matched for {len(common)}/{len(expr_df)} samples")

    metadata['tissue_type'] = metadata.get('tissue_type', pd.Series('Unknown', index=metadata.index)).fillna('Unknown')

    log(f"  Final: {expr_df.shape[0]} samples, {expr_df.shape[1]} genes")
    return expr_df, metadata


def load_tcga_pancan(probemap: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load TCGA PanCancer gene expression (11,069 samples, ~20.5K genes).

    The TSV file has:
      - Index 'gene_id': format 'symbol|entrezID' (e.g., 'TP53|7157')
      - Columns: TCGA barcode sample IDs
      - Values: EBPlusPlus batch-adjusted RSEM normalized counts
        (NOT TPM; column sums vary 759K-16.4M; ~0.55% values are negative
        from batch correction artifacts)
    """
    log("Loading TCGA PanCancer expression...")
    t0 = time.time()

    rsem = pd.read_csv(TCGA_PANCAN, sep='\t', index_col=0)
    log(f"  Raw matrix: {rsem.shape} ({time.time() - t0:.1f}s)")

    # Parse gene symbols from 'symbol|entrezID' index
    log("  Parsing gene symbols from symbol|entrezID format...")
    gene_symbols = []
    valid_rows = []

    for i, gid in enumerate(rsem.index):
        parts = str(gid).split('|')
        symbol = parts[0]
        if symbol and symbol != '?':
            gene_symbols.append(symbol)
            valid_rows.append(i)

    rsem = rsem.iloc[valid_rows]
    rsem.index = gene_symbols
    n_before = rsem.shape[0]
    n_dup = n_before - rsem.index.nunique()
    rsem = rsem.groupby(level=0).mean()
    log(f"  After gene mapping: {rsem.shape[0]} unique genes, {rsem.shape[1]} samples"
        f" ({n_dup} duplicates averaged)")

    # Clip negative values from EBPlusPlus batch correction artifacts
    n_neg = (rsem.values < 0).sum()
    if n_neg > 0:
        pct_neg = n_neg / rsem.values.size * 100
        log(f"  Clipping {n_neg} negative values ({pct_neg:.2f}%) from batch correction")
        rsem = rsem.clip(lower=0)

    # log2(x + 1) on clipped RSEM normalized counts
    log("  Applying log2(RSEM + 1) transform...")
    rsem = np.log2(rsem + 1)

    expr_df = rsem.T  # (samples x genes)

    # Build metadata from phenotype file
    # TCGA barcodes: PanCancer uses 7-part (TCGA-OR-A5J1-01A-11R-A29S-07),
    # phenotype uses 4-part (TCGA-OR-A5J1-01). Match on first 4 parts.
    metadata = pd.DataFrame(index=expr_df.index)
    metadata['dataset'] = 'TCGA'

    if PHENOTYPE_FILE.exists():
        log("  Loading phenotype metadata...")
        pheno = pd.read_csv(PHENOTYPE_FILE, sep='\t', compression='gzip', encoding='latin-1')
        pheno = pheno.set_index('sample')

        # Build mapping: 4-part barcode prefix -> phenotype row
        pheno_by_prefix = {}
        for sid in pheno.index:
            parts = str(sid).split('-')
            if len(parts) >= 4:
                prefix = '-'.join(parts[:4])
                pheno_by_prefix[prefix] = sid

        # Match PanCancer 7-part barcodes to phenotype 4-part barcodes
        # Expression: TCGA-OR-A5J1-01A-11R-A29S-07 (4th part has vial letter)
        # Phenotype:  TCGA-OR-A5J1-01              (4th part is 2-digit code only)
        matched = 0
        for sample_id in expr_df.index:
            parts = str(sample_id).split('-')
            if len(parts) >= 4:
                prefix = '-'.join(parts[:3] + [parts[3][:2]])
                if prefix in pheno_by_prefix:
                    pheno_sid = pheno_by_prefix[prefix]
                    row = pheno.loc[pheno_sid]
                    if '_primary_site' in pheno.columns:
                        metadata.loc[sample_id, 'cancer_site'] = row['_primary_site']
                    if 'detailed_category' in pheno.columns:
                        metadata.loc[sample_id, 'cancer_type'] = row['detailed_category']
                    matched += 1

        log(f"  Phenotype matched for {matched}/{len(expr_df)} samples")

    metadata['cancer_site'] = metadata.get('cancer_site', pd.Series('Unknown', index=metadata.index)).fillna('Unknown')
    metadata['cancer_type'] = metadata.get('cancer_type', pd.Series('Unknown', index=metadata.index)).fillna('Unknown')

    log(f"  Final: {expr_df.shape[0]} samples, {expr_df.shape[1]} genes")
    return expr_df, metadata


def load_toil_data(
    dataset: str,
    probemap: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load TOIL-recomputed TPM data for GTEx or TCGA (fallback).

    Args:
        dataset: 'gtex' or 'tcga'
        probemap: ENSG.version -> HGNC symbol mapping

    Returns:
        (expr_df, metadata_df) where expr_df is samples x genes (log2(TPM+1))
    """
    log(f"Loading TOIL combined TPM for {dataset} (fallback)...")

    t0 = time.time()
    tpm = pd.read_csv(TOIL_TPM, sep='\t', index_col=0, compression='gzip')
    log(f"  Raw matrix: {tpm.shape} ({time.time() - t0:.1f}s)")

    # Load phenotype metadata
    log("  Loading phenotype metadata...")
    pheno = pd.read_csv(PHENOTYPE_FILE, sep='\t', compression='gzip', encoding='latin-1')
    pheno = pheno.set_index('sample')

    # Filter samples by dataset
    if dataset == 'gtex':
        samples = [s for s in tpm.columns if s.startswith('GTEX-')]
        tpm = tpm[samples]
        log(f"  Filtered to {len(samples)} GTEx samples")
    elif dataset == 'tcga':
        samples = [s for s in tpm.columns if s.startswith('TCGA-')]
        tpm = tpm[samples]
        log(f"  Filtered to {len(samples)} TCGA samples")

    # Map gene IDs to HGNC symbols
    log("  Mapping gene IDs to symbols...")
    gene_symbols, valid_rows = _map_ensg_to_symbols(tpm.index.tolist(), probemap)

    tpm = tpm.iloc[valid_rows]
    tpm.index = gene_symbols
    n_before = tpm.shape[0]
    n_dup = n_before - tpm.index.nunique()
    tpm = tpm.groupby(level=0).mean()
    log(f"  After gene mapping: {tpm.shape[0]} unique genes ({n_dup} duplicates averaged)")

    # log2(TPM + 1)
    log("  Applying log2(TPM + 1) transform...")
    tpm = np.log2(tpm + 1)

    expr_df = tpm.T

    # Build metadata
    common_samples = expr_df.index.intersection(pheno.index)
    metadata = pheno.loc[common_samples].copy()

    if dataset == 'gtex':
        metadata = metadata.rename(columns={
            '_primary_site': 'tissue_type',
            '_study': 'study',
        })
        if 'detailed_category' in metadata.columns:
            metadata = metadata.rename(columns={'detailed_category': 'tissue_detail'})
    elif dataset == 'tcga':
        metadata = metadata.rename(columns={
            '_primary_site': 'cancer_site',
            '_study': 'study',
            'detailed_category': 'cancer_type',
        })

    metadata['dataset'] = dataset.upper()
    expr_df = expr_df.loc[common_samples]
    log(f"  Final: {expr_df.shape[0]} samples, {expr_df.shape[1]} genes")

    return expr_df, metadata


# =============================================================================
# Activity Inference
# =============================================================================

def run_activity_inference(
    expr_df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    data_format: str = 'TPM',
    force: bool = False,
    lambda_: float = 5e5,
    backend: str = 'auto',
) -> Dict[str, Path]:
    """
    Run activity inference on bulk expression data.

    For bulk data, each sample is one "donor" — no cell-type stratification.
    Mean-center genes across all samples, then run ridge regression.

    Args:
        expr_df: samples x genes DataFrame (log-transformed)
        output_dir: directory to save H5AD files
        dataset_name: 'gtex' or 'tcga'
        data_format: source data format for metadata tracking
            'TPM' for GTEx, 'EBPlusPlus_RSEM_normalized_counts' for TCGA
        force: overwrite existing files
        lambda_: ridge regularization parameter
        backend: 'auto', 'numpy', or 'cupy'

    Returns:
        Dict mapping signature_name -> output path
    """
    signatures = load_signatures()
    output_dir.mkdir(parents=True, exist_ok=True)

    gene_names = list(expr_df.columns)
    expr_matrix = expr_df.values  # (samples x genes)
    obs_df = pd.DataFrame(index=expr_df.index)

    output_paths = {}

    for sig_name, sig_matrix in signatures.items():
        act_path = output_dir / f"{dataset_name}_donor_only_{sig_name}.h5ad"

        if act_path.exists() and not force:
            log(f"  SKIP {sig_name} (exists): {act_path.name}")
            output_paths[sig_name] = act_path
            continue

        log(f"  Computing {sig_name}...")

        # Find common genes
        expr_genes = set(gene_names)
        sig_genes = set(sig_matrix.index)
        common = sorted(expr_genes & sig_genes)

        if len(common) < 100:
            log(f"    Too few common genes: {len(common)}, skipping")
            continue

        # Align expression to common genes
        gene_idx = [gene_names.index(g) for g in common]
        X = sig_matrix.loc[common].values.copy()  # (genes x targets)
        np.nan_to_num(X, copy=False, nan=0.0)

        t0 = time.time()

        # Mean-center across all samples, run ridge once
        Y = expr_matrix[:, gene_idx].T.astype(np.float64)  # (genes x samples)
        np.nan_to_num(Y, copy=False, nan=0.0)
        Y -= Y.mean(axis=1, keepdims=True)
        result = ridge(X, Y, lambda_=lambda_, n_rand=1000, backend=backend, verbose=False)
        activity = result['zscore'].T.astype(np.float32)

        elapsed = time.time() - t0

        # Create AnnData
        adata_act = ad.AnnData(
            X=activity,
            obs=obs_df.copy(),
            var=pd.DataFrame(index=list(sig_matrix.columns)),
        )
        adata_act.uns['common_genes'] = len(common)
        adata_act.uns['total_sig_genes'] = len(sig_genes)
        adata_act.uns['gene_coverage'] = len(common) / len(sig_genes)
        adata_act.uns['signature'] = sig_name
        adata_act.uns['source'] = f"{dataset_name}_donor_only_expression.h5ad"
        adata_act.uns['data_format'] = data_format
        if data_format == 'EBPlusPlus_RSEM_normalized_counts':
            adata_act.uns['transform'] = 'clip(0) -> log2(RSEM+1)'
        else:
            adata_act.uns['transform'] = f'log2({data_format}+1)'

        adata_act.write_h5ad(act_path, compression='gzip')
        output_paths[sig_name] = act_path
        log(f"    {sig_name}: {activity.shape}, {len(common)} genes, {elapsed:.1f}s -> {act_path.name}")

        del activity, adata_act, X, Y, result
        gc.collect()

    return output_paths


def run_stratified_activity_inference(
    expr_df: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    group_col: str,
    level_name: str,
    data_format: str = 'TPM',
    force: bool = False,
    lambda_: float = 5e5,
    backend: str = 'auto',
    min_samples: int = 30,
) -> Dict[str, Path]:
    """
    Run within-group stratified activity inference (Level 2 validation for bulk).

    For each tissue (GTEx) or cancer type (TCGA), mean-center genes within the
    group and run ridge regression separately. This asks: "Within lung samples,
    does the donor with higher IFNG expression also show higher IFNG activity?"

    Args:
        expr_df: samples x genes DataFrame (log-transformed)
        metadata: DataFrame with group_col annotation
        output_dir: directory to save H5AD files
        dataset_name: 'gtex' or 'tcga'
        group_col: column in metadata to group by ('tissue_type' or 'cancer_type')
        level_name: e.g. 'by_tissue' or 'by_cancer'
        data_format: source data format for metadata tracking
        force: overwrite existing files
        lambda_: ridge regularization parameter
        backend: 'auto', 'numpy', or 'cupy'
        min_samples: minimum samples per group (default 30)

    Returns:
        Dict mapping signature_name -> output path
    """
    signatures = load_signatures()
    output_dir.mkdir(parents=True, exist_ok=True)

    if group_col not in metadata.columns:
        log(f"  WARNING: '{group_col}' not in metadata, skipping stratified inference")
        return {}

    groups = metadata[group_col].dropna()
    groups = groups[groups != 'Unknown']
    group_counts = groups.value_counts()
    valid_groups = group_counts[group_counts >= min_samples].index.tolist()
    valid_mask = metadata[group_col].isin(valid_groups)

    log(f"\nStratified inference ({level_name}): {len(valid_groups)} groups "
        f"with >= {min_samples} samples (of {group_counts.shape[0]} total)")
    for g in sorted(valid_groups):
        log(f"  {g}: {group_counts[g]} samples")

    # Filter to valid samples
    expr_valid = expr_df.loc[valid_mask]
    meta_valid = metadata.loc[valid_mask].copy()

    # Save stratified expression H5AD
    pb_path = output_dir / f"{dataset_name}_{level_name}_expression.h5ad"
    if not pb_path.exists() or force:
        log(f"Saving stratified expression: {pb_path.name}")
        obs_df = meta_valid[[group_col]].copy()
        obs_df.columns = [group_col]
        adata_pb = ad.AnnData(
            X=expr_valid.values.astype(np.float32),
            obs=obs_df,
            var=pd.DataFrame(index=expr_valid.columns),
        )
        adata_pb.uns['data_format'] = data_format
        if data_format == 'EBPlusPlus_RSEM_normalized_counts':
            adata_pb.uns['transform'] = 'clip(0) -> log2(RSEM+1)'
        else:
            adata_pb.uns['transform'] = f'log2({data_format}+1)'
        adata_pb.uns['n_samples'] = expr_valid.shape[0]
        adata_pb.uns['n_groups'] = len(valid_groups)
        adata_pb.uns['group_col'] = group_col
        adata_pb.write_h5ad(pb_path, compression='gzip')
        log(f"  Saved: {pb_path.name} ({adata_pb.shape})")
        del adata_pb
    else:
        log(f"SKIP expression (exists): {pb_path.name}")

    # Run activity inference per group
    gene_names = list(expr_valid.columns)
    expr_matrix = expr_valid.values
    n_samples = expr_matrix.shape[0]
    group_labels = meta_valid[group_col].values

    output_paths = {}

    for sig_name, sig_matrix in signatures.items():
        act_path = output_dir / f"{dataset_name}_{level_name}_{sig_name}.h5ad"

        if act_path.exists() and not force:
            log(f"  SKIP {sig_name} (exists): {act_path.name}")
            output_paths[sig_name] = act_path
            continue

        log(f"  Computing {sig_name} (stratified by {group_col})...")

        # Find common genes
        expr_genes = set(gene_names)
        sig_genes = set(sig_matrix.index)
        common = sorted(expr_genes & sig_genes)

        if len(common) < 100:
            log(f"    Too few common genes: {len(common)}, skipping")
            continue

        gene_idx = [gene_names.index(g) for g in common]
        X = sig_matrix.loc[common].values.copy()  # (genes x targets)
        np.nan_to_num(X, copy=False, nan=0.0)

        n_targets = X.shape[1]
        activity = np.zeros((n_samples, n_targets), dtype=np.float32)

        t0 = time.time()

        # Run ridge per group with within-group mean centering
        for g in valid_groups:
            g_mask = (group_labels == g)
            n_g = g_mask.sum()
            if n_g < 3:
                continue

            Y_g = expr_matrix[g_mask][:, gene_idx].T.astype(np.float64)  # (genes x samples)
            np.nan_to_num(Y_g, copy=False, nan=0.0)
            Y_g -= Y_g.mean(axis=1, keepdims=True)  # within-group mean centering

            result_g = ridge(X, Y_g, lambda_=lambda_, n_rand=1000, backend=backend, verbose=False)
            activity[g_mask] = result_g['zscore'].T.astype(np.float32)

            del Y_g, result_g

        elapsed = time.time() - t0

        # Create AnnData
        obs_df = meta_valid[[group_col]].copy()
        obs_df.columns = [group_col]
        adata_act = ad.AnnData(
            X=activity,
            obs=obs_df,
            var=pd.DataFrame(index=list(sig_matrix.columns)),
        )
        adata_act.uns['common_genes'] = len(common)
        adata_act.uns['total_sig_genes'] = len(sig_genes)
        adata_act.uns['gene_coverage'] = len(common) / len(sig_genes)
        adata_act.uns['signature'] = sig_name
        adata_act.uns['source'] = f"{dataset_name}_{level_name}_expression.h5ad"
        adata_act.uns['data_format'] = data_format
        if data_format == 'EBPlusPlus_RSEM_normalized_counts':
            adata_act.uns['transform'] = 'clip(0) -> log2(RSEM+1)'
        else:
            adata_act.uns['transform'] = f'log2({data_format}+1)'
        adata_act.uns['group_col'] = group_col
        adata_act.uns['n_groups'] = len(valid_groups)
        adata_act.uns['stratified'] = True

        adata_act.write_h5ad(act_path, compression='gzip')
        output_paths[sig_name] = act_path
        log(f"    {sig_name}: {activity.shape}, {len(common)} genes, "
            f"{len(valid_groups)} groups, {elapsed:.1f}s -> {act_path.name}")

        del activity, adata_act, X
        gc.collect()

    return output_paths


# =============================================================================
# Main Pipeline
# =============================================================================

def run_dataset(
    dataset: str,
    force: bool = False,
    backend: str = 'auto',
) -> None:
    """Run full pipeline for one dataset (gtex or tcga)."""
    log(f"\n{'=' * 60}")
    log(f"DATASET: {dataset.upper()}")
    log(f"{'=' * 60}")

    output_dir = BASE_OUTPUT_DIR / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load gene ID mapping
    probemap = load_probemap(PROBEMAP)

    # Load expression data and determine data format
    # Prefer primary files (more samples); fall back to TOIL unified
    if dataset == 'gtex':
        data_format = 'TPM'
        if GTEX_V11_TPM.exists():
            expr_df, metadata = load_gtex_v11(probemap)
        elif TOIL_TPM.exists():
            expr_df, metadata = load_toil_data('gtex', probemap)
        else:
            log(f"ERROR: No GTEx data found. Need {GTEX_V11_TPM.name} or {TOIL_TPM.name}")
            return
    elif dataset == 'tcga':
        if TCGA_PANCAN.exists():
            data_format = 'EBPlusPlus_RSEM_normalized_counts'
            expr_df, metadata = load_tcga_pancan(probemap)
        elif TOIL_TPM.exists():
            data_format = 'TPM'  # TOIL recomputed as TPM
            expr_df, metadata = load_toil_data('tcga', probemap)
        else:
            log(f"ERROR: No TCGA data found. Need {TCGA_PANCAN.name} or {TOIL_TPM.name}")
            return
    else:
        log(f"ERROR: Unknown dataset '{dataset}'. Use 'gtex' or 'tcga'.")
        return

    log(f"  Data format: {data_format}")

    # Save bulk expression H5AD
    pb_path = output_dir / f"{dataset}_donor_only_expression.h5ad"
    if not pb_path.exists() or force:
        log(f"Saving bulk expression: {pb_path.name}")
        adata_pb = ad.AnnData(
            X=expr_df.values.astype(np.float32),
            obs=metadata,
            var=pd.DataFrame(index=expr_df.columns),
        )
        # Record format-specific metadata
        if dataset == 'gtex':
            adata_pb.uns['data_format'] = 'TPM'
            adata_pb.uns['transform'] = 'log2(TPM+1)'
        elif dataset == 'tcga':
            adata_pb.uns['data_format'] = 'EBPlusPlus_RSEM_normalized_counts'
            adata_pb.uns['transform'] = 'clip(0) -> log2(RSEM+1)'
        else:
            adata_pb.uns['data_format'] = 'unknown'
            adata_pb.uns['transform'] = 'log2(x+1)'
        adata_pb.uns['n_samples'] = expr_df.shape[0]
        adata_pb.uns['n_genes'] = expr_df.shape[1]
        adata_pb.write_h5ad(pb_path, compression='gzip')
        log(f"  Saved: {pb_path.name} ({adata_pb.shape})")
        del adata_pb
    else:
        log(f"SKIP expression (exists): {pb_path.name}")

    # Run activity inference
    log("\nRunning activity inference...")
    run_activity_inference(
        expr_df=expr_df,
        output_dir=output_dir,
        dataset_name=dataset,
        data_format=data_format,
        force=force,
        backend=backend,
    )

    # Run stratified activity inference (within-tissue / within-cancer-type)
    if dataset == 'gtex' and 'tissue_type' in metadata.columns:
        log("\n--- Stratified by tissue ---")
        run_stratified_activity_inference(
            expr_df=expr_df,
            metadata=metadata,
            output_dir=output_dir,
            dataset_name=dataset,
            group_col='tissue_type',
            level_name='by_tissue',
            data_format=data_format,
            force=force,
            backend=backend,
        )
    elif dataset == 'tcga' and 'cancer_type' in metadata.columns:
        log("\n--- Stratified by cancer type ---")
        run_stratified_activity_inference(
            expr_df=expr_df,
            metadata=metadata,
            output_dir=output_dir,
            dataset_name=dataset,
            group_col='cancer_type',
            level_name='by_cancer',
            data_format=data_format,
            force=force,
            backend=backend,
        )

    # Print summary
    if dataset == 'gtex' and 'tissue_type' in metadata.columns:
        n_tissues = metadata['tissue_type'].nunique()
        log(f"\nGTEx summary: {len(metadata)} samples, {n_tissues} tissue types")
    elif dataset == 'tcga':
        cancer_col = 'cancer_type' if 'cancer_type' in metadata.columns else 'cancer_site'
        if cancer_col in metadata.columns:
            n_types = metadata[cancer_col].nunique()
            log(f"\nTCGA summary: {len(metadata)} samples, {n_types} cancer types")

    del expr_df, metadata
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Bulk RNA-seq Validation Pipeline (TCGA + GTEx)"
    )
    parser.add_argument(
        '--dataset', nargs='+', default=['all'],
        help='Dataset(s) to process: gtex, tcga, or all (default: all)',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force overwrite existing files',
    )
    parser.add_argument(
        '--backend', default='auto', choices=['auto', 'numpy', 'cupy'],
        help='Computation backend (default: auto)',
    )

    args = parser.parse_args()

    # Check data directory
    if not BULK_DATA_DIR.exists():
        log(f"ERROR: Bulk data directory not found: {BULK_DATA_DIR}")
        log("Run scripts/15a_download_bulk_data.sh first.")
        sys.exit(1)

    if not PROBEMAP.exists():
        log(f"WARNING: Gene probemap not found: {PROBEMAP}")
        log("GTEx loading requires probemap. TCGA PanCancer can work without it.")

    # Determine datasets
    if 'all' in args.dataset:
        datasets = ['gtex', 'tcga']
    else:
        datasets = args.dataset

    log(f"Datasets: {datasets}")
    log(f"Backend: {args.backend}")
    log(f"Force: {args.force}")

    t_start = time.time()

    for dataset in datasets:
        run_dataset(dataset, force=args.force, backend=args.backend)

    elapsed = time.time() - t_start
    log(f"\n{'=' * 60}")
    log(f"ALL DONE ({elapsed / 60:.1f} min)")
    log(f"{'=' * 60}")

    # Verify output files
    for dataset in datasets:
        ddir = BASE_OUTPUT_DIR / dataset
        files = sorted(ddir.glob('*.h5ad'))
        log(f"\n{dataset} output files:")
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            log(f"  {f.name} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
