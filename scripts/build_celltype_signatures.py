#!/usr/bin/env python3
"""
Build Cell-Type-Specific Cytokine Signatures

This script creates cell-type-specific cytokine response signatures from the
CytoSig differential expression database. It uses semantic matching to harmonize
diverse cell type annotations into canonical categories.

Input:
    - diff.merge: Gene x Experiment matrix of differential expression values
    - meta_info: Experiment metadata with Treatment, Condition (cell type), Duration, Dose

Output:
    - Cell-type-specific signature matrices (genes x cytokines) for each cell type category
    - Mapping file showing how raw cell types map to canonical categories
    - Summary statistics

Author: Seongyong Park
Date: 2026-02-02
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

# ==============================================================================
# Configuration
# ==============================================================================

INPUT_DIR = Path('/data/parks34/projects/0sigdiscov/moran_i/results/cytosig')
OUTPUT_DIR = Path('/data/parks34/projects/2secactpy/results/celltype_signatures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum number of experiments required per cell type to include in signatures
MIN_EXPERIMENTS_PER_CELLTYPE = 5
# Minimum number of experiments required per cytokine within a cell type
MIN_EXPERIMENTS_PER_SIGNATURE = 3

# ==============================================================================
# Cell Type Semantic Mapping
# ==============================================================================

# Hierarchical cell type categories with pattern matching rules
# Order matters: more specific patterns should come before general ones

CELLTYPE_PATTERNS = {
    # === IMMUNE CELLS ===

    # Monocytes
    'Monocyte': [
        r'\bmonocyte\b',
        r'\bcd14\+?\s*mono',
        r'\bcd14pos.*mono',
        r'\bmono.*cd14',
        r'^mono$',  # Standalone "Mono"
    ],

    # Macrophages
    'Macrophage': [
        r'\bmacrophage\b',
        r'\balveolar\s+macro',
        r'\bactivated\s+macro',
        r'\bm1\s+macro',
        r'\bm2\s+macro',
        r'\bthp1\s+macro',
        r'\bwt\s+esc.*macro',
        r'^mac$',  # Standalone "Mac"
        r'\bm1\s+mdm',  # M1 monocyte-derived macrophage
        r'\bm2\s+mdm',
        r'\bmdm\b',
    ],

    # Dendritic cells
    'Dendritic_Cell': [
        r'\bdendritic\b',
        r'\bdc\b(?!.*cancer)',
        r'\bmdc\b',
        r'\bpdc\b',
        r'\bmodc\b',
    ],

    # T cells - CD4
    'T_CD4': [
        r'\bt\s*cd4\b',
        r'\bcd4\+?\s*t\b',
        r'\bcd4\s*helper',
        r'\bt\s*helper',
        r'\bth0\b',
        r'\bnaive\s*cd4',
        r'\bactivated.*cd4',
        r'\bcd4.*activated',
        r'\bcd4-positive',
        r'\bcd4\s*positive',
        r'\bmemory\s*t\s*cd4',
        r'\bt\s*cd4h\b',  # T CD4 helper
    ],

    # T cells - Th1
    'Th1': [
        r'\bth1\b',
    ],

    # T cells - Th2
    'Th2': [
        r'\bth2\b',
    ],

    # T cells - Th17
    'Th17': [
        r'\bth17\b',
    ],

    # T cells - Treg
    'Treg': [
        r'\btreg\b',
        r'\bcd4.*cd25\+',
        r'\bcd25\+.*cd4',
        r'\bregulatory\s*t',
    ],

    # T cells - CD8
    'T_CD8': [
        r'\bt\s*cd8\b',
        r'\bcd8\+?\s*t\b',
        r'\bcytotoxic\s*t',
        r'\bctl\b',
    ],

    # T cells - general
    'T_Cell': [
        r'^t$',  # Standalone "T"
        r'\bt\s*cell',
        r'\bt\s*lymph',
        r'\bjurkat',
        r'\bsupt1\b',
        r'\bt\s*leuk',
        r'\bprimary\s*t-cell',
    ],

    # B cells
    'B_Cell': [
        r'^b$',  # Standalone "B"
        r'\bb\s*cell',
        r'\bb\s*lymph',
        r'\bcd19\+',
        r'\bnaive\s*b\b',
        r'\bpurified\s*naive\s*b',
        r'\bplasma\s*cell',
        r'\bcd138\+',
        r'\blymphoma\b',
        r'\bmyeloma\b',
        r'\bleukaemia\b',
        r'\bleukemia\b',
        r'\bbcwm\b',
        r'\bbl2\b',
        r'\bsu-dhl',
    ],

    # NK cells
    'NK_Cell': [
        r'\bnk\s*cell',
        r'\bnk\b',
        r'\bnatural\s*killer',
    ],

    # ILC (Innate Lymphoid Cells)
    'ILC': [
        r'\bilc\d*\b',  # ILC2, ILC3, etc.
        r'\binnate\s*lymphoid',
    ],

    # Basophils
    'Basophil': [
        r'\bbasophil',
    ],

    # Neutrophils
    'Neutrophil': [
        r'\bneutrophil',
        r'\bpmn\b',  # Polymorphonuclear
        r'\bpolymorphonuclear',
    ],

    # Eosinophils
    'Eosinophil': [
        r'\beosinophil',
    ],

    # PBMC / Mixed immune
    'PBMC': [
        r'\bpbmc\b',
        r'\bperipheral\s*blood\s*mono',
        r'\bwhole\s*blood\b',
        r'\bblood\b(?!.*vessel)',
    ],

    # === EPITHELIAL CELLS ===

    # Airway/Bronchial epithelial
    'Airway_Epithelial': [
        r'\bairway\s*epi',
        r'\bbronch.*epi',
        r'\bbeas',
        r'\bnhbe\b',
        r'\bsmall\s*airway',
        r'\blung\s*epi',
        r'\balveolar.*epi',
    ],

    # Keratinocytes
    'Keratinocyte': [
        r'\bkeratinocyte',
        r'\bnhek\b',
        r'\bhacat\b',
        r'\bepiderm',
    ],

    # Intestinal epithelial
    'Intestinal_Epithelial': [
        r'\bcaco',
        r'\bht29\b',
        r'\bcolon.*epi',
        r'\bintestin.*epi',
        r'\bt84\b',
    ],

    # Hepatocytes
    'Hepatocyte': [
        r'\bhepatocyte',
        r'\bhepg2\b',
        r'\bhuh7',
        r'\bhep\s*g2',
        r'\bhep3b\b',  # Hepatoma line
        r'\bhep\s*3b',
        r'\bliver\b(?!.*fibro)',
        r'\bphh\b',  # Primary human hepatocyte
        r'\bihh\b',  # Immortalized human hepatocyte
        r'\bhepatoblast',
    ],

    # Renal epithelial
    'Renal_Epithelial': [
        r'\brptec\b',
        r'\bhk2\b',
        r'\brenal.*epi',
        r'\bkidney.*epi',
        r'\bhek293\b',  # Human embryonic kidney
        r'\bhek\s*293',
    ],

    # Retinal epithelial
    'Retinal_Epithelial': [
        r'\bretinal.*epi',
        r'\brpe\b',
    ],

    # General epithelial
    'Epithelial': [
        r'\bepi.*cell',
        r'\bepithelial',
        r'\bepithel',
    ],

    # === FIBROBLASTS ===

    # Dermal fibroblast
    'Dermal_Fibroblast': [
        r'\bdermal\s*fibro',
        r'\bskin\s*fibro',
        r'\bforeskin\s*fibro',
        r'\bforskin\s*fibro',
    ],

    # Lung fibroblast
    'Lung_Fibroblast': [
        r'\blung\s*fibro',
        r'\bpulmonary\s*fibro',
        r'\bimt90\b',
        r'\bccl210\b',
        r'\bccl171\b',
    ],

    # Synovial fibroblast
    'Synovial_Fibroblast': [
        r'\bsynovial\s*fibro',
        r'\bra\s*fibro',
        r'\bfls\b',
    ],

    # Cardiac fibroblast
    'Cardiac_Fibroblast': [
        r'\bcardiac\s*fibro',
        r'\bheart\s*fibro',
    ],

    # General fibroblast
    'Fibroblast': [
        r'\bfibroblast',
        r'\bfibro\b',
        r'\bbj\b',
        r'\bmrc5\b',
        r'\bwi38\b',
    ],

    # === ENDOTHELIAL CELLS ===

    # HUVEC
    'HUVEC': [
        r'\bhuvec\b',
        r'\bumbilic.*vein.*endo',
        r'\bumbilical.*endo',
    ],

    # Microvascular endothelial
    'Microvascular_Endothelial': [
        r'\bmicrovasc.*endo',
        r'\bhmvec\b',
        r'\bhdmec\b',
        r'\bhmec\d*\b',  # HMEC1
    ],

    # Arterial endothelial
    'Arterial_Endothelial': [
        r'\baort.*endo',
        r'\barteri.*endo',
        r'\bhcaec\b',
        r'\bhaec\b',  # Human aortic endothelial
        r'\bpulmonary\s*artery\s*endo',
    ],

    # Lymphatic endothelial
    'Lymphatic_Endothelial': [
        r'\blymphatic\s*endo',
        r'\bhlec\b',
    ],

    # General endothelial
    'Endothelial': [
        r'\bendothel',
        r'\bendo.*cell',
    ],

    # === SMOOTH MUSCLE ===

    'Smooth_Muscle': [
        r'\bsmooth\s*muscle',
        r'\bvsmc\b',
        r'\basm\b',
        r'\baosmc',
        r'\bhasmc',  # Human aortic SMC
        r'\bpasmc',  # Pulmonary artery SMC
    ],

    'Skeletal_Muscle': [
        r'\bskeletal\s*muscle',
        r'\bhskmcs\b',  # Human skeletal muscle cells
        r'\bmyoblast',
        r'\bmyotube',
        r'\bmyocyte\b',
    ],

    # === MESENCHYMAL / STROMAL ===

    'MSC': [
        r'\bmsc\b',
        r'\bmesenchymal\s*stem',
        r'\bmesenchymal\s*strom',
        r'\bstromal\s*cell',
        r'\bbone\s*marrow.*strom',
        r'\bbm\s*msc',
        r'\bhmsc\b',
        r'\bvertebral.*mesenchymal',
    ],

    # === ADIPOCYTES ===

    'Adipocyte': [
        r'\badipocyte',
        r'\badipos',
        r'\bsgbs\b',
    ],

    # === NEURAL ===

    'Neuron': [
        r'\bneuron',
        r'\bneuroblast',
        r'\bsh-sy5y',
        r'\bshsy5y',
        r'\bneuro\b',
    ],

    'Astrocyte': [
        r'\bastrocyte',
    ],

    'Microglia': [
        r'\bmicroglia',
    ],

    # === BONE / CARTILAGE ===

    'Osteoblast': [
        r'\bosteoblast',
    ],

    'Osteocyte': [
        r'\bosteocyte',
    ],

    'Osteosarcoma_Line': [
        r'\bu2os\b',
        r'\bosteosarcoma',
        r'\bsaos',
    ],

    'Chondrocyte': [
        r'\bchondrocyte',
    ],

    # === CANCER CELL LINES ===

    'Lung_Cancer_Line': [
        r'\ba549\b',
        r'\bh358\b',
        r'\bcalu',
        r'\bh1299\b',
        r'\bhcc827\b',  # Lung adenocarcinoma
        r'\blung\s*cancer',
        r'\bnsclc\b',
    ],

    'Breast_Cancer_Line': [
        r'\bmcf7\b',
        r'\bmcf-7\b',
        r'\bmcf10',
        r'\bmda.*mb',  # Covers MDA-MB231, MDA-MB468, etc.
        r'\bt47d\b',
        r'\bbt47',  # BT474, BT483
        r'\bbreast\s*cancer',
        r'\bsum52',
        r'\bbasal\s*breast',
        r'\btnbc\b',
    ],

    'Cervical_Cancer_Line': [
        r'\bhela\b',
        r'\bsiha\b',
        r'\bcaski\b',
    ],

    'Colorectal_Cancer_Line': [
        r'\bhct116\b',
        r'\bsw480\b',
        r'\bcolo\b',
        r'\bcolorectal.*cancer',
        r'\bcolon\s*cancer',
    ],

    'Melanoma_Line': [
        r'\bmelanoma',
        r'\bskmel',
        r'\bwm\d+',
        r'\ba375\b',
    ],

    'Ovarian_Cancer_Line': [
        r'\bskov',
        r'\bovarian.*cancer',
        r'\ba2780\b',
    ],

    'Prostate_Cancer_Line': [
        r'\blncap\b',
        r'\bpc3\b',
        r'\bdu145\b',
        r'\bprostate.*cancer',
    ],

    'Pancreatic_Cancer_Line': [
        r'\bpanc',
        r'\bbxpc',
        r'\baspc',
    ],

    'Leukemia_Line': [
        r'\bu937\b',
        r'\bthp1\b(?!.*macro)',
        r'\bk562\b',
        r'\bhl60\b',
        r'\bjurkat\b',
        r'\baml\b',
        r'\bcml\b',
    ],

    # === STEM CELLS ===

    'ESC': [
        r'\besc\b',
        r'\bembryonic\s*stem',
        r'\bhesc',
        r'\bipsc\b',
    ],

    # === OTHER ===

    'Trophoblast': [
        r'\btrophoblast',
        r'\bplacent',
    ],
}

# Cell type category hierarchy for grouping
CELLTYPE_HIERARCHY = {
    'Immune': [
        'Monocyte', 'Macrophage', 'Dendritic_Cell',
        'T_CD4', 'Th1', 'Th2', 'Th17', 'Treg', 'T_CD8', 'T_Cell',
        'B_Cell', 'NK_Cell', 'ILC', 'Basophil', 'Neutrophil', 'Eosinophil', 'PBMC'
    ],
    'Epithelial': [
        'Airway_Epithelial', 'Keratinocyte', 'Intestinal_Epithelial',
        'Hepatocyte', 'Renal_Epithelial', 'Retinal_Epithelial', 'Epithelial'
    ],
    'Fibroblast': [
        'Dermal_Fibroblast', 'Lung_Fibroblast', 'Synovial_Fibroblast',
        'Cardiac_Fibroblast', 'Fibroblast'
    ],
    'Endothelial': [
        'HUVEC', 'Microvascular_Endothelial', 'Arterial_Endothelial',
        'Lymphatic_Endothelial', 'Endothelial'
    ],
    'Muscle': ['Smooth_Muscle', 'Skeletal_Muscle'],
    'Stromal': ['MSC', 'Adipocyte'],
    'Neural': ['Neuron', 'Astrocyte', 'Microglia'],
    'Bone': ['Osteoblast', 'Osteocyte', 'Chondrocyte'],
    'Cancer_Line': [
        'Lung_Cancer_Line', 'Breast_Cancer_Line', 'Cervical_Cancer_Line',
        'Colorectal_Cancer_Line', 'Melanoma_Line', 'Ovarian_Cancer_Line',
        'Prostate_Cancer_Line', 'Pancreatic_Cancer_Line', 'Leukemia_Line',
        'Osteosarcoma_Line'
    ],
    'Stem': ['ESC'],
    'Other': ['Trophoblast']
}


def classify_celltype(raw_name: str) -> tuple:
    """
    Classify a raw cell type name into canonical category.

    Returns:
        (canonical_category, broad_category, confidence)
    """
    raw_lower = raw_name.lower().strip()

    for category, patterns in CELLTYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, raw_lower):
                # Find broad category
                broad = 'Unknown'
                for broad_cat, members in CELLTYPE_HIERARCHY.items():
                    if category in members:
                        broad = broad_cat
                        break
                return (category, broad, 'high')

    return ('Unknown', 'Unknown', 'low')


def load_data():
    """Load the diff.merge and meta_info files."""
    print("Loading data...")

    # Load meta_info
    meta = pd.read_csv(INPUT_DIR / 'meta_info', sep='\t', index_col=0)
    print(f"  Meta info: {meta.shape[0]} experiments")

    # Load diff.merge (large file - genes x experiments)
    # First row is header (experiment IDs), first column is gene names
    print("  Loading diff.merge (this may take a moment)...")
    diff = pd.read_csv(INPUT_DIR / 'diff.merge', sep='\t', index_col=0)
    print(f"  Diff matrix: {diff.shape[0]} genes x {diff.shape[1]} experiments")

    return meta, diff


def create_celltype_mapping(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Create mapping from raw cell types to canonical categories.
    """
    print("\nClassifying cell types...")

    # Get unique raw cell types from Condition column
    raw_celltypes = meta['Condition'].unique()
    print(f"  Found {len(raw_celltypes)} unique raw cell types")

    # Classify each
    mapping = []
    for raw in raw_celltypes:
        if pd.isna(raw):
            continue
        canonical, broad, confidence = classify_celltype(raw)
        mapping.append({
            'raw_celltype': raw,
            'canonical_celltype': canonical,
            'broad_category': broad,
            'confidence': confidence,
            'n_experiments': (meta['Condition'] == raw).sum()
        })

    mapping_df = pd.DataFrame(mapping)

    # Summary
    print(f"\n  Classification summary:")
    for broad in mapping_df['broad_category'].unique():
        subset = mapping_df[mapping_df['broad_category'] == broad]
        n_types = len(subset)
        n_exp = subset['n_experiments'].sum()
        print(f"    {broad}: {n_types} cell types, {n_exp} experiments")

    # Unknown
    unknown = mapping_df[mapping_df['canonical_celltype'] == 'Unknown']
    if len(unknown) > 0:
        print(f"\n  Unclassified cell types ({len(unknown)}):")
        for _, row in unknown.head(20).iterrows():
            print(f"    - {row['raw_celltype']} ({row['n_experiments']} exp)")
        if len(unknown) > 20:
            print(f"    ... and {len(unknown) - 20} more")

    return mapping_df


def build_celltype_signatures(meta: pd.DataFrame, diff: pd.DataFrame,
                               mapping: pd.DataFrame) -> dict:
    """
    Build cell-type-specific signature matrices.

    For each (cell_type, cytokine) pair, average the differential expression
    across all experiments matching that combination.

    Filters:
    - Cell types with < MIN_EXPERIMENTS_PER_CELLTYPE total experiments are excluded
    - Cytokine signatures with < MIN_EXPERIMENTS_PER_SIGNATURE experiments are excluded
    """
    print("\nBuilding cell-type-specific signatures...")
    print(f"  Minimum experiments per cell type: {MIN_EXPERIMENTS_PER_CELLTYPE}")
    print(f"  Minimum experiments per signature: {MIN_EXPERIMENTS_PER_SIGNATURE}")

    # Add canonical cell type to meta
    raw_to_canonical = dict(zip(mapping['raw_celltype'], mapping['canonical_celltype']))
    meta['canonical_celltype'] = meta['Condition'].map(raw_to_canonical)

    # Get unique cytokines (treatments) and cell types
    cytokines = meta['Treatment'].unique()
    cytokines = [c for c in cytokines if c != 'Condition' and pd.notna(c)]

    # Filter cell types by minimum experiment count
    celltype_counts = meta[meta['canonical_celltype'] != 'Unknown']['canonical_celltype'].value_counts()
    celltypes = [ct for ct in celltype_counts.index
                 if celltype_counts[ct] >= MIN_EXPERIMENTS_PER_CELLTYPE]

    excluded_celltypes = [ct for ct in celltype_counts.index
                          if celltype_counts[ct] < MIN_EXPERIMENTS_PER_CELLTYPE]
    if excluded_celltypes:
        print(f"  Excluded {len(excluded_celltypes)} cell types with < {MIN_EXPERIMENTS_PER_CELLTYPE} experiments:")
        for ct in excluded_celltypes[:10]:
            print(f"    - {ct} ({celltype_counts[ct]} experiments)")
        if len(excluded_celltypes) > 10:
            print(f"    ... and {len(excluded_celltypes) - 10} more")

    print(f"  {len(cytokines)} cytokines, {len(celltypes)} cell types (after filtering)")

    # Build signatures for each cell type
    signatures = {}

    for celltype in celltypes:
        # Get experiments for this cell type
        ct_meta = meta[meta['canonical_celltype'] == celltype]

        # For each cytokine, average the response
        ct_signatures = {}
        ct_counts = {}

        for cytokine in cytokines:
            # Get experiments for this cytokine + cell type
            exp_ids = ct_meta[ct_meta['Treatment'] == cytokine].index

            if len(exp_ids) == 0:
                continue

            # Get matching columns from diff matrix
            matching_cols = [col for col in exp_ids if col in diff.columns]

            # Filter by minimum experiments per signature
            if len(matching_cols) < MIN_EXPERIMENTS_PER_SIGNATURE:
                continue

            # Average expression across experiments
            avg_expr = diff[matching_cols].mean(axis=1)
            ct_signatures[cytokine] = avg_expr
            ct_counts[cytokine] = len(matching_cols)

        if ct_signatures:
            sig_df = pd.DataFrame(ct_signatures)
            signatures[celltype] = {
                'matrix': sig_df,
                'counts': ct_counts,
                'n_cytokines': len(ct_signatures),
                'n_experiments': sum(ct_counts.values())
            }
            print(f"    {celltype}: {len(ct_signatures)} cytokines from {sum(ct_counts.values())} experiments")

    return signatures


def save_signatures(signatures: dict, mapping: pd.DataFrame):
    """Save signature matrices and metadata."""
    print("\nSaving outputs...")

    # Save mapping
    mapping.to_csv(OUTPUT_DIR / 'celltype_mapping.csv', index=False)
    print(f"  Saved celltype_mapping.csv")

    # Save summary
    summary = []
    for celltype, data in signatures.items():
        summary.append({
            'celltype': celltype,
            'n_cytokines': data['n_cytokines'],
            'n_experiments': data['n_experiments'],
            'cytokines': ','.join(data['matrix'].columns.tolist())
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / 'signature_summary.csv', index=False)
    print(f"  Saved signature_summary.csv")

    # Save individual signature matrices
    sig_dir = OUTPUT_DIR / 'signatures'
    sig_dir.mkdir(exist_ok=True)

    for celltype, data in signatures.items():
        fname = f"{celltype}_signatures.csv"
        data['matrix'].to_csv(sig_dir / fname)
    print(f"  Saved {len(signatures)} signature matrices to signatures/")

    # Save combined matrix (all cell types as one file)
    # Columns: CellType_Cytokine
    combined_cols = {}
    for celltype, data in signatures.items():
        for cytokine in data['matrix'].columns:
            col_name = f"{celltype}__{cytokine}"
            combined_cols[col_name] = data['matrix'][cytokine]

    combined_df = pd.DataFrame(combined_cols)
    combined_df.to_csv(OUTPUT_DIR / 'celltype_cytokine_signatures.csv')
    print(f"  Saved combined matrix: {combined_df.shape[0]} genes x {combined_df.shape[1]} signatures")

    # Save metadata JSON
    metadata = {
        'n_celltypes': len(signatures),
        'n_total_signatures': sum(d['n_cytokines'] for d in signatures.values()),
        'celltypes': list(signatures.keys()),
        'cytokines': list(set(c for d in signatures.values() for c in d['matrix'].columns)),
        'celltype_details': {
            ct: {'n_cytokines': d['n_cytokines'], 'n_experiments': d['n_experiments']}
            for ct, d in signatures.items()
        }
    }
    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata.json")


def main():
    """Main execution."""
    print("=" * 60)
    print("Building Cell-Type-Specific Cytokine Signatures")
    print("=" * 60)

    # Load data
    meta, diff = load_data()

    # Create cell type mapping
    mapping = create_celltype_mapping(meta)

    # Build signatures
    signatures = build_celltype_signatures(meta, diff, mapping)

    # Save outputs
    save_signatures(signatures, mapping)

    print("\n" + "=" * 60)
    print("Complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
