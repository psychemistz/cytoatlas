# Signature Matrices

This document describes the signature matrices used for activity inference in the CytoAtlas project.

## Overview

| Signature | Proteins | Source | Primary Use |
|-----------|----------|--------|-------------|
| CytoSig | 44 | Jiang et al. | Cytokine response signatures |
| SecAct | 1,249 | Secreted protein annotations | Comprehensive secretome |

## CytoSig

### Description

CytoSig contains 44 cytokine response gene signatures derived from in vitro stimulation experiments. Each signature represents the transcriptional response to a specific cytokine.

### Loading

```python
from secactpy import load_cytosig

cytosig = load_cytosig()
print(f"Shape: {cytosig.shape}")  # (genes x 44)
print(f"Cytokines: {list(cytosig.columns)}")
```

### Cytokine List (44 signatures)

| Category | Cytokines |
|----------|-----------|
| Interleukins | IL1A, IL1B, IL2, IL4, IL6, IL7, IL10, IL12, IL13, IL15, IL17A, IL17F, IL18, IL21, IL22, IL23, IL27, IL33 |
| Interferons | IFNA, IFNB, IFNG |
| TNF Family | TNFA, TNFB, LTA, LTB, CD40L, TRAIL, TWEAK |
| TGF Family | TGFB1, TGFB2, TGFB3, BMP2, BMP4, BMP7 |
| Growth Factors | EGF, FGF2, VEGFA, PDGF, HGF, IGF1 |
| Chemokines | CXCL10, CCL2, CCL5 |
| Other | GM-CSF, M-CSF |

### Key Cytokines by Cell Type

| Cell Type | Expected High Cytokines |
|-----------|------------------------|
| Th1 | IFNG, IL12, TNFA |
| Th2 | IL4, IL13, IL5 |
| Th17 | IL17A, IL17F, IL22, IL23 |
| Treg | TGFB1, IL10 |
| CD8+ T | IFNG, TNFA, IL2 |
| NK | IFNG, TNFA |
| Monocyte | IL1B, IL6, TNFA |
| Macrophage (M1) | TNFA, IL1B, IL6, IFNG |
| Macrophage (M2) | IL10, TGFB1, IL4 |

## SecAct

### Description

SecAct contains 1,249 secreted protein signatures covering the comprehensive human secretome. These include cytokines, growth factors, hormones, enzymes, and other secreted proteins.

### Loading

```python
from secactpy import load_secact

secact = load_secact()
print(f"Shape: {secact.shape}")  # (genes x 1,249)
```

### Protein Categories

| Category | Count | Examples |
|----------|-------|----------|
| Cytokines | ~50 | IL1B, IL6, TNFA, IFNG |
| Growth Factors | ~100 | EGF, FGF, VEGF, PDGF |
| Chemokines | ~50 | CCL2, CXCL8, CXCL10 |
| Hormones | ~50 | Insulin, Leptin, Adiponectin |
| Proteases | ~200 | MMP1-28, ADAM family |
| ECM Proteins | ~150 | Collagens, Laminins |
| Complement | ~30 | C3, C4, C5 |
| Coagulation | ~30 | Fibrinogen, Thrombin |
| Enzymes | ~200 | Lipases, Phosphatases |
| Other | ~300 | Various secreted proteins |

### Notable Protein Families

#### Matrix Metalloproteinases (MMPs)
Involved in tissue remodeling and cancer invasion:
- MMP1, MMP2, MMP3, MMP7, MMP9, MMP13, MMP14

#### Complement System
Immune defense proteins:
- C1QA, C1QB, C1QC, C3, C4A, C4B, C5

#### Wnt Signaling
Developmental and cancer-related:
- WNT1-16, SFRP1-5, DKK1-4

#### TGF-β Superfamily
Growth and differentiation:
- TGFB1-3, BMP1-15, GDF1-15

## Activity Computation

### Ridge Regression

Both signature matrices are used in a ridge regression framework:

```python
from secactpy import ridge_batch

result = ridge_batch(
    X=signature.values,      # Signature matrix (genes x proteins)
    Y=expression.values,     # Expression matrix (genes x samples)
    lambda_=5e5,             # Regularization
    n_rand=1000,             # Permutations for p-values
    batch_size=10000,        # For large datasets
    backend='cupy'           # GPU acceleration
)
```

### Output Interpretation

| Output | Description | Interpretation |
|--------|-------------|----------------|
| `beta` | Activity coefficient | Strength of signature activity |
| `zscore` | Standardized activity | Primary metric for comparison |
| `pvalue` | Statistical significance | Confidence in activity estimate |
| `se` | Standard error | Uncertainty in estimate |

### Activity Z-scores

Activity values are z-scores with:
- Mean ≈ 0 across samples
- SD ≈ 1
- Range typically -3 to +3

**Important**: For differential analysis, use activity difference (not log2 fold change):
```python
activity_diff = group1_mean - group2_mean
```

## Gene Overlap

### Quality Metrics

For reliable activity inference, aim for:
- Gene overlap > 80% between expression data and signature
- At least 1,000 common genes with CytoSig
- At least 5,000 common genes with SecAct

### Checking Overlap

```python
# Check gene overlap
expr_genes = set(expression.index.str.upper())
sig_genes = set(signature.index.str.upper())
overlap = len(expr_genes & sig_genes)
coverage = overlap / len(sig_genes) * 100

print(f"Overlap: {overlap} genes ({coverage:.1f}%)")
```

## Validation

### Expected Patterns

CytoSig activities should show:
1. IL-17 high in Th17 cells
2. IFN-γ high in CD8+ T and NK cells
3. TNF-α high in activated monocytes
4. IL-10 high in Tregs and M2 macrophages
5. Type I IFN high in pDCs

### Correlation Validation

Cross-cohort validation typically shows:
- CytoSig: r > 0.9 between cohorts
- SecAct: r > 0.85 between cohorts

## Usage Notes

1. **Signature selection**: Use CytoSig for focused cytokine analysis, SecAct for comprehensive secretome profiling

2. **Preprocessing**: Expression should be log-transformed and mean-centered (differential expression)

3. **Batch effects**: Ridge regression is relatively robust but consider batch correction for multi-cohort analysis

4. **Multiple testing**: Apply FDR correction when comparing many signatures across conditions

## References

- CytoSig: Jiang P, et al. (2021) Systematic investigation of cytokine signaling activity at the tissue and single-cell levels. Nature Methods.
- SecAct: Internal signature matrix derived from comprehensive secreted protein annotations
