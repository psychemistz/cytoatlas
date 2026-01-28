# Pan-Disease Single-Cell Cytokine Activity Atlas

## Publication Target: Nature/Cell

**Title:** "A Pan-Disease Single-Cell Cytokine Activity Atlas Links Cell-Type-Specific Signaling to Treatment Response Across Inflammatory Diseases"

---

## Executive Summary

We present the **first comprehensive atlas of inferred cytokine and secreted protein signaling ACTIVITIES** (not expression) across 12+ million human immune cells spanning:
- **Healthy donors:** 6.5M cells from the Chinese Immune Multi-omics Atlas (CIMA)
- **20+ diseases:** 6.3M cells from the Inflammation Atlas
- **35 organs + cancers:** 6.4M cells from scAtlas (pre-computed)

### Key Claims for Publication
1. **Resource:** First pan-disease cytokine ACTIVITY atlas (complementary to gene expression atlases)
2. **Biology:** Conserved vs disease-specific cytokine programs identified across 20+ conditions
3. **Clinical:** Treatment response prediction from pre-treatment cytokine signatures (n=208 patients)
4. **Multi-omics:** Novel cytokine-metabolome correlations (CIMA: 1,549 metabolites)

### Why Nature/Cell?
- **Scale:** 12M+ cells across two major atlases + pre-computed scAtlas
- **Novelty:** Activity inference (not expression) at unprecedented scale
- **Clinical utility:** Treatment response prediction across multiple inflammatory diseases
- **Resource value:** Open data and tool (SecActpy)
- **Complementary to CIMA (Science 2026):** Different question (cytokine activity), same data

---

## Literature Context (2024-2026)

### Current Landscape - Key Gap Identified

| Atlas | Journal | Year | Focus | Gap |
|-------|---------|------|-------|-----|
| CIMA | Science | 2026 | Chinese healthy population, eQTLs, chromatin | No cytokine activity inference |
| Human Immune Health Atlas | Nature | 2025 | Aging, 16M cells | No disease, no cytokine activity |
| Anti-TNF IBD Atlas | Nature Immunol | 2024 | Treatment response, 1M cells | IBD only, no secretome-wide |
| IL-23 Psoriasis Atlas | Science Immunol | 2024 | Treatment response | Single disease, single cytokine |
| Pan-Cancer B Cell Atlas | Cell | 2024 | B cells across cancers | Cancer only, B cells only |

**No existing study has:**
1. Computed cytokine/secreted protein ACTIVITY across multiple large-scale atlases
2. Compared activities between healthy and 20+ disease conditions
3. Integrated single-cell cytokine activity with metabolomics/biochemistry
4. Built treatment response predictors from cytokine activity signatures

---

## Project Overview

This project computes cytokine and secreted protein activities across three large-scale single-cell RNA-seq atlases using the SecActpy package with GPU acceleration.

**Objectives:**
1. Compute cytokine activities using CytoSig signatures (44 cytokines)
2. Compute secreted protein activities using SecAct signatures (1,249 proteins)
3. Associate cytokine/secreted protein levels with patient clinical information
4. Perform comparative analysis across normal and disease conditions
5. **Build treatment response predictors** from pre-treatment cytokine activity signatures
6. **Identify conserved vs disease-specific** cytokine programs across 20+ conditions

---

## Data Sources Summary

| Atlas | Cells | Genes | Samples | Patient Info | Data Type |
|-------|-------|-------|---------|--------------|-----------|
| CIMA | 6,484,974 | 36,326 | 428 | Yes (biochemistry, metabolites) | Log-normalized |
| Inflammation Atlas | 6,340,934 | 22,826-37,124 | 1,047 | Yes (disease, treatment, response) | Raw counts |
| scAtlas Normal | 2,293,951 | 21,812 | Multiple | No (organ/cell type only) | Raw counts |
| scAtlas Cancer | 4,146,975 | 21,812 | Multiple | No (cancer type/tissue) | Raw counts |

---

## Dataset 1: CIMA Cell Atlas

### Files

```
/data/Jiang_Lab/Data/Seongyong/CIMA/
├── Cell_Atlas/
│   ├── CIMA_RNA_6484974cells_36326genes_compressed.h5ad (45 GB)
│   ├── CIMA_Sample_Blood_Biochemistry_Results.csv
│   ├── CIMA_Sample_Plasma_Metabolites_and_Lipids_Results.csv
│   └── CIMA_Cell_Type_Level_and_Marker.xlsx
└── Metadata/
    ├── CIMA_Sample_Information_Metadata.csv
    └── CIMA_Sample_Omics_Metadata.csv
```

### H5AD Structure

**Expression Data:**
- **Matrix (.X):** Log-normalized counts (log(counts+1)), sparse CSR, float32
- **Raw counts:** Available in `layers['counts']` as sparse integer matrix
- **Dimensions:** 6,484,974 cells × 36,326 genes
- **Gene format:** HGNC symbols (A1BG, A2M, TP53, etc.)

**Cell Metadata (.obs) - 41 columns:**

| Category | Columns | Description |
|----------|---------|-------------|
| **Linking** | `sample` | Sample ID (CIMA_H001, etc.) - KEY FOR JOINING |
| **Cell Types** | `cell_type_l1` | 6 broad types (B, CD4_T, CD8_T, Myeloid, ILC, HSPC) |
| | `cell_type_l2` | 27 intermediate types |
| | `cell_type_l3` | 38 subtypes |
| | `cell_type_l4` | 73 fine-grained types |
| **QC** | `n_counts`, `n_genes` | UMI counts (1000-25000), genes per cell |
| | `pct_counts_mt/hb/rb` | Mitochondrial, hemoglobin, ribosomal % |
| **Demographics** | `Age`, `Sex`, `Height`, `Weight`, `BMI`, `Blood_Type` | Patient characteristics |
| **Lifestyle** | `Smoking`, `Alcohol`, `Sleep_Duration`, `Exercise`, `Mood` | Behavioral factors |

### Metadata Files

**1. CIMA_Sample_Information_Metadata.csv** (428 samples, 28 columns)
- Demographics: Sex, Age, Height, Weight, BMI, Blood_Type
- Lifestyle: Smoking, Alcohol, Sleep_Duration, Exercise, Mood
- Geographic: City, Province
- **Linking column:** `Sample_name`

**2. CIMA_Sample_Blood_Biochemistry_Results.csv** (399 samples, 20 columns)
- Liver: ALT, AST, DBIL, IBIL, Tbil, ALB, GLOB, GGT
- Kidney: Cr, UR, UA
- Lipids: CHOL, TG, LDL-C, HDL-C
- Other: TP, GLU
- **Linking column:** `Sample`

**3. CIMA_Sample_Plasma_Metabolites_and_Lipids_Results.csv** (390 samples, 1550 columns)
- 1,549 metabolites/lipids including amino acids, organic acids, lipid species
- **Linking column:** `sample` (lowercase)

**4. CIMA_Cell_Type_Level_and_Marker.xlsx** (73 rows, 20 columns)
- Hierarchical cell type definitions (L1→L4)
- Gene markers for each cell type
- Cell ontology IDs

### Data Linking Strategy

```python
# Linking key: h5ad.obs["sample"] → metadata["Sample_name" or "Sample"]
h5ad.obs["sample"] ──→ CIMA_Sample_Information_Metadata["Sample_name"]
                   ──→ CIMA_Sample_Omics_Metadata["Sample"]
                   ──→ Blood_Biochemistry["Sample"] (399/428 samples)
                   ──→ Plasma_Metabolites["sample"] (390/428 samples)
```

### Analysis Considerations

1. **Data is log-normalized** - may need to use raw counts from `layers['counts']` for SecActpy
2. **Sample coverage:** 375 samples have complete data (scRNA + biochemistry + metabolites)
3. **Cell type resolution:** Use L2 or L3 for balanced granularity
4. **Association analysis:** Can correlate cytokine activities with:
   - Blood biochemistry (liver/kidney function, lipid profiles)
   - Metabolite levels (1,549 features)
   - Demographics and lifestyle factors

---

## Dataset 2: Inflammation Atlas

### Files

```
/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/
├── INFLAMMATION_ATLAS_main_afterQC.h5ad (17.2 GB, 4,918,140 cells)
├── INFLAMMATION_ATLAS_validation_afterQC.h5ad (2.4 GB, 849,922 cells)
├── INFLAMMATION_ATLAS_external_afterQC.h5ad (2.0 GB, 572,872 cells)
└── INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv (1,047 samples)
```

### H5AD Structure

| Dataset | Cells | Genes | Samples | Purpose |
|---------|-------|-------|---------|---------|
| **main** | 4,918,140 | 22,826 | 817 | Primary analysis |
| **validation** | 849,922 | 22,826 | 144 | Independent validation |
| **external** | 572,872 | 37,124 | 86 | External benchmarking |

**Expression Data:**
- **Matrix (.X):** RAW COUNTS (integers), sparse CSR
- **Gene format:** HGNC symbols with Ensembl IDs available
- **Note:** External dataset has larger gene set (37,124 vs 22,826)

**Cell Metadata (.obs) - 5 columns:**

| Column | Values | Description |
|--------|--------|-------------|
| `cellID` | Unique | Cell barcode |
| `sampleID` | 817/144/86 | **KEY FOR JOINING** to sample metadata |
| `libraryID` | 484 | Sequencing library ID |
| `Level1` | 17 types | Broad cell types (T cells, B cells, Monocytes, etc.) |
| `Level2` | 66 types | Fine cell subtypes |

**Gene Metadata (.var) - 12 columns:**
- `symbol`, `ensembl_gene_id`, `hgnc_id`
- Gene flags: `hb`, `mt`, `ribo`, `plt` (boolean)
- QC: `n_cells_by_counts`, `total_counts`

### Sample Metadata

**INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv** (1,047 samples, 18 columns)

| Column | Values | Missing | Description |
|--------|--------|---------|-------------|
| `sampleID` | 1,047 unique | 0 | **KEY FOR JOINING** |
| `patientID` | 1,047 unique | 0 | Patient identifier |
| `disease` | 20 diseases | 0 | RA, PSA, CD, COPD, COVID, etc. |
| `diseaseGroup` | 6 groups | 0 | IMIDs, respiratory, cancer, etc. |
| `diseaseStatus` | 59 values | 0 | Disease-specific status |
| `treatmentStatus` | 3 values | 0 | Treatment status |
| `therapyResponse` | R/NR/na | 839 (80%) | Response to therapy |
| `sex` | M/F/na | 0 | Biological sex |
| `age` | 8-92 years | 161 (15%) | Patient age |
| `BMI` | Numeric | 810 (77%) | Body mass index |
| `smokingStatus` | 4 values | 0 | Smoking history |
| `ethnicity` | 4 values | 0 | Ethnic background |
| `studyID` | 25 studies | 0 | Study/cohort identifier |

**Disease Distribution:**
- Inflammatory: RA (rheumatoid arthritis), PSA (psoriatic arthritis), PS (psoriasis), SLE (lupus)
- Gastrointestinal: CD (Crohn's disease), UC (ulcerative colitis)
- Respiratory: COPD, asthma, COVID, flu
- Cancer: CRC (colorectal), HNSCC, NPC
- Infectious: HBV, HIV, sepsis
- Other: MS (multiple sclerosis), cirrhosis, healthy

### Data Linking Strategy

```python
# Link cells to sample metadata
h5ad.obs["sampleID"] ──→ sampleMetadata["sampleID"]

# Example code:
import pandas as pd
import anndata as ad

metadata = pd.read_csv("INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv")
adata = ad.read_h5ad("INFLAMMATION_ATLAS_main_afterQC.h5ad", backed='r')
adata.obs = adata.obs.merge(metadata, on='sampleID', how='left')
```

### Analysis Considerations

1. **Raw counts available** - ideal for SecActpy input
2. **Disease-rich dataset** - 20 diseases across inflammatory/immune conditions
3. **Treatment response data** - available for ~20% of samples (208 with R/NR)
4. **Multi-cohort design** - main/validation/external for robust analysis
5. **Gene overlap** - handle different gene sets between main and external

---

## Dataset 3: scAtlas (Normal + Cancer)

### Files

```
/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/
├── igt_s9_fine_counts.h5ad (Normal organs, 2.3M cells)
├── PanCancer_igt_s9_fine_counts.h5ad (Cancer, 4.1M cells)
├── igt_s9_fine_CytoSig_activity.h5ad (Pre-computed, 44 features)
├── igt_s9_fine_SecAct_activity.h5ad (Pre-computed, 1,249 features)
├── PanCancer_igt_s9_fine_CytoSig_activity.h5ad (Pre-computed)
└── PanCancer_igt_s9_fine_SecAct_activity.h5ad (Pre-computed)
```

### Normal Organs Atlas

**igt_s9_fine_counts.h5ad:**
- **Cells:** 2,293,951
- **Genes:** 21,812
- **Data:** Raw counts, sparse matrix

**Cell Metadata (.obs) - 33 columns:**

| Category | Columns | Description |
|----------|---------|-------------|
| **Tissue** | `tissue` | 35 organs (Bladder, Blood, BoneMarrow, Breast, Heart, etc.) |
| | `region`, `system` | Anatomical region, body system |
| **Cell Types** | `cellType1` | 468 coarse types |
| | `cellType2` | 1,069 fine subtypes with tissue context |
| | `majorCluster`, `subCluster` | Clustering annotations |
| **Sample** | `cohortID`, `datasetID`, `sampleID`, `donorID` | Sample identifiers |
| **Donor** | `sex`, `age` | Demographics (limited) |
| **QC** | `doublet_score`, `n_genes_by_counts`, `pct_counts_mt` | Quality metrics |

### PanCancer Atlas

**PanCancer_igt_s9_fine_counts.h5ad:**
- **Cells:** 4,146,975
- **Genes:** 21,812 (same as normal)
- **Data:** Raw counts, sparse matrix

**Additional Cancer Metadata (.obs) - 39 columns:**

| Column | Description |
|--------|-------------|
| `cancerType` | Primary cancer type |
| `sub_cancerType` | Cancer subtype |
| `tissue` | 6 values: Tumor, Adjacent, Blood, Metastasis, PreLesion, PleuralFluids |
| `tumorPhase`, `TNM` | Tumor staging |
| `recurrence` | Recurrence status |
| `treatment`, `treatmentResponse`, `treatmentPhase` | Treatment info |

### Pre-computed Activity Files (ALREADY AVAILABLE)

| File | Cells | Features | Content |
|------|-------|----------|---------|
| `igt_s9_fine_CytoSig_activity.h5ad` | 2,293,951 | 44 | Cytokine activities (normal) |
| `igt_s9_fine_SecAct_activity.h5ad` | 2,293,951 | 1,249 | Secreted protein activities (normal) |
| `PanCancer_igt_s9_fine_CytoSig_activity.h5ad` | 4,146,975 | 44 | Cytokine activities (cancer) |
| `PanCancer_igt_s9_fine_SecAct_activity.h5ad` | 4,146,975 | 1,249 | Secreted protein activities (cancer) |

**CytoSig signatures (44):** Activin A, BDNF, BMP2, BMP4, BMP6, CD40L, CXCL12, EGF, FGF2, GCSF, GDF11, GMCSF, HGF, IFN1, IFNG, IL10, IL12A, IL13, IL17, IL2, IL4, IL6, IL9, LIF, LTA, TNF, VEGF, etc.

**SecAct signatures (1,249):** Comprehensive secreted proteins including A1BG, A2M, ADAM family, ADAMTS family, cytokines, growth factors, etc.

### Analysis Considerations

1. **Pre-computed activities available** - can directly use for downstream analysis
2. **No patient-level metadata** - analysis focuses on organ/cell type comparisons
3. **Normal vs Cancer comparison** - same gene set enables direct comparison
4. **Rich cell type annotations** - 468 coarse and 1,069 fine cell types

---

## SecActpy Package Usage

### Installation

```bash
pip install secactpy           # CPU only
pip install secactpy[gpu]      # With GPU support (CuPy)
```

### Package Location

```
/vf/users/parks34/projects/1ridgesig/SecActpy/
```

### Core Functions

#### 1. Single-Cell Analysis (Recommended for this project)

```python
from secactpy import secact_activity_inference_scrnaseq

# Cell-type level (pseudo-bulk) analysis
result = secact_activity_inference_scrnaseq(
    adata="path/to/data.h5ad",
    cell_type_col="cell_type_l2",       # Column for cell type grouping
    is_single_cell_level=False,          # Pseudo-bulk by cell type
    sig_matrix="cytosig",                # or "secact"
    backend="cupy",                       # GPU acceleration
    n_rand=1000,
    verbose=True
)

# Single-cell level analysis (memory intensive)
result = secact_activity_inference_scrnaseq(
    adata=adata,
    cell_type_col="cell_type_l2",
    is_single_cell_level=True,           # Per-cell activities
    sig_matrix="secact",
    backend="cupy",
    n_rand=1000
)
```

#### 2. Batch Processing for Large Datasets

```python
from secactpy import ridge_batch, load_cytosig, load_secact
import anndata as ad
import scipy.sparse as sp

# Load signature
sig = load_cytosig()  # or load_secact()

# Load expression data
adata = ad.read_h5ad("data.h5ad", backed='r')

# For CIMA (log-normalized), use raw counts layer
expr = adata.layers['counts']  # sparse matrix

# For Inflammation/scAtlas (raw counts), use .X directly
expr = adata.X

# Batch processing with GPU
result = ridge_batch(
    X=sig.values,                         # Signature matrix
    Y=expr.T,                             # Expression (genes x cells), transposed
    batch_size=10000,                     # Process 10k cells at a time
    n_rand=1000,
    backend='cupy',                       # GPU
    output_path="results.h5ad",           # Stream to disk
    verbose=True
)
```

#### 3. Load Built-in Signatures

```python
from secactpy import load_cytosig, load_secact, list_signatures

# CytoSig: 44 cytokines
cytosig = load_cytosig()
print(f"CytoSig: {cytosig.shape}")  # (genes, 44)

# SecAct: 1,249 secreted proteins
secact = load_secact()
print(f"SecAct: {secact.shape}")  # (genes, 1249)

# List all available
print(list_signatures())
```

### Output Format

```python
result = {
    'beta': DataFrame,      # Activity estimates (proteins × samples/cells)
    'se': DataFrame,        # Standard errors
    'zscore': DataFrame,    # Z-scores (significance)
    'pvalue': DataFrame,    # P-values from permutation test
    'n_genes': int,         # Genes used in analysis
    'genes': list,          # Gene names
    'method': str,          # "numpy" or "cupy"
    'time': float           # Execution time (seconds)
}
```

### GPU Acceleration

```python
# Check GPU availability
from secactpy import CUPY_AVAILABLE
print(f"GPU available: {CUPY_AVAILABLE}")

# Use GPU
result = secact_activity(expr, sig, backend='cupy')

# Performance: 10-34x speedup depending on dataset size
```

### Memory Estimation

```python
from secactpy import estimate_memory, estimate_batch_size

# Estimate memory for analysis
mem = estimate_memory(
    n_genes=21812,
    n_features=1249,      # SecAct
    n_samples=2_000_000,  # 2M cells
    n_rand=1000,
    batch_size=10000
)
print(f"Total: {mem['total']:.1f} GB, Per batch: {mem['per_batch']:.1f} GB")

# Estimate optimal batch size
batch_size = estimate_batch_size(n_genes=21812, n_features=1249)
```

---

## Publication Figures Plan

### Main Figures (6-8 figures)

1. **Figure 1 - Overview Schema**
   - Multi-atlas integration diagram with cell counts
   - Study design and workflow
   - Data summary statistics

2. **Figure 2 - Pan-Disease Cytokine Activity Landscape**
   - Heatmap: Cell types × cytokines across all diseases
   - UMAP colored by cytokine activity
   - Disease-specific activity profiles

3. **Figure 3 - Conserved vs Disease-Specific Programs**
   - Venn diagram / UpSet plot of shared signatures
   - Hierarchical clustering of diseases by cytokine profiles
   - Core inflammatory module identification

4. **Figure 4 - Cell-Type Resolution of Cytokine Signaling**
   - Cell-type specific activity heatmaps
   - Identification of responding cell populations per disease
   - Comparison across atlases

5. **Figure 5 - Treatment Response Prediction**
   - ROC curves for response prediction across diseases
   - Key predictive signatures per disease (RA, PSA, CD, UC, PS)
   - Cell types driving predictive signatures

6. **Figure 6 - Cytokine-Metabolome Axis (CIMA)**
   - Correlation network: cytokine activity ↔ metabolites
   - Top cytokine-metabolite associations
   - Biological pathway enrichment

7. **Figure 7 - Normal vs Disease Comparison**
   - CIMA healthy vs Inflammation Atlas healthy
   - Baseline activity differences
   - Population-specific signatures

### Extended Data / Supplementary

- ED1: Data quality and validation metrics
- ED2: Full cytokine × cell type × disease matrix
- ED3: Cross-cohort validation results
- ED4: Complete metabolite correlation results
- ED5: Treatment response per disease details

---

## Analysis Plan

### Analysis Approach

**Two levels of analysis:**
1. **Pseudo-bulk** - Aggregate expression by cell type first, then compute activities (fast, robust)
2. **Single-cell** - Compute per-cell activities (detailed, memory intensive)

**Two statistical approaches:**
1. **Correlation analysis** - Spearman correlations for continuous variables
2. **Differential analysis** - Wilcoxon rank-sum tests for categorical variables
3. Multiple testing correction using Benjamini-Hochberg FDR (q < 0.05)

**Treatment response prediction:**
1. Logistic regression / Random Forest classifiers
2. Leave-one-out cross-validation within diseases
3. Cross-disease generalization testing

### Phase 1: CIMA Atlas Analysis

**Objective:** Compute cytokine/secreted protein activities and associate with patient clinical data

**Steps:**
1. Load CIMA h5ad using backed mode
2. Extract raw counts from `layers['counts']`
3. **Pseudo-bulk analysis:**
   - Aggregate by cell type (L2) and sample
   - Compute CytoSig (44) and SecAct (1,249) activities
4. **Single-cell analysis:**
   - Compute per-cell activities using GPU batch processing
5. Merge with metadata:
   - Blood biochemistry (19 markers, 399 samples)
   - Plasma metabolites (1,549 features, 390 samples)
   - Demographics and lifestyle
6. Statistical analysis:
   - Correlations: activity vs biochemistry/metabolites (continuous)
   - Differential: activity by sex, smoking status, blood type (categorical)
   - Cell type-specific signature identification

**Output files:**
- `CIMA_CytoSig_pseudobulk.h5ad`
- `CIMA_SecAct_pseudobulk.h5ad`
- `CIMA_CytoSig_singlecell.h5ad`
- `CIMA_SecAct_singlecell.h5ad`
- `CIMA_correlation_biochemistry.csv`
- `CIMA_correlation_metabolites.csv`
- `CIMA_differential_demographics.csv`

### Phase 2: Inflammation Atlas Analysis

**Objective:** Compute activities and associate with disease/treatment response

**Steps:**
1. Process each dataset (main: 4.9M, validation: 850K, external: 573K)
2. Use raw counts directly from .X
3. **Pseudo-bulk analysis:**
   - Aggregate by cell type (Level2) and sample
   - Compute CytoSig and SecAct activities
4. **Single-cell analysis:**
   - Compute per-cell activities using GPU batch processing
5. Merge with sample metadata (disease, treatment, response)
6. Statistical analysis:
   - Differential: disease groups (20 diseases, 6 groups)
   - Differential: treatment response (R vs NR, 208 samples)
   - Correlations: activity vs age, BMI
7. Cross-validation using validation/external cohorts

**Output files:**
- `InflamAtlas_main_CytoSig_pseudobulk.h5ad`
- `InflamAtlas_main_SecAct_pseudobulk.h5ad`
- `InflamAtlas_main_CytoSig_singlecell.h5ad`
- `InflamAtlas_main_SecAct_singlecell.h5ad`
- `InflamAtlas_disease_differential.csv`
- `InflamAtlas_treatment_response.csv`
- `InflamAtlas_validation_comparison.csv`

### Phase 3: scAtlas Analysis

**Objective:** Use pre-computed activities for organ/cancer comparisons

**Steps:**
1. Load pre-computed activity files (already available)
2. Analyze normal organ signatures:
   - Organ-specific cytokine profiles
   - Cell type-specific activities across organs
3. Analyze cancer signatures:
   - Tumor vs adjacent tissue
   - Cancer type-specific profiles
   - Metastasis signatures
4. Compare normal vs cancer:
   - Matched organ comparisons
   - Pan-cancer common signatures

**Output files:**
- `scAtlas_organ_signatures.csv`
- `scAtlas_cancer_signatures.csv`
- `scAtlas_normal_vs_cancer.csv`

### Phase 4: Integrated Analysis

**Objective:** Cross-atlas comparisons and validation

**Steps:**
1. Harmonize cell type annotations across atlases
2. Compare cytokine profiles:
   - Healthy (CIMA) vs healthy (Inflammation Atlas)
   - Disease-specific signatures across atlases
3. Validate findings:
   - Consistency of cell type signatures
   - Reproducibility across cohorts

---

## Code Templates

### Template 1: Load H5AD with Backed Mode

```python
import anndata as ad
import h5py

# For exploration (metadata only)
with h5py.File("data.h5ad", 'r') as f:
    obs_keys = list(f['obs'].keys())
    var_keys = list(f['var'].keys())
    n_cells = f['X'].shape[0]
    n_genes = f['X'].shape[1]

# For analysis (backed mode)
adata = ad.read_h5ad("data.h5ad", backed='r')
```

### Template 2: Compute Activities with GPU

```python
from secactpy import ridge_batch, load_cytosig, load_secact
import anndata as ad

# Load data
adata = ad.read_h5ad("data.h5ad", backed='r')

# Get expression matrix (genes x cells)
# For CIMA: use layers['counts']
# For others: use .X
expr = adata.X.T  # Transpose to genes x cells

# Load signature
sig = load_cytosig()

# Compute with GPU
result = ridge_batch(
    X=sig.values,
    Y=expr,
    batch_size=10000,
    n_rand=1000,
    backend='cupy',
    output_path="output_activity.h5ad",
    verbose=True
)
```

### Template 3: Aggregate by Cell Type

```python
import pandas as pd
import numpy as np

# Get cell type assignments
cell_types = adata.obs['cell_type_l2'].values

# Aggregate expression by cell type (pseudo-bulk)
unique_types = np.unique(cell_types)
aggregated = {}
for ct in unique_types:
    mask = cell_types == ct
    aggregated[ct] = adata.X[mask].mean(axis=0)

pseudo_bulk = pd.DataFrame(aggregated, index=adata.var_names)
```

### Template 4: Merge with Metadata

```python
import pandas as pd

# Load sample-level activities
activities = pd.read_csv("sample_activities.csv", index_col=0)

# Load metadata
metadata = pd.read_csv("sample_metadata.csv")
biochemistry = pd.read_csv("blood_biochemistry.csv")

# Merge
merged = activities.merge(metadata, left_index=True, right_on='Sample')
merged = merged.merge(biochemistry, on='Sample', how='left')
```

---

## File Structure

```
/data/parks34/projects/2secactpy/
├── README.md                    # This file (publication-focused)
├── scripts/
│   ├── 00_pilot_analysis.py     # Exploratory pilot on 100K cell subsets
│   ├── 01_cima_activity.py      # CIMA analysis + metabolomics
│   ├── 02_inflam_activity.py    # Inflammation Atlas + treatment response
│   ├── 03_scatlas_analysis.py   # scAtlas organ/cancer comparison
│   ├── 04_integrated.py         # Cross-atlas integration
│   ├── 05_figures.py            # Publication figure generation
│   └── slurm/                   # SLURM job submission scripts
│       ├── run_cima.sh
│       ├── run_inflam.sh
│       └── run_all.sh
├── results/
│   ├── pilot/                   # Pilot analysis outputs
│   ├── cima/                    # CIMA activity results
│   ├── inflammation/            # Inflammation Atlas results
│   ├── scatlas/                 # scAtlas results
│   ├── integrated/              # Cross-atlas comparisons
│   └── figures/                 # Publication-ready figures
└── notebooks/
    └── exploratory_analysis.ipynb
```

---

## Implementation Phases

### Phase 0: Exploratory Pilot (00_pilot_analysis.py)
**Goal:** Validate pipeline and identify promising directions
- Run CytoSig on 100K cell subset from each atlas
- Verify biological sanity (IL-17 in Th17, IFNγ in CD8, TNF in monocytes)
- Test metadata linkage (CIMA metabolomics, Inflammation disease labels)
- **Decision point:** Confirm signals before full-scale computation

### Phase 1: CIMA Analysis (01_cima_activity.py)
- Compute CytoSig (44) + SecAct (1,249) on 6.5M cells
- Associate with biochemistry (19 markers) and metabolites (1,549 features)
- Identify cytokine-metabolome axis correlations

### Phase 2: Inflammation Atlas (02_inflam_activity.py)
- Process main (4.9M), validation (850K), external (573K) cohorts
- Disease differential analysis (20 diseases, 6 groups)
- **Treatment response prediction** (208 samples with R/NR)
- Cross-cohort validation

### Phase 3: scAtlas Integration (03_scatlas_analysis.py)
- Use pre-computed activities (already available)
- Normal vs cancer comparisons
- Organ-specific signatures

### Phase 4: Cross-Atlas Integration (04_integrated.py)
- Harmonize cell type annotations
- Compare healthy profiles (CIMA vs Inflammation healthy)
- Identify conserved vs atlas-specific signatures

### Phase 5: Figures and Manuscript (05_figures.py)
- Generate publication-quality figures
- Statistical summary tables
- Interactive visualizations (optional)

---

## Verification Plan

### Technical Validation
- Compare with scAtlas pre-computed activities (should correlate r > 0.9)
- Check activity distributions (z-scores typically -3 to +3)
- Verify gene overlap (>80% of signature genes)

### Biological Validation
- IL-17 elevated in Th17 cells
- IFNγ elevated in CD8 T cells and NK cells
- TNF elevated in monocytes/macrophages and IBD
- IL-6 elevated in RA and sepsis

### Clinical Validation
- Treatment response prediction AUC > 0.7
- Cross-cohort reproducibility (main → validation)
- Comparison with published disease associations

---

## Key Considerations

### Memory Management
- Use `backed='r'` mode for h5ad files
- Process in batches (10,000-50,000 cells)
- Stream results to disk with `output_path`

### GPU Usage
- Request GPU node for computation
- Use `backend='cupy'` in SecActpy functions
- Expected speedup: 10-34x over CPU

### Data Normalization
- CIMA: Use raw counts from `layers['counts']` (not log-normalized .X)
- Inflammation Atlas: Use .X directly (raw counts)
- scAtlas: Use .X directly (raw counts) or pre-computed activities

### Gene Overlap
- Check gene overlap between expression data and signatures
- SecActpy automatically handles missing genes
- Minimum 10 overlapping genes required

---

## Contact

For questions about this analysis, refer to the SecActpy documentation:
- GitHub: https://github.com/data2intelligence/SecActpy
- Local package: `/vf/users/parks34/projects/1ridgesig/SecActpy/`
