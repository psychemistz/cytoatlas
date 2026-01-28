# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## github instruction
Do not use "Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>" in github commit message

## Project Overview

Pan-Disease Single-Cell Cytokine Activity Atlas - computes cytokine and secreted protein activity signatures across 12+ million human immune cells from three major single-cell atlases (CIMA, Inflammation Atlas, scAtlas) to identify disease-specific and conserved signaling patterns, with applications to treatment response prediction.

## Development Environment

```bash
# Activate the conda environment
source ~/bin/myconda
conda activate secactpy
```

Required external package: `secactpy` from `/vf/users/parks34/projects/1ridgesig/SecActpy/`

## Running Analyses

### SLURM Job Submission (HPC)

```bash
# Full pipeline with job dependencies
sbatch scripts/slurm/run_all.sh

# Pilot only (quick validation, ~2 hours)
sbatch scripts/slurm/run_all.sh --pilot

# Main analyses only (skip pilot)
sbatch scripts/slurm/run_all.sh --main

# Individual analyses
sbatch scripts/slurm/run_pilot.sh          # Validation (2h, GPU)
sbatch scripts/slurm/run_cima.sh           # CIMA 6.5M cells (24h, GPU)
sbatch scripts/slurm/run_inflam.sh         # Inflammation 6.3M cells (48h, GPU)
sbatch scripts/slurm/run_scatlas.sh        # scAtlas 6.4M cells (24h, GPU)
sbatch scripts/slurm/run_integrated.sh     # Cross-atlas comparison (4h, CPU)
```

### Direct Script Execution

```bash
cd /data/parks34/projects/2secactpy
python scripts/00_pilot_analysis.py --n-cells 100000 --seed 42
python scripts/01_cima_activity.py --mode pseudobulk
python scripts/02_inflam_activity.py --mode both
python scripts/05_figures.py --all
python scripts/06_preprocess_viz_data.py
```

## Architecture

### 6-Phase Pipeline

1. **Phase 0 (00_pilot_analysis.py):** Pilot validation on 100K cell subsets - validates biology and metadata linkage before full-scale runs
2. **Phase 1 (01_cima_activity.py):** CIMA analysis - 6.5M cells, pseudo-bulk aggregation, correlation with biochemistry/metabolomics
3. **Phase 2 (02_inflam_activity.py):** Inflammation Atlas - 6.3M cells across 3 cohorts (main/validation/external), treatment response prediction with ML
4. **Phase 3 (03_scatlas_analysis.py):** scAtlas - 6.4M cells (normal organs + cancer), uses pre-computed activities
5. **Phase 4 (04_integrated.py):** Cross-atlas integration and comparison
6. **Phase 5 (05_figures.py, 06_preprocess_viz_data.py):** Publication figures and web visualization

### Key Design Patterns

- GPU acceleration via CuPy (10-34x speedup) with automatic fallback to NumPy
- Pseudo-bulk aggregation (cell type × sample) as primary analysis level
- Single-cell batch processing (10K cells/batch) for detailed analysis
- Backed mode (`ad.read_h5ad(..., backed='r')`) for memory efficiency on 6M+ cell files

## Analysis Details

### Two Levels of Analysis

#### 1. Pseudo-bulk Analysis (Primary)
Aggregates expression by cell type and sample before computing activities:
```python
# Aggregate by sample and cell type (sum counts)
groups = adata.obs.groupby([sample_col, cell_type_col]).groups
for (sample, celltype), indices in groups.items():
    group_sum = X[indices, :].sum(axis=0)

# TPM normalize and log2 transform
expr_tpm = expr_df / col_sums * 1e6
expr_log = np.log2(expr_tpm + 1)

# Compute differential (subtract row mean)
expr_diff = expr_log.subtract(expr_log.mean(axis=1), axis=0)
```

#### 2. Single-cell Analysis (Detailed)
Computes per-cell activities with batch processing:
```python
ridge_batch(
    X=sig_scaled.values,           # Signature matrix
    Y=X[:, common_idx].T,          # Expression (genes × cells)
    batch_size=10000,              # 10K cells per batch
    backend='cupy',                # GPU acceleration
    output_path=str(output_path),  # Stream to disk
)
```

### Statistical Analysis Methods

| Method | Use Case | Function |
|--------|----------|----------|
| Spearman correlation | Continuous variables (age, BMI, metabolites) | `correlation_analysis()` |
| Wilcoxon rank-sum | Categorical comparisons (disease vs healthy) | `differential_analysis()` |
| Benjamini-Hochberg FDR | Multiple testing correction | `multipletests(method='fdr_bh')` |
| Logistic Regression | Treatment response prediction | `build_response_predictor()` |
| Random Forest | Treatment response prediction | `build_response_predictor()` |

### Cell-Type Stratified Analysis

Computes disease differential within each cell type to identify driving populations:
```python
# In 02_inflam_activity.py
celltype_stratified_disease_analysis()  # Disease vs healthy per cell type
identify_driving_cell_populations()     # Top cell types per disease
identify_conserved_programs()           # Programs shared across 3+ diseases
```

## Visualization Plan

### Publication Figures (05_figures.py)

| Figure | Type | Description |
|--------|------|-------------|
| Fig 1 | Schema | Overview with multi-atlas integration diagram |
| Fig 2 | Heatmap | Cytokine activities: cell types × cytokines × diseases (clustered) |
| Fig 3 | Bar/ROC | Treatment response prediction performance by disease |
| Fig 4 | Volcano | Disease differential (log2FC vs -log10 p-value) |
| Fig 5 | Heatmap | Cytokine-metabolome correlations (CIMA) |
| Fig 6 | UpSet/Bar | Cross-disease comparison: shared vs disease-specific signatures |

### Cell-Type Specific Activity Visualizations

#### Box Plots by Age Bins
```python
# Age correlation analysis in 01_cima_activity.py
corr_age = correlation_analysis(activity_df, sample_meta, ['Age'], agg_meta_df=meta_df)

# For visualization, bin ages:
age_bins = [0, 30, 40, 50, 60, 70, 100]
age_labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70+']
sample_meta['age_bin'] = pd.cut(sample_meta['Age'], bins=age_bins, labels=age_labels)

# Box plot: cytokine activity by age bin per cell type
for celltype in celltypes:
    celltype_cols = [c for c in activity_df.columns if meta_df.loc[c, 'cell_type'] == celltype]
    # Plot activity distribution across age bins
```

#### Box Plots by BMI Bins
```python
# BMI correlation analysis in 01_cima_activity.py
corr_bmi = correlation_analysis(activity_df, sample_meta, ['BMI'], agg_meta_df=meta_df)

# For visualization, bin BMI:
bmi_bins = [0, 18.5, 25, 30, 35, 100]
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II+']
sample_meta['bmi_bin'] = pd.cut(sample_meta['BMI'], bins=bmi_bins, labels=bmi_labels)

# Box plot: cytokine activity by BMI category per cell type
```

#### Cell-Type Specific Cytokine Heatmaps
```python
# From celltype_stratified_disease_analysis()
# Pivot: cell_type × cytokine with disease-specific coloring
pivot_df = sig_df.pivot_table(
    index='cell_type',
    columns='protein',
    values='log2fc',
    aggfunc='mean'
)
sns.heatmap(pivot_df, cmap='RdBu_r', center=0)
```

### Interactive Web Visualization (06_preprocess_viz_data.py)

Preprocesses data for HTML dashboard:

| JSON File | Contents |
|-----------|----------|
| `cima_correlations.json` | Age, BMI, biochemistry correlations |
| `cima_metabolites_top.json` | Top 500 metabolite correlations |
| `cima_differential.json` | Sex, smoking, blood type differential |
| `inflammation_celltype.json` | Cell type × signature mean activities |
| `scatlas_organs.json` | Organ-specific signatures |
| `scatlas_celltypes.json` | Top 100 cell types × signatures |
| `summary_stats.json` | Overview statistics |

### Planned Visualization Types

1. **Activity Distribution Box Plots**
   - By age bins (decade intervals)
   - By BMI categories (WHO classification)
   - By sex, smoking status, blood type
   - Stratified by cell type and disease

2. **Cell-Type Specific Heatmaps**
   - Cell type × cytokine activity matrix per disease
   - Hierarchical clustering to identify cell type groups
   - Annotate driving cell populations

3. **Disease Comparison Visualizations**
   - Volcano plots per disease (log2FC vs significance)
   - UpSet plots for shared/unique signatures
   - Radar/spider plots for disease profiles

4. **Treatment Response Visualizations**
   - ROC curves per disease and model type
   - Feature importance bar plots (top predictive cytokines)
   - Prediction probability distributions (R vs NR)

5. **Cross-Atlas Comparisons**
   - Correlation scatter plots (CIMA vs Inflammation healthy)
   - Sankey diagrams for cell type mapping
   - Consistency heatmaps across cohorts

## Data Paths (Hardcoded in Scripts)

```python
# CIMA
CIMA_H5AD = '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad'
CIMA_BIOCHEM = '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_Sample_Blood_Biochemistry_Results.csv'
CIMA_METAB = '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_Sample_Plasma_Metabolites_and_Lipids_Results.csv'

# Inflammation Atlas
MAIN_H5AD = '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad'
VAL_H5AD = '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad'
EXT_H5AD = '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_external_afterQC.h5ad'

# scAtlas
NORMAL_COUNTS = '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad'
CANCER_COUNTS = '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad'
```

## Signature Matrices

- **CytoSig:** 44 cytokines/growth factors (IFNγ, IL-17, TNF, IL-6, IL-10, etc.)
- **SecAct:** 1,249 secreted proteins

Load via:
```python
from secactpy import load_cytosig, load_secact
cytosig = load_cytosig()  # (genes × 44)
secact = load_secact()    # (genes × 1249)
```

## Output Structure

```
results/
├── pilot/                # Pilot validation results
├── cima/                 # CIMA results
│   ├── CIMA_CytoSig_pseudobulk.h5ad
│   ├── CIMA_SecAct_pseudobulk.h5ad
│   ├── CIMA_CytoSig_singlecell.h5ad
│   ├── CIMA_SecAct_singlecell.h5ad
│   ├── CIMA_correlation_age.csv
│   ├── CIMA_correlation_bmi.csv
│   ├── CIMA_correlation_biochemistry.csv
│   ├── CIMA_correlation_metabolites.csv
│   └── CIMA_differential_demographics.csv
├── inflammation/         # Inflammation Atlas results
│   ├── main_CytoSig_pseudobulk.h5ad
│   ├── main_SecAct_pseudobulk.h5ad
│   ├── disease_differential.csv
│   ├── diseaseGroup_differential.csv
│   ├── treatment_response.csv
│   ├── treatment_prediction_summary.csv
│   ├── celltype_stratified_differential.csv
│   ├── driving_cell_populations.csv
│   ├── conserved_cytokine_programs.csv
│   ├── correlation_age.csv
│   ├── correlation_bmi.csv
│   └── cross_cohort_validation.csv
├── scatlas/              # scAtlas results
│   ├── normal_organ_signatures.csv
│   ├── normal_celltype_signatures.csv
│   ├── normal_top_organ_signatures.csv
│   └── cancer_signatures.csv
├── integrated/           # Cross-atlas comparisons
└── figures/              # Publication-ready figures

visualization/
├── data/                 # JSON files for web display
│   ├── cima_correlations.json
│   ├── cima_metabolites_top.json
│   ├── cima_differential.json
│   ├── inflammation_celltype.json
│   ├── scatlas_organs.json
│   ├── scatlas_celltypes.json
│   └── summary_stats.json
├── index.html            # Interactive dashboard
└── index_standalone.html # Standalone version
```

## Resource Requirements

| Analysis | Time | Memory | GPU |
|----------|------|--------|-----|
| Pilot | 2h | 128GB | A100 |
| CIMA | 24h | 128GB | A100 |
| Inflammation | 48h | 128GB | A100 |
| scAtlas | 24h | 128GB | A100 |
| Integrated | 4h | 64GB | No |
| Figures | 2h | 32GB | No |

## Validation Strategy

No formal test framework. Validation through:

1. **Pilot analysis:** Run on 100K cell subsets first to validate biology
   - IL-17 elevated in Th17 cells
   - IFNγ elevated in CD8 T cells and NK cells
   - TNF elevated in monocytes/macrophages

2. **Cross-cohort validation:** Main → validation → external cohort generalization

3. **Output verification:**
   - Activity z-scores typically -3 to +3
   - Gene overlap >80% of signature genes
   - Correlation with pre-computed scAtlas activities r > 0.9
   - Treatment response prediction AUC > 0.7

## Git Workflow

When committing changes:

```bash
# Configure git user for this repository
git config user.email "seongyong.park@nih.gov"
git config user.name "Seongyong Park"

# Commit with descriptive message
git commit -m "Brief description of changes"
```

Do not use co-authored-by or other attribution lines in commit messages.
