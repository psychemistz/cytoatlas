# Pipeline Documentation

This directory contains documentation for all analysis pipelines in the CytoAtlas project.

## Pipeline Overview

```mermaid
flowchart TB
    subgraph Input ["Input Data"]
        CIMA_H5AD[CIMA H5AD<br/>6.5M cells]
        INFLAM_H5AD[Inflammation H5AD<br/>6.3M cells]
        SCATLAS_H5AD[scAtlas H5AD<br/>6.4M cells]
        SIGS[Signature Matrices<br/>CytoSig + SecAct]
    end

    subgraph Scripts ["Analysis Scripts"]
        P00[00_pilot_analysis.py]
        P01[01_cima_activity.py]
        P02[02_inflam_activity.py]
        P03[03_scatlas_analysis.py]
        P07[07_scatlas_immune_analysis.py]
        P06[06_preprocess_viz_data.py]
    end

    subgraph Results ["Results"]
        CIMA_OUT[CIMA Results<br/>Correlations, Differential]
        INFLAM_OUT[Inflammation Results<br/>Disease, Treatment]
        SCATLAS_OUT[scAtlas Results<br/>Organs, Cancer]
    end

    subgraph Viz ["Visualization"]
        JSON[JSON Files<br/>30+ files]
        HTML[index.html<br/>Web Dashboard]
    end

    CIMA_H5AD --> P01
    INFLAM_H5AD --> P02
    SCATLAS_H5AD --> P03
    SCATLAS_H5AD --> P07
    SIGS --> P01 & P02 & P03

    P01 --> CIMA_OUT
    P02 --> INFLAM_OUT
    P03 --> SCATLAS_OUT
    P07 --> SCATLAS_OUT

    CIMA_OUT --> P06
    INFLAM_OUT --> P06
    SCATLAS_OUT --> P06

    P06 --> JSON
    JSON --> HTML
```

## Script Summary

| Phase | Script | Description | Runtime | GPU |
|-------|--------|-------------|---------|-----|
| 0 | `00_pilot_analysis.py` | Validation on 100K cell subsets | ~30 min | Yes |
| 1 | `01_cima_activity.py` | CIMA activity + correlations | ~2 hr | Yes |
| 2 | `02_inflam_activity.py` | Inflammation activity + disease analysis | ~3 hr | Yes |
| 3 | `03_scatlas_analysis.py` | scAtlas organs + cancer comparison | ~4 hr | Yes |
| 3b | `07_scatlas_immune_analysis.py` | Immune infiltration + exhaustion | ~2 hr | Yes |
| 4 | `06_preprocess_viz_data.py` | JSON preprocessing for web | ~30 min | No |

## Common Processing Steps

All activity pipelines follow this pattern:

```mermaid
flowchart LR
    A[Raw Counts] --> B[Aggregate<br/>Sample × Cell Type]
    B --> C[TPM Normalize]
    C --> D[Log2 Transform]
    D --> E[Differential<br/>Subtract Mean]
    E --> F[Ridge Regression]
    F --> G[Activity Z-scores]
```

### Step 1: Aggregation
Cells are grouped by sample and cell type to create pseudo-bulk profiles:
```python
expr_df, meta_df = aggregate_by_sample_celltype(adata, cell_type_col, sample_col)
```

### Step 2: Normalization
TPM normalization followed by log2 transformation:
```python
expr_log = normalize_and_transform(expr_df)
```

### Step 3: Differential Expression
Center by subtracting row means:
```python
expr_diff = compute_differential(expr_log)
```

### Step 4: Activity Inference
Ridge regression against signature matrices:
```python
result = run_activity_inference(expr_diff, signature, sig_name)
```

## Pipeline Details

### Phase 0: Pilot Analysis
- [Pilot Validation](pilot.md)

### Phase 1: CIMA Analysis
- [CIMA Activity Pipeline](cima/activity.md)
- [Age/BMI/Biochemistry Correlations](cima/panels/correlations.md)
- [Metabolite Analysis](cima/panels/metabolites.md)
- [Sex/Smoking Differential](cima/panels/differential.md)

### Phase 2: Inflammation Analysis
- [Inflammation Activity Pipeline](inflammation/activity.md)
- [Disease Differential](inflammation/panels/disease.md)
- [Treatment Response](inflammation/panels/treatment.md)
- [Cross-Cohort Validation](inflammation/panels/validation.md)

### Phase 3: scAtlas Analysis
- [scAtlas Activity Pipeline](scatlas/analysis.md)
- [Organ Signatures](scatlas/panels/organs.md)
- [Cancer Comparison](scatlas/panels/cancer.md)
- [T Cell Exhaustion](scatlas/panels/exhaustion.md)
- [Immune Infiltration](scatlas/immune.md)

### Phase 4: Visualization
- [JSON Preprocessing](visualization/preprocess.md)

## Execution

### SLURM (Recommended)

```bash
# Full pipeline
sbatch scripts/slurm/run_all.sh

# Individual analyses
sbatch scripts/slurm/run_cima.sh
sbatch scripts/slurm/run_inflam.sh
sbatch scripts/slurm/run_scatlas.sh
```

### Direct Execution

```bash
cd /data/parks34/projects/2secactpy

# Activate environment
source ~/bin/myconda
conda activate secactpy

# Run scripts
python scripts/01_cima_activity.py --mode pseudobulk
python scripts/02_inflam_activity.py --mode both
python scripts/03_scatlas_analysis.py
python scripts/06_preprocess_viz_data.py
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 10,000 | Cells per batch for single-cell analysis |
| `N_RAND` | 1,000 | Permutations for p-value calculation |
| `LAMBDA` | 5e5 | Ridge regression regularization |
| `SEED` | 0 | Random seed for reproducibility |
| `BACKEND` | cupy/numpy | Computation backend (GPU if available) |

## Data Flow Diagram

```mermaid
flowchart TB
    subgraph Phase1 ["Phase 1: CIMA"]
        C1[CIMA H5AD] --> C2[Pseudo-bulk]
        C2 --> C3[Activity Z-scores]
        C3 --> C4[Age/BMI Correlation]
        C3 --> C5[Biochemistry Correlation]
        C3 --> C6[Metabolite Correlation]
        C3 --> C7[Sex/Smoking Differential]
    end

    subgraph Phase2 ["Phase 2: Inflammation"]
        I1[Inflammation H5AD] --> I2[Pseudo-bulk]
        I2 --> I3[Activity Z-scores]
        I3 --> I4[Disease Differential]
        I3 --> I5[Treatment Response]
        I3 --> I6[Cohort Validation]
    end

    subgraph Phase3 ["Phase 3: scAtlas"]
        S1[scAtlas H5AD] --> S2[Normal Organs]
        S1 --> S3[PanCancer]
        S2 --> S4[Organ Signatures]
        S3 --> S5[Cancer Comparison]
        S3 --> S6[Exhaustion Analysis]
        S3 --> S7[Immune Infiltration]
    end

    subgraph Phase4 ["Phase 4: Visualization"]
        C4 & C5 & C6 & C7 --> V1[CIMA JSON]
        I4 & I5 & I6 --> V2[Inflammation JSON]
        S4 & S5 & S6 & S7 --> V3[scAtlas JSON]
        V1 & V2 & V3 --> V4[Web Dashboard]
    end
```
