# GPU Validation Test Matrix

Comprehensive validation testing for CytoAtlas activity inference across all atlases, signatures, and aggregation levels.

## Test Matrix

| Atlas | Signatures | Cell Type Levels | Aggregation Methods |
|-------|------------|------------------|---------------------|
| CIMA (6.5M cells) | CytoSig, LinCytoSig, SecAct | L1, L2, L3 | Pseudobulk, Resampled PB, Single-cell |
| Inflammation (6.3M cells) | CytoSig, LinCytoSig, SecAct | Level1, Level2 | Pseudobulk, Resampled PB, Single-cell |
| scAtlas Normal (2.3M cells) | CytoSig, LinCytoSig, SecAct | subCluster, cellType2 | Pseudobulk, Resampled PB, Single-cell |
| scAtlas Cancer (4.1M cells) | CytoSig, LinCytoSig, SecAct | subCluster, cellType2 | Pseudobulk, Resampled PB, Single-cell |

**Total combinations:** 4 datasets × 3 signatures × ~3 cell type levels × 3 aggregation = ~108 validation runs

## Data Paths

### CIMA
- **H5AD:** `/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad`
- **Cell type columns:** `cell_type_l1` (coarse), `cell_type_l2` (27 types), `cell_type_l3` (fine)
- **Sample column:** `sample`

### Inflammation Atlas
- **Main H5AD:** `/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad`
- **Validation H5AD:** `/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad`
- **Cell type columns:** `Level1` (coarse), `Level2` (66 types)
- **Sample column:** `sampleID`

### scAtlas
- **Normal H5AD:** `/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad`
- **Cancer H5AD:** `/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad`
- **Cell type columns:** `subCluster` (standardized), `cellType2` (fine)
- **Tissue column:** `tissue`

## Aggregation Methods

### 1. Pseudobulk (Standard)
- Aggregate expression by `cell_type × sample`
- Sum raw counts, normalize (CPM or TPM)
- Cell count varies per group

### 2. Resampled Pseudobulk
- Normalize cell counts across groups via bootstrap resampling
- For each cell type × sample, resample N cells (e.g., N=100 or min cell count)
- Reduces variance from unequal cell numbers
- Multiple replicates (e.g., 10) for stability

### 3. Single-cell Level
- Per-cell activity inference
- Batch processing (10K cells per batch)
- Validate by comparing to cell-level expression

## Validation Metrics

For each combination, compute:
1. **Pearson correlation (r):** Expression vs Activity
2. **Spearman correlation (ρ):** Rank-based
3. **R² (coefficient of determination)**
4. **P-value** with FDR correction
5. **Sample size (n)**

## Output Structure

```
results/validation/
├── cima/
│   ├── cytosig/
│   │   ├── pseudobulk_l1.csv
│   │   ├── pseudobulk_l2.csv
│   │   ├── pseudobulk_l3.csv
│   │   ├── resampled_l1.csv
│   │   ├── resampled_l2.csv
│   │   ├── resampled_l3.csv
│   │   └── singlecell.csv
│   ├── lincytosig/
│   │   └── ...
│   └── secact/
│       └── ...
├── inflammation/
│   └── ...
├── scatlas_normal/
│   └── ...
├── scatlas_cancer/
│   └── ...
└── summary/
    ├── validation_summary.json       # For CytoAtlas API
    └── scatter_plot_data/            # For visualization
        ├── cima_cytosig_l2.json
        └── ...
```

## Scatter Plot Data Format

For CytoAtlas service scatter plots:

```json
{
  "atlas": "CIMA",
  "signature_type": "CytoSig",
  "level": "pseudobulk_l2",
  "n_points": 1000,
  "signatures": [
    {
      "name": "IFNG",
      "data": [
        {"expression": 2.3, "activity": 1.8, "celltype": "CD8 T", "sample": "S001"},
        ...
      ],
      "correlation": 0.85,
      "pvalue": 1e-10,
      "n": 500
    },
    ...
  ]
}
```

## SLURM Configuration

- **Partition:** gpu
- **GPU:** A100 (40GB)
- **Memory:** 64GB RAM
- **Time:** 4-8 hours per atlas
- **Array jobs:** For parallel signature processing

## Execution Order

1. Task #14: Plan (this document) ✓
2. Task #15: Create validation runner script
3. Task #16: Implement resampling pseudobulk
4. Task #17: Create SLURM job scripts
5. Tasks #18-20: Run validation (can run in parallel)
6. Task #21: Generate scatter plot datasets
