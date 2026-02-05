# Atlas-Level Activity Inference Validation - Context Document

> **Purpose:** Cross-session context for the atlas-level activity inference validation pipeline.
> **Created:** 2026-02-03
> **Plan File:** `/data/parks34/.claude/plans/fancy-conjuring-finch.md`

---

## Quick Reference

### Job Name
**Atlas-Level Activity Inference Validation**

### Objective
Generate cell type-specific pseudobulk gene expression datasets for all 3 atlases using GPU-accelerated processing, then run activity inference with 3 signature matrices.

### Scope

| Atlas | Cohorts | Levels | Pseudobulk Files | Activity Files |
|-------|---------|--------|------------------|----------------|
| CIMA | 1 | 4 (L1-L4) | 4 | 12 (4×3 signatures) |
| Inflammation | 3 (main/val/ext) | 2 (L1-L2) | 6 | 18 (6×3 signatures) |
| scAtlas | 2 (normal/cancer) | 1 (organ×cellType) | 2 | 6 (2×3 signatures) |
| **Total** | - | - | **12** | **36** |

---

## Data Sources

### CIMA Atlas (6.5M cells)
```python
H5AD_PATH = '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad'
CELLTYPE_COLS = ['cell_type_l1', 'cell_type_l2', 'cell_type_l3', 'cell_type_l4']
SAMPLE_COL = 'sample_id'
```

### Inflammation Atlas (6.3M cells total)
```python
MAIN_H5AD = '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad'
VAL_H5AD = '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad'
EXT_H5AD = '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_external_afterQC.h5ad'
CELLTYPE_COLS = ['cell_type_level1', 'cell_type_level2']
SAMPLE_COL = 'sample_id'
```

### scAtlas (6.4M cells total)
```python
NORMAL_H5AD = '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad'
CANCER_H5AD = '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad'
CELLTYPE_COL = 'cellType1'
ORGAN_COL = 'organ'
SAMPLE_COL = 'sampleID'
# Pseudobulk grouping: organ × cellType1 × sampleID
```

---

## Signature Matrices

```python
import sys
sys.path.insert(0, '/vf/users/parks34/projects/1ridgesig/SecActpy')
from secactpy import load_cytosig, load_secact, load_lincytosig

# Signatures (genes × features)
cytosig = load_cytosig()        # Shape: (~2000, 44)
lincytosig = load_lincytosig()  # Shape: (~2000, 178)
secact = load_secact()          # Shape: (~2000, 1249)
```

---

## Output Directory Structure

```
results/atlas_validation/
├── cima/
│   ├── pseudobulk/
│   │   ├── CIMA_pseudobulk_l1.h5ad
│   │   ├── CIMA_pseudobulk_l2.h5ad
│   │   ├── CIMA_pseudobulk_l3.h5ad
│   │   └── CIMA_pseudobulk_l4.h5ad
│   └── activity/
│       ├── CIMA_CytoSig_l1.h5ad
│       ├── CIMA_LinCytoSig_l1.h5ad
│       ├── CIMA_SecAct_l1.h5ad
│       └── ... (12 total)
│
├── inflammation/
│   ├── main/pseudobulk/ & activity/
│   ├── validation/pseudobulk/ & activity/
│   └── external/pseudobulk/ & activity/
│
├── scatlas/
│   ├── normal/pseudobulk/ & activity/
│   └── cancer/pseudobulk/ & activity/
│
└── validation/
    ├── biological_validation.json
    ├── gene_coverage.json
    └── quality_scores.json
```

---

## Technical Stack

### Environment Setup
```bash
source ~/bin/myconda
conda activate secactpy
module load CUDA/12.8.1
module load cuDNN/9.12.0/CUDA-12
```

### Key Dependencies
- `secactpy` - Activity inference (from `/vf/users/parks34/projects/1ridgesig/SecActpy`)
- `cupy-cuda12x` - GPU acceleration
- `anndata` - H5AD file handling
- `cytoatlas-pipeline` - Custom pipeline package

### GPU Processing Pattern
```python
# Batch processing with backed H5AD (memory efficient)
adata = ad.read_h5ad(h5ad_path, backed='r')
batch_size = 50000

for start in range(0, adata.n_obs, batch_size):
    end = min(start + batch_size, adata.n_obs)
    X_batch = adata.X[start:end]

    # GPU aggregation
    if CUPY_AVAILABLE:
        X_gpu = cp.asarray(X_batch.toarray())
        # ... aggregate on GPU
        cp.get_default_memory_pool().free_all_blocks()
```

### Activity Inference Pattern
```python
from secactpy import ridge_batch

result = ridge_batch(
    X=signature.values,      # genes × signatures
    Y=expression.values,     # genes × samples
    lambda_=5e5,
    n_rand=1000,
    seed=0,
    batch_size=10000,
    backend='cupy',
    output_path=output_h5ad,
    verbose=True
)
```

---

## Implementation Phases

### Phase 1: Enhance cytoatlas-pipeline
- [ ] Create `batch/streaming_aggregator.py` - GPU-accelerated pseudobulk
- [ ] Create `batch/atlas_config.py` - Atlas metadata registry
- [ ] Enhance `activity/ridge.py` - Multi-signature inference

### Phase 2: Create Pipeline Scripts
- [ ] `scripts/09_atlas_validation_pseudobulk.py`
- [ ] `scripts/09_atlas_validation_activity.py`
- [ ] `scripts/09_atlas_validation_validate.py`

### Phase 3: Create SLURM Jobs
- [ ] `scripts/slurm/validation/run_all_validation.sh`
- [ ] Individual atlas job scripts

### Phase 4: Run & Validate
- [ ] Execute pipeline on all atlases
- [ ] Verify biological markers
- [ ] Generate quality reports

---

## Key Code References

### Existing Patterns to Reuse

| Pattern | Source File |
|---------|-------------|
| Backed H5AD + GPU batch | `scripts/create_cima_pseudobulk_batch.py` |
| GPU manager | `cytoatlas-pipeline/src/cytoatlas_pipeline/core/gpu_manager.py` |
| Ridge inference | `cytoatlas-pipeline/src/cytoatlas_pipeline/activity/ridge.py` |
| H5AD streaming | `cytoatlas-pipeline/src/cytoatlas_pipeline/activity/streaming.py` |
| Validation tests | `scripts/validation/02_run_activity_inference.py` |

### H5AD Output Formats

**Pseudobulk Expression:**
```python
adata = ad.AnnData(
    X=log1p_cpm,  # (n_groups, n_genes)
    obs=pd.DataFrame({'cell_type': ..., 'sample': ..., 'n_cells': ...}),
    var=pd.DataFrame(index=gene_names),
)
adata.layers['counts'] = raw_counts
```

**Activity Scores:**
```python
adata = ad.AnnData(
    X=zscore,  # (n_signatures, n_groups)
    obs=pd.DataFrame(index=signature_names),
    var=group_metadata,
)
adata.layers['beta'] = beta
adata.layers['se'] = se
adata.layers['pvalue'] = pvalue
adata.uns['signature'] = 'CytoSig'
```

---

## Validation Criteria

### Biological Markers (Must Pass)
| Cytokine | Expected High In |
|----------|------------------|
| IFNG | CD8+ T, NK, Th1 |
| IL17A | Th17 |
| IL4/IL13 | Th2 |
| TNF | Monocytes, Macrophages |
| IL10 | Tregs |

### Quality Thresholds
- Gene coverage: >80%
- Activity z-scores: -3 to +3 range
- CV stability: r > 0.9
- LinCytoSig specificity: Target cell type in top 3

---

## SLURM Resources

| Job Type | Time | Memory | GPU |
|----------|------|--------|-----|
| Pseudobulk | 4h | 128G | A100 |
| Activity | 2h | 128G | A100 |
| Validation | 1h | 64G | None |

**Total Pipeline Time:** ~20-24 hours with parallelization

---

## Resume Instructions

To continue this work in a new session:

1. Reference this context document
2. Check implementation status in the plan file
3. Review completed outputs in `results/atlas_validation/`

```bash
# Check current progress
ls -la results/atlas_validation/*/pseudobulk/
ls -la results/atlas_validation/*/activity/

# Check running jobs
squeue -u $USER
```

---

## Related Documentation

- [CLAUDE.md](../CLAUDE.md) - Project overview
- [docs/OVERVIEW.md](OVERVIEW.md) - Architecture
- [docs/pipelines/](pipelines/) - Pipeline documentation
- [cytoatlas-pipeline/README.md](../cytoatlas-pipeline/README.md) - Package documentation
