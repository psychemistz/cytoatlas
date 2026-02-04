# Session Handoff: Atlas Multi-Level Pseudobulk Generation

**Date:** 2026-02-04 01:35 EST
**Status:** Jobs running overnight
**Plan File:** `/data/parks34/.claude/plans/fancy-conjuring-finch.md`

---

## Current State

### Jobs Running/Queued

| Atlas | Job ID | Status | Location | ETA |
|-------|--------|--------|----------|-----|
| CIMA | nohup PID 446336 | RUNNING | cn4272 (current node) | ~30 min from 01:30 |
| inflammation_main | 10852732 | PENDING | SLURM queue | When GPU available |
| inflammation_val | 10852733 | PENDING | SLURM queue | When GPU available |
| inflammation_ext | 10852803 | PENDING | SLURM queue | When GPU available |
| scatlas_normal | 10852735 | PENDING | SLURM queue | When GPU available |
| scatlas_cancer | 10852737 | PENDING | SLURM queue | When GPU available |

### Log Files
- CIMA: `logs/validation/cima_multi_direct.log`
- SLURM jobs: `logs/validation/{atlas}_multi_{jobid}.out`

---

## What Was Done This Session

### 1. Created Multi-Level Aggregator
Single-pass processing of ALL annotation levels from one H5AD file opening.

**Key Files Created:**
- `cytoatlas-pipeline/src/cytoatlas_pipeline/batch/multi_level_aggregator.py`
- `scripts/09_atlas_multilevel_pseudobulk.py`
- `scripts/slurm/validation/*_multilevel.sh` (6 files)

### 2. Fixed Bugs
- **Chan's algorithm bug**: `gene_count` was passed as int (immutable) instead of array
- **CIMA sample column**: Changed from `donor_id` to `sample` in atlas_config.py
- **CLI bug**: `--atlas` was required even with `--list-atlases`

### 3. Atlas Registry
```python
# Available atlases in ATLAS_REGISTRY
ATLAS_REGISTRY = {
    'cima': AtlasConfig(levels=['L1', 'L2', 'L3', 'L4'], sample_col='sample'),
    'inflammation_main': AtlasConfig(levels=['L1', 'L2'], sample_col='sample_id'),
    'inflammation_val': AtlasConfig(levels=['L1', 'L2'], sample_col='sample_id'),
    'inflammation_ext': AtlasConfig(levels=['L1', 'L2'], sample_col='sample_id'),
    'scatlas_normal': AtlasConfig(levels=['organ_celltype', 'celltype', 'organ'], sample_col='sample_id'),
    'scatlas_cancer': AtlasConfig(levels=['organ_celltype', 'celltype', 'organ'], sample_col='sample_id'),
}
```

---

## Expected Output Files

Each atlas generates 2 files per level:
- `{atlas}_pseudobulk_{level}.h5ad` - Main pseudobulk (cell types × genes)
- `{atlas}_pseudobulk_{level}_resampled.h5ad` - Bootstrap samples (100 resamples)

### Output Directories
```
results/atlas_validation/
├── cima/pseudobulk/                    # 8 files (L1-L4 × 2)
├── inflammation_main/pseudobulk/       # 4 files (L1-L2 × 2)
├── inflammation_val/pseudobulk/        # 4 files (L1-L2 × 2)
├── inflammation_ext/pseudobulk/        # 4 files (L1-L2 × 2)
├── scatlas_normal/pseudobulk/          # 6 files (3 levels × 2)
└── scatlas_cancer/pseudobulk/          # 6 files (3 levels × 2)
```

**Total: 32 H5AD files**

---

## Morning Verification Commands

```bash
# 1. Check SLURM job status
sacct -u parks34 --starttime=2026-02-04 --format=JobID,JobName,State,Elapsed,ExitCode

# 2. List all output files
find results/atlas_validation -name "*.h5ad" -printf "%p %s\n" | sort

# 3. Check CIMA completion
tail -50 logs/validation/cima_multi_direct.log

# 4. Verify file contents
python -c "
import anndata as ad
from pathlib import Path

for f in Path('results/atlas_validation').rglob('*.h5ad'):
    try:
        adata = ad.read_h5ad(f, backed='r')
        print(f'{f.name}: {adata.shape[0]} groups × {adata.shape[1]} genes')
    except Exception as e:
        print(f'{f.name}: ERROR - {e}')
"
```

---

## Next Steps After Pseudobulk Completes

### 1. Activity Inference
Run CytoSig, LinCytoSig, SecAct on all pseudobulk files:
```bash
python scripts/09_atlas_validation_activity.py --input results/atlas_validation/cima/pseudobulk/*.h5ad
```

### 2. Cross-Atlas Validation
Compare signatures across atlases for consistency.

### 3. Update API
Integrate new pseudobulk data into CytoAtlas API endpoints.

---

## Key Code References

### Multi-Level Aggregator Pattern
```python
from cytoatlas_pipeline.batch import aggregate_all_levels

output_paths = aggregate_all_levels(
    atlas_name='cima',
    output_dir=Path('results/atlas_validation/cima/pseudobulk'),
    n_bootstrap=100,
    batch_size=50000,
    use_gpu=True,
    skip_existing=True,
)
```

### H5AD Output Structure
```python
# Main pseudobulk
adata.X           # log1p(CPM) expression (cell_types × genes)
adata.layers['counts']   # Raw sum counts
adata.layers['zscore']   # Atlas-level z-scored expression
adata.obs['cell_type']   # Cell type names
adata.obs['n_cells']     # Cell count per group
adata.uns['atlas_stats'] # Atlas-level mean/std for z-scoring

# Resampled pseudobulk
adata.X           # Bootstrap sample expression
adata.obs['cell_type']    # Cell type
adata.obs['bootstrap_idx'] # Bootstrap index (0-99)
```

---

## Troubleshooting

### If CIMA job died
```bash
# Check if process is still running
ps aux | grep "09_atlas_multilevel" | grep -v grep

# If not, restart with skip-existing
nohup python scripts/09_atlas_multilevel_pseudobulk.py --atlas cima --skip-existing \
    > logs/validation/cima_multi_restart.log 2>&1 &
```

### If SLURM jobs failed
```bash
# Check error logs
cat logs/validation/inflam_main_multi_*.err

# Resubmit
sbatch scripts/slurm/validation/inflam_main_multilevel.sh
```

### Memory issues
Reduce batch size:
```bash
python scripts/09_atlas_multilevel_pseudobulk.py --atlas cima --batch-size 25000
```

---

## Session Statistics

- **CIMA Processing Rate:** ~27 sec/batch, 130 batches total
- **Memory Usage:** 125-135 GB RAM for CIMA
- **H5AD Open Time:** 4.6 minutes (backed mode)
- **Cell Type Groups:** L1=6, L2=27, L3=38, L4=73

---

## Files Modified This Session

```
cytoatlas-pipeline/src/cytoatlas_pipeline/batch/
├── __init__.py                    # Added MultiLevelAggregator exports
├── atlas_config.py                # Fixed CIMA sample_col: donor_id → sample
└── multi_level_aggregator.py      # NEW: Multi-level streaming aggregator

scripts/
├── 09_atlas_multilevel_pseudobulk.py  # NEW: CLI for multi-level generation

scripts/slurm/validation/
├── cima_multilevel.sh             # NEW
├── inflam_main_multilevel.sh      # NEW
├── inflam_val_multilevel.sh       # NEW
├── inflam_ext_multilevel.sh       # NEW
├── scatlas_normal_multilevel.sh   # NEW
└── scatlas_cancer_multilevel.sh   # NEW
```
