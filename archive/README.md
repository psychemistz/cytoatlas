# Archive

Legacy and superseded files moved here during project maintenance. These files are preserved for reference but are no longer part of the active codebase.

## Directory Structure

| Directory | Contents |
|-----------|----------|
| `agents/` | Sprint-specific agent prompts (orchestrator, code_reviewer, frontend_checker, etc.) |
| `docs/` | Stale documentation (DECISIONS.md, QA_LOG.md) |
| `hooks/` | Retired git hooks (post-decision.sh) |
| `scripts/generation/` | One-off data generation and visualization scripts |
| `scripts/regeneration/` | Re-run and patching scripts |
| `scripts/utility/` | Cell type mapping, panel validation, orchestration scripts, R validation |
| `scripts/early_pipeline/` | Early pipeline iterations (04-13) superseded by current pipeline |
| `scripts/alphagenome/` | **Relocated** to `/data/parks34/projects/4germicb/data/alphagenome_eqtl/cytoatlas_outputs/` |
| `scripts/future/` | Placeholder scripts for future atlases (NicheFormer, scGPT, cellxgene) |
| `scripts/preprocessing/` | Superseded preprocessing scripts (singlecell validation) |
| `slurm/` | Legacy SLURM job scripts for archived analyses |
| `slurm/future/` | SLURM wrappers for future atlas scripts |
| `slurm/utility/` | SLURM wrappers for superseded utility scripts (Parquet conversion) |
| `api/` | Legacy API service files |
| `visualization/` | Old standalone visualization files |

## Current Active Pipeline

The active analysis scripts remain in `scripts/`:

- `00_pilot_analysis.py` - Pilot validation
- `01_cima_activity.py` - CIMA atlas analysis
- `02_inflam_activity.py` - Inflammation atlas analysis
- `03_scatlas_analysis.py` - scAtlas analysis
- `04_integrated.py` - Cross-atlas integration
- `05_figures.py` - Publication figures
- `06_preprocess_viz_data.py` - Web visualization preprocessing
- `07_cross_atlas_analysis.py` - Cross-atlas comparison
- `08_scatlas_immune_analysis.py` - scAtlas immune analysis
- `10_atlas_validation_pipeline.py` - Atlas validation
- `11_donor_level_pipeline.py` - Donor-level analysis
- `14_preprocess_bulk_validation.py` - Bulk validation preprocessing
- `15_bulk_validation.py` - Bulk RNA-seq validation
- `16_resampled_validation.py` - Resampled validation
