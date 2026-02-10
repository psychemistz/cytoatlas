# Archive

Legacy and superseded files moved here during the comprehensive project restructuring on 2026-02-09. These files are preserved for reference but are no longer part of the active codebase.

## Directory Structure

| Directory | Contents |
|-----------|----------|
| `scripts/generation/` | One-off data generation and visualization scripts (e.g., atlas comparisons, boxplots, signature correlations) |
| `scripts/regeneration/` | Re-run and patching scripts used to fix or regenerate specific outputs |
| `scripts/utility/` | Cell type mapping, panel validation, pseudobulk creation, gene name fixing, and other preprocessing utilities |
| `scripts/early_pipeline/` | Early pipeline iterations (04-13) superseded by the current pipeline (00-06 + 07, 10, 11, 14-16) |
| `scripts/alphagenome/` | AlphaGenome experimental pipeline (stages 1-5) |
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
