# Spatial Data Validation Skill

Usage: /validate-spatial [visium|xenium|all]

Validates spatial dataset processing and API access.

## Steps

1. **Check spatial output files** exist in `results/spatial/`:
   - `spatial_activity_by_technology.h5ad`
   - `spatial_activity_by_tissue.csv`
   - `spatial_neighborhood_activity.csv`
   - `spatial_technology_comparison.csv`
   - `spatial_dataset_metadata.csv`
   - `spatial_gene_coverage.csv`

2. **Validate visualization JSON files** in `visualization/data/`:
   - `spatial_tissue_activity.json`
   - `spatial_technology_comparison.json`
   - `spatial_gene_coverage.json`
   - `spatial_dataset_catalog.json`

3. **Verify technology stratification**:
   - Tier A (Visium): >80% gene coverage with CytoSig
   - Tier B (Xenium/MERFISH/CosMx): targeted scoring used
   - Tier C (ISS/mouse): correctly excluded

4. **Validate gene panel coverage**:
   - Check coverage percentages per technology
   - Verify Visium has full CytoSig + SecAct inference
   - Verify Tier B only has signatures with >50% gene overlap

5. **Check spatial coordinates** data integrity:
   - x/y coordinates are numeric and finite
   - Downsampled to reasonable size for API

6. **Validate DuckDB** (`spatial_data.duckdb`):
   - Check table existence and row counts
   - Verify index creation

7. **Test API endpoints** (if server running):
   - `/api/v1/spatial/summary`
   - `/api/v1/spatial/technologies`
   - `/api/v1/spatial/gene-coverage`

8. **Report** summary with pass/fail status per check.

## Arguments

- `visium` - Validate only Visium (Tier A) data
- `xenium` - Validate only Xenium/targeted (Tier B) data
- `all` (default) - Validate all spatial data
