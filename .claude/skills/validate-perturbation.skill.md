# Perturbation Data Validation Skill

Usage: /validate-perturbation [parse10m|tahoe|all]

Validates perturbation dataset processing and API access.

## Steps

1. **Check parse_10M output files** exist in `results/parse10m/`:
   - `parse10m_pseudobulk_activity.h5ad`
   - `parse10m_treatment_vs_control.csv`
   - `parse10m_cytokine_response_matrix.csv`
   - `parse10m_ground_truth_validation.csv`

2. **Check Tahoe output files** exist in `results/tahoe/`:
   - `tahoe_pseudobulk_activity.h5ad`
   - `tahoe_drug_vs_control.csv`
   - `tahoe_drug_sensitivity_matrix.csv`
   - `tahoe_dose_response.csv`
   - `tahoe_cytokine_pathway_activation.csv`

3. **Validate visualization JSON files** in `visualization/data/`:
   - `parse10m_cytokine_heatmap.json`
   - `parse10m_ground_truth.json`
   - `parse10m_donor_variability.json`
   - `tahoe_drug_sensitivity.json`
   - `tahoe_dose_response.json`
   - `tahoe_pathway_activation.json`

4. **Validate DuckDB** (`perturbation_data.duckdb`):
   - Check table existence and row counts
   - Verify index creation
   - Spot-check data integrity (no all-NaN columns)

5. **Test API endpoints** (if server running):
   - `/api/v1/perturbation/summary`
   - `/api/v1/perturbation/parse10m/ground-truth`
   - `/api/v1/perturbation/tahoe/sensitivity-matrix`

6. **Report** summary with pass/fail status per check.

## Arguments

- `parse10m` - Validate only parse_10M data
- `tahoe` - Validate only Tahoe data
- `all` (default) - Validate both datasets
