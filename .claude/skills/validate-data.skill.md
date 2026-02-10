# validate-data

Validate visualization data JSON files

## Instructions

When the user invokes /validate-data, perform the following:

1. **Check JSON file existence** in `visualization/data/`:
   - CIMA: `cima_celltype.json`, `cima_correlations.json`, `cima_differential.json`, `cima_metabolites_top.json`, `cima_biochem_scatter.json`, `cima_eqtl.json`, `cima_atlas_validation.json`, `cima_celltype_correlations.json`, `cima_signature_expression.json`, `cima_population_stratification.json`
   - Inflammation: `inflammation_celltype.json`, `inflammation_correlations.json`, `inflammation_differential.json`, `inflammation_disease.json`, `inflammation_severity.json`, `inflammation_celltype_correlations.json`, `inflammation_longitudinal.json`, `cohort_validation.json`, `treatment_response.json`
   - scAtlas: `scatlas_organs.json`, `scatlas_organs_top.json`, `scatlas_celltypes.json`, `cancer_comparison.json`, `exhaustion.json`, `immune_infiltration.json`, `caf_signatures.json`, `adjacent_tissue.json`
   - Cross-atlas: `cross_atlas.json`, `celltype_mapping.json`, `disease_sankey.json`
   - Validation: `validation_summary.json`, `validation_corr_boxplot.json`, `bulk_rnaseq_validation.json`
   - Gene/Search: `gene_list.json`, `gene_expression.json`, `search_index.json`, `summary_stats.json`

2. **Validate JSON structure**:
   - Parse each file for valid JSON
   - Check expected keys exist
   - Verify data types (numbers are numbers, not strings)

3. **Check for common issues**:
   - NaN/Inf values in numeric fields
   - Empty arrays where data expected
   - Missing required fields (signature, signature_type, cell_type, etc.)

4. **Cross-reference with EMBEDDED_DATA_CHECKLIST.md**:
   - Ensure all required files for embedded_data.js are present

5. **Report**:
   - List of valid files with record counts
   - List of missing files
   - Any validation errors found

## Arguments

- `/validate-data` - Check all files
- `/validate-data <filename>` - Check specific file
- `/validate-data --fix` - Attempt to fix common issues (NaN→null, etc.)
