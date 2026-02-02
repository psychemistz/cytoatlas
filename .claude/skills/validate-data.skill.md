# validate-data

Validate visualization data JSON files

## Instructions

When the user invokes /validate-data, perform the following:

1. **Check JSON file existence** in `visualization/data/`:
   - Core files: `cima_celltype.json`, `inflammation_celltype.json`, `scatlas_organs.json`
   - Cross-atlas: `cross_atlas.json`, `conserved_signatures.json`
   - Correlations: `cima_correlations.json`, `cima_metabolites_top.json`
   - Differential: `inflammation_differential.json`

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
