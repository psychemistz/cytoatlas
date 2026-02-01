# Embedded Data Checklist for visualization/index.html

This checklist tracks all JSON data files that should be embedded into `visualization/data/embedded_data.js` for the standalone visualization to work correctly.

**IMPORTANT:** Run `scripts/06_preprocess_viz_data.py` to regenerate `embedded_data.js` whenever the underlying data changes.

## Required JSON Files

### CIMA Atlas Data
| File | Embedded Key | Status | Notes |
|------|--------------|--------|-------|
| `cima_correlations.json` | `cimacorrelations` | Required | Age/BMI correlations |
| `cima_metabolites_top.json` | `cimametabolitestop` | Required | Top metabolite correlations |
| `cima_differential.json` | `cimadifferential` | Required | Sex/smoking differential |
| `cima_celltype.json` | `cimacelltype` | Required | Cell type signatures |
| `cima_celltype_correlations.json` | `cimacelltypecorrelations` | Required | Cell type-specific age/BMI correlations |
| `cima_biochem_scatter.json` | `cimabiochemscatter` | Required | Sample-level biochemistry scatter |
| `cima_population_stratification.json` | `cimapopulationstratification` | Required | Population stratification (sex, age, BMI, blood type, smoking) |
| `cima_eqtl.json` OR `cima_eqtl_top.json` | `cimaeqtl` | Required | eQTL browser data (71,530 eQTLs) |

### Inflammation Atlas Data
| File | Embedded Key | Status | Notes |
|------|--------------|--------|-------|
| `inflammation_celltype.json` | `inflammationcelltype` | Required | Cell type signatures |
| `inflammation_correlations.json` | `inflammationcorrelations` | Required | Age/BMI correlations |
| `inflammation_celltype_correlations.json` | `inflammationcelltypecorrelations` | Required | Cell type-specific correlations |
| `inflammation_disease_filtered.json` | `inflammationdisease` | Required | Disease activity (filtered version) |
| `inflammation_severity_filtered.json` | `inflammationseverity` | Required | Disease severity analysis |
| `inflammation_differential.json` | `inflammationdifferential` | Required | Disease vs healthy differential |
| `inflammation_longitudinal.json` | `inflammationlongitudinal` | Required | Longitudinal analysis |
| `inflammation_cell_drivers.json` | `inflammationcelldrivers` | Required | Cell type drivers by disease |
| `inflammation_demographics.json` | `inflammationdemographics` | Required | Demographic analysis |
| `treatment_response.json` | `treatmentresponse` | Required | Treatment response predictions |
| `cohort_validation.json` | `cohortvalidation` | Required | Cross-cohort validation |
| `disease_sankey.json` | `diseasesankey` | Required | Disease hierarchy Sankey |

### scAtlas Data
| File | Embedded Key | Status | Notes |
|------|--------------|--------|-------|
| `scatlas_organs.json` | `scatlasorgans` | Required | Organ signatures |
| `scatlas_organs_top.json` | `scatlasorganstop` | Required | Top organ signatures |
| `scatlas_celltypes.json` | `scatlascelltypes` | Required | Cell type signatures |
| `cancer_comparison.json` | `cancercomparison` | Required | Tumor vs adjacent normal |
| `cancer_types.json` | `cancertypes` | Required | Cancer type signatures |
| `immune_infiltration.json` | `immuneinfiltration` | Required | Immune infiltration |
| `exhaustion.json` | `exhaustion` | Required | T cell exhaustion |
| `caf_signatures.json` | `cafsignatures` | Required | CAF signatures |
| `organ_cancer_matrix.json` | `organcancermatrix` | Required | Organ-cancer matrix |
| `adjacent_tissue.json` | `adjacenttissue` | Required | Adjacent tissue signatures |

### Cross-Atlas Data
| File | Embedded Key | Status | Notes |
|------|--------------|--------|-------|
| `cross_atlas.json` | `crossatlas` | Required | Cross-atlas comparison |
| `summary_stats.json` | `summarystats` | Required | Summary statistics |

### Age/BMI Boxplot Data
| File | Embedded Key | Status | Notes |
|------|--------------|--------|-------|
| `age_bmi_boxplots.json` OR `age_bmi_boxplots_cytosig.json` | `agebmiboxplots` | Required | CytoSig boxplot data for CIMA & Inflammation |
| `age_bmi_boxplots_secact.json` | N/A | Lazy-loaded | SecAct boxplot data (loaded on demand) |

## Regeneration Command

```bash
cd /data/parks34/projects/2secactpy
python scripts/06_preprocess_viz_data.py
```

This will:
1. Process all JSON files from `results/` directories
2. Create individual JSON files in `visualization/data/`
3. Bundle selected files into `visualization/data/embedded_data.js`

## Verification

After regeneration, verify the embedded data has all required keys:

```bash
# Check embedded data keys
grep -o '"[a-z_]*":' visualization/data/embedded_data.js | head -50

# Check file size
ls -lh visualization/data/embedded_data.js
```

Expected size: 150-300 MB (depending on included data)

## Lazy Loading Fallbacks

The following files have fallback lazy loading if not embedded:
- `cima_population_stratification.json` - Loaded via `loadPopulationDataIfNeeded()`
- `cima_eqtl.json` - Loaded via `loadFullEqtlData()`
- `age_bmi_boxplots.json` - Loaded via `loadBoxplotDataIfNeeded()`

## Common Issues

1. **Population panel shows "No data"**: Check if `cima_population_stratification.json` exists and has `CytoSig`/`SecAct` top-level keys
2. **eQTL browser empty**: Check if `cima_eqtl.json` or `cima_eqtl_top.json` exists with `eqtls` array
3. **Age/BMI stratified panel empty**: Check if `age_bmi_boxplots.json` exists with `cima`/`inflammation` keys containing `age`/`bmi` arrays

## Last Updated

2026-02-01
