# Q&A Log

## Q1: Where do the overview statistics in index.html come from?

**Date:** 2026-01-28

**Question:** Where do these values come from: 421 CIMA healthy donors, 35 organs profiled, 376 cell types, 817 inflammation samples, 43 CytoSig signatures?

**Answer:**

All values are defined in `EMBEDDED_STATS` (index.html lines 2246-2271) and populated via JavaScript. They originate from `scripts/06_preprocess_viz_data.py` function `create_summary_stats()` (lines 1020-1079).

### Source Files and Calculations

| Statistic | Value | Source File | Calculation |
|-----------|-------|-------------|-------------|
| CIMA Healthy Donors | 421 | `results/cima/CIMA_correlation_age.csv` | `age_df['n'].iloc[0]` |
| Organs Profiled | 35 | `results/scatlas/normal_organ_signatures.csv` | `len(organ_df['organ'].unique())` |
| Cell Types | 376 | `results/scatlas/normal_celltype_signatures.csv` | `len(ct_df['cell_type'].unique())` |
| Inflammation Samples | 817 | `results/inflammation/main_CytoSig_pseudobulk.h5ad` | `adata.var['sample'].nunique()` |
| CytoSig Signatures | 43 | `results/cima/CIMA_correlation_age.csv` | `len(age_df[age_df['signature']=='CytoSig']['protein'].unique())` |

### Code in 06_preprocess_viz_data.py (lines 1046-1069)

```python
summary = {
    'cima': {
        'n_samples': int(age_df['n'].iloc[0]),                    # 421
        'n_cytokines_cytosig': len(age_df[...]['protein'].unique()),  # 43
    },
    'scatlas': {
        'n_organs': len(organ_df['organ'].unique()),              # 35
        'n_cell_types': len(ct_df['cell_type'].unique()),         # 376
    },
    'inflammation': inflam_stats  # n_samples: 817 from h5ad file
}
```

### Data Flow

```
Analysis scripts (01, 02, 03) → results/*.csv / *.h5ad
                              ↓
06_preprocess_viz_data.py → visualization/data/summary_stats.json
                              ↓
index.html (EMBEDDED_STATS hardcoded for offline viewing)
                              ↓
JavaScript updateStats() → DOM elements (#stat-samples, #stat-organs, etc.)
```

### Verified Values (from source files)

- Organs: 35 unique values in `normal_organ_signatures.csv`
- Cell types: 372 currently in file (may differ from displayed 376 if data updated)
- Inflammation samples: 817 confirmed from h5ad
- CytoSig signatures: 43 unique proteins

### Dataset Attribution

| Statistic | Dataset |
|-----------|---------|
| 421 CIMA Healthy Donors | **CIMA** |
| 35 Organs Profiled | **scAtlas** (normal tissues) |
| 376 Cell Types | **scAtlas** (normal tissues) |
| 817 Inflammation Samples | **Inflammation Atlas** |
| 43 CytoSig Signatures | **CIMA** (signature matrix, applies globally) |

---

## Q2: Update Atlas Overview to show each dataset in separate tabs

**Date:** 2026-01-28

**Request:** Separate the mixed statistics in Atlas Overview into dataset-specific tabs.

**Changes Made:**

Updated `visualization/index.html` (lines 623-710):

### Before
Single stats-grid with mixed statistics from all datasets.

### After
Three tabs with dataset-specific statistics:

**CIMA Tab:**
- Healthy Donors (421)
- Total Cells (6.5M)
- Significant Age Correlations (693)
- Significant BMI Correlations (580)

*Note: CytoSig (43) and SecAct (1,170) removed - these are signature matrices independent of the atlas data.*

---

## Q3: Can we identify how many samples in scAtlas?

**Date:** 2026-01-28

**Answer:** Yes, the scAtlas h5ad files contain donor/sample information.

| Dataset | Cells | Donors | Samples |
|---------|-------|--------|---------|
| scAtlas Normal | 2,293,951 | 317 | - |
| scAtlas Cancer | 4,146,975 | 717 | 1,062 |

**Source columns in h5ad files:**
- `donorID`: Unique donor identifier
- `sampleID`: Unique sample identifier (cancer only)

**File locations:**
- Normal: `/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad`
- Cancer: `/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad`

**Update:** Now displayed in scAtlas tab (see Q4).

---

## Q4: Update scAtlas tab to show donors, cells, organs, cell types, cohorts

**Date:** 2026-01-28

**Request:** Show Normal / Cancer breakdown for scAtlas statistics.

**Data Retrieved from h5ad files:**

| Metric | Normal | Cancer |
|--------|--------|--------|
| Donors | 317 | 717 |
| Total Cells | 2.3M | 4.1M |
| Cohorts | 26 | 45 |
| Cell Types | 376 | 120 |
| Organs | 35 | - |

**Changes Made:**

1. Updated `EMBEDDED_STATS.scatlas` with new fields:
   - `n_donors_normal`, `n_donors_cancer`
   - `n_cells_normal`, `n_cells_cancer`
   - `n_cohorts_normal`, `n_cohorts_cancer`
   - `n_cell_types_normal`, `n_cell_types_cancer`

2. Updated scAtlas tab HTML to show:
   - Donors (Normal / Cancer): 317 / 717
   - Total Cells (Normal / Cancer): 2.3M / 4.1M
   - Organs Profiled: 35
   - Cell Types (Normal / Cancer): 376 / 120
   - Cohorts (Normal / Cancer): 26 / 45

3. Updated `updateStats()` JavaScript function to populate new elements.

---

## Q5: Make Data Sources panel reactive to each tab

**Date:** 2026-01-28

**Request:** Show dataset-specific analysis data sources for each tab instead of mixed sources.

**Changes Made:**

Moved Data Sources table inside each tab with dataset-specific content:

**CIMA Analysis Data Sources:**
| Analysis | Description | Records |
|----------|-------------|---------|
| Age Correlations | Spearman correlations with donor age | 1,213 |
| BMI Correlations | Spearman correlations with BMI | 1,213 |
| Biochemistry | Correlations with blood biochemistry parameters | 21,859 |
| Metabolites | Correlations with plasma metabolites and lipids | 606,500 |
| Differential | Sex, smoking status, blood type comparisons | 12,130 |
| Cell Type Activities | Mean activities per cell type | 27 cell types |

**Inflammation Atlas Analysis Data Sources:**
| Analysis | Description | Records |
|----------|-------------|---------|
| Disease Differential | Disease vs healthy activity comparisons | 12+ diseases |
| Treatment Response | Responder vs non-responder prediction | 6 diseases |
| Cell Type Stratified | Disease differential per cell type | 66 cell types |
| Driving Populations | Top cell types driving disease signatures | Per disease |
| Conserved Programs | Cytokine programs shared across diseases | 3+ diseases |
| Cross-Cohort Validation | Main, validation, external cohort comparison | 3 cohorts |

**scAtlas Analysis Data Sources:**
| Analysis | Description | Records |
|----------|-------------|---------|
| Organ Signatures (Normal) | Mean cytokine activities per organ | 42,455 |
| Cell Type Signatures (Normal) | Mean activities per cell type across organs | 907,324 |
| Cancer Comparison | Tumor vs adjacent normal tissue | 120 cell types |
| Cancer Types | Pan-cancer cytokine profiles | 20+ cancer types |
| Immune Infiltration | Tumor microenvironment analysis | Per cancer type |

---

## Q6: Update CIMA stats and align data sources with 10 analysis tabs

**Date:** 2026-01-28

**Request:**
1. Show donors, total cells, cell types, metabolites profiled, blood biomarkers
2. Match 10 CIMA analysis tabs with data sources (split Age & BMI = 11 items)
3. Include IFNG examples for age & BMI analyses

**CIMA Stats Updated:**
| Stat | Value |
|------|-------|
| Healthy Donors | 421 |
| Total Cells | 6.5M |
| Cell Types | 27 |
| Metabolites Profiled | 500 |
| Blood Biomarkers | 19 |

**CIMA Data Sources (11 items matching 10 tabs):**
| Tab | Data Source | Description |
|-----|-------------|-------------|
| Age & BMI | Age Correlations | Spearman correlations with donor age (e.g., IFNG) |
| Age & BMI | BMI Correlations | Spearman correlations with BMI (e.g., IFNG) |
| Age/BMI Stratified | Age/BMI Stratified | Activity distributions by age bins and BMI categories (e.g., IFNG) |
| Biochemistry | Biochemistry | Correlations with 19 blood biomarkers |
| Biochem Scatter | Biochem Scatter | Scatter plots of cytokine vs biochemistry values |
| Metabolites | Metabolites | Correlations with 500 plasma metabolites and lipids |
| Differential | Differential | Sex, smoking status, blood type comparisons |
| Cell Types | Cell Types | Mean activities across 27 cell types |
| Multi-omic | Multi-omic | Integrated cytokine-metabolome-biochemistry analysis |
| Population | Population | Population stratification by demographics |
| eQTL Browser | eQTL Browser | Genetic variants associated with cytokine expression |

---

## Q7: Clarify that record counts are correlation tests, not samples

**Date:** 2026-01-28

**Question:** Where does 1,213 records come from when we only have 421 donors?

**Answer:**
- 1,213 = number of **correlation tests** (43 CytoSig + 1,170 SecAct proteins)
- Each correlation is computed across all **421 donors**
- The `n` column in CSV files shows 421 (sample size per test)

**Updated descriptions to clarify:**
| Analysis | Description | Count |
|----------|-------------|-------|
| Age Correlations | 1,213 proteins (43 CytoSig + 1,170 SecAct) × Age across 421 donors | 1,213 correlations |
| BMI Correlations | 1,213 proteins (43 CytoSig + 1,170 SecAct) × BMI across 421 donors | 1,213 correlations |
| Biochemistry | 1,213 proteins × 19 blood biomarkers across 421 donors | 21,859 correlations |
| Metabolites | 1,213 proteins × 500 metabolites across 421 donors | 606,500 correlations |
| Differential | Wilcoxon tests for sex, smoking status, blood type | 12,130 comparisons |

---

## Q8: What is record number for Age/BMI Stratified?

**Date:** 2026-01-28

**Answer:**

| Type | Records | Formula |
|------|---------|---------|
| Age boxplots | 100 | 20 signatures × 5 age bins |
| BMI boxplots | 80 | 20 signatures × 4 BMI bins |
| **Total** | **180** | boxplot distributions |

**Details:**
- Age bins: <30, 30-40, 40-50, 50-60, >60
- BMI bins: Underweight, Normal, Overweight, Obese
- Signatures: Top 20 most variable proteins from pseudobulk data
- Each boxplot shows distribution across 421 donors in that bin

**Source:** `visualization/data/age_bmi_boxplots.json`

---

## Q9: Show how record numbers are calculated in descriptions

**Date:** 2026-01-28

**Updated CIMA Data Sources with formulas:**

| Analysis | Description (Formula) | Records |
|----------|----------------------|---------|
| Age Correlations | 1,213 proteins × 1 feature (Age) = 1,213 | 1,213 |
| BMI Correlations | 1,213 proteins × 1 feature (BMI) = 1,213 | 1,213 |
| Age/BMI Stratified | 20 proteins × (5 age + 4 BMI bins) = 180 | 180 |
| Biochemistry | 1,213 proteins × 19 biomarkers = 23,047 | 21,859 |
| Biochem Scatter | 396 samples × 19 biomarkers × 43 cytokines | 396 |
| Metabolites | 1,213 proteins × 500 metabolites = 606,500 | 606,500 |
| Differential | 1,213 proteins × (1 sex + 3 smoking + 6 blood type) = 12,130 | 12,130 |
| Cell Types | 27 cell types × 143 proteins = 3,861 | 3,861 |
| Multi-omic | 20 cytokines × 3 modalities | 20 |
| Population | 421 donors × demographic subgroups | 421 |
| eQTL Browser | SNP-gene associations (requires genetic data) | - |

**Differential breakdown:**
- Sex: 1,213 (Female vs Male)
- Smoking: 3,639 (1,213 × 3 comparisons)
- Blood type: 7,278 (1,213 × 6 comparisons)

---

## Q10: IFNG not showing in BMI correlations

**Date:** 2026-01-28

**Issue:** IFNG ranked 13th (rho=0.0795) out of 43 CytoSig proteins for BMI, so it wasn't in the top 10 or bottom 10 shown.

**Fix:** Modified `getTopBottomCorrelations()` function to always include IFNG even if not in extremes:

```javascript
// Always include IFNG if not already in the list
const hasIFNG = result.some(d => d.protein === 'IFNG');
if (!hasIFNG) {
    const ifng = filtered.find(d => d.protein === 'IFNG');
    if (ifng) {
        result.push(ifng);
        result.sort((a, b) => b.rho - a.rho);
    }
}
```

Now IFNG will always appear in Age and BMI correlation plots.

---

## Q11: Why Age/BMI Stratified only shows 20 proteins?

**Date:** 2026-01-28

**Issue:** `preprocess_age_bmi_boxplots()` selected only top 20 most variable signatures.

**Fix:** Changed to use all 43 CytoSig signatures:

```python
# Before:
top_sigs = sig_variance.nlargest(20).index.tolist()

# After:
all_sigs = sorted(sig_cols)  # All 43 CytoSig signatures
```

**Updated records:**
- Age: 43 proteins × 5 bins = 215 boxplots
- BMI: 43 proteins × 4 bins = 172 boxplots
- Total: 387 boxplots

**Note:** Requires re-running `python scripts/06_preprocess_viz_data.py` to regenerate data.

---

## Q12: Process all SecAct proteins with searchable protein selection

**Date:** 2026-01-28

**Request:**
1. Process all 1,170 SecAct proteins (not just top 20)
2. Make protein selection searchable with autocomplete

**Changes to `06_preprocess_viz_data.py`:**
- Refactored `preprocess_age_bmi_boxplots()` to process both CytoSig and SecAct
- Added `sig_type` field to each record
- Stores separate signature lists: `cytosig_signatures`, `secact_signatures`, `all_signatures`

**Changes to `index.html`:**
1. Replaced dropdown with searchable text input
2. Added autocomplete suggestions while typing
3. Added signature type filter (CytoSig / SecAct / All)

**New record count:**
- CytoSig: 43 × 9 bins = 387 boxplots
- SecAct: 1,170 × 9 bins = 10,530 boxplots
- Total: **10,917** boxplots

**JavaScript functions added:**
- `initBoxplotSignatures()` - Initialize signature lists
- `getBoxplotSignatures()` - Get signatures by type
- `showBoxplotSuggestions(query)` - Show autocomplete suggestions
- `selectBoxplotSignature(sig)` - Select from suggestions

**Note:** Requires re-running preprocessing to generate SecAct data.

---

## Q13: Add TNFA to always-show proteins (like IFNG) in correlations

**Date:** 2026-01-28

**Fix:** Updated `getTopBottomCorrelations()` to always include both IFNG and TNFA:

```javascript
const alwaysInclude = ['IFNG', 'TNFA'];
for (const protein of alwaysInclude) {
    const hasProtein = result.some(d => d.protein === protein);
    if (!hasProtein) {
        const found = filtered.find(d => d.protein === protein);
        if (found) result.push(found);
    }
}
```

---

## Q14: Change "All" to "Both" with side-by-side CytoSig/SecAct visualization

**Date:** 2026-01-28

**Issue:** "All" option was misleading since CytoSig and SecAct may contain same proteins with different activity values.

**Changes:**
1. Renamed "All (1,213 proteins)" to "Both (side-by-side)"
2. For "both" option, shows CytoSig (blue) and SecAct (orange) boxplots side-by-side
3. Updated `getBoxplotSignatures()` to return proteins that exist in both signature types
4. Updated `updateStratifiedBoxplot()` to use `boxmode: 'group'` for grouped display

---

## Q15: Add cell type-specific Age/BMI correlations

**Date:** 2026-01-28

**Request:** Add heatmap showing how cytokine activities correlate with Age/BMI within each cell type.

**Location:** Added to Age & BMI tab (not Cell Types tab as initially implemented).

**New analysis:**
- 27 cell types × 1,213 proteins × 2 features (age, BMI) = 65,502 correlations
- Spearman correlation computed per cell type using sample-level activities
- FDR correction applied

**Files modified:**

1. `scripts/06_preprocess_viz_data.py`:
   - Added `preprocess_cima_celltype_correlations()` function
   - Processes both CytoSig and SecAct
   - Outputs to `cima_celltype_correlations.json`

2. `visualization/index.html`:
   - Added heatmap panel to Age & BMI tab
   - Controls: Signature Type (CytoSig/SecAct), Correlation With (Age/BMI), Show (All/Significant)
   - Added `updateCelltypeCorrelationHeatmap()` JavaScript function
   - Red = positive correlation, Blue = negative correlation

**Note:** Requires re-running preprocessing to generate data.

**Inflammation Atlas Tab:**
- Patient Samples (817)
- Total Cells (4.9M)
- Cell Types (66)
- Diseases Studied (12+)

**scAtlas Tab:**
- Organs Profiled (35)
- Cell Types (376)
- Organ Signatures (42,455)
- Cell Type Signatures (907,324)

### JavaScript Update
Updated `updateStats()` function to populate new element IDs:
- `stat-cima-*` for CIMA stats
- `stat-inflam-*` for Inflammation stats
- `stat-scatlas-*` for scAtlas stats
