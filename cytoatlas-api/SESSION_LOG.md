# CytoAtlas API Development Session Log

## Session Date: 2026-01-28 (Initial) | 2026-01-31 (Updated)

---

## Validation System: CytoSig/SecAct Inference Credibility

### Purpose

The validation system enables users to understand **how credible CytoSig and SecAct inference results are** for each atlas. This is critical because:
- Activity scores are inferred from gene expression using ridge regression
- The inference assumes signature genes are expressed and measurable
- Different atlases have varying data quality, cell type coverage, and gene detection rates

### The 5-Type Validation Framework

Based on user requirements, we implemented 5 levels of validation:

#### Type 1: Sample-Level Validation
**Question**: Does pseudobulk cytokine gene expression correlate with predicted activity across samples?

**Process**:
1. Generate pseudobulk expression by aggregating cells per sample
2. Compute sample-level CytoSig/SecAct activity predictions
3. For each signature, compare the signature gene's expression vs predicted activity
4. Visualize with scatter plot + regression line

**Output**: Spearman/Pearson correlation, R², p-value per signature
**Expected**: r > 0.5 for valid signatures (expression should predict activity)

#### Type 2: Cell Type-Level Validation
**Question**: Does cell type pseudobulk expression match cell type activity patterns?

**Process**:
1. Generate pseudobulk expression by aggregating cells per cell type
2. Compute cell type-level activity predictions
3. Compare expression vs activity across all cell types
4. Check if known producer cell types rank highest

**Output**: Correlation stats + biological concordance (do expected cell types rank high?)
**Expected**: IL17A should be high in Th17, IFNG in CD8/NK, TNF in Monocytes, etc.

#### Type 3: Pseudobulk vs Single-Cell Comparison
**Question**: Are aggregation methods consistent?

**Process**:
1. Generate cell type-level pseudobulk expression
2. Compute single-cell level activities, then aggregate to mean/median per cell type
3. Compare pseudobulk expression with mean/median single-cell activity

**Output**: Two scatter plots (vs mean, vs median) with correlations
**Expected**: High correlation indicates consistent inference across aggregation methods

#### Type 4: Single-Cell Direct Validation
**Question**: Do cells expressing the cytokine gene have higher activity than non-expressing cells?

**Process**:
1. Get single-cell gene expression for each signature gene
2. Get single-cell activity predictions
3. Classify cells as "expressing" (expr > 0) or "non-expressing"
4. Compare activity distributions between groups

**Output**: Box plots, fold change, Mann-Whitney p-value
**Expected**: Expressing cells should have significantly higher activity (fold change > 1.5, p < 0.05)

#### Type 5: Biological Association Validation
**Question**: Do predictions match known cytokine-cell type biology?

**Process**:
1. Define known biological associations (e.g., IL17A → Th17, IFNG → CD8/NK)
2. For each association, check if expected cell type ranks in top 3-5 for that signature
3. Count validated vs not validated associations

**Output**: Table of 12 known associations with validation status
**Expected**: >80% of known associations should validate

### Known Biological Associations (12 pairs)

| Signature | Expected Cell Type | Biological Basis |
|-----------|-------------------|------------------|
| IL17A | Th17 | Canonical Th17 cytokine |
| IFNG | CD8_CTL | IFN-γ from cytotoxic T cells |
| IFNG | NK | IFN-γ from NK cells |
| TNF | Mono | TNF from monocytes/macrophages |
| IL10 | Treg | Regulatory cytokine |
| IL4 | Th2 | Canonical Th2 cytokine |
| IL6 | Mono | Inflammatory cytokine |
| IL1B | Mono | Inflammasome activation |
| CXCL8 | Mono | IL-8 for neutrophil recruitment |
| IL21 | Tfh | Follicular helper cytokine |
| IL2 | CD4_helper | T cell growth factor |
| TGFB1 | Treg | TGF-β for immune suppression |

### Additional Validation Metrics

#### Gene Coverage
- How many signature genes are detected (non-zero) in the atlas?
- Coverage > 90% = excellent, 70-90% = good, 50-70% = moderate, < 50% = poor

#### CV Stability
- Cross-validation stability of predictions
- Low coefficient of variation = stable, reproducible predictions

### Quality Grading

The validation summary computes an overall quality score (0-100) based on:
- Sample-level correlation (20%)
- Cell type-level correlation (20%)
- Gene coverage (20%)
- CV stability (20%)
- Biological validation rate (20%)

Grades: A (>90), B (80-90), C (70-80), D (60-70), F (<60)

### Implementation Status

| Component | Status |
|-----------|--------|
| Validation schemas | Complete (596 lines) |
| Validation service | Complete (636 lines) |
| Validation router | Complete (all endpoints) |
| Validation JSON data | **GENERATED (2026-01-31)** |

### Validation Results (Generated 2026-01-31)

| Atlas | Quality Grade | Score | Biological Validation Rate | Cell Types |
|-------|---------------|-------|---------------------------|------------|
| CIMA | F | 49.9 | 41.7% (5/12) | 27 |
| Inflammation | F | 45.5 | 33.3% (4/12) | 30 |
| scAtlas | C | 72.4 | 16.7% (2/12) | 198 |

Note: The low grades (F) for CIMA and Inflammation reflect that the 5-type validation
system uses strict criteria. The biological validation rate measures how many of the
12 known cytokine-cell type associations (e.g., IL17A→Th17) are found in the top 3
ranking cell types. scAtlas has lower biological validation because it contains many
non-immune tissue-resident cells that are not part of the expected associations.

### Data File Requirements

Need to generate for each atlas:
```
visualization/data/validation/
├── cima_validation.json
├── inflammation_validation.json
└── scatlas_validation.json
```

Each file should contain:
```json
{
  "sample_validations": [...],
  "celltype_validations": [...],
  "pseudobulk_vs_sc": [...],
  "singlecell_validations": [...],
  "biological_associations": {...},
  "gene_coverage": [...],
  "cv_stability": [...]
}
```

---

## Original User Requirements (2026-01-28)

> there is one more objective in this service. that's validating cytosig & secact inference results with respect to atlas data.
> This step enable user to understand how cytosig and secact inference and association results are credible for each atlas. so, for example,
> 1) perform sample level pseudobulk generation & use that pseudobulk to predict sample level cytosig/secact signature activities. compare cytosig/secact signature gene expression level in pseudobulk samples and these activities. provide scatterplot to visualize sample level correlation between cytokine/secreted protein expression level and activities or additional visualizations
> 2) perform cell type level pseudobulk generation & use that pseudobulk to predict celltype level cytosig/secact signature activities. compare these gene levels in celltype agnostic manner and compare celltype specific activities and visualize with scatterplot and others
> 3) perform cell type level pseudobulk generation & use single cell level expressions to prediction single cell level cytosig/secact signature activities. compare pseudobulk gene expression & mean/median cytokine/secact activity levels. visualize with scatterplot & others.
> 4) perform comparison between single cell gene expression and activity inference directly, while control cytokine/secreted protein unexpressed cells. visualize.
> 5) any other process which can provide estimated credibility of cytosig/secact inference methods (reference cytosig paper)
>
> keep in mind that we are not restricting 3 atlas already archived here. the service should be also applicable to new atlas dataset published or deposited datasets from user.

---

## Session History

### 2026-01-28: Initial Setup

**Issues Fixed**:
1. `ALLOWED_ORIGINS` configuration for pydantic-settings
2. Database engine creation without DATABASE_URL
3. Health endpoint without database
4. Missing email-validator package
5. Inflammation schema mismatch

### 2026-01-31: Documentation Consolidation

**Actions**:
1. Consolidated 6 plan files into single master plan
2. Updated ARCHITECTURE.md with current API status
3. Updated CLAUDE.md with router summary
4. Added status section to README.md
5. Committed scAtlas immune analysis enhancements

---

## How to Run

```bash
cd /vf/users/parks34/projects/2secactpy/cytoatlas-api

# Activate environment
source ~/bin/myconda
conda activate secactpy

# Run server
./scripts/run_server.sh

# Access docs at http://localhost:8000/docs
```

## Verified Working Endpoints

```bash
# Health
curl --noproxy '*' http://localhost:8000/api/v1/health

# CIMA
curl --noproxy '*' http://localhost:8000/api/v1/cima/summary

# Inflammation
curl --noproxy '*' http://localhost:8000/api/v1/inflammation/diseases

# scAtlas
curl --noproxy '*' http://localhost:8000/api/v1/scatlas/organs

# Validation (returns empty until data generated)
curl --noproxy '*' http://localhost:8000/api/v1/validation/summary/cima
```
