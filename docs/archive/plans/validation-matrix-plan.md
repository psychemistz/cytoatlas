# CytoAtlas Validation Matrix — Formal Plan

## Purpose

Systematic validation of cytokine/secreted protein activity inference across **3 signature types**, **8 data sources**, **multiple aggregation levels**, and **3 validation strategies** (standard, resampled, single-cell). The core question: *Does predicted activity correlate with actual gene expression across independent samples?*

---

## 1. Validation Dimensions

### 1.1 Signature Types (3)

| Signature | Targets | Description |
|-----------|---------|-------------|
| **CytoSig** | 43 cytokines | Narrow, curated cytokine signatures |
| **LinCytoSig** | 178 (celltype-annotated) | Context-specific cytokine signatures (`CellType__Cytokine`) |
| **SecAct** | 1,170 secreted proteins | Comprehensive secretome coverage |

### 1.2 Data Sources (8)

| Source | Type | Samples | Cells | Script |
|--------|------|---------|-------|--------|
| **CIMA** | Single-cell | 421 donors | 6.5M | 12 |
| **Inflammation Main** | Single-cell | 817 donors | ~6.3M | 12 |
| **Inflammation Val** | Single-cell | 144 donors | ~6.3M | 12 |
| **Inflammation Ext** | Single-cell | 86 donors | ~6.3M | 12 |
| **scAtlas Normal** | Single-cell | ~500 donors | ~6.4M | 12 |
| **scAtlas Cancer** | Single-cell | ~1,062 donors | ~4.1M | 12 |
| **GTEx** | Bulk RNA-seq | 19,788 samples | — | 15 |
| **TCGA** | Bulk RNA-seq | 11,069 samples | — | 15 |

### 1.3 Aggregation Levels (per source)

#### CIMA (5 levels)
| Level | Grouping | Samples | Description |
|-------|----------|---------|-------------|
| `donor_only` | donor | 421 | All cell types pooled per donor |
| `donor_l1` | donor × `cell_type_l1` | 2,525 | Broad cell types (T, B, Mono, ...) |
| `donor_l2` | donor × `cell_type_l2` | 10,251 | Intermediate (CD4+, CD8+, ...) |
| `donor_l3` | donor × `cell_type_l3` | 14,304 | Fine (naive CD4+, activated CD4+, ...) |
| `donor_l4` | donor × `cell_type_l4` | 24,741 | Ultra-fine (central memory CD4+, ...) |

#### Inflammation Atlas — main/val/ext (3 levels each)
| Level | Grouping | Samples (main/val/ext) | Description |
|-------|----------|------------------------|-------------|
| `donor_only` | donor | 817 / 144 / 86 | All cell types pooled |
| `donor_l1` | donor × `Level1`/`Level1pred` | 11,327 / 1,617 / 948 | Broad cell types |
| `donor_l2` | donor × `Level2`/`Level2pred` | 30,227 / 4,540 / 2,754 | Intermediate cell types |

#### scAtlas Normal & Cancer (3 levels each)
| Level | Grouping | Samples (normal/cancer) | Description |
|-------|----------|------------------------|-------------|
| `donor_organ` | donor × `tissue` | ~800 / 1,062 | Per organ/tissue |
| `donor_organ_celltype1` | donor × `tissue` × `cellType1` | ~5,500 / 7,498 | Per organ + fine celltype |
| `donor_organ_celltype2` | donor × `tissue` × `cellType2` | ~3,600 / 2,685 | Per organ + ultra-fine subtype |

#### GTEx (2 levels)
| Level | Grouping | Samples | Description |
|-------|----------|---------|-------------|
| `donor_only` | sample | 19,788 | All tissues pooled |
| `by_tissue` | tissue_type (29 groups) | 19,759 (≥30/group) | Within-tissue normalization |

#### TCGA (2 levels)
| Level | Grouping | Samples | Description |
|-------|----------|---------|-------------|
| `donor_only` | sample | 11,069 | All cancer types pooled |
| `by_cancer` | cancer_type (≥30/group) | TBD | Within-cancer-type normalization |

### 1.4 Validation Strategies (3)

| Strategy | Description | Script | Output |
|----------|-------------|--------|--------|
| **Standard** | Single pseudobulk, Spearman rho per target | 12 → 13 | Per-target rho, pval |
| **Resampled** | 100 bootstrap resamples, CI on rho | 16 | Per-target rho + 95% CI |
| **Single-cell** | Per-cell expression vs activity | (atlas_validation) | Cell-level scatter |

---

## 2. Full Validation Matrix

### 2.1 Standard Cross-Sample Correlations

Each cell = {3 signatures × N targets each}. Metric: Spearman rho (expression vs predicted activity).

| Source | donor_only | L1/organ | L2/organ+ct1 | L3/organ+ct2 | L4 |
|--------|-----------|----------|------------|------------|-----|
| **CIMA** | DONE | DONE | DONE | DONE | DONE |
| **Inflam Main** | DONE | DONE | DONE | — | — |
| **Inflam Val** | DONE | DONE | DONE | — | — |
| **Inflam Ext** | DONE | DONE | DONE | — | — |
| **scAtlas Normal** | — | DONE | DONE | DONE | — |
| **scAtlas Cancer** | — | REGEN | REGEN | REGEN | — |
| **GTEx** | DONE | DONE (by_tissue) | — | — | — |
| **TCGA** | PENDING | PENDING (by_cancer) | — | — | — |

**Notes:**
- scAtlas has no `donor_only` (would be meaningless — each donor has only 1-2 organs)
- CIMA L3/L4 and Inflammation L2 are the finest available per atlas
- scAtlas Cancer REGEN = regenerating (SLURM job 11198496, zscore bug fix)
- TCGA PENDING = waiting on SLURM job 11209412

### 2.2 Resampled Bootstrap Validation

Input: resampled pseudobulk from `results/atlas_validation/`. 100 bootstrap iterations, 80% cell subsampling.

| Source | Resampled Levels Available | Activity Inference | Correlation CSVs |
|--------|---------------------------|-------------------|-----------------|
| **CIMA** | L1, L2, L3, L4 + L1_donor | NOT DONE | NOT DONE |
| **Inflam Main** | L1, L2 + L1_donor, L2_donor | NOT DONE | NOT DONE |
| **Inflam Val** | L1, L2 + L1_donor, L2_donor | NOT DONE | NOT DONE |
| **Inflam Ext** | L1, L2 + L1_donor, L2_donor | NOT DONE | NOT DONE |
| **scAtlas Normal** | celltype, donor_celltype, organ_celltype, donor_organ_celltype | NOT DONE | NOT DONE |
| **scAtlas Cancer** | celltype, donor_celltype, organ_celltype, donor_organ_celltype | NOT DONE | NOT DONE |
| **GTEx** | N/A (bulk, no cell-level resampling) | — | — |
| **TCGA** | N/A (bulk, no cell-level resampling) | — | — |

**Status:** Resampled pseudobulk H5AD files exist for all 6 single-cell atlases. Script 16 (`16_resampled_validation.py`) is written but has NOT been run. No activity H5AD or correlation CSV outputs yet.

### 2.3 LinCytoSig Cell-Type-Restricted Correlations

Special validation for LinCytoSig: correlate activity only within the cell type matching the signature's context annotation (e.g., `Monocyte__IL6` → correlate only in monocyte samples).

| Source | Matched Correlations | Status |
|--------|---------------------|--------|
| **CIMA** | L1-L4 | DONE (in cima_correlations.csv) |
| **Inflam Main** | L1-L2 | DONE |
| **Inflam Val** | L1-L2 | DONE |
| **Inflam Ext** | L1-L2 | DONE |
| **scAtlas Normal** | organ_celltype1/2 | DONE |
| **scAtlas Cancer** | organ_celltype1/2 | REGEN |
| **GTEx** | by_tissue | DONE |
| **TCGA** | by_cancer | PENDING |

### 2.4 Single-Cell Validation

Per-cell expression vs predicted activity (from earlier atlas_validation pipeline).

| Source | Status | Notes |
|--------|--------|-------|
| **CIMA** | EXISTS | In `cima_validation.json` (167 MB) |
| **Inflammation** | EXISTS | In `inflammation_validation.json` (174 MB) |
| **scAtlas** | EXISTS | In `scatlas_validation.json` (321 MB) |

---

## 3. Pipeline Architecture

### 3.1 Script Dependency Chain

```
Raw H5AD / Bulk Data
        │
        ├── Script 12: Pseudobulk aggregation + activity inference (single-cell)
        │       └── Output: {atlas}_{level}_pseudobulk.h5ad + {atlas}_{level}_{sig}.h5ad
        │
        ├── Script 15: Bulk RNA-seq activity inference (GTEx/TCGA)
        │       └── Output: {dataset}_{level}_expression.h5ad + {dataset}_{level}_{sig}.h5ad
        │
        ├── Script 16: Resampled activity inference + bootstrap CIs
        │       └── Input: atlas_validation/*_resampled.h5ad
        │       └── Output: {atlas}_resampled_{level}_{sig}.h5ad + {atlas}_resampled_{level}_correlations.csv
        │
        ▼
Script 13: Correlation analysis (Spearman rho, per-target, per-celltype)
        │       └── Input: H5AD files from scripts 12/15
        │       └── Output: {atlas}_correlations.csv + correlation_summary.csv + all_correlations.csv
        ▼
Script 14: JSON preprocessing for visualization
        │       └── Input: correlation CSVs + H5AD files + (optional) resampled CSVs
        │       └── Output: bulk_donor_correlations.json + bulk_rnaseq_validation.json
        ▼
visualization/index.html (5 tabs: Bulk, Donor, CellType, CellType-Resampled, SingleCell)
```

### 3.2 Key Design Decisions

- **Within-celltype normalization**: For celltype-stratified levels, genes are mean-centered *within* each celltype before ridge regression. This isolates donor-level variance from celltype-level variance.
- **Ridge regression**: λ=5e5, n_rand=1000 permutations, z-score output
- **Expression normalization**: Single-cell → log1p(CPM); Bulk → log2(TPM+1)
- **Gene name mapping**: CytoSig uses non-HGNC names (e.g., `TNFA` → `TNF`), mapped via `signature_gene_mapping.json`
- **Size optimization**: JSON scatter data subsampled to ≤2000 points; LinCytoSig/SecAct limited to top 30 targets by |rho|

---

## 4. Current Status & Gaps

### 4.1 Completed

| Component | Files | Status |
|-----------|-------|--------|
| CIMA pseudobulk + activity (5 levels × 3 sigs) | 20 H5AD | DONE |
| Inflammation main/val/ext (3 levels × 3 sigs each) | 36 H5AD | DONE |
| scAtlas Normal (3 levels × 3 sigs) | 12 H5AD | DONE |
| GTEx donor_only (3 sigs) | 4 H5AD | DONE |
| GTEx by_tissue CytoSig | 1 H5AD | DONE |
| All standard correlations (excl. TCGA, scAtlas cancer regen) | 8 CSV | DONE |
| Visualization JSON (bulk_donor_correlations.json) | 12 MB | DONE |
| Single-cell validation JSONs | 662 MB | DONE (earlier pipeline) |
| Resampled pseudobulk generation | 24 H5AD | DONE (atlas_validation/) |

### 4.2 In Progress (SLURM Jobs Running)

| Job | ID | Runtime | What Remains |
|-----|----|---------|-------------|
| scAtlas Cancer regen | 11198496 | ~6.5h | SecAct for donor_organ_celltype1, then all of donor_organ_celltype2 |
| GTEx/TCGA bulk | 11209412 | ~2.5h | GTEx by_tissue LinCytoSig/SecAct → TCGA full pipeline → correlations → JSON |

### 4.3 Not Started

| Component | Blocker | Script | Est. Time |
|-----------|---------|--------|-----------|
| **Resampled activity inference** (all 6 SC atlases) | None — resampled pseudobulk exists | 16 | 6-12h (GPU) |
| **Resampled correlation CSVs** | Needs resampled activity | 16 | Included above |
| **scAtlas cancer correlation refresh** | Waiting on job 11198496 | 13 | 30 min |
| **TCGA full pipeline** | Waiting on job 11209412 | 15 → 13 | Included in job |
| **JSON refresh** (incl. TCGA + resampled) | Needs all above | 14 | 15 min |

### 4.4 Known Gaps / Missing Combinations

1. **scAtlas `donor_only`**: Not computed (multi-organ donors make global pooling uninformative). This is intentional.
2. **CIMA L3/L4 in Inflammation**: Only L1/L2 exist (Inflammation Atlas has 2-level hierarchy). This is a data limitation.
3. **Resampled for bulk**: N/A (no cell-level resampling for GTEx/TCGA). This is by design.
4. **GTEx SecAct by_tissue**: Missing — SecAct by_tissue inference not yet complete. Will be done by job 11209412.
5. **Inflammation ext SecAct by level**: Missing `inflammation_ext_donor_l1_secact.h5ad` (exists on disk but only 11 files vs 12 for other atlases — need verification).

---

## 5. Execution Plan

### Phase 1: Complete In-Flight Jobs (ETA: today)

**Wait for SLURM jobs to finish:**
- Job 11198496 (`regen_ca`): scAtlas cancer pseudobulk + activity regeneration
- Job 11209412 (`bulk_val`): GTEx by_tissue + full TCGA pipeline

**After jobs complete:**
```bash
# Refresh scAtlas cancer correlations
python scripts/13_cross_sample_correlation_analysis.py --atlas scatlas_cancer

# If TCGA finished in bulk_val job, verify outputs
ls -la results/cross_sample_validation/tcga/
ls -la results/cross_sample_validation/correlations/tcga_correlations.csv
```

### Phase 2: Resampled Validation (new SLURM job)

Submit script 16 for all single-cell atlases:

```bash
# Create SLURM script: scripts/slurm/run_resampled_validation.sh
# Resources: 200GB RAM, 1 GPU, 8 CPUs, 24h
# Pipeline:
#   Step 1: python scripts/16_resampled_validation.py --atlas all
#   Step 2: (correlations computed within script 16)
```

**Expected outputs:**
- Activity H5AD: `{atlas}_resampled_{level}_{sig}.h5ad` (18 atlas×level combinations × 3 sigs = 54 files)
- Correlation CSV: `{atlas}_resampled_{level}_correlations.csv` (18 files)

### Phase 3: Final JSON Assembly

After Phases 1-2 complete:

```bash
python scripts/14_preprocess_bulk_validation.py
# Produces:
#   visualization/data/bulk_donor_correlations.json  (all atlases + resampled)
#   visualization/data/bulk_rnaseq_validation.json   (GTEx + TCGA)
```

### Phase 4: Verification

**Correlation sanity checks:**
1. All 8 sources × available levels × 3 signatures present in `correlation_summary.csv`
2. GTEx SecAct donor_only: median_rho ≈ 0.40 (strongest expected)
3. Resampled CIs: 95% CI width < 0.15 for well-powered targets
4. TCGA correlations: comparable to GTEx (external replication)
5. LinCytoSig matched correlations: higher rho than unmatched for well-characterized targets

**Expected final validation matrix size:**
- Standard correlations: ~70 rows in summary (8 sources × 2-5 levels × 3 sigs, minus N/A)
- Resampled correlations: ~54 rows (6 SC sources × 2-4 levels × 3 sigs)
- Total: ~124 summary rows covering the full matrix

---

## 6. Summary Table: Full Matrix

```
                        Standard                    Resampled               Single-Cell
Source          donor  L1/organ  L2/ct1  L3/ct2  L4   L1  L2  L3  L4      per-cell
─────────────── ────── ──────── ─────── ─────── ──── ─── ─── ─── ───── ─────────
CIMA             ✅     ✅       ✅      ✅     ✅   ⬜  ⬜  ⬜  ⬜      ✅
Inflam Main      ✅     ✅       ✅      —      —    ⬜  ⬜  —   —       ✅
Inflam Val       ✅     ✅       ✅      —      —    ⬜  ⬜  —   —       ✅
Inflam Ext       ✅     ✅       ✅      —      —    ⬜  ⬜  —   —       ✅
scAtlas Normal   —      ✅       ✅      ✅     —    ⬜  ⬜  ⬜  —       ✅
scAtlas Cancer   —      🔄       🔄      🔄     —    ⬜  ⬜  ⬜  —       ✅
GTEx             ✅     ✅ᵗ      —       —      —    —   —   —   —       —
TCGA             ⏳     ⏳ᶜ      —       —      —    —   —   —   —       —

✅ = Complete    🔄 = Regenerating    ⏳ = In progress    ⬜ = Not started    — = N/A
ᵗ = by_tissue    ᶜ = by_cancer

Each cell = CytoSig + LinCytoSig + SecAct (3 signature types)
```

---

## 7. File Inventory (Expected Final State)

### H5AD Files: `results/cross_sample_validation/`

| Directory | Standard Files | Resampled Files | Total |
|-----------|---------------|-----------------|-------|
| cima/ | 20 | 15 (5 levels × 3 sigs) | 35 |
| inflammation_main/ | 12 | 6 (2 levels × 3 sigs) | 18 |
| inflammation_val/ | 12 | 6 | 18 |
| inflammation_ext/ | 12 | 6 | 18 |
| scatlas_normal/ | 12 | 9 (3 levels × 3 sigs) | 21 |
| scatlas_cancer/ | 12 | 9 | 21 |
| gtex/ | 9 | — | 9 |
| tcga/ | 9 | — | 9 |
| **Total** | **98** | **51** | **149** |

### Correlation CSVs: `results/cross_sample_validation/correlations/`

| File | Description |
|------|-------------|
| `cima_correlations.csv` | Standard: 5 levels × 3 sigs |
| `inflammation_main_correlations.csv` | Standard: 3 levels × 3 sigs |
| `inflammation_val_correlations.csv` | Standard: 3 levels × 3 sigs |
| `inflammation_ext_correlations.csv` | Standard: 3 levels × 3 sigs |
| `scatlas_normal_correlations.csv` | Standard: 3 levels × 3 sigs |
| `scatlas_cancer_correlations.csv` | Standard: 3 levels × 3 sigs |
| `gtex_correlations.csv` | Standard: 2 levels × 3 sigs |
| `tcga_correlations.csv` | Standard: 2 levels × 3 sigs |
| `cima_resampled_{l1..l4}_correlations.csv` | Resampled: 4 files |
| `inflammation_main_resampled_{l1,l2}_correlations.csv` | Resampled: 2 files |
| `inflammation_val_resampled_{l1,l2}_correlations.csv` | Resampled: 2 files |
| `inflammation_ext_resampled_{l1,l2}_correlations.csv` | Resampled: 2 files |
| `scatlas_normal_resampled_{ct,dct,oct}_correlations.csv` | Resampled: 3 files |
| `scatlas_cancer_resampled_{ct,dct,oct}_correlations.csv` | Resampled: 3 files |
| `all_correlations.csv` | Combined standard |
| `correlation_summary.csv` | Aggregated statistics |
| **Total** | **~26 CSV files** |

### Visualization JSONs: `visualization/data/`

| File | Content |
|------|---------|
| `bulk_donor_correlations.json` | All SC atlases + GTEx: donor & celltype scatter, summary, resampled |
| `bulk_rnaseq_validation.json` | GTEx + TCGA: tissue/cancer-stratified scatter |
| `validation/cima_validation.json` | Single-cell level (existing) |
| `validation/inflammation_validation.json` | Single-cell level (existing) |
| `validation/scatlas_validation.json` | Single-cell level (existing) |
