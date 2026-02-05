# AlphaGenome eQTL Analysis Implementation Plan

## Overview

Use AlphaGenome API to prioritize regulatory variants from CIMA immune cell cis-eQTLs affecting cytokine/secreted protein expression.

## Status Summary (Updated 2026-02-05 09:00)

| Stage | Status | Actual Output |
|-------|--------|---------------|
| Stage 1 | ✅ COMPLETE | 48,627 cytokine eQTLs → 29,816 unique variants |
| Stage 2 | ✅ COMPLETE | 29,816 variants formatted (SNV/indel) |
| Stage 3 | 🔄 RUNNING | Job 10659620: 24,600/29,816 (82.5%), Job 10735825: 16,400/29,816 (55%) |
| Stage 4 | ✅ PARTIAL | 1,585 prioritized variants (from 3,713 with predictions) |
| Stage 5 | ✅ PARTIAL | GTEx: 64.7% concordance, **DICE: 75.1% concordance** |

### Current Progress

Two parallel jobs running:
- **Job 10659620**: 24,600/29,816 (82.5%) - ETA ~1 day
- **Job 10735825**: 16,400/29,816 (55.0%) - ETA ~2 days

Partial results available from intermediate checkpoint with 3,713 variants having non-zero predictions.

## Validation Results

### Direct Comparisons (Primary Validation)

**Important**: The strongest validation comes from direct comparison of CIMA eQTLs with independent datasets, NOT from AlphaGenome predictions.

| Comparison | Matches | Concordance | Pearson r | Spearman ρ |
|------------|---------|-------------|-----------|------------|
| **Direct CIMA vs DICE** | 165,064 | **83.5%** | **0.685** | **0.660** |
| **Direct CIMA vs GTEx** | 68,035 | **68.9%** | **0.370** | **0.377** |
| AlphaGenome-filtered CIMA vs DICE | 4,495 | 75.1% | 0.54 | 0.51 |
| AlphaGenome-filtered CIMA vs GTEx | 1,962 | 64.7% | 0.28 | 0.29 |

**Key findings**:
1. **DICE > GTEx**: Cell-type-specific DICE (83.5%) shows better concordance than bulk GTEx (68.9%), confirming cell-type specificity matters
2. **Direct > AlphaGenome-filtered**: In BOTH cases, direct comparison outperforms AlphaGenome-filtered (DICE: 83.5% vs 75.1%, GTEx: 68.9% vs 64.7%)
3. **AlphaGenome filtering reduces concordance**: May select variants with larger predicted effects that are more context-specific and less replicable

### AlphaGenome-Filtered Validation (Secondary)

| Source | Matches | Concordance | Pearson r | Spearman ρ | Notes |
|--------|---------|-------------|-----------|------------|-------|
| GTEx Bulk v10 | 1,962 | 64.7% | 0.28 | 0.29 | Potentially circular (same training data) |
| DICE (hg38) | 4,495 | 75.1% | 0.54 | 0.51 | Independent validation |
| GTEx ieQTL | 0 | - | - | - | Coordinate mismatch |

### DICE Concordance by Cell Type (AlphaGenome-filtered)

| Cell Type | Concordant | Total | Rate |
|-----------|------------|-------|------|
| TREG_NAIVE | 349 | 451 | 77.4% |
| TFH | 353 | 457 | 77.2% |
| TH2 | 344 | 448 | 76.8% |
| CD4_NAIVE | 321 | 420 | 76.4% |
| CD8_STIM | 181 | 237 | 76.4% |
| CD8_NAIVE | 346 | 459 | 75.4% |
| TH17 | 330 | 442 | 74.7% |
| CD4_STIM | 189 | 255 | 74.1% |
| TH1 | 240 | 325 | 73.8% |
| NK | 189 | 257 | 73.5% |
| TREG_MEM | 322 | 439 | 73.3% |
| B_CELL_NAIVE | 212 | 305 | 69.5% |

### What AlphaGenome Actually Provides

| Purpose | Useful? | Notes |
|---------|---------|-------|
| **Prioritization** | ✅ Yes | Identifies variants with largest predicted regulatory effects |
| **Validation** | ❌ No | GTEx validation is circular; DICE validation is weaker than direct comparison |
| **Mechanism annotation** | ❌ No | Only RNA tracks available, no chromatin data |

**Recommendation for paper**: Report direct CIMA-DICE concordance (83.5%) as the primary validation. Use AlphaGenome only for prioritizing variants for functional follow-up.

## Data Sources

| Resource | Location | Details |
|----------|----------|---------|
| eQTL data | `/data/Jiang_Lab/Data/Seongyong/CIMA/xQTL/CIMA_Lead_cis-xQTL.csv` | 223,405 records, 71,530 at FDR<0.05 |
| CytoSig genes | `secactpy.load_cytosig()` | 4,881 genes x 43 cytokines |
| SecAct genes | `secactpy.load_secact()` | 7,919 genes x 1,169 proteins |
| GTEx v10 | `results/alphagenome/gtex_data/` | 2,985,690 Whole Blood eQTLs |
| DICE | `results/alphagenome/dice_data/hg38/` | 3,706,208 eQTLs (12 cell types, lifted to hg38) |

## Implementation Stages

### Stage 1: Filter eQTLs to Cytokine Gene Sets ✅
**Script**: `scripts/08_alphagenome_stage1_filter.py`

**Actual results**:
- Input: 223,405 eQTLs
- FDR < 0.05: 71,530 eQTLs
- Cytokine/secreted protein genes: 48,627 eQTLs
- Unique variants: 29,816

**Output**: `results/alphagenome/stage1_cytokine_eqtls.csv`

### Stage 2: Format for AlphaGenome API ✅
**Script**: `scripts/08_alphagenome_stage2_format.py`

**Actual results**:
- 29,816 unique variants
- SNVs: 40,686 | Deletions: 4,322 | Insertions: 3,619
- Genome build: hg38

**Output**: `results/alphagenome/stage2_alphagenome_input.csv`

### Stage 3: Execute AlphaGenome Predictions 🔄
**Script**: `scripts/08_alphagenome_stage3_predict.py`

**Current job status**:
| Job ID | Node | Runtime | Progress | Completion |
|--------|------|---------|----------|------------|
| 10659620 | cn0008 | 3d 21h | 24,600/29,816 | 82.5% |
| 10735825 | cn4332 | 2d 16h | 16,400/29,816 | 55.0% |

**API behavior**:
- Returns **3 immune-relevant RNA-seq tracks** (not chromatin):
  - `RNA_SEQ_EFO:0000572 gtex Cells_EBV-transformed_lymphocytes polyA plus RNA-seq`
  - `RNA_SEQ_UBERON:0002106 gtex Spleen polyA plus RNA-seq`
  - `RNA_SEQ_UBERON:0013756 gtex Whole_Blood polyA plus RNA-seq`
- ~50% RESOURCE_EXHAUSTED rate limiting errors
- ~10-20 sec per variant (with retries)
- 12.5% of variants (3,713/29,816) have non-zero predictions

**Output**:
- `results/alphagenome/stage3_predictions_intermediate.h5ad` - 29,816 variants (3,713 with data)
- `results/alphagenome/stage3_checkpoint.json` - 24,600 variants processed

### Stage 4: Interpret and Score Predictions ✅ PARTIAL
**Script**: `scripts/08_alphagenome_stage4_interpret.py`

**Partial run results** (on intermediate data):
- Input: 29,816 variants (3,713 with non-zero predictions)
- Prioritized: **1,585 variants** (5.3%)
- Unique genes: 1,045
- Unique cell types: 69

**Variant type distribution**:
- SNV: 972 (61%)
- Deletion: 334 (21%)
- Insertion: 279 (18%)

**Top prioritized variants**:
| Variant | Gene | Cell Type | Impact | eQTL β |
|---------|------|-----------|--------|--------|
| chr6:29520616 | HLA-F | CD8_Tcm | 0.00331 | +0.55 |
| chr8:144725016 | ZNF34 | ncMono | 0.00258 | +0.34 |
| chr7:99381143 | ARPC1B | CD8_Tem | 0.00248 | -0.88 |
| chr3:52859176 | ITIH4 | CD4_Tn | 0.00241 | +0.72 |
| chr3:49813942 | UBA7 | Bn_TCL1A | 0.00235 | -0.39 |

**Output**:
- `results/alphagenome/stage4_scored_variants_partial.csv`
- `results/alphagenome/stage4_prioritized_partial.csv`
- `results/alphagenome/stage4_summary_partial.json`

### Stage 5: Multi-Source Validation ✅ PARTIAL
**Script**: `scripts/08_alphagenome_stage5_validate.py`

**Validation sources**:
1. **GTEx Bulk v10**: 2,985,690 Whole Blood eQTLs (hg38)
2. **GTEx ieQTL v8**: 20,315 Neutrophil interaction eQTLs
3. **DICE**: 3,706,208 immune cell eQTLs (12 cell types, lifted hg19→hg38)

**Results**:
| Source | Matches | Concordance | Correlation |
|--------|---------|-------------|-------------|
| GTEx Bulk | 1,962 | 64.7% | r=0.28 |
| **DICE** | **4,495** | **75.1%** | **r=0.54** |
| GTEx ieQTL | 0 | - | - |

**Output**:
- `results/alphagenome/stage5_gtex_matched_partial_v2.csv`
- `results/alphagenome/stage5_dice_matched_partial_v2.csv`
- `results/alphagenome/stage5_validation_metrics_partial_v2.json`
- `results/alphagenome/stage5_report_partial_v2.md`

## DICE Liftover (hg19 → hg38) ✅

**Script**: `scripts/liftover_dice_hg19_to_hg38.py`

DICE eQTLs were originally in hg19 coordinates, causing 0 matches with CIMA (hg38). Implemented liftover using `pyliftover`:

| Cell Type | Total | Lifted | Rate |
|-----------|-------|--------|------|
| B_CELL_NAIVE | 305,710 | 304,690 | 99.7% |
| CD4_NAIVE | 347,035 | 346,370 | 99.8% |
| CD8_NAIVE | 365,996 | 365,460 | 99.9% |
| NK | 238,011 | 237,652 | 99.8% |
| TFH | 348,119 | 347,453 | 99.8% |
| TH1 | 250,464 | 249,972 | 99.8% |
| TH2 | 339,160 | 338,514 | 99.8% |
| TH17 | 359,195 | 358,502 | 99.8% |
| TREG_MEM | 338,083 | 337,434 | 99.8% |
| TREG_NAIVE | 355,560 | 354,985 | 99.8% |
| **Total** | **3,713,804** | **3,706,208** | **99.8%** |

**Output**: `results/alphagenome/dice_data/hg38/*_hg38.tsv`

## File Structure

```
scripts/
  08_alphagenome_stage1_filter.py      ✅
  08_alphagenome_stage2_format.py      ✅
  08_alphagenome_stage3_predict.py     ✅
  08_alphagenome_stage4_interpret.py   ✅
  08_alphagenome_stage5_validate.py    ✅ (updated for hg38 DICE)
  liftover_dice_hg19_to_hg38.py        ✅ NEW
  slurm/run_alphagenome.sh             ✅

results/alphagenome/
  stage1_cytokine_eqtls.csv            ✅ 48,627 eQTLs
  stage1_summary.json                  ✅
  stage2_alphagenome_input.csv         ✅ 29,816 variants
  stage2_summary.json                  ✅
  stage3_predictions_intermediate.h5ad 🔄 29,816 variants (3,713 with data)
  stage3_checkpoint.json               🔄 24,600 variants processed
  stage4_scored_variants_partial.csv   ✅ 29,816 scored
  stage4_prioritized_partial.csv       ✅ 1,585 prioritized
  stage4_summary_partial.json          ✅
  stage5_gtex_matched_partial_v2.csv   ✅ 1,962 GTEx matches
  stage5_dice_matched_partial_v2.csv   ✅ 4,495 DICE matches
  stage5_validation_metrics_partial_v2.json ✅
  stage5_report_partial_v2.md          ✅
  gtex_data/                           ✅ V10 parquet (2,985,690 eQTLs)
  dice_data/hg38/                      ✅ Lifted DICE (3,706,208 eQTLs)
```

## SLURM Jobs

```bash
#SBATCH --job-name=alphagenome
#SBATCH --time=168:00:00   # 7 days
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=norm
```

**Active jobs**:
- 10659620: cn0008 (82.5% complete)
- 10735825: cn4332 (55.0% complete)

## Key Findings

### AlphaGenome API Limitations

| Aspect | Expected | Actual |
|--------|----------|--------|
| Track count | 7,000+ | 3 (RNA-seq only) |
| Track types | ATAC, H3K27ac, H3K4me3, ChIP | GTEx RNA-seq only |
| Mechanism classification | Enhancer/promoter/TF | Not possible |

**Implication**: AlphaGenome predictions are limited to RNA expression changes. The 3 GTEx RNA-seq tracks returned are the same data used to train the model, creating potential circularity in GTEx validation.

### AlphaGenome Filtering Does NOT Improve GTEx Concordance

Despite being trained on GTEx data, AlphaGenome filtering **reduces** concordance with GTEx:

| Comparison | Concordance |
|------------|-------------|
| Direct CIMA vs GTEx | **68.9%** |
| AlphaGenome-filtered CIMA vs GTEx | **64.7%** |
| Difference | **-4.2%** (worse) |

**Concordance by AlphaGenome impact quartile**:

| Impact Quartile | Concordance |
|-----------------|-------------|
| Q1 (low) | 65.6% |
| Q2 | 63.8% |
| Q3 | 63.9% |
| Q4 (high) | 65.6% |

- Point-biserial correlation: r = 0.034, p = 0.14 (not significant)
- Higher impact scores do NOT predict better GTEx concordance

**Why this happens**: AlphaGenome optimizes for **predicted effect magnitude**, not **cross-study replicability**. Variants with large predicted effects may be more cell-type-specific or context-dependent, making them less likely to replicate across datasets.

### Validation Interpretation

1. **Direct CIMA-DICE concordance (83.5%)**: Strong independent validation - the primary result for publication
2. **AlphaGenome-filtered concordance (75.1%)**: Lower than direct comparison - AlphaGenome filtering does not improve validation
3. **GTEx concordance (64.7%)**: Potentially circular since AlphaGenome was trained on GTEx
4. **Cell-type specificity confirmed**: Cell-type-specific DICE data replicates CIMA single-cell eQTLs better than bulk GTEx

### Top Validated Variants (Direct CIMA-DICE)

Variants with concordant effects across CIMA and DICE (10,394 unique variants matched):

| Variant | Gene | CIMA β | DICE β | Notes |
|---------|------|--------|--------|-------|
| chr3:49813942 | UBA7 | -0.39 | -0.63 | Consistent negative effect |
| chr3:40411308 | RPL14 | +0.67 | +0.68 | Strong positive concordance |
| chr2:54560455 | SPTBN1 | +0.54 | +0.79 | Large effect, same direction |

## Next Steps

1. **Wait for Stage 3 completion** (~1-2 days remaining)

2. **Re-run Stage 4/5 on complete data**:
   ```bash
   python scripts/08_alphagenome_stage4_interpret.py
   python scripts/08_alphagenome_stage5_validate.py
   ```

3. **Report direct CIMA-DICE validation** as primary result (83.5% concordance, r=0.685)

4. **Use AlphaGenome for prioritization only**:
   - Select top variants by predicted regulatory impact for functional follow-up
   - Do NOT claim AlphaGenome "validates" eQTLs (validation comes from DICE replication)

5. **Consider additional validation**:
   - eQTLGen (31K blood samples, largest eQTL meta-analysis)
   - BLUEPRINT epigenomics
   - OneK1K single-cell eQTLs

## Conclusions

### Primary Finding: CIMA eQTLs Strongly Replicate

**Direct CIMA vs DICE comparison** (without AlphaGenome filtering):
- 165,064 matched eQTL pairs across 10,394 unique variants
- **83.5% direction concordance**
- **Pearson r = 0.685** (strong correlation)

This is the primary validation result demonstrating that CIMA single-cell eQTLs replicate in independent cell-type-specific data.

### AlphaGenome Role: Prioritization, Not Validation

| AlphaGenome Purpose | Status |
|---------------------|--------|
| Prioritize variants by predicted regulatory effect | ✅ Useful |
| Validate eQTLs | ❌ Not useful (direct comparison is better) |
| Annotate regulatory mechanisms | ❌ Not possible (no chromatin tracks) |

### Summary

1. **CIMA eQTLs are validated** by direct comparison with DICE (83.5% concordance)
2. **AlphaGenome adds prioritization**, not validation - use for selecting variants for follow-up
3. **1,585 AlphaGenome-prioritized variants** available for functional studies
4. **GTEx validation is circular** - AlphaGenome was trained on GTEx data
