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

## Validation Results (Partial Data)

### Multi-Source Validation Summary

| Source | Matches | Concordance | Pearson r | Spearman ρ | Notes |
|--------|---------|-------------|-----------|------------|-------|
| **GTEx Bulk v10** | 1,962 | 64.7% | 0.28 | 0.29 | Whole blood bulk tissue |
| **DICE (hg38)** | 4,495 | **75.1%** | **0.54** | **0.51** | Cell-type-specific eQTLs |
| GTEx ieQTL | 0 | - | - | - | Coordinate mismatch |

### DICE Concordance by Cell Type

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

**Key finding**: DICE cell-type-specific eQTLs show higher concordance (75.1%) than GTEx bulk (64.7%), validating that CIMA single-cell eQTLs capture true cell-type-specific regulatory effects.

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

### Validation Interpretation

1. **GTEx concordance (64.7%)**: Moderate, but potentially circular since AlphaGenome was trained on GTEx
2. **DICE concordance (75.1%)**: **Strong independent validation** - DICE is a separate dataset not used in AlphaGenome training
3. **Cell-type specificity confirmed**: Higher concordance with cell-type-specific DICE than bulk GTEx

### Top Validated Variants

Variants with concordant effects across CIMA, GTEx, and DICE:

| Variant | Gene | CIMA β | GTEx β | DICE β | Notes |
|---------|------|--------|--------|--------|-------|
| chr3:49813942 | UBA7 | -0.39 | -0.28 | -0.63 | Consistent across all sources |
| chr1:192817615 | RGS2 | +0.30 | +0.18 | - | GTEx concordant |
| chr2:54560455 | SPTBN1 | +0.54 | +0.79 | - | Strong GTEx match |

## Next Steps

1. **Wait for Stage 3 completion** (~1-2 days remaining)

2. **Re-run Stage 4/5 on complete data**:
   ```bash
   python scripts/08_alphagenome_stage4_interpret.py
   python scripts/08_alphagenome_stage5_validate.py
   ```

3. **Consider additional validation**:
   - eQTLGen (31K blood samples, largest eQTL meta-analysis)
   - BLUEPRINT epigenomics
   - OneK1K single-cell eQTLs

4. **Functional interpretation**:
   - Focus on variants concordant across all 3 sources
   - Prioritize immune-relevant genes (HLA, cytokines, receptors)
   - Link to disease associations via GWAS catalog

## Conclusions

Despite AlphaGenome API limitations (RNA-seq only, no chromatin), the analysis provides valuable validation:

1. **CIMA eQTLs replicate** in both GTEx (bulk) and DICE (cell-type-specific)
2. **Cell-type specificity matters**: 75% concordance with DICE vs 65% with GTEx bulk
3. **1,585 high-confidence variants** prioritized for follow-up
4. **Independent validation achieved** through DICE (not used in AlphaGenome training)
