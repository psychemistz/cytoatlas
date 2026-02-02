# AlphaGenome eQTL Analysis Implementation Plan

## Overview

Use AlphaGenome API to prioritize regulatory variants from CIMA immune cell cis-eQTLs affecting cytokine/secreted protein expression.

## Status Summary (Updated 2026-02-02 15:00)

| Stage | Status | Actual Output |
|-------|--------|---------------|
| Stage 1 | ✅ COMPLETE | 48,627 cytokine eQTLs → 29,816 unique variants |
| Stage 2 | ✅ COMPLETE | 29,816 variants formatted (SNV/indel) |
| Stage 3 | 🔄 RUNNING | Job 10659620: 6,500/29,816 (~22%), ETA ~95h |
| Stage 4 | ⚠️ MOCK ONLY | Existing output from simulated data - needs re-run |
| Stage 5 | ✅ GTEx READY | GTEx V10 downloaded (2,985,690 Whole Blood eQTLs) |

### ⚠️ Critical: Mock vs Real Data

**Existing Stage 4/5 outputs are based on MOCK predictions, not real AlphaGenome API results.**

| Aspect | Mock Run (existing files) | Real API (running job) |
|--------|---------------------------|------------------------|
| Source | `--mock` flag (simulated) | AlphaGenome API |
| Variants | 29,816 | 2,100 in checkpoint |
| Tracks | 19 (chromatin + TF) | **3 (RNA-seq only)** |
| Track types | ATAC, H3K27ac, H3K4me3, DNase, ChIP | GTEx lymphocytes, spleen, whole blood |
| Mechanism classification | Full (enhancer/promoter/TF) | Limited (RNA-only) |
| File | Overwritten by real run | `stage3_predictions.h5ad` (3 variants) |

**Action required**: Stage 4/5 must be re-run after Stage 3 completes with real API data.

## Data Sources

| Resource | Location | Details |
|----------|----------|---------|
| eQTL data | `/data/Jiang_Lab/Data/Seongyong/CIMA/xQTL/CIMA_Lead_cis-xQTL.csv` | 223,405 records, 71,530 at FDR<0.05 |
| CytoSig genes | `secactpy.load_cytosig()` | 4,881 genes x 43 cytokines |
| SecAct genes | `secactpy.load_secact()` | 7,919 genes x 1,169 proteins |

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

**Current job status** (Job 10659620):
- Progress: 2,188 / 29,816 variants (~7.3%)
- Checkpoint: 2,100 variants saved
- Successful API calls: ~1,055
- Rate limiting: ~50% RESOURCE_EXHAUSTED errors
- Runtime: ~9 hours elapsed
- ETA: ~110-120 hours remaining (4-5 days)

**Key findings from actual execution**:
- API returns **3 immune-relevant RNA-seq tracks** (not 7000+):
  - `RNA_SEQ_EFO:0000572 gtex Cells_EBV-transformed_lymphocytes polyA plus RNA-seq`
  - `RNA_SEQ_UBERON:0002106 gtex Spleen polyA plus RNA-seq`
  - `RNA_SEQ_UBERON:0013756 gtex Whole_Blood polyA plus RNA-seq`
- **No chromatin tracks** (ATAC, H3K27ac, H3K4me3) returned for immune filtering
- Rate limiting causes ~50% RESOURCE_EXHAUSTED errors
- ~15-20 sec per variant (with retries)

**Features implemented**:
- Checkpoint every 100 variants for resume
- `--resume` flag to continue from last checkpoint
- `--mock` flag for testing without API
- Immune track filtering (EBV-lymphocytes, Spleen, Whole Blood, PBMC)

**Output**:
- `results/alphagenome/stage3_predictions.h5ad` - Currently 3 variants (will grow as job runs)
- `results/alphagenome/stage3_checkpoint.json` - 2,100 variants processed
- `results/alphagenome/stage3_test_predictions.h5ad` - 12 variants for testing

**Runtime**: ~125 hours for 29,816 variants (168h allocated)

### Stage 4: Interpret and Score Predictions ⚠️
**Script**: `scripts/08_alphagenome_stage4_interpret.py`

**Current status**: Existing outputs are from **MOCK data** (simulated chromatin tracks). Must re-run after Stage 3 completes.

**Mock run results** (not valid for final analysis):
- 29,816 variants scored with simulated data
- 1,231 prioritized variants (based on fake chromatin signals)
- Mechanism classification used non-existent tracks

**Updated approach** (RNA-seq only, no chromatin):
1. Compute regulatory impact scores from RNA-seq track differences
2. ~~Classify mechanism~~ → Limited without chromatin marks
3. Check direction concordance: AlphaGenome RNA prediction vs eQTL beta
4. Prioritize variants with highest track differences

**CLI options**:
- `--test` - Use test predictions file
- `--input FILE` - Custom input h5ad
- `--output-suffix SUFFIX` - Custom output suffix

**Output** (to be regenerated):
- `results/alphagenome/stage4_scored_variants.csv`
- `results/alphagenome/stage4_prioritized.csv`
- `results/alphagenome/stage4_summary.json`

### Stage 5: Validate Against GTEx ⚠️
**Script**: `scripts/08_alphagenome_stage5_validate.py`

**Current status**:
- Existing outputs are from **MOCK data** - not valid
- GTEx data directory is **EMPTY** - manual download still required

**Mock run results** (not valid):
- 369 variants matched to GTEx
- 48.8% direction concordance (meaningless with fake predictions)

**Updated approach**:
1. Load GTEx v8 whole blood eQTLs (manual download required)
2. Match variants by position
3. Compute concordance: CIMA beta vs GTEx beta vs AlphaGenome prediction
4. Generate validation report

**Supported GTEx files** (auto-detected):
- `Whole_Blood.v8.signif_variant_gene_pairs.txt.gz` (preferred, ~50 MB)
- `Whole_Blood.nominal.allpairs.txt.gz` (filtered by p < 1e-5)

**CLI options**:
- `--test` - Use test prioritized file
- `--input FILE` - Custom input CSV

**Output** (to be regenerated):
- `results/alphagenome/stage5_gtex_matched.csv`
- `results/alphagenome/stage5_validation_metrics.json`
- `results/alphagenome/stage5_report.md`

## File Structure

```
scripts/
  08_alphagenome_stage1_filter.py      ✅
  08_alphagenome_stage2_format.py      ✅
  08_alphagenome_stage3_predict.py     ✅
  08_alphagenome_stage4_interpret.py   ✅
  08_alphagenome_stage5_validate.py    ✅
  slurm/run_alphagenome.sh             ✅

results/alphagenome/
  stage1_cytokine_eqtls.csv            ✅ 48,627 eQTLs
  stage1_summary.json                  ✅
  stage2_alphagenome_input.csv         ✅ 29,816 variants
  stage2_summary.json                  ✅
  stage3_predictions.h5ad              🔄 3 variants (growing)
  stage3_checkpoint.json               🔄 2,100 variants processed
  stage3_test_predictions.h5ad         ✅ 12 variants for testing
  stage4_scored_variants.csv           ⚠️ MOCK DATA - needs re-run
  stage4_prioritized.csv               ⚠️ MOCK DATA - needs re-run
  stage4_summary.json                  ⚠️ MOCK DATA - needs re-run
  stage5_gtex_matched.csv              ⚠️ MOCK DATA - needs re-run
  stage5_validation_metrics.json       ⚠️ MOCK DATA - needs re-run
  stage5_report.md                     ⚠️ MOCK DATA - needs re-run
  gtex_data/                           ✅ V10 parquet (2,985,690 eQTLs)
```

## SLURM Job

```bash
#SBATCH --job-name=alphagenome
#SBATCH --time=168:00:00   # 7 days (was 48h in original plan)
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=norm
```

**Current job**: 10659620, running on cn0008 (~9h elapsed)

## Key Deviations from Original Plan

| Aspect | Planned | Actual |
|--------|---------|--------|
| Variant count | 500-2,000 | 29,816 |
| Track count | 7,000+ | 3 (RNA-seq only) |
| Runtime | 2-5 hours | ~125 hours |
| Chromatin data | ATAC, H3K27ac, etc. | **Not available** |
| Mechanism classification | Enhancer/promoter/TF | Limited (RNA-only) |

## GTEx Data Download ✅ COMPLETE

GTEx V10 eQTL data downloaded and extracted:
```
/data/parks34/projects/2secactpy/results/alphagenome/gtex_data/
  GTEx_Analysis_v10_eQTL_updated/
    Whole_Blood.v10.eQTLs.signif_pairs.parquet  # 2,985,690 eQTLs
```

Stage 5 script updated to support V10 parquet format.

## Next Steps

1. **Monitor Stage 3 job** - Check progress with:
   ```bash
   tail -20 logs/alphagenome_10659620.out
   python3 -c "import json; d=json.load(open('results/alphagenome/stage3_checkpoint.json')); print(f'Processed: {len(d[\"processed_variants\"])}/29816')"
   ```

2. **Wait for Stage 3 completion** (~4 days remaining, ETA Feb 6)

3. ~~**Download GTEx data**~~ ✅ COMPLETE - V10 parquet downloaded

4. **Re-run Stage 4** on real predictions (after Stage 3 completes):
   ```bash
   python scripts/08_alphagenome_stage4_interpret.py
   ```

5. **Re-run Stage 5** validation:
   ```bash
   python scripts/08_alphagenome_stage5_validate.py
   ```

6. **Adjust analysis approach** - Focus on RNA-seq expression concordance rather than chromatin mechanism classification (limited by API track availability)

## Implications for Final Analysis

Given that only RNA-seq tracks are available (no chromatin data):

1. **Cannot classify mechanisms** as enhancer/promoter/TF binding
2. **Focus on expression concordance**: Do AlphaGenome RNA predictions match eQTL direction?
3. **Validation strategy**: Compare CIMA eQTL beta vs GTEx beta vs AlphaGenome prediction
4. **Prioritization**: Rank by magnitude of predicted RNA expression change

The analysis will be more limited than originally planned but still valuable for identifying variants with consistent expression effects across prediction methods.
