# AlphaGenome eQTL Analysis Implementation Plan

## Overview

Use AlphaGenome API to prioritize regulatory variants from CIMA immune cell cis-eQTLs affecting cytokine/secreted protein expression.

## Status Summary (Updated 2026-02-01)

| Stage | Status | Actual Output |
|-------|--------|---------------|
| Stage 1 | ✅ COMPLETE | 48,627 cytokine eQTLs → 29,816 unique variants |
| Stage 2 | ✅ COMPLETE | 29,816 variants formatted (SNV/indel) |
| Stage 3 | 🔄 RUNNING | Job 10659620: ~900/29,816 (~3%), ETA ~115h |
| Stage 4 | ⏳ READY | Can run on sample data |
| Stage 5 | ⏳ READY | Needs GTEx data (manual download) |

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
- `results/alphagenome/stage3_predictions.h5ad` (written at job completion)
- `results/alphagenome/stage3_checkpoint.json` (updated every 100 variants)
- `results/alphagenome/stage3_test_predictions.h5ad` (12 variants for testing)

**Runtime**: ~125 hours for 29,816 variants (168h allocated)

### Stage 4: Interpret and Score Predictions
**Script**: `scripts/08_alphagenome_stage4_interpret.py`

**Updated approach** (RNA-seq only, no chromatin):
1. Compute regulatory impact scores from RNA-seq track differences
2. ~~Classify mechanism~~ → Limited without chromatin marks
3. Check direction concordance: AlphaGenome RNA prediction vs eQTL beta
4. Prioritize variants with highest track differences

**CLI options**:
- `--test` - Use test predictions file
- `--input FILE` - Custom input h5ad
- `--output-suffix SUFFIX` - Custom output suffix

**Output**:
- `results/alphagenome/stage4_scored_variants.csv`
- `results/alphagenome/stage4_prioritized.csv`
- `results/alphagenome/stage4_summary.json`

### Stage 5: Validate Against GTEx
**Script**: `scripts/08_alphagenome_stage5_validate.py`

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

**Output**:
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
  stage3_predictions.h5ad              🔄 (written at job completion)
  stage3_checkpoint.json               ✅ ~900 variants processed
  stage3_test_predictions.h5ad         ✅ 12 variants for testing
  stage4_scored_variants.csv           ⏳
  stage4_prioritized.csv               ⏳
  stage5_gtex_matched.csv              ⏳
  stage5_report.md                     ⏳
  gtex_data/                           ⏳ (manual download needed)
```

## SLURM Job

```bash
#SBATCH --job-name=alphagenome
#SBATCH --time=168:00:00   # 7 days (was 48h in original plan)
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=norm
```

**Current job**: 10659620, running on cn0008

## Key Deviations from Original Plan

| Aspect | Planned | Actual |
|--------|---------|--------|
| Variant count | 500-2,000 | 29,816 |
| Track count | 7,000+ | 3 (RNA-seq only) |
| Runtime | 2-5 hours | ~125 hours |
| Chromatin data | ATAC, H3K27ac, etc. | Not available |
| Mechanism classification | Enhancer/promoter/TF | Limited (RNA-only) |

## GTEx Data Download (Manual)

HPC proxy blocks external downloads. On local machine:
```bash
wget "https://storage.gtexportal.org/public/GTEx_Analysis_v8/single_tissue_qtl_data/GTEx_Analysis_v8_eQTL.tar"
tar -xvf GTEx_Analysis_v8_eQTL.tar --wildcards '*/Whole_Blood*signif*'
scp GTEx_Analysis_v8_eQTL/Whole_Blood.v8.signif_variant_gene_pairs.txt.gz \
  parks34@biowulf.nih.gov:/vf/users/parks34/projects/2secactpy/results/alphagenome/gtex_data/
```

## Next Steps

1. **Wait for Stage 3 job completion** (~4-5 days)
2. **Run Stage 4** on full predictions: `python scripts/08_alphagenome_stage4_interpret.py`
3. **Download GTEx data** manually and transfer to HPC
4. **Run Stage 5** validation: `python scripts/08_alphagenome_stage5_validate.py`
5. **Analyze results** - focus on RNA-seq concordance rather than chromatin mechanisms
