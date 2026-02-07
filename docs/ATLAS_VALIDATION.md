# Atlas Validation: Expression vs Activity Concordance

Multi-level validation framework for assessing whether predicted cytokine/secreted protein activity scores correlate with actual gene expression across samples. Both axes are z-scored for consistency across all levels.

---

## Level 0: Bulk RNA-seq (External Validation)

External validation against independent bulk RNA-seq datasets (GTEx ~17K normal tissues, TCGA ~11K tumor samples). Demonstrates that single-cell-derived cytokine signatures generalize beyond the single-cell atlases used to train them.

### Expression

1. Download uniformly-processed TPM from TOIL recompute (hg38, GENCODE v23)
2. Map ENSG.version gene IDs to HGNC symbols via GENCODE v23 probemap
3. Transform: `log2(TPM + 1)` — standard for bulk RNA-seq (NOT log1p(CPM))
4. Gene-wise z-score across all samples

### Activity

1. Same log2(TPM+1) matrix `[genes x samples]`
2. Mean-center each gene across all samples
3. Run SecActPy ridge regression -> output z-scores `[cytokines x samples]`

### Key Differences from Single-Cell Levels

| | Bulk (Level 0) | Single-Cell (Levels 1-4) |
|---|---|---|
| **Data source** | GTEx/TCGA (independent) | CIMA/Inflammation/scAtlas |
| **Normalization** | log2(TPM+1) | log1p(CPM) |
| **Sample definition** | One bulk sample | One donor (pseudobulk) |
| **Cell type** | None (whole-tissue) | Pooled or stratified |
| **N samples** | ~11K-17K | ~20-400 donors |

### Visualization

- Tab: "Bulk RNA-seq" (first tab in Atlas Validation section)
- Scatter: expression z-score vs activity z-score per target
- Comparison: GTEx vs TCGA rho concordance
- Subsampled to 2,000 points for visualization; rho computed on all samples

### Scripts

- `scripts/15a_download_bulk_data.sh` — Download TOIL/GTEx/TCGA data
- `scripts/15_bulk_validation.py` — Activity inference
- `scripts/13_cross_sample_correlation_analysis.py --atlas gtex tcga` — Correlations
- `scripts/14_preprocess_bulk_validation.py` — JSON generation

### Output

- `visualization/data/bulk_rnaseq_validation.json` — Separate JSON for bulk tab
- `results/cross_sample_validation/{gtex,tcga}/` — H5AD files

---

## Level 1: Donor Level

Each sample = one donor (all cell types pooled).

### Expression

1. Sum raw counts across all cells per donor -> matrix `[genes x donors]`
2. CPM normalize per donor -> log1p -> `[genes x donors]` in log1p(CPM) space
3. Gene-wise z-score across donors -> each gene has mean=0, sd=1 across donors

### Activity

1. Same log1p(CPM) matrix `[genes x donors]`
2. Mean-center each gene across donors (subtract row mean)
3. Run SecActPy ridge regression -> output z-scores `[cytokines x donors]`

### Scatter Plot

For cytokine *k*: x-axis = expression z-score, y-axis = activity z-score, each dot = one donor.

---

## Level 2: Donor x Cell Type

Each sample = one (donor, celltype) pair. **Within-celltype normalization is mandatory.**

### Why Within-Celltype?

If you z-score across all donor-celltype samples together, the dominant axis of variation is celltype identity, not donor biology. For example, monocytes constitutively express IL6 far more than T cells. A global z-score would just separate celltypes on the x-axis.

The actual question: **"Within T cells, does the donor with relatively high IFNG expression also show relatively high IFNG activity?"**

### Pipeline

1. **Pseudobulk**: Sum raw counts per (donor, celltype) -> matrix `[genes x (donor x celltype)]`
2. **logCPM**: CPM normalize each sample independently -> log1p
3. **Split by celltype**: Subset into per-celltype matrices
4. **Expression z-score**: Within each celltype, z-score each gene across donors
5. **Activity prediction**: Within each celltype separately:
   - Mean-center each gene across donors (within that celltype)
   - Run SecActPy -> activity z-scores `[cytokines x donors_in_ct]`
6. **Scatter plot**: All celltypes overlaid, colored by celltype
   - Per-celltype correlation (more informative)
   - Global correlation (mixes celltypes)

### Implementation

`run_activity_inference()` in `scripts/12_cross_sample_correlation.py` detects celltype columns in obs and automatically splits by the last grouping column. Ridge regression runs per celltype with within-celltype mean centering.

---

## Level 3: Donor x Cell Type x Resampling

Quantifies estimation uncertainty from unequal cell counts per donor-celltype pair.

### The Problem

A donor-celltype pair with 5,000 cells produces a precise pseudobulk estimate while one with 30 cells is noisy. Both become single dots with equal weight in Level 2.

### Subsampling Strategy

- Per-celltype subsampling size N_ct = 30th percentile of cell counts
- Hard floor: N_ct = max(p30, 50)
- Exclude donor-celltype pairs with fewer than N_ct cells

### Pipeline

For R=100 iterations:

1. Randomly sample N_ct cells (without replacement) per valid donor-celltype pair
2. Sum raw counts -> CPM -> log1p per iteration
3. Within-celltype z-score and SecActPy per iteration
4. Aggregate: mean +/- 95% CI across resamples

### Visualization

- Scatter with 95% CI error bars (horizontal for expression, vertical for activity)
- Color by celltype
- Report distribution of Spearman rho across iterations

---

## Level 4: Single-Cell Level

Tests whether expression-activity coupling holds at single-cell resolution.

### Approach: Filter Expressing Cells

The entire zero-inflation problem comes from non-expressing cells. Once you condition on expression > 0, the remaining values are continuous enough for a standard scatter.

```python
mask = (donor == selected) & (celltype == selected_ct) & (raw_counts[gene_k] > 0)
expr = log_normalized[gene_k, mask]
expr_zscore = (expr - mean(expr)) / sd(expr)
act = activity_zscore[k, mask]
scatter(expr_zscore, act)
```

### Why This Works

- Among expressing cells, log-normalized expression has reasonable dynamic range
- Activity scores are continuous across all cells, so no issue on the y-axis
- Immediately interpretable: "among cells that produce this cytokine, does higher expression correspond to higher predicted downstream activity?"

### Activity Scoring

Per celltype: mean-center each gene across cells within the celltype -> SecActPy -> subset results to expressing cells for display.

### UI Selectors

- **Cytokine**: which target to display
- **Cell type**: which cell population
- **Donor**: specific donor or "All donors"

Per-donor plots immediately reveal whether concordance is universal or donor-specific, without hidden statistical machinery.

### Z-score Scope Follows the Donor Selector

| Selector | Z-score computed across |
|---|---|
| Specific donor | Expressing cells of that donor x celltype |
| All donors | Expressing cells of all donors x celltype |

### Report Alongside Each Scatter

- N expressing / N total cells (% detected)
- Spearman rho
- If only ~30 cells express the cytokine, note the scatter is unreliable

---

## Summary Table

| | Level 0 | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---|---|---|---|---|
| **Unit** | Bulk sample | Donor | Donor x CT | Donor x CT x resample | Single cell |
| **Aggregation** | None (bulk) | Sum all cells per donor | Sum cells per donor per CT | Sum subsampled N_ct cells | No aggregation |
| **Normalization** | log2(TPM+1) | logCPM | logCPM | logCPM per iteration | log-normalized (standard scRNA-seq) |
| **Expression measure** | Gene-wise z-score across samples | Gene-wise z-score across donors | Gene-wise z-score within CT | Gene-wise z-score within CT per iter | Filter > 0 -> z-score within displayed subset |
| **Activity method** | SecActPy (mean-centered across samples) | SecActPy (mean-centered across donors) | SecActPy per CT (mean-centered within CT) | SecActPy per CT per iteration | SecActPy per CT (mean-centered within CT) |
| **Sparsity handling** | N/A (bulk) | N/A (pseudobulk) | N/A (pseudobulk) | N/A (pseudobulk) | Exclude non-expressing cells |
| **Both axes** | Z-scores | Z-scores | Z-scores | Z-scores | Z-scores |
| **N points** | ~11K-17K samples | ~20-400 donors | ~160 (donor x CT) | ~160 with 95% CI | Hundreds-thousands (expressing cells) |
| **Visualization** | Scatter + GTEx/TCGA comparison | Scatter | Scatter (color=CT) | Scatter + error bars | Scatter with donor/CT/cytokine selectors |
| **Correlation** | Spearman rho | Spearman rho | Spearman rho (global + per CT) | Spearman rho distribution | Spearman rho |
| **UI selectors** | Dataset + Cytokine | Cytokine | Cytokine | Cytokine | Cytokine + Celltype + Donor |
| **Report alongside** | N samples, dataset | N donors | N donors per CT | N valid donors, N_ct | N expressing / N total (% detected) |
| **Biological question** | External generalization | Donor concordance | CT-specific concordance | Robustness to sampling | Single-cell coupling |
| **Narrative role** | Independent external validation | Establishes phenomenon | Localizes to celltypes | Confirms robustness | Tests single-cell resolution |

---

## Implementation Status

| Level | Script | Status |
|-------|--------|--------|
| Level 0 (Bulk) | `scripts/15_bulk_validation.py` | Done: GTEx + TCGA, log2(TPM+1), ridge regression |
| Level 1 (Donor) | `scripts/12_cross_sample_correlation.py` | Done: sum counts -> CPM -> log1p, mean-center, zscore output |
| Level 2 (Donor x CT) | `scripts/12_cross_sample_correlation.py` | Done: within-celltype ridge via `_get_celltype_col()` auto-detection |
| Level 3 (Resampling) | TBD | Not implemented |
| Level 4 (Single-cell) | TBD | Not implemented |

### Key Code Changes (2026-02-06)

1. **Pseudobulk**: Accumulate raw count sums (not per-cell CPM) -> CPM normalize summed counts -> log1p
2. **Activity output**: Store `result['zscore']` not `result['beta']`
3. **Mean-centering**: Added `Y -= Y.mean(axis=1, keepdims=True)` before ridge regression
4. **Within-celltype**: For celltype-stratified levels, ridge runs per celltype separately with within-group mean centering
