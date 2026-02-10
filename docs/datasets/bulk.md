# Bulk RNA-seq Datasets (GTEx + TCGA)

External validation datasets for demonstrating that single-cell-derived cytokine signatures generalize to independent bulk RNA-seq data.

## GTEx v8

| Property | Value |
|----------|-------|
| **Samples** | ~17,382 |
| **Tissues** | 54 tissue types |
| **Donors** | 948 |
| **Quantification** | RNA-SeQC v1.1.9 TPM |
| **Genome** | hg38 (GRCh38) |
| **Gene annotation** | GENCODE v26 |
| **Source** | GTEx Portal / TOIL recompute |

### Key Metadata Columns

| Column | Description | Example |
|--------|-------------|---------|
| `tissue_type` | Broad tissue category | Brain, Heart, Lung |
| `tissue_detail` | Detailed tissue subtype | Brain - Cortex |
| `dataset` | Source identifier | GTEX |

## TCGA

| Property | Value |
|----------|-------|
| **Samples** | ~11,000 |
| **Cancer types** | 33 |
| **Quantification** | RSEM TPM via TOIL recompute |
| **Genome** | hg38 (GRCh38) |
| **Gene annotation** | GENCODE v23 |
| **Source** | UCSC Xena TOIL hub |

### Key Metadata Columns

| Column | Description | Example |
|--------|-------------|---------|
| `cancer_type` | Detailed cancer type | Lung Adenocarcinoma |
| `cancer_site` | Primary site | Lung |
| `dataset` | Source identifier | TCGA |

## Gene ID Mapping

Both datasets use Ensembl gene IDs (ENSG.version format). We use the GENCODE v23 probemap to map to HGNC symbols.

| File | Description |
|------|-------------|
| `gencode.v23.annotation.gene.probemap` | ENSG.version to HGNC symbol mapping |

Mapping strategy:
1. Match ENSG.version directly
2. Fall back to ENSG (without version) for cross-version compatibility
3. Keep first occurrence to avoid duplicate symbols

## Expression Transform

**log2(TPM + 1)** — standard for bulk RNA-seq. This differs from single-cell which uses log1p(CPM).

## Data Files

All stored in `/data/parks34/projects/2cytoatlas/data/bulk/`:

| File | Size | Description |
|------|------|-------------|
| `TcgaTargetGtex_rsem_gene_tpm.gz` | ~4 GB | TOIL combined TPM (TCGA + GTEx) |
| `GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz` | ~1.5 GB | GTEx v8 standalone |
| `tcga_RSEM_gene_tpm.gz` | ~2 GB | TCGA standalone |
| `TcgaTargetGTEX_phenotype.txt.gz` | Small | TOIL phenotype metadata |
| `GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt` | Small | GTEx sample attributes |
| `gencode.v23.annotation.gene.probemap` | Small | Gene ID mapping |

## Output Files

```
results/cross_sample_validation/gtex/
  gtex_donor_only_pseudobulk.h5ad     # obs: sample_id, tissue_type, dataset
  gtex_donor_only_cytosig.h5ad        # Activity z-scores (CytoSig)
  gtex_donor_only_lincytosig.h5ad     # Activity z-scores (LinCytoSig)
  gtex_donor_only_secact.h5ad         # Activity z-scores (SecAct)

results/cross_sample_validation/tcga/
  tcga_donor_only_pseudobulk.h5ad     # obs: sample_id, cancer_type, dataset
  tcga_donor_only_cytosig.h5ad
  tcga_donor_only_lincytosig.h5ad
  tcga_donor_only_secact.h5ad
```

## Pipeline

1. **Download**: `scripts/15a_download_bulk_data.sh`
2. **Activity inference**: `scripts/15_bulk_validation.py --dataset all --backend auto`
3. **Correlations**: `scripts/13_cross_sample_correlation_analysis.py --atlas gtex tcga`
4. **JSON for viz**: `scripts/14_preprocess_bulk_validation.py`

## References

- GTEx Consortium. "The GTEx Consortium atlas of genetic regulatory effects across human tissues." Science 369.6509 (2020).
- Vivian et al. "Toil enables reproducible, open source, big biomedical data analyses." Nature Biotechnology 35.4 (2017).
