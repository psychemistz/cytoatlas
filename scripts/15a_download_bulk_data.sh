#!/usr/bin/env bash
# Download bulk RNA-seq data for TCGA/GTEx external validation.
#
# Downloads TOIL recomputed TCGA+GTEx TPM (fallback), standalone GTEx v8 (fallback),
# standalone TCGA, metadata, and gene ID mapping (GENCODE v23 probemap).
#
# NOTE: The pipeline prefers GTEx v11 data (parquet, 19,788 samples) which must
# be obtained separately from the GTEx Portal (requires registration):
#   - GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet  (4.1 GB)
#   - GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt           (37 MB)
# Place these in the output directory below. The TOIL combined file serves as
# fallback if v11 files are absent.
#
# Usage:
#   bash scripts/15a_download_bulk_data.sh
#   sbatch scripts/15a_download_bulk_data.sh  # via SLURM
#
# Output: /data/parks34/projects/2cytoatlas/data/bulk/

set -uo pipefail

BULK_DIR="/data/parks34/projects/2cytoatlas/data/bulk"
mkdir -p "$BULK_DIR"
cd "$BULK_DIR"

echo "=== Downloading bulk RNA-seq data to $BULK_DIR ==="
echo "Start: $(date)"

# ---- UCSC TOIL combined (TCGA + GTEx, uniformly processed, hg38) ----
if [ ! -f "TcgaTargetGtex_rsem_gene_tpm.gz" ]; then
    echo "[1/6] Downloading TOIL combined TPM (~4 GB)..."
    wget -c https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/TcgaTargetGtex_rsem_gene_tpm.gz
else
    echo "[1/6] TOIL combined TPM already exists, skipping"
fi

# ---- GTEx v8 standalone (optional fallback - TOIL combined is preferred) ----
if [ ! -f "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz" ]; then
    echo "[2/6] Downloading GTEx v8 TPM (~1.5 GB, optional fallback)..."
    wget -c https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz || echo "  WARNING: GTEx v8 standalone not available (not critical if TOIL combined exists)"
else
    echo "[2/6] GTEx v8 TPM already exists, skipping"
fi

# ---- TCGA standalone (optional fallback - TOIL combined is preferred) ----
if [ ! -f "tcga_RSEM_gene_tpm.gz" ]; then
    echo "[3/6] Downloading TCGA TPM (~2 GB, optional fallback)..."
    wget -c https://toil.xenahubs.net/download/tcga_RSEM_gene_tpm.gz || echo "  WARNING: TCGA standalone not available (not critical if TOIL combined exists)"
else
    echo "[3/6] TCGA TPM already exists, skipping"
fi

# ---- Metadata: tissue/cancer type annotation ----
if [ ! -f "TcgaTargetGTEX_phenotype.txt.gz" ]; then
    echo "[4/6] Downloading TOIL phenotype metadata..."
    wget -c https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/TcgaTargetGTEX_phenotype.txt.gz
else
    echo "[4/6] TOIL phenotype metadata already exists, skipping"
fi

if [ ! -f "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt" ]; then
    echo "[5/6] Downloading GTEx v8 sample attributes..."
    wget -c https://storage.googleapis.com/gtex_analysis_v8/annotations/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt
else
    echo "[5/6] GTEx v8 sample attributes already exists, skipping"
fi

# ---- Gene ID mapping (ENSG.version -> HGNC symbol) ----
if [ ! -f "gencode.v23.annotation.gene.probemap" ]; then
    echo "[6/6] Downloading GENCODE v23 gene probemap..."
    wget -c https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/probeMap/gencode.v23.annotation.gene.probemap
else
    echo "[6/6] GENCODE v23 probemap already exists, skipping"
fi

echo ""
echo "=== Download complete ==="
echo "End: $(date)"
echo ""
echo "Files:"
ls -lh "$BULK_DIR"
