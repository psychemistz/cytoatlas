# Feasibility of using AlphaGenome for cytokine eQTL variant prioritization

Using AlphaGenome to analyze 71,530 cis-eQTLs across 69 immune cell types is **technically feasible and scientifically promising**, though certain practical constraints warrant careful planning. The free API can process this dataset in approximately **20-30 hours** with no computational cost, while AlphaGenome's state-of-the-art performance on eQTL prediction—recovering **41% of GTEx eQTLs at 90% accuracy** versus 19% for the next-best model—makes it the optimal choice for identifying regulatory variants affecting cytokine and secreted protein expression.

## AlphaGenome delivers unprecedented resolution for regulatory variant analysis

AlphaGenome, published by Google DeepMind in Nature (January 2026), represents a significant advance over previous models like Enformer and Borzoi. Its architecture combines a **U-Net-inspired backbone with transformer blocks** to process 1 million base pairs of genomic context—5× longer than Enformer—while achieving **single base-pair resolution** for most outputs, a 128-fold improvement over Enformer's 128-bp bins.

The model simultaneously predicts **7,000+ functional genomic tracks across 11 modalities**: RNA-seq expression, chromatin accessibility (ATAC-seq, DNase-seq), histone modifications (H3K27ac, H3K4me1, H3K36me3), transcription factor binding, transcription initiation (CAGE, PRO-cap), splice site usage, and chromatin contact maps. This multimodal approach enables mechanistic interpretation—when a variant affects both local chromatin accessibility and downstream gene expression, you can infer a causal regulatory pathway rather than mere correlation.

For eQTL analysis specifically, AlphaGenome achieves **Spearman ρ = 0.49** correlation between predicted and observed effect sizes (versus 0.39 for Borzoi) and **0.80 auROC** for predicting the direction of expression change. At high-confidence thresholds (>99th percentile quantile scores), this correlation rises to **ρ = 0.73**—strong enough to prioritize variants with confidence.

## API access is immediate and free for academic research

Google DeepMind provides **immediate, free API access** for non-commercial research through their portal at deepmind.google.com/science/alphagenome. No waitlist exists—access is granted upon accepting the terms of service. The API accepts variants in a simple format requiring only chromosome, position, reference allele, and alternate allele (hg38 coordinates).

For local deployment, model weights are available on **Kaggle** (kaggle.com/models/google/alphagenome) and **Hugging Face** (huggingface.co/google/alphagenome-all-folds) under non-commercial terms. The research codebase at github.com/google-deepmind/alphagenome_research includes full JAX implementation, variant scorers, and evaluation notebooks.

The input/output specifications work directly with eQTL data:

| Parameter | Specification |
|-----------|--------------|
| Input format | VCF-like: CHROM (chr1-22, chrX), POS (1-based), REF, ALT |
| Coordinate system | hg38 (GRCh38) for human |
| Sequence context | 16KB, 100KB, 500KB, or **1MB (recommended)** |
| Variant types | SNVs, insertions, deletions all supported |
| Output format | AnnData objects with raw and quantile-normalized scores |

## Analyzing 71,530 variants is computationally tractable

The API processes approximately **3,600 variants per hour** (~1 second per variant), meaning the full dataset of 71,530 cis-eQTLs requires roughly **20 hours of continuous processing**. This falls within the API's acceptable range for "medium-scale analyses" (the documentation notes it's "not suitable" only for analyses exceeding 1 million predictions). Dynamic rate limiting may extend this timeline during high-demand periods, so building in retry logic with exponential backoff is advisable.

For local deployment, an NVIDIA H100 GPU (80GB HBM3) is the minimum recommended hardware. Cloud costs are modest:

| Provider | Cost/Hour | 20-Hour Total |
|----------|-----------|---------------|
| AWS P5 (H100) | $3.93 | ~$79 |
| Google Cloud A3 | $3.00 | ~$60 |
| Lambda Labs | $2.99 | ~$60 |
| Vast.ai | $1.49 | ~$30 |

The free API makes the most sense for a one-time analysis of this scale.

## Practical workflow for cytokine-focused analysis

A realistic workflow integrates gene annotation, data formatting, prediction, and interpretation in five stages.

**Stage 1: Define cytokine and secreted protein gene sets.** Query Gene Ontology term **GO:0005125** (cytokine activity) to capture ~150 classical cytokines including interleukins, interferons, and chemokines. For broader secreted proteins, use UniProt keyword KW-0964 (Secreted) or the Human Protein Atlas secretome, which defines **2,623 proteins** classified into blood proteins (~625), extracellular matrix components, and tissue-specific secreted factors. Cross-reference with your eQTL target genes to identify the subset requiring analysis—likely **500-2,000 variants** affecting cytokine/secreted protein expression from your 71,530 total eQTLs.

**Stage 2: Format and validate eQTL coordinates.** Convert your cis-eQTL coordinates to hg38 if currently in GRCh37 (DICE database uses GRCh37). Extract the essential columns: variant ID, chromosome, position, reference allele, alternate allele. Verify that positions are 1-based and chromosomes use "chr" prefix (chr1 not 1). Example input format:

```
variant_id,CHROM,POS,REF,ALT,target_gene
rs12345_IL6,chr7,22766645,A,G,IL6
rs67890_CXCL8,chr4,74606209,C,T,CXCL8
```

**Stage 3: Execute AlphaGenome predictions with relevant track filtering.** For immune cell eQTLs, filter output tracks to relevant cell types using ontology terms. The model includes tracks for GM12878 (B lymphoblastoid), CD34+ myeloid progenitors, PBMCs, and various hematopoietic lineages from ENCODE and GTEx. While exact matches to DICE's 15 immune cell types (naive B cells, classical/non-classical monocytes, CD4/CD8 T cell subsets, NK cells) may not exist, map to the closest available lineage:

```python
from alphagenome import genome, dna_client

# Score variant for expression and accessibility
variant = genome.Variant(
    chromosome='chr7',
    position=22766645,
    reference_bases='A',
    alternate_bases='G'
)

# Use 1MB context for optimal enhancer detection
sequence_length = dna_client.SUPPORTED_SEQUENCE_LENGTHS['SEQUENCE_LENGTH_1MB']
scores = await client.score_variant(variant, sequence_length=sequence_length)
```

**Stage 4: Interpret multimodal predictions for mechanism identification.** AlphaGenome returns scores across all modalities, enabling mechanistic inference. A variant affecting cytokine expression might show: (1) disrupted TF binding at the variant site, (2) reduced H3K27ac signal indicating enhancer deactivation, (3) decreased chromatin accessibility, and (4) reduced RNA-seq signal at the gene body. This multimodal pattern indicates an enhancer variant, whereas a variant affecting only splicing scores suggests a splicing QTL. Filter for variants exceeding the **99th percentile quantile score**—these represent predicted effects larger than 99% of common variants and correlate most strongly with observed eQTL effects.

**Stage 5: Prioritize and validate candidates.** Rank variants by predicted effect magnitude and concordance with observed eQTL direction. Variants where AlphaGenome predicts the same direction as the measured eQTL effect, with high confidence scores, represent top candidates. For experimental validation, MPRA in primary CD4+ T cells has successfully tested >18,000 autoimmune variants with 9-24% sensitivity for detecting causal eQTL alleles.

## Key limitations require careful interpretation

**Cell-type specificity remains an ongoing challenge.** The Nature paper explicitly acknowledges that "predictions for underrepresented tissues or rare cell types remain limited." AlphaGenome's training data from ENCODE and GTEx includes general immune populations (whole blood, PBMCs) but may lack coverage for specialized subtypes like TH17 cells or tissue-resident memory T cells. Cross-reference predicted effects with cell-type-specific chromatin data from DICE or ImmGen when available.

**Distal regulatory elements show reduced accuracy.** While AlphaGenome's 1MB context window captures most cis-regulatory elements, prediction accuracy decreases for variants >100kb from their target gene TSS. Since **92% of lead cis-eQTL SNPs** fall within 100kb of the TSS, this affects a minority of variants, but cytokine genes with complex super-enhancer regulation may have important distal variants that are harder to interpret.

**Coordinate system compatibility requires attention.** The DICE eQTL browser uses GRCh37 coordinates while AlphaGenome requires hg38. LiftOver conversion is necessary but may fail for ~0.5-1% of variants in complex genomic regions.

**The model is not diploid-aware.** AlphaGenome processes single sequences, so compound heterozygous effects or haplotype-specific regulation cannot be directly modeled.

**Commercial use requires separate licensing.** The free API and model weights are restricted to non-commercial research. Commercial applications require contacting DeepMind directly.

## Validation strategy should leverage existing immune datasets

Given the scale of analysis, a tiered validation approach maximizes efficiency. First, check predicted effects against independent eQTL datasets—GTEx whole blood eQTLs provide an orthogonal validation set. Variants where AlphaGenome predictions match both DICE and GTEx effect directions represent highest-confidence candidates.

Second, leverage existing MPRA datasets for immune variants. Recent studies have tested thousands of autoimmune disease variants in primary T cells, with results available for comparison. Third, for top candidates affecting key cytokines like IL-6, TNF, or interferon pathway genes, consider CRISPR allelic replacement in relevant cell lines (Jurkat for T cells, THP-1 for monocytes) to demonstrate endogenous expression effects.

## Conclusion

AlphaGenome is well-suited for prioritizing regulatory variants from your 71,530 immune cell eQTLs, with particular advantages for cytokine biology: its multimodal predictions can distinguish enhancer variants from promoter variants from splicing variants, its state-of-the-art eQTL prediction accuracy minimizes false positives, and its free API makes the analysis immediately accessible. The realistic timeline is **1-2 days for full analysis** at zero cost using the API, with results that include predicted effect sizes, directions, and mechanistic annotations across chromatin accessibility, TF binding, histone modifications, and gene expression. Focus initial analysis on the ~500-2,000 variants affecting annotated cytokines and secreted proteins, validate predictions against GTEx whole blood eQTLs, and prioritize high-confidence variants (>99th percentile scores with concordant effect directions) for experimental follow-up.

## Execution Plan
stage 1: we have cytosig and secact gene sets which defines genes related to cytokine/secreted proteins

stage2. review eqtl data and extract valid input data (if anything not clear, let me know)

stage3. generate alphagenome prediction script and execute it.

stage4. generate outcome interpretation (or scoring) logic & script and execute it.

stage5. validate alphagenome prediction using proposed validation strategy.
