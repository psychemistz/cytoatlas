# GPU-Accelerated Data Processing Pipeline for CytoAtlas

## Overview

Create a comprehensive **cytoatlas-pipeline** package for GPU-accelerated data processing covering ALL CytoAtlas services: activity inference, correlation analysis, differential analysis, validation, cross-atlas integration, disease analysis, cancer analysis, and search indexing.

## Problem Statement

Current limitations:
- Celery hard time limits (1-2 hours) crash on 19M+ cell datasets
- Single-worker processing with no parallel computation
- Naive correlation scoring instead of ridge regression in submission processing
- No checkpointing for crash recovery
- No external data source integration (cellxgene)
- Processing pipelines scattered across scripts/ and services/

## Architecture Decision

**Separate package**: `cytoatlas-pipeline/` - consolidates all data processing into a modular, GPU-accelerated package usable by API, scripts, and CLI.

---

## Complete Package Structure

```
cytoatlas-pipeline/
├── pyproject.toml
├── src/cytoatlas_pipeline/
│   ├── __init__.py
│   │
│   ├── core/                          # Infrastructure
│   │   ├── __init__.py
│   │   ├── config.py                  # Pipeline configuration
│   │   ├── gpu_manager.py             # GPU resource management
│   │   ├── checkpoint.py              # Checkpoint/recovery system
│   │   ├── memory.py                  # Memory estimation utilities
│   │   └── cache.py                   # Result caching layer
│   │
│   ├── ingest/                        # Data Sources
│   │   ├── __init__.py
│   │   ├── base.py                    # DataSource interface
│   │   ├── local_h5ad.py              # Chunked H5AD loader
│   │   ├── cellxgene.py               # cellxgene Census connector
│   │   ├── remote_h5ad.py             # Remote/streaming H5AD
│   │   └── formats.py                 # Loom, 10X format support
│   │
│   ├── aggregation/                   # Aggregation Strategies
│   │   ├── __init__.py
│   │   ├── base.py                    # AggregationStrategy interface
│   │   ├── celltype.py                # Cell type hierarchy (L1/L2/L3)
│   │   ├── pseudobulk.py              # Standard cell_type × sample
│   │   ├── resampling.py              # Bootstrap-based pseudobulk
│   │   └── singlecell.py              # Per-cell streaming
│   │
│   ├── activity/                      # Activity Inference (GPU)
│   │   ├── __init__.py
│   │   ├── ridge.py                   # Ridge regression wrapper
│   │   ├── parallel.py                # Multi-GPU signature processing
│   │   ├── streaming.py               # Streaming result writer
│   │   └── signatures.py              # CytoSig, SecAct, custom loaders
│   │
│   ├── correlation/                   # Correlation Analysis (GPU)
│   │   ├── __init__.py
│   │   ├── pearson.py                 # Pearson correlation (GPU)
│   │   ├── spearman.py                # Spearman correlation (GPU)
│   │   ├── partial.py                 # Partial correlation
│   │   ├── continuous.py              # Age, BMI, metabolites
│   │   └── biochemistry.py            # Blood marker correlations
│   │
│   ├── differential/                  # Differential Analysis (GPU)
│   │   ├── __init__.py
│   │   ├── wilcoxon.py                # Wilcoxon rank-sum (GPU)
│   │   ├── ttest.py                   # T-test variants
│   │   ├── effect_size.py             # Activity difference computation
│   │   ├── fdr.py                     # Benjamini-Hochberg FDR
│   │   └── stratified.py              # Multi-level stratification
│   │
│   ├── validation/                    # Validation Pipeline (GPU)
│   │   ├── __init__.py
│   │   ├── sample_level.py            # Expression vs activity regression
│   │   ├── celltype_level.py          # Cell type validation
│   │   ├── pseudobulk_vs_sc.py        # Aggregation comparison
│   │   ├── singlecell.py              # Expressing vs non-expressing
│   │   ├── biological.py              # Known marker validation
│   │   ├── gene_coverage.py           # Signature gene detection
│   │   ├── cv_stability.py            # Cross-validation robustness
│   │   └── quality_score.py           # Composite quality scoring
│   │
│   ├── cross_atlas/                   # Cross-Atlas Integration (GPU)
│   │   ├── __init__.py
│   │   ├── harmonization.py           # Activity harmonization
│   │   ├── celltype_mapping.py        # Cell type alignment
│   │   ├── conserved.py               # Conserved signature detection
│   │   ├── meta_analysis.py           # Multi-atlas aggregation
│   │   └── consistency.py             # CV, std, correlation scoring
│   │
│   ├── disease/                       # Disease Analysis
│   │   ├── __init__.py
│   │   ├── activity.py                # Disease-specific activity
│   │   ├── differential.py            # Disease vs healthy
│   │   ├── treatment_response.py      # Response prediction (ML)
│   │   ├── temporal.py                # Longitudinal analysis
│   │   ├── conserved_programs.py      # Multi-disease signatures
│   │   └── driving_populations.py     # Cell type contributions
│   │
│   ├── cancer/                        # Cancer Analysis
│   │   ├── __init__.py
│   │   ├── tumor_adjacent.py          # Tumor vs adjacent comparison
│   │   ├── infiltration.py            # Immune infiltration (TME)
│   │   ├── exhaustion.py              # T cell exhaustion states
│   │   ├── caf_subtypes.py            # CAF classification
│   │   └── cancer_type.py             # Cancer type stratification
│   │
│   ├── organ/                         # Organ/Tissue Analysis
│   │   ├── __init__.py
│   │   ├── signatures.py              # Organ-specific signatures
│   │   ├── specificity.py             # Tissue specificity scoring
│   │   └── cell_composition.py        # Cell type proportions
│   │
│   ├── search/                        # Search Index Pipeline
│   │   ├── __init__.py
│   │   ├── indexer.py                 # Entity extraction & indexing
│   │   ├── gene_mapping.py            # HGNC ↔ CytoSig mapping
│   │   ├── fuzzy.py                   # Fuzzy matching (Levenshtein)
│   │   └── ranking.py                 # Relevance scoring
│   │
│   ├── export/                        # Output Generation
│   │   ├── __init__.py
│   │   ├── json_writer.py             # Visualization JSON
│   │   ├── csv_writer.py              # Tabular exports
│   │   ├── h5ad_writer.py             # AnnData output
│   │   └── parquet_writer.py          # Columnar format
│   │
│   └── orchestration/                 # Job Management
│       ├── __init__.py
│       ├── job.py                     # Job definition and state
│       ├── celery_tasks.py            # Improved Celery tasks
│       ├── scheduler.py               # Task scheduling
│       └── recovery.py                # Crash recovery
│
└── tests/
    ├── unit/
    ├── integration/
    └── benchmarks/
```

---

## Pipeline Modules

### 1. Activity Inference Pipeline (`activity/`)

**Purpose:** GPU-accelerated signature scoring using ridge regression

**Components:**
| File | Function |
|------|----------|
| `ridge.py` | SecActpy ridge regression wrapper |
| `parallel.py` | Multi-GPU signature processing |
| `streaming.py` | Disk-streaming for large outputs |
| `signatures.py` | CytoSig (44), SecAct (1,249), custom |

**GPU Acceleration:** 10-34x speedup via CuPy

### 2. Correlation Analysis Pipeline (`correlation/`)

**Purpose:** Compute correlations between activity and continuous variables

**Methods:**
- Pearson correlation (GPU-accelerated matrix ops)
- Spearman correlation (GPU rank transform)
- Partial correlation (confound adjustment)
- Age/BMI binned correlations
- Biochemistry markers (ALT, AST, etc.)
- Metabolite/lipid correlations

**Key Formula:**
```python
# GPU Spearman
ranks = cp.argsort(cp.argsort(X, axis=0), axis=0)
rho = cp.corrcoef(ranks.T)
```

### 3. Differential Analysis Pipeline (`differential/`)

**Purpose:** Compare groups (disease vs healthy, tumor vs adjacent)

**Methods:**
- Wilcoxon rank-sum test (GPU-accelerated)
- Activity difference (not log2FC - values are z-scores)
- Benjamini-Hochberg FDR correction
- Multi-level stratification (cell type × disease × treatment)

**Key Output Fields:**
```python
{
    "activity_diff": mean_a - mean_b,  # NOT log2FC
    "pvalue": float,
    "qvalue": float,  # FDR-corrected
    "neg_log10_pval": -np.log10(pvalue)
}
```

### 4. Validation Pipeline (`validation/`)

**Purpose:** 5-type credibility assessment

| Validation Type | Module | Metric |
|-----------------|--------|--------|
| Sample-level | `sample_level.py` | Expression vs activity R² |
| Cell-type level | `celltype_level.py` | Pseudobulk correlation |
| Pseudobulk vs SC | `pseudobulk_vs_sc.py` | Mean/median concordance |
| Single-cell | `singlecell.py` | Expressing vs non-expressing |
| Biological | `biological.py` | Known marker validation |
| Gene coverage | `gene_coverage.py` | Signature gene detection % |
| CV stability | `cv_stability.py` | Cross-validation robustness |

**Quality Score Formula:**
```python
quality = (
    sample_r2 * 0.20 +
    celltype_r2 * 0.20 +
    gene_coverage * 0.20 +
    cv_stability * 0.20 +
    bio_concordance * 0.20
)
```

### 5. Cross-Atlas Integration Pipeline (`cross_atlas/`)

**Purpose:** Harmonize and compare across CIMA, Inflammation, scAtlas

**Components:**
- Activity harmonization (batch correction)
- Cell type mapping (fuzzy + manual curation)
- Conserved signature detection (Jaccard similarity)
- Meta-analysis aggregation (weighted mean)
- Consistency scoring (CV, Spearman rho)

### 6. Disease Analysis Pipeline (`disease/`)

**Purpose:** Inflammation Atlas disease-specific analyses

**Components:**
| Module | Function |
|--------|----------|
| `activity.py` | Per-disease activity profiles |
| `differential.py` | Disease vs healthy comparison |
| `treatment_response.py` | ML prediction (logistic, RF) |
| `temporal.py` | Longitudinal/timepoint analysis |
| `conserved_programs.py` | Multi-disease shared signatures |
| `driving_populations.py` | Cell type contribution ranking |

**Treatment Response Pipeline:**
```python
# Logistic regression + Random Forest ensemble
X = activity_matrix  # (samples × signatures)
y = response_labels  # 0/1

lr_model = LogisticRegression().fit(X_train, y_train)
rf_model = RandomForestClassifier().fit(X_train, y_train)

# Feature importance (mean of both models)
importance = (lr_coef + rf_importance) / 2
```

### 7. Cancer Analysis Pipeline (`cancer/`)

**Purpose:** scAtlas tumor/cancer analyses

**Components:**
| Module | Function |
|--------|----------|
| `tumor_adjacent.py` | Tumor vs adjacent tissue |
| `infiltration.py` | Immune cell proportions (TME) |
| `exhaustion.py` | T cell exhaustion classification |
| `caf_subtypes.py` | CAF phenotype analysis |
| `cancer_type.py` | Cancer-type stratification |

### 8. Search Index Pipeline (`search/`)

**Purpose:** Build and query search indices

**Entity Types:**
- Cytokines (44 CytoSig)
- Proteins (1,249 SecAct)
- Genes (HGNC symbols)
- Cell types (hierarchical)
- Diseases (20+)
- Organs (35)

**Ranking Algorithm:**
```python
def score(query, entity):
    if exact_match: return 100
    if prefix_match: return 80
    if contains: return 60
    if levenshtein_distance <= 2: return 40
    return 0
```

---

## GPU Acceleration Summary

| Pipeline | GPU Method | Speedup |
|----------|------------|---------|
| Activity inference | CuPy ridge regression | 10-34x |
| Correlation | CuPy matrix ops | 5-15x |
| Differential (Wilcoxon) | CuPy rank operations | 5-10x |
| Validation regression | CuPy least squares | 5-10x |
| Cross-atlas consistency | CuPy correlation | 5-10x |

---

## Data Flow Architecture

```
                    ┌─────────────────────────────────────┐
                    │         DATA SOURCES                │
                    │  ┌─────────┐ ┌─────────┐ ┌───────┐ │
                    │  │Local    │ │cellxgene│ │Remote │ │
                    │  │H5AD     │ │Census   │ │H5AD   │ │
                    │  └────┬────┘ └────┬────┘ └───┬───┘ │
                    └───────┼───────────┼─────────┼──────┘
                            │           │         │
                            ▼           ▼         ▼
                    ┌─────────────────────────────────────┐
                    │         INGEST LAYER                │
                    │   DataSource.iter_chunks()          │
                    │   (chunked, memory-efficient)       │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │       AGGREGATION LAYER             │
                    │  ┌────────┐ ┌──────────┐ ┌───────┐ │
                    │  │CellType│ │Pseudobulk│ │Single │ │
                    │  │L1/L2/L3│ │CT×Sample │ │Cell   │ │
                    │  └────┬───┘ └────┬─────┘ └───┬───┘ │
                    └───────┼──────────┼───────────┼─────┘
                            │          │           │
                            ▼          ▼           ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                    COMPUTE LAYER (GPU)                        │
    │  ┌──────────┐ ┌───────────┐ ┌────────────┐ ┌──────────────┐  │
    │  │Activity  │ │Correlation│ │Differential│ │Validation    │  │
    │  │Inference │ │Analysis   │ │Analysis    │ │Pipeline      │  │
    │  └────┬─────┘ └─────┬─────┘ └──────┬─────┘ └──────┬───────┘  │
    │       │             │              │              │          │
    │  ┌────┴─────┐ ┌─────┴─────┐ ┌──────┴──────┐ ┌────┴────────┐ │
    │  │Cross-    │ │Disease    │ │Cancer       │ │Search       │ │
    │  │Atlas     │ │Analysis   │ │Analysis     │ │Indexing     │ │
    │  └────┬─────┘ └─────┬─────┘ └──────┬──────┘ └──────┬──────┘ │
    └───────┼─────────────┼──────────────┼───────────────┼────────┘
            │             │              │               │
            ▼             ▼              ▼               ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                    EXPORT LAYER                               │
    │  ┌──────┐ ┌─────┐ ┌───────┐ ┌─────────┐ ┌─────────────────┐  │
    │  │JSON  │ │CSV  │ │H5AD   │ │Parquet  │ │Streaming Writer │  │
    │  └──────┘ └─────┘ └───────┘ └─────────┘ └─────────────────┘  │
    └───────────────────────────────────────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                 ORCHESTRATION                                 │
    │  ┌────────────┐ ┌────────────┐ ┌───────────────────────────┐ │
    │  │Celery Jobs │ │Checkpoint  │ │Progress Broadcasting      │ │
    │  │(4h limit)  │ │Recovery    │ │(Redis pub/sub)            │ │
    │  └────────────┘ └────────────┘ └───────────────────────────┘ │
    └───────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (3-4 days)
**Files:**
- `core/config.py`, `core/gpu_manager.py`
- `core/checkpoint.py`, `core/memory.py`, `core/cache.py`

**Tasks:**
- [ ] GPUManager with device selection and memory pooling
- [ ] CheckpointManager with atomic saves
- [ ] Memory estimation (leverage SecActpy)
- [ ] Result caching layer

### Phase 2: Data Ingestion (3-4 days)
**Files:**
- `ingest/base.py`, `ingest/local_h5ad.py`
- `ingest/cellxgene.py`, `ingest/formats.py`

**Tasks:**
- [ ] DataSource interface with `iter_chunks()`
- [ ] LocalH5ADSource with backed mode
- [ ] CellxgeneCensusSource with filtering
- [ ] Format converters (Loom, 10X)

### Phase 3: Aggregation Strategies (2-3 days)
**Files:**
- `aggregation/base.py`, `aggregation/celltype.py`
- `aggregation/pseudobulk.py`, `aggregation/resampling.py`
- `aggregation/singlecell.py`

**Tasks:**
- [ ] Cell type hierarchy (L1/L2/L3 mapping)
- [ ] Standard pseudobulk
- [ ] Bootstrap resampling
- [ ] Single-cell streaming

### Phase 4: Activity Pipeline (3-4 days)
**Files:**
- `activity/ridge.py`, `activity/parallel.py`
- `activity/streaming.py`, `activity/signatures.py`

**Tasks:**
- [ ] SecActpy wrapper with error handling
- [ ] Multi-GPU parallel processing
- [ ] Streaming result writer
- [ ] Signature loading (CytoSig, SecAct, custom)

### Phase 5: Correlation Pipeline (2-3 days)
**Files:**
- `correlation/pearson.py`, `correlation/spearman.py`
- `correlation/continuous.py`, `correlation/biochemistry.py`

**Tasks:**
- [ ] GPU Pearson correlation
- [ ] GPU Spearman (rank transform)
- [ ] Age/BMI binned correlations
- [ ] Biochemistry/metabolite correlations

### Phase 6: Differential Pipeline (2-3 days)
**Files:**
- `differential/wilcoxon.py`, `differential/effect_size.py`
- `differential/fdr.py`, `differential/stratified.py`

**Tasks:**
- [ ] GPU Wilcoxon rank-sum
- [ ] Activity difference (not log2FC)
- [ ] FDR correction
- [ ] Multi-level stratification

### Phase 7: Validation Pipeline (3-4 days)
**Files:**
- `validation/sample_level.py`, `validation/celltype_level.py`
- `validation/biological.py`, `validation/quality_score.py`

**Tasks:**
- [ ] 5-type validation implementation
- [ ] GPU regression
- [ ] Quality score computation
- [ ] Validation JSON generation

### Phase 8: Cross-Atlas Pipeline (2-3 days)
**Files:**
- `cross_atlas/harmonization.py`, `cross_atlas/conserved.py`
- `cross_atlas/meta_analysis.py`, `cross_atlas/consistency.py`

**Tasks:**
- [ ] Cell type mapping
- [ ] Conserved signature detection
- [ ] Meta-analysis aggregation
- [ ] Consistency scoring

### Phase 9: Disease & Cancer Pipelines (3-4 days)
**Files:**
- `disease/treatment_response.py`, `disease/conserved_programs.py`
- `cancer/tumor_adjacent.py`, `cancer/exhaustion.py`

**Tasks:**
- [ ] Treatment response ML models
- [ ] Tumor vs adjacent comparison
- [ ] T cell exhaustion classification
- [ ] CAF subtype analysis

### Phase 10: Search & Export (2 days)
**Files:**
- `search/indexer.py`, `search/gene_mapping.py`
- `export/json_writer.py`, `export/h5ad_writer.py`

**Tasks:**
- [ ] Search index building
- [ ] Gene name mapping (HGNC ↔ CytoSig)
- [ ] JSON/CSV/H5AD writers
- [ ] Streaming export

### Phase 11: Orchestration (2-3 days)
**Files:**
- `orchestration/job.py`, `orchestration/celery_tasks.py`
- `orchestration/recovery.py`

**Tasks:**
- [ ] Job state machine
- [ ] Celery tasks with 4-hour limits
- [ ] Checkpoint recovery
- [ ] Progress broadcasting

### Phase 12: API Integration (2-3 days)
**Files to modify:**
- `cytoatlas-api/app/tasks/process_atlas.py`
- `cytoatlas-api/app/services/submit_service.py`

**Tasks:**
- [ ] Replace naive processing with pipeline
- [ ] Add pipeline configuration endpoints
- [ ] Update WebSocket progress

### Phase 13: Testing & Documentation (3-4 days)
- [ ] Unit tests for each module
- [ ] Integration tests with real data
- [ ] Benchmark suite
- [ ] API documentation

---

## Dependencies

```toml
[project]
name = "cytoatlas-pipeline"
version = "0.1.0"

[project.dependencies]
numpy = ">=1.26.0"
pandas = ">=2.1.0"
scipy = ">=1.12.0"
anndata = ">=0.10.0"
h5py = ">=3.10.0"
cellxgene-census = ">=1.0.0"
tiledbsoma = ">=1.0.0"
cupy-cuda12x = ">=13.0.0"
celery = {extras = ["redis"], version = ">=5.3.0"}
scikit-learn = ">=1.4.0"
statsmodels = ">=0.14.0"

[project.optional-dependencies]
secactpy = {path = "/vf/users/parks34/projects/1ridgesig/SecActpy"}
```

---

## Key Files to Reference

| File | Purpose |
|------|---------|
| `/vf/users/parks34/projects/1ridgesig/SecActpy/secactpy/ridge.py` | GPU ridge regression |
| `/vf/users/parks34/projects/1ridgesig/SecActpy/secactpy/batch.py` | Batch processing |
| `/vf/users/parks34/projects/2secactpy/scripts/01_cima_activity.py` | CIMA pipeline reference |
| `/vf/users/parks34/projects/2secactpy/scripts/02_inflam_activity.py` | Inflammation reference |
| `/vf/users/parks34/projects/2secactpy/scripts/03_scatlas_analysis.py` | scAtlas reference |
| `/vf/users/parks34/projects/2secactpy/scripts/08_atlas_validation.py` | Validation reference |
| `/vf/users/parks34/projects/2secactpy/cytoatlas-api/app/services/` | All service implementations |

---

## Verification Plan

### Unit Tests
```bash
pytest tests/unit/ -v --cov=cytoatlas_pipeline
```

### Integration Tests
```bash
# Test each pipeline with real data
pytest tests/integration/test_activity.py -v
pytest tests/integration/test_correlation.py -v
pytest tests/integration/test_validation.py -v
pytest tests/integration/test_cellxgene.py -v
```

### Benchmark Suite
```bash
python -m cytoatlas_pipeline.benchmark \
    --pipeline activity,correlation,differential \
    --dataset cima \
    --gpu 0,1
```

### End-to-End Validation
1. Process CIMA through full pipeline
2. Compare outputs with existing `visualization/data/` JSONs
3. Verify correlation r > 0.99 between old and new outputs
4. Test checkpoint recovery by killing mid-process
5. Test cellxgene data fetch and processing

---

## Success Criteria

1. **Performance:** Process 19M+ cells without timeout
2. **GPU Utilization:** >80% during compute phases
3. **Accuracy:** Results match existing pipeline (r > 0.99)
4. **Recovery:** Checkpoint recovery works after crash
5. **Integration:** cellxgene queries work end-to-end
6. **Coverage:** All 8 major service pipelines included
7. **Testing:** >80% code coverage

---

## Configuration Example

```python
from cytoatlas_pipeline import Pipeline, Config
from cytoatlas_pipeline.ingest import CellxgeneCensusSource

# Configure pipeline
config = Config(
    gpu_devices=[0, 1],
    checkpoint_interval=300,
    batch_size=10000,
    n_rand=1000,
)

pipeline = Pipeline(config)

# Example: Process user submission
results = pipeline.process_submission(
    h5ad_path="/data/user_upload.h5ad",
    signatures=["CytoSig", "SecAct"],
    aggregation_levels=["celltype_L2", "pseudobulk"],
    analyses=["activity", "correlation", "differential", "validation"],
    output_dir="/data/results/"
)

# Example: Query cellxgene
source = CellxgeneCensusSource().query(
    tissue="blood",
    cell_type=["T cell", "B cell"],
    disease="COVID-19",
    max_cells=500_000
)

results = pipeline.process(
    source=source,
    signatures=["CytoSig"],
    analyses=["activity", "disease_differential"],
)
```

---

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 1-3 | Week 1 | Core + Ingest + Aggregation |
| 4-6 | Week 2 | Activity + Correlation + Differential |
| 7-9 | Week 3 | Validation + Cross-Atlas + Disease/Cancer |
| 10-11 | Week 4 | Search + Export + Orchestration |
| 12-13 | Week 5 | API Integration + Testing |

**Total: ~5 weeks**
