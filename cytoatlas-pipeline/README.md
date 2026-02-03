# CytoAtlas Pipeline

GPU-accelerated data processing pipeline for single-cell cytokine activity analysis.

## Installation

```bash
pip install -e .
```

With GPU support:
```bash
pip install -e ".[gpu]"
```

With all optional dependencies:
```bash
pip install -e ".[all]"
```

## Quick Start

```python
from cytoatlas_pipeline import Pipeline, Config
from cytoatlas_pipeline.ingest import LocalH5ADSource

# Configure pipeline
config = Config(gpu_devices=[0], batch_size=10000)
pipeline = Pipeline(config)

# Load data and run analysis
source = LocalH5ADSource("/path/to/data.h5ad")
results = pipeline.process(
    source=source,
    signatures=["CytoSig"],
    analyses=["activity", "correlation"]
)
```

## Features

- **Activity Inference**: Ridge regression with permutation-based significance testing
- **Correlation Analysis**: Pearson, Spearman, and partial correlations
- **Differential Analysis**: Wilcoxon rank-sum with FDR correction
- **Validation**: 5-type credibility assessment
- **Cross-Atlas Integration**: Harmonization and meta-analysis
- **GPU Acceleration**: 10-34x speedup via CuPy

## Running Tests

```bash
pytest tests/ -v
```
