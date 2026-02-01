# {DATASET_NAME} Dataset

## Overview

| Property | Value |
|----------|-------|
| **Cells** | {N_CELLS} |
| **Samples** | {N_SAMPLES} |
| **Genes** | {N_GENES} |
| **Source** | {SOURCE} |
| **File Size** | {FILE_SIZE} |

## File Paths

```python
# Main H5AD file
H5AD_PATH = '{H5AD_PATH}'

# Metadata files
METADATA_PATH = '{METADATA_PATH}'
```

## Cell Observations (`.obs`)

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `{COL_NAME}` | {TYPE} | {DESCRIPTION} | {EXAMPLES} |

## Gene Variables (`.var`)

| Column | Type | Description |
|--------|------|-------------|
| `{COL_NAME}` | {TYPE} | {DESCRIPTION} |

## Metadata Files

### {METADATA_FILE_NAME}

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `{COL_NAME}` | {TYPE} | {DESCRIPTION} | {EXAMPLES} |

## Cell Type Hierarchy

```
Level 1 (coarse)
├── {L1_TYPE_1}
│   ├── {L2_TYPE_1}
│   └── {L2_TYPE_2}
└── {L1_TYPE_2}
    └── {L2_TYPE_3}
```

## Usage Examples

### Loading the dataset

```python
import anndata as ad

# Full load (requires ~{MEMORY}GB RAM)
adata = ad.read_h5ad('{H5AD_PATH}')

# Backed mode (memory efficient)
adata = ad.read_h5ad('{H5AD_PATH}', backed='r')
```

### Accessing metadata

```python
# Cell type distribution
adata.obs['{CELLTYPE_COL}'].value_counts()

# Sample metadata
sample_meta = pd.read_csv('{METADATA_PATH}')
```

## Related Pipelines

- [{PIPELINE_NAME}](../pipelines/{ATLAS}/activity.md)
