# CytoAtlas Validate Menu Implementation Plan

## Overview

Implement 4-tab validation panel in CytoAtlas mirroring visualization/index.html Atlas Validation section.

## Current State

- **visualization/index.html**: Has 4 tabs (Atlas Level, Pseudobulk Level, Single-Cell Level, Summary)
- **cytoatlas-api/validate.js**: Has signature-specific 5-type credibility assessment - needs tab restructure
- **Validation JSON files**: Exist for all 3 atlases (~175-336MB each) with comprehensive data

### Validation Data Structure (per atlas JSON)
```json
{
    "atlas": "cima",
    "signature_types": ["CytoSig", "LinCytoSig", "SecAct"],
    "sample_validations": [...],      // Pseudobulk level data
    "celltype_validations": [...],    // Atlas level data
    "singlecell_validations": [...],  // Single-cell level data
    "gene_coverage": [...],
    "biological_associations": [...]
}
```

## Architecture Design

### Tab Structure
```
/validate
├── Tab 1: Atlas Level      - Cell type-aggregated (1 point per cell type)
├── Tab 2: Pseudobulk Level - Sample-level (sample × cell type)
├── Tab 3: Single-Cell Level - Per-cell correlation
└── Tab 4: Summary          - Overview metrics & comparison
```

### Controls (shared across tabs)
- Atlas selector: CIMA / Inflammation / scAtlas
- Signature type: CytoSig (43) / LinCytoSig / SecAct (1,170)
- Signature selector: Dropdown + search
- Cell type filter (where applicable)

## Implementation Steps

### Phase 1: Backend Endpoints

**File**: `cytoatlas-api/app/routers/validation.py`

| Endpoint | Purpose | Data Source |
|----------|---------|-------------|
| `GET /validation/atlas-level` | Cell type means scatter | `celltype_validations` |
| `GET /validation/atlas-level/ranking` | Signature correlation ranking | `celltype_validations` |
| `GET /validation/atlas-level/heatmap` | Cross-atlas cell type heatmap | All 3 atlases |
| `GET /validation/pseudobulk` | Sample × celltype scatter | `sample_validations` |
| `GET /validation/pseudobulk/distribution` | Correlation distribution | `sample_validations` |
| `GET /validation/pseudobulk/heatmap` | Per-signature sorted bar | `sample_validations` |
| `GET /validation/singlecell/scatter` | Cell-level scatter | `singlecell_validations` |
| `GET /validation/singlecell/celltype` | By cell type correlation | `singlecell_validations` |
| `GET /validation/summary/sigtype` | Mean r by signature type | Aggregated |
| `GET /validation/summary/atlas` | Mean r by atlas | Aggregated |
| `GET /validation/summary/levels` | Comparison across levels | Aggregated |

### Phase 2: API Client Updates

**File**: `cytoatlas-api/static/js/api.js`

Add methods:
```javascript
// Atlas Level
async getAtlasLevelValidation(atlas, sigType, signature) { ... }
async getAtlasLevelRanking(atlas, sigType) { ... }
async getAtlasLevelHeatmap(sigType) { ... }

// Pseudobulk Level
async getPseudobulkValidation(atlas, sigType, signature, celltype) { ... }
async getPseudobulkDistribution(atlas, sigType) { ... }
async getPseudobulkHeatmap(atlas, sigType) { ... }

// Single-Cell Level
async getSingleCellScatter(atlas, sigType, signature) { ... }
async getSingleCellByCelltype(atlas, sigType, signature) { ... }

// Summary
async getValidationSummaryBySignatureType() { ... }
async getValidationSummaryByAtlas() { ... }
async getValidationSummaryLevels() { ... }
```

### Phase 3: Frontend Restructure

**File**: `cytoatlas-api/static/js/pages/validate.js` (expand from ~465 to ~900 lines)

```javascript
const ValidatePage = {
    // State
    currentAtlas: 'cima',
    signatureType: 'CytoSig',
    selectedSignature: null,
    selectedCelltype: 'all',
    activeTab: 'atlas-level',

    // Signature lists (cached)
    signatures: {
        CytoSig: [],
        LinCytoSig: [],
        SecAct: []
    },
    celltypes: [],

    // Methods
    async init(params, query) { ... },
    render() { ... },

    // Tab navigation
    switchTab(tab) { ... },

    // Tab loaders
    async loadAtlasLevel() { ... },
    async loadPseudobulkLevel() { ... },
    async loadSingleCellLevel() { ... },
    async loadSummary() { ... },

    // Plot creators
    createExpressionActivityScatter(containerId, data) { ... },
    createCorrelationDistribution(containerId, data) { ... },
    createSignatureRanking(containerId, data) { ... },
    createCrossAtlasHeatmap(containerId, data) { ... },
};
```

### Phase 4: Template Update

**File**: `cytoatlas-api/static/index.html`

Update validate-template:
```html
<template id="validate-template">
    <div class="page validate-page">
        <div class="page-header">
            <h1>Activity Validation</h1>
            <p>Validation of activity predictions by correlating inferred scores with signature gene expression</p>
        </div>

        <div class="page-controls">
            <select id="val-atlas">
                <option value="cima">CIMA</option>
                <option value="inflammation">Inflammation</option>
                <option value="scatlas">scAtlas</option>
            </select>
            <select id="val-sigtype">
                <option value="CytoSig">CytoSig (43)</option>
                <option value="LinCytoSig">LinCytoSig</option>
                <option value="SecAct">SecAct (1,170)</option>
            </select>
        </div>

        <div class="tab-navigation">
            <button data-tab="atlas-level" class="active">Atlas Level</button>
            <button data-tab="pseudobulk">Pseudobulk Level</button>
            <button data-tab="singlecell">Single-Cell Level</button>
            <button data-tab="summary">Summary</button>
        </div>

        <div class="tab-content" id="validation-content"></div>
    </div>
</template>
```

### Phase 5: Panel Implementations

#### Tab 1: Atlas Level
- **Left panel**: Expression vs Activity scatter (1 point per cell type)
- **Right panel**: Signature correlation ranking (sorted bar chart)
- **Bottom panel**: Cross-atlas cell type heatmap (all 3 atlases)
- Controls: Signature selector with search

#### Tab 2: Pseudobulk Level
- **Left panel**: Expression vs Activity scatter (1 point per sample)
- **Right panel**: Correlation distribution histogram
- **Bottom panel**: Per-signature correlations (sorted bar chart)
- Controls: Signature selector, Cell type filter

#### Tab 3: Single-Cell Level
- **Left panel**: Expression vs Activity scatter (subsample of cells)
- **Right panel**: Correlation by cell type (grouped bar)
- **Bottom panel**: Summary table or additional viz
- Controls: Signature selector

#### Tab 4: Summary
- **Left panel**: Mean correlation by signature type (grouped bar)
- **Right panel**: Mean correlation by atlas (grouped bar)
- **Bottom panel**: Validation level comparison (pseudobulk vs sc vs atlas)
- Key findings text panel

## Files to Modify

| File | Changes |
|------|---------|
| `cytoatlas-api/app/routers/validation.py` | Add ~11 new endpoints |
| `cytoatlas-api/app/services/validation_service.py` | Add new service methods |
| `cytoatlas-api/static/js/pages/validate.js` | Full restructure with tabs |
| `cytoatlas-api/static/index.html` | Update validate-template |
| `cytoatlas-api/static/js/api.js` | Add ~11 API methods |
| `cytoatlas-api/static/css/main.css` | Add validation tab styles |

## Data Flow

```
User selects Atlas + Signature Type + Signature
        ↓
API endpoint receives params
        ↓
Service loads {atlas}_validation.json
        ↓
Filters by signature_type, signature
        ↓
Returns formatted data for Plotly
        ↓
Frontend creates visualization
```

## Verification

1. **Backend**: Test each endpoint
   ```bash
   curl "http://localhost:8000/api/v1/validation/atlas-level?atlas=cima&sig_type=CytoSig&signature=IFNG"
   curl "http://localhost:8000/api/v1/validation/pseudobulk?atlas=cima&sig_type=CytoSig"
   ```

2. **Frontend**: Navigate to `/validate` and verify:
   - All 4 tabs load correctly
   - Atlas/signature type/signature filters work
   - Scatter plots show proper correlations
   - Cross-atlas comparisons render

## Implementation Order

1. Backend endpoints (validation.py + validation_service.py)
2. API client methods (api.js)
3. Template update (index.html)
4. validate.js restructure with tab system
5. Individual tab implementations
6. CSS styling refinements
7. Testing across all atlases
