# CytoAtlas Compare Menu Implementation Plan

## Overview

Implement 5-panel Compare menu in CytoAtlas mirroring visualization/index.html cross-atlas section, with extensibility for user-submitted datasets.

## Current State

- **visualization/index.html**: Has 5 tabs (Overview, Cell Type Mapping, Atlas Comparison, Conserved Signatures, Meta-Analysis)
- **cytoatlas-api/compare.js**: Has 3 basic panels (quality, correlation, consistency) - needs full restructure
- **cross_atlas.py router**: 28 endpoints exist but need 5 new ones for complete feature parity
- **cross_atlas.json**: Contains all necessary data (5.5MB with summary, celltype_mapping, atlas_comparison, signature_reliability, meta_analysis)

## Architecture Design

### Tab Structure
```
/compare
├── Tab 1: Overview          - Atlas summary statistics
├── Tab 2: Cell Type Mapping - Sankey diagrams + heatmaps
├── Tab 3: Atlas Comparison  - Pairwise scatter plots
├── Tab 4: Conserved         - Signature reliability heatmap
└── Tab 5: Meta-Analysis     - Forest plots
```

### Extensibility Design
- **Dynamic Atlas Selector**: Checkbox multi-select for N atlases (not hardcoded 3)
- **Atlas Registry Integration**: Use existing `atlas_registry.py` for available atlases
- **Pairwise Generation**: Auto-generate (N choose 2) pairs for comparisons
- **Future-proof API**: Pass `atlases[]` parameter to all endpoints

## Implementation Steps

### Phase 1: Backend Endpoints

**File**: `cytoatlas-api/app/routers/cross_atlas.py`

| Endpoint | Purpose | Data Source |
|----------|---------|-------------|
| `GET /cross-atlas/summary` | Overview stats | `cross_atlas.json > summary` |
| `GET /cross-atlas/celltype-sankey` | Sankey diagram | `cross_atlas.json > celltype_mapping.sankey` |
| `GET /cross-atlas/pairwise-scatter` | Scatter data | `cross_atlas.json > atlas_comparison` |
| `GET /cross-atlas/signature-reliability` | Reliability data | `cross_atlas.json > signature_reliability` |
| `GET /cross-atlas/meta-analysis/forest` | Forest plot | `cross_atlas.json > meta_analysis` |

**File**: `cytoatlas-api/app/services/cross_atlas_service.py`

Add methods:
- `get_summary()` - Returns aggregated stats
- `get_celltype_sankey(level, lineage)` - Returns Sankey nodes/links
- `get_pairwise_scatter(atlas1, atlas2, sig_type, level)` - Returns scatter data
- `get_signature_reliability(sig_type)` - Returns per-signature correlations
- `get_meta_analysis_forest(analysis, sig_type)` - Returns forest plot data

### Phase 2: Frontend Restructure

**File**: `cytoatlas-api/static/js/pages/compare.js` (expand from 214 to ~800 lines)

```javascript
const ComparePage = {
    // State
    signatureType: 'CytoSig',
    selectedAtlases: [],  // Dynamic, not hardcoded
    activeTab: 'overview',
    availableAtlases: [],

    // Tab-specific state
    tabs: {
        celltypeMapping: { level: 'coarse', lineage: 'all' },
        atlasComparison: { pair: null, level: 'coarse' },
        conserved: { filter: 'all' },
        metaAnalysis: { analysis: 'age' }
    },

    // Methods
    async init() { ... },
    render() { ... },

    // Tab loaders
    async loadOverview() { ... },
    async loadCelltypeMapping() { ... },
    async loadAtlasComparison() { ... },
    async loadConservedSignatures() { ... },
    async loadMetaAnalysis() { ... },

    // Atlas selector
    renderAtlasSelector() { ... },
    onAtlasSelectionChange() { ... },

    // Utility
    getAtlasPairs() { ... },  // Generate (N choose 2)
};
```

**File**: `cytoatlas-api/static/index.html`

Update compare-template:
```html
<template id="compare-template">
    <div class="page compare-page">
        <div class="page-header">
            <h1>Cross-Atlas Comparison</h1>
            <div class="page-controls">
                <div class="atlas-selector" id="atlas-selector"></div>
                <select id="signature-type-select">
                    <option value="CytoSig">CytoSig (43)</option>
                    <option value="SecAct">SecAct (1,249)</option>
                </select>
            </div>
        </div>
        <div class="tab-navigation">
            <button data-tab="overview" class="active">Overview</button>
            <button data-tab="celltype-mapping">Cell Type Mapping</button>
            <button data-tab="atlas-comparison">Atlas Comparison</button>
            <button data-tab="conserved">Conserved Signatures</button>
            <button data-tab="meta-analysis">Meta-Analysis</button>
        </div>
        <div class="tab-content" id="compare-content"></div>
    </div>
</template>
```

### Phase 3: Panel Implementations

#### Tab 1: Overview
- 4 stat cards (Total Cells, Samples, Cell Types, Signatures)
- Bar chart: cells/samples/types by atlas
- Uses `Plotly.newPlot()` with dual Y-axis

#### Tab 2: Cell Type Mapping
- Sankey diagram (atlas → harmonized lineage)
- Grouped bar chart (types per lineage per atlas)
- Controls: Coarse/Fine level, Lineage filter
- **New component**: `SankeyChart.create(containerId, nodes, links)`

#### Tab 3: Atlas Comparison
- Scatter plot (X: Atlas1 activity, Y: Atlas2 activity)
- Bar chart (per-cell-type correlation)
- Controls: Atlas pair dropdown (dynamic from selected), Level, View type
- Uses existing `Scatter.createCorrelationScatter()`

#### Tab 4: Conserved Signatures
- Summary cards (highly/moderately/atlas-specific counts)
- Heatmap (signatures × atlas pairs, r values)
- Sortable table with search
- Uses existing `Heatmap.createCorrelationHeatmap()`

#### Tab 5: Meta-Analysis
- Summary cards (replicated, consistent direction, significant)
- Forest plot (individual + pooled effects)
- I² heterogeneity bar chart
- Controls: Analysis type (age/bmi/sex), Gene search
- **New component**: `ForestPlot.create(containerId, data)`

### Phase 4: New Components

**File**: `cytoatlas-api/static/js/components/sankey.js` (~100 lines)
```javascript
const SankeyChart = {
    create(containerId, nodes, links, options = {}) {
        const trace = {
            type: 'sankey',
            node: { label: nodes.map(n => n.label), color: nodes.map(n => n.color) },
            link: { source: links.map(l => l.source), target: links.map(l => l.target), value: links.map(l => l.value) }
        };
        Plotly.newPlot(containerId, [trace], layout, {responsive: true});
    }
};
```

**File**: `cytoatlas-api/static/js/components/forest-plot.js` (~150 lines)
```javascript
const ForestPlot = {
    create(containerId, signatures, options = {}) {
        // Horizontal dot plot with error bars
        // Diamond for pooled estimates
        // Vertical line at 0
    }
};
```

### Phase 5: API Client Updates

**File**: `cytoatlas-api/static/js/api.js`

```javascript
// Add to cross-atlas section
async getCrossAtlasSummary() {
    return this.get('/cross-atlas/summary');
},
async getCelltypeSankey(params = {}) {
    return this.get('/cross-atlas/celltype-sankey', params);
},
async getPairwiseScatter(params = {}) {
    return this.get('/cross-atlas/pairwise-scatter', params);
},
async getSignatureReliability(params = {}) {
    return this.get('/cross-atlas/signature-reliability', params);
},
async getMetaAnalysisForest(params = {}) {
    return this.get('/cross-atlas/meta-analysis/forest', params);
},
```

## Files to Modify

| File | Changes |
|------|---------|
| `cytoatlas-api/app/routers/cross_atlas.py` | Add 5 new endpoints |
| `cytoatlas-api/app/services/cross_atlas_service.py` | Add 5 new service methods |
| `cytoatlas-api/static/js/pages/compare.js` | Full restructure (214→800 lines) |
| `cytoatlas-api/static/index.html` | Update compare-template |
| `cytoatlas-api/static/js/api.js` | Add 5 API methods |
| `cytoatlas-api/static/js/components/sankey.js` | New file |
| `cytoatlas-api/static/js/components/forest-plot.js` | New file |
| `cytoatlas-api/static/css/main.css` | Add compare page styles |

## Extensibility for User Datasets

1. **Atlas Registry Integration**: `get_available_atlases()` reads from registry
2. **Dynamic Pair Generation**: Frontend generates pairs from selected atlases
3. **Lazy Computation**: When new atlas added, comparisons computed on-demand
4. **Feature Flags**: Atlases can have `supports_comparison: true/false`

## Verification

1. **Backend**: Test each new endpoint with curl
   ```bash
   curl http://localhost:8000/api/v1/cross-atlas/summary
   curl http://localhost:8000/api/v1/cross-atlas/celltype-sankey?level=coarse
   curl http://localhost:8000/api/v1/cross-atlas/pairwise-scatter?atlas1=CIMA&atlas2=Inflammation
   ```

2. **Frontend**: Navigate to `/compare` and verify:
   - All 5 tabs load correctly
   - Atlas selector shows all available atlases
   - Signature type filter works
   - Tab-specific controls function
   - Visualizations render correctly

3. **Extensibility**: Add test atlas and verify it appears in selector

## Implementation Order

1. Backend endpoints (cross_atlas.py + cross_atlas_service.py)
2. API client methods (api.js)
3. New components (sankey.js, forest-plot.js)
4. compare.js restructure with tab system
5. Individual tab implementations
6. CSS styling
7. Testing and refinement
