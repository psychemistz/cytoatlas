/**
 * Compare Page Handler
 * Cross-atlas comparison views with 5-panel tab system
 */

const ComparePage = {
    // State
    signatureType: 'CytoSig',
    selectedAtlases: ['CIMA', 'Inflammation', 'scAtlas'],
    availableAtlases: [],
    activeTab: 'overview',

    // Tab-specific state
    tabs: {
        celltypeMapping: { level: 'coarse', lineage: 'all' },
        atlasComparison: { pair: 'CIMA-Inflammation', level: 'coarse' },
        conserved: { filter: 'all' },
        metaAnalysis: { analysis: 'age', signature: null },
    },

    /**
     * Initialize the compare page
     */
    async init(params, query) {
        // Parse query parameters
        if (query.type) {
            this.signatureType = query.type;
        }
        if (query.tab) {
            this.activeTab = query.tab;
        }

        // Load available atlases
        await this.loadAvailableAtlases();

        // Render template
        this.render();

        // Load initial tab content
        await this.loadActiveTab();
    },

    /**
     * Load available atlases from API
     */
    async loadAvailableAtlases() {
        try {
            const atlases = await API.getCrossAtlasAtlases();
            this.availableAtlases = atlases || ['CIMA', 'Inflammation', 'scAtlas'];
            this.selectedAtlases = [...this.availableAtlases];
        } catch (error) {
            console.error('Failed to load atlases:', error);
            this.availableAtlases = ['CIMA', 'Inflammation', 'scAtlas'];
            this.selectedAtlases = [...this.availableAtlases];
        }
    },

    /**
     * Generate atlas pairs from selected atlases
     */
    getAtlasPairs() {
        const pairs = [];
        for (let i = 0; i < this.selectedAtlases.length; i++) {
            for (let j = i + 1; j < this.selectedAtlases.length; j++) {
                pairs.push({
                    key: `${this.selectedAtlases[i]}-${this.selectedAtlases[j]}`,
                    label: `${this.selectedAtlases[i]} vs ${this.selectedAtlases[j]}`,
                    atlas1: this.selectedAtlases[i],
                    atlas2: this.selectedAtlases[j],
                });
            }
        }
        return pairs;
    },

    /**
     * Render the page template
     */
    render() {
        const app = document.getElementById('app');
        if (!app) return;

        app.innerHTML = `
            <div class="page compare-page">
                <div class="page-header">
                    <h1>Cross-Atlas Comparison</h1>
                    <div class="page-controls">
                        <div class="atlas-selector" id="atlas-selector">
                            ${this.renderAtlasSelector()}
                        </div>
                        <select id="signature-type-select" class="filter-select" onchange="ComparePage.changeSignatureType(this.value)">
                            <option value="CytoSig" ${this.signatureType === 'CytoSig' ? 'selected' : ''}>CytoSig (43)</option>
                            <option value="SecAct" ${this.signatureType === 'SecAct' ? 'selected' : ''}>SecAct (1,249)</option>
                        </select>
                    </div>
                </div>

                <div class="tab-navigation">
                    <button class="tab-btn ${this.activeTab === 'overview' ? 'active' : ''}" data-tab="overview" onclick="ComparePage.switchTab('overview')">
                        Overview
                    </button>
                    <button class="tab-btn ${this.activeTab === 'celltype-mapping' ? 'active' : ''}" data-tab="celltype-mapping" onclick="ComparePage.switchTab('celltype-mapping')">
                        Cell Type Mapping
                    </button>
                    <button class="tab-btn ${this.activeTab === 'atlas-comparison' ? 'active' : ''}" data-tab="atlas-comparison" onclick="ComparePage.switchTab('atlas-comparison')">
                        Atlas Comparison
                    </button>
                    <button class="tab-btn ${this.activeTab === 'conserved' ? 'active' : ''}" data-tab="conserved" onclick="ComparePage.switchTab('conserved')">
                        Conserved Signatures
                    </button>
                    <button class="tab-btn ${this.activeTab === 'meta-analysis' ? 'active' : ''}" data-tab="meta-analysis" onclick="ComparePage.switchTab('meta-analysis')">
                        Meta-Analysis
                    </button>
                </div>

                <div class="tab-content" id="compare-content">
                    <div class="loading">Loading...</div>
                </div>
            </div>
        `;
    },

    /**
     * Render atlas selector checkboxes
     */
    renderAtlasSelector() {
        return this.availableAtlases.map(atlas => `
            <label class="atlas-checkbox">
                <input type="checkbox" value="${atlas}"
                       ${this.selectedAtlases.includes(atlas) ? 'checked' : ''}
                       onchange="ComparePage.onAtlasSelectionChange()">
                <span>${atlas}</span>
            </label>
        `).join('');
    },

    /**
     * Handle atlas selection change
     */
    onAtlasSelectionChange() {
        const checkboxes = document.querySelectorAll('.atlas-checkbox input[type="checkbox"]');
        this.selectedAtlases = Array.from(checkboxes)
            .filter(cb => cb.checked)
            .map(cb => cb.value);

        // Need at least 2 atlases for comparison
        if (this.selectedAtlases.length < 2) {
            alert('Please select at least 2 atlases for comparison');
            return;
        }

        // Update pair dropdown if on atlas comparison tab
        if (this.activeTab === 'atlas-comparison') {
            this.updatePairDropdown();
        }

        // Reload current tab
        this.loadActiveTab();
    },

    /**
     * Update the pair dropdown for atlas comparison
     */
    updatePairDropdown() {
        const select = document.getElementById('pair-select');
        if (!select) return;

        const pairs = this.getAtlasPairs();
        select.innerHTML = pairs.map(p => `
            <option value="${p.key}" ${this.tabs.atlasComparison.pair === p.key ? 'selected' : ''}>
                ${p.label}
            </option>
        `).join('');

        // Ensure selected pair is valid
        if (!pairs.find(p => p.key === this.tabs.atlasComparison.pair)) {
            this.tabs.atlasComparison.pair = pairs[0]?.key || null;
        }
    },

    /**
     * Switch to a different tab
     */
    switchTab(tab) {
        this.activeTab = tab;

        // Update tab button states
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tab);
        });

        // Load tab content
        this.loadActiveTab();
    },

    /**
     * Load content for the active tab
     */
    async loadActiveTab() {
        const content = document.getElementById('compare-content');
        if (!content) return;

        content.innerHTML = '<div class="loading">Loading...</div>';

        try {
            switch (this.activeTab) {
                case 'overview':
                    await this.loadOverview();
                    break;
                case 'celltype-mapping':
                    await this.loadCelltypeMapping();
                    break;
                case 'atlas-comparison':
                    await this.loadAtlasComparison();
                    break;
                case 'conserved':
                    await this.loadConservedSignatures();
                    break;
                case 'meta-analysis':
                    await this.loadMetaAnalysis();
                    break;
                default:
                    content.innerHTML = '<p>Unknown tab</p>';
            }
        } catch (error) {
            content.innerHTML = `<p class="error">Failed to load: ${error.message}</p>`;
            console.error('Tab load error:', error);
        }
    },

    /**
     * Tab 1: Overview - Atlas summary statistics
     */
    async loadOverview() {
        const content = document.getElementById('compare-content');
        const data = await API.getCrossAtlasSummary();

        content.innerHTML = `
            <div class="overview-tab">
                <div class="stat-cards">
                    <div class="stat-card">
                        <div class="stat-value">${this.formatNumber(data.total_cells)}</div>
                        <div class="stat-label">Total Cells</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${this.formatNumber(data.total_samples)}</div>
                        <div class="stat-label">Total Samples</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${data.total_cell_types}</div>
                        <div class="stat-label">Cell Types</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${this.signatureType === 'CytoSig' ? data.n_signatures_cytosig : data.n_signatures_secact}</div>
                        <div class="stat-label">Signatures (${this.signatureType})</div>
                    </div>
                </div>

                <div class="overview-grid">
                    <div class="overview-panel">
                        <h3>Cells by Atlas</h3>
                        <div id="overview-cells-chart" class="plot-container"></div>
                    </div>
                    <div class="overview-panel">
                        <h3>Samples by Atlas</h3>
                        <div id="overview-samples-chart" class="plot-container"></div>
                    </div>
                </div>

                <div class="atlas-details">
                    <h3>Atlas Details</h3>
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Atlas</th>
                                <th>Cells</th>
                                <th>Samples</th>
                                <th>Cell Types</th>
                                <th>Focus</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${this.renderAtlasDetailsRows(data.atlases)}
                        </tbody>
                    </table>
                </div>
            </div>
        `;

        // Render charts
        this.renderOverviewCharts(data.atlases);
    },

    /**
     * Render atlas details table rows
     */
    renderAtlasDetailsRows(atlases) {
        const focusMap = {
            cima: 'Healthy aging (multi-ethnic)',
            inflammation: 'Inflammatory diseases',
            scatlas_normal: 'Normal organs (35 tissues)',
            scatlas_cancer: 'Pan-cancer',
        };

        return Object.entries(atlases).map(([key, atlas]) => `
            <tr>
                <td><strong>${this.formatAtlasName(key)}</strong></td>
                <td>${this.formatNumber(atlas.cells)}</td>
                <td>${this.formatNumber(atlas.samples)}</td>
                <td>${atlas.cell_types}</td>
                <td>${focusMap[key] || ''}</td>
            </tr>
        `).join('');
    },

    /**
     * Render overview bar charts
     */
    renderOverviewCharts(atlases) {
        const atlasNames = Object.keys(atlases).map(k => this.formatAtlasName(k));
        const cells = Object.values(atlases).map(a => a.cells);
        const samples = Object.values(atlases).map(a => a.samples);

        const atlasColors = ['#3b82f6', '#f59e0b', '#10b981', '#8b5cf6'];

        // Cells chart
        Plotly.newPlot('overview-cells-chart', [{
            type: 'bar',
            x: atlasNames,
            y: cells,
            marker: { color: atlasColors },
            text: cells.map(c => this.formatNumber(c)),
            textposition: 'outside',
        }], {
            margin: { l: 60, r: 20, t: 20, b: 60 },
            yaxis: { title: 'Cells' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
        }, { responsive: true });

        // Samples chart
        Plotly.newPlot('overview-samples-chart', [{
            type: 'bar',
            x: atlasNames,
            y: samples,
            marker: { color: atlasColors },
            text: samples.map(s => this.formatNumber(s)),
            textposition: 'outside',
        }], {
            margin: { l: 60, r: 20, t: 20, b: 60 },
            yaxis: { title: 'Samples' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
        }, { responsive: true });
    },

    /**
     * Tab 2: Cell Type Mapping - Sankey diagram + heatmap
     */
    async loadCelltypeMapping() {
        const content = document.getElementById('compare-content');

        content.innerHTML = `
            <div class="celltype-mapping-tab">
                <div class="tab-controls">
                    <div class="control-group">
                        <label>Mapping Level</label>
                        <select id="mapping-level-select" class="filter-select" onchange="ComparePage.changeMappingLevel(this.value)">
                            <option value="coarse" ${this.tabs.celltypeMapping.level === 'coarse' ? 'selected' : ''}>Coarse (Lineages)</option>
                            <option value="fine" ${this.tabs.celltypeMapping.level === 'fine' ? 'selected' : ''}>Fine (Cell Types)</option>
                        </select>
                    </div>
                </div>

                <div class="sankey-container">
                    <h3>Cell Type Harmonization</h3>
                    <p class="viz-subtitle">Flow diagram showing cell type mapping across atlases</p>
                    <div id="celltype-sankey" class="plot-container plot-large"></div>
                </div>

                <div class="mapping-summary" id="mapping-summary">
                    <!-- Summary will be loaded -->
                </div>
            </div>
        `;

        await this.loadSankeyDiagram();
    },

    /**
     * Load Sankey diagram data
     */
    async loadSankeyDiagram() {
        const data = await API.getCelltypeSankey({
            level: this.tabs.celltypeMapping.level,
        });

        if (data && data.nodes && data.links) {
            SankeyChart.createCelltypeMapping('celltype-sankey', data, {
                title: '',
            });

            // Update summary
            const summaryEl = document.getElementById('mapping-summary');
            if (summaryEl && data.summary) {
                summaryEl.innerHTML = `
                    <div class="summary-cards">
                        <div class="summary-card">
                            <div class="summary-value">${data.summary.n_shared || 0}</div>
                            <div class="summary-label">Shared Cell Types</div>
                        </div>
                        <div class="summary-card">
                            <div class="summary-value">${data.summary.n_cima_unique || 0}</div>
                            <div class="summary-label">CIMA Unique</div>
                        </div>
                        <div class="summary-card">
                            <div class="summary-value">${data.summary.n_inflammation_unique || 0}</div>
                            <div class="summary-label">Inflammation Unique</div>
                        </div>
                        <div class="summary-card">
                            <div class="summary-value">${data.summary.n_scatlas_unique || 0}</div>
                            <div class="summary-label">scAtlas Unique</div>
                        </div>
                    </div>
                `;
            }
        } else {
            document.getElementById('celltype-sankey').innerHTML =
                '<p class="loading">No cell type mapping data available</p>';
        }
    },

    /**
     * Change mapping level
     */
    changeMappingLevel(level) {
        this.tabs.celltypeMapping.level = level;
        this.loadSankeyDiagram();
    },

    /**
     * Tab 3: Atlas Comparison - Pairwise scatter plots
     */
    async loadAtlasComparison() {
        const content = document.getElementById('compare-content');
        const pairs = this.getAtlasPairs();

        // Set default pair if not set
        if (!this.tabs.atlasComparison.pair && pairs.length > 0) {
            this.tabs.atlasComparison.pair = pairs[0].key;
        }

        content.innerHTML = `
            <div class="atlas-comparison-tab">
                <div class="tab-controls">
                    <div class="control-group">
                        <label>Atlas Pair</label>
                        <select id="pair-select" class="filter-select" onchange="ComparePage.changePair(this.value)">
                            ${pairs.map(p => `
                                <option value="${p.key}" ${this.tabs.atlasComparison.pair === p.key ? 'selected' : ''}>
                                    ${p.label}
                                </option>
                            `).join('')}
                        </select>
                    </div>
                    <div class="control-group">
                        <label>Aggregation</label>
                        <select id="comparison-level-select" class="filter-select" onchange="ComparePage.changeComparisonLevel(this.value)">
                            <option value="coarse" ${this.tabs.atlasComparison.level === 'coarse' ? 'selected' : ''}>Coarse (Lineages)</option>
                            <option value="fine" ${this.tabs.atlasComparison.level === 'fine' ? 'selected' : ''}>Fine (Cell Types)</option>
                        </select>
                    </div>
                </div>

                <div class="comparison-grid">
                    <div class="comparison-panel main-panel">
                        <h3>Activity Correlation</h3>
                        <div id="scatter-plot" class="plot-container plot-large"></div>
                    </div>
                    <div class="comparison-panel side-panel">
                        <h3>Per-Cell-Type Correlations</h3>
                        <div id="celltype-correlations" class="plot-container"></div>
                    </div>
                </div>

                <div class="correlation-stats" id="correlation-stats">
                    <!-- Stats will be loaded -->
                </div>
            </div>
        `;

        await this.loadPairwiseScatter();
    },

    /**
     * Load pairwise scatter plot
     */
    async loadPairwiseScatter() {
        const pair = this.getAtlasPairs().find(p => p.key === this.tabs.atlasComparison.pair);
        if (!pair) return;

        const data = await API.getPairwiseScatter({
            atlas1: pair.atlas1,
            atlas2: pair.atlas2,
            signature_type: this.signatureType,
            level: this.tabs.atlasComparison.level,
        });

        if (data && data.data && data.data.length > 0) {
            // Main scatter plot
            const scatterData = data.data;
            const cellTypes = [...new Set(scatterData.map(d => d.cell_type))];
            const colors = this.generateColors(cellTypes.length);

            const traces = cellTypes.map((ct, i) => {
                const ctData = scatterData.filter(d => d.cell_type === ct);
                return {
                    type: 'scatter',
                    mode: 'markers',
                    name: ct,
                    x: ctData.map(d => d.x),
                    y: ctData.map(d => d.y),
                    text: ctData.map(d => `${d.signature}<br>${d.cell_type}`),
                    marker: {
                        color: colors[i],
                        size: 8,
                        opacity: 0.7,
                    },
                    hovertemplate: '%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                };
            });

            // Add identity line
            const allX = scatterData.map(d => d.x);
            const allY = scatterData.map(d => d.y);
            const minVal = Math.min(...allX, ...allY);
            const maxVal = Math.max(...allX, ...allY);

            traces.push({
                type: 'scatter',
                mode: 'lines',
                x: [minVal, maxVal],
                y: [minVal, maxVal],
                line: { color: '#94a3b8', dash: 'dash', width: 1 },
                showlegend: false,
                hoverinfo: 'skip',
            });

            Plotly.newPlot('scatter-plot', traces, {
                xaxis: { title: `${pair.atlas1} Activity (z-score)`, zeroline: true },
                yaxis: { title: `${pair.atlas2} Activity (z-score)`, zeroline: true },
                legend: { orientation: 'v', x: 1.02, y: 1 },
                margin: { l: 60, r: 120, t: 20, b: 60 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                hovermode: 'closest',
            }, { responsive: true });

            // Per-cell-type correlation bar chart
            const ctCorrelations = this.calculatePerCelltypeCorrelations(scatterData, cellTypes);

            Plotly.newPlot('celltype-correlations', [{
                type: 'bar',
                y: ctCorrelations.map(c => c.cellType),
                x: ctCorrelations.map(c => c.r),
                orientation: 'h',
                marker: {
                    color: ctCorrelations.map(c =>
                        c.r > 0.8 ? '#10b981' :
                        c.r > 0.5 ? '#3b82f6' :
                        c.r > 0.3 ? '#f59e0b' : '#ef4444'
                    ),
                },
                text: ctCorrelations.map(c => c.r.toFixed(2)),
                textposition: 'outside',
            }], {
                xaxis: { title: 'Correlation (r)', range: [0, 1.1] },
                yaxis: { automargin: true },
                margin: { l: 100, r: 40, t: 20, b: 60 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
            }, { responsive: true });

            // Update stats
            document.getElementById('correlation-stats').innerHTML = `
                <div class="stats-row">
                    <div class="stat-item">
                        <span class="stat-label">Overall Correlation:</span>
                        <span class="stat-value">${data.correlation?.toFixed(3) || 'N/A'}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">P-value:</span>
                        <span class="stat-value">${data.pvalue ? data.pvalue.toExponential(2) : 'N/A'}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Data Points:</span>
                        <span class="stat-value">${data.n || scatterData.length}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Cell Types:</span>
                        <span class="stat-value">${data.n_celltypes || cellTypes.length}</span>
                    </div>
                </div>
            `;
        } else {
            document.getElementById('scatter-plot').innerHTML =
                '<p class="loading">No comparison data available for this pair</p>';
        }
    },

    /**
     * Calculate per-cell-type correlations
     */
    calculatePerCelltypeCorrelations(data, cellTypes) {
        return cellTypes.map(ct => {
            const ctData = data.filter(d => d.cell_type === ct);
            const x = ctData.map(d => d.x);
            const y = ctData.map(d => d.y);
            const r = this.pearsonCorrelation(x, y);
            return { cellType: ct, r: r || 0, n: ctData.length };
        }).sort((a, b) => b.r - a.r);
    },

    /**
     * Simple Pearson correlation calculation
     */
    pearsonCorrelation(x, y) {
        if (x.length < 2 || x.length !== y.length) return 0;
        const n = x.length;
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((total, xi, i) => total + xi * y[i], 0);
        const sumX2 = x.reduce((a, b) => a + b * b, 0);
        const sumY2 = y.reduce((a, b) => a + b * b, 0);
        const num = n * sumXY - sumX * sumY;
        const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        return den === 0 ? 0 : num / den;
    },

    /**
     * Change selected pair
     */
    changePair(pairKey) {
        this.tabs.atlasComparison.pair = pairKey;
        this.loadPairwiseScatter();
    },

    /**
     * Change comparison level
     */
    changeComparisonLevel(level) {
        this.tabs.atlasComparison.level = level;
        this.loadPairwiseScatter();
    },

    /**
     * Tab 4: Conserved Signatures - Reliability heatmap
     */
    async loadConservedSignatures() {
        const content = document.getElementById('compare-content');

        content.innerHTML = `
            <div class="conserved-tab">
                <div class="tab-controls">
                    <div class="control-group">
                        <label>Filter</label>
                        <select id="conserved-filter" class="filter-select" onchange="ComparePage.changeConservedFilter(this.value)">
                            <option value="all" ${this.tabs.conserved.filter === 'all' ? 'selected' : ''}>All Signatures</option>
                            <option value="highly_conserved" ${this.tabs.conserved.filter === 'highly_conserved' ? 'selected' : ''}>Highly Conserved (r > 0.7)</option>
                            <option value="moderately_conserved" ${this.tabs.conserved.filter === 'moderately_conserved' ? 'selected' : ''}>Moderately Conserved (r > 0.5)</option>
                            <option value="atlas_specific" ${this.tabs.conserved.filter === 'atlas_specific' ? 'selected' : ''}>Atlas-Specific (r < 0.5)</option>
                        </select>
                    </div>
                </div>

                <div class="conserved-summary" id="conserved-summary">
                    <!-- Summary cards will be loaded -->
                </div>

                <div class="conserved-grid">
                    <div class="conserved-panel main-panel">
                        <h3>Signature Reliability Heatmap</h3>
                        <p class="viz-subtitle">Per-signature correlations across atlas pairs</p>
                        <div id="reliability-heatmap" class="plot-container plot-large"></div>
                    </div>
                </div>

                <div class="conserved-table-container">
                    <h3>Signature Details</h3>
                    <input type="text" id="signature-search" class="search-input"
                           placeholder="Search signatures..."
                           oninput="ComparePage.filterSignatureTable(this.value)">
                    <div class="table-wrapper">
                        <table class="data-table" id="signature-table">
                            <thead>
                                <tr>
                                    <th>Signature</th>
                                    <th>Category</th>
                                    <th>Mean r</th>
                                    <th>CIMA-Inflam</th>
                                    <th>CIMA-scAtlas</th>
                                    <th>Inflam-scAtlas</th>
                                </tr>
                            </thead>
                            <tbody id="signature-table-body">
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;

        await this.loadSignatureReliability();
    },

    /**
     * Load signature reliability data
     */
    async loadSignatureReliability() {
        const data = await API.getSignatureReliability({
            signature_type: this.signatureType,
        });

        if (!data || !data.signatures) {
            document.getElementById('reliability-heatmap').innerHTML =
                '<p class="loading">No reliability data available</p>';
            return;
        }

        // Update summary
        const summary = data.summary || {};
        document.getElementById('conserved-summary').innerHTML = `
            <div class="summary-cards">
                <div class="summary-card highlight-green">
                    <div class="summary-value">${summary.highly_conserved || 0}</div>
                    <div class="summary-label">Highly Conserved</div>
                </div>
                <div class="summary-card highlight-blue">
                    <div class="summary-value">${summary.moderately_conserved || 0}</div>
                    <div class="summary-label">Moderately Conserved</div>
                </div>
                <div class="summary-card highlight-yellow">
                    <div class="summary-value">${summary.atlas_specific || 0}</div>
                    <div class="summary-label">Atlas-Specific</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value">${summary.total || data.signatures.length}</div>
                    <div class="summary-label">Total Signatures</div>
                </div>
            </div>
        `;

        // Filter signatures
        let signatures = data.signatures;
        if (this.tabs.conserved.filter !== 'all') {
            signatures = signatures.filter(s => s.category === this.tabs.conserved.filter);
        }

        // Store for table filtering
        this.currentSignatures = signatures;

        // Build heatmap
        const pairs = data.pairs || ['cima_vs_inflammation', 'cima_vs_scatlas', 'inflammation_vs_scatlas'];
        const pairLabels = data.pair_labels || ['CIMA vs Inflam', 'CIMA vs scAtlas', 'Inflam vs scAtlas'];

        const z = signatures.map(s =>
            pairs.map(p => s.correlations[p]?.r || null)
        );

        const hovertext = signatures.map(s =>
            pairs.map(p => {
                const c = s.correlations[p];
                return c ? `${s.signature}<br>r = ${c.r?.toFixed(3)}<br>p = ${c.p?.toExponential(2)}<br>n = ${c.n}` : 'No data';
            })
        );

        Plotly.newPlot('reliability-heatmap', [{
            type: 'heatmap',
            z: z,
            x: pairLabels,
            y: signatures.map(s => s.signature),
            colorscale: [
                [0, '#ef4444'],
                [0.5, '#fbbf24'],
                [0.7, '#a3e635'],
                [1, '#10b981'],
            ],
            zmin: 0,
            zmax: 1,
            text: hovertext,
            hovertemplate: '%{text}<extra></extra>',
            colorbar: {
                title: 'Correlation (r)',
                titleside: 'right',
            },
        }], {
            margin: { l: 100, r: 80, t: 20, b: 80 },
            xaxis: { side: 'bottom' },
            yaxis: { automargin: true, tickfont: { size: 10 } },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
        }, { responsive: true });

        // Populate table
        this.populateSignatureTable(signatures);
    },

    /**
     * Populate signature table
     */
    populateSignatureTable(signatures) {
        const tbody = document.getElementById('signature-table-body');
        if (!tbody) return;

        tbody.innerHTML = signatures.map(s => {
            const getCellClass = (r) => {
                if (r === null || r === undefined) return '';
                if (r > 0.7) return 'cell-green';
                if (r > 0.5) return 'cell-blue';
                if (r > 0.3) return 'cell-yellow';
                return 'cell-red';
            };

            const formatR = (r) => r !== null && r !== undefined ? r.toFixed(3) : '-';

            const c = s.correlations;
            return `
                <tr>
                    <td><strong>${s.signature}</strong></td>
                    <td><span class="category-badge ${s.category}">${this.formatCategory(s.category)}</span></td>
                    <td>${s.mean_correlation?.toFixed(3) || '-'}</td>
                    <td class="${getCellClass(c.cima_vs_inflammation?.r)}">${formatR(c.cima_vs_inflammation?.r)}</td>
                    <td class="${getCellClass(c.cima_vs_scatlas?.r)}">${formatR(c.cima_vs_scatlas?.r)}</td>
                    <td class="${getCellClass(c.inflammation_vs_scatlas?.r)}">${formatR(c.inflammation_vs_scatlas?.r)}</td>
                </tr>
            `;
        }).join('');
    },

    /**
     * Filter signature table by search term
     */
    filterSignatureTable(term) {
        if (!this.currentSignatures) return;
        const filtered = this.currentSignatures.filter(s =>
            s.signature.toLowerCase().includes(term.toLowerCase())
        );
        this.populateSignatureTable(filtered);
    },

    /**
     * Change conserved filter
     */
    changeConservedFilter(filter) {
        this.tabs.conserved.filter = filter;
        this.loadSignatureReliability();
    },

    /**
     * Tab 5: Meta-Analysis - Forest plots
     */
    async loadMetaAnalysis() {
        const content = document.getElementById('compare-content');

        content.innerHTML = `
            <div class="meta-analysis-tab">
                <div class="tab-controls">
                    <div class="control-group">
                        <label>Analysis Type</label>
                        <select id="analysis-select" class="filter-select" onchange="ComparePage.changeAnalysis(this.value)">
                            <option value="age" ${this.tabs.metaAnalysis.analysis === 'age' ? 'selected' : ''}>Age Correlation</option>
                            <option value="bmi" ${this.tabs.metaAnalysis.analysis === 'bmi' ? 'selected' : ''}>BMI Correlation</option>
                            <option value="sex" ${this.tabs.metaAnalysis.analysis === 'sex' ? 'selected' : ''}>Sex Difference</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>Signature (optional)</label>
                        <input type="text" id="meta-signature-filter" class="filter-input"
                               placeholder="Filter by signature..."
                               value="${this.tabs.metaAnalysis.signature || ''}"
                               onchange="ComparePage.changeMetaSignature(this.value)">
                    </div>
                </div>

                <div class="meta-summary" id="meta-summary">
                    <!-- Summary will be loaded -->
                </div>

                <div class="meta-grid">
                    <div class="meta-panel main-panel">
                        <h3>Forest Plot</h3>
                        <p class="viz-subtitle">Individual atlas effects and pooled estimates</p>
                        <div id="forest-plot" class="plot-container plot-large"></div>
                    </div>
                    <div class="meta-panel side-panel">
                        <h3>Heterogeneity (I²)</h3>
                        <p class="viz-subtitle">Variation between atlas estimates</p>
                        <div id="heterogeneity-chart" class="plot-container"></div>
                    </div>
                </div>
            </div>
        `;

        await this.loadForestPlot();
    },

    /**
     * Load forest plot data
     */
    async loadForestPlot() {
        const data = await API.getMetaAnalysisForest({
            analysis: this.tabs.metaAnalysis.analysis,
            signature_type: this.signatureType,
            signature: this.tabs.metaAnalysis.signature || undefined,
        });

        if (!data || !data.forest_data || data.forest_data.length === 0) {
            document.getElementById('forest-plot').innerHTML =
                '<p class="loading">No meta-analysis data available</p>';
            return;
        }

        // Update summary
        const summary = data.summary || {};
        document.getElementById('meta-summary').innerHTML = `
            <div class="summary-cards">
                <div class="summary-card">
                    <div class="summary-value">${summary.n_signatures || 0}</div>
                    <div class="summary-label">Signatures Analyzed</div>
                </div>
                <div class="summary-card highlight-green">
                    <div class="summary-value">${summary.n_significant || 0}</div>
                    <div class="summary-label">Significant Effects</div>
                </div>
                <div class="summary-card highlight-blue">
                    <div class="summary-value">${summary.n_consistent_direction || 0}</div>
                    <div class="summary-label">Consistent Direction</div>
                </div>
                <div class="summary-card highlight-yellow">
                    <div class="summary-value">${summary.n_heterogeneous || 0}</div>
                    <div class="summary-label">High Heterogeneity</div>
                </div>
            </div>
        `;

        // Create forest plot
        ForestPlot.create('forest-plot', data.forest_data, {
            title: '',
            xLabel: this.tabs.metaAnalysis.analysis === 'sex' ? 'Effect Size (M-F)' : 'Effect Size (Correlation)',
            maxSignatures: 25,
        });

        // Create heterogeneity chart
        ForestPlot.createHeterogeneityChart('heterogeneity-chart', data.forest_data, {
            title: '',
            maxSignatures: 20,
        });
    },

    /**
     * Change analysis type
     */
    changeAnalysis(analysis) {
        this.tabs.metaAnalysis.analysis = analysis;
        this.loadForestPlot();
    },

    /**
     * Change meta signature filter
     */
    changeMetaSignature(signature) {
        this.tabs.metaAnalysis.signature = signature || null;
        this.loadForestPlot();
    },

    /**
     * Change signature type
     */
    changeSignatureType(type) {
        this.signatureType = type;
        this.loadActiveTab();
    },

    // ==================== Utility Methods ====================

    /**
     * Format number with commas
     */
    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        }
        if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num?.toLocaleString() || '0';
    },

    /**
     * Format atlas name
     */
    formatAtlasName(key) {
        const names = {
            cima: 'CIMA',
            inflammation: 'Inflammation',
            scatlas_normal: 'scAtlas (Normal)',
            scatlas_cancer: 'scAtlas (Cancer)',
        };
        return names[key] || key;
    },

    /**
     * Format category name
     */
    formatCategory(cat) {
        const labels = {
            highly_conserved: 'Highly Conserved',
            moderately_conserved: 'Moderately Conserved',
            atlas_specific: 'Atlas-Specific',
            insufficient_data: 'Insufficient Data',
        };
        return labels[cat] || cat;
    },

    /**
     * Generate color palette
     */
    generateColors(n) {
        const palette = [
            '#3b82f6', '#f59e0b', '#10b981', '#8b5cf6', '#ef4444',
            '#06b6d4', '#f97316', '#84cc16', '#ec4899', '#6366f1',
            '#14b8a6', '#eab308', '#22c55e', '#a855f7', '#f43f5e',
        ];
        const colors = [];
        for (let i = 0; i < n; i++) {
            colors.push(palette[i % palette.length]);
        }
        return colors;
    },
};

// Make available globally
window.ComparePage = ComparePage;
