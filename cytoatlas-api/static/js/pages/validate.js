/**
 * Validation Page Handler
 * 4-tab validation dashboard matching visualization/index.html
 */

const ValidatePage = {
    // State
    currentAtlas: 'cima',
    signatureType: 'CytoSig',
    activeTab: 'atlas-level',

    // Tab-specific state
    atlasLevel: {
        signature: 'IFNG',
        signatures: []
    },
    pseudobulk: {
        signature: 'all',
        celltype: 'all',
        signatures: [],
        celltypes: []
    },
    singlecell: {
        signature: 'IFNG',
        signatures: []
    },

    /**
     * Initialize the validation page
     */
    async init(params, query) {
        // Get params from query
        if (query.atlas) this.currentAtlas = query.atlas;
        if (query.type) this.signatureType = query.type;
        if (query.tab) this.activeTab = query.tab;

        // Render template
        this.render();

        // Set up event listeners
        this.setupEventListeners();

        // Load initial data
        await this.loadSignatures();
        await this.loadTab(this.activeTab);
    },

    /**
     * Render the page template
     */
    render() {
        const app = document.getElementById('app');
        const template = document.getElementById('validate-template');

        if (app && template) {
            app.innerHTML = template.innerHTML;
        }

        // Set control values
        const atlasSelect = document.getElementById('val-atlas');
        const sigtypeSelect = document.getElementById('val-sigtype');

        if (atlasSelect) atlasSelect.value = this.currentAtlas;
        if (sigtypeSelect) sigtypeSelect.value = this.signatureType;
    },

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Atlas selector
        const atlasSelect = document.getElementById('val-atlas');
        if (atlasSelect) {
            atlasSelect.addEventListener('change', async (e) => {
                this.currentAtlas = e.target.value;
                await this.loadSignatures();
                await this.loadTab(this.activeTab);
            });
        }

        // Signature type selector
        const sigtypeSelect = document.getElementById('val-sigtype');
        if (sigtypeSelect) {
            sigtypeSelect.addEventListener('change', async (e) => {
                this.signatureType = e.target.value;
                await this.loadSignatures();
                await this.loadTab(this.activeTab);
            });
        }

        // Tab buttons
        const tabBtns = document.querySelectorAll('#validation-tabs .tab-btn');
        tabBtns.forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const tab = e.target.dataset.tab;
                this.switchTab(tab);
            });
        });
    },

    /**
     * Switch active tab
     */
    async switchTab(tab) {
        this.activeTab = tab;

        // Update tab button styles
        const tabBtns = document.querySelectorAll('#validation-tabs .tab-btn');
        tabBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tab);
        });

        // Load tab content
        await this.loadTab(tab);
    },

    /**
     * Load signatures for current atlas/type
     */
    async loadSignatures() {
        try {
            const signatures = await API.getValidationSignatures(this.currentAtlas, this.signatureType);
            this.atlasLevel.signatures = signatures;
            this.pseudobulk.signatures = signatures;
            this.singlecell.signatures = signatures;

            // Set default signature
            if (signatures.length > 0) {
                if (!signatures.includes(this.atlasLevel.signature)) {
                    this.atlasLevel.signature = signatures[0];
                }
                if (!signatures.includes(this.singlecell.signature)) {
                    this.singlecell.signature = signatures[0];
                }
            }
        } catch (error) {
            console.error('Failed to load signatures:', error);
            // Use fallback
            const fallback = this.signatureType === 'CytoSig'
                ? ['IFNG', 'TNF', 'IL6', 'IL10', 'IL17A', 'TGFB1']
                : ['IFNG', 'TNF', 'IL6'];
            this.atlasLevel.signatures = fallback;
            this.pseudobulk.signatures = fallback;
            this.singlecell.signatures = fallback;
        }
    },

    /**
     * Load tab content
     */
    async loadTab(tab) {
        const content = document.getElementById('validation-content');
        if (!content) return;

        content.innerHTML = '<div class="loading"><div class="spinner"></div>Loading...</div>';

        switch (tab) {
            case 'atlas-level':
                await this.loadAtlasLevel();
                break;
            case 'pseudobulk':
                await this.loadPseudobulkLevel();
                break;
            case 'singlecell':
                await this.loadSingleCellLevel();
                break;
            case 'summary':
                await this.loadSummary();
                break;
        }
    },

    // ==================== Atlas Level Tab ====================

    async loadAtlasLevel() {
        const content = document.getElementById('validation-content');

        content.innerHTML = `
            <div class="tab-panel">
                <h3>Atlas Level Validation</h3>
                <p>Cell type-aggregated correlation: one data point per cell type. Compares mean expression of signature genes with mean activity across all cells of each cell type.</p>

                <div class="filter-bar">
                    <label>Signature:
                        <select id="val-atlas-signature">
                            ${this.atlasLevel.signatures.map(s =>
                                `<option value="${s}" ${s === this.atlasLevel.signature ? 'selected' : ''}>${s}</option>`
                            ).join('')}
                        </select>
                    </label>
                    <label>
                        <input type="text" id="val-atlas-search" placeholder="Search signature...">
                    </label>
                </div>

                <div class="panel-grid">
                    <div class="panel">
                        <div class="viz-title">Expression vs Activity Correlation</div>
                        <div class="viz-subtitle">Each point = cell type mean</div>
                        <div id="val-atlas-scatter" class="plot-container" style="height: 400px;"></div>
                    </div>
                    <div class="panel">
                        <div class="viz-title">Signature Correlation Ranking</div>
                        <div class="viz-subtitle">Signatures ranked by cell type correlation</div>
                        <div id="val-atlas-ranking" class="plot-container" style="height: 400px;"></div>
                    </div>
                </div>

                <div class="panel-grid">
                    <div class="panel full-width">
                        <div class="viz-title">Cross-Atlas Cell Type Correlation</div>
                        <div class="viz-subtitle">Correlation by cell type across all three atlases</div>
                        <div id="val-atlas-heatmap" class="plot-container" style="height: 350px;"></div>
                    </div>
                </div>
            </div>
        `;

        // Set up signature selector
        const sigSelect = document.getElementById('val-atlas-signature');
        if (sigSelect) {
            sigSelect.addEventListener('change', async (e) => {
                this.atlasLevel.signature = e.target.value;
                await this.updateAtlasLevelScatter();
            });
        }

        // Set up search
        const searchInput = document.getElementById('val-atlas-search');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                const query = e.target.value.toLowerCase();
                const options = sigSelect.options;
                for (let i = 0; i < options.length; i++) {
                    const match = options[i].value.toLowerCase().includes(query);
                    options[i].style.display = match ? '' : 'none';
                }
            });
        }

        // Load visualizations
        await Promise.all([
            this.updateAtlasLevelScatter(),
            this.loadAtlasLevelRanking(),
            this.loadAtlasLevelHeatmap()
        ]);
    },

    async updateAtlasLevelScatter() {
        const container = document.getElementById('val-atlas-scatter');
        if (!container) return;

        try {
            const data = await API.getCellTypeLevelValidation(
                this.currentAtlas,
                this.atlasLevel.signature,
                this.signatureType
            );

            if (data && data.points) {
                const trace = {
                    x: data.points.map(p => p.expression),
                    y: data.points.map(p => p.activity),
                    mode: 'markers+text',
                    type: 'scatter',
                    text: data.points.map(p => p.cell_type),
                    textposition: 'top center',
                    textfont: { size: 9 },
                    marker: {
                        size: 10,
                        color: '#3b82f6',
                        opacity: 0.7
                    },
                    hovertemplate: '<b>%{text}</b><br>Expression: %{x:.3f}<br>Activity: %{y:.3f}<extra></extra>'
                };

                // Add trend line
                const trendline = this.calculateTrendline(
                    data.points.map(p => p.expression),
                    data.points.map(p => p.activity)
                );

                const layout = {
                    xaxis: { title: 'Mean Expression (log1p)' },
                    yaxis: { title: 'Mean Activity (z-score)' },
                    margin: { l: 60, r: 20, t: 40, b: 60 },
                    annotations: [{
                        x: 0.02,
                        y: 0.98,
                        xref: 'paper',
                        yref: 'paper',
                        text: `r = ${data.stats?.pearson_r?.toFixed(3) || 'N/A'}`,
                        showarrow: false,
                        font: { size: 14 }
                    }]
                };

                Plotly.newPlot(container, [trace, trendline], layout, {responsive: true});
            } else {
                container.innerHTML = '<p class="no-data">No data available</p>';
            }
        } catch (error) {
            container.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        }
    },

    async loadAtlasLevelRanking() {
        const container = document.getElementById('val-atlas-ranking');
        if (!container) return;

        try {
            // Get correlations for all signatures
            const rankings = [];
            for (const sig of this.atlasLevel.signatures.slice(0, 50)) { // Limit for performance
                try {
                    const data = await API.getCellTypeLevelValidation(
                        this.currentAtlas, sig, this.signatureType
                    );
                    if (data?.stats?.pearson_r !== undefined) {
                        rankings.push({
                            signature: sig,
                            r: data.stats.pearson_r
                        });
                    }
                } catch (e) {
                    // Skip signatures without data
                }
            }

            if (rankings.length > 0) {
                // Sort by correlation
                rankings.sort((a, b) => b.r - a.r);

                const trace = {
                    x: rankings.map(r => r.r),
                    y: rankings.map(r => r.signature),
                    type: 'bar',
                    orientation: 'h',
                    marker: {
                        color: rankings.map(r => r.r > 0 ? '#22c55e' : '#ef4444')
                    }
                };

                const layout = {
                    xaxis: { title: 'Pearson r', range: [-1, 1] },
                    yaxis: { automargin: true },
                    margin: { l: 100, r: 20, t: 20, b: 60 },
                    height: 400
                };

                Plotly.newPlot(container, [trace], layout, {responsive: true});
            } else {
                container.innerHTML = '<p class="no-data">No ranking data available</p>';
            }
        } catch (error) {
            container.innerHTML = `<p class="error">Error loading ranking</p>`;
        }
    },

    async loadAtlasLevelHeatmap() {
        const container = document.getElementById('val-atlas-heatmap');
        if (!container) return;

        try {
            // Get data for all atlases
            const atlases = ['cima', 'inflammation', 'scatlas'];
            const celltypes = new Set();
            const atlasData = {};

            for (const atlas of atlases) {
                try {
                    const data = await API.getCellTypeLevelValidation(
                        atlas, this.atlasLevel.signature, this.signatureType
                    );
                    if (data?.points) {
                        atlasData[atlas] = {};
                        data.points.forEach(p => {
                            celltypes.add(p.cell_type);
                            atlasData[atlas][p.cell_type] = p.activity;
                        });
                    }
                } catch (e) {
                    atlasData[atlas] = {};
                }
            }

            const celltypeList = Array.from(celltypes).sort();

            if (celltypeList.length > 0) {
                const traces = atlases.map((atlas, i) => ({
                    x: celltypeList,
                    y: celltypeList.map(ct => atlasData[atlas]?.[ct] || null),
                    name: atlas.charAt(0).toUpperCase() + atlas.slice(1),
                    type: 'bar'
                }));

                const layout = {
                    barmode: 'group',
                    xaxis: { title: 'Cell Type', tickangle: 45 },
                    yaxis: { title: 'Mean Activity' },
                    margin: { l: 60, r: 20, t: 20, b: 120 },
                    legend: { orientation: 'h', y: 1.1 }
                };

                Plotly.newPlot(container, traces, layout, {responsive: true});
            } else {
                container.innerHTML = '<p class="no-data">No cross-atlas data available</p>';
            }
        } catch (error) {
            container.innerHTML = `<p class="error">Error loading heatmap</p>`;
        }
    },

    // ==================== Pseudobulk Level Tab ====================

    async loadPseudobulkLevel() {
        const content = document.getElementById('validation-content');

        content.innerHTML = `
            <div class="tab-panel">
                <h3>Pseudobulk Level Validation</h3>
                <p>Correlation between mean signature gene expression and predicted activity per sample. Pseudobulk samples are created by aggregating cells within each sample × cell type combination.</p>

                <div class="filter-bar">
                    <label>Signature:
                        <select id="val-pb-signature">
                            <option value="all">All Signatures</option>
                            ${this.pseudobulk.signatures.map(s =>
                                `<option value="${s}" ${s === this.pseudobulk.signature ? 'selected' : ''}>${s}</option>`
                            ).join('')}
                        </select>
                    </label>
                    <label>
                        <input type="text" id="val-pb-search" placeholder="Search signature...">
                    </label>
                </div>

                <div class="panel-grid">
                    <div class="panel">
                        <div class="viz-title">Expression vs Activity Correlation</div>
                        <div class="viz-subtitle">Each point represents a pseudobulk sample</div>
                        <div id="val-pb-scatter" class="plot-container" style="height: 400px;"></div>
                    </div>
                    <div class="panel">
                        <div class="viz-title">Correlation Distribution</div>
                        <div class="viz-subtitle">Distribution of Pearson r across all signatures</div>
                        <div id="val-pb-distribution" class="plot-container" style="height: 400px;"></div>
                    </div>
                </div>

                <div class="panel-grid">
                    <div class="panel full-width">
                        <div class="viz-title">Per-Signature Correlations (Sorted)</div>
                        <div class="viz-subtitle">Pearson r between expression and activity, sorted from highest to lowest</div>
                        <div id="val-pb-heatmap" class="plot-container" style="height: 300px;"></div>
                    </div>
                </div>
            </div>
        `;

        // Set up signature selector
        const sigSelect = document.getElementById('val-pb-signature');
        if (sigSelect) {
            sigSelect.addEventListener('change', async (e) => {
                this.pseudobulk.signature = e.target.value;
                await this.updatePseudobulkScatter();
            });
        }

        // Load visualizations
        await Promise.all([
            this.updatePseudobulkScatter(),
            this.loadPseudobulkDistribution(),
            this.loadPseudobulkHeatmap()
        ]);
    },

    async updatePseudobulkScatter() {
        const container = document.getElementById('val-pb-scatter');
        if (!container) return;

        const sig = this.pseudobulk.signature === 'all'
            ? this.pseudobulk.signatures[0]
            : this.pseudobulk.signature;

        try {
            const data = await API.getSampleLevelValidation(
                this.currentAtlas, sig, this.signatureType
            );

            if (data && data.points) {
                const trace = {
                    x: data.points.map(p => p.expression),
                    y: data.points.map(p => p.activity),
                    mode: 'markers',
                    type: 'scatter',
                    marker: {
                        size: 6,
                        color: '#3b82f6',
                        opacity: 0.5
                    },
                    hovertemplate: 'Expression: %{x:.3f}<br>Activity: %{y:.3f}<extra></extra>'
                };

                const trendline = this.calculateTrendline(
                    data.points.map(p => p.expression),
                    data.points.map(p => p.activity)
                );

                const layout = {
                    xaxis: { title: 'Mean Expression (log1p)' },
                    yaxis: { title: 'Activity (z-score)' },
                    margin: { l: 60, r: 20, t: 40, b: 60 },
                    annotations: [{
                        x: 0.02,
                        y: 0.98,
                        xref: 'paper',
                        yref: 'paper',
                        text: `r = ${data.stats?.pearson_r?.toFixed(3) || 'N/A'}<br>n = ${data.points.length}`,
                        showarrow: false,
                        font: { size: 12 }
                    }]
                };

                Plotly.newPlot(container, [trace, trendline], layout, {responsive: true});
            } else {
                container.innerHTML = '<p class="no-data">No data available</p>';
            }
        } catch (error) {
            container.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        }
    },

    async loadPseudobulkDistribution() {
        const container = document.getElementById('val-pb-distribution');
        if (!container) return;

        try {
            // Get correlations for all signatures
            const correlations = [];
            for (const sig of this.pseudobulk.signatures.slice(0, 50)) {
                try {
                    const data = await API.getSampleLevelValidation(
                        this.currentAtlas, sig, this.signatureType
                    );
                    if (data?.stats?.pearson_r !== undefined) {
                        correlations.push(data.stats.pearson_r);
                    }
                } catch (e) {
                    // Skip
                }
            }

            if (correlations.length > 0) {
                const trace = {
                    x: correlations,
                    type: 'histogram',
                    nbinsx: 20,
                    marker: { color: '#3b82f6' }
                };

                const layout = {
                    xaxis: { title: 'Pearson r', range: [-1, 1] },
                    yaxis: { title: 'Count' },
                    margin: { l: 60, r: 20, t: 20, b: 60 },
                    shapes: [{
                        type: 'line',
                        x0: 0, x1: 0,
                        y0: 0, y1: 1,
                        yref: 'paper',
                        line: { color: 'red', dash: 'dash' }
                    }]
                };

                Plotly.newPlot(container, [trace], layout, {responsive: true});
            } else {
                container.innerHTML = '<p class="no-data">No distribution data available</p>';
            }
        } catch (error) {
            container.innerHTML = `<p class="error">Error loading distribution</p>`;
        }
    },

    async loadPseudobulkHeatmap() {
        const container = document.getElementById('val-pb-heatmap');
        if (!container) return;

        try {
            // Get correlations for all signatures
            const rankings = [];
            for (const sig of this.pseudobulk.signatures.slice(0, 50)) {
                try {
                    const data = await API.getSampleLevelValidation(
                        this.currentAtlas, sig, this.signatureType
                    );
                    if (data?.stats?.pearson_r !== undefined) {
                        rankings.push({
                            signature: sig,
                            r: data.stats.pearson_r
                        });
                    }
                } catch (e) {
                    // Skip
                }
            }

            if (rankings.length > 0) {
                rankings.sort((a, b) => b.r - a.r);

                const trace = {
                    x: rankings.map(r => r.signature),
                    y: rankings.map(r => r.r),
                    type: 'bar',
                    marker: {
                        color: rankings.map(r => r.r > 0 ? '#22c55e' : '#ef4444')
                    }
                };

                const layout = {
                    xaxis: { title: 'Signature', tickangle: 45 },
                    yaxis: { title: 'Pearson r', range: [-1, 1] },
                    margin: { l: 60, r: 20, t: 20, b: 100 }
                };

                Plotly.newPlot(container, [trace], layout, {responsive: true});
            } else {
                container.innerHTML = '<p class="no-data">No data available</p>';
            }
        } catch (error) {
            container.innerHTML = `<p class="error">Error loading heatmap</p>`;
        }
    },

    // ==================== Single-Cell Level Tab ====================

    async loadSingleCellLevel() {
        const content = document.getElementById('validation-content');

        content.innerHTML = `
            <div class="tab-panel">
                <h3>Single-Cell Level Validation</h3>
                <p>Comparison of activity between cells that express the signature gene vs those that don't. If inference is valid, expressing cells should have higher activity.</p>

                <div class="filter-bar">
                    <label>Signature/Gene:
                        <select id="val-sc-signature">
                            ${this.singlecell.signatures.map(s =>
                                `<option value="${s}" ${s === this.singlecell.signature ? 'selected' : ''}>${s}</option>`
                            ).join('')}
                        </select>
                    </label>
                    <label>
                        <input type="text" id="val-sc-search" placeholder="Search gene...">
                    </label>
                </div>

                <div class="panel-grid">
                    <div class="panel">
                        <div class="viz-title">Activity: Expressing vs Non-Expressing Cells</div>
                        <div class="viz-subtitle">Box plot comparing activity distributions</div>
                        <div id="val-sc-boxplot" class="plot-container" style="height: 400px;"></div>
                    </div>
                    <div class="panel">
                        <div class="viz-title">Validation by Cell Type</div>
                        <div class="viz-subtitle">Fold change (expressing/non-expressing) per cell type</div>
                        <div id="val-sc-celltype" class="plot-container" style="height: 400px;"></div>
                    </div>
                </div>

                <div class="panel-grid">
                    <div class="panel full-width">
                        <div class="viz-title">Validation Summary</div>
                        <div class="viz-subtitle">Key metrics for single-cell validation</div>
                        <div id="val-sc-summary" class="summary-cards"></div>
                    </div>
                </div>
            </div>
        `;

        // Set up signature selector
        const sigSelect = document.getElementById('val-sc-signature');
        if (sigSelect) {
            sigSelect.addEventListener('change', async (e) => {
                this.singlecell.signature = e.target.value;
                await this.updateSingleCellPlots();
            });
        }

        // Load visualizations
        await this.updateSingleCellPlots();
    },

    async updateSingleCellPlots() {
        await Promise.all([
            this.loadSingleCellBoxplot(),
            this.loadSingleCellByType(),
            this.loadSingleCellSummary()
        ]);
    },

    async loadSingleCellBoxplot() {
        const container = document.getElementById('val-sc-boxplot');
        if (!container) return;

        try {
            const data = await API.getSingleCellDistribution(
                this.currentAtlas,
                this.singlecell.signature,
                this.signatureType
            );

            if (data) {
                const traces = [
                    {
                        y: data.expressing_activities || [],
                        type: 'box',
                        name: 'Expressing',
                        marker: { color: '#22c55e' }
                    },
                    {
                        y: data.non_expressing_activities || [],
                        type: 'box',
                        name: 'Non-Expressing',
                        marker: { color: '#94a3b8' }
                    }
                ];

                const layout = {
                    yaxis: { title: 'Activity (z-score)' },
                    margin: { l: 60, r: 20, t: 20, b: 60 },
                    showlegend: false
                };

                Plotly.newPlot(container, traces, layout, {responsive: true});
            } else {
                container.innerHTML = '<p class="no-data">No data available</p>';
            }
        } catch (error) {
            container.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        }
    },

    async loadSingleCellByType() {
        const container = document.getElementById('val-sc-celltype');
        if (!container) return;

        try {
            const data = await API.getSingleCellDirect(
                this.currentAtlas,
                this.singlecell.signature,
                this.signatureType
            );

            if (data) {
                // Show fold change as summary
                const trace = {
                    x: ['Overall'],
                    y: [data.fold_change || 1],
                    type: 'bar',
                    marker: {
                        color: data.fold_change > 1 ? '#22c55e' : '#ef4444'
                    },
                    text: [`FC: ${data.fold_change?.toFixed(2)}x`],
                    textposition: 'outside'
                };

                const layout = {
                    yaxis: { title: 'Fold Change (Expr / Non-Expr)' },
                    margin: { l: 60, r: 20, t: 40, b: 60 },
                    shapes: [{
                        type: 'line',
                        x0: -0.5, x1: 0.5,
                        y0: 1, y1: 1,
                        line: { color: 'red', dash: 'dash' }
                    }]
                };

                Plotly.newPlot(container, [trace], layout, {responsive: true});
            } else {
                container.innerHTML = '<p class="no-data">No data available</p>';
            }
        } catch (error) {
            container.innerHTML = `<p class="error">Error loading by cell type</p>`;
        }
    },

    async loadSingleCellSummary() {
        const container = document.getElementById('val-sc-summary');
        if (!container) return;

        try {
            const data = await API.getSingleCellDirect(
                this.currentAtlas,
                this.singlecell.signature,
                this.signatureType
            );

            if (data) {
                const isValid = data.fold_change > 1 && data.p_value < 0.05;

                container.innerHTML = `
                    <div class="summary-card">
                        <div class="summary-value">${data.n_expressing?.toLocaleString() || 'N/A'}</div>
                        <div class="summary-label">Expressing Cells</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-value">${data.n_non_expressing?.toLocaleString() || 'N/A'}</div>
                        <div class="summary-label">Non-Expressing Cells</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-value">${data.fold_change?.toFixed(2) || 'N/A'}x</div>
                        <div class="summary-label">Fold Change</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-value ${isValid ? 'valid' : 'invalid'}">${isValid ? 'Valid' : 'Not Valid'}</div>
                        <div class="summary-label">p = ${data.p_value?.toExponential(2) || 'N/A'}</div>
                    </div>
                `;
            } else {
                container.innerHTML = '<p class="no-data">No summary data available</p>';
            }
        } catch (error) {
            container.innerHTML = `<p class="error">Error loading summary</p>`;
        }
    },

    // ==================== Summary Tab ====================

    async loadSummary() {
        const content = document.getElementById('validation-content');

        content.innerHTML = `
            <div class="tab-panel">
                <h3>Validation Summary</h3>
                <p>Overall validation metrics comparing expression-activity correlations across atlases and signature types.</p>

                <div class="panel-grid">
                    <div class="panel">
                        <div class="viz-title">Mean Correlation by Signature Type</div>
                        <div class="viz-subtitle">Comparison of CytoSig and SecAct</div>
                        <div id="val-summary-sigtype" class="plot-container" style="height: 350px;"></div>
                    </div>
                    <div class="panel">
                        <div class="viz-title">Mean Correlation by Atlas</div>
                        <div class="viz-subtitle">Comparison across CIMA, Inflammation, and scAtlas</div>
                        <div id="val-summary-atlas" class="plot-container" style="height: 350px;"></div>
                    </div>
                </div>

                <div class="panel-grid">
                    <div class="panel full-width">
                        <div class="viz-title">Validation Level Comparison</div>
                        <div class="viz-subtitle">Correlation strength at cell type vs sample level</div>
                        <div id="val-summary-levels" class="plot-container" style="height: 300px;"></div>
                    </div>
                </div>

                <div class="panel-grid">
                    <div class="panel full-width">
                        <h4>Key Findings</h4>
                        <div id="val-summary-findings" class="findings">
                            <ul>
                                <li><strong>Pseudobulk Level:</strong> Highest correlations due to noise reduction from cell aggregation</li>
                                <li><strong>Single-Cell Level:</strong> Lower but significant correlations reflecting cell-to-cell variability</li>
                                <li><strong>Atlas Level:</strong> Cell type-specific patterns reveal biological validity</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Load visualizations
        await Promise.all([
            this.loadSummaryBySigType(),
            this.loadSummaryByAtlas(),
            this.loadSummaryByLevel()
        ]);
    },

    async loadSummaryBySigType() {
        const container = document.getElementById('val-summary-sigtype');
        if (!container) return;

        try {
            const results = [];
            for (const sigType of ['CytoSig', 'SecAct']) {
                try {
                    const summary = await API.getValidationSummary(this.currentAtlas, sigType);
                    results.push({
                        type: sigType,
                        sample_r: summary.sample_level_mean_r || 0,
                        celltype_r: summary.celltype_level_mean_r || 0
                    });
                } catch (e) {
                    // Skip
                }
            }

            if (results.length > 0) {
                const traces = [
                    {
                        x: results.map(r => r.type),
                        y: results.map(r => r.sample_r),
                        name: 'Sample Level',
                        type: 'bar'
                    },
                    {
                        x: results.map(r => r.type),
                        y: results.map(r => r.celltype_r),
                        name: 'Cell Type Level',
                        type: 'bar'
                    }
                ];

                const layout = {
                    barmode: 'group',
                    yaxis: { title: 'Mean Pearson r', range: [0, 1] },
                    margin: { l: 60, r: 20, t: 20, b: 60 },
                    legend: { orientation: 'h', y: 1.1 }
                };

                Plotly.newPlot(container, traces, layout, {responsive: true});
            } else {
                container.innerHTML = '<p class="no-data">No data available</p>';
            }
        } catch (error) {
            container.innerHTML = `<p class="error">Error loading summary</p>`;
        }
    },

    async loadSummaryByAtlas() {
        const container = document.getElementById('val-summary-atlas');
        if (!container) return;

        try {
            const atlases = ['cima', 'inflammation', 'scatlas'];
            const results = [];

            for (const atlas of atlases) {
                try {
                    const summary = await API.getValidationSummary(atlas, this.signatureType);
                    results.push({
                        atlas: atlas.charAt(0).toUpperCase() + atlas.slice(1),
                        sample_r: summary.sample_level_mean_r || 0,
                        celltype_r: summary.celltype_level_mean_r || 0
                    });
                } catch (e) {
                    // Skip
                }
            }

            if (results.length > 0) {
                const traces = [
                    {
                        x: results.map(r => r.atlas),
                        y: results.map(r => r.sample_r),
                        name: 'Sample Level',
                        type: 'bar'
                    },
                    {
                        x: results.map(r => r.atlas),
                        y: results.map(r => r.celltype_r),
                        name: 'Cell Type Level',
                        type: 'bar'
                    }
                ];

                const layout = {
                    barmode: 'group',
                    yaxis: { title: 'Mean Pearson r', range: [0, 1] },
                    margin: { l: 60, r: 20, t: 20, b: 60 },
                    legend: { orientation: 'h', y: 1.1 }
                };

                Plotly.newPlot(container, traces, layout, {responsive: true});
            } else {
                container.innerHTML = '<p class="no-data">No data available</p>';
            }
        } catch (error) {
            container.innerHTML = `<p class="error">Error loading summary</p>`;
        }
    },

    async loadSummaryByLevel() {
        const container = document.getElementById('val-summary-levels');
        if (!container) return;

        try {
            const summary = await API.getValidationSummary(this.currentAtlas, this.signatureType);

            const levels = ['Cell Type', 'Sample'];
            const values = [
                summary.celltype_level_mean_r || 0,
                summary.sample_level_mean_r || 0
            ];

            const trace = {
                x: levels,
                y: values,
                type: 'bar',
                marker: {
                    color: ['#3b82f6', '#22c55e']
                },
                text: values.map(v => v.toFixed(3)),
                textposition: 'outside'
            };

            const layout = {
                yaxis: { title: 'Mean Pearson r', range: [0, 1] },
                margin: { l: 60, r: 20, t: 40, b: 60 }
            };

            Plotly.newPlot(container, [trace], layout, {responsive: true});
        } catch (error) {
            container.innerHTML = `<p class="error">Error loading level comparison</p>`;
        }
    },

    // ==================== Utility Functions ====================

    calculateTrendline(x, y) {
        if (!x.length || !y.length) return {};

        const n = x.length;
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
        const sumX2 = x.reduce((acc, xi) => acc + xi * xi, 0);

        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;

        const minX = Math.min(...x);
        const maxX = Math.max(...x);

        return {
            x: [minX, maxX],
            y: [slope * minX + intercept, slope * maxX + intercept],
            mode: 'lines',
            type: 'scatter',
            line: { color: 'red', dash: 'dash', width: 1 },
            showlegend: false,
            hoverinfo: 'skip'
        };
    }
};

// Make available globally
window.ValidatePage = ValidatePage;
