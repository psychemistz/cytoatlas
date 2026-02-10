/**
 * Spatial Page
 * Displays SpatialCorpus-110M spatial transcriptomics activity data.
 *
 * Tabs:
 * 1. Overview - Technology summary, dataset catalog, tissue distribution
 * 2. Tissue Activity - Heatmap of tissues x signatures
 * 3. Technology Comparison - Cross-technology reproducibility scatter
 * 4. Gene Coverage - Gene panel overlap with CytoSig/SecAct per technology
 * 5. Spatial Map - Interactive tissue coordinate plot with activity overlay
 */

window.SpatialPage = (function () {
    'use strict';

    const API_BASE = '/api/v1/spatial';
    let tabPanel = null;

    function init(params, query) {
        const app = document.getElementById('app');
        if (!app) return;

        app.innerHTML = `
            <div class="page-header">
                <h1>Spatial Transcriptomics</h1>
                <p class="page-subtitle">
                    Cytokine and secreted protein activity across 251 spatial datasets
                    from 8 technologies (SpatialCorpus-110M, ~110M cells)
                </p>
            </div>
            <div id="spatial-tabs" class="tab-container"></div>
            <div id="spatial-content" class="tab-content-area"></div>
        `;

        initTabs();
    }

    function initTabs() {
        tabPanel = new TabPanel('spatial-tabs', { orientation: 'horizontal' });

        tabPanel.addTab('overview', 'Overview', loadOverview);
        tabPanel.addTab('tissue-activity', 'Tissue Activity', loadTissueActivity);
        tabPanel.addTab('tech-comparison', 'Technology Comparison', loadTechComparison);
        tabPanel.addTab('gene-coverage', 'Gene Coverage', loadGeneCoverage);
        tabPanel.addTab('spatial-map', 'Spatial Map', loadSpatialMap);

        tabPanel.init();
    }

    // =========================================================================
    // Tab 1: Overview
    // =========================================================================
    async function loadOverview() {
        const container = document.getElementById('spatial-content');
        container.innerHTML = '<div class="loading-skeleton"></div>';

        try {
            const summary = await API.get(`${API_BASE}/summary`);

            container.innerHTML = `
                <div class="panel">
                    <div class="panel-header">
                        <h3>SpatialCorpus-110M Overview</h3>
                    </div>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value">${(summary.total_cells || 110000000).toLocaleString()}</div>
                            <div class="stat-label">Total Cells</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${summary.total_datasets || 251}</div>
                            <div class="stat-label">Datasets</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${summary.technologies || 8}</div>
                            <div class="stat-label">Technologies</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${summary.tissues || '30+'}</div>
                            <div class="stat-label">Tissues</div>
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-header">
                        <h3>Technology Tiers</h3>
                        <p>Gene panel size determines the analysis strategy</p>
                    </div>
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Tier</th>
                                <th>Technologies</th>
                                <th>Files</th>
                                <th>Genes</th>
                                <th>Strategy</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>A</strong> (Full)</td>
                                <td>Visium</td>
                                <td>171</td>
                                <td>15K-20K</td>
                                <td>Full CytoSig + SecAct ridge regression</td>
                            </tr>
                            <tr>
                                <td><strong>B</strong> (Targeted)</td>
                                <td>Xenium, MERFISH, MERSCOPE, CosMx</td>
                                <td>51</td>
                                <td>150-1,000</td>
                                <td>Subset signatures with sufficient gene overlap</td>
                            </tr>
                            <tr>
                                <td><strong>C</strong> (Skip)</td>
                                <td>ISS, mouse datasets</td>
                                <td>12</td>
                                <td>&lt;150</td>
                                <td>Excluded</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <div class="panel">
                    <div class="panel-header">
                        <h3>Dataset Catalog</h3>
                    </div>
                    <div id="dataset-catalog"></div>
                </div>
            `;

            loadDatasetCatalog();
        } catch (error) {
            container.innerHTML = `<div class="error-message">Failed to load spatial overview: ${error.message}</div>`;
        }
    }

    async function loadDatasetCatalog() {
        try {
            const datasets = await API.get(`${API_BASE}/datasets`);
            const catalogEl = document.getElementById('dataset-catalog');

            if (!datasets || !Array.isArray(datasets) || datasets.length === 0) {
                catalogEl.innerHTML = '<p class="no-data">Dataset catalog not available yet. Run scripts/20_spatial_activity.py first.</p>';
                return;
            }

            const rows = datasets.slice(0, 50).map(d => `
                <tr>
                    <td>${d.dataset_id || d.filename || ''}</td>
                    <td>${d.technology || ''}</td>
                    <td>${d.tissue || ''}</td>
                    <td>${(d.n_cells || 0).toLocaleString()}</td>
                    <td>${d.n_genes || ''}</td>
                </tr>
            `).join('');

            catalogEl.innerHTML = `
                <table class="data-table">
                    <thead>
                        <tr><th>Dataset</th><th>Technology</th><th>Tissue</th><th>Cells</th><th>Genes</th></tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
                <p class="table-footer">Showing ${Math.min(datasets.length, 50)} of ${datasets.length} datasets</p>
            `;
        } catch (error) {
            document.getElementById('dataset-catalog').innerHTML =
                `<p class="error-message">Failed to load dataset catalog: ${error.message}</p>`;
        }
    }

    // =========================================================================
    // Tab 2: Tissue Activity
    // =========================================================================
    async function loadTissueActivity() {
        const container = document.getElementById('spatial-content');
        container.innerHTML = '<div class="loading-skeleton"></div>';

        try {
            const data = await API.get(`${API_BASE}/tissue-summary`);

            container.innerHTML = `
                <div class="panel">
                    <div class="panel-header">
                        <h3>Tissue-Level Activity</h3>
                        <p>Cytokine/secreted protein activity across tissues from spatial transcriptomics</p>
                        <div class="panel-controls">
                            <select id="tissue-sig-type" class="select-control">
                                <option value="CytoSig">CytoSig</option>
                                <option value="SecAct">SecAct</option>
                            </select>
                        </div>
                    </div>
                    <div id="tissue-heatmap" class="chart-container" style="height: 500px;"></div>
                </div>
            `;

            renderTissueHeatmap(data);
        } catch (error) {
            container.innerHTML = `<div class="error-message">Failed to load tissue activity: ${error.message}</div>`;
        }
    }

    function renderTissueHeatmap(data) {
        if (!data || !Array.isArray(data) || data.length === 0) {
            document.getElementById('tissue-heatmap').innerHTML =
                '<p class="no-data">No tissue activity data available yet. Run scripts/20_spatial_activity.py first.</p>';
            return;
        }

        const tissues = [...new Set(data.map(d => d.tissue))];
        const signatures = [...new Set(data.map(d => d.signature))].slice(0, 30);

        const zValues = tissues.map(tissue =>
            signatures.map(sig => {
                const record = data.find(d => d.tissue === tissue && d.signature === sig);
                return record ? record.mean_activity : 0;
            })
        );

        Plotly.newPlot('tissue-heatmap', [{
            type: 'heatmap',
            z: zValues,
            x: signatures,
            y: tissues,
            colorscale: 'Viridis',
            colorbar: { title: { text: 'Mean Activity' } },
        }], {
            margin: { l: 120, r: 40, t: 20, b: 100 },
            xaxis: { title: 'Signature', tickangle: -45 },
            yaxis: { title: 'Tissue' },
        }, { responsive: true });
    }

    // =========================================================================
    // Tab 3: Technology Comparison
    // =========================================================================
    async function loadTechComparison() {
        const container = document.getElementById('spatial-content');
        container.innerHTML = '<div class="loading-skeleton"></div>';

        try {
            const data = await API.get(`${API_BASE}/technology-comparison`);

            container.innerHTML = `
                <div class="panel">
                    <div class="panel-header">
                        <h3>Cross-Technology Reproducibility</h3>
                        <p>Activity correlation between different spatial technologies for the same tissues</p>
                    </div>
                    <div id="tech-comparison-scatter" class="chart-container" style="height: 500px;"></div>
                </div>
            `;

            renderTechComparison(data);
        } catch (error) {
            container.innerHTML = `<div class="error-message">Failed to load technology comparison: ${error.message}</div>`;
        }
    }

    function renderTechComparison(data) {
        if (!data || !Array.isArray(data) || data.length === 0) {
            document.getElementById('tech-comparison-scatter').innerHTML =
                '<p class="no-data">No technology comparison data available yet. Run scripts/20_spatial_activity.py first.</p>';
            return;
        }

        const techPairs = [...new Set(data.map(d => `${d.technology_1} vs ${d.technology_2}`))];

        const traces = techPairs.map(pair => {
            const pairData = data.filter(d => `${d.technology_1} vs ${d.technology_2}` === pair);
            return {
                type: 'scatter',
                mode: 'markers',
                x: pairData.map(d => d.activity_tech1),
                y: pairData.map(d => d.activity_tech2),
                text: pairData.map(d => d.tissue),
                name: pair,
                marker: { size: 6, opacity: 0.6 },
            };
        });

        Plotly.newPlot('tech-comparison-scatter', traces, {
            xaxis: { title: 'Activity (Technology 1)' },
            yaxis: { title: 'Activity (Technology 2)' },
            margin: { l: 60, r: 20, t: 20, b: 60 },
            showlegend: true,
        }, { responsive: true });
    }

    // =========================================================================
    // Tab 4: Gene Coverage
    // =========================================================================
    async function loadGeneCoverage() {
        const container = document.getElementById('spatial-content');
        container.innerHTML = '<div class="loading-skeleton"></div>';

        try {
            const data = await API.get(`${API_BASE}/gene-coverage`);

            container.innerHTML = `
                <div class="panel">
                    <div class="panel-header">
                        <h3>Gene Panel Coverage</h3>
                        <p>Fraction of CytoSig/SecAct signature genes present in each technology's gene panel</p>
                    </div>
                    <div id="gene-coverage-chart" class="chart-container" style="height: 400px;"></div>
                </div>
            `;

            renderGeneCoverage(data);
        } catch (error) {
            container.innerHTML = `<div class="error-message">Failed to load gene coverage: ${error.message}</div>`;
        }
    }

    function renderGeneCoverage(data) {
        if (!data || !Array.isArray(data) || data.length === 0) {
            document.getElementById('gene-coverage-chart').innerHTML =
                '<p class="no-data">No gene coverage data available yet. Run scripts/20_spatial_activity.py first.</p>';
            return;
        }

        const technologies = data.map(d => d.technology);

        Plotly.newPlot('gene-coverage-chart', [
            {
                type: 'bar',
                x: technologies,
                y: data.map(d => (d.cytosig_coverage || 0) * 100),
                name: 'CytoSig Coverage (%)',
                marker: { color: '#3498db' },
            },
            {
                type: 'bar',
                x: technologies,
                y: data.map(d => (d.secact_coverage || 0) * 100),
                name: 'SecAct Coverage (%)',
                marker: { color: '#e74c3c' },
            },
        ], {
            barmode: 'group',
            xaxis: { title: 'Technology' },
            yaxis: { title: 'Gene Coverage (%)', range: [0, 100] },
            margin: { l: 60, r: 20, t: 20, b: 60 },
            showlegend: true,
        }, { responsive: true });
    }

    // =========================================================================
    // Tab 5: Spatial Map
    // =========================================================================
    async function loadSpatialMap() {
        const container = document.getElementById('spatial-content');
        container.innerHTML = '<div class="loading-skeleton"></div>';

        try {
            const datasets = await API.get(`${API_BASE}/datasets`);

            container.innerHTML = `
                <div class="panel">
                    <div class="panel-header">
                        <h3>Spatial Activity Map</h3>
                        <p>Select a dataset to view spatial coordinates colored by activity</p>
                        <div class="panel-controls">
                            <select id="spatial-dataset-select" class="select-control">
                                <option value="">Select a dataset...</option>
                            </select>
                        </div>
                    </div>
                    <div id="spatial-map-plot" class="chart-container" style="height: 600px;"></div>
                </div>
            `;

            if (datasets && Array.isArray(datasets)) {
                const select = document.getElementById('spatial-dataset-select');
                datasets.slice(0, 100).forEach(ds => {
                    const opt = document.createElement('option');
                    opt.value = ds.dataset_id || ds.filename || '';
                    opt.textContent = `${ds.tissue || 'Unknown'} - ${ds.technology || ''} (${(ds.n_cells || 0).toLocaleString()} cells)`;
                    select.appendChild(opt);
                });

                select.addEventListener('change', async () => {
                    if (select.value) {
                        await loadSpatialCoordinates(select.value);
                    }
                });
            }

            document.getElementById('spatial-map-plot').innerHTML =
                '<p class="no-data">Select a dataset above to view spatial coordinates.</p>';
        } catch (error) {
            container.innerHTML = `<div class="error-message">Failed to load spatial map: ${error.message}</div>`;
        }
    }

    async function loadSpatialCoordinates(datasetId) {
        const plotEl = document.getElementById('spatial-map-plot');
        plotEl.innerHTML = '<div class="loading-skeleton"></div>';

        try {
            const coords = await API.get(`${API_BASE}/coordinates/${encodeURIComponent(datasetId)}`);

            if (!coords || !Array.isArray(coords) || coords.length === 0) {
                plotEl.innerHTML = '<p class="no-data">No spatial coordinates available for this dataset.</p>';
                return;
            }

            Plotly.newPlot('spatial-map-plot', [{
                type: 'scattergl',
                mode: 'markers',
                x: coords.map(c => c.x_coord),
                y: coords.map(c => c.y_coord),
                marker: {
                    size: 3,
                    color: coords.map(c => c.activity || 0),
                    colorscale: 'Viridis',
                    colorbar: { title: { text: 'Activity' } },
                    opacity: 0.6,
                },
                text: coords.map(c => c.cell_type || ''),
                hovertemplate: 'x: %{x:.0f}<br>y: %{y:.0f}<br>Activity: %{marker.color:.2f}<br>%{text}<extra></extra>',
            }], {
                xaxis: { title: 'X', scaleanchor: 'y' },
                yaxis: { title: 'Y' },
                margin: { l: 60, r: 40, t: 20, b: 60 },
            }, { responsive: true });
        } catch (error) {
            plotEl.innerHTML = `<p class="error-message">Failed to load coordinates: ${error.message}</p>`;
        }
    }

    return { init };
})();
