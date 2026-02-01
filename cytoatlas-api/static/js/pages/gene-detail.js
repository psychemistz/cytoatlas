/**
 * Gene Detail Page Handler
 * Displays gene/signature-centric views across all atlases
 */

const GeneDetailPage = {
    signature: null,
    signatureType: 'CytoSig',
    activeTab: 'cell-types',

    tabs: [
        { id: 'cell-types', label: 'Cell Types', icon: '&#128300;' },
        { id: 'tissues', label: 'Tissues', icon: '&#128149;' },
        { id: 'diseases', label: 'Diseases', icon: '&#129658;' },
        { id: 'correlations', label: 'Correlations', icon: '&#128200;' },
        { id: 'cross-atlas', label: 'Cross-Atlas', icon: '&#128202;' },
    ],

    /**
     * Initialize the gene detail page
     */
    async init(params, query) {
        this.signature = params.signature;
        this.signatureType = query.type || 'CytoSig';
        this.activeTab = query.tab || 'cell-types';

        this.render();
        await this.loadOverview();
    },

    /**
     * Render the page template
     */
    render() {
        const app = document.getElementById('app');

        app.innerHTML = `
            <div class="gene-detail-page">
                <header class="gene-header">
                    <div class="header-top">
                        <a href="/search" class="back-link">&#8592; Back to Search</a>
                        <div class="signature-type-toggle">
                            <button class="toggle-btn ${this.signatureType === 'CytoSig' ? 'active' : ''}"
                                    onclick="GeneDetailPage.changeSignatureType('CytoSig')">
                                CytoSig
                            </button>
                            <button class="toggle-btn ${this.signatureType === 'SecAct' ? 'active' : ''}"
                                    onclick="GeneDetailPage.changeSignatureType('SecAct')">
                                SecAct
                            </button>
                        </div>
                    </div>
                    <div id="gene-title" class="gene-title">
                        <h1>${this.signature}</h1>
                        <span class="signature-type-badge ${this.signatureType.toLowerCase()}">${this.signatureType}</span>
                    </div>
                    <div id="gene-summary" class="gene-summary">Loading...</div>
                </header>

                <nav class="gene-tabs" id="gene-tabs">
                    ${this.tabs.map(tab => `
                        <button class="gene-tab ${tab.id === this.activeTab ? 'active' : ''}"
                                data-tab="${tab.id}"
                                onclick="GeneDetailPage.switchTab('${tab.id}')">
                            <span>${tab.icon}</span> ${tab.label}
                        </button>
                    `).join('')}
                </nav>

                <main class="gene-content" id="gene-content">
                    <div class="loading"><div class="spinner"></div>Loading...</div>
                </main>
            </div>
        `;
    },

    /**
     * Load gene overview and summary
     */
    async loadOverview() {
        try {
            const overview = await API.get(`/gene/${encodeURIComponent(this.signature)}`, {
                signature_type: this.signatureType,
            });

            // Update title
            const titleEl = document.getElementById('gene-title');
            if (titleEl) {
                titleEl.innerHTML = `
                    <h1>${overview.signature}</h1>
                    <span class="signature-type-badge ${overview.signature_type.toLowerCase()}">${overview.signature_type}</span>
                    <span class="atlas-badges">
                        ${overview.atlases.map(a => `<span class="atlas-badge">${a}</span>`).join('')}
                    </span>
                `;
            }

            // Update summary
            const summaryEl = document.getElementById('gene-summary');
            if (summaryEl) {
                const stats = overview.summary_stats;
                summaryEl.innerHTML = `
                    <div class="summary-stats">
                        <div class="stat">
                            <span class="value">${stats.n_atlases}</span>
                            <span class="label">Atlases</span>
                        </div>
                        <div class="stat">
                            <span class="value">${stats.n_cell_types}</span>
                            <span class="label">Cell Types</span>
                        </div>
                        <div class="stat">
                            <span class="value">${stats.n_tissues}</span>
                            <span class="label">Tissues</span>
                        </div>
                        <div class="stat">
                            <span class="value">${stats.n_diseases}</span>
                            <span class="label">Diseases</span>
                        </div>
                        <div class="stat">
                            <span class="value">${stats.n_correlations}</span>
                            <span class="label">Correlations</span>
                        </div>
                    </div>
                    ${stats.top_cell_type ? `<p class="top-feature">Top cell type: <strong>${stats.top_cell_type}</strong></p>` : ''}
                    ${stats.top_tissue ? `<p class="top-feature">Top tissue: <strong>${stats.top_tissue}</strong></p>` : ''}
                `;
            }

            // Load the active tab
            await this.loadTabContent(this.activeTab);

        } catch (error) {
            console.error('Failed to load gene overview:', error);
            const content = document.getElementById('gene-content');
            if (content) {
                content.innerHTML = `
                    <div class="error-message">
                        <h3>Signature Not Found</h3>
                        <p>${this.signature} was not found in ${this.signatureType}.</p>
                        <p>Try switching to ${this.signatureType === 'CytoSig' ? 'SecAct' : 'CytoSig'} or search for a different signature.</p>
                        <a href="/search" class="btn btn-primary">Back to Search</a>
                    </div>
                `;
            }
        }
    },

    /**
     * Switch signature type
     */
    changeSignatureType(type) {
        if (type === this.signatureType) return;

        this.signatureType = type;

        // Update URL
        const url = new URL(window.location);
        url.searchParams.set('type', type);
        window.history.replaceState({}, '', url);

        // Reload page content
        this.render();
        this.loadOverview();
    },

    /**
     * Switch to a different tab
     */
    async switchTab(tabId) {
        // Update active state
        document.querySelectorAll('.gene-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabId);
        });

        this.activeTab = tabId;

        // Update URL
        const url = new URL(window.location);
        url.searchParams.set('tab', tabId);
        window.history.replaceState({}, '', url);

        await this.loadTabContent(tabId);
    },

    /**
     * Load tab content
     */
    async loadTabContent(tabId) {
        const content = document.getElementById('gene-content');
        if (!content) return;

        content.innerHTML = '<div class="loading"><div class="spinner"></div>Loading...</div>';

        try {
            switch (tabId) {
                case 'cell-types':
                    await this.loadCellTypesTab(content);
                    break;
                case 'tissues':
                    await this.loadTissuesTab(content);
                    break;
                case 'diseases':
                    await this.loadDiseasesTab(content);
                    break;
                case 'correlations':
                    await this.loadCorrelationsTab(content);
                    break;
                case 'cross-atlas':
                    await this.loadCrossAtlasTab(content);
                    break;
                default:
                    content.innerHTML = '<p>Tab not found</p>';
            }
        } catch (error) {
            console.error(`Failed to load ${tabId} tab:`, error);
            content.innerHTML = `<div class="error-message">Failed to load data: ${error.message}</div>`;
        }
    },

    // ==================== Cell Types Tab ====================

    async loadCellTypesTab(content) {
        const data = await API.get(`/gene/${encodeURIComponent(this.signature)}/cell-types`, {
            signature_type: this.signatureType,
        });

        if (!data || !data.length) {
            content.innerHTML = '<div class="no-data">No cell type data available.</div>';
            return;
        }

        // Group by atlas
        const byAtlas = {};
        data.forEach(d => {
            if (!byAtlas[d.atlas]) byAtlas[d.atlas] = [];
            byAtlas[d.atlas].push(d);
        });

        content.innerHTML = `
            <div class="tab-header">
                <h2>Cell Type Activity</h2>
                <p>Activity of ${this.signature} across cell types in each atlas</p>
                <div class="tab-filters">
                    <select id="atlas-filter" onchange="GeneDetailPage.filterCellTypesByAtlas()">
                        <option value="all">All Atlases</option>
                        ${Object.keys(byAtlas).map(a => `<option value="${a}">${a.toUpperCase()}</option>`).join('')}
                    </select>
                </div>
            </div>
            <div class="viz-container">
                <div id="celltype-chart" class="chart-container"></div>
            </div>
            <div class="data-table-container">
                <table class="data-table" id="celltype-table">
                    <thead>
                        <tr>
                            <th>Cell Type</th>
                            <th>Atlas</th>
                            <th>Mean Activity</th>
                            <th>N Cells</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.slice(0, 50).map(d => `
                            <tr>
                                <td>${d.cell_type}</td>
                                <td><span class="atlas-badge">${d.atlas}</span></td>
                                <td class="${d.mean_activity >= 0 ? 'positive' : 'negative'}">${d.mean_activity.toFixed(4)}</td>
                                <td>${d.n_cells ? d.n_cells.toLocaleString() : '-'}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;

        // Store data for filtering
        this._cellTypeData = data;

        // Create chart
        this.createCellTypeChart(data);
    },

    createCellTypeChart(data) {
        // Take top 30 for visualization
        const topData = data.slice(0, 30);

        const traces = [];
        const atlases = [...new Set(topData.map(d => d.atlas))];

        atlases.forEach(atlas => {
            const atlasData = topData.filter(d => d.atlas === atlas);
            traces.push({
                type: 'bar',
                name: atlas.toUpperCase(),
                y: atlasData.map(d => d.cell_type),
                x: atlasData.map(d => d.mean_activity),
                orientation: 'h',
                text: atlasData.map(d => d.mean_activity.toFixed(3)),
                textposition: 'auto',
                hovertemplate: '<b>%{y}</b><br>Activity: %{x:.4f}<extra></extra>',
            });
        });

        const layout = {
            title: `${this.signature} Activity by Cell Type`,
            barmode: 'group',
            xaxis: {
                title: 'Mean Activity (z-score)',
                zeroline: true,
                zerolinecolor: '#999',
            },
            yaxis: {
                title: '',
                automargin: true,
            },
            height: Math.max(400, topData.length * 20),
            margin: { l: 200, r: 50, t: 50, b: 50 },
            legend: { orientation: 'h', y: 1.1 },
        };

        Plotly.newPlot('celltype-chart', traces, layout, { responsive: true });
    },

    filterCellTypesByAtlas() {
        const atlas = document.getElementById('atlas-filter').value;
        let filtered = this._cellTypeData;

        if (atlas !== 'all') {
            filtered = filtered.filter(d => d.atlas === atlas);
        }

        // Update chart
        this.createCellTypeChart(filtered);

        // Update table
        const tbody = document.querySelector('#celltype-table tbody');
        if (tbody) {
            tbody.innerHTML = filtered.slice(0, 50).map(d => `
                <tr>
                    <td>${d.cell_type}</td>
                    <td><span class="atlas-badge">${d.atlas}</span></td>
                    <td class="${d.mean_activity >= 0 ? 'positive' : 'negative'}">${d.mean_activity.toFixed(4)}</td>
                    <td>${d.n_cells ? d.n_cells.toLocaleString() : '-'}</td>
                </tr>
            `).join('');
        }
    },

    // ==================== Tissues Tab ====================

    async loadTissuesTab(content) {
        const data = await API.get(`/gene/${encodeURIComponent(this.signature)}/tissues`, {
            signature_type: this.signatureType,
        });

        if (!data || !data.length) {
            content.innerHTML = '<div class="no-data">No tissue data available. Tissue data is from scAtlas.</div>';
            return;
        }

        content.innerHTML = `
            <div class="tab-header">
                <h2>Tissue Activity</h2>
                <p>Activity of ${this.signature} across organs from scAtlas</p>
            </div>
            <div class="viz-container">
                <div id="tissue-chart" class="chart-container"></div>
            </div>
            <div class="data-table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Organ</th>
                            <th>Mean Activity</th>
                            <th>Specificity</th>
                            <th>N Cells</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.map(d => `
                            <tr>
                                <td>${d.rank || '-'}</td>
                                <td>${d.organ}</td>
                                <td class="${d.mean_activity >= 0 ? 'positive' : 'negative'}">${d.mean_activity.toFixed(4)}</td>
                                <td>${d.specificity_score ? d.specificity_score.toFixed(4) : '-'}</td>
                                <td>${d.n_cells ? d.n_cells.toLocaleString() : '-'}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;

        // Create chart
        const trace = {
            type: 'bar',
            y: data.map(d => d.organ),
            x: data.map(d => d.mean_activity),
            orientation: 'h',
            marker: {
                color: data.map(d => d.mean_activity >= 0 ? '#2ecc71' : '#e74c3c'),
            },
            text: data.map(d => d.mean_activity.toFixed(3)),
            textposition: 'auto',
            hovertemplate: '<b>%{y}</b><br>Activity: %{x:.4f}<extra></extra>',
        };

        const layout = {
            title: `${this.signature} Activity by Organ`,
            xaxis: {
                title: 'Mean Activity (z-score)',
                zeroline: true,
                zerolinecolor: '#999',
            },
            yaxis: {
                title: '',
                automargin: true,
            },
            height: Math.max(400, data.length * 25),
            margin: { l: 150, r: 50, t: 50, b: 50 },
        };

        Plotly.newPlot('tissue-chart', [trace], layout, { responsive: true });
    },

    // ==================== Diseases Tab ====================

    async loadDiseasesTab(content) {
        const response = await API.get(`/gene/${encodeURIComponent(this.signature)}/diseases`, {
            signature_type: this.signatureType,
        });

        const data = response.data || [];

        if (!data.length) {
            content.innerHTML = '<div class="no-data">No disease data available. Disease data is from the Inflammation Atlas.</div>';
            return;
        }

        content.innerHTML = `
            <div class="tab-header">
                <h2>Disease Associations</h2>
                <p>Differential activity of ${this.signature} (disease vs healthy) from the Inflammation Atlas</p>
                <div class="stats-inline">
                    <span><strong>${response.n_diseases}</strong> diseases</span>
                    <span><strong>${response.n_significant}</strong> significant (FDR < 0.05)</span>
                </div>
            </div>
            <div class="viz-container">
                <div id="volcano-chart" class="chart-container"></div>
            </div>
            <div class="data-table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Disease</th>
                            <th>Group</th>
                            <th>&Delta; Activity</th>
                            <th>P-value</th>
                            <th>FDR</th>
                            <th>Significant</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.map(d => `
                            <tr class="${d.is_significant ? 'significant' : ''}">
                                <td>${d.disease}</td>
                                <td>${d.disease_group}</td>
                                <td class="${d.activity_diff >= 0 ? 'positive' : 'negative'}">${d.activity_diff.toFixed(4)}</td>
                                <td>${d.p_value < 0.001 ? d.p_value.toExponential(2) : d.p_value.toFixed(4)}</td>
                                <td>${d.q_value ? (d.q_value < 0.001 ? d.q_value.toExponential(2) : d.q_value.toFixed(4)) : '-'}</td>
                                <td>${d.is_significant ? '&#10003;' : ''}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;

        // Create volcano plot
        const sigData = data.filter(d => d.is_significant);
        const nsData = data.filter(d => !d.is_significant);

        const traces = [];

        if (nsData.length) {
            traces.push({
                type: 'scatter',
                mode: 'markers',
                name: 'Not Significant',
                x: nsData.map(d => d.activity_diff),
                y: nsData.map(d => d.neg_log10_pval || -Math.log10(d.p_value)),
                text: nsData.map(d => d.disease),
                marker: { color: '#999', size: 8 },
                hovertemplate: '<b>%{text}</b><br>&Delta; Activity: %{x:.3f}<br>-log10(p): %{y:.2f}<extra></extra>',
            });
        }

        if (sigData.length) {
            traces.push({
                type: 'scatter',
                mode: 'markers+text',
                name: 'Significant (FDR < 0.05)',
                x: sigData.map(d => d.activity_diff),
                y: sigData.map(d => d.neg_log10_pval || -Math.log10(d.p_value)),
                text: sigData.map(d => d.disease),
                textposition: 'top center',
                textfont: { size: 10 },
                marker: {
                    color: sigData.map(d => d.activity_diff >= 0 ? '#2ecc71' : '#e74c3c'),
                    size: 12,
                },
                hovertemplate: '<b>%{text}</b><br>&Delta; Activity: %{x:.3f}<br>-log10(p): %{y:.2f}<extra></extra>',
            });
        }

        const layout = {
            title: `${this.signature} Disease Volcano Plot`,
            xaxis: {
                title: '&Delta; Activity (disease - healthy)',
                zeroline: true,
                zerolinecolor: '#999',
            },
            yaxis: {
                title: '-log10(p-value)',
            },
            height: 500,
            showlegend: true,
            legend: { orientation: 'h', y: -0.15 },
            shapes: [{
                type: 'line',
                x0: 0, x1: 0,
                y0: 0, y1: Math.max(...data.map(d => d.neg_log10_pval || -Math.log10(d.p_value))) + 1,
                line: { color: '#999', dash: 'dot' },
            }],
        };

        Plotly.newPlot('volcano-chart', traces, layout, { responsive: true });
    },

    // ==================== Correlations Tab ====================

    async loadCorrelationsTab(content) {
        const data = await API.get(`/gene/${encodeURIComponent(this.signature)}/correlations`, {
            signature_type: this.signatureType,
        });

        content.innerHTML = `
            <div class="tab-header">
                <h2>Correlations</h2>
                <p>Correlations of ${this.signature} with clinical and molecular variables (from CIMA)</p>
            </div>
            <div class="correlation-sections">
                <div class="correlation-section">
                    <h3>&#128197; Age Correlations (${data.n_significant_age} significant)</h3>
                    ${this.renderCorrelationList(data.age, 'age')}
                </div>
                <div class="correlation-section">
                    <h3>&#9878; BMI Correlations (${data.n_significant_bmi} significant)</h3>
                    ${this.renderCorrelationList(data.bmi, 'bmi')}
                </div>
                <div class="correlation-section">
                    <h3>&#129514; Biochemistry Correlations (${data.n_significant_biochem} significant)</h3>
                    ${this.renderCorrelationList(data.biochemistry, 'biochemistry')}
                </div>
                <div class="correlation-section">
                    <h3>&#9879; Metabolite Correlations (${data.n_significant_metabol} significant)</h3>
                    ${this.renderCorrelationList(data.metabolites, 'metabolites')}
                </div>
            </div>
        `;
    },

    renderCorrelationList(correlations, category) {
        if (!correlations || !correlations.length) {
            return '<p class="no-data">No data available.</p>';
        }

        // Take top 10
        const topCorr = correlations.slice(0, 10);

        return `
            <div class="correlation-list">
                ${topCorr.map(c => {
                    const isSignificant = c.q_value && c.q_value < 0.05;
                    return `
                        <div class="correlation-item ${isSignificant ? 'significant' : ''}">
                            <span class="variable">${c.variable}${c.cell_type && c.cell_type !== 'All' ? ` (${c.cell_type})` : ''}</span>
                            <span class="rho ${c.rho >= 0 ? 'positive' : 'negative'}">&rho; = ${c.rho.toFixed(3)}</span>
                            <span class="pval">p = ${c.p_value < 0.001 ? c.p_value.toExponential(2) : c.p_value.toFixed(4)}</span>
                            ${isSignificant ? '<span class="sig-badge">*</span>' : ''}
                        </div>
                    `;
                }).join('')}
            </div>
            ${correlations.length > 10 ? `<p class="more-link">...and ${correlations.length - 10} more</p>` : ''}
        `;
    },

    // ==================== Cross-Atlas Tab ====================

    async loadCrossAtlasTab(content) {
        const data = await API.get(`/gene/${encodeURIComponent(this.signature)}/cross-atlas`, {
            signature_type: this.signatureType,
        });

        if (!data.atlases || !data.atlases.length) {
            content.innerHTML = '<div class="no-data">No cross-atlas data available.</div>';
            return;
        }

        content.innerHTML = `
            <div class="tab-header">
                <h2>Cross-Atlas Comparison</h2>
                <p>${this.signature} activity across ${data.n_atlases} atlases</p>
            </div>
            <div class="viz-container">
                <div id="crossatlas-chart" class="chart-container"></div>
            </div>
            <div class="atlas-cards">
                ${data.activity_by_atlas.map(a => `
                    <div class="atlas-card">
                        <h4>${a.atlas.toUpperCase()}</h4>
                        <div class="activity-value ${a.mean_activity >= 0 ? 'positive' : 'negative'}">
                            ${a.mean_activity.toFixed(4)}
                        </div>
                        <div class="cell-type">${a.cell_type}</div>
                    </div>
                `).join('')}
            </div>
            ${data.consistency_score ? `
                <div class="consistency-score">
                    <h4>Consistency Score</h4>
                    <div class="score-value">${data.consistency_score.toFixed(3)}</div>
                    <p class="score-desc">Mean pairwise correlation of activities</p>
                </div>
            ` : ''}
        `;

        // Create bar chart
        const trace = {
            type: 'bar',
            x: data.activity_by_atlas.map(a => a.atlas.toUpperCase()),
            y: data.activity_by_atlas.map(a => a.mean_activity),
            marker: {
                color: ['#3498db', '#e74c3c', '#2ecc71'],
            },
            text: data.activity_by_atlas.map(a => a.mean_activity.toFixed(3)),
            textposition: 'auto',
        };

        const layout = {
            title: `${this.signature} Mean Activity by Atlas`,
            xaxis: { title: 'Atlas' },
            yaxis: { title: 'Mean Activity (z-score)', zeroline: true },
            height: 400,
        };

        Plotly.newPlot('crossatlas-chart', [trace], layout, { responsive: true });
    },
};

// Make available globally
window.GeneDetailPage = GeneDetailPage;
