/**
 * Atlas Detail Page Handler
 * Shows atlas analysis panels with tabs - migrated from visualization/index.html
 */

const AtlasDetailPage = {
    currentAtlas: null,
    currentTab: null,
    signatureType: 'CytoSig',

    /**
     * Atlas configurations with available analysis tabs
     * Matches the panels from visualization/index.html
     */
    atlasConfigs: {
        cima: {
            displayName: 'CIMA',
            description: 'Chinese Immune Multi-omics Atlas - 6.5M cells from 421 healthy adults with matched biochemistry and metabolomics',
            tabs: [
                { id: 'overview', label: 'Overview', icon: '&#127968;' },
                { id: 'celltypes', label: 'Cell Types', icon: '&#128300;' },
                { id: 'age-bmi', label: 'Age & BMI', icon: '&#128200;' },
                { id: 'age-bmi-stratified', label: 'Age/BMI Stratified', icon: '&#128202;' },
                { id: 'biochemistry', label: 'Biochemistry', icon: '&#129514;' },
                { id: 'biochem-scatter', label: 'Biochem Scatter', icon: '&#128201;' },
                { id: 'metabolites', label: 'Metabolites', icon: '&#9879;' },
                { id: 'differential', label: 'Differential', icon: '&#128209;' },
                { id: 'multiomics', label: 'Multi-omics', icon: '&#128300;' },
                { id: 'population', label: 'Population', icon: '&#128101;' },
                { id: 'eqtl', label: 'eQTL Browser', icon: '&#129516;' },
            ],
        },
        inflammation: {
            displayName: 'Inflammation Atlas',
            description: 'Pan-disease immune profiling - 4.9M cells across 12+ inflammatory diseases with treatment response data',
            tabs: [
                { id: 'overview', label: 'Overview', icon: '&#127968;' },
                { id: 'celltypes', label: 'Cell Types', icon: '&#128300;' },
                { id: 'age-bmi', label: 'Age & BMI', icon: '&#128200;' },
                { id: 'age-bmi-stratified', label: 'Age/BMI Stratified', icon: '&#128202;' },
                { id: 'disease', label: 'Disease', icon: '&#129658;' },
                { id: 'differential', label: 'Differential', icon: '&#128209;' },
                { id: 'treatment', label: 'Treatment Response', icon: '&#128137;' },
                { id: 'sankey', label: 'Disease Flow', icon: '&#128260;' },
                { id: 'validation', label: 'Cohort Validation', icon: '&#9989;' },
                { id: 'longitudinal', label: 'Longitudinal', icon: '&#128197;' },
                { id: 'severity', label: 'Severity', icon: '&#128200;' },
                { id: 'drivers', label: 'Cell Drivers', icon: '&#128302;' },
            ],
        },
        scatlas: {
            displayName: 'scAtlas',
            description: 'Human tissue reference atlas - 6.4M cells across 35 organs with pan-cancer immune profiling',
            tabs: [
                { id: 'overview', label: 'Overview', icon: '&#127968;' },
                { id: 'celltypes', label: 'Cell Types', icon: '&#128300;' },
                { id: 'organ-map', label: 'Organ Map', icon: '&#128149;' },
                { id: 'cancer-comparison', label: 'Tumor vs Adjacent', icon: '&#128201;' },
                { id: 'cancer-types', label: 'Cancer Types', icon: '&#129656;' },
                { id: 'immune-infiltration', label: 'Immune Infiltration', icon: '&#128300;' },
                { id: 'exhaustion', label: 'T Cell Exhaustion', icon: '&#128546;' },
                { id: 'caf', label: 'CAF Types', icon: '&#128302;' },
                { id: 'organ-cancer-matrix', label: 'Organ-Cancer', icon: '&#128202;' },
                { id: 'adjacent-tissue', label: 'Adjacent Tissue', icon: '&#129516;' },
            ],
        },
    },

    /**
     * Initialize the atlas detail page
     */
    async init(params) {
        this.currentAtlas = params.name;
        this.currentTab = null;

        // Render template
        this.render();

        // Load atlas info and tabs
        await this.loadAtlasInfo();
    },

    /**
     * Render the page template
     */
    render() {
        const app = document.getElementById('app');
        const template = document.getElementById('atlas-detail-template');

        if (app && template) {
            app.innerHTML = template.innerHTML;
        }
    },

    /**
     * Load atlas info and set up tabs
     */
    async loadAtlasInfo() {
        const config = this.atlasConfigs[this.currentAtlas] || {
            displayName: this.currentAtlas,
            description: '',
            tabs: [{ id: 'celltypes', label: 'Cell Types', icon: '&#128300;' }],
        };

        // Set default tab
        this.currentTab = config.tabs[0].id;

        // Render header
        this.renderHeader(config);

        // Render tabs
        this.renderTabs(config.tabs);

        // Load default tab content
        await this.loadTabContent(this.currentTab);
    },

    /**
     * Render atlas header
     */
    renderHeader(config) {
        const header = document.getElementById('atlas-header');
        if (!header) return;

        header.innerHTML = `
            <div class="page-header">
                <h1>${config.displayName}</h1>
                <p>${config.description}</p>
            </div>
            <div class="atlas-controls">
                <select id="signature-type-select" class="filter-select" onchange="AtlasDetailPage.changeSignatureType(this.value)">
                    <option value="CytoSig">CytoSig (44 cytokines)</option>
                    <option value="SecAct">SecAct (1,249 proteins)</option>
                </select>
                <a href="/validate?atlas=${this.currentAtlas}" class="btn btn-secondary">View Validation</a>
                <button class="btn btn-secondary" onclick="AtlasDetailPage.exportData()">Export CSV</button>
            </div>
        `;
    },

    /**
     * Render analysis tabs
     */
    renderTabs(tabs) {
        const tabsContainer = document.getElementById('analysis-tabs');
        if (!tabsContainer) return;

        tabsContainer.innerHTML = tabs.map(tab => `
            <button class="analysis-tab ${tab.id === this.currentTab ? 'active' : ''}"
                    data-tab="${tab.id}"
                    onclick="AtlasDetailPage.switchTab('${tab.id}')">
                <span>${tab.icon}</span> ${tab.label}
            </button>
        `).join('');
    },

    /**
     * Switch to a different tab
     */
    async switchTab(tabId) {
        // Update active state
        document.querySelectorAll('.analysis-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabId);
        });

        this.currentTab = tabId;
        await this.loadTabContent(tabId);
    },

    /**
     * Load content for a tab
     */
    async loadTabContent(tabId) {
        const content = document.getElementById('analysis-content');
        if (!content) return;

        content.innerHTML = '<div class="loading"><div class="spinner"></div>Loading...</div>';

        try {
            // Route to appropriate loader based on atlas and tab
            if (this.currentAtlas === 'cima') {
                await this.loadCimaTab(tabId, content);
            } else if (this.currentAtlas === 'inflammation') {
                await this.loadInflammationTab(tabId, content);
            } else if (this.currentAtlas === 'scatlas') {
                await this.loadScatlasTab(tabId, content);
            } else {
                content.innerHTML = '<p>Content not available for this atlas</p>';
            }
        } catch (error) {
            console.error('Failed to load tab content:', error);
            content.innerHTML = `<p class="loading">Failed to load data: ${error.message}</p>`;
        }
    },

    // ==================== CIMA Tab Loaders ====================

    async loadCimaTab(tabId, content) {
        switch (tabId) {
            case 'overview':
                await this.loadCimaOverview(content);
                break;
            case 'celltypes':
                await this.loadCimaCelltypes(content);
                break;
            case 'age-bmi':
                await this.loadCimaAgeBmi(content);
                break;
            case 'age-bmi-stratified':
                await this.loadCimaAgeBmiStratified(content);
                break;
            case 'biochemistry':
                await this.loadCimaBiochemistry(content);
                break;
            case 'biochem-scatter':
                await this.loadCimaBiochemScatter(content);
                break;
            case 'metabolites':
                await this.loadCimaMetabolites(content);
                break;
            case 'differential':
                await this.loadCimaDifferential(content);
                break;
            case 'multiomics':
                await this.loadCimaMultiomics(content);
                break;
            case 'population':
                await this.loadCimaPopulation(content);
                break;
            case 'eqtl':
                await this.loadCimaEqtl(content);
                break;
            default:
                content.innerHTML = '<p>Panel not implemented</p>';
        }
    },

    async loadCimaOverview(content) {
        // Load summary stats
        const stats = await API.get('/cima/summary');

        content.innerHTML = `
            <div class="overview-section">
                <div class="panel-header">
                    <h3>CIMA: Chinese Immune Multi-omics Atlas</h3>
                    <p>Comprehensive immune monitoring atlas of healthy donors with matched blood biochemistry and plasma metabolomics data.</p>
                    <p class="citation" style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">
                        <strong>Citation:</strong> Yin et al. (2026) <em>Science</em>.
                        <a href="https://www.science.org/doi/10.1126/science.adt3130" target="_blank" rel="noopener">DOI: 10.1126/science.adt3130</a>
                    </p>
                </div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">${stats?.n_samples || 421}</div>
                        <div class="stat-label">Healthy Donors</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">6.5M</div>
                        <div class="stat-label">Total Cells</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${stats?.n_cell_types || 27}</div>
                        <div class="stat-label">Cell Types</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">500</div>
                        <div class="stat-label">Metabolites Profiled</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">19</div>
                        <div class="stat-label">Blood Biomarkers</div>
                    </div>
                </div>

                <div class="card" style="margin-top: 1.5rem;">
                    <div class="card-title">Analysis Data Sources</div>
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Analysis</th>
                                <th>Description</th>
                                <th>Records</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Cell Type Activity</td>
                                <td>Mean ${this.signatureType} activity per cell type</td>
                                <td>${this.signatureType === 'CytoSig' ? '27 × 43' : '27 × 1,170'}</td>
                            </tr>
                            <tr>
                                <td>Age Correlations</td>
                                <td>Spearman correlation with donor age</td>
                                <td>${stats?.n_age_correlations || 'N/A'}</td>
                            </tr>
                            <tr>
                                <td>BMI Correlations</td>
                                <td>Spearman correlation with body mass index</td>
                                <td>${stats?.n_bmi_correlations || 'N/A'}</td>
                            </tr>
                            <tr>
                                <td>Biochemistry Correlations</td>
                                <td>Correlation with 19 blood markers</td>
                                <td>${stats?.n_biochem_correlations || 'N/A'}</td>
                            </tr>
                            <tr>
                                <td>Metabolite Correlations</td>
                                <td>Correlation with plasma metabolomics</td>
                                <td>${stats?.n_metabolite_correlations || 'N/A'}</td>
                            </tr>
                            <tr>
                                <td>Differential Analysis</td>
                                <td>Sex, smoking, blood type comparisons</td>
                                <td>${stats?.n_differential_tests || 'N/A'}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

            </div>
        `;
    },

    async loadCimaCelltypes(content) {
        content.innerHTML = `
            <div class="viz-grid">
                <!-- Sub-panel 1: Activity Profile with Search -->
                <div class="sub-panel">
                    <div class="panel-header">
                        <h3>Cell Type Activity Profile</h3>
                        <p>Search and view activity of a specific ${this.signatureType === 'CytoSig' ? 'cytokine' : 'protein'}</p>
                    </div>
                    <div class="search-controls">
                        <input type="text" id="cima-protein-search" class="search-input"
                               placeholder="Search ${this.signatureType === 'CytoSig' ? 'cytokine (e.g., IFNG, IL17A, TNF)' : 'protein'}..."
                               onkeyup="AtlasDetailPage.filterProteinList(this.value, 'cima')">
                        <select id="cima-protein-select" class="filter-select" onchange="AtlasDetailPage.updateCimaActivityProfile()">
                            <option value="">Select ${this.signatureType === 'CytoSig' ? 'cytokine' : 'protein'}...</option>
                        </select>
                    </div>
                    <div id="cima-activity-profile" class="plot-container" style="height: 450px;">
                        <p class="loading">Select a ${this.signatureType === 'CytoSig' ? 'cytokine' : 'protein'} to view its activity profile</p>
                    </div>
                </div>

                <!-- Sub-panel 2: Activity Heatmap -->
                <div class="sub-panel">
                    <div class="panel-header">
                        <h3>Activity Heatmap</h3>
                        <p>Mean ${this.signatureType} activity z-scores across all cell types</p>
                    </div>
                    <div id="cima-celltype-heatmap" class="plot-container" style="height: 500px;"></div>
                </div>
            </div>
        `;

        // Load signatures for dropdown and heatmap data
        const [signatures, heatmapData] = await Promise.all([
            API.get('/cima/signatures', { signature_type: this.signatureType }),
            API.get('/cima/heatmap/activity', { signature_type: this.signatureType }),
        ]);

        // Populate protein dropdown
        const select = document.getElementById('cima-protein-select');
        if (signatures && select) {
            this.cimaSignatures = signatures; // Store for filtering
            select.innerHTML = `<option value="">Select ${this.signatureType === 'CytoSig' ? 'cytokine' : 'protein'}...</option>` +
                signatures.map(s => `<option value="${s}">${s}</option>`).join('');

            // Auto-select first signature
            if (signatures.length > 0) {
                select.value = signatures[0];
                this.updateCimaActivityProfile();
            }
        }

        // Render heatmap
        if (heatmapData && heatmapData.values) {
            const data = {
                z: heatmapData.values,
                cell_types: heatmapData.rows,
                signatures: heatmapData.columns,
            };
            Heatmap.createActivityHeatmap('cima-celltype-heatmap', data, {
                title: `${this.signatureType} Activity by Cell Type`,
            });
        } else {
            document.getElementById('cima-celltype-heatmap').innerHTML = '<p class="loading">No heatmap data available</p>';
        }
    },

    async updateCimaActivityProfile() {
        const protein = document.getElementById('cima-protein-select')?.value;
        const container = document.getElementById('cima-activity-profile');
        if (!container || !protein) return;

        container.innerHTML = '<p class="loading">Loading...</p>';

        try {
            const data = await API.get('/cima/activity', { signature_type: this.signatureType });

            if (data && data.length > 0) {
                // Filter for selected protein
                const proteinData = data.filter(d => d.signature === protein);

                if (proteinData.length > 0) {
                    // Sort by activity
                    proteinData.sort((a, b) => b.mean_activity - a.mean_activity);

                    const cellTypes = proteinData.map(d => d.cell_type);
                    const activities = proteinData.map(d => d.mean_activity);
                    const colors = activities.map(v => v >= 0 ? '#ef4444' : '#2563eb');

                    Plotly.newPlot('cima-activity-profile', [{
                        type: 'bar',
                        x: activities,
                        y: cellTypes,
                        orientation: 'h',
                        marker: { color: colors },
                        hovertemplate: '<b>%{y}</b><br>Activity: %{x:.3f}<extra></extra>',
                    }], {
                        title: `${protein} [${this.signatureType}] Activity Across Cell Types`,
                        xaxis: { title: 'Activity (z-score)', zeroline: true, zerolinecolor: '#888' },
                        yaxis: { title: '', automargin: true },
                        margin: { l: 150, r: 20, t: 40, b: 40 },
                        font: { family: 'Inter, sans-serif' },
                    });
                } else {
                    container.innerHTML = `<p class="loading">No data found for ${protein} [${this.signatureType}]</p>`;
                }
            } else {
                container.innerHTML = '<p class="loading">No activity data available</p>';
            }
        } catch (e) {
            container.innerHTML = `<p class="loading">Error: ${e.message}</p>`;
        }
    },

    filterProteinList(searchText, atlas) {
        const selectId = atlas === 'cima' ? 'cima-protein-select' :
                         atlas === 'inflammation' ? 'inflam-protein-select' : 'scatlas-protein-select';
        const select = document.getElementById(selectId);
        const signatures = atlas === 'cima' ? this.cimaSignatures :
                          atlas === 'inflammation' ? this.inflamSignatures : this.scatlasSignatures;

        if (!select || !signatures) return;

        const filtered = searchText ?
            signatures.filter(s => s.toLowerCase().includes(searchText.toLowerCase())) :
            signatures;

        const currentValue = select.value;
        select.innerHTML = `<option value="">Select ${this.signatureType === 'CytoSig' ? 'cytokine' : 'protein'}...</option>` +
            filtered.map(s => `<option value="${s}">${s}</option>`).join('');

        // Restore selection if still in filtered list
        if (filtered.includes(currentValue)) {
            select.value = currentValue;
        }
    },

    async loadCimaAgeBmi(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Age & BMI Correlations</h3>
                <p>Spearman correlations between ${this.signatureType} activities and donor age/BMI</p>
            </div>

            <!-- Sub-panel 1 & 2: Age and BMI Correlations (lollipop charts) -->
            <div class="viz-grid">
                <div class="sub-panel">
                    <div class="panel-header">
                        <h4>Age Correlations</h4>
                        <p>Top 10 positive and negative correlations with donor age</p>
                    </div>
                    <div id="cima-age-lollipop" class="plot-container" style="height: 450px;"></div>
                </div>
                <div class="sub-panel">
                    <div class="panel-header">
                        <h4>BMI Correlations</h4>
                        <p>Top 10 positive and negative correlations with BMI</p>
                    </div>
                    <div id="cima-bmi-lollipop" class="plot-container" style="height: 450px;"></div>
                </div>
            </div>

            <!-- Sub-panel 3: Cell Type-Specific Correlations -->
            <div class="sub-panel" style="margin-top: var(--spacing-lg);">
                <div class="panel-header">
                    <h4>Cell Type-Specific Correlations</h4>
                    <p>Heatmap showing how activities correlate with Age/BMI <strong>within each cell type</strong></p>
                </div>
                <div class="search-controls">
                    <select id="cima-ct-corr-feature" class="filter-select" onchange="AtlasDetailPage.updateCimaCellTypeCorrelation()">
                        <option value="age">Correlation with Age</option>
                        <option value="bmi">Correlation with BMI</option>
                    </select>
                    <select id="cima-ct-corr-metric" class="filter-select" onchange="AtlasDetailPage.updateCimaCellTypeCorrelation()">
                        <option value="all">All correlations (ρ)</option>
                        <option value="sig">Significant only (q<0.05)</option>
                    </select>
                </div>
                <div id="cima-celltype-corr-heatmap" class="plot-container" style="height: 500px;"></div>
            </div>
        `;

        // Load correlation data
        let ageData = null, bmiData = null;
        try {
            [ageData, bmiData] = await Promise.all([
                API.get('/cima/correlations/age', { signature_type: this.signatureType }),
                API.get('/cima/correlations/bmi', { signature_type: this.signatureType }),
            ]);
            console.log('CIMA Age data:', ageData?.length, 'records', ageData?.[0]);
            console.log('CIMA BMI data:', bmiData?.length, 'records');
        } catch (e) {
            console.error('Failed to load CIMA correlations:', e);
        }

        // Store data for cell type heatmap
        this.cimaAgeCorrelations = ageData;
        this.cimaBmiCorrelations = bmiData;

        // Render lollipop charts
        if (ageData && ageData.length > 0) {
            this.renderCorrelationLollipop('cima-age-lollipop', ageData, 'Age');
        } else {
            console.warn('No CIMA age data - ageData:', ageData);
            document.getElementById('cima-age-lollipop').innerHTML = '<p class="loading">No age correlation data</p>';
        }

        if (bmiData && bmiData.length > 0) {
            this.renderCorrelationLollipop('cima-bmi-lollipop', bmiData, 'BMI');
        } else {
            document.getElementById('cima-bmi-lollipop').innerHTML = '<p class="loading">No BMI correlation data</p>';
        }

        // Render cell type correlation heatmap
        this.updateCimaCellTypeCorrelation();
    },

    renderCorrelationLollipop(containerId, data, title) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Aggregate by signature (average across cell types)
        const sigMap = {};
        data.forEach(d => {
            if (!sigMap[d.signature]) {
                sigMap[d.signature] = { rhos: [], pvals: [] };
            }
            sigMap[d.signature].rhos.push(d.rho);
            sigMap[d.signature].pvals.push(d.p_value || d.pvalue);
        });

        const aggregated = Object.entries(sigMap).map(([sig, vals]) => ({
            signature: sig,
            rho: vals.rhos.reduce((a, b) => a + b, 0) / vals.rhos.length,
            p_value: Math.min(...vals.pvals),
        }));

        // Sort by rho and get top 10 + bottom 10
        aggregated.sort((a, b) => b.rho - a.rho);
        const top10 = aggregated.slice(0, 10);
        const bottom10 = aggregated.slice(-10);
        const combined = [...top10, ...bottom10];

        // Reverse for display (highest at top)
        combined.reverse();

        const y = combined.map(d => d.signature);
        const x = combined.map(d => d.rho);
        const colors = x.map(v => v >= 0 ? '#ef4444' : '#2563eb');

        Plotly.newPlot(containerId, [{
            type: 'bar',
            x: x,
            y: y,
            orientation: 'h',
            marker: { color: colors },
            hovertemplate: '<b>%{y}</b><br>ρ = %{x:.3f}<extra></extra>',
        }], {
            title: `${title} Correlation (${this.signatureType})`,
            xaxis: {
                title: 'Spearman ρ',
                zeroline: true,
                zerolinecolor: '#888',
                range: [-0.5, 0.5],
            },
            yaxis: { title: '', automargin: true, tickfont: { size: 10 } },
            margin: { l: 100, r: 20, t: 40, b: 40 },
            font: { family: 'Inter, sans-serif' },
        });
    },

    updateCimaCellTypeCorrelation() {
        const container = document.getElementById('cima-celltype-corr-heatmap');
        if (!container) return;

        const feature = document.getElementById('cima-ct-corr-feature')?.value || 'age';
        const metric = document.getElementById('cima-ct-corr-metric')?.value || 'all';

        const data = feature === 'age' ? this.cimaAgeCorrelations : this.cimaBmiCorrelations;

        if (!data || data.length === 0) {
            container.innerHTML = '<p class="loading">No correlation data available</p>';
            return;
        }

        // Get unique cell types and signatures
        const cellTypes = [...new Set(data.map(d => d.cell_type))].sort();
        const signatures = [...new Set(data.map(d => d.signature))].sort();

        // Create lookup map
        const lookup = {};
        data.forEach(d => {
            lookup[`${d.cell_type}|${d.signature}`] = d;
        });

        // Build heatmap matrix
        const zValues = [];
        const hoverText = [];
        const sigThreshold = 0.05;

        for (const ct of cellTypes) {
            const row = [];
            const hoverRow = [];
            for (const sig of signatures) {
                const record = lookup[`${ct}|${sig}`];
                if (record) {
                    const qval = record.q_value || record.qvalue;
                    const isSignificant = qval !== undefined ? qval < sigThreshold : record.p_value < sigThreshold;

                    if (metric === 'sig' && !isSignificant) {
                        row.push(null);
                        hoverRow.push(`${ct}<br>${sig}<br>Not significant`);
                    } else {
                        row.push(record.rho);
                        const sigMark = isSignificant ? ' *' : '';
                        hoverRow.push(`<b>${ct}</b><br>${sig}<br>ρ = ${record.rho.toFixed(3)}${sigMark}`);
                    }
                } else {
                    row.push(null);
                    hoverRow.push(`${ct}<br>${sig}<br>No data`);
                }
            }
            zValues.push(row);
            hoverText.push(hoverRow);
        }

        // Count significant
        const nSig = data.filter(d => {
            const qval = d.q_value || d.qvalue;
            return qval !== undefined ? qval < sigThreshold : d.p_value < sigThreshold;
        }).length;

        Plotly.newPlot(container, [{
            z: zValues,
            x: signatures,
            y: cellTypes,
            type: 'heatmap',
            colorscale: 'RdBu',
            reversescale: true,
            zmin: -0.5,
            zmax: 0.5,
            hoverinfo: 'text',
            text: hoverText,
            colorbar: { title: 'ρ', titleside: 'right' },
        }], {
            title: `${feature === 'age' ? 'Age' : 'BMI'} Correlation by Cell Type [${this.signatureType}] (${nSig} significant)`,
            xaxis: { title: 'Signature', tickangle: -45, tickfont: { size: 9 } },
            yaxis: { title: 'Cell Type', tickfont: { size: 10 }, automargin: true },
            margin: { l: 120, r: 50, t: 50, b: 100 },
            font: { family: 'Inter, sans-serif' },
        });
    },

    async loadCimaAgeBmiStratified(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Age/BMI Stratified Activity</h3>
                <p>Activity distribution across age groups and BMI categories, with cell type stratification</p>
            </div>
            <div class="stratified-controls">
                <select id="stratified-variable" class="filter-select" onchange="AtlasDetailPage.updateStratifiedPlot()">
                    <option value="age">Age Groups</option>
                    <option value="bmi">BMI Categories</option>
                </select>
                <select id="cima-strat-celltype" class="filter-select" onchange="AtlasDetailPage.updateStratifiedPlot()">
                    <option value="All">All Cell Types (Sample-level)</option>
                </select>
                <div class="search-controls" style="position: relative;">
                    <input type="text" id="stratified-signature-search" placeholder="Search signature (e.g., IFNG, IL6...)"
                           style="width: 200px; padding: 8px; border: 1px solid #ddd; border-radius: 4px;" autocomplete="off" value="IFNG">
                    <div id="stratified-signature-suggestions" class="suggestions-dropdown"></div>
                </div>
            </div>
            <div class="card" style="margin-bottom: 1rem; padding: 0.75rem; font-size: 0.9rem;">
                <strong>Age Bins:</strong> &lt;30, 30-39, 40-49, 50-59, 60-69, 70+ years<br>
                <strong>BMI Categories:</strong> Underweight (&lt;18.5), Normal (18.5-25), Overweight (25-30), Obese (30+)
            </div>
            <div class="stacked-panels">
                <div class="panel-section">
                    <h4 style="margin: 0 0 0.5rem 0; font-size: 1rem;">Activity Boxplot</h4>
                    <div id="stratified-plot" class="plot-container" style="height: 400px;"></div>
                </div>
                <div class="panel-section" style="margin-top: 1.5rem;">
                    <h4 style="margin: 0 0 0.5rem 0; font-size: 1rem;">Cell Type Heatmap</h4>
                    <div id="stratified-heatmap" class="plot-container" style="height: 500px;"></div>
                </div>
            </div>
        `;

        // Load signatures for autocomplete and cell types for dropdown
        await Promise.all([
            this.loadStratifiedSignatures(),
            this.loadCimaStratifiedCellTypes(),
        ]);

        // Set up search autocomplete
        this.setupStratifiedSignatureSearch();

        // Initial plot
        await this.updateStratifiedPlot();
    },

    async loadCimaStratifiedCellTypes() {
        try {
            const cellTypes = await API.get('/cima/cell-types');
            const select = document.getElementById('cima-strat-celltype');
            if (select && cellTypes && cellTypes.length > 0) {
                select.innerHTML = '<option value="All">All Cell Types (Sample-level)</option>' +
                    cellTypes.map(ct => `<option value="${ct}">${ct}</option>`).join('');
            }
        } catch (e) {
            console.warn('Failed to load CIMA cell types:', e);
        }
    },

    async loadStratifiedSignatures() {
        try {
            const signatures = await API.get('/cima/signatures', { signature_type: this.signatureType });
            this.stratifiedSignatures = signatures || [];
        } catch (e) {
            console.warn('Failed to load stratified signatures:', e);
            this.stratifiedSignatures = [];
        }
    },

    setupStratifiedSignatureSearch() {
        const searchInput = document.getElementById('stratified-signature-search');
        const suggestionsDiv = document.getElementById('stratified-signature-suggestions');
        if (!searchInput || !suggestionsDiv) return;

        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            if (!query) {
                suggestionsDiv.style.display = 'none';
                return;
            }

            const matches = (this.stratifiedSignatures || [])
                .filter(s => s.toLowerCase().includes(query))
                .slice(0, 10);

            if (matches.length > 0) {
                suggestionsDiv.innerHTML = matches.map(s =>
                    `<div class="suggestion-item" onclick="AtlasDetailPage.selectStratifiedSignature('${s}')">${s}</div>`
                ).join('');
                suggestionsDiv.style.display = 'block';
            } else {
                suggestionsDiv.style.display = 'none';
            }
        });

        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                suggestionsDiv.style.display = 'none';
                this.updateStratifiedPlot();
            }
        });

        searchInput.addEventListener('blur', () => {
            setTimeout(() => { suggestionsDiv.style.display = 'none'; }, 200);
        });
    },

    selectStratifiedSignature(signature) {
        const searchInput = document.getElementById('stratified-signature-search');
        if (searchInput) {
            searchInput.value = signature;
        }
        document.getElementById('stratified-signature-suggestions').style.display = 'none';
        this.updateStratifiedPlot();
    },

    async loadCimaBiochemistry(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Biochemistry Correlations</h3>
                <p>Correlation between cytokine activity and blood biochemistry markers</p>
            </div>
            <div id="biochem-heatmap" class="plot-container" style="height: 600px;"></div>
        `;

        const data = await API.get('/cima/correlations/biochemistry', { signature_type: this.signatureType });
        if (data && data.length > 0) {
            this.renderBiochemHeatmap('biochem-heatmap', data);
        } else {
            document.getElementById('biochem-heatmap').innerHTML = '<p class="loading">No biochemistry data available</p>';
        }
    },

    async loadCimaBiochemScatter(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Biochemistry Scatter Plots</h3>
                <p>Individual correlation plots for top associations</p>
            </div>
            <div class="scatter-controls">
                <select id="biochem-marker" class="filter-select" onchange="AtlasDetailPage.updateBiochemScatter()">
                    <option value="">Select Marker</option>
                </select>
                <select id="biochem-signature" class="filter-select" onchange="AtlasDetailPage.updateBiochemScatter()">
                    <option value="IFNG">IFNG</option>
                </select>
            </div>
            <div id="biochem-scatter" class="plot-container" style="height: 500px;"></div>
        `;

        await this.populateBiochemDropdowns();
    },

    async loadCimaMetabolites(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Metabolite Correlations</h3>
                <p>Top correlations with plasma metabolites and lipids (${this.signatureType === 'CytoSig' ? '43' : '1,170'} signatures x 500 metabolites)</p>
            </div>
            <div id="metabolite-heatmap" class="plot-container" style="height: 600px;"></div>
        `;

        const data = await API.get('/cima/correlations/metabolites', { signature_type: this.signatureType, limit: 500 });
        if (data && data.correlations) {
            this.renderMetaboliteHeatmap('metabolite-heatmap', data);
        } else {
            document.getElementById('metabolite-heatmap').innerHTML = '<p class="loading">No metabolite data available</p>';
        }
    },

    async loadCimaDifferential(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Differential Analysis</h3>
                <p>Activity differences by sex, smoking status, and blood type</p>
            </div>
            <div class="differential-controls">
                <select id="diff-comparison" class="filter-select" onchange="AtlasDetailPage.updateDifferentialPlot()">
                    <option value="sex">Sex (Male vs Female)</option>
                    <option value="smoking">Smoking Status</option>
                    <option value="blood_type">Blood Type</option>
                </select>
            </div>
            <div id="differential-volcano" class="plot-container" style="height: 500px;"></div>
        `;

        await this.updateDifferentialPlot();
    },

    async loadCimaMultiomics(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Multi-omics Integration</h3>
                <p>Correlation network between cytokine activity, biochemistry, and metabolites</p>
            </div>
            <div id="multiomics-viz" class="plot-container" style="height: 600px;">
                <p class="loading">Multi-omics network visualization coming soon</p>
            </div>
        `;
    },

    async loadCimaPopulation(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Population Stratification</h3>
                <p>Activity patterns across demographic groups</p>
            </div>
            <div id="population-viz" class="plot-container" style="height: 500px;"></div>
        `;

        // Population stratification endpoint not yet available
        document.getElementById('population-viz').innerHTML = '<p class="loading">Population stratification visualization coming soon</p>';
    },

    async loadCimaEqtl(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>eQTL Browser</h3>
                <p>Genetic variants associated with cytokine activity (cis-eQTLs within 1Mb)</p>
            </div>
            <div class="eqtl-controls">
                <select id="eqtl-signature" class="filter-select" onchange="AtlasDetailPage.updateEqtlPlot()">
                    <option value="IFNG">IFNG</option>
                </select>
                <input type="text" id="eqtl-search" class="filter-select" placeholder="Search gene or SNP..." onchange="AtlasDetailPage.updateEqtlPlot()">
            </div>
            <div id="eqtl-table" style="margin-top: 1rem;"></div>
            <div id="eqtl-plot" class="plot-container" style="height: 400px;"></div>
        `;

        await this.populateEqtlDropdowns();
        await this.updateEqtlPlot();
    },

    // ==================== Inflammation Tab Loaders ====================

    async loadInflammationTab(tabId, content) {
        switch (tabId) {
            case 'overview':
                await this.loadInflamOverview(content);
                break;
            case 'celltypes':
                await this.loadInflamCelltypes(content);
                break;
            case 'age-bmi':
                await this.loadInflamAgeBmi(content);
                break;
            case 'age-bmi-stratified':
                await this.loadInflamAgeBmiStratified(content);
                break;
            case 'disease':
                await this.loadInflamDisease(content);
                break;
            case 'differential':
                await this.loadInflamDifferential(content);
                break;
            case 'treatment':
                await this.loadInflamTreatment(content);
                break;
            case 'sankey':
                await this.loadInflamSankey(content);
                break;
            case 'validation':
                await this.loadInflamValidation(content);
                break;
            case 'longitudinal':
                await this.loadInflamLongitudinal(content);
                break;
            case 'severity':
                await this.loadInflamSeverity(content);
                break;
            case 'drivers':
                await this.loadInflamDrivers(content);
                break;
            default:
                content.innerHTML = '<p>Panel not implemented</p>';
        }
    },

    async loadInflamOverview(content) {
        // Load summary stats
        const stats = await API.get('/inflammation/summary');

        content.innerHTML = `
            <div class="overview-section">
                <div class="panel-header">
                    <h3>Inflammation Atlas</h3>
                    <p>Multi-disease cohort spanning RA, IBD, MS, SLE, and other inflammatory conditions with treatment response data.</p>
                    <p class="citation" style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">
                        <strong>Citation:</strong> Jiménez-Gracia et al. (2026) <em>Nature Medicine</em>.
                    </p>
                </div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">${stats?.n_samples || 817}</div>
                        <div class="stat-label">Patient Samples</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">4.9M</div>
                        <div class="stat-label">Total Cells</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${stats?.n_cell_types || 35}</div>
                        <div class="stat-label">Cell Types</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${stats?.n_diseases || 12}+</div>
                        <div class="stat-label">Diseases Studied</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">3</div>
                        <div class="stat-label">Cohorts (Main/Validation/External)</div>
                    </div>
                </div>

                <div class="card" style="margin-top: 1.5rem;">
                    <div class="card-title">Diseases Covered</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem;">
                        ${(stats?.diseases || ['RA', 'IBD', 'MS', 'SLE', 'COVID-19', 'UC', 'CD', 'PsA', 'SSc', 'Sjögren', 'ANCA Vasculitis', 'JIA']).map(d =>
                            `<span style="background: var(--bg-secondary); padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.875rem;">${d}</span>`
                        ).join('')}
                    </div>
                </div>

                <div class="card" style="margin-top: 1rem;">
                    <div class="card-title">Analysis Data Sources</div>
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Analysis</th>
                                <th>Description</th>
                                <th>Records</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Cell Type Activity</td>
                                <td>Mean ${this.signatureType} activity per cell type and disease</td>
                                <td>${this.signatureType === 'CytoSig' ? '35 × 43' : '35 × 1,170'}</td>
                            </tr>
                            <tr>
                                <td>Disease Differential</td>
                                <td>Disease vs healthy comparison per cell type</td>
                                <td>${stats?.n_diseases || 12} diseases</td>
                            </tr>
                            <tr>
                                <td>Treatment Response</td>
                                <td>Responder vs non-responder prediction</td>
                                <td>4 diseases with treatment data</td>
                            </tr>
                            <tr>
                                <td>Age/BMI Correlations</td>
                                <td>Spearman correlation with clinical parameters</td>
                                <td>Cell type-stratified</td>
                            </tr>
                            <tr>
                                <td>Cross-Cohort Validation</td>
                                <td>Main → Validation → External consistency</td>
                                <td>3 cohorts</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

            </div>
        `;
    },

    async loadInflamCelltypes(content) {
        content.innerHTML = `
            <div class="viz-grid">
                <!-- Sub-panel 1: Activity Profile with Search -->
                <div class="sub-panel">
                    <div class="panel-header">
                        <h3>Cell Type Activity Profile</h3>
                        <p>Search and view activity of a specific ${this.signatureType === 'CytoSig' ? 'cytokine' : 'protein'}</p>
                    </div>
                    <div class="search-controls">
                        <input type="text" id="inflam-protein-search" class="search-input"
                               placeholder="Search ${this.signatureType === 'CytoSig' ? 'cytokine (e.g., IFNG, IL17A, TNF)' : 'protein'}..."
                               onkeyup="AtlasDetailPage.filterProteinList(this.value, 'inflammation')">
                        <select id="inflam-protein-select" class="filter-select" onchange="AtlasDetailPage.updateInflamActivityProfile()">
                            <option value="">Select ${this.signatureType === 'CytoSig' ? 'cytokine' : 'protein'}...</option>
                        </select>
                    </div>
                    <div id="inflam-activity-profile" class="plot-container" style="height: 450px;">
                        <p class="loading">Select a ${this.signatureType === 'CytoSig' ? 'cytokine' : 'protein'} to view its activity profile</p>
                    </div>
                </div>

                <!-- Sub-panel 2: Activity Heatmap -->
                <div class="sub-panel">
                    <div class="panel-header">
                        <h3>Activity Heatmap</h3>
                        <p>Mean ${this.signatureType} activity z-scores across all cell types</p>
                    </div>
                    <div id="inflam-celltype-heatmap" class="plot-container" style="height: 500px;"></div>
                </div>
            </div>
        `;

        // Load signatures for dropdown and heatmap data
        const [signatures, heatmapData] = await Promise.all([
            API.get('/cima/signatures', { signature_type: this.signatureType }), // Use CIMA signatures as reference
            API.get('/inflammation/heatmap/activity', { signature_type: this.signatureType }),
        ]);

        // Populate protein dropdown
        const select = document.getElementById('inflam-protein-select');
        if (signatures && select) {
            this.inflamSignatures = signatures;
            select.innerHTML = `<option value="">Select ${this.signatureType === 'CytoSig' ? 'cytokine' : 'protein'}...</option>` +
                signatures.map(s => `<option value="${s}">${s}</option>`).join('');

            if (signatures.length > 0) {
                select.value = signatures[0];
                this.updateInflamActivityProfile();
            }
        }

        // Render heatmap
        if (heatmapData && heatmapData.values) {
            const data = {
                z: heatmapData.values,
                cell_types: heatmapData.rows,
                signatures: heatmapData.columns,
            };
            Heatmap.createActivityHeatmap('inflam-celltype-heatmap', data, {
                title: `${this.signatureType} Activity by Cell Type`,
            });
        } else {
            document.getElementById('inflam-celltype-heatmap').innerHTML = '<p class="loading">No heatmap data available</p>';
        }
    },

    async updateInflamActivityProfile() {
        const protein = document.getElementById('inflam-protein-select')?.value;
        const container = document.getElementById('inflam-activity-profile');
        if (!container || !protein) return;

        container.innerHTML = '<p class="loading">Loading...</p>';

        try {
            const data = await API.get('/inflammation/activity', { signature_type: this.signatureType });

            if (data && data.length > 0) {
                const proteinData = data.filter(d => d.signature === protein);

                if (proteinData.length > 0) {
                    proteinData.sort((a, b) => b.mean_activity - a.mean_activity);

                    const cellTypes = proteinData.map(d => d.cell_type);
                    const activities = proteinData.map(d => d.mean_activity);
                    const colors = activities.map(v => v >= 0 ? '#ef4444' : '#2563eb');

                    Plotly.newPlot('inflam-activity-profile', [{
                        type: 'bar',
                        x: activities,
                        y: cellTypes,
                        orientation: 'h',
                        marker: { color: colors },
                        hovertemplate: '<b>%{y}</b><br>Activity: %{x:.3f}<extra></extra>',
                    }], {
                        title: `${protein} [${this.signatureType}] Activity Across Cell Types`,
                        xaxis: { title: 'Activity (z-score)', zeroline: true, zerolinecolor: '#888' },
                        yaxis: { title: '', automargin: true },
                        margin: { l: 150, r: 20, t: 40, b: 40 },
                        font: { family: 'Inter, sans-serif' },
                    });
                } else {
                    container.innerHTML = `<p class="loading">No data found for ${protein} [${this.signatureType}]</p>`;
                }
            } else {
                container.innerHTML = '<p class="loading">No activity data available</p>';
            }
        } catch (e) {
            container.innerHTML = `<p class="loading">Error: ${e.message}</p>`;
        }
    },

    async loadInflamAgeBmi(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Age & BMI Correlations</h3>
                <p>Spearman correlations between ${this.signatureType} activities and patient age/BMI</p>
            </div>

            <!-- Sub-panel 1 & 2: Age and BMI Correlations (lollipop charts) -->
            <div class="viz-grid">
                <div class="sub-panel">
                    <div class="panel-header">
                        <h4>Age Correlations</h4>
                        <p>Top 10 positive and negative correlations with patient age</p>
                    </div>
                    <div id="inflam-age-lollipop" class="plot-container" style="height: 450px;"></div>
                </div>
                <div class="sub-panel">
                    <div class="panel-header">
                        <h4>BMI Correlations</h4>
                        <p>Top 10 positive and negative correlations with BMI</p>
                    </div>
                    <div id="inflam-bmi-lollipop" class="plot-container" style="height: 450px;"></div>
                </div>
            </div>

            <!-- Sub-panel 3: Cell Type-Specific Correlations -->
            <div class="sub-panel" style="margin-top: var(--spacing-lg);">
                <div class="panel-header">
                    <h4>Cell Type-Specific Correlations</h4>
                    <p>Heatmap showing how activities correlate with Age/BMI <strong>within each cell type</strong></p>
                </div>
                <div class="search-controls">
                    <select id="inflam-ct-corr-feature" class="filter-select" onchange="AtlasDetailPage.updateInflamCellTypeCorrelation()">
                        <option value="age">Correlation with Age</option>
                        <option value="bmi">Correlation with BMI</option>
                    </select>
                    <select id="inflam-ct-corr-metric" class="filter-select" onchange="AtlasDetailPage.updateInflamCellTypeCorrelation()">
                        <option value="all">All correlations (ρ)</option>
                        <option value="sig">Significant only (q<0.05)</option>
                    </select>
                </div>
                <div id="inflam-celltype-corr-heatmap" class="plot-container" style="height: 500px;"></div>
            </div>
        `;

        // Load correlation data
        const [ageData, bmiData] = await Promise.all([
            API.get('/inflammation/correlations/age', { signature_type: this.signatureType }),
            API.get('/inflammation/correlations/bmi', { signature_type: this.signatureType }),
        ]);

        // Store data for cell type heatmap
        this.inflamAgeCorrelations = ageData;
        this.inflamBmiCorrelations = bmiData;

        // Render lollipop charts
        if (ageData && ageData.length > 0) {
            this.renderCorrelationLollipop('inflam-age-lollipop', ageData, 'Age');
        } else {
            document.getElementById('inflam-age-lollipop').innerHTML = '<p class="loading">No age correlation data</p>';
        }

        if (bmiData && bmiData.length > 0) {
            this.renderCorrelationLollipop('inflam-bmi-lollipop', bmiData, 'BMI');
        } else {
            document.getElementById('inflam-bmi-lollipop').innerHTML = '<p class="loading">No BMI correlation data</p>';
        }

        // Render cell type correlation heatmap
        this.updateInflamCellTypeCorrelation();
    },

    updateInflamCellTypeCorrelation() {
        const container = document.getElementById('inflam-celltype-corr-heatmap');
        if (!container) return;

        const feature = document.getElementById('inflam-ct-corr-feature')?.value || 'age';
        const metric = document.getElementById('inflam-ct-corr-metric')?.value || 'all';

        const data = feature === 'age' ? this.inflamAgeCorrelations : this.inflamBmiCorrelations;

        if (!data || data.length === 0) {
            container.innerHTML = '<p class="loading">No correlation data available</p>';
            return;
        }

        // Get unique cell types and signatures
        const cellTypes = [...new Set(data.map(d => d.cell_type))].sort();
        const signatures = [...new Set(data.map(d => d.signature))].sort();

        // Create lookup map
        const lookup = {};
        data.forEach(d => {
            lookup[`${d.cell_type}|${d.signature}`] = d;
        });

        // Build heatmap matrix
        const zValues = [];
        const hoverText = [];
        const sigThreshold = 0.05;

        for (const ct of cellTypes) {
            const row = [];
            const hoverRow = [];
            for (const sig of signatures) {
                const record = lookup[`${ct}|${sig}`];
                if (record) {
                    const qval = record.q_value || record.qvalue;
                    const isSignificant = qval !== undefined ? qval < sigThreshold : record.p_value < sigThreshold;

                    if (metric === 'sig' && !isSignificant) {
                        row.push(null);
                        hoverRow.push(`${ct}<br>${sig}<br>Not significant`);
                    } else {
                        row.push(record.rho);
                        const sigMark = isSignificant ? ' *' : '';
                        hoverRow.push(`<b>${ct}</b><br>${sig}<br>ρ = ${record.rho.toFixed(3)}${sigMark}`);
                    }
                } else {
                    row.push(null);
                    hoverRow.push(`${ct}<br>${sig}<br>No data`);
                }
            }
            zValues.push(row);
            hoverText.push(hoverRow);
        }

        // Count significant
        const nSig = data.filter(d => {
            const qval = d.q_value || d.qvalue;
            return qval !== undefined ? qval < sigThreshold : d.p_value < sigThreshold;
        }).length;

        Plotly.newPlot(container, [{
            z: zValues,
            x: signatures,
            y: cellTypes,
            type: 'heatmap',
            colorscale: 'RdBu',
            reversescale: true,
            zmin: -0.5,
            zmax: 0.5,
            hoverinfo: 'text',
            text: hoverText,
            colorbar: { title: 'ρ', titleside: 'right' },
        }], {
            title: `${feature === 'age' ? 'Age' : 'BMI'} Correlation by Cell Type [${this.signatureType}] (${nSig} significant)`,
            xaxis: { title: 'Signature', tickangle: -45, tickfont: { size: 9 } },
            yaxis: { title: 'Cell Type', tickfont: { size: 10 }, automargin: true },
            margin: { l: 120, r: 50, t: 50, b: 100 },
            font: { family: 'Inter, sans-serif' },
        });
    },

    async loadInflamAgeBmiStratified(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Age/BMI Stratified Activity</h3>
                <p>Activity distribution across age groups and BMI categories, with cell type stratification</p>
            </div>
            <div class="stratified-controls">
                <select id="inflam-strat-variable" class="filter-select" onchange="AtlasDetailPage.updateInflamStratifiedPlot()">
                    <option value="age">Age Groups</option>
                    <option value="bmi">BMI Categories</option>
                </select>
                <select id="inflam-strat-celltype" class="filter-select" onchange="AtlasDetailPage.updateInflamStratifiedPlot()">
                    <option value="All">All Cell Types (Aggregated)</option>
                </select>
                <div class="search-controls" style="position: relative;">
                    <input type="text" id="inflam-strat-signature-search" placeholder="Search signature (e.g., IFNG, IL6...)"
                           style="width: 200px; padding: 8px; border: 1px solid #ddd; border-radius: 4px;" autocomplete="off" value="IFNG">
                    <div id="inflam-strat-signature-suggestions" class="suggestions-dropdown"></div>
                </div>
            </div>
            <div class="card" style="margin-bottom: 1rem; padding: 0.75rem; font-size: 0.9rem;">
                <strong>Age Bins:</strong> &lt;30, 30-39, 40-49, 50-59, 60-69, 70+ years<br>
                <strong>BMI Categories:</strong> Underweight (&lt;18.5), Normal (18.5-25), Overweight (25-30), Obese (30+)
            </div>
            <div class="stacked-panels">
                <div class="panel-section">
                    <h4 style="margin: 0 0 0.5rem 0; font-size: 1rem;">Activity Boxplot</h4>
                    <div id="inflam-stratified-plot" class="plot-container" style="height: 400px;"></div>
                </div>
                <div class="panel-section" style="margin-top: 1.5rem;">
                    <h4 style="margin: 0 0 0.5rem 0; font-size: 1rem;">Cell Type Heatmap</h4>
                    <div id="inflam-stratified-heatmap" class="plot-container" style="height: 500px;"></div>
                </div>
            </div>
        `;

        // Load signatures for autocomplete and cell types for dropdown
        await Promise.all([
            this.loadInflamStratifiedSignatures(),
            this.loadInflamStratifiedCellTypes(),
        ]);

        // Set up search autocomplete
        this.setupInflamStratifiedSignatureSearch();

        // Initial plot
        await this.updateInflamStratifiedPlot();
    },

    async loadInflamStratifiedCellTypes() {
        try {
            const cellTypes = await API.get('/inflammation/cell-types');
            const select = document.getElementById('inflam-strat-celltype');
            if (select && cellTypes && cellTypes.length > 0) {
                // Keep "All" as first option, then add specific cell types
                select.innerHTML = '<option value="All">All Cell Types (Aggregated)</option>' +
                    cellTypes.map(ct => `<option value="${ct}">${ct}</option>`).join('');
            }
        } catch (e) {
            console.warn('Failed to load inflammation cell types:', e);
        }
    },

    async loadInflamStratifiedSignatures() {
        try {
            const signatures = await API.get('/inflammation/signatures', { signature_type: this.signatureType });
            this.inflamStratifiedSignatures = signatures || [];
        } catch (e) {
            console.warn('Failed to load inflammation stratified signatures:', e);
            this.inflamStratifiedSignatures = [];
        }
    },

    setupInflamStratifiedSignatureSearch() {
        const searchInput = document.getElementById('inflam-strat-signature-search');
        const suggestionsDiv = document.getElementById('inflam-strat-signature-suggestions');
        if (!searchInput || !suggestionsDiv) return;

        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            if (!query) {
                suggestionsDiv.style.display = 'none';
                return;
            }

            const matches = (this.inflamStratifiedSignatures || [])
                .filter(s => s.toLowerCase().includes(query))
                .slice(0, 10);

            if (matches.length > 0) {
                suggestionsDiv.innerHTML = matches.map(s =>
                    `<div class="suggestion-item" onclick="AtlasDetailPage.selectInflamStratifiedSignature('${s}')">${s}</div>`
                ).join('');
                suggestionsDiv.style.display = 'block';
            } else {
                suggestionsDiv.style.display = 'none';
            }
        });

        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                suggestionsDiv.style.display = 'none';
                this.updateInflamStratifiedPlot();
            }
        });

        searchInput.addEventListener('blur', () => {
            setTimeout(() => { suggestionsDiv.style.display = 'none'; }, 200);
        });
    },

    selectInflamStratifiedSignature(signature) {
        const searchInput = document.getElementById('inflam-strat-signature-search');
        if (searchInput) {
            searchInput.value = signature;
        }
        document.getElementById('inflam-strat-signature-suggestions').style.display = 'none';
        this.updateInflamStratifiedPlot();
    },

    async loadInflamDisease(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Disease-Specific Activity</h3>
                <p>Cytokine activity patterns across inflammatory diseases</p>
            </div>
            <div class="disease-controls">
                <select id="disease-select" class="filter-select" onchange="AtlasDetailPage.updateDiseaseHeatmap()">
                    <option value="">All Diseases</option>
                </select>
            </div>
            <div id="disease-heatmap" class="plot-container" style="height: 600px;"></div>
        `;

        await this.populateDiseaseDropdown();
        await this.updateDiseaseHeatmap();
    },

    async loadInflamDifferential(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Disease Differential Analysis</h3>
                <p>Volcano plot showing disease vs healthy activity changes</p>
            </div>
            <div class="differential-controls">
                <select id="inflam-diff-disease" class="filter-select" onchange="AtlasDetailPage.updateInflamDifferential()">
                    <option value="">Select Disease</option>
                </select>
            </div>
            <div id="inflam-volcano" class="plot-container" style="height: 500px;"></div>
        `;

        await this.populateDiseaseDropdown('inflam-diff-disease');
        await this.updateInflamDifferential();
    },

    async loadInflamTreatment(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Treatment Response Prediction</h3>
                <p>ROC curves from cross-validated response prediction models</p>
            </div>
            <div class="treatment-controls">
                <select id="treatment-disease" class="filter-select" onchange="AtlasDetailPage.updateTreatmentResponse()">
                    <option value="">All Diseases</option>
                </select>
            </div>
            <div id="treatment-roc" class="plot-container" style="height: 500px;"></div>
            <div id="treatment-features" class="plot-container" style="height: 400px;"></div>
        `;

        await this.updateTreatmentResponse();
    },

    async loadInflamSankey(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Disease Flow Diagram</h3>
                <p>Sample distribution across disease categories and treatment outcomes</p>
            </div>
            <div id="disease-sankey" class="plot-container" style="height: 600px;"></div>
        `;

        const data = await API.get('/inflammation/sankey');
        if (data) {
            this.renderSankeyDiagram('disease-sankey', data);
        } else {
            document.getElementById('disease-sankey').innerHTML = '<p class="loading">No Sankey data available</p>';
        }
    },

    async loadInflamValidation(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Cross-Cohort Validation</h3>
                <p>Consistency of findings across main, validation, and external cohorts</p>
            </div>
            <div id="cohort-validation" class="plot-container" style="height: 500px;"></div>
        `;

        const data = await API.get('/inflammation/cohort-validation', { signature_type: this.signatureType });
        if (data) {
            this.renderCohortValidation('cohort-validation', data);
        } else {
            document.getElementById('cohort-validation').innerHTML = '<p class="loading">No validation data available</p>';
        }
    },

    async loadInflamLongitudinal(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Longitudinal Analysis</h3>
                <p>Activity changes over time for patients with multiple timepoints</p>
            </div>
            <div id="longitudinal-plot" class="plot-container" style="height: 500px;">
                <p class="loading">Longitudinal analysis coming soon</p>
            </div>
        `;
    },

    async loadInflamSeverity(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Disease Severity Correlation</h3>
                <p>Correlation between cytokine activity and disease severity scores</p>
            </div>
            <div id="severity-plot" class="plot-container" style="height: 500px;">
                <p class="loading">Severity correlation coming soon</p>
            </div>
        `;
    },

    async loadInflamDrivers(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Cell Type Drivers</h3>
                <p>Identification of cell populations driving disease-specific signatures</p>
            </div>
            <div id="drivers-plot" class="plot-container" style="height: 500px;">
                <p class="loading">Cell drivers analysis coming soon</p>
            </div>
        `;
    },

    // ==================== scAtlas Tab Loaders ====================

    async loadScatlasTab(tabId, content) {
        switch (tabId) {
            case 'overview':
                await this.loadScatlasOverview(content);
                break;
            case 'celltypes':
                await this.loadScatlasCelltypes(content);
                break;
            case 'organ-map':
                await this.loadScatlasOrganMap(content);
                break;
            case 'cancer-comparison':
                await this.loadScatlasCancerComparison(content);
                break;
            case 'cancer-types':
                await this.loadScatlasCancerTypes(content);
                break;
            case 'immune-infiltration':
                await this.loadScatlasImmuneInfiltration(content);
                break;
            case 'exhaustion':
                await this.loadScatlasExhaustion(content);
                break;
            case 'caf':
                await this.loadScatlasCaf(content);
                break;
            case 'organ-cancer-matrix':
                await this.loadScatlasOrganCancerMatrix(content);
                break;
            case 'adjacent-tissue':
                await this.loadScatlasAdjacentTissue(content);
                break;
            default:
                content.innerHTML = '<p>Panel not implemented</p>';
        }
    },

    async loadScatlasOverview(content) {
        // Load summary stats
        const stats = await API.get('/scatlas/summary');

        content.innerHTML = `
            <div class="overview-section">
                <div class="panel-header">
                    <h3>scAtlas: Human Tissue Reference Atlas</h3>
                    <p>Human Cell Atlas data spanning normal organs and pan-cancer datasets with detailed cell type annotations.</p>
                    <p class="citation" style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">
                        <strong>Citation:</strong> Shi et al. (2025) Cross-tissue multicellular coordination and its rewiring in cancer. <em>Nature</em>.
                    </p>
                </div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">${stats?.n_normal_donors || 145} / ${stats?.n_cancer_donors || 325}</div>
                        <div class="stat-label">Donors (Normal / Cancer)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${stats?.n_normal_cells ? (stats.n_normal_cells / 1e6).toFixed(1) + 'M' : '2.8M'} / ${stats?.n_cancer_cells ? (stats.n_cancer_cells / 1e6).toFixed(1) + 'M' : '3.6M'}</div>
                        <div class="stat-label">Total Cells (Normal / Cancer)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${stats?.n_organs || 35}</div>
                        <div class="stat-label">Organs Profiled</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${stats?.n_normal_celltypes || 192} / ${stats?.n_cancer_celltypes || 214}</div>
                        <div class="stat-label">Cell Types (Normal / Cancer)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${stats?.n_cancer_types || 25}</div>
                        <div class="stat-label">Cancer Types</div>
                    </div>
                </div>

                <div class="card" style="margin-top: 1.5rem;">
                    <div class="card-title">Organs Covered</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; max-height: 150px; overflow-y: auto;">
                        ${(stats?.organs || ['Lung', 'Liver', 'Heart', 'Kidney', 'Brain', 'Pancreas', 'Colon', 'Skin', 'Spleen', 'Bone Marrow', 'Thymus', 'Lymph Node', 'Blood', 'Breast', 'Prostate', 'Ovary', 'Stomach', 'Small Intestine', 'Bladder', 'Esophagus']).map(o =>
                            `<span style="background: var(--bg-secondary); padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.875rem;">${o}</span>`
                        ).join('')}
                    </div>
                </div>

                <div class="card" style="margin-top: 1rem;">
                    <div class="card-title">Analysis Data Sources</div>
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Analysis</th>
                                <th>Description</th>
                                <th>Records</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Normal Organ Signatures</td>
                                <td>Mean ${this.signatureType} activity per organ and cell type</td>
                                <td>${stats?.n_organs || 35} organs</td>
                            </tr>
                            <tr>
                                <td>Cancer Cell Types</td>
                                <td>Activity in tumor-infiltrating immune cells</td>
                                <td>${stats?.n_cancer_celltypes || 214} cell types</td>
                            </tr>
                            <tr>
                                <td>Tumor vs Adjacent</td>
                                <td>Paired tumor-normal tissue comparison</td>
                                <td>${stats?.n_cancer_types || 25} cancer types</td>
                            </tr>
                            <tr>
                                <td>Immune Infiltration</td>
                                <td>Cytokine activity by immune subset in tumors</td>
                                <td>Per cancer type</td>
                            </tr>
                            <tr>
                                <td>T Cell Exhaustion</td>
                                <td>Exhaustion signature across tumor types</td>
                                <td>Exhaustion markers</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

            </div>
        `;
    },

    async loadScatlasOrganMap(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Organ-Specific Activity</h3>
                <p>Mean cytokine activity across 35 human organs</p>
            </div>
            <div class="organ-controls">
                <select id="organ-signature" class="filter-select" onchange="AtlasDetailPage.updateOrganMap()">
                    <option value="IFNG">IFNG</option>
                </select>
            </div>
            <div id="organ-bar" class="plot-container" style="height: 500px;"></div>
            <div id="organ-heatmap" class="plot-container" style="height: 600px;"></div>
        `;

        await this.populateSignatureDropdown('organ-signature');
        await this.updateOrganMap();
    },

    async loadScatlasCelltypes(content) {
        content.innerHTML = `
            <div class="viz-grid">
                <!-- Sub-panel 1: Activity Profile with Search -->
                <div class="sub-panel">
                    <div class="panel-header">
                        <h3>Cell Type Activity Profile</h3>
                        <p>Search and view activity of a specific ${this.signatureType === 'CytoSig' ? 'cytokine' : 'protein'}</p>
                    </div>
                    <div class="search-controls">
                        <input type="text" id="scatlas-protein-search" class="search-input"
                               placeholder="Search ${this.signatureType === 'CytoSig' ? 'cytokine (e.g., IFNG, IL17A, TNF)' : 'protein'}..."
                               onkeyup="AtlasDetailPage.filterProteinList(this.value, 'scatlas')">
                        <select id="scatlas-protein-select" class="filter-select" onchange="AtlasDetailPage.updateScatlasActivityProfile()">
                            <option value="">Select ${this.signatureType === 'CytoSig' ? 'cytokine' : 'protein'}...</option>
                        </select>
                    </div>
                    <div id="scatlas-activity-profile" class="plot-container" style="height: 450px;">
                        <p class="loading">Select a ${this.signatureType === 'CytoSig' ? 'cytokine' : 'protein'} to view its activity profile</p>
                    </div>
                </div>

                <!-- Sub-panel 2: Activity Heatmap -->
                <div class="sub-panel">
                    <div class="panel-header">
                        <h3>Activity Heatmap</h3>
                        <p>Mean ${this.signatureType} activity z-scores (top 50 cell types)</p>
                    </div>
                    <div id="scatlas-celltype-heatmap" class="plot-container" style="height: 500px;"></div>
                </div>
            </div>
        `;

        // Load signatures for dropdown and heatmap data
        const [signatures, heatmapData] = await Promise.all([
            API.get('/cima/signatures', { signature_type: this.signatureType }), // Use CIMA signatures as reference
            API.get('/scatlas/heatmap/celltype', { signature_type: this.signatureType }),
        ]);

        // Populate protein dropdown
        const select = document.getElementById('scatlas-protein-select');
        if (signatures && select) {
            this.scatlasSignatures = signatures;
            select.innerHTML = `<option value="">Select ${this.signatureType === 'CytoSig' ? 'cytokine' : 'protein'}...</option>` +
                signatures.map(s => `<option value="${s}">${s}</option>`).join('');

            if (signatures.length > 0) {
                select.value = signatures[0];
                this.updateScatlasActivityProfile();
            }
        }

        // Render heatmap
        if (heatmapData && heatmapData.values) {
            const data = {
                z: heatmapData.values,
                cell_types: heatmapData.rows,
                signatures: heatmapData.columns,
            };
            Heatmap.createActivityHeatmap('scatlas-celltype-heatmap', data, {
                title: `${this.signatureType} Activity by Cell Type`,
            });
        } else {
            document.getElementById('scatlas-celltype-heatmap').innerHTML = '<p class="loading">No heatmap data available</p>';
        }
    },

    async updateScatlasActivityProfile() {
        const protein = document.getElementById('scatlas-protein-select')?.value;
        const container = document.getElementById('scatlas-activity-profile');
        if (!container || !protein) return;

        container.innerHTML = '<p class="loading">Loading...</p>';

        try {
            // Get cell type signatures data
            const response = await API.get('/scatlas/celltype-signatures', { signature_type: this.signatureType });

            if (response && response.data && response.data.length > 0) {
                const proteinData = response.data.filter(d => d.signature === protein);

                if (proteinData.length > 0) {
                    proteinData.sort((a, b) => b.mean_activity - a.mean_activity);

                    // Take top 30 for readability
                    const topData = proteinData.slice(0, 30);
                    const cellTypes = topData.map(d => d.cell_type);
                    const activities = topData.map(d => d.mean_activity);
                    const colors = activities.map(v => v >= 0 ? '#ef4444' : '#2563eb');

                    Plotly.newPlot('scatlas-activity-profile', [{
                        type: 'bar',
                        x: activities,
                        y: cellTypes,
                        orientation: 'h',
                        marker: { color: colors },
                        hovertemplate: '<b>%{y}</b><br>Activity: %{x:.3f}<extra></extra>',
                    }], {
                        title: `${protein} [${this.signatureType}] Activity (Top 30 Cell Types)`,
                        xaxis: { title: 'Activity (z-score)', zeroline: true, zerolinecolor: '#888' },
                        yaxis: { title: '', automargin: true },
                        margin: { l: 150, r: 20, t: 40, b: 40 },
                        font: { family: 'Inter, sans-serif' },
                    });
                } else {
                    container.innerHTML = `<p class="loading">No data found for ${protein} [${this.signatureType}]</p>`;
                }
            } else {
                container.innerHTML = '<p class="loading">No activity data available</p>';
            }
        } catch (e) {
            container.innerHTML = `<p class="loading">Error: ${e.message}</p>`;
        }
    },

    async loadScatlasCancerComparison(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Tumor vs Adjacent Normal</h3>
                <p>Activity changes in tumor microenvironment compared to adjacent normal tissue</p>
            </div>
            <div id="cancer-comparison-plot" class="plot-container" style="height: 600px;"></div>
        `;

        const data = await API.get('/scatlas/heatmap/cancer-comparison', { signature_type: this.signatureType });
        if (data && data.values) {
            // Transform API response to heatmap component format
            Heatmap.create('cancer-comparison-plot', {
                z: data.values,
                x: data.columns,
                y: data.rows,
            }, {
                title: `Tumor vs Adjacent (${data.n_paired_donors || 0} paired donors)`,
                colorbarTitle: 'Mean Difference',
                symmetric: true,
                xLabel: 'Signature',
                yLabel: 'Cell Type',
            });
        } else {
            document.getElementById('cancer-comparison-plot').innerHTML = '<p class="loading">No cancer comparison data available</p>';
        }
    },

    async loadScatlasCancerTypes(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Cancer Type Signatures</h3>
                <p>Activity patterns across different cancer types</p>
            </div>
            <div id="cancer-types-heatmap" class="plot-container" style="height: 600px;">
                <p class="loading">Cancer types heatmap coming soon</p>
            </div>
        `;
    },

    async loadScatlasImmuneInfiltration(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Immune Infiltration</h3>
                <p>Immune cell composition in tumor microenvironment</p>
            </div>
            <div id="immune-infiltration-plot" class="plot-container" style="height: 500px;">
                <p class="loading">Immune infiltration analysis coming soon</p>
            </div>
        `;
    },

    async loadScatlasExhaustion(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>T Cell Exhaustion</h3>
                <p>Exhaustion signatures in tumor-infiltrating T cells</p>
            </div>
            <div id="exhaustion-plot" class="plot-container" style="height: 500px;">
                <p class="loading">T cell exhaustion analysis coming soon</p>
            </div>
        `;
    },

    async loadScatlasCaf(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Cancer-Associated Fibroblast Types</h3>
                <p>CAF classification and cytokine signatures</p>
            </div>
            <div id="caf-plot" class="plot-container" style="height: 500px;">
                <p class="loading">CAF classification coming soon</p>
            </div>
        `;
    },

    async loadScatlasOrganCancerMatrix(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Organ-Cancer Matrix</h3>
                <p>Cross-tabulation of cytokine patterns by organ and cancer type</p>
            </div>
            <div id="organ-cancer-matrix" class="plot-container" style="height: 600px;">
                <p class="loading">Organ-cancer matrix coming soon</p>
            </div>
        `;
    },

    async loadScatlasAdjacentTissue(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Adjacent Tissue Analysis</h3>
                <p>Comparison of tumor-adjacent tissue with matched normal</p>
            </div>
            <div id="adjacent-tissue-plot" class="plot-container" style="height: 500px;">
                <p class="loading">Adjacent tissue analysis coming soon</p>
            </div>
        `;
    },

    // ==================== Helper Functions ====================

    renderCorrelationHeatmap(containerId, data, title) {
        // Group data by cell type and signature
        const cellTypes = [...new Set(data.map(d => d.cell_type))];
        const signatures = [...new Set(data.map(d => d.signature || d.protein))];

        const z = cellTypes.map(ct =>
            signatures.map(sig => {
                const item = data.find(d => d.cell_type === ct && (d.signature === sig || d.protein === sig));
                return item ? item.correlation : 0;
            })
        );

        Heatmap.create(containerId, {
            z, x: signatures, y: cellTypes,
            colorscale: 'RdBu', reversescale: true,
        }, {
            title,
            xLabel: 'Signature',
            yLabel: 'Cell Type',
            colorbarTitle: 'Correlation (r)',
            symmetric: true,
        });
    },

    renderBiochemHeatmap(containerId, data) {
        // Group by signature and biochem marker
        const signatures = [...new Set(data.map(d => d.signature || d.protein))];
        const markers = [...new Set(data.map(d => d.variable || d.marker))];

        const z = signatures.map(sig =>
            markers.map(m => {
                const item = data.find(d => (d.signature === sig || d.protein === sig) && (d.variable === m || d.marker === m));
                return item ? item.correlation : 0;
            })
        );

        Heatmap.create(containerId, {
            z, x: markers, y: signatures,
            colorscale: 'RdBu', reversescale: true,
        }, {
            title: 'Biochemistry Correlations',
            xLabel: 'Biochemistry Marker',
            yLabel: 'Signature',
            colorbarTitle: 'Correlation (r)',
            symmetric: true,
        });
    },

    renderMetaboliteHeatmap(containerId, data) {
        if (data.z) {
            Heatmap.create(containerId, data, {
                title: 'Top Metabolite Correlations',
                colorbarTitle: 'Correlation (r)',
                symmetric: true,
            });
        }
    },

    renderBoxplotFromStats(containerId, data, options = {}) {
        // Render boxplot from pre-computed statistics
        // New format: data is array of {signature, signature_type, bin, min, q1, median, q3, max, mean, n}
        const container = document.getElementById(containerId);
        if (!container || !data || data.length === 0) {
            if (container) container.innerHTML = '<p class="loading">No data available</p>';
            return;
        }

        // Data is now flat - each record is one bin's statistics
        // Sort bins in proper order
        const binOrder = {
            // Age bins
            '<30': 0, '30-39': 1, '40-49': 2, '50-59': 3, '60-69': 4, '70+': 5,
            // BMI bins
            'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3, 'Obese I': 3, 'Obese II+': 4,
        };
        const sortedData = [...data].sort((a, b) => (binOrder[a.bin] ?? 99) - (binOrder[b.bin] ?? 99));

        // Build single box plot trace with all bins
        // Using arrays for x, lowerfence, q1, median, q3, upperfence
        const xLabels = sortedData.map(stat => stat.bin);
        const lowerfences = sortedData.map(stat => stat.min);
        const q1s = sortedData.map(stat => stat.q1);
        const medians = sortedData.map(stat => stat.median);
        const q3s = sortedData.map(stat => stat.q3);
        const upperfences = sortedData.map(stat => stat.max);
        const means = sortedData.map(stat => stat.mean);
        const ns = sortedData.map(stat => stat.n);

        // Create custom hover text
        const hoverText = sortedData.map(stat =>
            `<b>${stat.bin}</b><br>` +
            `Median: ${stat.median?.toFixed(3)}<br>` +
            `Q1: ${stat.q1?.toFixed(3)}<br>` +
            `Q3: ${stat.q3?.toFixed(3)}<br>` +
            `Mean: ${stat.mean?.toFixed(3)}<br>` +
            `n=${stat.n}`
        );

        const trace = {
            type: 'box',
            x: xLabels,
            lowerfence: lowerfences,
            q1: q1s,
            median: medians,
            q3: q3s,
            upperfence: upperfences,
            mean: means,
            boxpoints: false,
            hoverinfo: 'text',
            hovertext: hoverText,
            marker: { color: '#3b82f6' },
            fillcolor: 'rgba(59, 130, 246, 0.5)',
        };

        Plotly.newPlot(containerId, [trace], {
            title: options.title || 'Activity by Group',
            xaxis: {
                title: options.xLabel || 'Group',
                type: 'category',
                categoryorder: 'array',
                categoryarray: xLabels,
            },
            yaxis: { title: options.yLabel || 'Value' },
            showlegend: false,
            font: { family: 'Inter, sans-serif' },
            margin: { t: 50, b: 60 },
        });
    },

    renderPopulationViz(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container) return;

        if (data.groups && data.values) {
            Scatter.createBoxPlot(containerId, data, {
                title: 'Population Stratification',
                yLabel: 'Activity (z-score)',
            });
        } else {
            container.innerHTML = '<p class="loading">No population data</p>';
        }
    },

    renderSankeyDiagram(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container || !data.nodes || !data.links) {
            container.innerHTML = '<p class="loading">No Sankey data</p>';
            return;
        }

        const trace = {
            type: 'sankey',
            orientation: 'h',
            node: {
                pad: 15,
                thickness: 20,
                line: { color: 'black', width: 0.5 },
                label: data.nodes.map(n => n.name || n),
                color: data.nodes.map((n, i) => `hsl(${(i * 30) % 360}, 70%, 50%)`),
            },
            link: {
                source: data.links.map(l => l.source),
                target: data.links.map(l => l.target),
                value: data.links.map(l => l.value),
            },
        };

        Plotly.newPlot(containerId, [trace], {
            title: 'Disease Flow',
            font: { family: 'Inter, sans-serif' },
        });
    },

    renderCohortValidation(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container) return;

        if (data.cohorts && data.correlations) {
            Plotly.newPlot(containerId, [{
                x: data.cohorts,
                y: data.correlations,
                type: 'bar',
                marker: { color: '#2563eb' },
            }], {
                title: 'Cross-Cohort Correlation',
                xaxis: { title: 'Cohort Pair' },
                yaxis: { title: 'Correlation (r)', range: [0, 1] },
            });
        } else {
            container.innerHTML = '<p class="loading">No validation data</p>';
        }
    },

    async populateStratifiedDropdowns() {
        try {
            const [cellTypes, signatures] = await Promise.all([
                API.get('/cima/cell-types'),
                API.get('/cima/signatures', { signature_type: this.signatureType }),
            ]);

            const ctSelect = document.getElementById('stratified-celltype');
            const sigSelect = document.getElementById('stratified-signature');

            if (cellTypes && ctSelect) {
                cellTypes.forEach(ct => {
                    ctSelect.innerHTML += `<option value="${ct}">${ct}</option>`;
                });
            }

            if (signatures && sigSelect) {
                sigSelect.innerHTML = signatures.map(s => `<option value="${s}">${s}</option>`).join('');
            }
        } catch (e) {
            console.warn('Failed to populate stratified dropdowns:', e);
        }
    },

    async populateSignatureDropdown(selectId) {
        try {
            const signatures = await API.get(`/${this.currentAtlas}/signatures`, { signature_type: this.signatureType });
            const select = document.getElementById(selectId);
            if (signatures && select) {
                select.innerHTML = signatures.map(s => `<option value="${s}">${s}</option>`).join('');
            }
        } catch (e) {
            console.warn('Failed to populate signature dropdown:', e);
        }
    },

    async populateDiseaseDropdown(selectId = 'disease-select') {
        try {
            const diseases = await API.get('/inflammation/diseases');
            const select = document.getElementById(selectId);
            if (diseases && select) {
                select.innerHTML = '<option value="">All Diseases</option>' +
                    diseases.map(d => `<option value="${d}">${d}</option>`).join('');
            }
        } catch (e) {
            console.warn('Failed to populate disease dropdown:', e);
        }
    },

    async populateBiochemDropdowns() {
        try {
            const markers = await API.get('/cima/biochem-variables');
            const signatures = await API.get('/cima/signatures', { signature_type: this.signatureType });

            const markerSelect = document.getElementById('biochem-marker');
            const sigSelect = document.getElementById('biochem-signature');

            if (markers && markerSelect) {
                markerSelect.innerHTML = '<option value="">Select Marker</option>' +
                    markers.map(m => `<option value="${m}">${m}</option>`).join('');
            }
            if (signatures && sigSelect) {
                sigSelect.innerHTML = signatures.map(s => `<option value="${s}">${s}</option>`).join('');
            }
        } catch (e) {
            console.warn('Failed to populate biochem dropdowns:', e);
        }
    },

    async populateEqtlDropdowns() {
        try {
            const signatures = await API.get('/cima/signatures', { signature_type: 'CytoSig' });
            const select = document.getElementById('eqtl-signature');
            if (signatures && select) {
                select.innerHTML = signatures.map(s => `<option value="${s}">${s}</option>`).join('');
            }
        } catch (e) {
            console.warn('Failed to populate eQTL dropdowns:', e);
        }
    },

    // Update functions for interactive controls
    async updateStratifiedPlot() {
        const variable = document.getElementById('stratified-variable')?.value || 'age';
        const signature = document.getElementById('stratified-signature-search')?.value || 'IFNG';
        const cellType = document.getElementById('cima-strat-celltype')?.value || 'All';

        const plotContainer = document.getElementById('stratified-plot');
        const heatmapContainer = document.getElementById('stratified-heatmap');

        if (!signature) {
            if (plotContainer) plotContainer.innerHTML = '<p class="loading">Enter a signature name to view distribution</p>';
            if (heatmapContainer) heatmapContainer.innerHTML = '';
            return;
        }

        const endpoint = variable === 'age' ? 'age' : 'bmi';

        // Load boxplot
        if (plotContainer) {
            try {
                plotContainer.innerHTML = '<p class="loading">Loading...</p>';

                const params = { signature_type: this.signatureType };
                if (cellType && cellType !== 'All') {
                    params.cell_type = cellType;
                }

                const data = await API.get(`/cima/boxplots/${endpoint}/${signature}`, params);

                if (data && data.length > 0) {
                    const titleCellType = cellType === 'All' ? '' : ` (${cellType})`;
                    this.renderBoxplotFromStats('stratified-plot', data, {
                        title: `${signature} [${this.signatureType}] Activity by ${variable === 'age' ? 'Age Group' : 'BMI Category'}${titleCellType}`,
                        yLabel: 'Activity (z-score)',
                    });
                } else {
                    plotContainer.innerHTML = `<p class="loading">No data available for "${signature}" [${this.signatureType}]${cellType !== 'All' ? ` in ${cellType}` : ''}</p>`;
                }
            } catch (e) {
                plotContainer.innerHTML = `<p class="loading">Error: ${e.message}</p>`;
            }
        }

        // Load heatmap
        if (heatmapContainer) {
            try {
                heatmapContainer.innerHTML = '<p class="loading">Loading heatmap...</p>';

                const data = await API.get(`/cima/boxplots/${endpoint}/${signature}/heatmap`, {
                    signature_type: this.signatureType
                });

                if (data && data.cell_types && data.bins && data.medians && data.cell_types.length > 0) {
                    this.renderCellTypeHeatmap('stratified-heatmap', data, {
                        title: `${signature} [${this.signatureType}] Activity: Cell Types × ${variable === 'age' ? 'Age Groups' : 'BMI Categories'}`,
                        xLabel: variable === 'age' ? 'Age Group' : 'BMI Category',
                        yLabel: 'Cell Type',
                    });
                } else {
                    heatmapContainer.innerHTML = `<p class="loading">No cell-type specific data available for "${signature}" [${this.signatureType}]</p>`;
                }
            } catch (e) {
                heatmapContainer.innerHTML = `<p class="loading">Error loading heatmap: ${e.message}</p>`;
            }
        }
    },

    renderCellTypeHeatmap(containerId, data, options = {}) {
        const { cell_types, bins, medians } = data;

        // medians is a 2D array: cell_types × bins
        const trace = {
            type: 'heatmap',
            z: medians,
            x: bins,
            y: cell_types,
            colorscale: 'RdBu',
            zmid: 0,
            colorbar: {
                title: 'Median Activity',
                titleside: 'right',
            },
            hovertemplate: '<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>',
        };

        Plotly.newPlot(containerId, [trace], {
            title: options.title || 'Cell Type Activity Heatmap',
            xaxis: {
                title: options.xLabel || 'Bin',
                tickangle: -45,
            },
            yaxis: {
                title: options.yLabel || 'Cell Type',
                automargin: true,
                tickfont: { size: 10 },
            },
            margin: { l: 150, r: 80, t: 50, b: 80 },
            font: { family: 'Inter, sans-serif' },
        });
    },

    async updateInflamStratifiedPlot() {
        const variable = document.getElementById('inflam-strat-variable')?.value || 'age';
        const signature = document.getElementById('inflam-strat-signature-search')?.value || 'IFNG';
        const cellType = document.getElementById('inflam-strat-celltype')?.value || 'All';

        const plotContainer = document.getElementById('inflam-stratified-plot');
        const heatmapContainer = document.getElementById('inflam-stratified-heatmap');

        if (!signature) {
            if (plotContainer) plotContainer.innerHTML = '<p class="loading">Enter a signature name to view distribution</p>';
            if (heatmapContainer) heatmapContainer.innerHTML = '';
            return;
        }

        const endpoint = variable === 'age' ? 'age' : 'bmi';

        // Load boxplot
        if (plotContainer) {
            try {
                plotContainer.innerHTML = '<p class="loading">Loading...</p>';

                const params = { signature_type: this.signatureType };
                if (cellType && cellType !== 'All') {
                    params.cell_type = cellType;
                }

                const data = await API.get(`/inflammation/boxplots/${endpoint}/${signature}`, params);

                if (data && data.length > 0) {
                    const titleCellType = cellType === 'All' ? '' : ` (${cellType})`;
                    this.renderBoxplotFromStats('inflam-stratified-plot', data, {
                        title: `${signature} [${this.signatureType}] Activity by ${variable === 'age' ? 'Age Group' : 'BMI Category'}${titleCellType}`,
                        yLabel: 'Activity (z-score)',
                    });
                } else {
                    plotContainer.innerHTML = `<p class="loading">No data available for "${signature}" [${this.signatureType}]${cellType !== 'All' ? ` in ${cellType}` : ''}</p>`;
                }
            } catch (e) {
                plotContainer.innerHTML = `<p class="loading">Error: ${e.message}</p>`;
            }
        }

        // Load heatmap
        if (heatmapContainer) {
            try {
                heatmapContainer.innerHTML = '<p class="loading">Loading heatmap...</p>';

                const data = await API.get(`/inflammation/boxplots/${endpoint}/${signature}/heatmap`, {
                    signature_type: this.signatureType
                });

                if (data && data.cell_types && data.bins && data.medians && data.cell_types.length > 0) {
                    this.renderCellTypeHeatmap('inflam-stratified-heatmap', data, {
                        title: `${signature} [${this.signatureType}] Activity: Cell Types × ${variable === 'age' ? 'Age Groups' : 'BMI Categories'}`,
                        xLabel: variable === 'age' ? 'Age Group' : 'BMI Category',
                        yLabel: 'Cell Type',
                    });
                } else {
                    heatmapContainer.innerHTML = `<p class="loading">No cell-type specific data available for "${signature}" [${this.signatureType}]</p>`;
                }
            } catch (e) {
                heatmapContainer.innerHTML = `<p class="loading">Error loading heatmap: ${e.message}</p>`;
            }
        }
    },

    async updateDifferentialPlot() {
        const comparison = document.getElementById('diff-comparison')?.value || 'sex';
        const plotContainer = document.getElementById('differential-volcano');
        if (!plotContainer) return;

        try {
            const data = await API.get('/cima/differential', {
                comparison, signature_type: this.signatureType,
            });

            if (data && data.length > 0) {
                // Volcano plot
                const x = data.map(d => d.log2fc || d.effect_size || 0);
                const y = data.map(d => -Math.log10(d.pvalue || d.p_value || 1));
                const labels = data.map(d => d.signature || d.protein);

                Plotly.newPlot('differential-volcano', [{
                    x, y, text: labels,
                    mode: 'markers',
                    type: 'scatter',
                    marker: {
                        color: x.map(v => v > 0 ? '#ef4444' : '#2563eb'),
                        size: 8,
                    },
                    hovertemplate: '%{text}<br>Log2FC: %{x:.2f}<br>-log10(p): %{y:.2f}<extra></extra>',
                }], {
                    title: `Differential Analysis: ${comparison} [${this.signatureType}]`,
                    xaxis: { title: 'Log2 Fold Change', zeroline: true },
                    yaxis: { title: '-log10(p-value)' },
                    shapes: [
                        { type: 'line', x0: 0, x1: 0, y0: 0, y1: Math.max(...y), line: { dash: 'dash', color: 'gray' } },
                        { type: 'line', x0: Math.min(...x), x1: Math.max(...x), y0: -Math.log10(0.05), y1: -Math.log10(0.05), line: { dash: 'dash', color: 'red' } },
                    ],
                });
            } else {
                plotContainer.innerHTML = `<p class="loading">No differential data available [${this.signatureType}]</p>`;
            }
        } catch (e) {
            plotContainer.innerHTML = `<p class="loading">Error: ${e.message}</p>`;
        }
    },

    async updateDiseaseHeatmap() {
        const disease = document.getElementById('disease-select')?.value;
        const plotContainer = document.getElementById('disease-heatmap');
        if (!plotContainer) return;

        try {
            const data = await API.get('/inflammation/disease-activity', {
                disease, signature_type: this.signatureType,
            });

            if (data && data.z) {
                Heatmap.createActivityHeatmap('disease-heatmap', data, {
                    title: disease || 'All Diseases',
                });
            } else {
                plotContainer.innerHTML = '<p class="loading">No disease data available</p>';
            }
        } catch (e) {
            plotContainer.innerHTML = `<p class="loading">Error: ${e.message}</p>`;
        }
    },

    async updateInflamDifferential() {
        const disease = document.getElementById('inflam-diff-disease')?.value;
        const plotContainer = document.getElementById('inflam-volcano');
        if (!plotContainer) return;

        try {
            const data = await API.get('/inflammation/celltype-stratified', {
                disease, signature_type: this.signatureType,
            });

            if (data && data.length > 0) {
                const x = data.map(d => d.log2fc || 0);
                const y = data.map(d => -Math.log10(d.p_value || 1));
                const labels = data.map(d => `${d.signature} (${d.cell_type})`);

                Plotly.newPlot('inflam-volcano', [{
                    x, y, text: labels,
                    mode: 'markers',
                    type: 'scatter',
                    marker: { color: x.map(v => v > 0 ? '#ef4444' : '#2563eb'), size: 8 },
                    hovertemplate: '%{text}<br>Log2FC: %{x:.2f}<br>-log10(p): %{y:.2f}<extra></extra>',
                }], {
                    title: `Disease vs Healthy: ${disease || 'All'} [${this.signatureType}]`,
                    xaxis: { title: 'Log2 Fold Change', zeroline: true },
                    yaxis: { title: '-log10(p-value)' },
                });
            } else {
                plotContainer.innerHTML = `<p class="loading">No differential data available [${this.signatureType}]</p>`;
            }
        } catch (e) {
            plotContainer.innerHTML = `<p class="loading">Error: ${e.message}</p>`;
        }
    },

    async updateTreatmentResponse() {
        const disease = document.getElementById('treatment-disease')?.value;
        const rocContainer = document.getElementById('treatment-roc');
        if (!rocContainer) return;

        try {
            const params = disease ? { disease } : {};
            const data = await API.get('/inflammation/treatment-response/roc', params);

            if (data && data.length > 0) {
                // Plot ROC curves - data is a list of {disease, model, auc, fpr, tpr}
                const traces = data.map(curve => ({
                    x: curve.fpr,
                    y: curve.tpr,
                    name: `${curve.model} (AUC=${curve.auc?.toFixed(2) || 'N/A'})`,
                    mode: 'lines',
                    type: 'scatter',
                }));

                Plotly.newPlot('treatment-roc', traces, {
                    title: `Treatment Response Prediction${disease ? ` - ${disease}` : ''} [${this.signatureType}]`,
                    xaxis: { title: 'False Positive Rate', range: [0, 1] },
                    yaxis: { title: 'True Positive Rate', range: [0, 1] },
                    shapes: [{
                        type: 'line', x0: 0, x1: 1, y0: 0, y1: 1,
                        line: { dash: 'dash', color: 'gray' },
                    }],
                });
            } else {
                rocContainer.innerHTML = `<p class="loading">No treatment response data available [${this.signatureType}]</p>`;
            }
        } catch (e) {
            rocContainer.innerHTML = `<p class="loading">Error: ${e.message}</p>`;
        }
    },

    async updateOrganMap() {
        const signature = document.getElementById('organ-signature')?.value || 'IFNG';

        try {
            const data = await API.get('/scatlas/organ-signatures', { signature_type: this.signatureType });

            if (data && data.length > 0) {
                // Filter for selected signature
                const sigData = data.filter(d => d.signature === signature || d.protein === signature);

                if (sigData.length > 0) {
                    const organs = sigData.map(d => d.organ);
                    const values = sigData.map(d => d.mean_activity || d.value);

                    Plotly.newPlot('organ-bar', [{
                        x: values,
                        y: organs,
                        type: 'bar',
                        orientation: 'h',
                        marker: { color: values.map(v => v > 0 ? '#ef4444' : '#2563eb') },
                    }], {
                        title: `${signature} [${this.signatureType}] Activity by Organ`,
                        xaxis: { title: 'Mean Activity (z-score)' },
                        margin: { l: 150 },
                    });
                }
            }
        } catch (e) {
            document.getElementById('organ-bar').innerHTML = `<p class="loading">Error: ${e.message}</p>`;
        }
    },

    async updateBiochemScatter() {
        const marker = document.getElementById('biochem-marker')?.value;
        const signature = document.getElementById('biochem-signature')?.value || 'IFNG';
        const plotContainer = document.getElementById('biochem-scatter');
        if (!plotContainer || !marker) return;

        try {
            // Endpoint format: /cima/scatter/biochem/{signature}/{variable}
            const data = await API.get(`/cima/scatter/biochem/${signature}/${marker}`, {
                signature_type: this.signatureType
            });
            if (data && data.x && data.y) {
                Scatter.create('biochem-scatter', data, {
                    title: `${signature} [${this.signatureType}] vs ${marker}`,
                    xLabel: marker,
                    yLabel: `${signature} Activity`,
                    showTrendLine: true,
                });
            }
        } catch (e) {
            plotContainer.innerHTML = `<p class="loading">Error: ${e.message}</p>`;
        }
    },

    async updateEqtlPlot() {
        const signature = document.getElementById('eqtl-signature')?.value || 'IFNG';
        const search = document.getElementById('eqtl-search')?.value || '';
        const tableContainer = document.getElementById('eqtl-table');

        try {
            const data = await API.get('/cima/eqtl', { signature, search, limit: 20 });

            if (data && data.length > 0) {
                // Render as table
                tableContainer.innerHTML = `
                    <table class="validation-table">
                        <thead>
                            <tr><th>SNP</th><th>Gene</th><th>Beta</th><th>P-value</th><th>Distance</th></tr>
                        </thead>
                        <tbody>
                            ${data.map(d => `
                                <tr>
                                    <td>${d.snp || d.variant}</td>
                                    <td>${d.gene}</td>
                                    <td>${d.beta?.toFixed(3) || 'N/A'}</td>
                                    <td>${d.pvalue?.toExponential(2) || 'N/A'}</td>
                                    <td>${d.distance?.toLocaleString() || 'N/A'}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;
            } else {
                tableContainer.innerHTML = '<p class="loading">No eQTL data available</p>';
            }
        } catch (e) {
            tableContainer.innerHTML = `<p class="loading">Error: ${e.message}</p>`;
        }
    },

    /**
     * Change signature type
     */
    changeSignatureType(type) {
        this.signatureType = type;
        this.loadTabContent(this.currentTab);
    },

    /**
     * Export current data
     */
    exportData() {
        const url = API.getExportUrl(this.currentAtlas, 'csv', 'activity');
        window.open(url, '_blank');
    },
};

// Make available globally
window.AtlasDetailPage = AtlasDetailPage;
