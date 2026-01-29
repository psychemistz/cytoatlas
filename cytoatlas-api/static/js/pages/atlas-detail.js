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
                                <td>Cell Types</td>
                                <td>27 cell types × (43 CytoSig + 1,170 SecAct)</td>
                                <td>32,751</td>
                            </tr>
                            <tr>
                                <td>Age Correlations</td>
                                <td>1,213 proteins × Age</td>
                                <td>${stats?.n_age_correlations?.toLocaleString() || 'N/A'}</td>
                            </tr>
                            <tr>
                                <td>BMI Correlations</td>
                                <td>1,213 proteins × BMI</td>
                                <td>${stats?.n_bmi_correlations?.toLocaleString() || 'N/A'}</td>
                            </tr>
                            <tr>
                                <td>Age/BMI Stratified</td>
                                <td>1,213 proteins × 28 cell types × 9 bins</td>
                                <td>21,366</td>
                            </tr>
                            <tr>
                                <td>Biochemistry</td>
                                <td>1,190 proteins × 19 biomarkers</td>
                                <td>${stats?.n_biochem_correlations?.toLocaleString() || 'N/A'}</td>
                            </tr>
                            <tr>
                                <td>Biochem Scatter</td>
                                <td>396 samples × 19 biomarkers × (43 CytoSig + 1,170 SecAct)</td>
                                <td>${(396 * (43 + 1170)).toLocaleString()}</td>
                            </tr>
                            <tr>
                                <td>Metabolites</td>
                                <td>1,213 proteins × 500 metabolites (top)</td>
                                <td>${stats?.n_metabolite_correlations?.toLocaleString() || 'N/A'}</td>
                            </tr>
                            <tr>
                                <td>Differential</td>
                                <td>1,213 proteins × 10 categories</td>
                                <td>${stats?.n_differential_tests?.toLocaleString() || '12,130'}</td>
                            </tr>
                            <tr>
                                <td>Cell Type Correlations</td>
                                <td>27 cell types × (43 CytoSig + 100 SecAct) × 2 (age, BMI)</td>
                                <td>7,722</td>
                            </tr>
                            <tr>
                                <td>Multi-omics</td>
                                <td>Cytokine-Biochemistry (21,859) + Cytokine-Metabolite (500)</td>
                                <td>22,359</td>
                            </tr>
                            <tr>
                                <td>Population</td>
                                <td>421 donors × demographic subgroups (sex, age, BMI, blood type, smoking)</td>
                                <td>6,065</td>
                            </tr>
                            <tr>
                                <td>eQTL Browser</td>
                                <td>71,530 significant cis-eQTLs across 69 cell types (9,600 genes)</td>
                                <td>71,530</td>
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

    // Store biochemistry proteins for autocomplete
    biochemAllProteins: { CytoSig: [], SecAct: [] },
    biochemData: null,

    async loadCimaBiochemistry(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Biochemistry Correlations</h3>
                <p>Spearman correlations between cytokine activities and blood biochemistry parameters</p>
            </div>

            <div class="controls" style="display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem;">
                <div class="control-group" style="position: relative;">
                    <label>Search Protein</label>
                    <input type="text" id="biochem-protein-search" class="filter-select"
                           placeholder="Type to filter (e.g., IFNG, IL6...)"
                           style="width: 180px;" autocomplete="off"
                           oninput="AtlasDetailPage.showBiochemSuggestions(this.value)"
                           onkeyup="if(event.key==='Enter') AtlasDetailPage.updateBiochemHeatmap()">
                    <div id="biochem-suggestions" style="position: absolute; top: 100%; left: 0; width: 180px; max-height: 200px; overflow-y: auto; background: white; border: 1px solid #ddd; border-radius: 4px; display: none; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"></div>
                </div>
                <div class="control-group">
                    <label>Top N Proteins</label>
                    <select id="biochem-top-n" class="filter-select" onchange="AtlasDetailPage.updateBiochemHeatmap()">
                        <option value="20">Top 20</option>
                        <option value="50" selected>Top 50</option>
                        <option value="100">Top 100</option>
                        <option value="all">All</option>
                    </select>
                </div>
            </div>

            <details class="card" style="margin-bottom: 1rem; padding: 1rem;">
                <summary style="cursor: pointer; font-weight: 600; color: var(--primary-color);">Blood Marker Abbreviations (click to expand)</summary>
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 0.5rem; margin-top: 1rem; font-size: 0.9rem;">
                    <div><strong>ALB</strong> - Albumin (liver function, nutrition)</div>
                    <div><strong>GLOB</strong> - Globulin (immune proteins)</div>
                    <div><strong>A/G</strong> - Albumin/Globulin ratio</div>
                    <div><strong>TP</strong> - Total Protein</div>
                    <div><strong>ALT</strong> - Alanine Aminotransferase (liver enzyme)</div>
                    <div><strong>AST</strong> - Aspartate Aminotransferase (liver/heart enzyme)</div>
                    <div><strong>AST/ALT</strong> - AST to ALT ratio (liver damage pattern)</div>
                    <div><strong>GGT</strong> - Gamma-Glutamyl Transferase (liver/bile duct)</div>
                    <div><strong>Tbil</strong> - Total Bilirubin (liver function)</div>
                    <div><strong>DBIL</strong> - Direct Bilirubin (conjugated)</div>
                    <div><strong>IBIL</strong> - Indirect Bilirubin (unconjugated)</div>
                    <div><strong>CHOL</strong> - Total Cholesterol</div>
                    <div><strong>HDL-C</strong> - HDL Cholesterol ("good" cholesterol)</div>
                    <div><strong>LDL-C</strong> - LDL Cholesterol ("bad" cholesterol)</div>
                    <div><strong>TG</strong> - Triglycerides (blood fat)</div>
                    <div><strong>GLU</strong> - Glucose (blood sugar)</div>
                    <div><strong>Cr</strong> - Creatinine (kidney function)</div>
                    <div><strong>UR</strong> - Urea/BUN (kidney function)</div>
                    <div><strong>UA</strong> - Uric Acid (gout, kidney)</div>
                </div>
            </details>

            <div id="biochem-heatmap" class="plot-container" style="height: 600px;">
                <p class="loading">Loading biochemistry data...</p>
            </div>
        `;

        // Load all biochemistry data
        await this.loadBiochemData();
        this.updateBiochemHeatmap();
    },

    async loadBiochemData() {
        // Load both CytoSig and SecAct data
        const [cytosigData, secactData] = await Promise.all([
            API.get('/cima/correlations/biochemistry', { signature_type: 'CytoSig' }),
            API.get('/cima/correlations/biochemistry', { signature_type: 'SecAct' }),
        ]);

        this.biochemData = {
            CytoSig: cytosigData || [],
            SecAct: secactData || [],
        };

        // Build protein lists for autocomplete
        // API returns: signature (protein name), variable (blood marker)
        this.biochemAllProteins.CytoSig = [...new Set(this.biochemData.CytoSig.map(d => d.signature))].sort();
        this.biochemAllProteins.SecAct = [...new Set(this.biochemData.SecAct.map(d => d.signature))].sort();
    },

    showBiochemSuggestions(query) {
        const suggestionsDiv = document.getElementById('biochem-suggestions');
        if (!suggestionsDiv) return;

        const sigType = this.signatureType;
        const proteins = this.biochemAllProteins[sigType] || [];

        if (!query || query.length < 1) {
            suggestionsDiv.style.display = 'none';
            return;
        }

        const matches = proteins.filter(p => p.toLowerCase().includes(query.toLowerCase())).slice(0, 10);

        if (matches.length === 0) {
            suggestionsDiv.style.display = 'none';
            return;
        }

        suggestionsDiv.innerHTML = matches.map(p =>
            `<div style="padding: 8px; cursor: pointer; border-bottom: 1px solid #eee;"
                  onmouseover="this.style.background='#f0f0f0'"
                  onmouseout="this.style.background='white'"
                  onclick="AtlasDetailPage.selectBiochemProtein('${p}')">${p}</div>`
        ).join('');
        suggestionsDiv.style.display = 'block';
    },

    selectBiochemProtein(protein) {
        const searchInput = document.getElementById('biochem-protein-search');
        const suggestionsDiv = document.getElementById('biochem-suggestions');
        if (searchInput) searchInput.value = protein;
        if (suggestionsDiv) suggestionsDiv.style.display = 'none';
        this.updateBiochemHeatmap();
    },

    updateBiochemHeatmap() {
        const container = document.getElementById('biochem-heatmap');
        if (!container || !this.biochemData) return;

        const sigType = this.signatureType;
        const searchQuery = document.getElementById('biochem-protein-search')?.value?.trim() || '';
        const topN = document.getElementById('biochem-top-n')?.value || '50';

        let data = this.biochemData[sigType] || [];

        if (data.length === 0) {
            container.innerHTML = '<p class="loading">No biochemistry data available</p>';
            return;
        }

        // Get all unique proteins and their max correlations
        // API format: {signature (protein name), variable (blood marker), rho, ...}
        const proteinMaxCorr = {};
        data.forEach(d => {
            const protein = d.signature;  // API uses 'signature' for protein name
            const absRho = Math.abs(d.rho);
            if (!proteinMaxCorr[protein] || absRho > proteinMaxCorr[protein]) {
                proteinMaxCorr[protein] = absRho;
            }
        });

        // Filter by search query if provided
        let proteins;
        if (searchQuery) {
            proteins = Object.keys(proteinMaxCorr)
                .filter(p => p.toLowerCase().includes(searchQuery.toLowerCase()))
                .sort((a, b) => proteinMaxCorr[b] - proteinMaxCorr[a]);
        } else {
            // Sort by max correlation and take top N
            proteins = Object.keys(proteinMaxCorr)
                .sort((a, b) => proteinMaxCorr[b] - proteinMaxCorr[a]);

            if (topN !== 'all') {
                proteins = proteins.slice(0, parseInt(topN));
            }
        }

        if (proteins.length === 0) {
            container.innerHTML = `<p class="loading">No proteins match "${searchQuery}"</p>`;
            return;
        }

        // Filter data to selected proteins
        const filteredData = data.filter(d => proteins.includes(d.signature));

        this.renderBiochemHeatmap('biochem-heatmap', filteredData, proteins);
    },

    // Store scatter data
    biochemScatterData: null,

    async loadCimaBiochemScatter(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Biochemistry Scatter Plots</h3>
                <p>Interactive exploration of clinical-cytokine relationships</p>
            </div>

            <div class="controls" style="display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem;">
                <div class="control-group">
                    <label>X-Axis (Biochemistry)</label>
                    <select id="biochem-x-axis" class="filter-select" onchange="AtlasDetailPage.updateBiochemScatterPlot()">
                        <option value="ALT">ALT (Liver enzyme)</option>
                        <option value="AST">AST (Liver enzyme)</option>
                        <option value="GGT">GGT (Liver/bile duct)</option>
                        <option value="CHOL">Total Cholesterol</option>
                        <option value="HDL-C">HDL Cholesterol</option>
                        <option value="LDL-C">LDL Cholesterol</option>
                        <option value="TG">Triglycerides</option>
                        <option value="GLU">Glucose</option>
                        <option value="Cr">Creatinine</option>
                        <option value="UA">Uric Acid</option>
                    </select>
                </div>
                <div class="control-group" style="position: relative;">
                    <label>Y-Axis (Protein)</label>
                    <input type="text" id="biochem-scatter-protein" class="filter-select"
                           placeholder="Search protein..."
                           value="IFNG"
                           style="width: 150px;" autocomplete="off"
                           oninput="AtlasDetailPage.showScatterProteinSuggestions(this.value)"
                           onkeyup="if(event.key==='Enter') AtlasDetailPage.updateBiochemScatterPlot()">
                    <div id="scatter-protein-suggestions" style="position: absolute; top: 100%; left: 0; width: 150px; max-height: 200px; overflow-y: auto; background: white; border: 1px solid #ddd; border-radius: 4px; display: none; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"></div>
                </div>
                <div class="control-group">
                    <label>Color By</label>
                    <select id="biochem-color-by" class="filter-select" onchange="AtlasDetailPage.updateBiochemScatterPlot()">
                        <option value="sex">Sex</option>
                        <option value="age_bin">Age Group</option>
                        <option value="bmi_bin">BMI Category</option>
                    </select>
                </div>
            </div>

            <div class="viz-grid">
                <div class="sub-panel" style="flex: 2;">
                    <div id="biochem-scatter-plot" class="plot-container" style="height: 500px;">
                        <p class="loading">Loading scatter data...</p>
                    </div>
                </div>
                <div class="sub-panel" style="flex: 1;">
                    <div class="panel-header">
                        <h4>Regression Statistics</h4>
                    </div>
                    <div id="biochem-regression-stats" style="padding: 1rem; font-size: 0.9rem;">
                        <p class="loading">Loading...</p>
                    </div>
                </div>
            </div>
        `;

        // Load scatter data
        await this.loadBiochemScatterData();
        this.updateBiochemScatterPlot();
    },

    async loadBiochemScatterData() {
        this.biochemScatterData = await API.get('/cima/scatter/biochem-samples');
    },

    showScatterProteinSuggestions(query) {
        const suggestionsDiv = document.getElementById('scatter-protein-suggestions');
        if (!suggestionsDiv || !this.biochemScatterData) return;

        const proteins = this.signatureType === 'CytoSig'
            ? this.biochemScatterData.cytokines || []
            : this.biochemScatterData.secact_proteins || [];

        if (!query || query.length < 1) {
            suggestionsDiv.style.display = 'none';
            return;
        }

        const matches = proteins.filter(p => p.toLowerCase().includes(query.toLowerCase())).slice(0, 10);

        if (matches.length === 0) {
            suggestionsDiv.style.display = 'none';
            return;
        }

        suggestionsDiv.innerHTML = matches.map(p =>
            `<div style="padding: 8px; cursor: pointer; border-bottom: 1px solid #eee;"
                  onmouseover="this.style.background='#f0f0f0'"
                  onmouseout="this.style.background='white'"
                  onclick="AtlasDetailPage.selectScatterProtein('${p}')">${p}</div>`
        ).join('');
        suggestionsDiv.style.display = 'block';
    },

    selectScatterProtein(protein) {
        const input = document.getElementById('biochem-scatter-protein');
        const suggestionsDiv = document.getElementById('scatter-protein-suggestions');
        if (input) input.value = protein;
        if (suggestionsDiv) suggestionsDiv.style.display = 'none';
        this.updateBiochemScatterPlot();
    },

    updateBiochemScatterPlot() {
        const container = document.getElementById('biochem-scatter-plot');
        const statsContainer = document.getElementById('biochem-regression-stats');
        if (!container) return;

        const scatterData = this.biochemScatterData;
        if (!scatterData || !scatterData.samples) {
            container.innerHTML = '<p class="loading">Scatter data not available</p>';
            return;
        }

        const xAxis = document.getElementById('biochem-x-axis')?.value || 'ALT';
        const yProtein = document.getElementById('biochem-scatter-protein')?.value || 'IFNG';
        const colorBy = document.getElementById('biochem-color-by')?.value || 'sex';

        const samples = scatterData.samples;
        const activityKey = this.signatureType === 'CytoSig' ? 'activity' : 'secact_activity';

        // Color mapping
        const colorMap = {
            'Male': '#1f77b4', 'Female': '#ff7f0e', 'Unknown': '#999',
            '<30': '#1f77b4', '30-39': '#2ca02c', '40-49': '#ff7f0e', '50-59': '#d62728', '60+': '#9467bd',
            'Normal': '#2ca02c', 'Overweight': '#ff7f0e', 'Obese': '#d62728', 'Underweight': '#17becf'
        };

        // Process data
        const plotData = [];
        samples.forEach(s => {
            const xVal = s.biochem?.[xAxis];
            const yVal = s[activityKey]?.[yProtein];
            if (xVal == null || yVal == null) return;

            let group;
            if (colorBy === 'sex') {
                group = s.sex || 'Unknown';
            } else if (colorBy === 'age_bin') {
                const age = s.age;
                if (age == null) group = 'Unknown';
                else if (age < 30) group = '<30';
                else if (age < 40) group = '30-39';
                else if (age < 50) group = '40-49';
                else if (age < 60) group = '50-59';
                else group = '60+';
            } else if (colorBy === 'bmi_bin') {
                const bmi = s.bmi;
                if (bmi == null) group = 'Unknown';
                else if (bmi < 18.5) group = 'Underweight';
                else if (bmi < 25) group = 'Normal';
                else if (bmi < 30) group = 'Overweight';
                else group = 'Obese';
            } else {
                group = 'All';
            }

            plotData.push({ x: xVal, y: yVal, group, sample: s.sample, age: s.age, sex: s.sex, bmi: s.bmi });
        });

        if (plotData.length === 0) {
            container.innerHTML = `<p class="loading">No data available for "${yProtein}" in ${this.signatureType}</p>`;
            return;
        }

        // Get unique groups
        let groups = [...new Set(plotData.map(d => d.group))];
        if (colorBy === 'sex') {
            groups = groups.sort((a, b) => {
                const order = ['Male', 'Female', 'Unknown'];
                return order.indexOf(a) - order.indexOf(b);
            });
        } else {
            groups = groups.sort();
        }

        // Create traces
        const traces = groups.map(group => {
            const groupData = plotData.filter(d => d.group === group);
            return {
                type: 'scatter',
                mode: 'markers',
                name: `${group} (n=${groupData.length})`,
                x: groupData.map(d => d.x),
                y: groupData.map(d => d.y),
                text: groupData.map(d => `${d.sample}<br>Sex: ${d.sex || 'NA'}<br>Age: ${d.age || 'NA'}<br>BMI: ${d.bmi?.toFixed(1) || 'NA'}`),
                marker: { size: 8, color: colorMap[group] || '#999', opacity: 0.7 },
                hovertemplate: '<b>%{text}</b><br>' + xAxis + ': %{x:.2f}<br>' + yProtein + ': %{y:.3f}<extra></extra>'
            };
        });

        // Calculate trend line
        const allX = plotData.map(d => d.x);
        const allY = plotData.map(d => d.y);
        const n = allX.length;
        const sumX = allX.reduce((a, b) => a + b, 0);
        const sumY = allY.reduce((a, b) => a + b, 0);
        const sumXY = allX.reduce((sum, x, i) => sum + x * allY[i], 0);
        const sumX2 = allX.reduce((sum, x) => sum + x * x, 0);

        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;

        // Correlation
        const meanX = sumX / n, meanY = sumY / n;
        const ssX = allX.reduce((sum, x) => sum + Math.pow(x - meanX, 2), 0);
        const ssY = allY.reduce((sum, y) => sum + Math.pow(y - meanY, 2), 0);
        const ssXY = allX.reduce((sum, x, i) => sum + (x - meanX) * (allY[i] - meanY), 0);
        const r = ssXY / Math.sqrt(ssX * ssY);
        const r2 = r * r;

        // Add trend line
        const xMin = Math.min(...allX), xMax = Math.max(...allX);
        traces.push({
            type: 'scatter',
            mode: 'lines',
            name: 'Trend',
            x: [xMin, xMax],
            y: [slope * xMin + intercept, slope * xMax + intercept],
            line: { color: '#333', dash: 'dash', width: 2 },
            hoverinfo: 'skip'
        });

        Plotly.newPlot(container, traces, {
            title: `${yProtein} vs ${xAxis}`,
            xaxis: { title: xAxis },
            yaxis: { title: `${yProtein} Activity (z-score)` },
            margin: { l: 60, r: 20, t: 40, b: 50 },
            legend: { x: 1, xanchor: 'right', y: 1 },
            font: { family: 'Inter, sans-serif' }
        }, { responsive: true });

        // Update stats
        if (statsContainer) {
            statsContainer.innerHTML = `
                <table style="width: 100%; border-collapse: collapse;">
                    <tr><td style="padding: 0.5rem; border-bottom: 1px solid #eee;"><strong>N samples</strong></td><td style="padding: 0.5rem; border-bottom: 1px solid #eee;">${n}</td></tr>
                    <tr><td style="padding: 0.5rem; border-bottom: 1px solid #eee;"><strong>Pearson r</strong></td><td style="padding: 0.5rem; border-bottom: 1px solid #eee;">${r.toFixed(4)}</td></tr>
                    <tr><td style="padding: 0.5rem; border-bottom: 1px solid #eee;"><strong>R²</strong></td><td style="padding: 0.5rem; border-bottom: 1px solid #eee;">${r2.toFixed(4)}</td></tr>
                    <tr><td style="padding: 0.5rem; border-bottom: 1px solid #eee;"><strong>Slope</strong></td><td style="padding: 0.5rem; border-bottom: 1px solid #eee;">${slope.toFixed(4)}</td></tr>
                    <tr><td style="padding: 0.5rem;"><strong>Intercept</strong></td><td style="padding: 0.5rem;">${intercept.toFixed(4)}</td></tr>
                </table>
            `;
        }
    },

    // Store metabolite data
    metaboliteData: null,

    async loadCimaMetabolites(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Metabolite Correlations</h3>
                <p>Top 500 cytokine-metabolite correlations from plasma metabolomics data</p>
            </div>

            <div class="controls" style="display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem;">
                <div class="control-group">
                    <label>Metabolite Category</label>
                    <select id="metab-category" class="filter-select" onchange="AtlasDetailPage.updateMetabolitePlots()">
                        <option value="all">All Categories</option>
                        <option value="lipid">Lipids</option>
                        <option value="amino_acid">Amino Acids</option>
                        <option value="carbohydrate">Carbohydrates</option>
                        <option value="nucleotide">Nucleotides</option>
                        <option value="cofactor">Cofactors & Vitamins</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Min |Correlation|</label>
                    <select id="metab-threshold" class="filter-select" onchange="AtlasDetailPage.updateMetabolitePlots()">
                        <option value="0.2">0.2</option>
                        <option value="0.3" selected>0.3</option>
                        <option value="0.4">0.4</option>
                        <option value="0.5">0.5</option>
                    </select>
                </div>
            </div>

            <div class="viz-grid">
                <div class="sub-panel">
                    <div class="panel-header">
                        <h4>Top Metabolite Correlations</h4>
                        <p>Ranked by absolute correlation strength</p>
                    </div>
                    <div id="metabolite-lollipop" class="plot-container" style="height: 550px;"></div>
                </div>
                <div class="sub-panel">
                    <div class="panel-header">
                        <h4>Correlation Heatmap</h4>
                        <p>Cytokines × Metabolites</p>
                    </div>
                    <div id="metabolite-heatmap" class="plot-container" style="height: 550px;"></div>
                </div>
            </div>
        `;

        // Load metabolite data
        this.metaboliteData = await API.get('/cima/correlations/metabolites', { signature_type: this.signatureType, limit: 500 });
        this.updateMetabolitePlots();
    },

    updateMetabolitePlots() {
        const data = this.metaboliteData;
        if (!data || !data.correlations) {
            document.getElementById('metabolite-lollipop').innerHTML = '<p class="loading">No metabolite data</p>';
            return;
        }

        const threshold = parseFloat(document.getElementById('metab-threshold')?.value || '0.3');
        const category = document.getElementById('metab-category')?.value || 'all';

        // Filter by threshold
        let filtered = data.correlations.filter(d => Math.abs(d.rho || d.correlation) >= threshold);

        // Filter by category (simple keyword matching)
        if (category !== 'all') {
            const categoryKeywords = {
                'lipid': ['lipid', 'fatty', 'cholesterol', 'phospho', 'sphingo', 'sterol', 'PS ', 'PC ', 'PE ', 'LPC', 'LPE'],
                'amino_acid': ['amino', 'amine', 'glutam', 'glycine', 'alanine', 'serine', 'proline', 'valine', 'leucine', 'isoleucine'],
                'carbohydrate': ['glucose', 'fructose', 'sugar', 'sacchar', 'hexose', 'pentose'],
                'nucleotide': ['nucleotide', 'purine', 'pyrimidine', 'adenine', 'guanine', 'cytosine', 'uracil'],
                'cofactor': ['vitamin', 'cofactor', 'NAD', 'FAD', 'coenzyme']
            };
            const keywords = categoryKeywords[category] || [];
            filtered = filtered.filter(d => {
                const metab = (d.metabolite || d.feature || '').toLowerCase();
                return keywords.some(k => metab.includes(k.toLowerCase()));
            });
        }

        // Sort by absolute correlation
        filtered.sort((a, b) => Math.abs(b.rho || b.correlation) - Math.abs(a.rho || a.correlation));
        const top50 = filtered.slice(0, 50);

        // Lollipop chart
        if (top50.length > 0) {
            const labels = top50.map(d => `${d.signature || d.protein} × ${d.metabolite || d.feature}`);
            const values = top50.map(d => d.rho || d.correlation);
            const colors = values.map(v => v >= 0 ? '#ef4444' : '#2563eb');

            Plotly.newPlot('metabolite-lollipop', [{
                type: 'bar',
                y: labels.reverse(),
                x: values.reverse(),
                orientation: 'h',
                marker: { color: colors.reverse() },
                hovertemplate: '<b>%{y}</b><br>ρ = %{x:.3f}<extra></extra>'
            }], {
                title: `Top ${top50.length} Correlations (|ρ| ≥ ${threshold})`,
                xaxis: { title: 'Spearman ρ', range: [-1, 1] },
                yaxis: { tickfont: { size: 9 } },
                margin: { l: 200, r: 20, t: 40, b: 40 },
                font: { family: 'Inter, sans-serif' }
            }, { responsive: true });
        } else {
            document.getElementById('metabolite-lollipop').innerHTML = '<p class="loading">No correlations above threshold</p>';
        }

        // Heatmap
        if (data.z && data.x && data.y) {
            Heatmap.create('metabolite-heatmap', data, {
                title: 'Metabolite Correlations',
                colorbarTitle: 'Spearman ρ',
                symmetric: true,
            });
        } else {
            document.getElementById('metabolite-heatmap').innerHTML = '<p class="loading">Heatmap data not available</p>';
        }
    },

    // Store differential data
    differentialData: null,

    async loadCimaDifferential(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Differential Analysis</h3>
                <p>Activity differences by sex, smoking status, and blood type</p>
            </div>

            <div class="controls" style="display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem;">
                <div class="control-group">
                    <label>Comparison</label>
                    <select id="diff-comparison" class="filter-select" onchange="AtlasDetailPage.updateDifferentialPlots()">
                        <option value="sex|Female|Male">Sex (Female vs Male)</option>
                        <optgroup label="Smoking Status">
                            <option value="smoking_status|current|never">Current vs Never</option>
                            <option value="smoking_status|current|past">Current vs Past</option>
                            <option value="smoking_status|never|past">Never vs Past</option>
                        </optgroup>
                        <optgroup label="Blood Type">
                            <option value="blood_type|O|A">O vs A</option>
                            <option value="blood_type|O|B">O vs B</option>
                            <option value="blood_type|O|AB">O vs AB</option>
                            <option value="blood_type|B|A">B vs A</option>
                        </optgroup>
                    </select>
                </div>
            </div>

            <div class="viz-grid">
                <div class="sub-panel">
                    <div class="panel-header">
                        <h4>Volcano Plot</h4>
                        <p id="volcano-subtitle">Effect size vs significance</p>
                    </div>
                    <div id="differential-volcano" class="plot-container" style="height: 500px;"></div>
                </div>
                <div class="sub-panel">
                    <div class="panel-header">
                        <h4>Top Differential Signatures</h4>
                        <p>Sorted by significance score</p>
                    </div>
                    <div id="differential-bar" class="plot-container" style="height: 500px;"></div>
                </div>
            </div>
        `;

        // Load differential data
        this.differentialData = await API.get('/cima/differential', { signature_type: this.signatureType });
        this.updateDifferentialPlots();
    },

    updateDifferentialPlots() {
        const data = this.differentialData;
        if (!data || data.length === 0) {
            document.getElementById('differential-volcano').innerHTML = '<p class="loading">No differential data</p>';
            return;
        }

        const comparisonValue = document.getElementById('diff-comparison')?.value || 'sex|Female|Male';
        const [variable, group1, group2] = comparisonValue.split('|');

        // Filter data for this comparison
        const filtered = data.filter(d => {
            const comp = d.comparison || '';
            return comp.includes(variable) || comp.toLowerCase().includes(group1.toLowerCase());
        });

        if (filtered.length === 0) {
            document.getElementById('differential-volcano').innerHTML = '<p class="loading">No data for this comparison</p>';
            document.getElementById('differential-bar').innerHTML = '';
            return;
        }

        // Volcano plot
        const log2fc = filtered.map(d => d.log2fc || d.effect_size || 0);
        const negLogP = filtered.map(d => -Math.log10(d.pvalue || d.p_value || 1));
        const signatures = filtered.map(d => d.signature || d.protein);
        const significant = filtered.map(d => (d.qvalue || d.q_value || 1) < 0.05);

        Plotly.newPlot('differential-volcano', [{
            type: 'scatter',
            mode: 'markers',
            x: log2fc,
            y: negLogP,
            text: signatures,
            marker: {
                size: 8,
                color: significant.map(s => s ? '#ef4444' : '#999'),
                opacity: 0.7
            },
            hovertemplate: '<b>%{text}</b><br>log2FC: %{x:.3f}<br>-log10(p): %{y:.2f}<extra></extra>'
        }], {
            title: `${group1} vs ${group2}`,
            xaxis: { title: 'log2 Fold Change', zeroline: true, zerolinecolor: '#ccc' },
            yaxis: { title: '-log10(p-value)' },
            shapes: [{
                type: 'line', x0: -10, x1: 10, y0: -Math.log10(0.05), y1: -Math.log10(0.05),
                line: { color: 'red', dash: 'dash', width: 1 }
            }],
            margin: { l: 50, r: 20, t: 40, b: 50 },
            font: { family: 'Inter, sans-serif' }
        }, { responsive: true });

        // Bar chart - top differential
        const scored = filtered.map(d => ({
            ...d,
            score: Math.abs(d.log2fc || 0) * negLogP[filtered.indexOf(d)]
        }));
        scored.sort((a, b) => b.score - a.score);
        const top20 = scored.slice(0, 20);

        Plotly.newPlot('differential-bar', [{
            type: 'bar',
            y: top20.map(d => d.signature || d.protein).reverse(),
            x: top20.map(d => d.log2fc || d.effect_size || 0).reverse(),
            orientation: 'h',
            marker: {
                color: top20.map(d => (d.log2fc || 0) >= 0 ? '#ef4444' : '#2563eb').reverse()
            },
            hovertemplate: '<b>%{y}</b><br>log2FC: %{x:.3f}<extra></extra>'
        }], {
            title: 'Top 20 by Significance',
            xaxis: { title: 'log2 Fold Change' },
            margin: { l: 120, r: 20, t: 40, b: 50 },
            font: { family: 'Inter, sans-serif' }
        }, { responsive: true });

        // Update subtitle
        const subtitle = document.getElementById('volcano-subtitle');
        if (subtitle) subtitle.textContent = `${group1} vs ${group2}: ${filtered.length} signatures tested`;
    },

    // Store population data
    populationData: null,

    async loadCimaMultiomics(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Multi-omics Integration</h3>
                <p>Integrates cytokine activity with blood biochemistry and plasma metabolomics</p>
            </div>

            <div class="controls" style="display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem;">
                <div class="control-group">
                    <label>Cytokine Subset</label>
                    <select id="multiomics-cytokines" class="filter-select" onchange="AtlasDetailPage.updateMultiomicsPlot()">
                        <option value="all">All CytoSig (43)</option>
                        <option value="inflammatory">Inflammatory (IL-1B, IL-6, TNF)</option>
                        <option value="regulatory">Regulatory (IL-10, TGF-β, IL-4)</option>
                        <option value="th17">Th17 axis (IL-17A, IL-21, IL-22)</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Min |Correlation|</label>
                    <select id="multiomics-threshold" class="filter-select" onchange="AtlasDetailPage.updateMultiomicsPlot()">
                        <option value="0.2">0.2</option>
                        <option value="0.3" selected>0.3</option>
                        <option value="0.4">0.4</option>
                        <option value="0.5">0.5</option>
                    </select>
                </div>
            </div>

            <div class="card" style="margin-bottom: 1rem; padding: 1rem;">
                <strong>Multi-omics Integration:</strong> Integrates cytokine activity signatures with blood biochemistry (19 markers) and plasma metabolomics (65 metabolites) from CIMA healthy donors.
            </div>

            <div class="viz-grid">
                <div class="sub-panel" style="flex: 2;">
                    <div class="panel-header">
                        <h4>Cross-omic Correlations</h4>
                    </div>
                    <div id="multiomics-summary" class="plot-container" style="height: 500px;"></div>
                </div>
                <div class="sub-panel">
                    <div class="panel-header">
                        <h4>Top Associations</h4>
                    </div>
                    <div id="multiomics-top" style="padding: 1rem; max-height: 500px; overflow-y: auto;"></div>
                </div>
            </div>
        `;

        this.updateMultiomicsPlot();
    },

    async updateMultiomicsPlot() {
        const subset = document.getElementById('multiomics-cytokines')?.value || 'all';
        const threshold = parseFloat(document.getElementById('multiomics-threshold')?.value || '0.3');

        // Load biochemistry and metabolite correlations
        const [biochemData, metabData] = await Promise.all([
            API.get('/cima/correlations/biochemistry', { signature_type: 'CytoSig' }),
            API.get('/cima/correlations/metabolites', { signature_type: 'CytoSig', limit: 500 })
        ]);

        // Define cytokine subsets
        const subsetMap = {
            'inflammatory': ['IL1B', 'IL6', 'TNF', 'IL1A', 'IL18'],
            'regulatory': ['IL10', 'TGFB1', 'IL4', 'IL13', 'IL35'],
            'th17': ['IL17A', 'IL17F', 'IL21', 'IL22', 'IL23A']
        };
        const cytokineFilter = subset === 'all' ? null : subsetMap[subset] || null;

        // Filter and combine data
        let biochemFiltered = (biochemData || []).filter(d => Math.abs(d.rho) >= threshold);
        let metabFiltered = ((metabData?.correlations) || []).filter(d => Math.abs(d.rho || d.correlation) >= threshold);

        if (cytokineFilter) {
            biochemFiltered = biochemFiltered.filter(d => cytokineFilter.includes(d.signature));
            metabFiltered = metabFiltered.filter(d => cytokineFilter.includes(d.signature || d.protein));
        }

        // Create summary bar chart
        const biochemCount = biochemFiltered.length;
        const metabCount = metabFiltered.length;

        Plotly.newPlot('multiomics-summary', [{
            type: 'bar',
            x: ['Biochemistry', 'Metabolites'],
            y: [biochemCount, metabCount],
            marker: { color: ['#3b82f6', '#10b981'] },
            text: [biochemCount, metabCount],
            textposition: 'auto'
        }], {
            title: `Correlations with |ρ| ≥ ${threshold}`,
            yaxis: { title: 'Number of Significant Correlations' },
            margin: { l: 60, r: 20, t: 40, b: 50 },
            font: { family: 'Inter, sans-serif' }
        }, { responsive: true });

        // Top associations list
        const allCorr = [
            ...biochemFiltered.map(d => ({ type: 'Biochem', cytokine: d.signature, feature: d.variable, rho: d.rho })),
            ...metabFiltered.map(d => ({ type: 'Metab', cytokine: d.signature || d.protein, feature: d.metabolite || d.feature, rho: d.rho || d.correlation }))
        ];
        allCorr.sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho));
        const top20 = allCorr.slice(0, 20);

        const topContainer = document.getElementById('multiomics-top');
        if (topContainer) {
            topContainer.innerHTML = `
                <table style="width: 100%; font-size: 0.85rem; border-collapse: collapse;">
                    <thead>
                        <tr style="border-bottom: 2px solid #ddd;">
                            <th style="text-align: left; padding: 0.5rem;">Type</th>
                            <th style="text-align: left; padding: 0.5rem;">Cytokine</th>
                            <th style="text-align: left; padding: 0.5rem;">Feature</th>
                            <th style="text-align: right; padding: 0.5rem;">ρ</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${top20.map(d => `
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 0.4rem;"><span style="background: ${d.type === 'Biochem' ? '#dbeafe' : '#d1fae5'}; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem;">${d.type}</span></td>
                                <td style="padding: 0.4rem;">${d.cytokine}</td>
                                <td style="padding: 0.4rem;">${d.feature}</td>
                                <td style="padding: 0.4rem; text-align: right; color: ${d.rho >= 0 ? '#dc2626' : '#2563eb'};">${d.rho.toFixed(3)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        }
    },

    async loadCimaPopulation(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Population Stratification</h3>
                <p>Activity patterns across demographic groups</p>
            </div>

            <div class="controls" style="display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem;">
                <div class="control-group">
                    <label>Stratification Variable</label>
                    <select id="pop-stratify" class="filter-select" onchange="AtlasDetailPage.updatePopulationPlots()">
                        <option value="sex">Sex (Male vs Female)</option>
                        <option value="age">Age Group</option>
                        <option value="bmi">BMI Category</option>
                        <option value="blood_type">Blood Type</option>
                        <option value="smoking">Smoking Status</option>
                    </select>
                </div>
            </div>

            <div class="viz-grid">
                <div class="sub-panel">
                    <div class="panel-header">
                        <h4>Population Distribution</h4>
                        <p>Sample counts across groups (N=421)</p>
                    </div>
                    <div id="pop-distribution" class="plot-container" style="height: 300px;"></div>
                </div>
                <div class="sub-panel">
                    <div class="panel-header">
                        <h4>Top Effect Sizes</h4>
                        <p>Signatures with largest group differences</p>
                    </div>
                    <div id="pop-effect-sizes" class="plot-container" style="height: 300px;"></div>
                </div>
            </div>

            <div class="sub-panel" style="margin-top: 1rem;">
                <div class="panel-header">
                    <h4>Population-Stratified Heatmap</h4>
                    <p>Mean activity across demographic groups</p>
                </div>
                <div id="pop-heatmap" class="plot-container" style="height: 400px;"></div>
            </div>
        `;

        // Load population data
        this.populationData = await API.get('/cima/population-stratification');
        this.updatePopulationPlots();
    },

    updatePopulationPlots() {
        const data = this.populationData;
        if (!data) {
            document.getElementById('pop-distribution').innerHTML = '<p class="loading">Population data not available</p>';
            return;
        }

        const stratify = document.getElementById('pop-stratify')?.value || 'sex';
        const groups = data.groups?.[stratify] || {};
        const effects = data.effect_sizes?.[stratify] || [];
        const cytokines = data.cytokines || [];

        // Distribution chart
        const groupNames = Object.keys(groups);
        const groupCounts = Object.values(groups);

        if (groupNames.length > 0) {
            Plotly.newPlot('pop-distribution', [{
                type: 'bar',
                x: groupNames,
                y: groupCounts,
                marker: { color: '#3b82f6' },
                text: groupCounts,
                textposition: 'auto'
            }], {
                title: `Distribution by ${stratify}`,
                yaxis: { title: 'Sample Count' },
                margin: { l: 50, r: 20, t: 40, b: 50 },
                font: { family: 'Inter, sans-serif' }
            }, { responsive: true });
        }

        // Effect sizes
        if (effects.length > 0) {
            const sorted = [...effects].sort((a, b) => Math.abs(b.effect_size || b.log2fc || 0) - Math.abs(a.effect_size || a.log2fc || 0));
            const top10 = sorted.slice(0, 10);
            const bottom10 = sorted.slice(-10).reverse();
            const combined = [...top10, ...bottom10];

            Plotly.newPlot('pop-effect-sizes', [{
                type: 'bar',
                y: combined.map(d => d.signature || d.cytokine).reverse(),
                x: combined.map(d => d.effect_size || d.log2fc || 0).reverse(),
                orientation: 'h',
                marker: {
                    color: combined.map(d => (d.effect_size || d.log2fc || 0) >= 0 ? '#ef4444' : '#2563eb').reverse()
                }
            }], {
                title: 'Top & Bottom Effect Sizes',
                xaxis: { title: 'Effect Size' },
                margin: { l: 100, r: 20, t: 40, b: 50 },
                font: { family: 'Inter, sans-serif' }
            }, { responsive: true });
        } else {
            document.getElementById('pop-effect-sizes').innerHTML = '<p class="loading">Effect size data not available</p>';
        }

        // Heatmap placeholder
        document.getElementById('pop-heatmap').innerHTML = '<p class="loading">Heatmap visualization requires additional data processing</p>';
    },

    // Store eQTL data
    eqtlData: null,

    async loadCimaEqtl(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>eQTL Browser</h3>
                <p>Genetic variants associated with gene expression in immune cells</p>
            </div>

            <div class="controls" style="display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem;">
                <div class="control-group">
                    <label>Search Gene/Variant</label>
                    <input type="text" id="eqtl-search" class="filter-select" placeholder="e.g., IFNG, IL6" style="width: 150px;"
                           onkeyup="if(event.key==='Enter') AtlasDetailPage.updateEqtlPlots()">
                </div>
                <div class="control-group">
                    <label>Cell Type</label>
                    <select id="eqtl-celltype" class="filter-select" onchange="AtlasDetailPage.updateEqtlPlots()">
                        <option value="all">All Cell Types</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Significance</label>
                    <select id="eqtl-threshold" class="filter-select" onchange="AtlasDetailPage.updateEqtlPlots()">
                        <option value="1e-3">p < 1e-3</option>
                        <option value="1e-5">p < 1e-5</option>
                        <option value="5e-8" selected>Genome-wide (5e-8)</option>
                    </select>
                </div>
            </div>

            <div class="card" style="margin-bottom: 1rem; padding: 1rem;">
                <strong>cis-eQTL Browser:</strong> <span id="eqtl-count">Loading...</span> cis-eQTLs across immune cell types.
                <br><small style="color: #666;">Data: CIMA Lead cis-eQTL (study-wise FDR < 0.05)</small>
            </div>

            <div class="viz-grid">
                <div class="sub-panel" style="flex: 2;">
                    <div class="panel-header">
                        <h4>Manhattan Plot</h4>
                        <p>Genomic position vs -log10(p-value)</p>
                    </div>
                    <div id="eqtl-manhattan" class="plot-container" style="height: 400px;"></div>
                </div>
                <div class="sub-panel">
                    <div class="panel-header">
                        <h4>Top eQTL Results</h4>
                    </div>
                    <div id="eqtl-table" style="max-height: 400px; overflow-y: auto;"></div>
                </div>
            </div>

            <div class="sub-panel" style="margin-top: 1rem;">
                <div class="panel-header">
                    <h4>eQTLs by Cell Type</h4>
                </div>
                <div id="eqtl-celltype-bar" class="plot-container" style="height: 300px;"></div>
            </div>
        `;

        // Load eQTL data
        this.eqtlData = await API.get('/cima/eqtl');
        this.populateEqtlCellTypes();
        this.updateEqtlPlots();
    },

    populateEqtlCellTypes() {
        const data = this.eqtlData;
        if (!data || !data.cell_types) return;

        const select = document.getElementById('eqtl-celltype');
        if (select) {
            select.innerHTML = '<option value="all">All Cell Types</option>' +
                data.cell_types.map(ct => `<option value="${ct}">${ct}</option>`).join('');
        }

        // Update count
        const countEl = document.getElementById('eqtl-count');
        if (countEl && data.summary) {
            countEl.textContent = `${(data.summary.total_eqtls || data.eqtls?.length || 0).toLocaleString()}`;
        }
    },

    updateEqtlPlots() {
        const data = this.eqtlData;
        if (!data || !data.eqtls) {
            document.getElementById('eqtl-manhattan').innerHTML = '<p class="loading">eQTL data not available</p>';
            return;
        }

        const searchQuery = document.getElementById('eqtl-search')?.value?.toLowerCase() || '';
        const cellType = document.getElementById('eqtl-celltype')?.value || 'all';
        const threshold = parseFloat(document.getElementById('eqtl-threshold')?.value || '5e-8');

        // Filter eQTLs
        let filtered = data.eqtls.filter(d => d.pvalue <= threshold);

        if (cellType !== 'all') {
            filtered = filtered.filter(d => d.celltype === cellType);
        }

        if (searchQuery) {
            filtered = filtered.filter(d =>
                (d.gene || '').toLowerCase().includes(searchQuery) ||
                (d.variant || '').toLowerCase().includes(searchQuery)
            );
        }

        // Manhattan plot - sample for performance
        const plotData = filtered.length > 5000 ? filtered.slice(0, 5000) : filtered;

        // Assign colors by chromosome
        const chrColors = {};
        const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'];
        plotData.forEach(d => {
            const chr = d.chr || 'unknown';
            if (!chrColors[chr]) chrColors[chr] = colors[Object.keys(chrColors).length % colors.length];
        });

        Plotly.newPlot('eqtl-manhattan', [{
            type: 'scatter',
            mode: 'markers',
            x: plotData.map(d => d.pos || 0),
            y: plotData.map(d => -Math.log10(d.pvalue || 1)),
            text: plotData.map(d => `${d.gene} (${d.variant})`),
            marker: {
                size: 5,
                color: plotData.map(d => chrColors[d.chr || 'unknown']),
                opacity: 0.6
            },
            hovertemplate: '<b>%{text}</b><br>-log10(p): %{y:.2f}<extra></extra>'
        }], {
            title: `${filtered.length.toLocaleString()} eQTLs (p < ${threshold})`,
            xaxis: { title: 'Genomic Position' },
            yaxis: { title: '-log10(p-value)' },
            shapes: [{
                type: 'line', x0: 0, x1: Math.max(...plotData.map(d => d.pos || 0)),
                y0: -Math.log10(5e-8), y1: -Math.log10(5e-8),
                line: { color: 'red', dash: 'dash', width: 1 }
            }],
            margin: { l: 50, r: 20, t: 40, b: 50 },
            font: { family: 'Inter, sans-serif' }
        }, { responsive: true });

        // Top results table
        const top20 = filtered.slice(0, 20);
        document.getElementById('eqtl-table').innerHTML = `
            <table style="width: 100%; font-size: 0.8rem; border-collapse: collapse;">
                <thead>
                    <tr style="border-bottom: 2px solid #ddd;">
                        <th style="padding: 0.4rem; text-align: left;">Gene</th>
                        <th style="padding: 0.4rem; text-align: left;">Variant</th>
                        <th style="padding: 0.4rem; text-align: left;">Cell Type</th>
                        <th style="padding: 0.4rem; text-align: right;">p-value</th>
                    </tr>
                </thead>
                <tbody>
                    ${top20.map(d => `
                        <tr style="border-bottom: 1px solid #eee;">
                            <td style="padding: 0.3rem;">${d.gene}</td>
                            <td style="padding: 0.3rem; font-size: 0.7rem;">${d.variant}</td>
                            <td style="padding: 0.3rem; font-size: 0.7rem;">${d.celltype}</td>
                            <td style="padding: 0.3rem; text-align: right;">${d.pvalue?.toExponential(2)}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;

        // Cell type distribution
        if (data.cell_types && cellType === 'all') {
            const ctCounts = {};
            filtered.forEach(d => {
                ctCounts[d.celltype] = (ctCounts[d.celltype] || 0) + 1;
            });
            const sortedCts = Object.entries(ctCounts).sort((a, b) => b[1] - a[1]).slice(0, 20);

            Plotly.newPlot('eqtl-celltype-bar', [{
                type: 'bar',
                y: sortedCts.map(d => d[0]).reverse(),
                x: sortedCts.map(d => d[1]).reverse(),
                orientation: 'h',
                marker: { color: '#3b82f6' }
            }], {
                title: 'eQTLs by Cell Type (Top 20)',
                xaxis: { title: 'Number of eQTLs' },
                margin: { l: 150, r: 20, t: 40, b: 50 },
                font: { family: 'Inter, sans-serif' }
            }, { responsive: true });
        }
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
                                <td>Cell Types</td>
                                <td>66 cell types × (43 CytoSig + 1,170 SecAct proteins)</td>
                                <td>80,058</td>
                            </tr>
                            <tr>
                                <td>Age Correlations</td>
                                <td>1,213 proteins × Age correlation per cell type</td>
                                <td>~80,000</td>
                            </tr>
                            <tr>
                                <td>BMI Correlations</td>
                                <td>1,213 proteins × BMI correlation per cell type</td>
                                <td>~80,000</td>
                            </tr>
                            <tr>
                                <td>Age/BMI Stratified</td>
                                <td>1,213 proteins × 66 cell types × bins (boxplots)</td>
                                <td>~500,000</td>
                            </tr>
                            <tr>
                                <td>Disease</td>
                                <td>Disease vs healthy differential across 12+ diseases</td>
                                <td>54,137</td>
                            </tr>
                            <tr>
                                <td>Differential</td>
                                <td>Sex, smoking differential activity analysis</td>
                                <td>Per comparison</td>
                            </tr>
                            <tr>
                                <td>Treatment Response</td>
                                <td>Responder vs non-responder ML prediction (8 diseases)</td>
                                <td>288</td>
                            </tr>
                            <tr>
                                <td>Disease Flow</td>
                                <td>Sankey diagram: 20 diseases × 6 disease groups × 3 cohorts</td>
                                <td>136</td>
                            </tr>
                            <tr>
                                <td>Validation</td>
                                <td>Cross-cohort consistency (main, validation, external)</td>
                                <td>23</td>
                            </tr>
                            <tr>
                                <td>Longitudinal</td>
                                <td>Time-course cytokine dynamics during treatment</td>
                                <td>Per disease</td>
                            </tr>
                            <tr>
                                <td>Severity</td>
                                <td>Disease severity correlation with cytokine activity</td>
                                <td>Per disease</td>
                            </tr>
                            <tr>
                                <td>Cell Drivers</td>
                                <td>Top driving cell types per disease-cytokine pair</td>
                                <td>Per disease</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

            </div>
        `;
    },

    async loadInflamCelltypes(content) {
        content.innerHTML = `
            <div class="controls" style="display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem;">
                <div class="control-group" style="position: relative;">
                    <label>Search Signature</label>
                    <input type="text" id="inflam-ct-search" class="search-input"
                           placeholder="Type to search (e.g., IFNG, IL6)"
                           value="IFNG"
                           style="width: 180px;">
                    <div id="inflam-ct-suggestions" style="position: absolute; top: 100%; left: 0; width: 180px; max-height: 200px; overflow-y: auto; background: white; border: 1px solid #ddd; border-radius: 4px; display: none; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"></div>
                </div>
            </div>

            <div class="viz-grid">
                <!-- Sub-panel 1: Activity Profile with Search -->
                <div class="sub-panel">
                    <div class="panel-header">
                        <h3>Cell Type Activity Profile</h3>
                        <p>Mean cytokine activity across immune cell populations</p>
                    </div>
                    <div id="inflam-activity-profile" class="plot-container" style="height: 450px;">
                        <p class="loading">Loading...</p>
                    </div>
                </div>

                <!-- Sub-panel 2: Activity Heatmap -->
                <div class="sub-panel">
                    <div class="panel-header">
                        <h3>Activity Heatmap</h3>
                        <p>Top variable cell types × signatures</p>
                    </div>
                    <div id="inflam-celltype-heatmap" class="plot-container" style="height: 500px;"></div>
                </div>
            </div>
        `;

        // Load activity data for current signature type
        const activityData = await API.get('/inflammation/activity', { signature_type: this.signatureType });
        this.inflamActivityData = { [this.signatureType]: activityData };

        // Get unique signatures for autocomplete
        if (activityData && activityData.length > 0) {
            this.inflamCTSignatures = {
                [this.signatureType]: [...new Set(activityData.map(d => d.signature))].sort()
            };
        }

        // Set up autocomplete
        this.setupInflamCTAutocomplete();

        // Initial render
        this.updateInflamCelltypes();
    },

    setupInflamCTAutocomplete() {
        const input = document.getElementById('inflam-ct-search');
        const suggestionsDiv = document.getElementById('inflam-ct-suggestions');
        if (!input || !suggestionsDiv) return;

        input.addEventListener('focus', () => this.showInflamCTSuggestions());
        input.addEventListener('input', () => this.showInflamCTSuggestions());
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                suggestionsDiv.style.display = 'none';
                this.updateInflamActivityProfile();
            }
        });
        input.addEventListener('blur', () => {
            setTimeout(() => { suggestionsDiv.style.display = 'none'; }, 200);
        });
    },

    showInflamCTSuggestions() {
        const input = document.getElementById('inflam-ct-search');
        const suggestionsDiv = document.getElementById('inflam-ct-suggestions');
        if (!input || !suggestionsDiv) return;

        const sigType = this.signatureType;
        const signatures = this.inflamCTSignatures?.[sigType] || [];
        const query = input.value.toLowerCase();

        const filtered = signatures.filter(s => s.toLowerCase().includes(query)).slice(0, 15);

        if (filtered.length === 0) {
            suggestionsDiv.style.display = 'none';
            return;
        }

        suggestionsDiv.innerHTML = filtered.map(s =>
            `<div style="padding: 6px 10px; cursor: pointer; border-bottom: 1px solid #eee;"
                 onmouseover="this.style.background='#f0f0f0'"
                 onmouseout="this.style.background='white'"
                 onclick="AtlasDetailPage.selectInflamCTSignature('${s}')">${s}</div>`
        ).join('');
        suggestionsDiv.style.display = 'block';
    },

    selectInflamCTSignature(sig) {
        const input = document.getElementById('inflam-ct-search');
        const suggestionsDiv = document.getElementById('inflam-ct-suggestions');
        if (input) input.value = sig;
        if (suggestionsDiv) suggestionsDiv.style.display = 'none';
        this.updateInflamActivityProfile();
    },

    async updateInflamCelltypes() {
        const sigType = this.signatureType;

        // Reset search to default on signature type change
        const input = document.getElementById('inflam-ct-search');
        if (input) input.value = 'IFNG';

        // Load data for this signature type if not cached
        if (!this.inflamActivityData[sigType]) {
            const container = document.getElementById('inflam-activity-profile');
            if (container) container.innerHTML = '<p class="loading">Loading...</p>';

            const data = await API.get('/inflammation/activity', { signature_type: sigType });
            this.inflamActivityData[sigType] = data;

            // Update signatures for autocomplete
            if (data && data.length > 0) {
                this.inflamCTSignatures[sigType] = [...new Set(data.map(d => d.signature))].sort();
            }
        }

        // Update both visualizations
        this.updateInflamActivityProfile();
        this.updateInflamCelltypeHeatmap();
    },

    async updateInflamActivityProfile() {
        const container = document.getElementById('inflam-activity-profile');
        if (!container) return;

        const sigType = this.signatureType;
        const signature = document.getElementById('inflam-ct-search')?.value || 'IFNG';

        const data = this.inflamActivityData?.[sigType];
        if (!data || data.length === 0) {
            container.innerHTML = '<p class="loading">No activity data available</p>';
            return;
        }

        // Filter to selected signature
        const filtered = data.filter(d => d.signature === signature);
        filtered.sort((a, b) => b.mean_activity - a.mean_activity);

        // Take top 30 cell types
        const top30 = filtered.slice(0, 30);

        if (top30.length === 0) {
            container.innerHTML = `<p class="loading">No data available for '${signature}' [${sigType}]</p>`;
            return;
        }

        Plotly.newPlot(container, [{
            y: top30.map(d => d.cell_type),
            x: top30.map(d => d.mean_activity),
            type: 'bar',
            orientation: 'h',
            marker: {
                color: top30.map(d => d.mean_activity),
                colorscale: [[0, '#a8d4e6'], [0.5, '#f5f5f5'], [1, '#f4a6a6']],
                cmid: 0
            },
            hovertemplate: '<b>%{y}</b><br>Activity: %{x:.2f}<br>Samples: %{customdata}<extra></extra>',
            customdata: top30.map(d => d.n_samples || 'N/A')
        }], {
            title: `${signature} [${sigType}] Activity`,
            margin: { l: 180, r: 30, t: 40, b: 50 },
            xaxis: { title: 'Mean Activity (z-score)' },
            height: 450,
            font: { family: 'Inter, sans-serif' }
        }, { responsive: true });
    },

    async updateInflamCelltypeHeatmap() {
        const container = document.getElementById('inflam-celltype-heatmap');
        if (!container) return;

        const sigType = this.signatureType;
        const data = this.inflamActivityData?.[sigType];

        if (!data || data.length === 0) {
            container.innerHTML = '<p class="loading">No heatmap data available</p>';
            return;
        }

        // Build lookup index
        const dataIndex = {};
        data.forEach(d => {
            const key = `${d.cell_type}|${d.signature}`;
            dataIndex[key] = d.mean_activity;
        });

        const cellTypes = [...new Set(data.map(d => d.cell_type))];
        const allSignatures = [...new Set(data.map(d => d.signature))];

        // Find most variable signatures
        const sigVariance = {};
        allSignatures.forEach(sig => {
            const vals = cellTypes.map(ct => dataIndex[`${ct}|${sig}`] || 0);
            if (vals.length === 0) return;
            const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
            const variance = vals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / vals.length;
            sigVariance[sig] = variance;
        });

        // Limit to top 50 for SecAct, all for CytoSig
        const maxSigs = sigType === 'CytoSig' ? allSignatures.length : 50;
        const topSignatures = Object.entries(sigVariance)
            .sort((a, b) => b[1] - a[1])
            .slice(0, maxSigs)
            .map(d => d[0])
            .sort();

        // Find most variable cell types
        const ctVariance = {};
        cellTypes.forEach(ct => {
            const vals = topSignatures.map(sig => dataIndex[`${ct}|${sig}`] || 0);
            if (vals.length === 0) return;
            const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
            const variance = vals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / vals.length;
            ctVariance[ct] = variance;
        });

        // Top 25 cell types
        const topCts = Object.entries(ctVariance)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 25)
            .map(d => d[0]);

        // Build z-matrix
        const z = topCts.map(ct =>
            topSignatures.map(sig => dataIndex[`${ct}|${sig}`] || 0)
        );

        Plotly.newPlot(container, [{
            z: z,
            x: topSignatures,
            y: topCts,
            type: 'heatmap',
            colorscale: [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
            zmid: 0,
            hovertemplate: '<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>'
        }], {
            title: `${sigType} Activity: Top ${topCts.length} Cell Types × ${topSignatures.length} Signatures`,
            margin: { l: 180, r: 30, t: 40, b: 100 },
            xaxis: { tickangle: -45, tickfont: { size: 10 } },
            yaxis: { tickfont: { size: 10 } },
            height: 500,
            font: { family: 'Inter, sans-serif' }
        }, { responsive: true });
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
                                <td>Organ Signatures</td>
                                <td>Mean cytokine activities per organ (normal tissue)</td>
                                <td>${stats?.n_organ_signatures?.toLocaleString() || 'N/A'}</td>
                            </tr>
                            <tr>
                                <td>Cell Type Signatures</td>
                                <td>Mean activities per cell type across organs</td>
                                <td>${stats?.n_celltype_signatures?.toLocaleString() || 'N/A'}</td>
                            </tr>
                            <tr>
                                <td>Tumor vs Adjacent</td>
                                <td>Differential activity between tumor and adjacent tissue</td>
                                <td>${stats?.n_cancer_celltypes || 214} cell types</td>
                            </tr>
                            <tr>
                                <td>Normal vs Cancer</td>
                                <td>Matched organ-cancer type comparisons</td>
                                <td>13 organ-cancer pairs</td>
                            </tr>
                            <tr>
                                <td>Immune Infiltration</td>
                                <td>Immune cell proportions and activity per cancer type</td>
                                <td>Per cancer type</td>
                            </tr>
                            <tr>
                                <td>T Cell Exhaustion</td>
                                <td>Exhausted vs non-exhausted T cell signatures</td>
                                <td>Tex vs non-Tex</td>
                            </tr>
                            <tr>
                                <td>CAF Signatures</td>
                                <td>Cancer-associated fibroblast subtype activities</td>
                                <td>Per cancer type</td>
                            </tr>
                            <tr>
                                <td>Adjacent Tissue</td>
                                <td>Field effect analysis in tumor-adjacent tissue</td>
                                <td>Per cancer type</td>
                            </tr>
                            <tr>
                                <td>Pan-Cancer Signatures</td>
                                <td>Signatures consistent across multiple cancers</td>
                                <td>Cross-cancer</td>
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

    renderBiochemHeatmap(containerId, data, orderedProteins = null) {
        // Group by protein and biochem marker
        // API format: {signature (protein name), variable (blood marker), rho, pvalue, qvalue, n_samples}
        const proteins = orderedProteins || [...new Set(data.map(d => d.signature))];
        const markers = [...new Set(data.map(d => d.variable))].filter(m => m);

        if (markers.length === 0 || proteins.length === 0) {
            document.getElementById(containerId).innerHTML = '<p class="loading">No data available</p>';
            return;
        }

        const z = proteins.map(protein =>
            markers.map(m => {
                const item = data.find(d => d.signature === protein && d.variable === m);
                return item ? (item.rho ?? 0) : 0;
            })
        );

        const sigType = this.signatureType;

        Heatmap.create(containerId, {
            z, x: markers, y: proteins,
            colorscale: 'RdBu', reversescale: true,
        }, {
            title: `Biochemistry Correlations [${sigType}]`,
            xLabel: 'Blood Marker',
            yLabel: 'Protein/Cytokine',
            colorbarTitle: 'Spearman ρ',
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
