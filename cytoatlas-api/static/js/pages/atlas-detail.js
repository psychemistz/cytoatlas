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
                { id: 'sankey', label: 'Disease Flow', icon: '&#128260;' },
                { id: 'disease', label: 'Disease', icon: '&#129658;' },
                { id: 'severity', label: 'Severity', icon: '&#128200;' },
                { id: 'differential', label: 'Differential', icon: '&#128209;' },
                { id: 'treatment', label: 'Treatment Response', icon: '&#128137;' },
                { id: 'validation', label: 'Cohort Validation', icon: '&#9989;' },
                { id: 'longitudinal', label: 'Longitudinal', icon: '&#128197;' },
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
                <p>Top 500 cytokine-metabolite correlations from plasma metabolomics data (${this.signatureType})</p>
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
                        <h4>Metabolite Correlation Network</h4>
                        <p>Force-directed graph of cytokine-metabolite associations</p>
                    </div>
                    <div id="metabolite-network" class="plot-container" style="height: 550px;"></div>
                </div>
                <div class="sub-panel">
                    <div class="panel-header">
                        <h4>Top Metabolite Correlations</h4>
                        <p>Ranked by absolute correlation strength</p>
                    </div>
                    <div id="metabolite-lollipop" class="plot-container" style="height: 550px;"></div>
                </div>
            </div>
        `;

        // Load metabolite data with current signature type
        this.metaboliteData = await API.get('/cima/correlations/metabolites', { signature_type: this.signatureType, limit: 500 });
        this.updateMetabolitePlots();
    },

    updateMetabolitePlots() {
        // API returns list directly, not {correlations: [...]}
        const data = Array.isArray(this.metaboliteData) ? this.metaboliteData : (this.metaboliteData?.correlations || []);
        if (!data || data.length === 0) {
            document.getElementById('metabolite-network').innerHTML = '<p class="loading">No metabolite data</p>';
            document.getElementById('metabolite-lollipop').innerHTML = '<p class="loading">No metabolite data</p>';
            return;
        }

        const threshold = parseFloat(document.getElementById('metab-threshold')?.value || '0.3');
        const category = document.getElementById('metab-category')?.value || 'all';

        // Filter by threshold
        let filtered = data.filter(d => Math.abs(d.rho || d.correlation) >= threshold);

        // Filter by category using prefix matching (like visualization/index.html)
        if (category !== 'all') {
            const categoryPrefixes = {
                'lipid': ['PS ', 'PC ', 'PE ', 'PG ', 'PI ', 'SM ', 'LPC', 'Cer', 'TG ', 'DG ', 'LPE', 'FA '],
                'amino_acid': ['L-Histidine', 'L-Tryptophan', '3-Nitrotyrosine'],
                'carbohydrate': ['Gluconic', 'Glucosamine', 'D-Gluconolactone', 'Glycolic', 'Citric', 'Erythronic'],
                'nucleotide': ['ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'UTP', 'CTP', 'NAD', 'NADP'],
                'cofactor': ['CoA', 'FAD', 'FMN', 'B12', 'Biotin', 'Folate', 'Thiamine']
            };
            const prefixes = categoryPrefixes[category] || [];
            filtered = filtered.filter(d => {
                const metab = d.metabolite || d.feature || '';
                return prefixes.some(p => metab.startsWith(p));
            });
        }

        // Limit for network visualization
        const networkData = filtered.slice(0, 100);

        // D3 Force-directed network
        this.renderMetaboliteNetwork(networkData, threshold);

        // Sort by absolute correlation for lollipop
        filtered.sort((a, b) => Math.abs(b.rho || b.correlation) - Math.abs(a.rho || a.correlation));
        const top50 = filtered.slice(0, 50);

        // Lollipop chart
        if (top50.length > 0) {
            const labels = top50.map(d => `${d.signature} × ${d.metabolite}`);
            const values = top50.map(d => d.rho || d.correlation);
            const colors = values.map(v => v >= 0 ? '#b2182b' : '#2166ac');

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
                margin: { l: 180, r: 20, t: 40, b: 40 },
                font: { family: 'Inter, sans-serif' }
            }, { responsive: true });
        } else {
            document.getElementById('metabolite-lollipop').innerHTML = '<p class="loading">No correlations above threshold</p>';
        }
    },

    renderMetaboliteNetwork(data, threshold) {
        const container = document.getElementById('metabolite-network');
        if (!container) return;

        // Clear previous content
        container.innerHTML = '';

        if (!data || data.length === 0) {
            container.innerHTML = '<p class="loading">No correlations above threshold</p>';
            return;
        }

        // Build nodes and links
        const nodesSet = new Set();
        data.forEach(d => {
            nodesSet.add(d.signature);
            nodesSet.add(d.metabolite);
        });

        const nodes = Array.from(nodesSet).map(id => ({
            id,
            type: data.some(d => d.signature === id) ? 'cytokine' : 'metabolite'
        }));

        const links = data.map(d => ({
            source: d.signature,
            target: d.metabolite,
            value: Math.abs(d.rho),
            sign: (d.rho || 0) > 0 ? 'positive' : 'negative'
        }));

        // Set up SVG
        const width = container.clientWidth || 500;
        const height = 500;

        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        // Force simulation
        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(80))
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(width / 2, height / 2));

        // Draw links
        const link = svg.append('g')
            .selectAll('line')
            .data(links)
            .join('line')
            .attr('stroke', d => d.sign === 'positive' ? '#b2182b' : '#2166ac')
            .attr('stroke-opacity', 0.6)
            .attr('stroke-width', d => d.value * 3);

        // Draw nodes
        const node = svg.append('g')
            .selectAll('circle')
            .data(nodes)
            .join('circle')
            .attr('r', d => d.type === 'cytokine' ? 8 : 5)
            .attr('fill', d => d.type === 'cytokine' ? '#1f77b4' : '#ff7f0e')
            .call(d3.drag()
                .on('start', (event) => {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    event.subject.fx = event.subject.x;
                    event.subject.fy = event.subject.y;
                })
                .on('drag', (event) => {
                    event.subject.fx = event.x;
                    event.subject.fy = event.y;
                })
                .on('end', (event) => {
                    if (!event.active) simulation.alphaTarget(0);
                    event.subject.fx = null;
                    event.subject.fy = null;
                }));

        node.append('title').text(d => d.id);

        // Labels for cytokines only
        const labels = svg.append('g')
            .selectAll('text')
            .data(nodes.filter(d => d.type === 'cytokine'))
            .join('text')
            .text(d => d.id)
            .attr('font-size', '10px')
            .attr('dx', 10)
            .attr('dy', 3);

        // Legend annotations
        const anno1 = svg.append('text').attr('x', 10).attr('y', height - 45).attr('font-size', '11px');
        anno1.append('tspan').attr('fill', '#666').text('Lines: ');
        anno1.append('tspan').attr('fill', '#b2182b').text('red');
        anno1.append('tspan').attr('fill', '#666').text(' = positive, ');
        anno1.append('tspan').attr('fill', '#2166ac').text('blue');
        anno1.append('tspan').attr('fill', '#666').text(' = negative');

        const anno2 = svg.append('text').attr('x', 10).attr('y', height - 28).attr('font-size', '11px');
        anno2.append('tspan').attr('fill', '#666').text('Nodes: ');
        anno2.append('tspan').attr('fill', '#1f77b4').text('blue');
        anno2.append('tspan').attr('fill', '#666').text(' = cytokine, ');
        anno2.append('tspan').attr('fill', '#ff7f0e').text('orange');
        anno2.append('tspan').attr('fill', '#666').text(' = metabolite');

        const anno3 = svg.append('text').attr('x', 10).attr('y', height - 11).attr('font-size', '11px').attr('fill', '#666');
        anno3.text(`Showing ${data.length} correlations (|ρ| ≥ ${threshold})`);

        // Update positions on tick
        simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            labels
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });
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
                            <option value="blood_type|B|AB">B vs AB</option>
                            <option value="blood_type|A|AB">A vs AB</option>
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
            document.getElementById('differential-bar').innerHTML = '<p class="loading">No differential data</p>';
            return;
        }

        const comparisonValue = document.getElementById('diff-comparison')?.value || 'sex|Female|Male';
        const [comparison, group1, group2] = comparisonValue.split('|');

        // Filter data by exact match of comparison, group1, group2
        let filtered = data.filter(d =>
            d.comparison === comparison &&
            d.group1 === group1 &&
            d.group2 === group2 &&
            !isNaN(d.log2fc) &&
            d.neg_log10_pval !== null && d.neg_log10_pval !== undefined
        );

        // For SecAct (many proteins), limit to top 200 by significance score
        if (this.signatureType === 'SecAct' && filtered.length > 200) {
            filtered = [...filtered].sort((a, b) => {
                const scoreA = (a.neg_log10_pval || 0) * Math.abs(a.log2fc || 0);
                const scoreB = (b.neg_log10_pval || 0) * Math.abs(b.log2fc || 0);
                return scoreB - scoreA;
            }).slice(0, 200);
        }

        if (filtered.length === 0) {
            document.getElementById('differential-volcano').innerHTML = `<p class="loading">No data for ${group1} vs ${group2}</p>`;
            document.getElementById('differential-bar').innerHTML = '';
            return;
        }

        // Update subtitle
        const subtitle = document.getElementById('volcano-subtitle');
        if (subtitle) subtitle.textContent = `Positive log2FC = higher in ${group1}, Negative = higher in ${group2}`;

        // Categorize points: significant up, significant down, not significant
        const significantUp = filtered.filter(d => {
            const qval = d.q_value ?? 1;
            return qval < 0.05 && d.log2fc > 0.5;
        });
        const significantDown = filtered.filter(d => {
            const qval = d.q_value ?? 1;
            return qval < 0.05 && d.log2fc < -0.5;
        });
        const notSignificant = filtered.filter(d => {
            const qval = d.q_value ?? 1;
            return qval >= 0.05 || Math.abs(d.log2fc) <= 0.5;
        });

        // Identify top hits by significance score (|log2FC| * -log10(p))
        const allSignificant = [...significantUp, ...significantDown];
        const scoredSignificant = allSignificant.map(d => ({
            ...d,
            score: Math.abs(d.log2fc || 0) * (d.neg_log10_pval || 0)
        }));
        scoredSignificant.sort((a, b) => b.score - a.score);
        const topHits = scoredSignificant.slice(0, 10);  // Top 10 hits for annotation
        const topHitSignatures = new Set(topHits.map(d => d.signature));

        // Separate top hits from other significant points
        const topHitsUp = significantUp.filter(d => topHitSignatures.has(d.signature));
        const topHitsDown = significantDown.filter(d => topHitSignatures.has(d.signature));
        const otherSigUp = significantUp.filter(d => !topHitSignatures.has(d.signature));
        const otherSigDown = significantDown.filter(d => !topHitSignatures.has(d.signature));

        // Dynamic x-axis range
        const maxAbsFC = Math.max(3, Math.ceil(Math.max(...filtered.map(d => Math.abs(d.log2fc || 0)))));
        const maxNegLogP = Math.max(10, ...filtered.map(d => d.neg_log10_pval || 0));

        // Build traces
        const traces = [];

        // Non-significant points (gray, small, no text)
        if (notSignificant.length > 0) {
            traces.push({
                type: 'scatter',
                mode: 'markers',
                name: 'Not significant',
                x: notSignificant.map(d => d.log2fc),
                y: notSignificant.map(d => d.neg_log10_pval),
                text: notSignificant.map(d => d.signature),
                marker: { color: '#cccccc', size: 6, opacity: 0.5 },
                hovertemplate: '<b>%{text}</b><br>log2FC: %{x:.3f}<br>-log10(p): %{y:.2f}<extra></extra>'
            });
        }

        // Other significant up (colored, medium, no text)
        if (otherSigUp.length > 0) {
            traces.push({
                type: 'scatter',
                mode: 'markers',
                name: `Higher in ${group1}`,
                x: otherSigUp.map(d => d.log2fc),
                y: otherSigUp.map(d => d.neg_log10_pval),
                text: otherSigUp.map(d => d.signature),
                marker: { color: '#f4a6a6', size: 8, opacity: 0.7 },
                hovertemplate: '<b>%{text}</b><br>log2FC: %{x:.3f}<br>-log10(p): %{y:.2f}<extra></extra>'
            });
        }

        // Other significant down (colored, medium, no text)
        if (otherSigDown.length > 0) {
            traces.push({
                type: 'scatter',
                mode: 'markers',
                name: `Higher in ${group2}`,
                x: otherSigDown.map(d => d.log2fc),
                y: otherSigDown.map(d => d.neg_log10_pval),
                text: otherSigDown.map(d => d.signature),
                marker: { color: '#a8d4e6', size: 8, opacity: 0.7 },
                hovertemplate: '<b>%{text}</b><br>log2FC: %{x:.3f}<br>-log10(p): %{y:.2f}<extra></extra>'
            });
        }

        // Top hits up (large, with text labels)
        if (topHitsUp.length > 0) {
            traces.push({
                type: 'scatter',
                mode: 'markers+text',
                name: `Top hits (${group1})`,
                x: topHitsUp.map(d => d.log2fc),
                y: topHitsUp.map(d => d.neg_log10_pval),
                text: topHitsUp.map(d => d.signature),
                textposition: 'top center',
                textfont: { size: 10, color: '#b91c1c' },
                marker: { color: '#dc2626', size: 12, opacity: 0.9, line: { color: '#fff', width: 1 } },
                hovertemplate: '<b>%{text}</b><br>log2FC: %{x:.3f}<br>-log10(p): %{y:.2f}<extra></extra>'
            });
        }

        // Top hits down (large, with text labels)
        if (topHitsDown.length > 0) {
            traces.push({
                type: 'scatter',
                mode: 'markers+text',
                name: `Top hits (${group2})`,
                x: topHitsDown.map(d => d.log2fc),
                y: topHitsDown.map(d => d.neg_log10_pval),
                text: topHitsDown.map(d => d.signature),
                textposition: 'top center',
                textfont: { size: 10, color: '#1e40af' },
                marker: { color: '#2563eb', size: 12, opacity: 0.9, line: { color: '#fff', width: 1 } },
                hovertemplate: '<b>%{text}</b><br>log2FC: %{x:.3f}<br>-log10(p): %{y:.2f}<extra></extra>'
            });
        }

        // Volcano plot with multiple traces
        Plotly.newPlot('differential-volcano', traces, {
            xaxis: {
                title: `log2 Fold Change (${group1} / ${group2})`,
                zeroline: true,
                zerolinecolor: '#ccc',
                range: [-maxAbsFC, maxAbsFC]
            },
            yaxis: { title: '-log10(p-value)', range: [0, maxNegLogP * 1.15] },
            shapes: [
                // Horizontal line at p=0.05
                { type: 'line', x0: -maxAbsFC, x1: maxAbsFC, y0: -Math.log10(0.05), y1: -Math.log10(0.05),
                  line: { color: '#999', dash: 'dash', width: 1 } },
                // Vertical lines at FC thresholds
                { type: 'line', x0: -0.5, x1: -0.5, y0: 0, y1: maxNegLogP * 1.1,
                  line: { color: '#999', dash: 'dash', width: 1 } },
                { type: 'line', x0: 0.5, x1: 0.5, y0: 0, y1: maxNegLogP * 1.1,
                  line: { color: '#999', dash: 'dash', width: 1 } }
            ],
            annotations: [
                { x: -maxAbsFC * 0.8, y: -0.08, xref: 'x', yref: 'paper',
                  text: `← Higher in ${group2}`, showarrow: false, font: { size: 11, color: '#2563eb' } },
                { x: maxAbsFC * 0.8, y: -0.08, xref: 'x', yref: 'paper',
                  text: `Higher in ${group1} →`, showarrow: false, font: { size: 11, color: '#dc2626' } }
            ],
            legend: { orientation: 'h', y: -0.18, x: 0.5, xanchor: 'center' },
            margin: { l: 60, r: 30, t: 30, b: 80 },
            font: { family: 'Inter, sans-serif' }
        }, { responsive: true });

        // Bar chart - top differential by significance score
        const scored = filtered.map(d => ({
            ...d,
            score: Math.abs(d.log2fc || 0) * (d.neg_log10_pval || 0)
        }));
        scored.sort((a, b) => b.score - a.score);
        const top20 = scored.slice(0, 20).reverse();  // Reverse for horizontal bar (top at top)

        Plotly.newPlot('differential-bar', [{
            type: 'bar',
            orientation: 'h',
            y: top20.map(d => d.signature),
            x: top20.map(d => d.log2fc),
            marker: {
                color: top20.map(d => d.log2fc > 0 ? '#f4a6a6' : '#a8d4e6')
            },
            text: top20.map(d => d.log2fc.toFixed(2)),
            textposition: 'outside',
            textfont: { size: 9 },
            hovertemplate: '<b>%{y}</b><br>log2FC: %{x:.3f}<br>q = %{customdata}<extra></extra>',
            customdata: top20.map(d => d.q_value?.toExponential(2) || 'N/A')
        }], {
            xaxis: { title: 'log2 Fold Change', zeroline: true, zerolinecolor: '#ccc' },
            yaxis: { automargin: true, tickfont: { size: 10 } },
            margin: { l: 120, r: 50, t: 30, b: 50 },
            font: { family: 'Inter, sans-serif' }
        }, { responsive: true });
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
                    <label>Analysis Type</label>
                    <select id="multiomics-analysis" class="filter-select" onchange="AtlasDetailPage.updateMultiomicsPlot()">
                        <option value="correlation">Correlation Network</option>
                        <option value="heatmap">Top Correlations Bar</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Signature Subset</label>
                    <select id="multiomics-cytokines" class="filter-select" onchange="AtlasDetailPage.updateMultiomicsPlot()">
                        <option value="all">All Signatures</option>
                        <option value="inflammatory">Inflammatory (IL-1B, IL-6, TNF)</option>
                        <option value="regulatory">Regulatory (IL-10, TGF-β, IL-4)</option>
                        <option value="th17">Th17 axis (IL-17A, IL-21, IL-22)</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Min |Correlation|</label>
                    <select id="multiomics-threshold" class="filter-select" onchange="AtlasDetailPage.updateMultiomicsPlot()">
                        <option value="0.2">0.2</option>
                        <option value="0.3">0.3</option>
                        <option value="0.4" selected>0.4</option>
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
                    <div id="multiomics-network" class="plot-container" style="height: 550px;"></div>
                </div>
                <div class="sub-panel">
                    <div class="panel-header">
                        <h4>Summary</h4>
                    </div>
                    <div id="multiomics-summary" style="padding: 1rem; max-height: 550px; overflow-y: auto;"></div>
                </div>
            </div>
        `;

        this.updateMultiomicsPlot();
    },

    async updateMultiomicsPlot() {
        const sigType = this.signatureType || 'CytoSig';
        const analysisType = document.getElementById('multiomics-analysis')?.value || 'correlation';
        const subset = document.getElementById('multiomics-cytokines')?.value || 'all';
        const threshold = parseFloat(document.getElementById('multiomics-threshold')?.value || '0.4');

        const container = document.getElementById('multiomics-network');
        const summaryContainer = document.getElementById('multiomics-summary');

        if (!container) return;
        container.innerHTML = '<div class="loading">Loading multi-omics data...</div>';

        // Load biochemistry and metabolite correlations
        const [biochemData, metabData] = await Promise.all([
            API.get('/cima/correlations/biochemistry', { signature_type: sigType }),
            API.get('/cima/correlations/metabolites', { signature_type: sigType, limit: 500 })
        ]);

        // Define cytokine subsets
        const subsetMap = {
            'inflammatory': ['IL1B', 'IL6', 'TNF', 'IL1A', 'IL18'],
            'regulatory': ['IL10', 'TGFB1', 'IL4', 'IL13', 'IL35'],
            'th17': ['IL17A', 'IL17F', 'IL21', 'IL22', 'IL23A']
        };
        const cytokineFilter = subset === 'all' ? null : subsetMap[subset] || null;

        // Filter and combine data (API returns arrays directly)
        let biochemFiltered = (biochemData || []).filter(d => Math.abs(d.rho) >= threshold);
        let metabFiltered = (Array.isArray(metabData) ? metabData : (metabData?.correlations || [])).filter(d => Math.abs(d.rho) >= threshold);

        if (cytokineFilter) {
            biochemFiltered = biochemFiltered.filter(d => cytokineFilter.includes(d.signature));
            metabFiltered = metabFiltered.filter(d => cytokineFilter.includes(d.signature));
        }

        // Limit to top correlations for visualization
        biochemFiltered = biochemFiltered.sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho)).slice(0, 50);
        metabFiltered = metabFiltered.sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho)).slice(0, 50);

        // Build nodes
        const nodesMap = new Map();

        biochemFiltered.forEach(d => {
            const cytokine = d.signature;
            const feature = d.variable;
            if (!nodesMap.has(cytokine)) nodesMap.set(cytokine, { id: cytokine, group: 'cytokine', color: '#1f77b4' });
            if (!nodesMap.has(feature)) nodesMap.set(feature, { id: feature, group: 'biochem', color: '#2ca02c' });
        });

        metabFiltered.forEach(d => {
            const cytokine = d.signature;
            const feature = d.metabolite;
            if (!nodesMap.has(cytokine)) nodesMap.set(cytokine, { id: cytokine, group: 'cytokine', color: '#1f77b4' });
            if (feature && !nodesMap.has(feature)) nodesMap.set(feature, { id: feature, group: 'metabolite', color: '#ff7f0e' });
        });

        const nodes = Array.from(nodesMap.values());

        // Build edges
        const edges = [
            ...biochemFiltered.map(d => ({ source: d.signature, target: d.variable, correlation: d.rho, type: 'biochem' })),
            ...metabFiltered.filter(d => d.metabolite).map(d => ({ source: d.signature, target: d.metabolite, correlation: d.rho, type: 'metabolite' }))
        ];

        container.innerHTML = '';

        if (nodes.length === 0 || edges.length === 0) {
            container.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">No correlations above threshold for selected filters</p>';
            if (summaryContainer) summaryContainer.innerHTML = '';
            return;
        }

        if (analysisType === 'correlation') {
            // D3 force-directed network
            const containerWidth = container.offsetWidth || 600;
            const width = containerWidth - 40;
            const height = 500;

            const svg = d3.select(container)
                .append('svg')
                .attr('width', width)
                .attr('height', height);

            const simulation = d3.forceSimulation(nodes)
                .force('link', d3.forceLink(edges).id(d => d.id).distance(80))
                .force('charge', d3.forceManyBody().strength(-200))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(25));

            const link = svg.append('g')
                .selectAll('line')
                .data(edges)
                .enter().append('line')
                .attr('stroke', d => d.correlation > 0 ? '#b2182b' : '#2166ac')
                .attr('stroke-opacity', 0.6)
                .attr('stroke-width', d => Math.abs(d.correlation) * 4);

            const node = svg.append('g')
                .selectAll('g')
                .data(nodes)
                .enter().append('g')
                .call(d3.drag()
                    .on('start', (event, d) => {
                        if (!event.active) simulation.alphaTarget(0.3).restart();
                        d.fx = d.x; d.fy = d.y;
                    })
                    .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
                    .on('end', (event, d) => {
                        if (!event.active) simulation.alphaTarget(0);
                        d.fx = null; d.fy = null;
                    }));

            node.append('circle')
                .attr('r', d => d.group === 'cytokine' ? 10 : 6)
                .attr('fill', d => d.color)
                .attr('stroke', '#fff')
                .attr('stroke-width', 1.5);

            node.append('title').text(d => d.id);

            // Only label cytokines to reduce clutter
            node.filter(d => d.group === 'cytokine')
                .append('text')
                .text(d => d.id)
                .attr('x', 12)
                .attr('y', 4)
                .style('font-size', '10px')
                .style('fill', '#333');

            simulation.on('tick', () => {
                link
                    .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
                node.attr('transform', d => `translate(${d.x},${d.y})`);
            });

            // Add annotation legend
            const anno1 = svg.append('text').attr('x', 10).attr('y', height - 45).attr('font-size', '11px');
            anno1.append('tspan').attr('fill', '#666').text('Lines: ');
            anno1.append('tspan').attr('fill', '#b2182b').text('red');
            anno1.append('tspan').attr('fill', '#666').text(' = positive, ');
            anno1.append('tspan').attr('fill', '#2166ac').text('blue');
            anno1.append('tspan').attr('fill', '#666').text(' = negative');

            const anno2 = svg.append('text').attr('x', 10).attr('y', height - 28).attr('font-size', '11px');
            anno2.append('tspan').attr('fill', '#666').text('Nodes: ');
            anno2.append('tspan').attr('fill', '#1f77b4').text('blue');
            anno2.append('tspan').attr('fill', '#666').text(' = cytokine, ');
            anno2.append('tspan').attr('fill', '#2ca02c').text('green');
            anno2.append('tspan').attr('fill', '#666').text(' = biochemistry, ');
            anno2.append('tspan').attr('fill', '#ff7f0e').text('orange');
            anno2.append('tspan').attr('fill', '#666').text(' = metabolite');

            const anno3 = svg.append('text').attr('x', 10).attr('y', height - 11).attr('font-size', '11px').attr('fill', '#666');
            anno3.append('tspan').text('Line ');
            anno3.append('tspan').attr('font-weight', 'bold').text('thickness');
            anno3.append('tspan').text(' = correlation strength');

        } else if (analysisType === 'heatmap') {
            // Top correlations bar chart
            const allCorr = [...biochemFiltered, ...metabFiltered].sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho)).slice(0, 30);

            const trace = {
                type: 'bar',
                x: allCorr.map(d => d.rho),
                y: allCorr.map(d => `${d.signature} - ${(d.variable || d.metabolite || '').substring(0, 15)}`),
                orientation: 'h',
                marker: {
                    color: allCorr.map(d => d.rho > 0 ? '#b2182b' : '#2166ac')
                },
                hovertemplate: '%{y}<br>ρ = %{x:.3f}<extra></extra>'
            };

            Plotly.newPlot(container, [trace], {
                xaxis: { title: 'Spearman ρ', range: [-1, 1] },
                yaxis: { automargin: true },
                margin: { l: 180, r: 30, t: 30, b: 50 },
                height: 500
            }, { responsive: true });
        }

        // Update summary panel
        if (summaryContainer) {
            const cytoBiochemCount = edges.filter(e => e.type === 'biochem').length;
            const cytoMetabCount = edges.filter(e => e.type === 'metabolite').length;
            const cytokineCount = nodes.filter(n => n.group === 'cytokine').length;
            const biochemCount = nodes.filter(n => n.group === 'biochem').length;
            const metabCount = nodes.filter(n => n.group === 'metabolite').length;

            // Get top correlations for display
            const topBiochem = biochemFiltered.slice(0, 5);
            const topMetab = metabFiltered.slice(0, 5);

            summaryContainer.innerHTML = `
                <div style="padding: 0;">
                    <h4 style="margin-top: 0;">Cross-omic Summary</h4>
                    <table class="data-table" style="width: 100%; font-size: 0.85rem; margin-bottom: 1rem;">
                        <tr><th style="text-align: left;">Metric</th><th style="text-align: right;">Count</th></tr>
                        <tr><td>Cytokines</td><td style="text-align: right;">${cytokineCount}</td></tr>
                        <tr><td>Biochemistry markers</td><td style="text-align: right;">${biochemCount}</td></tr>
                        <tr><td>Metabolites</td><td style="text-align: right;">${metabCount}</td></tr>
                        <tr><td>Cytokine ↔ Biochemistry edges</td><td style="text-align: right;">${cytoBiochemCount}</td></tr>
                        <tr><td>Cytokine ↔ Metabolite edges</td><td style="text-align: right;">${cytoMetabCount}</td></tr>
                        <tr><td><strong>Total edges</strong></td><td style="text-align: right;"><strong>${edges.length}</strong></td></tr>
                    </table>

                    <h4 style="margin-top: 1rem;">Top Biochemistry Correlations</h4>
                    <table class="data-table" style="width: 100%; font-size: 0.8rem; margin-bottom: 1rem;">
                        <tr><th>Cytokine</th><th>Marker</th><th>ρ</th></tr>
                        ${topBiochem.map(d => `<tr><td>${d.signature}</td><td>${d.variable}</td><td style="color:${d.rho > 0 ? '#b2182b' : '#2166ac'}">${d.rho.toFixed(3)}</td></tr>`).join('')}
                    </table>

                    <h4 style="margin-top: 1rem;">Top Metabolite Correlations</h4>
                    <table class="data-table" style="width: 100%; font-size: 0.8rem;">
                        <tr><th>Cytokine</th><th>Metabolite</th><th>ρ</th></tr>
                        ${topMetab.map(d => `<tr><td>${d.signature}</td><td>${(d.metabolite || '').substring(0, 18)}</td><td style="color:${d.rho > 0 ? '#b2182b' : '#2166ac'}">${d.rho.toFixed(3)}</td></tr>`).join('')}
                    </table>

                    <p style="margin-top: 1rem; color: #666; font-size: 0.8rem;">
                        Showing top correlations with |ρ| ≥ ${threshold}
                    </p>
                </div>
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
        await this.reloadPopulationData();
    },

    async reloadPopulationData() {
        const sigType = this.signatureType || 'CytoSig';
        const distContainer = document.getElementById('pop-distribution');
        if (distContainer) distContainer.innerHTML = '<div class="loading">Loading data...</div>';

        this.populationData = await API.get('/cima/population-stratification', { signature_type: sigType });
        this.updatePopulationPlots();
    },

    updatePopulationPlots() {
        const data = this.populationData;
        const distContainer = document.getElementById('pop-distribution');
        const effectContainer = document.getElementById('pop-effect-sizes');
        const heatmapContainer = document.getElementById('pop-heatmap');

        if (!data) {
            if (distContainer) distContainer.innerHTML = '<p class="no-data-msg" style="text-align:center; padding:2rem; color:#666;">Population data not available</p>';
            return;
        }

        const stratify = document.getElementById('pop-stratify')?.value || 'sex';
        const groupMeans = data.groups?.[stratify] || {};
        const effects = data.effect_sizes?.[stratify] || [];
        const cytokines = data.cytokines || [];

        const groupNames = Object.keys(groupMeans);

        // Define stratification labels and colors
        const stratLabels = {
            'sex': 'Sex', 'age': 'Age Group', 'bmi': 'BMI Category',
            'blood_type': 'Blood Type', 'smoking': 'Smoking Status'
        };
        const pieColors = {
            'sex': ['#e377c2', '#1f77b4'],
            'age': ['#2ca02c', '#ff7f0e', '#9467bd'],
            'bmi': ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd'],
            'blood_type': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            'smoking': ['#2ca02c', '#ff7f0e', '#d62728']
        };
        const compLabels = {
            'sex': 'Male vs Female', 'age': 'Older vs Young', 'bmi': 'Obese vs Normal',
            'blood_type': 'Blood Type Comparison', 'smoking': 'Smoker vs Non-Smoker'
        };

        // Get sample counts from first effect record
        let groupCounts = [];
        if (effects.length > 0) {
            const first = effects[0];
            const n1 = first.n_male || first.n_young || first.n_normal || 100;
            const n2 = first.n_female || first.n_older || first.n_obese || 100;
            groupCounts = groupNames.map((g, i) => i === 0 ? n1 : i === 1 ? n2 : Math.round((n1 + n2) / 2));
        } else {
            // Fallback: estimate from groupMeans keys
            groupCounts = groupNames.map(() => 100);
        }

        // 1. Distribution pie chart
        if (distContainer && groupNames.length > 0) {
            Plotly.purge(distContainer);
            Plotly.newPlot(distContainer, [{
                type: 'pie',
                labels: groupNames,
                values: groupCounts,
                textinfo: 'label+percent',
                marker: { colors: pieColors[stratify] || ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] },
                hovertemplate: '<b>%{label}</b><br>N ≈ %{value}<br>%{percent}<extra></extra>'
            }], {
                margin: { l: 30, r: 30, t: 40, b: 30 },
                height: 300,
                title: { text: `Distribution by ${stratLabels[stratify] || stratify}`, font: { size: 14 } }
            }, { responsive: true });
        }

        // 2. Effect sizes - Top 10 + Bottom 10 + always IFNG/TNFA
        if (effectContainer) {
            if (effects.length > 0) {
                Plotly.purge(effectContainer);

                // Sort by effect size (positive to negative)
                const sortedByEffect = [...effects].sort((a, b) => (b.effect || b.effect_size || b.log2fc || 0) - (a.effect || a.effect_size || a.log2fc || 0));

                // Get top 10 (most positive) and bottom 10 (most negative)
                const top10 = sortedByEffect.slice(0, 10);
                const bottom10 = sortedByEffect.slice(-10);

                // Always include IFNG and TNFA
                const alwaysInclude = ['IFNG', 'TNFA', 'TNF'];
                const mustHave = effects.filter(d => alwaysInclude.includes(((d.cytokine || d.signature || '')).toUpperCase()));

                // Combine and deduplicate
                const combined = new Map();
                [...top10, ...bottom10, ...mustHave].forEach(d => combined.set(d.cytokine || d.signature, d));

                // Sort final list by effect size
                const selectedEffects = Array.from(combined.values()).sort((a, b) => (b.effect || b.effect_size || b.log2fc || 0) - (a.effect || a.effect_size || a.log2fc || 0));

                const effectSigs = selectedEffects.map(d => d.cytokine || d.signature);
                const effectVals = selectedEffects.map(d => d.effect || d.effect_size || d.log2fc || 0);
                const pvals = selectedEffects.map(d => d.pvalue || d.p_value || 1);

                Plotly.newPlot(effectContainer, [{
                    type: 'bar',
                    orientation: 'h',
                    y: effectSigs,
                    x: effectVals,
                    marker: {
                        color: effectVals.map((e, i) => pvals[i] < 0.05 ? (e > 0 ? '#f4a6a6' : '#a8d4e6') : '#ccc')
                    },
                    text: pvals.map(p => p < 0.001 ? '***' : p < 0.01 ? '**' : p < 0.05 ? '*' : ''),
                    textposition: 'outside',
                    hovertemplate: '<b>%{y}</b><br>Effect: %{x:.3f}<br>p = %{customdata:.2e}<extra></extra>',
                    customdata: pvals
                }], {
                    xaxis: { title: `Effect Size (${compLabels[stratify] || stratify})`, zeroline: true },
                    yaxis: { automargin: true, tickfont: { size: 11 } },
                    margin: { l: 140, r: 60, t: 40, b: 50 },
                    height: Math.max(300, selectedEffects.length * 20),
                    title: { text: `Signature Variance by ${stratLabels[stratify] || stratify}`, font: { size: 14 } },
                    annotations: [{
                        x: 0.95, y: 1.02, xref: 'paper', yref: 'paper',
                        text: '* p<0.05, ** p<0.01, *** p<0.001', showarrow: false, font: { size: 10 }
                    }]
                }, { responsive: true });
            } else {
                effectContainer.innerHTML = '<p class="no-data-msg" style="text-align:center; padding:2rem; color:#666;">Effect size data not available for this stratification.</p>';
            }
        }

        // 3. Population heatmap - show mean activity by group
        if (heatmapContainer) {
            if (effects.length > 0 && groupNames.length > 0) {
                Plotly.purge(heatmapContainer);

                // Use same signatures from effect chart for heatmap
                const sortedByEffect = [...effects].sort((a, b) => Math.abs(b.effect || b.effect_size || b.log2fc || 0) - Math.abs(a.effect || a.effect_size || a.log2fc || 0));
                const top10 = sortedByEffect.slice(0, 10);
                const bottom10 = sortedByEffect.slice(-10);
                const alwaysInclude = ['IFNG', 'TNFA', 'TNF'];
                const mustHave = effects.filter(d => alwaysInclude.includes(((d.cytokine || d.signature || '')).toUpperCase()));
                const combined = new Map();
                [...top10, ...bottom10, ...mustHave].forEach(d => combined.set(d.cytokine || d.signature, d));
                const heatmapSigs = Array.from(combined.values()).sort((a, b) => (b.effect || b.effect_size || b.log2fc || 0) - (a.effect || a.effect_size || a.log2fc || 0)).map(d => d.cytokine || d.signature);

                // Build z-matrix from group means
                const zData = groupNames.map(group => {
                    return heatmapSigs.map(sig => groupMeans[group]?.[sig] ?? 0);
                });

                // Calculate z-score range for colorscale
                const allValues = zData.flat();
                const maxAbs = Math.max(...allValues.map(Math.abs), 1);

                Plotly.newPlot(heatmapContainer, [{
                    type: 'heatmap',
                    z: zData,
                    x: heatmapSigs,
                    y: groupNames,
                    colorscale: [[0, '#a8d4e6'], [0.5, '#f5f5f5'], [1, '#f4a6a6']],
                    zmid: 0,
                    zmin: -maxAbs,
                    zmax: maxAbs,
                    colorbar: { title: 'Mean Activity<br>(z-score)' },
                    hovertemplate: '<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>'
                }], {
                    margin: { l: 120, r: 50, t: 40, b: 120 },
                    xaxis: { tickangle: 45, title: 'Cytokine' },
                    yaxis: { title: stratLabels[stratify] || stratify },
                    height: 400,
                    title: { text: `Activity by ${stratLabels[stratify] || stratify}`, font: { size: 14 } }
                }, { responsive: true });
            } else {
                heatmapContainer.innerHTML = '<p class="no-data-msg" style="text-align:center; padding:2rem; color:#666;">Heatmap data not available for this stratification.</p>';
            }
        }
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
        const manhattanContainer = document.getElementById('eqtl-manhattan');
        const tableContainer = document.getElementById('eqtl-table');
        const cellTypeBarContainer = document.getElementById('eqtl-celltype-bar');

        if (!data || !data.eqtls) {
            if (manhattanContainer) manhattanContainer.innerHTML = '<p style="text-align:center; color:#d62728; padding:2rem;">eQTL data not available</p>';
            return;
        }

        const searchQuery = document.getElementById('eqtl-search')?.value?.toLowerCase() || '';
        const cellType = document.getElementById('eqtl-celltype')?.value || 'all';
        const threshold = parseFloat(document.getElementById('eqtl-threshold')?.value || '5e-8');

        const summary = data.summary || {};
        const totalEqtls = summary.total_eqtls || data.eqtls.length;

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

        // Chromosomes list
        const chromosomes = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y'];

        // 1. Manhattan plot with cumulative chromosome positions
        if (manhattanContainer) {
            Plotly.purge(manhattanContainer);

            if (filtered.length === 0) {
                manhattanContainer.innerHTML = `
                    <p style="text-align:center; color:#666; padding:2rem;">
                        No eQTLs found matching criteria.<br>
                        <small>Total: ${totalEqtls.toLocaleString()} cis-eQTLs</small>
                    </p>`;
            } else {
                // Calculate cumulative positions for Manhattan plot
                const chrLengths = {};
                chromosomes.forEach(chr => { chrLengths[chr] = 250000000; }); // Approximate chromosome length
                let cumPos = 0;
                const chrOffsets = {};
                chromosomes.forEach(chr => {
                    chrOffsets[chr] = cumPos;
                    cumPos += chrLengths[chr];
                });

                // Sample for performance if needed
                const plotData = filtered.length > 5000 ? filtered.slice(0, 5000) : filtered;

                const xPositions = plotData.map(d => (chrOffsets[d.chr] || 0) + (d.pos || 0));
                const yValues = plotData.map(d => -Math.log10(d.pvalue));
                const colors = plotData.map(d => chromosomes.indexOf(d.chr) % 2 === 0 ? '#1f77b4' : '#ff7f0e');

                Plotly.newPlot(manhattanContainer, [{
                    type: 'scatter',
                    mode: 'markers',
                    x: xPositions,
                    y: yValues,
                    text: plotData.map(d => `<b>${d.gene}</b><br>${d.variant}<br>Cell: ${d.celltype}<br>p=${d.pvalue.toExponential(2)}<br>β=${(d.beta || 0).toFixed(3)}`),
                    marker: { color: colors, size: 6, opacity: 0.7 },
                    hoverinfo: 'text'
                }], {
                    xaxis: {
                        title: 'Chromosome',
                        tickvals: chromosomes.slice(0, 22).map(c => chrOffsets[c] + chrLengths[c] / 2),
                        ticktext: chromosomes.slice(0, 22),
                        showgrid: false
                    },
                    yaxis: { title: '-log10(p-value)' },
                    shapes: [{
                        type: 'line',
                        x0: 0, x1: cumPos, y0: -Math.log10(5e-8), y1: -Math.log10(5e-8),
                        line: { dash: 'dash', color: '#d62728', width: 1 }
                    }],
                    margin: { l: 60, r: 30, t: 40, b: 50 },
                    height: 400,
                    title: { text: `cis-eQTLs (${filtered.length.toLocaleString()} shown, ${totalEqtls.toLocaleString()} total)`, font: { size: 14 } }
                }, { responsive: true });
            }
        }

        // 2. Top eQTL table
        if (tableContainer) {
            const topHits = [...filtered].sort((a, b) => a.pvalue - b.pvalue).slice(0, 15);

            if (topHits.length === 0) {
                tableContainer.innerHTML = '<p style="text-align:center; color:#666; padding:1rem;">No results</p>';
            } else {
                tableContainer.innerHTML = `
                    <div style="max-height: 400px; overflow-y: auto;">
                        <table class="data-table" style="width: 100%; font-size: 0.85rem;">
                            <thead>
                                <tr><th>Gene</th><th>Variant</th><th>Cell Type</th><th>P-value</th><th>Beta</th><th>AF</th></tr>
                            </thead>
                            <tbody>
                                ${topHits.map(d => `
                                    <tr>
                                        <td><strong>${d.gene}</strong></td>
                                        <td style="font-size:0.8rem;">${d.variant}</td>
                                        <td style="font-size:0.8rem;">${(d.celltype || '').replace(/_/g, ' ')}</td>
                                        <td>${d.pvalue.toExponential(2)}</td>
                                        <td style="color: ${(d.beta || 0) > 0 ? '#d62728' : '#1f77b4'}">${(d.beta || 0).toFixed(3)}</td>
                                        <td>${d.af !== undefined ? d.af.toFixed(3) : 'N/A'}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                    <p style="margin-top: 0.5rem; color: #666; font-size: 0.85rem;">
                        Showing top ${topHits.length} of ${filtered.length.toLocaleString()} filtered eQTLs
                    </p>
                `;
            }
        }

        // 3. Cell type distribution bar chart
        if (cellTypeBarContainer) {
            Plotly.purge(cellTypeBarContainer);

            // Count eQTLs per cell type
            const ctCounts = {};
            filtered.forEach(d => {
                ctCounts[d.celltype] = (ctCounts[d.celltype] || 0) + 1;
            });

            const sortedCts = Object.entries(ctCounts)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 15);

            if (sortedCts.length > 0) {
                Plotly.newPlot(cellTypeBarContainer, [{
                    type: 'bar',
                    orientation: 'h',
                    y: sortedCts.map(d => d[0].replace(/_/g, ' ')),
                    x: sortedCts.map(d => d[1]),
                    marker: { color: '#1f77b4' },
                    hovertemplate: '<b>%{y}</b><br>%{x} eQTLs<extra></extra>'
                }], {
                    xaxis: { title: 'Number of eQTLs' },
                    yaxis: { automargin: true },
                    margin: { l: 150, r: 30, t: 40, b: 50 },
                    height: 300,
                    title: { text: 'eQTLs by Cell Type', font: { size: 14 } }
                }, { responsive: true });
            } else {
                cellTypeBarContainer.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">No data for cell type distribution</p>';
            }
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

    // Store disease activity data and signatures
    diseaseActivityData: null,
    diseaseSignatures: { cytosig: [], secact: [] },

    async loadInflamDisease(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Disease-Specific Activity</h3>
                <p>Cytokine activity patterns across inflammatory diseases</p>
            </div>

            <div class="controls" style="display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem;">
                <div class="control-group">
                    <label>Disease Group</label>
                    <select id="inflam-disease-group" class="filter-select" onchange="AtlasDetailPage.updateInflamDiseaseBar(); AtlasDetailPage.updateInflamDiseaseHeatmap();">
                        <option value="all">All Diseases</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Signature Type</label>
                    <select id="inflam-disease-sig-type" class="filter-select" onchange="AtlasDetailPage.initDiseaseSignatures(); AtlasDetailPage.updateInflamDiseaseBar(); AtlasDetailPage.updateInflamDiseaseHeatmap();">
                        <option value="CytoSig">CytoSig (44 cytokines)</option>
                        <option value="SecAct">SecAct (1,249 proteins)</option>
                    </select>
                </div>
                <div class="control-group" style="position: relative;">
                    <label>Search Signature</label>
                    <input type="text" id="inflam-disease-search" class="filter-select" placeholder="e.g., IFNG, IL6"
                           style="width: 150px;" autocomplete="off" value="IFNG"
                           oninput="AtlasDetailPage.showDiseaseSuggestions()"
                           onkeyup="if(event.key==='Enter') AtlasDetailPage.updateInflamDiseaseBar()">
                    <div id="inflam-disease-suggestions" style="position: absolute; top: 100%; left: 0; width: 150px; max-height: 200px; overflow-y: auto; background: white; border: 1px solid #ddd; border-radius: 4px; display: none; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"></div>
                </div>
            </div>

            <div class="viz-grid">
                <div class="sub-panel">
                    <div class="panel-header">
                        <h4>Disease-Cell Type Activity</h4>
                        <p>Activity profile across cell types per disease</p>
                    </div>
                    <div id="inflam-disease-bar" class="plot-container" style="height: 500px;"></div>
                </div>
                <div class="sub-panel">
                    <div class="panel-header">
                        <h4>Disease Activity Heatmap</h4>
                        <p>Diseases × Signatures</p>
                    </div>
                    <div id="inflam-disease-heatmap" class="plot-container" style="height: 500px;"></div>
                </div>
            </div>
        `;

        // Load disease activity data
        this.diseaseActivityData = await API.get('/inflammation/disease-activity', { signature_type: this.signatureType });
        this.initDiseaseSignatures();
        this.populateInflamDiseaseGroups();
        this.updateInflamDiseaseBar();
        this.updateInflamDiseaseHeatmap();
    },

    initDiseaseSignatures() {
        const data = this.diseaseActivityData;
        if (!data || !Array.isArray(data)) return;

        const sigType = document.getElementById('inflam-disease-sig-type')?.value || 'CytoSig';

        // Group signatures by type
        this.diseaseSignatures.cytosig = [...new Set(
            data.filter(d => d.signature_type === 'CytoSig').map(d => d.signature)
        )].sort();
        this.diseaseSignatures.secact = [...new Set(
            data.filter(d => d.signature_type === 'SecAct').map(d => d.signature)
        )].sort();

        // Fallback if no signature_type field
        if (this.diseaseSignatures.cytosig.length === 0 && this.diseaseSignatures.secact.length === 0) {
            this.diseaseSignatures.cytosig = [...new Set(data.map(d => d.signature))].sort();
        }
    },

    getDiseaseSignatures() {
        const sigType = document.getElementById('inflam-disease-sig-type')?.value || 'CytoSig';
        return sigType === 'SecAct' ? this.diseaseSignatures.secact : this.diseaseSignatures.cytosig;
    },

    showDiseaseSuggestions() {
        const input = document.getElementById('inflam-disease-search');
        const div = document.getElementById('inflam-disease-suggestions');
        if (!input || !div) return;

        const query = input.value.toLowerCase();
        const sigs = this.getDiseaseSignatures();
        const filtered = sigs.filter(s => s.toLowerCase().includes(query)).slice(0, 15);

        if (filtered.length === 0 || !query) {
            div.style.display = 'none';
            return;
        }

        div.innerHTML = filtered.map(s =>
            `<div style="padding:6px 10px;cursor:pointer;border-bottom:1px solid #eee"
                 onmouseover="this.style.background='#f0f0f0'" onmouseout="this.style.background='white'"
                 onclick="AtlasDetailPage.selectDiseaseSig('${s}')">${s}</div>`
        ).join('');
        div.style.display = 'block';
    },

    selectDiseaseSig(sig) {
        const input = document.getElementById('inflam-disease-search');
        const div = document.getElementById('inflam-disease-suggestions');
        if (input) input.value = sig;
        if (div) div.style.display = 'none';
        this.updateInflamDiseaseBar();
    },

    populateInflamDiseaseGroups() {
        const data = this.diseaseActivityData;
        if (!data || !Array.isArray(data)) return;

        const diseaseGroups = [...new Set(data.map(d => d.disease_group).filter(Boolean))].sort();
        const select = document.getElementById('inflam-disease-group');
        if (select && diseaseGroups.length > 0) {
            select.innerHTML = '<option value="all">All Diseases</option>' +
                diseaseGroups.map(g => `<option value="${g}">${g}</option>`).join('');
        }
    },

    updateInflamDiseaseBar() {
        const container = document.getElementById('inflam-disease-bar');
        if (!container) return;

        const data = this.diseaseActivityData;
        if (!data || !Array.isArray(data)) {
            container.innerHTML = '<p style="text-align: center; color: #666; padding: 2rem;">Disease activity data not available</p>';
            return;
        }

        const diseaseGroup = document.getElementById('inflam-disease-group')?.value || 'all';
        const sigType = document.getElementById('inflam-disease-sig-type')?.value || 'CytoSig';
        const signature = document.getElementById('inflam-disease-search')?.value || 'IFNG';

        // Filter by signature
        let filtered = data.filter(d => d.signature === signature);

        // Filter by signature_type if field exists
        if (data[0]?.signature_type) {
            filtered = filtered.filter(d => d.signature_type === sigType);
        }
        if (diseaseGroup !== 'all') {
            filtered = filtered.filter(d => d.disease_group === diseaseGroup);
        }

        // Aggregate by cell type
        const cellTypeActivity = {};
        filtered.forEach(d => {
            if (!cellTypeActivity[d.cell_type]) {
                cellTypeActivity[d.cell_type] = [];
            }
            cellTypeActivity[d.cell_type].push(d.mean_activity || d.activity || 0);
        });

        const aggData = Object.entries(cellTypeActivity).map(([ct, vals]) => ({
            cell_type: ct,
            mean_activity: vals.reduce((a, b) => a + b, 0) / vals.length
        })).sort((a, b) => b.mean_activity - a.mean_activity).slice(0, 30);

        Plotly.purge(container);

        if (aggData.length === 0) {
            container.innerHTML = `<p style="text-align: center; color: #666; padding: 2rem;">No data for "${signature}" in ${diseaseGroup === 'all' ? 'all diseases' : diseaseGroup}</p>`;
            return;
        }

        Plotly.newPlot(container, [{
            y: aggData.map(d => d.cell_type),
            x: aggData.map(d => d.mean_activity),
            type: 'bar',
            orientation: 'h',
            marker: {
                color: aggData.map(d => d.mean_activity),
                colorscale: [[0, '#a8d4e6'], [0.5, '#f5f5f5'], [1, '#f4a6a6']],
                cmid: 0
            },
            hovertemplate: '<b>%{y}</b><br>Activity: %{x:.4f}<extra></extra>'
        }], {
            margin: { l: 180, r: 30, t: 30, b: 50 },
            xaxis: { title: 'Mean Activity' },
            height: 450
        }, { responsive: true });
    },

    updateInflamDiseaseHeatmap() {
        const container = document.getElementById('inflam-disease-heatmap');
        if (!container) return;

        const data = this.diseaseActivityData;
        if (!data || !Array.isArray(data)) {
            container.innerHTML = '<p style="text-align: center; color: #666; padding: 2rem;">Disease activity data not available</p>';
            return;
        }

        const diseaseGroup = document.getElementById('inflam-disease-group')?.value || 'all';
        const sigType = document.getElementById('inflam-disease-sig-type')?.value || 'CytoSig';

        let filtered = data;
        // Filter by signature type if field exists
        if (data[0]?.signature_type) {
            filtered = filtered.filter(d => d.signature_type === sigType);
        }
        if (diseaseGroup !== 'all') {
            filtered = filtered.filter(d => d.disease_group === diseaseGroup);
        }

        // Get unique diseases and signatures
        const diseases = [...new Set(filtered.map(d => d.disease))];
        const signatures = [...new Set(filtered.map(d => d.signature))].sort();

        // Aggregate by disease (mean across cell types)
        const diseaseActivity = {};
        filtered.forEach(d => {
            const key = `${d.disease}|${d.signature}`;
            if (!diseaseActivity[key]) {
                diseaseActivity[key] = [];
            }
            diseaseActivity[key].push(d.mean_activity || d.activity || 0);
        });

        // Create matrix
        const zData = diseases.map(disease =>
            signatures.map(sig => {
                const key = `${disease}|${sig}`;
                const vals = diseaseActivity[key] || [0];
                return vals.reduce((a, b) => a + b, 0) / vals.length;
            })
        );

        Plotly.purge(container);

        if (diseases.length === 0 || signatures.length === 0) {
            container.innerHTML = '<p style="text-align: center; color: #666; padding: 2rem;">No data available for heatmap</p>';
            return;
        }

        Plotly.newPlot(container, [{
            z: zData,
            x: signatures,
            y: diseases,
            type: 'heatmap',
            colorscale: [[0, '#a8d4e6'], [0.5, '#f5f5f5'], [1, '#f4a6a6']],
            zmid: 0,
            colorbar: { title: 'Activity' },
            hovertemplate: '<b>%{y}</b><br>%{x}: %{z:.4f}<extra></extra>'
        }], {
            margin: { l: 120, r: 50, t: 40, b: 120 },
            xaxis: { tickangle: 45, title: 'Signature' },
            yaxis: { title: 'Disease' },
            title: { text: 'Disease × Signature Activity', font: { size: 14 } },
            height: 480
        }, { responsive: true });
    },

    async loadInflamDifferential(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Disease Differential Analysis</h3>
                <p>Compare cytokine activity between disease and healthy samples</p>
            </div>
            <div class="controls" style="margin-bottom: 16px; display: flex; gap: 16px; flex-wrap: wrap;">
                <div class="control-group">
                    <label>Disease</label>
                    <select id="inflam-diff-disease" class="filter-select" onchange="AtlasDetailPage.updateInflamDifferential()">
                        <option value="all">All Diseases vs Healthy</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Signature Type</label>
                    <select id="inflam-diff-sig-type" class="filter-select" onchange="AtlasDetailPage.updateInflamDifferential()">
                        <option value="CytoSig">CytoSig (43 cytokines)</option>
                        <option value="SecAct">SecAct (1,170 proteins)</option>
                    </select>
                </div>
            </div>
            <div class="viz-grid">
                <div class="sub-panel">
                    <h4 id="inflam-diff-volcano-title">Volcano Plot: Disease vs Healthy</h4>
                    <p style="color: #666; font-size: 0.9rem;">Effect size (log2FC) vs significance (-log10 p-value)</p>
                    <div id="inflam-volcano" class="plot-container" style="height: 500px;"></div>
                </div>
                <div class="sub-panel">
                    <h4 id="inflam-diff-bar-title">Top Differential Signatures</h4>
                    <p style="color: #666; font-size: 0.9rem;">Sorted by significance score (|log2FC| × -log10 p)</p>
                    <div id="inflam-diff-bar" class="plot-container" style="height: 500px;"></div>
                </div>
            </div>
        `;

        await this.populateDiseaseDropdown('inflam-diff-disease');
        await this.updateInflamDifferential();
    },

    async loadInflamTreatment(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Treatment Response Prediction</h3>
                <p>Machine learning models trained to predict treatment response from baseline cytokine signatures</p>
            </div>
            <div class="controls" style="margin-bottom: 16px; display: flex; gap: 16px; flex-wrap: wrap;">
                <div class="control-group">
                    <label>Disease</label>
                    <select id="treatment-disease" class="filter-select" onchange="AtlasDetailPage.updateTreatmentResponse()">
                        <option value="all">All Diseases</option>
                        <option value="RA">Rheumatoid Arthritis (RA)</option>
                        <option value="PS">Psoriasis (PS)</option>
                        <option value="PSA">Psoriatic Arthritis (PSA)</option>
                        <option value="CD">Crohn's Disease (CD)</option>
                        <option value="UC">Ulcerative Colitis (UC)</option>
                        <option value="SLE">Systemic Lupus (SLE)</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Signature Type</label>
                    <select id="treatment-sig-type" class="filter-select" onchange="AtlasDetailPage.updateTreatmentResponse()">
                        <option value="CytoSig">CytoSig (43 cytokines)</option>
                        <option value="SecAct">SecAct (1,170 proteins)</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Model</label>
                    <select id="treatment-model" class="filter-select" onchange="AtlasDetailPage.updateTreatmentResponse()">
                        <option value="all">All Models</option>
                        <option value="Logistic Regression">Logistic Regression</option>
                        <option value="Random Forest">Random Forest</option>
                    </select>
                </div>
            </div>
            <div class="viz-grid">
                <div class="sub-panel">
                    <h4>ROC Curves: Treatment Response Prediction</h4>
                    <p style="color: #666; font-size: 0.9rem;">AUC performance for responder vs non-responder classification</p>
                    <div id="treatment-roc" class="plot-container" style="height: 450px;"></div>
                </div>
                <div class="sub-panel">
                    <h4>Feature Importance</h4>
                    <p style="color: #666; font-size: 0.9rem;">Top predictive cytokine signatures</p>
                    <div id="treatment-importance" class="plot-container" style="height: 450px;"></div>
                </div>
            </div>
            <div class="sub-panel" style="margin-top: 16px;">
                <h4>Response Prediction Scores</h4>
                <p style="color: #666; font-size: 0.9rem;">Distribution of predicted probabilities for responders vs non-responders</p>
                <div id="treatment-violin" class="plot-container" style="height: 350px;"></div>
            </div>
        `;

        await this.updateTreatmentResponse();
    },

    async loadInflamSankey(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Sample Flow: Study → Disease → Disease Group</h3>
                <p>Distribution of samples across studies, diseases, and disease groups</p>
            </div>
            <div id="disease-sankey" class="plot-container" style="height: 650px;"></div>
        `;

        const data = await API.get('/inflammation/sankey');
        if (data && data.nodes && data.links) {
            this.renderSankeyDiagram('disease-sankey', data);
        } else {
            document.getElementById('disease-sankey').innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">No Sankey data available</p>';
        }
    },

    async loadInflamValidation(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Cross-Cohort Validation</h3>
                <p>Consistency of cytokine/protein activity signatures across main, validation, and external cohorts.
                   <strong>Click on points or rows to highlight across all views.</strong></p>
            </div>
            <div class="validation-controls" style="margin-bottom: 16px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap;">
                <div class="control-group">
                    <label for="validation-sig-type" style="font-weight: 500; margin-right: 8px;">Signature Type:</label>
                    <select id="validation-sig-type" class="filter-select" onchange="AtlasDetailPage.updateInflamValidation()">
                        <option value="CytoSig" ${this.signatureType === 'CytoSig' ? 'selected' : ''}>CytoSig (44 cytokines)</option>
                        <option value="SecAct" ${this.signatureType === 'SecAct' ? 'selected' : ''}>SecAct (1,249 proteins)</option>
                    </select>
                </div>
                <div class="control-group">
                    <label for="validation-search" style="font-weight: 500; margin-right: 8px;">Search:</label>
                    <input type="text" id="validation-search" class="filter-input" placeholder="Filter signatures..."
                           oninput="AtlasDetailPage.filterValidationViews()" style="width: 200px;">
                </div>
                <div class="control-group">
                    <label style="font-weight: 500; margin-right: 8px;">Quality Filter:</label>
                    <div id="quality-filter-btns" style="display: inline-flex; gap: 4px;">
                        <button class="quality-btn active" data-quality="all" onclick="AtlasDetailPage.filterByQuality('all')"
                                style="padding: 4px 12px; border: 1px solid #d1d5db; border-radius: 4px; background: #3b82f6; color: white; cursor: pointer; font-size: 13px;">All</button>
                        <button class="quality-btn" data-quality="excellent" onclick="AtlasDetailPage.filterByQuality('excellent')"
                                style="padding: 4px 12px; border: 1px solid #10b981; border-radius: 4px; background: white; color: #10b981; cursor: pointer; font-size: 13px;">Excellent</button>
                        <button class="quality-btn" data-quality="good" onclick="AtlasDetailPage.filterByQuality('good')"
                                style="padding: 4px 12px; border: 1px solid #f59e0b; border-radius: 4px; background: white; color: #f59e0b; cursor: pointer; font-size: 13px;">Good</button>
                        <button class="quality-btn" data-quality="fair" onclick="AtlasDetailPage.filterByQuality('fair')"
                                style="padding: 4px 12px; border: 1px solid #6b7280; border-radius: 4px; background: white; color: #6b7280; cursor: pointer; font-size: 13px;">Fair/Poor</button>
                    </div>
                </div>
                <button onclick="AtlasDetailPage.clearValidationSelection()"
                        style="padding: 4px 12px; border: 1px solid #ef4444; border-radius: 4px; background: white; color: #ef4444; cursor: pointer; font-size: 13px;">
                    Clear Selection
                </button>
            </div>
            <div id="validation-summary" class="plot-container" style="height: 280px; margin-bottom: 20px;"></div>
            <div id="validation-detail-card" style="display: none; margin-bottom: 20px; padding: 16px; background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%); border-radius: 12px; border: 2px solid #3b82f6; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 id="validation-detail-name" style="margin: 0 0 8px 0; font-size: 18px; color: #1e40af;"></h4>
                        <div id="validation-detail-info" style="display: flex; gap: 24px; font-size: 14px;"></div>
                    </div>
                    <button onclick="AtlasDetailPage.clearValidationSelection()" style="padding: 6px 12px; background: #ef4444; color: white; border: none; border-radius: 6px; cursor: pointer;">×</button>
                </div>
            </div>
            <div id="validation-scatter" class="plot-container" style="height: 450px; margin-bottom: 24px;"></div>
            <div id="validation-table-container" style="max-height: 400px; overflow-y: auto; border: 1px solid #e5e7eb; border-radius: 8px;">
                <table id="validation-table" class="data-table" style="width: 100%; border-collapse: collapse;">
                    <thead style="position: sticky; top: 0; background: #f9fafb; z-index: 1;">
                        <tr>
                            <th style="padding: 12px; text-align: left; border-bottom: 2px solid #e5e7eb; cursor: pointer;" onclick="AtlasDetailPage.sortValidationTable('signature')">Signature ↕</th>
                            <th style="padding: 12px; text-align: center; border-bottom: 2px solid #e5e7eb; cursor: pointer;" onclick="AtlasDetailPage.sortValidationTable('main_validation_r')">Main↔Val (r) ↕</th>
                            <th style="padding: 12px; text-align: center; border-bottom: 2px solid #e5e7eb; cursor: pointer;" onclick="AtlasDetailPage.sortValidationTable('main_external_r')">Main↔Ext (r) ↕</th>
                            <th style="padding: 12px; text-align: center; border-bottom: 2px solid #e5e7eb;">Quality</th>
                        </tr>
                    </thead>
                    <tbody id="validation-table-body"></tbody>
                </table>
            </div>
            <div id="validation-stats" style="margin-top: 16px; padding: 12px; background: #f0f9ff; border-radius: 8px; font-size: 14px;"></div>
        `;

        await this.updateInflamValidation();
    },

    // Validation state
    validationData: null,
    validationSortColumn: 'signature',
    validationSortAsc: true,
    validationSelectedSignature: null,
    validationQualityFilter: 'all',
    validationSearchTerm: '',

    async updateInflamValidation() {
        const sigType = document.getElementById('validation-sig-type')?.value || this.signatureType;
        this.signatureType = sigType;

        // Reset selection state when switching signature type
        this.validationSelectedSignature = null;
        this.validationQualityFilter = 'all';
        this.validationSearchTerm = '';

        const data = await API.get('/inflammation/cohort-validation', { signature_type: sigType });
        if (data && data.correlations) {
            this.validationData = data;
            this.renderCohortValidation(data);
        } else {
            document.getElementById('validation-summary').innerHTML = '<p class="loading">No validation data available</p>';
            document.getElementById('validation-scatter').innerHTML = '';
            document.getElementById('validation-table-body').innerHTML = '';
        }
    },

    getFilteredValidationData() {
        if (!this.validationData?.correlations) return [];

        let filtered = [...this.validationData.correlations];

        // Apply search filter
        if (this.validationSearchTerm) {
            filtered = filtered.filter(c =>
                c.signature.toLowerCase().includes(this.validationSearchTerm.toLowerCase())
            );
        }

        // Apply quality filter
        if (this.validationQualityFilter !== 'all') {
            filtered = filtered.filter(c => {
                const avgR = (c.main_validation_r + c.main_external_r) / 2;
                switch (this.validationQualityFilter) {
                    case 'excellent': return avgR >= 0.8;
                    case 'good': return avgR >= 0.6 && avgR < 0.8;
                    case 'fair': return avgR < 0.6;
                    default: return true;
                }
            });
        }

        return filtered;
    },

    filterValidationViews() {
        this.validationSearchTerm = document.getElementById('validation-search')?.value || '';
        this.refreshValidationViews();
    },

    filterByQuality(quality) {
        this.validationQualityFilter = quality;

        // Update button states
        document.querySelectorAll('#quality-filter-btns .quality-btn').forEach(btn => {
            const btnQuality = btn.dataset.quality;
            if (btnQuality === quality) {
                btn.style.background = btnQuality === 'all' ? '#3b82f6' :
                    btnQuality === 'excellent' ? '#10b981' :
                    btnQuality === 'good' ? '#f59e0b' : '#6b7280';
                btn.style.color = 'white';
            } else {
                btn.style.background = 'white';
                btn.style.color = btnQuality === 'all' ? '#3b82f6' :
                    btnQuality === 'excellent' ? '#10b981' :
                    btnQuality === 'good' ? '#f59e0b' : '#6b7280';
            }
        });

        this.refreshValidationViews();
    },

    refreshValidationViews() {
        const filtered = this.getFilteredValidationData();
        this.renderValidationScatter(filtered);
        this.renderValidationTable(filtered);
        this.renderValidationStats({ correlations: filtered, consistency: this.validationData?.consistency });

        // Maintain selection highlight if still in filtered data
        if (this.validationSelectedSignature) {
            const stillVisible = filtered.some(c => c.signature === this.validationSelectedSignature);
            if (stillVisible) {
                this.highlightValidationSignature(this.validationSelectedSignature);
            } else {
                this.clearValidationSelection();
            }
        }
    },

    selectValidationSignature(signature) {
        this.validationSelectedSignature = signature;
        this.highlightValidationSignature(signature);
        this.showValidationDetail(signature);
    },

    highlightValidationSignature(signature) {
        // Highlight in scatter plot
        this.highlightScatterPoint(signature);

        // Highlight in table
        this.highlightTableRow(signature);
    },

    highlightScatterPoint(signature) {
        const scatterDiv = document.getElementById('validation-scatter');
        if (!scatterDiv || !scatterDiv.data) return;

        const filtered = this.getFilteredValidationData();
        const idx = filtered.findIndex(c => c.signature === signature);

        if (idx === -1) return;

        // Update marker sizes and colors to highlight selected point
        const sizes = filtered.map((c, i) => i === idx ? 16 : 8);
        const opacities = filtered.map((c, i) => i === idx ? 1 : 0.5);
        const lineWidths = filtered.map((c, i) => i === idx ? 3 : 1);
        const lineColors = filtered.map((c, i) => i === idx ? '#1e40af' : '#fff');

        Plotly.restyle('validation-scatter', {
            'marker.size': [sizes],
            'marker.opacity': [opacities],
            'marker.line.width': [lineWidths],
            'marker.line.color': [lineColors],
        }, [1]); // Index 1 is the scatter trace (0 is ref line)
    },

    highlightTableRow(signature) {
        // Remove previous highlight
        document.querySelectorAll('#validation-table-body tr').forEach(row => {
            row.classList.remove('selected-row');
            row.style.outline = 'none';
        });

        // Find and highlight the row
        const rows = document.querySelectorAll('#validation-table-body tr');
        rows.forEach(row => {
            if (row.dataset.signature === signature) {
                row.classList.add('selected-row');
                row.style.outline = '3px solid #3b82f6';
                row.style.outlineOffset = '-3px';
                row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        });
    },

    showValidationDetail(signature) {
        const data = this.validationData?.correlations?.find(c => c.signature === signature);
        if (!data) return;

        const detailCard = document.getElementById('validation-detail-card');
        const nameEl = document.getElementById('validation-detail-name');
        const infoEl = document.getElementById('validation-detail-info');

        if (!detailCard || !nameEl || !infoEl) return;

        const avgR = (data.main_validation_r + data.main_external_r) / 2;
        const quality = avgR >= 0.8 ? 'Excellent' : avgR >= 0.6 ? 'Good' : avgR >= 0.4 ? 'Fair' : 'Poor';
        const qualityColor = avgR >= 0.8 ? '#10b981' : avgR >= 0.6 ? '#f59e0b' : avgR >= 0.4 ? '#6b7280' : '#ef4444';

        nameEl.textContent = data.signature;
        infoEl.innerHTML = `
            <span><strong>Main ↔ Validation:</strong> <span style="font-size: 16px; color: #1e40af;">${data.main_validation_r.toFixed(3)}</span></span>
            <span><strong>Main ↔ External:</strong> <span style="font-size: 16px; color: #1e40af;">${data.main_external_r.toFixed(3)}</span></span>
            <span><strong>Average:</strong> <span style="font-size: 16px; color: #1e40af;">${avgR.toFixed(3)}</span></span>
            <span><strong>Quality:</strong> <span style="font-size: 16px; color: ${qualityColor}; font-weight: 600;">${quality}</span></span>
            <span><strong>p-value:</strong> ${data.pvalue < 0.001 ? '<0.001' : data.pvalue.toFixed(3)}</span>
        `;

        detailCard.style.display = 'block';
    },

    clearValidationSelection() {
        this.validationSelectedSignature = null;

        // Hide detail card
        const detailCard = document.getElementById('validation-detail-card');
        if (detailCard) detailCard.style.display = 'none';

        // Reset scatter plot highlighting
        const filtered = this.getFilteredValidationData();
        if (filtered.length > 0) {
            const avgR = filtered.map(c => (c.main_validation_r + c.main_external_r) / 2);
            const colors = avgR.map(r => r >= 0.8 ? '#10b981' : r >= 0.6 ? '#f59e0b' : '#ef4444');

            Plotly.restyle('validation-scatter', {
                'marker.size': [filtered.map(() => 8)],
                'marker.opacity': [filtered.map(() => 0.7)],
                'marker.line.width': [filtered.map(() => 1)],
                'marker.line.color': [filtered.map(() => '#fff')],
                'marker.color': [colors],
            }, [1]);
        }

        // Remove table row highlights
        document.querySelectorAll('#validation-table-body tr').forEach(row => {
            row.classList.remove('selected-row');
            row.style.outline = 'none';
        });
    },

    sortValidationTable(column) {
        if (this.validationSortColumn === column) {
            this.validationSortAsc = !this.validationSortAsc;
        } else {
            this.validationSortColumn = column;
            this.validationSortAsc = true;
        }
        const filtered = this.getFilteredValidationData();
        this.renderValidationTable(filtered);

        // Re-apply highlight if there's a selection
        if (this.validationSelectedSignature) {
            this.highlightTableRow(this.validationSelectedSignature);
        }
    },

    async loadInflamLongitudinal(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Temporal Distribution Analysis</h3>
                <p>Cross-sectional comparison of cytokine activity across sampling timepoints (T0, T1, T2).
                   <em>Note: Different timepoints represent different patients, not the same patients followed over time.</em></p>
            </div>
            <div class="temporal-controls" style="margin-bottom: 16px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap;">
                <div class="control-group">
                    <label for="temporal-disease" style="font-weight: 500; margin-right: 8px;">Disease:</label>
                    <select id="temporal-disease" class="filter-select" onchange="AtlasDetailPage.updateTemporalPanel()">
                        <option value="">Loading...</option>
                    </select>
                </div>
                <div class="control-group">
                    <label for="temporal-signature" style="font-weight: 500; margin-right: 8px;">Signature:</label>
                    <select id="temporal-signature" class="filter-select" onchange="AtlasDetailPage.updateTemporalLineChart()">
                        <option value="">All signatures (heatmap)</option>
                    </select>
                </div>
            </div>
            <div id="temporal-info" style="margin-bottom: 16px; padding: 12px; background: #fef3c7; border-radius: 8px; border-left: 4px solid #f59e0b; font-size: 14px;"></div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                <div id="temporal-distribution" class="plot-container" style="height: 300px;"></div>
                <div id="temporal-treatment" class="plot-container" style="height: 300px;"></div>
            </div>
            <div id="temporal-heatmap" class="plot-container" style="height: 400px; margin-bottom: 20px;"></div>
            <div id="temporal-linechart" class="plot-container" style="height: 350px; display: none;"></div>
            <div id="temporal-table-container" style="max-height: 350px; overflow-y: auto; border: 1px solid #e5e7eb; border-radius: 8px; margin-top: 16px;">
                <table id="temporal-table" class="data-table" style="width: 100%; border-collapse: collapse;">
                    <thead style="position: sticky; top: 0; background: #f9fafb; z-index: 1;">
                        <tr>
                            <th style="padding: 10px; text-align: left; border-bottom: 2px solid #e5e7eb;">Signature</th>
                            <th style="padding: 10px; text-align: center; border-bottom: 2px solid #e5e7eb;">T0 → T1 Change</th>
                            <th style="padding: 10px; text-align: center; border-bottom: 2px solid #e5e7eb;">Trend</th>
                        </tr>
                    </thead>
                    <tbody id="temporal-table-body"></tbody>
                </table>
            </div>
        `;

        await this.initTemporalPanel();
    },

    temporalData: null,
    temporalDiseases: [],

    async initTemporalPanel() {
        // Load full temporal analysis
        this.temporalData = await API.get('/inflammation/temporal') || {};

        // Populate disease dropdown - only diseases with multiple timepoints
        this.temporalDiseases = await API.get('/inflammation/temporal/diseases') || [];

        const diseaseSelect = document.getElementById('temporal-disease');
        if (diseaseSelect) {
            const allDiseases = Object.keys(this.temporalData.disease_timepoints || {});
            // Show diseases with multi-timepoint data first, then others
            const sortedDiseases = [
                ...this.temporalDiseases,
                ...allDiseases.filter(d => !this.temporalDiseases.includes(d))
            ];
            diseaseSelect.innerHTML = sortedDiseases.map(d => {
                const hasMulti = this.temporalDiseases.includes(d);
                return `<option value="${d}" ${d === 'healthy' ? 'selected' : ''}>${d}${hasMulti ? ' (multi-TP)' : ''}</option>`;
            }).join('');
        }

        // Load signatures for dropdown
        const signatures = await API.get('/inflammation/signatures', { signature_type: this.signatureType }) || [];
        const sigSelect = document.getElementById('temporal-signature');
        if (sigSelect && signatures.length > 0) {
            sigSelect.innerHTML = '<option value="">All signatures (heatmap)</option>' +
                signatures.map(s => `<option value="${s}">${s}</option>`).join('');
        }

        // Render distribution charts
        this.renderTemporalDistribution();
        this.renderTreatmentDistribution();

        // Update panel with selected disease
        await this.updateTemporalPanel();
    },

    renderTemporalDistribution() {
        const container = document.getElementById('temporal-distribution');
        if (!container || !this.temporalData?.timepoint_distribution) return;

        const dist = this.temporalData.timepoint_distribution;
        const timepoints = Object.keys(dist).sort();
        const counts = timepoints.map(tp => dist[tp]);

        Plotly.newPlot(container, [{
            x: timepoints,
            y: counts,
            type: 'bar',
            marker: { color: ['#3b82f6', '#10b981', '#f59e0b'] },
            text: counts.map(c => c.toString()),
            textposition: 'auto',
            hovertemplate: '<b>%{x}</b><br>Samples: %{y}<extra></extra>',
        }], {
            title: { text: 'Sample Distribution by Timepoint', font: { size: 14 } },
            xaxis: { title: 'Timepoint' },
            yaxis: { title: 'Number of Samples' },
            margin: { l: 50, r: 20, t: 40, b: 40 },
            font: { family: 'Inter, sans-serif' },
        });
    },

    renderTreatmentDistribution() {
        const container = document.getElementById('temporal-treatment');
        if (!container || !this.temporalData?.treatment_by_timepoint) return;

        const treatment = this.temporalData.treatment_by_timepoint;
        const statuses = Object.keys(treatment);
        const timepoints = ['T0', 'T1', 'T2'];

        const traces = statuses.filter(s => s !== 'Unknown' && s !== 'na').map((status) => ({
            x: timepoints,
            y: timepoints.map(tp => treatment[status]?.[tp] || 0),
            name: status,
            type: 'bar',
            hovertemplate: `<b>${status}</b><br>%{x}: %{y} samples<extra></extra>`,
        }));

        Plotly.newPlot(container, traces, {
            title: { text: 'Treatment Status by Timepoint', font: { size: 14 } },
            xaxis: { title: 'Timepoint' },
            yaxis: { title: 'Number of Samples' },
            barmode: 'stack',
            margin: { l: 50, r: 20, t: 40, b: 40 },
            font: { family: 'Inter, sans-serif' },
            legend: { orientation: 'h', y: -0.2 },
        });
    },

    async updateTemporalPanel() {
        const disease = document.getElementById('temporal-disease')?.value;
        if (!disease) return;

        // Update info panel
        const infoDiv = document.getElementById('temporal-info');
        const diseaseTp = this.temporalData?.disease_timepoints?.[disease] || {};
        const timepoints = Object.keys(diseaseTp);
        const hasMultiple = timepoints.length > 1;

        if (infoDiv) {
            if (hasMultiple) {
                infoDiv.style.background = '#f0fdf4';
                infoDiv.style.borderColor = '#10b981';
                infoDiv.innerHTML = `
                    <strong>${disease}</strong> has samples at <strong>${timepoints.length}</strong> timepoints:
                    ${timepoints.map(tp => `<strong>${tp}</strong> (n=${diseaseTp[tp]})`).join(', ')}
                `;
            } else {
                infoDiv.style.background = '#fef3c7';
                infoDiv.style.borderColor = '#f59e0b';
                infoDiv.innerHTML = `
                    <strong>${disease}</strong> has samples only at <strong>${timepoints[0] || 'T0'}</strong> (n=${diseaseTp[timepoints[0]] || 0}).
                    <em>Select a disease with "(multi-TP)" for temporal comparison.</em>
                `;
            }
        }

        // Get heatmap data if multiple timepoints
        if (hasMultiple) {
            const heatmapData = await API.get(`/inflammation/temporal/heatmap/${disease}`, {
                signature_type: this.signatureType
            });
            this.renderTemporalHeatmap(heatmapData, disease);

            // Get activity data for table
            const activityData = await API.get('/inflammation/temporal/activity', {
                disease: disease,
                signature_type: this.signatureType
            }) || [];
            this.renderTemporalTable(activityData);
        } else {
            document.getElementById('temporal-heatmap').innerHTML =
                '<p style="text-align: center; padding: 2rem; color: #666;">Select a disease with multiple timepoints to view heatmap</p>';
            document.getElementById('temporal-table-body').innerHTML = '';
        }

        // Reset line chart
        document.getElementById('temporal-linechart').style.display = 'none';
        const sigSelect = document.getElementById('temporal-signature');
        if (sigSelect) sigSelect.value = '';
    },

    renderTemporalHeatmap(data, disease) {
        const container = document.getElementById('temporal-heatmap');
        if (!container || !data || !data.rows || data.rows.length === 0) {
            container.innerHTML = '<p class="loading">No temporal heatmap data available</p>';
            return;
        }

        const trace = {
            z: data.values,
            x: data.columns,
            y: data.rows,
            type: 'heatmap',
            colorscale: [[0, '#3b82f6'], [0.5, '#f5f5f5'], [1, '#ef4444']],
            zmid: 0,
            colorbar: { title: 'Activity', titleside: 'right' },
            hovertemplate: '<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>',
        };

        Plotly.newPlot(container, [trace], {
            title: { text: `${disease}: Cytokine Activity by Timepoint`, font: { size: 16 } },
            xaxis: { title: 'Cytokine Signature', tickangle: 45, tickfont: { size: 10 } },
            yaxis: { title: 'Timepoint' },
            margin: { l: 80, r: 80, t: 50, b: 150 },
            font: { family: 'Inter, sans-serif' },
        });

        // Add click handler
        container.on('plotly_click', (eventData) => {
            if (eventData.points && eventData.points.length > 0) {
                const signature = eventData.points[0].x;
                const sigSelect = document.getElementById('temporal-signature');
                if (sigSelect) {
                    sigSelect.value = signature;
                    this.updateTemporalLineChart();
                }
            }
        });
    },

    async updateTemporalLineChart() {
        const signature = document.getElementById('temporal-signature')?.value;
        const disease = document.getElementById('temporal-disease')?.value;
        const lineChartDiv = document.getElementById('temporal-linechart');

        if (!signature || !disease) {
            lineChartDiv.style.display = 'none';
            return;
        }

        lineChartDiv.style.display = 'block';

        // Get activity data
        const activityData = await API.get('/inflammation/temporal/activity', {
            disease: disease,
            signature_type: this.signatureType
        }) || [];

        // Filter for selected signature
        const sigData = activityData.filter(d => d.signature === signature);

        if (sigData.length === 0) {
            lineChartDiv.innerHTML = '<p class="loading">No data for selected signature</p>';
            return;
        }

        // Sort by timepoint
        sigData.sort((a, b) => a.timepoint_num - b.timepoint_num);

        const timepoints = sigData.map(d => d.timepoint);
        const means = sigData.map(d => d.mean_activity);
        const stds = sigData.map(d => d.std_activity);
        const nSamples = sigData.map(d => d.n_samples);

        const lineTrace = {
            x: timepoints,
            y: means,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Mean Activity',
            line: { color: '#3b82f6', width: 3 },
            marker: { size: 12, color: '#3b82f6' },
            error_y: { type: 'data', array: stds, visible: true, color: 'rgba(59, 130, 246, 0.4)' },
            hovertemplate: '<b>%{x}</b><br>Mean: %{y:.3f}<br>n=%{customdata}<extra></extra>',
            customdata: nSamples,
        };

        Plotly.newPlot(lineChartDiv, [lineTrace], {
            title: { text: `${signature} Activity Across Timepoints (${disease})`, font: { size: 15 } },
            xaxis: { title: 'Timepoint', type: 'category' },
            yaxis: { title: 'Mean Activity (z-score)' },
            margin: { l: 60, r: 30, t: 50, b: 60 },
            font: { family: 'Inter, sans-serif' },
            showlegend: false,
        });

        // Highlight table row
        document.querySelectorAll('#temporal-table-body tr').forEach(row => {
            row.style.background = row.dataset.signature === signature ? '#dbeafe' : '';
        });
    },

    renderTemporalTable(data) {
        const tbody = document.getElementById('temporal-table-body');
        if (!tbody || !data || data.length === 0) {
            tbody.innerHTML = '<tr><td colspan="3" style="text-align: center; padding: 20px;">No temporal data available</td></tr>';
            return;
        }

        // Group by signature
        const signatureData = {};
        data.forEach(d => {
            if (!signatureData[d.signature]) signatureData[d.signature] = {};
            signatureData[d.signature][d.timepoint] = d.mean_activity;
        });

        // Calculate change from T0 to T1
        const rows = [];
        for (const [sig, tps] of Object.entries(signatureData)) {
            const timepoints = Object.keys(tps).sort();
            if (timepoints.length < 2) continue;

            const t0 = tps[timepoints[0]] || 0;
            const t1 = tps[timepoints[1]] || 0;
            const change = t1 - t0;

            let trend = '→';
            let trendColor = '#6b7280';
            if (change > 0.3) { trend = '↑'; trendColor = '#ef4444'; }
            else if (change < -0.3) { trend = '↓'; trendColor = '#3b82f6'; }

            rows.push({ signature: sig, t0, t1, change, trend, trendColor });
        }

        rows.sort((a, b) => Math.abs(b.change) - Math.abs(a.change));

        tbody.innerHTML = rows.map(r => `
            <tr data-signature="${r.signature}" style="cursor: pointer;"
                onclick="AtlasDetailPage.selectTemporalSignature('${r.signature}')"
                onmouseover="this.style.background='#f0f9ff'"
                onmouseout="this.style.background=''">
                <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; font-weight: 500;">${r.signature}</td>
                <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; text-align: center;">
                    ${r.t0.toFixed(2)} → ${r.t1.toFixed(2)}
                    <span style="color: ${r.trendColor}; margin-left: 8px;">(${r.change >= 0 ? '+' : ''}${r.change.toFixed(2)})</span>
                </td>
                <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; text-align: center;">
                    <span style="color: ${r.trendColor}; font-weight: 600; font-size: 18px;">${r.trend}</span>
                </td>
            </tr>
        `).join('');
    },

    selectTemporalSignature(signature) {
        const sigSelect = document.getElementById('temporal-signature');
        if (sigSelect) {
            sigSelect.value = signature;
            this.updateTemporalLineChart();
        }
    },

    async loadInflamSeverity(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Disease Severity Correlation</h3>
                <p>Cytokine activity patterns across disease severity levels. Select a disease to explore how signatures change with severity.</p>
            </div>
            <div class="severity-controls" style="margin-bottom: 16px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap;">
                <div class="control-group">
                    <label for="severity-disease" style="font-weight: 500; margin-right: 8px;">Disease:</label>
                    <select id="severity-disease" class="filter-select" onchange="AtlasDetailPage.updateSeverityPanel()">
                        <option value="">Loading...</option>
                    </select>
                </div>
                <div class="control-group">
                    <label for="severity-signature" style="font-weight: 500; margin-right: 8px;">Signature:</label>
                    <select id="severity-signature" class="filter-select" onchange="AtlasDetailPage.updateSeverityLineChart()">
                        <option value="">All signatures (heatmap)</option>
                    </select>
                </div>
            </div>
            <div id="severity-info" style="margin-bottom: 16px; padding: 12px; background: #f0f9ff; border-radius: 8px; font-size: 14px;"></div>
            <div id="severity-heatmap" class="plot-container" style="height: 500px; margin-bottom: 24px;"></div>
            <div id="severity-linechart" class="plot-container" style="height: 350px; display: none;"></div>
            <div id="severity-table-container" style="max-height: 400px; overflow-y: auto; border: 1px solid #e5e7eb; border-radius: 8px; margin-top: 16px;">
                <table id="severity-table" class="data-table" style="width: 100%; border-collapse: collapse;">
                    <thead style="position: sticky; top: 0; background: #f9fafb; z-index: 1;">
                        <tr>
                            <th style="padding: 10px; text-align: left; border-bottom: 2px solid #e5e7eb;">Signature</th>
                            <th style="padding: 10px; text-align: center; border-bottom: 2px solid #e5e7eb;">Severity Trend</th>
                            <th style="padding: 10px; text-align: center; border-bottom: 2px solid #e5e7eb;">Min → Max</th>
                            <th style="padding: 10px; text-align: center; border-bottom: 2px solid #e5e7eb;">Change</th>
                        </tr>
                    </thead>
                    <tbody id="severity-table-body"></tbody>
                </table>
            </div>
        `;

        await this.initSeverityPanel();
    },

    severityData: null,
    severityDiseases: [],

    async initSeverityPanel() {
        // Load list of diseases with severity data
        this.severityDiseases = await API.get('/inflammation/severity/diseases') || [];

        const diseaseSelect = document.getElementById('severity-disease');
        if (diseaseSelect && this.severityDiseases.length > 0) {
            diseaseSelect.innerHTML = this.severityDiseases.map(d =>
                `<option value="${d}" ${d === 'COVID' ? 'selected' : ''}>${d}</option>`
            ).join('');
        }

        // Load signatures for dropdown
        const signatures = await API.get('/inflammation/signatures', { signature_type: this.signatureType }) || [];
        const sigSelect = document.getElementById('severity-signature');
        if (sigSelect && signatures.length > 0) {
            sigSelect.innerHTML = '<option value="">All signatures (heatmap)</option>' +
                signatures.map(s => `<option value="${s}">${s}</option>`).join('');
        }

        await this.updateSeverityPanel();
    },

    async updateSeverityPanel() {
        const disease = document.getElementById('severity-disease')?.value;
        if (!disease) return;

        // Get severity levels for this disease
        const levels = await API.get(`/inflammation/severity/levels/${disease}`) || [];

        // Get heatmap data
        const heatmapData = await API.get(`/inflammation/severity/heatmap/${disease}`, {
            signature_type: this.signatureType
        });

        // Get full severity data for this disease
        this.severityData = await API.get('/inflammation/severity', {
            disease: disease,
            signature_type: this.signatureType
        }) || [];

        // Update info panel
        const infoDiv = document.getElementById('severity-info');
        if (infoDiv) {
            const nSignatures = new Set(this.severityData.map(d => d.signature)).size;
            const nSamples = this.severityData.length > 0 ? this.severityData[0].n_samples : 0;
            infoDiv.innerHTML = `
                <strong>${disease}</strong>: ${levels.length} severity levels
                <span style="margin-left: 16px;">Signatures: <strong>${nSignatures}</strong></span>
                <span style="margin-left: 16px;">Severity progression: <strong>${levels.join(' → ')}</strong></span>
            `;
        }

        // Render heatmap
        this.renderSeverityHeatmap(heatmapData, disease, levels);

        // Render table with severity trends
        this.renderSeverityTable(this.severityData, levels);

        // Reset signature dropdown and hide line chart
        const sigSelect = document.getElementById('severity-signature');
        if (sigSelect) sigSelect.value = '';
        document.getElementById('severity-linechart').style.display = 'none';
    },

    renderSeverityHeatmap(data, disease, levels) {
        const container = document.getElementById('severity-heatmap');
        if (!container || !data || !data.rows || data.rows.length === 0) {
            container.innerHTML = '<p class="loading">No severity heatmap data available</p>';
            return;
        }

        // data.rows = severity levels, data.columns = signatures, data.values = matrix
        const trace = {
            z: data.values,
            x: data.columns,
            y: data.rows,
            type: 'heatmap',
            colorscale: [
                [0, '#3b82f6'],      // Blue for low/negative
                [0.5, '#f5f5f5'],    // White for zero
                [1, '#ef4444']       // Red for high/positive
            ],
            zmid: 0,
            colorbar: {
                title: 'Activity',
                titleside: 'right'
            },
            hovertemplate: '<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>',
        };

        Plotly.newPlot(container, [trace], {
            title: { text: `${disease}: Cytokine Activity by Severity`, font: { size: 16 } },
            xaxis: {
                title: 'Cytokine Signature',
                tickangle: 45,
                tickfont: { size: 10 }
            },
            yaxis: {
                title: 'Severity Level',
                categoryorder: 'array',
                categoryarray: levels,
            },
            margin: { l: 150, r: 80, t: 50, b: 150 },
            font: { family: 'Inter, sans-serif' },
        });

        // Add click handler to select signature
        container.on('plotly_click', (eventData) => {
            if (eventData.points && eventData.points.length > 0) {
                const signature = eventData.points[0].x;
                const sigSelect = document.getElementById('severity-signature');
                if (sigSelect) {
                    sigSelect.value = signature;
                    this.updateSeverityLineChart();
                }
            }
        });
    },

    updateSeverityLineChart() {
        const signature = document.getElementById('severity-signature')?.value;
        const lineChartDiv = document.getElementById('severity-linechart');

        if (!signature) {
            lineChartDiv.style.display = 'none';
            return;
        }

        lineChartDiv.style.display = 'block';

        // Filter data for selected signature
        const sigData = this.severityData.filter(d => d.signature === signature);

        if (sigData.length === 0) {
            lineChartDiv.innerHTML = '<p class="loading">No data for selected signature</p>';
            return;
        }

        // Sort by severity order
        sigData.sort((a, b) => a.severity_order - b.severity_order);

        const severities = sigData.map(d => d.severity);
        const means = sigData.map(d => d.mean_activity);
        const stds = sigData.map(d => d.std_activity);

        // Line trace
        const lineTrace = {
            x: severities,
            y: means,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Mean Activity',
            line: { color: '#3b82f6', width: 3 },
            marker: { size: 10, color: '#3b82f6' },
            hovertemplate: '<b>%{x}</b><br>Mean: %{y:.3f}<extra></extra>',
        };

        // Error bars
        const errorTrace = {
            x: severities,
            y: means,
            type: 'scatter',
            mode: 'markers',
            error_y: {
                type: 'data',
                array: stds,
                visible: true,
                color: 'rgba(59, 130, 246, 0.4)',
            },
            marker: { size: 0.1, color: 'rgba(0,0,0,0)' },
            hoverinfo: 'skip',
            showlegend: false,
        };

        const disease = document.getElementById('severity-disease')?.value || '';

        Plotly.newPlot(lineChartDiv, [errorTrace, lineTrace], {
            title: { text: `${signature} Activity Across ${disease} Severity`, font: { size: 15 } },
            xaxis: {
                title: 'Severity Level',
                categoryorder: 'array',
                categoryarray: severities,
            },
            yaxis: { title: 'Mean Activity (z-score)' },
            margin: { l: 60, r: 30, t: 50, b: 80 },
            font: { family: 'Inter, sans-serif' },
            showlegend: false,
        });

        // Highlight corresponding row in table
        document.querySelectorAll('#severity-table-body tr').forEach(row => {
            if (row.dataset.signature === signature) {
                row.style.background = '#dbeafe';
                row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else {
                row.style.background = '';
            }
        });
    },

    renderSeverityTable(data, levels) {
        const tbody = document.getElementById('severity-table-body');
        if (!tbody || !data || data.length === 0) return;

        // Group by signature and compute trend
        const signatureStats = {};
        data.forEach(d => {
            if (!signatureStats[d.signature]) {
                signatureStats[d.signature] = {};
            }
            signatureStats[d.signature][d.severity] = {
                mean: d.mean_activity,
                order: d.severity_order
            };
        });

        // Compute trend for each signature
        const rows = [];
        for (const [sig, severities] of Object.entries(signatureStats)) {
            const orderedSeverities = Object.entries(severities)
                .sort((a, b) => a[1].order - b[1].order);

            if (orderedSeverities.length < 2) continue;

            const values = orderedSeverities.map(([_, v]) => v.mean);
            const minVal = Math.min(...values);
            const maxVal = Math.max(...values);
            const firstVal = values[0];
            const lastVal = values[values.length - 1];
            const change = lastVal - firstVal;

            // Determine trend direction
            let trend = '→';
            let trendColor = '#6b7280';
            if (change > 0.3) {
                trend = '↑';
                trendColor = '#ef4444';
            } else if (change < -0.3) {
                trend = '↓';
                trendColor = '#3b82f6';
            }

            // Create sparkline data
            const sparkline = values.map((v, i) => {
                const normalized = (v - minVal) / (maxVal - minVal + 0.001);
                return `${i * 20},${30 - normalized * 25}`;
            }).join(' ');

            rows.push({
                signature: sig,
                trend,
                trendColor,
                minVal,
                maxVal,
                change,
                sparkline,
                values
            });
        }

        // Sort by absolute change (most variable first)
        rows.sort((a, b) => Math.abs(b.change) - Math.abs(a.change));

        tbody.innerHTML = rows.map(r => `
            <tr data-signature="${r.signature}" style="cursor: pointer;"
                onclick="AtlasDetailPage.selectSeveritySignature('${r.signature}')"
                onmouseover="this.style.background='#f0f9ff'"
                onmouseout="this.style.background=''">
                <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; font-weight: 500;">${r.signature}</td>
                <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; text-align: center;">
                    <svg width="${r.values.length * 20}" height="30" style="vertical-align: middle;">
                        <polyline points="${r.sparkline}" fill="none" stroke="${r.trendColor}" stroke-width="2"/>
                    </svg>
                </td>
                <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; text-align: center;">
                    ${r.minVal.toFixed(2)} → ${r.maxVal.toFixed(2)}
                </td>
                <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; text-align: center;">
                    <span style="color: ${r.trendColor}; font-weight: 600; font-size: 18px;">${r.trend}</span>
                    <span style="color: ${r.trendColor}; margin-left: 4px;">${r.change >= 0 ? '+' : ''}${r.change.toFixed(2)}</span>
                </td>
            </tr>
        `).join('');
    },

    selectSeveritySignature(signature) {
        const sigSelect = document.getElementById('severity-signature');
        if (sigSelect) {
            sigSelect.value = signature;
            this.updateSeverityLineChart();
        }
    },

    async loadInflamDrivers(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Cell Type Drivers</h3>
                <p>Identification of cell populations driving disease-specific cytokine signatures.
                   Shows which cell types contribute most to each disease's signature profile.</p>
            </div>
            <div class="controls" style="margin-bottom: 16px; display: flex; gap: 16px; flex-wrap: wrap;">
                <div class="control-group">
                    <label>Signature Type</label>
                    <select id="drivers-sig-type" class="filter-select" onchange="AtlasDetailPage.updateDriversPanel()">
                        <option value="CytoSig">CytoSig (43 cytokines)</option>
                        <option value="SecAct">SecAct (1,170 proteins)</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Disease</label>
                    <select id="drivers-disease" class="filter-select" onchange="AtlasDetailPage.updateDriversPanel()">
                        <option value="">Loading...</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Cytokine</label>
                    <select id="drivers-cytokine" class="filter-select" onchange="AtlasDetailPage.updateDriversBar()">
                        <option value="">All (summary)</option>
                    </select>
                </div>
            </div>
            <div class="card" style="margin-bottom: 16px; padding: 12px; background: #f0f9ff; border-radius: 8px;">
                <strong>Driving Cell Populations:</strong> Identify which cell types contribute most to disease-specific cytokine signatures.
                Shows disease vs healthy differential activity stratified by cell type.
            </div>
            <div class="viz-grid">
                <div class="sub-panel" style="grid-column: span 2;">
                    <h4>Cell Type Contribution</h4>
                    <p style="color: #666; font-size: 0.9rem;">Disease vs Healthy effect size (log2FC) by cell type</p>
                    <div id="drivers-bar" class="plot-container" style="height: 450px;"></div>
                </div>
            </div>
            <div class="viz-grid" style="margin-top: 16px;">
                <div class="sub-panel">
                    <h4>Cell Type Heatmap</h4>
                    <p style="color: #666; font-size: 0.9rem;">Disease-specific signatures across cell types</p>
                    <div id="drivers-heatmap" class="plot-container" style="height: 450px;"></div>
                </div>
                <div class="sub-panel">
                    <h4>Cell Type Importance Ranking</h4>
                    <p style="color: #666; font-size: 0.9rem;">Top contributing cell types by number of significant signatures</p>
                    <div id="drivers-importance" class="plot-container" style="height: 450px;"></div>
                </div>
            </div>
        `;

        await this.initDriversPanel();
    },

    driversData: null,
    driversStratifiedData: null,

    async initDriversPanel() {
        // Populate disease dropdown
        const diseases = await API.get('/inflammation/diseases') || [];
        const diseaseSelect = document.getElementById('drivers-disease');
        if (diseaseSelect && diseases.length > 0) {
            const filteredDiseases = diseases.filter(d => d !== 'healthy');
            diseaseSelect.innerHTML = filteredDiseases.map((d, i) =>
                `<option value="${d}" ${i === 0 ? 'selected' : ''}>${d}</option>`
            ).join('');
        }

        // Populate cytokine dropdown
        const sigType = document.getElementById('drivers-sig-type')?.value || 'CytoSig';
        const signatures = await API.get('/inflammation/signatures', { signature_type: sigType }) || [];
        const cytokineSelect = document.getElementById('drivers-cytokine');
        if (cytokineSelect && signatures.length > 0) {
            cytokineSelect.innerHTML = '<option value="">All (summary)</option>' +
                signatures.map(s => `<option value="${s}">${s}</option>`).join('');
        }

        await this.updateDriversPanel();
    },

    async updateDriversPanel() {
        const disease = document.getElementById('drivers-disease')?.value;
        const sigType = document.getElementById('drivers-sig-type')?.value || 'CytoSig';

        if (!disease) return;

        // Get cell type stratified data for this disease
        this.driversStratifiedData = await API.get('/inflammation/celltype-stratified', {
            disease: disease,
            signature_type: sigType,
        }) || [];

        // Get driving populations summary
        this.driversData = await API.get('/inflammation/driving-populations', { disease, signature_type: sigType }) || [];

        // Update all visualizations
        this.updateDriversBar();
        this.updateDriversHeatmap();
        this.updateDriversImportance();
    },

    updateDriversBar() {
        const container = document.getElementById('drivers-bar');
        if (!container || !this.driversStratifiedData) return;

        const cytokine = document.getElementById('drivers-cytokine')?.value;
        const disease = document.getElementById('drivers-disease')?.value;

        let data = this.driversStratifiedData;

        if (cytokine) {
            // Filter for specific cytokine
            data = data.filter(d => d.signature === cytokine);
        }

        if (data.length === 0) {
            container.innerHTML = '<p class="loading">No data available</p>';
            return;
        }

        // Group by cell type and compute mean log2FC
        const cellTypeStats = {};
        data.forEach(d => {
            if (!cellTypeStats[d.cell_type]) {
                cellTypeStats[d.cell_type] = { values: [], sigCount: 0 };
            }
            cellTypeStats[d.cell_type].values.push(d.log2fc);
            if ((d.q_value || d.p_value) < 0.05 && Math.abs(d.log2fc) > 0.5) {
                cellTypeStats[d.cell_type].sigCount++;
            }
        });

        const cellTypes = Object.keys(cellTypeStats);
        const meanLog2FCs = cellTypes.map(ct => {
            const vals = cellTypeStats[ct].values;
            return vals.reduce((a, b) => a + b, 0) / vals.length;
        });

        // Sort by mean absolute log2FC
        const sorted = cellTypes.map((ct, i) => ({ ct, mean: meanLog2FCs[i] }))
            .sort((a, b) => Math.abs(b.mean) - Math.abs(a.mean))
            .slice(0, 25);  // Top 25

        Plotly.newPlot(container, [{
            type: 'bar',
            x: sorted.map(d => d.mean),
            y: sorted.map(d => d.ct),
            orientation: 'h',
            marker: { color: sorted.map(d => d.mean > 0 ? '#ef4444' : '#2563eb') },
            hovertemplate: '<b>%{y}</b><br>Mean log2FC: %{x:.3f}<extra></extra>',
        }], {
            title: cytokine ? `${cytokine} Effect by Cell Type (${disease})` : `Mean Effect by Cell Type (${disease})`,
            xaxis: { title: 'Log2 Fold Change (Disease vs Healthy)', zeroline: true },
            yaxis: { automargin: true, tickfont: { size: 10 } },
            margin: { l: 150, r: 30, t: 50, b: 50 },
            font: { family: 'Inter, sans-serif' },
        });
    },

    updateDriversHeatmap() {
        const container = document.getElementById('drivers-heatmap');
        if (!container || !this.driversStratifiedData) return;

        const disease = document.getElementById('drivers-disease')?.value;
        const data = this.driversStratifiedData;

        // Get unique cell types and signatures
        const cellTypes = [...new Set(data.map(d => d.cell_type))].sort();
        const signatures = [...new Set(data.map(d => d.signature))].sort();

        // Limit for visualization
        const topCellTypes = cellTypes.slice(0, 30);
        const topSignatures = signatures.slice(0, 30);

        // Build lookup
        const lookup = {};
        data.forEach(d => {
            lookup[`${d.cell_type}|${d.signature}`] = d.log2fc;
        });

        // Build matrix
        const zValues = topCellTypes.map(ct =>
            topSignatures.map(sig => lookup[`${ct}|${sig}`] || 0)
        );

        Plotly.newPlot(container, [{
            z: zValues,
            x: topSignatures,
            y: topCellTypes,
            type: 'heatmap',
            colorscale: 'RdBu',
            reversescale: true,
            zmid: 0,
            colorbar: { title: 'log2FC', titleside: 'right' },
            hovertemplate: '<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>',
        }], {
            title: `Cell Type × Signature Differential (${disease})`,
            xaxis: { title: 'Signature', tickangle: 45, tickfont: { size: 9 } },
            yaxis: { title: 'Cell Type', tickfont: { size: 9 }, automargin: true },
            margin: { l: 120, r: 50, t: 50, b: 100 },
            font: { family: 'Inter, sans-serif' },
        });
    },

    updateDriversImportance() {
        const container = document.getElementById('drivers-importance');
        if (!container) return;

        const disease = document.getElementById('drivers-disease')?.value;

        if (this.driversData && this.driversData.length > 0) {
            // Filter for current disease if available
            let data = this.driversData;
            if (disease) {
                data = data.filter(d => d.disease === disease);
            }

            if (data.length === 0) {
                container.innerHTML = '<p class="loading">No driving population data for this disease</p>';
                return;
            }

            // Sort by number of signatures
            data.sort((a, b) => b.n_signatures - a.n_signatures);
            const top15 = data.slice(0, 15);

            Plotly.newPlot(container, [{
                type: 'bar',
                x: top15.map(d => d.n_signatures),
                y: top15.map(d => d.cell_type),
                orientation: 'h',
                marker: {
                    color: top15.map((d, i) => `rgba(16, 185, 129, ${1 - i * 0.05})`),
                },
                text: top15.map(d => d.top_signatures?.slice(0, 3).join(', ') || ''),
                textposition: 'inside',
                hovertemplate: '<b>%{y}</b><br>Significant signatures: %{x}<br>Top: %{text}<extra></extra>',
            }], {
                title: `Driving Cell Types (${disease})`,
                xaxis: { title: 'Number of Significant Signatures' },
                yaxis: { automargin: true, tickfont: { size: 10 } },
                margin: { l: 120, r: 30, t: 50, b: 50 },
                font: { family: 'Inter, sans-serif' },
            });
        } else {
            // Fallback: compute from stratified data
            if (!this.driversStratifiedData) {
                container.innerHTML = '<p class="loading">No data available</p>';
                return;
            }

            const sigThreshold = 0.05;
            const fcThreshold = 0.5;

            // Count significant signatures per cell type
            const cellTypeCounts = {};
            this.driversStratifiedData.forEach(d => {
                if ((d.q_value || d.p_value) < sigThreshold && Math.abs(d.log2fc) > fcThreshold) {
                    cellTypeCounts[d.cell_type] = (cellTypeCounts[d.cell_type] || 0) + 1;
                }
            });

            const sorted = Object.entries(cellTypeCounts)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 15);

            Plotly.newPlot(container, [{
                type: 'bar',
                x: sorted.map(d => d[1]),
                y: sorted.map(d => d[0]),
                orientation: 'h',
                marker: {
                    color: sorted.map((d, i) => `rgba(16, 185, 129, ${1 - i * 0.05})`),
                },
                hovertemplate: '<b>%{y}</b><br>Significant signatures: %{x}<extra></extra>',
            }], {
                title: `Driving Cell Types (${disease})`,
                xaxis: { title: 'Number of Significant Signatures' },
                yaxis: { automargin: true, tickfont: { size: 10 } },
                margin: { l: 120, r: 30, t: 50, b: 50 },
                font: { family: 'Inter, sans-serif' },
            });
        }
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
            if (container) container.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">No Sankey data available</p>';
            return;
        }

        // Get node labels
        const nodeLabels = data.nodes.map(n => n.name || n);

        // Assign colors based on node type
        const nodeColors = data.nodes.map(n => {
            const nodeType = n.type || '';
            if (nodeType === 'cohort' || nodeType === 'study') return '#1f77b4';      // Blue for studies/cohorts
            if (nodeType === 'disease') return '#ff7f0e';                              // Orange for diseases
            if (nodeType === 'disease_group') return '#2ca02c';                        // Green for disease groups
            return '#7f7f7f';                                                          // Gray for unknown
        });

        Plotly.purge(container);
        Plotly.newPlot(containerId, [{
            type: 'sankey',
            orientation: 'h',
            node: {
                pad: 15,
                thickness: 20,
                line: { color: 'black', width: 0.5 },
                label: nodeLabels,
                color: nodeColors,
                hovertemplate: '<b>%{label}</b><br>Samples: %{value}<extra></extra>'
            },
            link: {
                source: data.links.map(l => l.source),
                target: data.links.map(l => l.target),
                value: data.links.map(l => l.value),
                hovertemplate: '%{source.label} → %{target.label}<br>Samples: %{value}<extra></extra>'
            }
        }], {
            margin: { t: 30, b: 30, l: 50, r: 50 },
            height: 600,
            font: { family: 'Inter, sans-serif' }
        }, { responsive: true });
    },

    renderCohortValidation(data) {
        // Render summary bar chart
        this.renderValidationSummary(data.consistency);

        // Render scatter plot of all signatures
        const filtered = this.getFilteredValidationData();
        this.renderValidationScatter(filtered);

        // Render table of all signatures
        this.renderValidationTable(filtered);

        // Update stats
        this.renderValidationStats({ correlations: filtered, consistency: data.consistency });
    },

    renderValidationSummary(consistency) {
        const container = document.getElementById('validation-summary');
        if (!container || !consistency || consistency.length === 0) {
            container.innerHTML = '<p class="loading">No summary data available</p>';
            return;
        }

        // Also show quality distribution from correlations
        const correlations = this.validationData?.correlations || [];
        const excellent = correlations.filter(c => (c.main_validation_r + c.main_external_r) / 2 >= 0.8).length;
        const good = correlations.filter(c => {
            const avg = (c.main_validation_r + c.main_external_r) / 2;
            return avg >= 0.6 && avg < 0.8;
        }).length;
        const fair = correlations.length - excellent - good;

        // Create two subplots: mean correlation bars and quality distribution pie
        const meanTrace = {
            x: consistency.map(c => c.cohort_pair),
            y: consistency.map(c => c.mean_r),
            type: 'bar',
            name: 'Mean Correlation',
            marker: {
                color: consistency.map(c => c.mean_r >= 0.8 ? '#10b981' : c.mean_r >= 0.6 ? '#f59e0b' : '#ef4444'),
            },
            text: consistency.map(c => `r=${c.mean_r.toFixed(3)}`),
            textposition: 'auto',
            hovertemplate: '<b>%{x}</b><br>Mean r: %{y:.3f}<br>n=%{customdata}<extra></extra>',
            customdata: consistency.map(c => c.n_signatures),
            xaxis: 'x',
            yaxis: 'y',
        };

        const pieTrace = {
            values: [excellent, good, fair],
            labels: ['Excellent (r≥0.8)', 'Good (0.6≤r<0.8)', 'Fair/Poor (r<0.6)'],
            type: 'pie',
            marker: { colors: ['#10b981', '#f59e0b', '#ef4444'] },
            textinfo: 'value+percent',
            hovertemplate: '<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>',
            domain: { x: [0.6, 1], y: [0, 1] },
            hole: 0.4,
        };

        Plotly.newPlot('validation-summary', [meanTrace, pieTrace], {
            title: { text: 'Cross-Cohort Validation Summary', font: { size: 16 } },
            xaxis: { title: 'Cohort Comparison', domain: [0, 0.5] },
            yaxis: { title: 'Mean Correlation (r)', range: [0, 1] },
            showlegend: false,
            font: { family: 'Inter, sans-serif' },
            margin: { t: 50, b: 60, l: 60, r: 30 },
            annotations: [{
                text: `n=${correlations.length}`,
                x: 0.8, y: 0.5,
                font: { size: 14, color: '#666' },
                showarrow: false,
                xref: 'paper', yref: 'paper',
            }],
        });
    },

    renderValidationScatter(correlations) {
        const container = document.getElementById('validation-scatter');
        if (!container || !correlations || correlations.length === 0) {
            container.innerHTML = '<p class="loading">No correlation data available for current filters</p>';
            return;
        }

        const mainValR = correlations.map(c => c.main_validation_r);
        const mainExtR = correlations.map(c => c.main_external_r);
        const signatures = correlations.map(c => c.signature);

        // Color by average quality
        const avgR = correlations.map(c => (c.main_validation_r + c.main_external_r) / 2);
        const colors = avgR.map(r => r >= 0.8 ? '#10b981' : r >= 0.6 ? '#f59e0b' : '#ef4444');

        // Highlight selected point if any
        const selectedIdx = this.validationSelectedSignature
            ? correlations.findIndex(c => c.signature === this.validationSelectedSignature)
            : -1;

        const sizes = correlations.map((c, i) => i === selectedIdx ? 16 : 8);
        const opacities = correlations.map((c, i) => selectedIdx === -1 ? 0.7 : (i === selectedIdx ? 1 : 0.4));
        const lineWidths = correlations.map((c, i) => i === selectedIdx ? 3 : 1);
        const lineColors = correlations.map((c, i) => i === selectedIdx ? '#1e40af' : '#fff');

        const trace = {
            x: mainValR,
            y: mainExtR,
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: sizes,
                color: colors,
                opacity: opacities,
                line: { color: lineColors, width: lineWidths },
            },
            text: signatures,
            customdata: correlations.map(c => c.signature),
            hovertemplate: '<b>%{text}</b><br>Main↔Val: %{x:.3f}<br>Main↔Ext: %{y:.3f}<br><i>Click to select</i><extra></extra>',
        };

        // Diagonal reference line
        const refLine = {
            x: [0, 1],
            y: [0, 1],
            mode: 'lines',
            type: 'scatter',
            line: { color: '#9ca3af', dash: 'dash', width: 1 },
            hoverinfo: 'skip',
            showlegend: false,
        };

        // Quality threshold lines
        const thresholdLine08 = {
            x: [0.8, 0.8, 1],
            y: [0.8, 1, 0.8],
            mode: 'lines',
            type: 'scatter',
            line: { color: '#10b981', dash: 'dot', width: 1 },
            hoverinfo: 'skip',
            showlegend: false,
        };

        Plotly.newPlot('validation-scatter', [refLine, thresholdLine08, trace], {
            title: { text: `Per-Signature Correlations (${correlations.length} signatures)`, font: { size: 16 } },
            xaxis: { title: 'Main ↔ Validation (r)', range: [-0.05, 1.05] },
            yaxis: { title: 'Main ↔ External (r)', range: [-0.05, 1.05] },
            showlegend: false,
            font: { family: 'Inter, sans-serif' },
            margin: { t: 50, b: 60, l: 60, r: 30 },
            hovermode: 'closest',
            shapes: [
                // Excellent zone background
                { type: 'rect', x0: 0.8, x1: 1, y0: 0.8, y1: 1, fillcolor: 'rgba(16, 185, 129, 0.1)', line: { width: 0 } },
            ],
            annotations: [
                { x: 0.9, y: 0.95, text: 'Excellent', font: { size: 11, color: '#10b981' }, showarrow: false },
            ],
        });

        // Add click handler for scatter plot
        container.removeAllListeners?.('plotly_click');
        container.on('plotly_click', (data) => {
            if (data.points && data.points.length > 0) {
                const point = data.points[0];
                // Only handle clicks on the scatter trace (index 2)
                if (point.curveNumber === 2 && point.customdata) {
                    this.selectValidationSignature(point.customdata);
                }
            }
        });
    },

    renderValidationTable(correlations) {
        const tbody = document.getElementById('validation-table-body');
        if (!tbody || !correlations) return;

        // Sort data
        const sorted = [...correlations].sort((a, b) => {
            const aVal = this.validationSortColumn === 'signature' ? a.signature : a[this.validationSortColumn];
            const bVal = this.validationSortColumn === 'signature' ? b.signature : b[this.validationSortColumn];

            if (typeof aVal === 'string') {
                return this.validationSortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            }
            return this.validationSortAsc ? aVal - bVal : bVal - aVal;
        });

        // Generate table rows with click handlers
        tbody.innerHTML = sorted.map(c => {
            const avgR = (c.main_validation_r + c.main_external_r) / 2;
            const quality = avgR >= 0.8 ? 'Excellent' : avgR >= 0.6 ? 'Good' : avgR >= 0.4 ? 'Fair' : 'Poor';
            const qualityColor = avgR >= 0.8 ? '#10b981' : avgR >= 0.6 ? '#f59e0b' : avgR >= 0.4 ? '#6b7280' : '#ef4444';
            const bgColor = avgR >= 0.8 ? '#f0fdf4' : avgR >= 0.6 ? '#fffbeb' : '#fff';
            const isSelected = c.signature === this.validationSelectedSignature;
            const outline = isSelected ? '3px solid #3b82f6' : 'none';

            return `
                <tr data-signature="${c.signature}"
                    style="background: ${bgColor}; cursor: pointer; outline: ${outline}; outline-offset: -3px; transition: all 0.2s;"
                    onclick="AtlasDetailPage.selectValidationSignature('${c.signature}')"
                    onmouseover="this.style.background='#e0f2fe'"
                    onmouseout="this.style.background='${bgColor}'">
                    <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; font-weight: 500;">${c.signature}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; text-align: center;">${c.main_validation_r.toFixed(3)}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; text-align: center;">${c.main_external_r.toFixed(3)}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; text-align: center;">
                        <span style="color: ${qualityColor}; font-weight: 500;">${quality}</span>
                    </td>
                </tr>
            `;
        }).join('');
    },

    renderValidationStats(data) {
        const statsContainer = document.getElementById('validation-stats');
        if (!statsContainer || !data.correlations) return;

        const correlations = data.correlations;
        const n = correlations.length;

        if (n === 0) {
            statsContainer.innerHTML = '<strong>No signatures match current filters</strong>';
            return;
        }

        const avgMainVal = correlations.reduce((sum, c) => sum + c.main_validation_r, 0) / n;
        const avgMainExt = correlations.reduce((sum, c) => sum + c.main_external_r, 0) / n;
        const excellent = correlations.filter(c => (c.main_validation_r + c.main_external_r) / 2 >= 0.8).length;
        const good = correlations.filter(c => {
            const avg = (c.main_validation_r + c.main_external_r) / 2;
            return avg >= 0.6 && avg < 0.8;
        }).length;

        const totalInData = this.validationData?.correlations?.length || n;
        const filterNote = n < totalInData ? ` (filtered from ${totalInData})` : '';

        statsContainer.innerHTML = `
            <strong>Summary Statistics${filterNote}:</strong>
            <span style="margin-left: 16px;">Showing: <strong>${n}</strong></span>
            <span style="margin-left: 16px;">Mean Main↔Val: <strong>${avgMainVal.toFixed(3)}</strong></span>
            <span style="margin-left: 16px;">Mean Main↔Ext: <strong>${avgMainExt.toFixed(3)}</strong></span>
            <span style="margin-left: 16px; color: #10b981;">Excellent (r≥0.8): <strong>${excellent}</strong></span>
            <span style="margin-left: 16px; color: #f59e0b;">Good (0.6≤r<0.8): <strong>${good}</strong></span>
        `;
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
        const disease = document.getElementById('inflam-diff-disease')?.value || 'all';
        const sigType = document.getElementById('inflam-diff-sig-type')?.value || this.signatureType;
        const volcanoContainer = document.getElementById('inflam-volcano');
        const barContainer = document.getElementById('inflam-diff-bar');

        if (!volcanoContainer) return;

        try {
            // Use the differential endpoint for disease-level volcano plot data
            const data = await API.get('/inflammation/differential', {
                disease: disease === 'all' ? null : disease,
                signature_type: sigType,
            });

            if (data && data.length > 0) {
                // Add computed fields (neg_log10_pval may already be present)
                data.forEach(d => {
                    if (!d.neg_log10_pval) {
                        d.neg_log10_pval = -Math.log10(d.p_value || 1);
                    }
                    d.score = Math.abs(d.log2fc || 0) * d.neg_log10_pval;
                });

                // ===== Volcano Plot =====
                const sigThreshold = 0.05;
                const fcThreshold = 0.5;

                // Separate into categories
                const sigUp = data.filter(d => (d.q_value || d.p_value) < sigThreshold && d.log2fc > fcThreshold);
                const sigDown = data.filter(d => (d.q_value || d.p_value) < sigThreshold && d.log2fc < -fcThreshold);
                const notSig = data.filter(d => !((d.q_value || d.p_value) < sigThreshold && Math.abs(d.log2fc) > fcThreshold));

                // Get top hits (by signature name for disease-level data)
                const allSig = [...sigUp, ...sigDown].sort((a, b) => b.score - a.score);
                const topHits = new Set(allSig.slice(0, 10).map(d => d.signature + '|' + d.disease));

                const topHitsUp = sigUp.filter(d => topHits.has(d.signature + '|' + d.disease));
                const topHitsDown = sigDown.filter(d => topHits.has(d.signature + '|' + d.disease));
                const regularUp = sigUp.filter(d => !topHits.has(d.signature + '|' + d.disease));
                const regularDown = sigDown.filter(d => !topHits.has(d.signature + '|' + d.disease));

                const traces = [];

                // Not significant (gray)
                if (notSig.length > 0) {
                    traces.push({
                        type: 'scatter', mode: 'markers', name: 'Not significant',
                        x: notSig.map(d => d.log2fc), y: notSig.map(d => d.neg_log10_pval),
                        text: notSig.map(d => `${d.signature} (${d.disease})`),
                        marker: { color: '#9ca3af', size: 6, opacity: 0.5 },
                        hovertemplate: '<b>%{text}</b><br>log2FC: %{x:.3f}<br>-log10(p): %{y:.2f}<extra></extra>',
                    });
                }

                // Regular up (red)
                if (regularUp.length > 0) {
                    traces.push({
                        type: 'scatter', mode: 'markers', name: 'Up (Disease)',
                        x: regularUp.map(d => d.log2fc), y: regularUp.map(d => d.neg_log10_pval),
                        text: regularUp.map(d => `${d.signature} (${d.disease})`),
                        marker: { color: '#ef4444', size: 8, opacity: 0.7 },
                        hovertemplate: '<b>%{text}</b><br>log2FC: %{x:.3f}<br>-log10(p): %{y:.2f}<extra></extra>',
                    });
                }

                // Regular down (blue)
                if (regularDown.length > 0) {
                    traces.push({
                        type: 'scatter', mode: 'markers', name: 'Down (Disease)',
                        x: regularDown.map(d => d.log2fc), y: regularDown.map(d => d.neg_log10_pval),
                        text: regularDown.map(d => `${d.signature} (${d.disease})`),
                        marker: { color: '#2563eb', size: 8, opacity: 0.7 },
                        hovertemplate: '<b>%{text}</b><br>log2FC: %{x:.3f}<br>-log10(p): %{y:.2f}<extra></extra>',
                    });
                }

                // Top hits up (large red with labels)
                if (topHitsUp.length > 0) {
                    traces.push({
                        type: 'scatter', mode: 'markers+text', name: 'Top Up',
                        x: topHitsUp.map(d => d.log2fc), y: topHitsUp.map(d => d.neg_log10_pval),
                        text: topHitsUp.map(d => d.signature),
                        textposition: 'top center', textfont: { size: 10, color: '#b91c1c' },
                        marker: { color: '#dc2626', size: 12, opacity: 0.9, line: { color: '#fff', width: 1 } },
                        hovertemplate: '<b>%{text}</b><br>log2FC: %{x:.3f}<br>-log10(p): %{y:.2f}<extra></extra>',
                    });
                }

                // Top hits down (large blue with labels)
                if (topHitsDown.length > 0) {
                    traces.push({
                        type: 'scatter', mode: 'markers+text', name: 'Top Down',
                        x: topHitsDown.map(d => d.log2fc), y: topHitsDown.map(d => d.neg_log10_pval),
                        text: topHitsDown.map(d => d.signature),
                        textposition: 'top center', textfont: { size: 10, color: '#1e40af' },
                        marker: { color: '#1d4ed8', size: 12, opacity: 0.9, line: { color: '#fff', width: 1 } },
                        hovertemplate: '<b>%{text}</b><br>log2FC: %{x:.3f}<br>-log10(p): %{y:.2f}<extra></extra>',
                    });
                }

                Plotly.newPlot('inflam-volcano', traces, {
                    xaxis: { title: 'Log2 Fold Change', zeroline: true, zerolinecolor: '#aaa' },
                    yaxis: { title: '-log10(p-value)' },
                    margin: { l: 60, r: 30, t: 30, b: 50 },
                    font: { family: 'Inter, sans-serif' },
                    showlegend: true,
                    legend: { orientation: 'h', y: -0.15 },
                    shapes: [
                        { type: 'line', x0: fcThreshold, x1: fcThreshold, y0: 0, y1: 1, yref: 'paper', line: { dash: 'dash', color: '#ccc' } },
                        { type: 'line', x0: -fcThreshold, x1: -fcThreshold, y0: 0, y1: 1, yref: 'paper', line: { dash: 'dash', color: '#ccc' } },
                        { type: 'line', x0: 0, x1: 1, y0: -Math.log10(sigThreshold), y1: -Math.log10(sigThreshold), xref: 'paper', line: { dash: 'dash', color: '#ccc' } },
                    ],
                });

                // ===== Bar Chart: Top Differential =====
                if (barContainer) {
                    const top20 = allSig.slice(0, 20);
                    const barColors = top20.map(d => d.log2fc > 0 ? '#ef4444' : '#2563eb');

                    Plotly.newPlot('inflam-diff-bar', [{
                        type: 'bar',
                        x: top20.map(d => d.log2fc),
                        y: top20.map(d => `${d.signature} (${d.disease})`),
                        orientation: 'h',
                        marker: { color: barColors },
                        text: top20.map(d => `q=${(d.q_value || d.p_value)?.toExponential(1)}`),
                        textposition: 'outside',
                        hovertemplate: '<b>%{y}</b><br>log2FC: %{x:.3f}<extra></extra>',
                    }], {
                        xaxis: { title: 'Log2 Fold Change', zeroline: true },
                        yaxis: { automargin: true, tickfont: { size: 10 } },
                        margin: { l: 180, r: 60, t: 30, b: 50 },
                        font: { family: 'Inter, sans-serif' },
                    });
                }

                // Update titles
                const diseaseLabel = disease === 'all' ? 'All Diseases' : disease;
                const volcanoTitle = document.getElementById('inflam-diff-volcano-title');
                const barTitle = document.getElementById('inflam-diff-bar-title');
                if (volcanoTitle) volcanoTitle.textContent = `Volcano Plot: ${diseaseLabel} vs Healthy`;
                if (barTitle) barTitle.textContent = `Top Differential Signatures (${diseaseLabel})`;

            } else {
                volcanoContainer.innerHTML = `<p class="loading">No differential data available [${sigType}]</p>`;
                if (barContainer) barContainer.innerHTML = `<p class="loading">No data</p>`;
            }
        } catch (e) {
            volcanoContainer.innerHTML = `<p class="loading">Error: ${e.message}</p>`;
        }
    },

    async updateTreatmentResponse() {
        const disease = document.getElementById('treatment-disease')?.value || 'all';
        const sigType = document.getElementById('treatment-sig-type')?.value || this.signatureType;
        const modelFilter = document.getElementById('treatment-model')?.value || 'all';

        const rocContainer = document.getElementById('treatment-roc');
        const importanceContainer = document.getElementById('treatment-importance');
        const violinContainer = document.getElementById('treatment-violin');

        if (!rocContainer) return;

        try {
            const params = {};
            if (disease !== 'all') params.disease = disease;
            if (modelFilter !== 'all') params.model = modelFilter;

            // Get ROC data
            const rocData = await API.get('/inflammation/treatment-response/roc', params);

            if (rocData && rocData.length > 0) {
                // Color palette for different disease/model combos
                const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];

                // Plot ROC curves
                const rocTraces = rocData.map((curve, i) => ({
                    x: curve.fpr,
                    y: curve.tpr,
                    name: disease === 'all'
                        ? `${curve.disease} - ${curve.model} (AUC=${curve.auc?.toFixed(2)})`
                        : `${curve.model} (AUC=${curve.auc?.toFixed(2)})`,
                    mode: 'lines',
                    type: 'scatter',
                    line: { color: colors[i % colors.length], width: 2 },
                    hovertemplate: `${curve.disease} - ${curve.model}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>`,
                }));

                // Add diagonal reference line
                rocTraces.push({
                    x: [0, 1], y: [0, 1],
                    mode: 'lines',
                    name: 'Random',
                    line: { dash: 'dash', color: '#9ca3af', width: 1 },
                    showlegend: false,
                    hoverinfo: 'skip',
                });

                Plotly.newPlot('treatment-roc', rocTraces, {
                    xaxis: { title: 'False Positive Rate', range: [0, 1] },
                    yaxis: { title: 'True Positive Rate', range: [0, 1] },
                    margin: { l: 60, r: 30, t: 30, b: 50 },
                    font: { family: 'Inter, sans-serif' },
                    legend: { orientation: 'v', x: 1.02, y: 0.5 },
                });
            } else {
                rocContainer.innerHTML = `<p class="loading">No ROC data available [${sigType}]</p>`;
            }

            // Get feature importance
            if (importanceContainer) {
                const featureParams = { ...params };
                if (modelFilter === 'all') featureParams.model = 'Random Forest';  // Default to RF for importance

                const featData = await API.get('/inflammation/treatment-response/features', featureParams);

                if (featData && featData.length > 0) {
                    // Sort by importance and take top 15
                    const top15 = featData.sort((a, b) => b.importance - a.importance).slice(0, 15);

                    Plotly.newPlot('treatment-importance', [{
                        type: 'bar',
                        x: top15.map(d => d.importance),
                        y: top15.map(d => d.feature),
                        orientation: 'h',
                        marker: {
                            color: top15.map((d, i) => `rgba(59, 130, 246, ${1 - i * 0.05})`),
                        },
                        hovertemplate: '<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
                    }], {
                        xaxis: { title: 'Feature Importance' },
                        yaxis: { automargin: true, tickfont: { size: 10 } },
                        margin: { l: 120, r: 30, t: 30, b: 50 },
                        font: { family: 'Inter, sans-serif' },
                    });
                } else {
                    importanceContainer.innerHTML = `<p class="loading">No feature importance data</p>`;
                }
            }

            // Placeholder for violin plot (requires prediction probability data)
            if (violinContainer) {
                // Note: This would require a new API endpoint that returns prediction probabilities
                // For now, show a placeholder or summary
                const summaryData = await API.get('/inflammation/treatment-response', params);

                if (summaryData && summaryData.length > 0) {
                    // Show summary as grouped bar chart instead
                    const diseases = [...new Set(summaryData.map(d => d.disease))];
                    const models = [...new Set(summaryData.map(d => d.model))];

                    const traces = models.map((model, i) => ({
                        type: 'bar',
                        name: model,
                        x: diseases,
                        y: diseases.map(dis => {
                            const match = summaryData.find(d => d.disease === dis && d.model === model);
                            return match?.auc || 0;
                        }),
                        marker: { color: i === 0 ? '#3b82f6' : '#ef4444' },
                        hovertemplate: '<b>%{x}</b><br>%{data.name}: AUC=%{y:.3f}<extra></extra>',
                    }));

                    Plotly.newPlot('treatment-violin', traces, {
                        barmode: 'group',
                        xaxis: { title: 'Disease' },
                        yaxis: { title: 'AUC Score', range: [0, 1] },
                        margin: { l: 60, r: 30, t: 30, b: 50 },
                        font: { family: 'Inter, sans-serif' },
                        legend: { orientation: 'h', y: -0.15 },
                        shapes: [{
                            type: 'line', x0: 0, x1: 1, y0: 0.5, y1: 0.5, xref: 'paper',
                            line: { dash: 'dash', color: '#9ca3af', width: 1 },
                        }],
                    });
                } else {
                    violinContainer.innerHTML = `<p class="loading">No prediction summary data</p>`;
                }
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
