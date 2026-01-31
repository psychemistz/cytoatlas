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
            description: 'Pan-disease immune profiling - 4.9M cells across 20 inflammatory diseases with treatment response data',
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
                { id: 'drivers', label: 'Cell Drivers', icon: '&#128302;' },
            ],
        },
        scatlas: {
            displayName: 'scAtlas',
            description: 'Human tissue reference atlas - 6.4M cells across 35 organs with pan-cancer immune profiling',
            tabs: [
                { id: 'overview', label: 'Overview', icon: '&#127968;' },
                { id: 'celltypes', label: 'Cell Types', icon: '&#128300;' },
                { id: 'tissue-atlas', label: 'Tissue Atlas', icon: '&#128149;' },
                { id: 'differential-analysis', label: 'Differential Analysis', icon: '&#128201;' },
                { id: 'immune-infiltration', label: 'Immune Infiltration', icon: '&#128300;' },
                { id: 'exhaustion', label: 'T Cell Exhaustion', icon: '&#128546;' },
                { id: 'caf', label: 'CAF Types', icon: '&#128302;' },
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
                    <option value="CytoSig">CytoSig (43 cytokines)</option>
                    <option value="SecAct">SecAct (1,170 proteins)</option>
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
                    <div class="search-controls" style="position: relative;">
                        <input type="text" id="cima-protein-search" class="search-input"
                               placeholder="Search ${this.signatureType === 'CytoSig' ? 'cytokine (e.g., IFNG, IL17A, TNF)' : 'protein'}..."
                               oninput="AtlasDetailPage.showCimaProteinSuggestions()"
                               onkeyup="if(event.key==='Enter') AtlasDetailPage.selectFilteredProtein('cima')"
                               onblur="setTimeout(() => document.getElementById('cima-protein-suggestions').style.display = 'none', 200)">
                        <div id="cima-protein-suggestions" style="position: absolute; top: 100%; left: 0; width: 250px; max-height: 200px; overflow-y: auto; background: white; border: 1px solid #ddd; border-radius: 4px; display: none; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"></div>
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

        const searchLower = searchText.toLowerCase();
        const filtered = searchText ?
            signatures.filter(s => s.toLowerCase().includes(searchLower)) :
            signatures;

        const currentValue = select.value;
        select.innerHTML = `<option value="">Select ${this.signatureType === 'CytoSig' ? 'cytokine' : 'protein'}...</option>` +
            filtered.map(s => `<option value="${s}">${s}</option>`).join('');

        // Check for exact match (case-insensitive) and auto-select
        const exactMatch = filtered.find(s => s.toLowerCase() === searchLower);
        if (exactMatch) {
            select.value = exactMatch;
            // Trigger the update function
            if (atlas === 'cima') {
                this.updateCimaActivityProfile();
            } else if (atlas === 'inflammation') {
                this.updateInflamActivityProfile?.();
            } else if (atlas === 'scatlas') {
                this.updateScatlasActivityProfile?.();
            }
        } else if (filtered.includes(currentValue)) {
            // Restore selection if still in filtered list
            select.value = currentValue;
        }
    },

    selectFilteredProtein(atlas) {
        const selectId = atlas === 'cima' ? 'cima-protein-select' :
                         atlas === 'inflammation' ? 'inflam-protein-select' : 'scatlas-protein-select';
        const searchId = atlas === 'cima' ? 'cima-protein-search' :
                         atlas === 'inflammation' ? 'inflam-protein-search' : 'scatlas-protein-search';
        const select = document.getElementById(selectId);
        const searchInput = document.getElementById(searchId);

        if (!select || !searchInput) return;

        // Get visible options (excluding the placeholder)
        const options = Array.from(select.options).filter(opt => opt.value);

        // If there's exactly one option or an exact match, select it
        if (options.length === 1) {
            select.value = options[0].value;
            searchInput.value = options[0].value;
        } else if (options.length > 0) {
            // Check for exact match
            const searchLower = searchInput.value.toLowerCase();
            const exactMatch = options.find(opt => opt.value.toLowerCase() === searchLower);
            if (exactMatch) {
                select.value = exactMatch.value;
                searchInput.value = exactMatch.value;
            } else {
                // Select first match
                select.value = options[0].value;
                searchInput.value = options[0].value;
            }
        }

        // Trigger the update function
        if (atlas === 'cima') {
            this.updateCimaActivityProfile();
        } else if (atlas === 'inflammation') {
            this.updateInflamActivityProfile?.();
        } else if (atlas === 'scatlas') {
            this.updateScatlasActivityProfile?.();
        }
    },

    showCimaProteinSuggestions() {
        const input = document.getElementById('cima-protein-search');
        const div = document.getElementById('cima-protein-suggestions');
        const select = document.getElementById('cima-protein-select');
        if (!input || !div || !this.cimaSignatures) return;

        const query = input.value.toLowerCase();
        if (!query) {
            div.style.display = 'none';
            return;
        }

        const filtered = this.cimaSignatures.filter(s => s.toLowerCase().includes(query));

        // Auto-update dropdown to show filtered options
        if (select) {
            const placeholder = this.signatureType === 'CytoSig' ? 'cytokine' : 'protein';
            select.innerHTML = `<option value="">Select ${placeholder}...</option>` +
                filtered.map(s => `<option value="${s}">${s}</option>`).join('');

            // Auto-select if exact match or single result
            const exactMatch = filtered.find(s => s.toLowerCase() === query);
            if (exactMatch) {
                select.value = exactMatch;
            } else if (filtered.length === 1) {
                select.value = filtered[0];
            }
        }

        // Show suggestions dropdown
        const suggestions = filtered.slice(0, 15);
        if (suggestions.length === 0) {
            div.style.display = 'none';
            return;
        }

        div.innerHTML = suggestions.map(s =>
            `<div style="padding:6px 10px;cursor:pointer;border-bottom:1px solid #eee"
                 onmouseover="this.style.background='#f0f0f0'" onmouseout="this.style.background='white'"
                 onclick="AtlasDetailPage.selectCimaProtein('${s}')">${s}</div>`
        ).join('');
        div.style.display = 'block';
    },

    selectCimaProtein(sig) {
        const input = document.getElementById('cima-protein-search');
        const select = document.getElementById('cima-protein-select');
        const div = document.getElementById('cima-protein-suggestions');
        if (input) input.value = sig;
        if (select) select.value = sig;
        if (div) div.style.display = 'none';
        this.updateCimaActivityProfile();
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

    // Store population stratification data (matching index.html)
    populationStratificationData: null,

    async loadCimaDifferential(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Population Differential Analysis</h3>
                <p>Compare cytokine activity across demographic groups in healthy donors</p>
            </div>

            <!-- Explanation card matching Inflammation style -->
            <div class="card" style="margin-bottom: 1rem; padding: 1rem; background: #f9fafb; border-radius: 8px; border: 1px solid #e5e7eb;">
                <strong>Population Comparison:</strong> Wilcoxon rank-sum test comparing cytokine activity between demographic groups.
                <br><em style="color: #666;">
                Positive effect = higher in Group 1 (e.g., Male, Older, Obese). Negative effect = higher in Group 2 (e.g., Female, Young, Normal).
                </em>
            </div>

            <div class="controls" style="margin-bottom: 16px; display: flex; gap: 16px; flex-wrap: wrap;">
                <div class="control-group">
                    <label>Comparison</label>
                    <select id="pop-stratify" class="filter-select" onchange="AtlasDetailPage.updatePopulationStratification()">
                        <option value="sex">Sex (Male vs Female)</option>
                        <option value="age">Age Group (Older vs Young)</option>
                        <option value="bmi">BMI Category (Obese vs Normal)</option>
                        <option value="blood_type">Blood Type (O vs A)</option>
                        <option value="smoking">Smoking Status (Smoker vs Never)</option>
                    </select>
                </div>
            </div>

            <div class="viz-grid">
                <div class="sub-panel">
                    <h4 id="cima-diff-volcano-title">Volcano Plot: Male vs Female</h4>
                    <p id="cima-diff-volcano-subtitle" style="color: #666; font-size: 0.9rem;">Effect size (activity difference) vs significance (-log10 p-value)</p>
                    <div id="cima-volcano" class="plot-container" style="height: 500px; max-height: 500px; overflow: hidden;"></div>
                </div>
                <div class="sub-panel">
                    <h4 id="cima-diff-bar-title">Top Differential Signatures</h4>
                    <p id="cima-diff-bar-subtitle" style="color: #666; font-size: 0.9rem;">Sorted by significance (|effect| × -log10 p)</p>
                    <div id="cima-diff-bar" class="plot-container" style="height: 500px; max-height: 500px; overflow: hidden;"></div>
                </div>
            </div>
        `;

        // Load population stratification data
        this.populationStratificationData = await API.get('/cima/population-stratification', { signature_type: this.signatureType });
        this.updatePopulationStratification();
    },

    updatePopulationStratification() {
        const volcanoContainer = document.getElementById('cima-volcano');
        const barContainer = document.getElementById('cima-diff-bar');

        if (!volcanoContainer) return;

        const stratifyBy = document.getElementById('pop-stratify')?.value || 'sex';
        const popData = this.populationStratificationData;

        // Define comparison labels
        const compLabels = {
            'sex': { label: 'Male vs Female', g1: 'Male', g2: 'Female' },
            'age': { label: 'Older vs Young', g1: 'Older (≥50)', g2: 'Young (<50)' },
            'bmi': { label: 'Obese vs Normal', g1: 'Obese (≥30)', g2: 'Normal (<25)' },
            'blood_type': { label: 'O vs A', g1: 'Type O', g2: 'Type A' },
            'smoking': { label: 'Smoker vs Never', g1: 'Current/Former', g2: 'Never' }
        };
        const comp = compLabels[stratifyBy] || { label: stratifyBy, g1: 'Group 1', g2: 'Group 2' };

        // Update titles
        const volcanoTitle = document.getElementById('cima-diff-volcano-title');
        const volcanoSubtitle = document.getElementById('cima-diff-volcano-subtitle');
        const barTitle = document.getElementById('cima-diff-bar-title');
        const barSubtitle = document.getElementById('cima-diff-bar-subtitle');

        if (volcanoTitle) volcanoTitle.textContent = `Volcano Plot: ${comp.label}`;
        if (volcanoSubtitle) volcanoSubtitle.innerHTML = `Positive = higher in ${comp.g1}, Negative = higher in ${comp.g2}`;
        if (barTitle) barTitle.textContent = `Top Differential Signatures`;
        if (barSubtitle) barSubtitle.textContent = `Sorted by significance (|effect| × -log10 p)`;

        if (!popData || !popData.effect_sizes || !popData.effect_sizes[stratifyBy]) {
            const noDataMsg = '<p style="text-align:center; padding:2rem; color:#666;">No data available for this comparison.</p>';
            volcanoContainer.innerHTML = noDataMsg;
            if (barContainer) barContainer.innerHTML = noDataMsg;
            return;
        }

        // Get effect sizes array for selected stratification
        const effects = popData.effect_sizes[stratifyBy];

        if (!effects || effects.length === 0) {
            const noDataMsg = '<p style="text-align:center; padding:2rem; color:#666;">No effect data for this comparison.</p>';
            volcanoContainer.innerHTML = noDataMsg;
            if (barContainer) barContainer.innerHTML = noDataMsg;
            return;
        }

        // Transform effects to volcano-style data
        const volcanoData = effects.map(d => ({
            cytokine: d.cytokine,
            effect: d.effect || 0,
            pvalue: d.pvalue || 1,
            neg_log10_pval: -Math.log10(Math.max(d.pvalue || 1, 1e-300)),
            qvalue: d.qvalue || d.pvalue || 1
        }));

        // For SecAct (many proteins), limit to top 200 by significance score
        let filteredData = volcanoData;
        if (this.signatureType === 'SecAct' && filteredData.length > 200) {
            filteredData = [...filteredData].sort((a, b) => {
                const scoreA = a.neg_log10_pval * Math.abs(a.effect);
                const scoreB = b.neg_log10_pval * Math.abs(b.effect);
                return scoreB - scoreA;
            }).slice(0, 200);
        }

        // 1. Volcano plot
        Plotly.purge(volcanoContainer);

        // Color by significance (matching Inflammation style)
        const pThreshold = 0.05;
        const effectThreshold = 0.3;
        const colors = filteredData.map(d => {
            if (d.pvalue < pThreshold && Math.abs(d.effect) > effectThreshold) {
                return d.effect > 0 ? '#f4a6a6' : '#a8d4e6';
            }
            return '#cccccc';
        });

        // Only show text labels for significant points
        const textLabels = filteredData.map(d => {
            if (d.pvalue < pThreshold && Math.abs(d.effect) > effectThreshold) {
                return d.cytokine;
            }
            return '';
        });

        // Dynamic axis range
        const maxAbsEffect = Math.max(1, Math.ceil(Math.max(...filteredData.map(d => Math.abs(d.effect)))));
        const maxY = Math.max(4, Math.ceil(Math.max(...filteredData.map(d => d.neg_log10_pval))));

        Plotly.newPlot(volcanoContainer, [{
            x: filteredData.map(d => d.effect),
            y: filteredData.map(d => d.neg_log10_pval),
            text: textLabels,
            customdata: filteredData.map(d => d.cytokine),
            mode: 'markers+text',
            type: 'scatter',
            marker: {
                color: colors,
                size: 10,
                opacity: 0.7
            },
            textposition: 'top center',
            textfont: { size: 10 },
            hovertemplate: '<b>%{customdata}</b><br>Effect: %{x:.3f}<br>-log10(p): %{y:.2f}<extra></extra>'
        }], {
            xaxis: {
                title: `Effect Size (${comp.g1} - ${comp.g2})`,
                zeroline: true,
                zerolinecolor: '#ccc',
                range: [-maxAbsEffect, maxAbsEffect]
            },
            yaxis: {
                title: '-log10(p-value)',
                range: [0, maxY * 1.1]
            },
            shapes: [
                // Horizontal line at p=0.05
                { type: 'line', x0: -maxAbsEffect, x1: maxAbsEffect, y0: -Math.log10(pThreshold), y1: -Math.log10(pThreshold), line: { color: '#999', dash: 'dash', width: 1 } },
                // Vertical lines at effect thresholds
                { type: 'line', x0: -effectThreshold, x1: -effectThreshold, y0: 0, y1: maxY * 1.1, line: { color: '#999', dash: 'dash', width: 1 } },
                { type: 'line', x0: effectThreshold, x1: effectThreshold, y0: 0, y1: maxY * 1.1, line: { color: '#999', dash: 'dash', width: 1 } }
            ],
            margin: { l: 60, r: 30, t: 30, b: 60 },
            height: 450,
            font: { family: 'Inter, sans-serif' },
            annotations: [{
                x: -maxAbsEffect * 0.8,
                y: -0.08,
                xref: 'x',
                yref: 'paper',
                text: `← Higher in ${comp.g2}`,
                showarrow: false,
                font: { size: 11, color: '#a8d4e6' }
            }, {
                x: maxAbsEffect * 0.8,
                y: -0.08,
                xref: 'x',
                yref: 'paper',
                text: `Higher in ${comp.g1} →`,
                showarrow: false,
                font: { size: 11, color: '#f4a6a6' }
            }]
        }, { responsive: true });

        // 2. Top differential bar chart - sorted by significance score
        if (barContainer) {
            Plotly.purge(barContainer);

            // Calculate significance score and sort
            const scoredData = filteredData.map(d => ({
                ...d,
                score: Math.abs(d.effect) * d.neg_log10_pval
            }));

            // Sort by score descending and take top 20
            const sorted = [...scoredData].sort((a, b) => b.score - a.score);
            const top20 = sorted.slice(0, 20).reverse();  // Reverse for horizontal bar (top at top)

            Plotly.newPlot(barContainer, [{
                type: 'bar',
                orientation: 'h',
                y: top20.map(d => d.cytokine),
                x: top20.map(d => d.effect),
                marker: {
                    color: top20.map(d => d.effect > 0 ? '#f4a6a6' : '#a8d4e6')
                },
                text: top20.map(d => d.effect.toFixed(3)),
                textposition: 'outside',
                textfont: { size: 9 },
                hovertemplate: '<b>%{y}</b><br>Effect: %{x:.3f}<br>p = %{customdata}<extra></extra>',
                customdata: top20.map(d => d.pvalue?.toExponential(2) || 'N/A')
            }], {
                xaxis: { title: 'Effect Size', zeroline: true, zerolinecolor: '#ccc' },
                yaxis: { automargin: true, tickfont: { size: 10 } },
                margin: { l: 120, r: 50, t: 30, b: 50 },
                height: 500,
                font: { family: 'Inter, sans-serif' }
            }, { responsive: true });
        }
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
        const signatures = data.signatures || data.cytokines || [];

        // Use signature type to determine label
        const isSecAct = this.signatureType === 'SecAct';
        const signatureLabel = isSecAct ? 'Protein' : 'Cytokine';

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
                    colorscale: [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
                    zmid: 0,
                    zmin: -maxAbs,
                    zmax: maxAbs,
                    colorbar: { title: 'Mean Activity<br>(z-score)' },
                    hovertemplate: '<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>'
                }], {
                    margin: { l: 120, r: 50, t: 40, b: 120 },
                    xaxis: { tickangle: 45, title: signatureLabel },
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
                <div class="control-group" style="position: relative;">
                    <label>Search Gene/Variant</label>
                    <input type="text" id="eqtl-search" class="filter-select" placeholder="e.g., IFNG, IL6" style="width: 150px;"
                           oninput="AtlasDetailPage.showEqtlSuggestions()"
                           onkeyup="if(event.key==='Enter') { document.getElementById('eqtl-suggestions').style.display='none'; AtlasDetailPage.updateEqtlPlots(); }"
                           onblur="setTimeout(() => document.getElementById('eqtl-suggestions').style.display = 'none', 200)">
                    <div id="eqtl-suggestions" style="position: absolute; top: 100%; left: 0; width: 150px; max-height: 200px; overflow-y: auto; background: white; border: 1px solid #ddd; border-radius: 4px; display: none; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"></div>
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

    showEqtlSuggestions() {
        const input = document.getElementById('eqtl-search');
        const div = document.getElementById('eqtl-suggestions');
        if (!input || !div || !this.eqtlData?.eqtls) return;

        const query = input.value.toLowerCase();
        if (!query) {
            div.style.display = 'none';
            return;
        }

        // Get unique genes from eQTL data
        const genes = [...new Set(this.eqtlData.eqtls.map(d => d.gene).filter(Boolean))];
        const filtered = genes.filter(g => g.toLowerCase().includes(query)).slice(0, 15);

        if (filtered.length === 0) {
            div.style.display = 'none';
            return;
        }

        div.innerHTML = filtered.map(g =>
            `<div style="padding:6px 10px;cursor:pointer;border-bottom:1px solid #eee"
                 onmouseover="this.style.background='#f0f0f0'" onmouseout="this.style.background='white'"
                 onclick="AtlasDetailPage.selectEqtlGene('${g}')">${g}</div>`
        ).join('');
        div.style.display = 'block';
    },

    selectEqtlGene(gene) {
        const input = document.getElementById('eqtl-search');
        const div = document.getElementById('eqtl-suggestions');
        if (input) input.value = gene;
        if (div) div.style.display = 'none';
        this.updateEqtlPlots();
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
                    <div class="card-title">Inflammation Atlas Analysis Data Sources</div>
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
                                <td>66 cell types × (43 CytoSig + 504 SecAct proteins)</td>
                                <td>36,390</td>
                            </tr>
                            <tr>
                                <td>Age/BMI Correlations</td>
                                <td>Sample-level (1,213) + cell type-level (9,438 each)</td>
                                <td>20,089</td>
                            </tr>
                            <tr>
                                <td>Age/BMI Stratified</td>
                                <td>Activity boxplots by age decade and BMI category</td>
                                <td>164,255</td>
                            </tr>
                            <tr>
                                <td>Disease</td>
                                <td>20 diseases × 66 cell types × (43 CytoSig + 186 SecAct)</td>
                                <td>310,973</td>
                            </tr>
                            <tr>
                                <td>Differential</td>
                                <td>Disease vs healthy differential (study-matched controls)</td>
                                <td>4,693</td>
                            </tr>
                            <tr>
                                <td>Severity</td>
                                <td>Disease severity correlation with cytokine activity</td>
                                <td>9,633</td>
                            </tr>
                            <tr>
                                <td>Treatment Response</td>
                                <td>ROC curves (28) + feature importance (560) + predictions</td>
                                <td>1,780</td>
                            </tr>
                            <tr>
                                <td>Validation</td>
                                <td>Cross-cohort correlation (main vs validation vs external)</td>
                                <td>1,219</td>
                            </tr>
                            <tr>
                                <td>Disease Flow</td>
                                <td>Sankey: 20 diseases → 6 disease groups → 3 cohorts</td>
                                <td>116</td>
                            </tr>
                            <tr>
                                <td>Cell Drivers</td>
                                <td>15 diseases × 66 cell types × 143 signatures (study-matched)</td>
                                <td>44,695</td>
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
                colorscale: [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
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
    // Pre-aggregated disease activity data from summary endpoint
    diseaseActivitySummary: null,

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
                <div class="control-group" style="position: relative;">
                    <label>Search ${this.signatureType === 'SecAct' ? 'Protein' : 'Cytokine'}</label>
                    <input type="text" id="inflam-disease-search" class="filter-select"
                           placeholder="${this.signatureType === 'SecAct' ? 'e.g., IFNG, CCL2, MMP9' : 'e.g., IFNG, IL6, TNF'}"
                           style="width: 180px;" autocomplete="off" value="IFNG"
                           oninput="AtlasDetailPage.showDiseaseSuggestions()"
                           onkeyup="if(event.key==='Enter') AtlasDetailPage.updateInflamDiseaseBar()">
                    <div id="inflam-disease-suggestions" style="position: absolute; top: 100%; left: 0; width: 180px; max-height: 200px; overflow-y: auto; background: white; border: 1px solid #ddd; border-radius: 4px; display: none; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"></div>
                </div>
            </div>

            <div class="viz-grid">
                <div class="sub-panel">
                    <div class="panel-header">
                        <h4>Disease-Cell Type Activity</h4>
                        <p id="inflam-disease-bar-subtitle">Activity profile across cell types per disease</p>
                    </div>
                    <div id="inflam-disease-bar" class="plot-container" style="height: 450px;">Loading...</div>
                </div>
                <div class="sub-panel">
                    <div class="panel-header">
                        <h4>Disease Activity Heatmap</h4>
                        <p id="inflam-disease-heatmap-subtitle">Diseases × Signatures</p>
                    </div>
                    <div id="inflam-disease-heatmap" class="plot-container" style="height: 450px;">Loading...</div>
                </div>
            </div>
        `;

        // Load pre-aggregated disease activity summary (much smaller than raw data)
        this.diseaseActivitySummary = await API.get('/inflammation/disease-activity-summary', { signature_type: this.signatureType });
        this.populateInflamDiseaseGroups();
        this.updateInflamDiseaseBar();
        this.updateInflamDiseaseHeatmap();
    },

    getDiseaseSignatures() {
        return this.diseaseActivitySummary?.signatures || [];
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
        const summary = this.diseaseActivitySummary;
        if (!summary?.disease_groups) return;

        const select = document.getElementById('inflam-disease-group');
        if (select && summary.disease_groups.length > 0) {
            select.innerHTML = '<option value="all">All Diseases</option>' +
                summary.disease_groups.map(g => `<option value="${g}">${g}</option>`).join('');
        }
    },

    updateInflamDiseaseBar() {
        const container = document.getElementById('inflam-disease-bar');
        if (!container) return;

        const summary = this.diseaseActivitySummary;
        if (!summary?.bar_data) {
            container.innerHTML = '<p style="text-align: center; color: #666; padding: 2rem;">Disease activity data not available</p>';
            return;
        }

        const diseaseGroup = document.getElementById('inflam-disease-group')?.value || 'all';
        const signature = document.getElementById('inflam-disease-search')?.value || 'IFNG';

        // Update subtitle with current signature
        const subtitle = document.getElementById('inflam-disease-bar-subtitle');
        if (subtitle) {
            subtitle.textContent = `${signature} activity across cell types`;
        }

        // Get pre-aggregated bar data for this signature and disease group
        const sigData = summary.bar_data[signature];
        if (!sigData) {
            container.innerHTML = `<p style="text-align: center; color: #666; padding: 2rem;">No data for "${signature}"</p>`;
            return;
        }

        // Get cell type -> mean_activity mapping for selected disease group
        const cellTypeData = sigData[diseaseGroup] || sigData['all'] || {};

        // Convert to array and sort
        const aggData = Object.entries(cellTypeData)
            .map(([ct, val]) => ({ cell_type: ct, mean_activity: val }))
            .sort((a, b) => b.mean_activity - a.mean_activity)
            .slice(0, 30);

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
                colorscale: [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
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

        const summary = this.diseaseActivitySummary;
        if (!summary?.heatmap) {
            container.innerHTML = '<p style="text-align: center; color: #666; padding: 2rem;">Disease activity data not available</p>';
            return;
        }

        const diseaseGroup = document.getElementById('inflam-disease-group')?.value || 'all';
        const sigType = this.signatureType || 'CytoSig';

        // Update subtitle based on signature type
        const subtitle = document.getElementById('inflam-disease-heatmap-subtitle');
        if (subtitle) {
            subtitle.textContent = sigType === 'SecAct'
                ? 'Top 50 most variable proteins × Diseases'
                : 'Diseases × Signatures';
        }

        // Use pre-computed heatmap data
        let { z: zData, x: signatures, y: diseases } = summary.heatmap;

        // Filter by disease group if specified
        if (diseaseGroup !== 'all' && summary.disease_to_group) {
            const filteredIndices = diseases
                .map((d, i) => summary.disease_to_group[d] === diseaseGroup ? i : -1)
                .filter(i => i !== -1);
            diseases = filteredIndices.map(i => diseases[i]);
            zData = filteredIndices.map(i => zData[i]);
        }

        // For SecAct, limit to top 50 most variable signatures
        if (sigType === 'SecAct' && signatures.length > 50) {
            // Calculate variance for each signature column
            const sigVariance = signatures.map((sig, colIdx) => {
                const vals = zData.map(row => row[colIdx]);
                const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                const variance = vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / vals.length;
                return { sig, colIdx, variance };
            });
            sigVariance.sort((a, b) => b.variance - a.variance);
            const topIndices = sigVariance.slice(0, 50).map(s => s.colIdx).sort((a, b) => a - b);
            signatures = topIndices.map(i => signatures[i]);
            zData = zData.map(row => topIndices.map(i => row[i]));
        }

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
            colorscale: [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
            zmid: 0,
            colorbar: { title: 'Activity' },
            hovertemplate: '<b>%{y}</b><br>%{x}: %{z:.4f}<extra></extra>'
        }], {
            margin: { l: 120, r: 50, t: 30, b: 100 },
            xaxis: { tickangle: 45 },
            height: 450
        }, { responsive: true });
    },

    async loadInflamDifferential(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Disease Differential Analysis</h3>
                <p>Compare cytokine activity between disease and healthy samples</p>
            </div>

            <!-- Explanation card about study-matched vs pooled healthy -->
            <div class="card" style="margin-bottom: 1rem; padding: 1rem; background: #f9fafb; border-radius: 8px; border: 1px solid #e5e7eb;">
                <strong>Disease vs Healthy Comparison:</strong> Wilcoxon rank-sum test comparing cytokine activity between disease and healthy samples.
                <br><em style="color: #666;">
                <strong>Study-matched:</strong> CD, COPD, COVID, HBV, HIV, HNSCC, MS, PS, PSA, RA, SLE, UC, asthma, flu, sepsis - compared to healthy from same study.
                <br><strong style="color: #d62728;">Pooled healthy ⚠️:</strong> BRCA, CRC, NPC, cirrhosis - no matched controls in study; compared to all healthy samples (interpret with caution).
                </em>
            </div>

            <div class="controls" style="margin-bottom: 16px; display: flex; gap: 16px; flex-wrap: wrap;">
                <div class="control-group">
                    <label>Disease</label>
                    <select id="inflam-diff-disease" class="filter-select" onchange="AtlasDetailPage.updateInflamDifferential()">
                        <option value="all">All Diseases vs Healthy</option>
                    </select>
                </div>
            </div>
            <div class="viz-grid">
                <div class="sub-panel">
                    <h4 id="inflam-diff-volcano-title">Volcano Plot: Disease vs Healthy</h4>
                    <p id="inflam-diff-volcano-subtitle" style="color: #666; font-size: 0.9rem;">Effect size (activity difference) vs significance (-log10 p-value)</p>
                    <div id="inflam-volcano" class="plot-container" style="height: 500px;"></div>
                </div>
                <div class="sub-panel">
                    <h4 id="inflam-diff-bar-title">Top Differential Signatures</h4>
                    <p id="inflam-diff-bar-subtitle" style="color: #666; font-size: 0.9rem;">Sorted by significance (|effect| × -log10 p)</p>
                    <div id="inflam-diff-bar" class="plot-container" style="height: 500px;"></div>
                </div>
            </div>
        `;

        // Load raw differential data
        this.inflamDifferentialRaw = await API.get('/inflammation/differential-raw');

        // Populate disease dropdown from raw data
        if (this.inflamDifferentialRaw && this.inflamDifferentialRaw.length > 0) {
            const diseases = [...new Set(this.inflamDifferentialRaw.map(d => d.disease))].filter(d => d !== 'healthy').sort();
            const select = document.getElementById('inflam-diff-disease');
            if (select) {
                select.innerHTML = '<option value="all">All Diseases vs Healthy</option>' +
                    diseases.map(d => `<option value="${d}">${d} vs Healthy</option>`).join('');
            }
        }

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

        // Load raw treatment response data (same format as index.html)
        this.treatmentResponseRaw = await API.get('/inflammation/treatment-response-raw');

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
                <p>Consistency of cytokine/protein activity signatures across main, validation, and external cohorts.</p>
            </div>
            <div class="controls" style="margin-bottom: 16px;">
                <div class="control-group">
                    <label for="validation-cohorts" style="font-weight: 500; margin-right: 8px;">Comparison:</label>
                    <select id="validation-cohorts" class="filter-select" onchange="AtlasDetailPage.updateCohortValidation()">
                        <option value="main-validation">Main vs Validation</option>
                        <option value="main-external">Main vs External</option>
                        <option value="validation-external">Validation vs External</option>
                    </select>
                </div>
            </div>

            <div class="two-col" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div class="viz-container" style="min-height: 450px;">
                    <div class="viz-title" style="font-weight: 600; font-size: 14px; margin-bottom: 4px;">Cross-Cohort Correlation</div>
                    <div class="viz-subtitle" style="color: #666; font-size: 12px; margin-bottom: 8px;">Signature activity correlation between cohorts</div>
                    <div id="cohort-scatter" class="plot-container" style="height: 400px;">Loading data...</div>
                </div>
                <div class="viz-container" style="min-height: 450px;">
                    <div class="viz-title" style="font-weight: 600; font-size: 14px; margin-bottom: 4px;">Consistency Metrics</div>
                    <div class="viz-subtitle" style="color: #666; font-size: 12px; margin-bottom: 8px;">Per-signature correlation coefficients</div>
                    <div id="cohort-consistency" class="plot-container" style="height: 400px;">Loading data...</div>
                </div>
            </div>
        `;

        await this.updateInflamValidation();
    },

    // Validation state
    validationData: null,

    async updateInflamValidation() {
        const sigType = this.signatureType;

        const data = await API.get('/inflammation/cohort-validation', { signature_type: sigType });
        if (data && (data.correlations?.length || data.consistency?.length)) {
            this.validationData = data;
            this.updateCohortValidation();
        } else {
            document.getElementById('cohort-scatter').innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">Cross-cohort validation data will be available after running the full pipeline.</p>';
            document.getElementById('cohort-consistency').innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">Cross-cohort validation data will be available after running the full pipeline.</p>';
        }
    },

    updateCohortValidation() {
        const container1 = document.getElementById('cohort-scatter');
        const container2 = document.getElementById('cohort-consistency');

        const data = this.validationData;
        if (!data || (!data.correlations?.length && !data.consistency?.length)) {
            [container1, container2].forEach(c => {
                if (c) {
                    c.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">Cross-cohort validation data will be available after running the full pipeline.</p>';
                }
            });
            return;
        }

        const sigType = this.signatureType;
        const cohortComparison = document.getElementById('validation-cohorts')?.value || 'main-validation';

        // Filter correlations by signature type
        let correlations = data.correlations || [];
        if (sigType !== 'both') {
            correlations = correlations.filter(d => d.signature_type === sigType);
        }

        // Determine which correlation values to use based on cohort comparison
        let xKey, xLabel;
        if (cohortComparison === 'main-validation') {
            xKey = 'main_validation_r';
            xLabel = 'Main vs Validation (r)';
        } else if (cohortComparison === 'main-external') {
            xKey = 'main_external_r';
            xLabel = 'Main vs External (r)';
        } else {
            xKey = 'validation_external_r';
            xLabel = 'Validation vs External (r)';
        }

        // 1. Histogram of correlation values for selected comparison
        if (container1) {
            Plotly.purge(container1);

            if (correlations.length === 0) {
                container1.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">No correlation data available for selected signature type</p>';
            } else {
                // Get correlation values for the selected comparison
                const rValues = correlations
                    .map(d => d[xKey])
                    .filter(v => v !== null && v !== undefined && !isNaN(v));

                if (rValues.length === 0) {
                    container1.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">No data available for selected cohort comparison</p>';
                } else {
                    // Calculate statistics
                    const meanR = rValues.reduce((a, b) => a + b, 0) / rValues.length;
                    const sortedR = [...rValues].sort((a, b) => a - b);
                    const medianR = sortedR[Math.floor(sortedR.length / 2)];

                    // Create histogram
                    const histTrace = {
                        type: 'histogram',
                        x: rValues,
                        name: 'Distribution',
                        marker: {
                            color: '#1f77b4',
                            opacity: 0.7
                        },
                        xbins: { start: 0, end: 1, size: 0.05 },
                        hovertemplate: 'r: %{x:.2f}<br>Count: %{y}<extra></extra>'
                    };

                    // Add vertical lines for mean and median
                    const shapes = [
                        {
                            type: 'line',
                            x0: meanR, x1: meanR, y0: 0, y1: 1, yref: 'paper',
                            line: { color: '#d62728', width: 2, dash: 'dash' }
                        },
                        {
                            type: 'line',
                            x0: medianR, x1: medianR, y0: 0, y1: 1, yref: 'paper',
                            line: { color: '#2ca02c', width: 2, dash: 'dot' }
                        }
                    ];

                    // Add annotations
                    const annotations = [
                        {
                            x: meanR, y: 1, yref: 'paper', xanchor: 'left',
                            text: ` Mean: ${meanR.toFixed(3)}`, showarrow: false,
                            font: { color: '#d62728', size: 11 }
                        },
                        {
                            x: medianR, y: 0.9, yref: 'paper', xanchor: 'left',
                            text: ` Median: ${medianR.toFixed(3)}`, showarrow: false,
                            font: { color: '#2ca02c', size: 11 }
                        }
                    ];

                    Plotly.newPlot(container1, [histTrace], {
                        xaxis: { title: xLabel, range: [0, 1] },
                        yaxis: { title: 'Number of Signatures' },
                        margin: { l: 60, r: 30, t: 40, b: 60 },
                        height: 400,
                        shapes: shapes,
                        annotations: annotations,
                        title: {
                            text: `${sigType} Signature Reproducibility (n=${rValues.length})`,
                            font: { size: 14 }
                        }
                    }, { responsive: true });
                }
            }
        }

        // 2. Consistency comparison bar chart - show all three comparisons for selected sig type
        if (container2) {
            Plotly.purge(container2);

            let consistency = data.consistency || [];
            // Filter by signature type
            if (sigType !== 'both') {
                consistency = consistency.filter(d => d.signature_type === sigType);
            }

            if (consistency.length === 0) {
                container2.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">No consistency data available</p>';
            } else {
                // Sort by cohort_pair for consistent ordering
                const order = ['Main vs Validation', 'Main vs External', 'Validation vs External'];
                consistency = consistency.sort((a, b) => order.indexOf(a.cohort_pair) - order.indexOf(b.cohort_pair));

                // Highlight the selected comparison
                const selectedLabel = cohortComparison === 'main-validation' ? 'Main vs Validation' :
                                     cohortComparison === 'main-external' ? 'Main vs External' :
                                     'Validation vs External';

                const colors = consistency.map(d =>
                    d.cohort_pair === selectedLabel ? '#1f77b4' : '#aec7e8'
                );

                Plotly.newPlot(container2, [{
                    type: 'bar',
                    x: consistency.map(d => d.cohort_pair),
                    y: consistency.map(d => d.mean_r),
                    text: consistency.map(d => `r=${d.mean_r.toFixed(3)}`),
                    textposition: 'auto',
                    marker: { color: colors },
                    hovertemplate: '<b>%{x}</b><br>Mean r: %{y:.3f}<br>n = %{customdata} signatures<extra></extra>',
                    customdata: consistency.map(d => d.n_signatures)
                }], {
                    xaxis: { title: 'Cohort Comparison' },
                    yaxis: { title: 'Mean Correlation', range: [0, 1] },
                    margin: { l: 50, r: 30, t: 40, b: 100 },
                    height: 400,
                    title: {
                        text: `${sigType} Cross-Cohort Consistency`,
                        font: { size: 14 }
                    }
                }, { responsive: true });
            }
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
            colorscale: [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
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
        // Match index.html layout exactly
        content.innerHTML = `
            <div class="panel-header">
                <h3>Disease Severity Analysis</h3>
                <p>Compare cytokine activity across severity stages within each disease.</p>
            </div>
            <div class="controls" style="margin-bottom: 16px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap;">
                <div class="control-group">
                    <label>Disease</label>
                    <select id="severity-disease" class="filter-select" onchange="AtlasDetailPage.updateSeverityCorrelation()">
                        <option value="">Loading...</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Select Signature</label>
                    <select id="severity-dropdown" class="filter-select" style="width: 150px;" onchange="AtlasDetailPage.onSeverityDropdownChange()">
                        <option value="IFNG">IFNG</option>
                    </select>
                </div>
                <div class="control-group" style="position: relative;">
                    <label>Or Search</label>
                    <input type="text" id="severity-search" placeholder="Search..."
                           style="width: 120px; padding: 8px; border: 1px solid #ddd; border-radius: 4px;"
                           autocomplete="off" value="IFNG"
                           onfocus="AtlasDetailPage.showSeveritySuggestions()"
                           oninput="AtlasDetailPage.showSeveritySuggestions()"
                           onkeydown="if(event.key==='Enter'){document.getElementById('severity-suggestions').style.display='none';AtlasDetailPage.updateSeverityCorrelation();}">
                    <div id="severity-suggestions" style="position: absolute; top: 100%; left: 0; width: 150px; max-height: 200px; overflow-y: auto; background: white; border: 1px solid #ddd; border-radius: 4px; display: none; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"></div>
                </div>
            </div>
            <div class="card" style="margin-bottom: 16px; padding: 12px; background: #f0f9ff; border-radius: 8px;">
                <strong>Disease Severity Analysis:</strong> Compare cytokine activity across severity stages within each disease.
                Select a disease and signature to explore activity patterns.
            </div>
            <div class="viz-grid">
                <div class="sub-panel" style="min-height: 450px;">
                    <h4>Activity by Severity Stage</h4>
                    <p id="severity-heatmap-subtitle" style="color: #666; font-size: 0.9rem;">Mean activity across severity categories</p>
                    <div id="severity-heatmap" class="plot-container" style="height: 420px;"></div>
                </div>
                <div class="sub-panel" style="min-height: 450px;">
                    <h4>Severity Progression</h4>
                    <p id="severity-line-subtitle" style="color: #666; font-size: 0.9rem;">Activity trend across severity stages</p>
                    <div id="severity-line" class="plot-container" style="height: 420px;"></div>
                </div>
            </div>
            <div class="sub-panel" style="margin-top: 16px; min-height: 400px;">
                <h4>Activity Distribution by Severity</h4>
                <p id="severity-boxplot-subtitle" style="color: #666; font-size: 0.9rem;">Distribution across severity categories</p>
                <div id="severity-boxplot" class="plot-container" style="height: 380px;"></div>
            </div>
        `;

        await this.initSeverityPanel();
    },

    // Raw severity data (matching index.html format)
    severityRawData: null,
    severitySignatures: { cytosig: [], secact: [] },

    async initSeverityPanel() {
        // Load raw severity data (same format as index.html)
        this.severityRawData = await API.get('/inflammation/severity-raw') || [];

        if (!this.severityRawData || this.severityRawData.length === 0) {
            console.log('initSeverityPanel: No severity data available');
            return;
        }

        // Extract unique diseases
        const diseases = [...new Set(this.severityRawData.map(d => d.disease))].sort();

        // Disease display names (matching index.html)
        const diseaseLabels = {
            'COVID': 'COVID-19 (7 severity stages)',
            'COPD': 'COPD (GOLD 3-4)',
            'asthma': 'Asthma (Severity levels)',
            'sepsis': 'Sepsis (Severity stages)',
            'HBV': 'Hepatitis B (Fibrosis stages)',
            'cirrhosis': 'Cirrhosis (Child-Pugh)',
            'SLE': 'SLE (SLEDAI scores)',
        };

        const diseaseSelect = document.getElementById('severity-disease');
        if (diseaseSelect && diseases.length > 0) {
            diseaseSelect.innerHTML = diseases.map(d =>
                `<option value="${d}" ${d === 'COVID' ? 'selected' : ''}>${diseaseLabels[d] || d}</option>`
            ).join('');
        }

        // Extract signatures by type
        this.severitySignatures.cytosig = [...new Set(
            this.severityRawData.filter(d => d.signature_type === 'CytoSig').map(d => d.signature)
        )].sort();
        this.severitySignatures.secact = [...new Set(
            this.severityRawData.filter(d => d.signature_type === 'SecAct').map(d => d.signature)
        )].sort();

        // Populate signature dropdown
        this.populateSeverityDropdown();

        // Add blur handler to hide suggestions
        document.getElementById('severity-search')?.addEventListener('blur', () => {
            setTimeout(() => {
                const div = document.getElementById('severity-suggestions');
                if (div) div.style.display = 'none';
            }, 200);
        });

        this.updateSeverityCorrelation();
    },

    populateSeverityDropdown() {
        const sigType = this.signatureType;
        const sigs = sigType === 'SecAct' ? this.severitySignatures.secact : this.severitySignatures.cytosig;

        const dropdown = document.getElementById('severity-dropdown');
        if (dropdown && sigs.length > 0) {
            const defaultSig = sigs.includes('IFNG') ? 'IFNG' : sigs[0];
            dropdown.innerHTML = sigs.map(s =>
                `<option value="${s}" ${s === defaultSig ? 'selected' : ''}>${s}</option>`
            ).join('');
        }
    },

    getSeveritySignatures() {
        const sigType = this.signatureType;
        return sigType === 'SecAct' ? this.severitySignatures.secact : this.severitySignatures.cytosig;
    },

    showSeveritySuggestions() {
        const input = document.getElementById('severity-search');
        const div = document.getElementById('severity-suggestions');
        if (!input || !div) return;

        const query = input.value.toLowerCase();
        const sigs = this.getSeveritySignatures();
        const filtered = sigs.filter(s => s.toLowerCase().includes(query)).slice(0, 15);

        if (filtered.length > 0 && query.length > 0) {
            div.innerHTML = filtered.map(s =>
                `<div style="padding: 8px; cursor: pointer; border-bottom: 1px solid #eee;"
                      onmouseover="this.style.background='#f0f0f0'"
                      onmouseout="this.style.background='white'"
                      onclick="AtlasDetailPage.selectSeveritySig('${s}')">${s}</div>`
            ).join('');
            div.style.display = 'block';
        } else {
            div.style.display = 'none';
        }
    },

    selectSeveritySig(sig) {
        const input = document.getElementById('severity-search');
        const div = document.getElementById('severity-suggestions');
        if (input) input.value = sig;
        if (div) div.style.display = 'none';
        this.updateSeverityCorrelation();
    },

    onSeveritySigTypeChange() {
        // Reset to appropriate default signature when global signature type changes
        const sigType = this.signatureType;
        const input = document.getElementById('severity-search');
        if (input) {
            const sigs = this.getSeveritySignatures();
            if (sigType === 'SecAct' && sigs.length > 0) {
                const defaultSig = ['IFNG', 'CCL3', 'S100A8', 'GRN'].find(s => sigs.includes(s)) || sigs[0];
                input.value = defaultSig;
            } else if (sigs.length > 0) {
                input.value = sigs.includes('IFNG') ? 'IFNG' : sigs[0];
            }
        }
        this.populateSeverityDropdown();
        this.updateSeverityCorrelation();
    },

    onSeverityDropdownChange() {
        const dropdown = document.getElementById('severity-dropdown');
        const search = document.getElementById('severity-search');
        if (dropdown && search) {
            search.value = dropdown.value;
        }
        this.updateSeverityCorrelation();
    },

    // Main update function matching index.html's updateSeverityCorrelation
    updateSeverityCorrelation() {
        const heatmapContainer = document.getElementById('severity-heatmap');
        const lineContainer = document.getElementById('severity-line');
        const boxplotContainer = document.getElementById('severity-boxplot');

        const data = this.severityRawData;
        if (!data || data.length === 0) {
            [heatmapContainer, lineContainer, boxplotContainer].forEach(c => {
                if (c) {
                    c.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">Severity data not available.</p>';
                }
            });
            return;
        }

        const disease = document.getElementById('severity-disease')?.value || 'COVID';
        const sigType = this.signatureType;
        const selectedSig = document.getElementById('severity-search')?.value || 'IFNG';

        // Filter data for this disease and signature type
        let filtered = data.filter(d => d.disease === disease && d.signature_type === sigType);

        if (filtered.length === 0) {
            [heatmapContainer, lineContainer, boxplotContainer].forEach(c => {
                if (c) {
                    c.innerHTML = `<p style="text-align:center; color:#666; padding:2rem;">No ${sigType} data for ${disease}</p>`;
                }
            });
            return;
        }

        // Get severity stages in order
        const severityOrder = [...new Set(filtered.map(d => JSON.stringify({s: d.severity, o: d.severity_order})))]
            .map(s => JSON.parse(s))
            .sort((a, b) => a.o - b.o)
            .map(x => x.s);

        // Get top signatures by variance across severity stages
        const sigVariance = {};
        const signatures = [...new Set(filtered.map(d => d.signature))];
        signatures.forEach(sig => {
            const vals = severityOrder.map(sev => {
                const rec = filtered.find(d => d.signature === sig && d.severity === sev);
                return rec ? rec.mean_activity : 0;
            });
            const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
            sigVariance[sig] = vals.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / vals.length;
        });
        const topSigs = Object.entries(sigVariance)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 20)
            .map(x => x[0]);

        // 1. Heatmap: Signatures × Severity stages (same as index.html)
        if (heatmapContainer) {
            const zData = topSigs.map(sig =>
                severityOrder.map(sev => {
                    const rec = filtered.find(d => d.signature === sig && d.severity === sev);
                    return rec ? rec.mean_activity : null;
                })
            );

            Plotly.newPlot(heatmapContainer, [{
                z: zData,
                x: severityOrder,
                y: topSigs,
                type: 'heatmap',
                colorscale: [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
                zmid: 0,
                colorbar: { title: 'Activity', len: 0.8 },
                hovertemplate: '<b>%{y}</b><br>%{x}<br>Activity: %{z:.3f}<extra></extra>'
            }], {
                margin: { l: 100, r: 60, t: 30, b: 100 },
                xaxis: { title: 'Severity Stage', tickangle: 45 },
                yaxis: { automargin: true },
                height: 420
            }, { responsive: true });

            const subtitle = document.getElementById('severity-heatmap-subtitle');
            if (subtitle) subtitle.textContent = `Top 20 ${sigType} signatures by variance across ${disease} severity stages`;
        }

        // 2. Line chart: Selected signature across severity (same as index.html)
        if (lineContainer) {
            // Get data for selected signature
            const sigData = severityOrder.map(sev => {
                const rec = filtered.find(d => d.signature === selectedSig && d.severity === sev);
                return {
                    severity: sev,
                    mean: rec ? rec.mean_activity : null,
                    std: rec ? rec.std_activity : 0,
                    n: rec ? rec.n_samples : 0
                };
            }).filter(d => d.mean !== null);

            if (sigData.length === 0) {
                lineContainer.innerHTML = `<p style="text-align:center; color:#666; padding:2rem;">No data for ${selectedSig} in ${disease}</p>`;
            } else {
                Plotly.newPlot(lineContainer, [{
                    x: sigData.map(d => d.severity),
                    y: sigData.map(d => d.mean),
                    error_y: {
                        type: 'data',
                        array: sigData.map(d => d.std / Math.sqrt(d.n)),
                        visible: true,
                        color: 'rgba(31, 119, 180, 0.5)'
                    },
                    type: 'scatter',
                    mode: 'lines+markers',
                    marker: { size: 10, color: '#1f77b4' },
                    line: { width: 2, color: '#1f77b4' },
                    hovertemplate: '<b>%{x}</b><br>Activity: %{y:.3f}<br>n=%{customdata}<extra></extra>',
                    customdata: sigData.map(d => d.n)
                }], {
                    margin: { l: 60, r: 30, t: 30, b: 100 },
                    xaxis: { title: 'Severity Stage', tickangle: 45 },
                    yaxis: { title: `${selectedSig} Activity`, zeroline: true },
                    height: 420
                }, { responsive: true });
            }

            const subtitle = document.getElementById('severity-line-subtitle');
            if (subtitle) subtitle.textContent = `${selectedSig} activity across ${disease} severity stages (mean ± SE)`;
        }

        // 3. Grouped bar chart: Top 5 most variable signatures (same as index.html)
        if (boxplotContainer) {
            const top5 = topSigs.slice(0, 5);
            const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'];

            const traces = top5.map((sig, i) => ({
                type: 'bar',
                name: sig,
                x: severityOrder,
                y: severityOrder.map(sev => {
                    const rec = filtered.find(d => d.signature === sig && d.severity === sev);
                    return rec ? rec.mean_activity : 0;
                }),
                error_y: {
                    type: 'data',
                    array: severityOrder.map(sev => {
                        const rec = filtered.find(d => d.signature === sig && d.severity === sev);
                        return rec ? rec.std_activity : 0;
                    }),
                    visible: true
                },
                marker: { color: colors[i] },
                hovertemplate: '<b>%{x}</b><br>' + sig + ': %{y:.3f}<extra></extra>'
            }));

            Plotly.newPlot(boxplotContainer, traces, {
                barmode: 'group',
                margin: { l: 60, r: 30, t: 30, b: 100 },
                xaxis: { title: 'Severity Stage', tickangle: 45 },
                yaxis: { title: 'Activity (mean ± SD)', zeroline: true },
                legend: { orientation: 'h', y: -0.25 },
                height: 380
            }, { responsive: true });

            const subtitle = document.getElementById('severity-boxplot-subtitle');
            if (subtitle) subtitle.textContent = `Top 5 most variable ${sigType} signatures in ${disease}`;
        }
    },

    // Keep for backward compatibility
    severityData: null,
    severityDiseases: [],
    async updateSeverityPanel() { this.updateSeverityCorrelation(); },
    updateSeverityLineChart() { this.updateSeverityCorrelation(); },
    selectSeveritySignature(sig) { this.selectSeveritySig(sig); },

    async loadInflamDrivers(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Cell Type Drivers</h3>
                <p>Identification of cell populations driving disease-specific cytokine signatures.
                   Shows which cell types contribute most to each disease's signature profile.</p>
            </div>
            <div class="controls" style="margin-bottom: 16px; display: flex; gap: 16px; flex-wrap: wrap;">
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

    // Cell Drivers data (raw format matching index.html)
    cellDriversData: null,

    async initDriversPanel() {
        // Load raw cell drivers data (same format as index.html)
        this.cellDriversData = await API.get('/inflammation/cell-drivers');

        if (!this.cellDriversData || !this.cellDriversData.effects) {
            console.log('initDriversPanel: No cell drivers data available');
            return;
        }

        // Populate dropdowns from the data (same as index.html)
        this.populateDriversDropdowns();
        this.updateCellDrivers();
    },

    populateDriversDropdowns() {
        const data = this.cellDriversData;
        if (!data || !data.effects || data.effects.length === 0) {
            console.log('populateDriversDropdowns: No cell drivers data available');
            return;
        }

        const sigType = this.signatureType;

        // Get unique diseases and signatures for the selected signature type
        const filtered = data.effects.filter(d => d.signature_type === sigType);
        const diseases = [...new Set(filtered.map(d => d.disease))].sort();
        const signatures = [...new Set(filtered.map(d => d.signature))].sort();

        // Populate disease dropdown
        const diseaseSelect = document.getElementById('drivers-disease');
        if (diseaseSelect && diseases.length > 0) {
            diseaseSelect.innerHTML = diseases.map((d, i) =>
                `<option value="${d}" ${i === 0 ? 'selected' : ''}>${d}</option>`
            ).join('');
        }

        // Populate signature/cytokine dropdown
        const cytokineSelect = document.getElementById('drivers-cytokine');
        if (cytokineSelect && signatures.length > 0) {
            // Set default to IFNG if available, otherwise first signature
            const defaultSig = signatures.includes('IFNG') ? 'IFNG' : signatures[0];
            cytokineSelect.innerHTML = signatures.map(s =>
                `<option value="${s}" ${s === defaultSig ? 'selected' : ''}>${s}</option>`
            ).join('');
        }

        console.log(`populateDriversDropdowns: ${diseases.length} diseases, ${signatures.length} signatures for ${sigType}`);
    },

    async updateDriversPanel() {
        // Re-populate dropdowns when signature type changes
        this.populateDriversDropdowns();
        this.updateCellDrivers();
    },

    // Main update function matching index.html's updateCellDrivers
    updateCellDrivers() {
        const barContainer = document.getElementById('drivers-bar');
        const heatmapContainer = document.getElementById('drivers-heatmap');
        const importanceContainer = document.getElementById('drivers-importance');

        if (!barContainer) return;

        const data = this.cellDriversData;

        // Check if data is available
        if (!data || !data.effects || data.effects.length === 0) {
            [barContainer, heatmapContainer, importanceContainer].forEach(c => {
                if (c) {
                    c.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">Cell type driver data not available.</p>';
                }
            });
            return;
        }

        const disease = document.getElementById('drivers-disease')?.value || data.diseases[0];
        const signature = document.getElementById('drivers-cytokine')?.value || 'IFNG';
        const sigType = this.signatureType;

        // Filter effects by disease, signature, and signature type (same as index.html)
        let filtered = data.effects.filter(d =>
            d.disease === disease &&
            d.signature === signature &&
            d.signature_type === sigType
        );

        if (filtered.length === 0) {
            // Try without signature type filter
            filtered = data.effects.filter(d => d.disease === disease && d.signature === signature);
        }

        if (filtered.length === 0) {
            barContainer.innerHTML = `<p style="text-align:center; color:#666; padding:2rem;">No data for ${signature} in ${disease}</p>`;
            return;
        }

        // Sort by effect size
        const effects = filtered.sort((a, b) => b.effect - a.effect).slice(0, 20);

        // 1. Bar chart of effect sizes (same as index.html)
        Plotly.newPlot(barContainer, [{
            type: 'bar',
            orientation: 'h',
            y: effects.map(d => d.cell_type),
            x: effects.map(d => d.effect),
            marker: {
                color: effects.map(d => d.pvalue < 0.05 ? (d.effect > 0 ? '#d62728' : '#2ca02c') : '#ccc')
            },
            text: effects.map(d => d.pvalue < 0.05 ? '*' : ''),
            textposition: 'outside',
            hovertemplate: '<b>%{y}</b><br>Effect: %{x:.3f}<br>p = %{customdata[0]:.4f}<br>n_healthy = %{customdata[1]}, n_disease = %{customdata[2]}<extra></extra>',
            customdata: effects.map(d => [d.pvalue, d.n_healthy, d.n_disease])
        }], {
            xaxis: { title: `${signature} Effect Size (Disease vs Matched Healthy)`, zeroline: true },
            yaxis: { automargin: true },
            margin: { l: 150, r: 50, t: 30, b: 50 },
            height: Math.max(300, effects.length * 25),
            annotations: [{
                x: 0.95, y: 1.02, xref: 'paper', yref: 'paper',
                text: '<span style="color:#d62728">↑Disease</span> | <span style="color:#2ca02c">↑Healthy</span>', showarrow: false, font: { size: 10 }
            }]
        }, { responsive: true });

        // 2. Heatmap of signatures × cell types for this disease (same as index.html)
        if (heatmapContainer) {
            // Get all effects for this disease
            const diseaseEffects = data.effects.filter(d => d.disease === disease && d.signature_type === sigType);

            // Get unique signatures and cell types
            const signatures = [...new Set(diseaseEffects.map(d => d.signature))].slice(0, 15);
            const cellTypes = [...new Set(diseaseEffects.map(d => d.cell_type))].slice(0, 15);

            // Build matrix
            const zData = cellTypes.map(ct => {
                return signatures.map(sig => {
                    const effect = diseaseEffects.find(d => d.cell_type === ct && d.signature === sig);
                    return effect ? effect.effect : 0;
                });
            });

            Plotly.newPlot(heatmapContainer, [{
                type: 'heatmap',
                z: zData,
                x: signatures,
                y: cellTypes,
                colorscale: [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
                zmid: 0,
                colorbar: { title: 'Effect' },
                hovertemplate: '<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>'
            }], {
                title: `${disease}: Cell Type × Signature Effects`,
                xaxis: { tickangle: 45, tickfont: { size: 9 } },
                yaxis: { automargin: true, tickfont: { size: 9 } },
                margin: { l: 120, r: 50, t: 50, b: 80 }
            }, { responsive: true });
        }

        // 3. Cell type importance ranking (same as index.html)
        if (importanceContainer) {
            // Count significant signatures per cell type for this disease
            const diseaseEffects = data.effects.filter(d => d.disease === disease && d.signature_type === sigType);
            const cellTypeCounts = {};

            diseaseEffects.forEach(d => {
                if (d.pvalue < 0.05 && Math.abs(d.effect) > 0.5) {
                    cellTypeCounts[d.cell_type] = (cellTypeCounts[d.cell_type] || 0) + 1;
                }
            });

            const sorted = Object.entries(cellTypeCounts)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 15);

            if (sorted.length > 0) {
                Plotly.newPlot(importanceContainer, [{
                    type: 'bar',
                    orientation: 'h',
                    y: sorted.map(d => d[0]),
                    x: sorted.map(d => d[1]),
                    marker: { color: '#10b981' },
                    hovertemplate: '<b>%{y}</b><br>Significant signatures: %{x}<extra></extra>'
                }], {
                    title: `${disease}: Driving Cell Types`,
                    xaxis: { title: 'Number of Significant Signatures (p<0.05, |effect|>0.5)' },
                    yaxis: { automargin: true, tickfont: { size: 9 } },
                    margin: { l: 120, r: 30, t: 50, b: 50 }
                }, { responsive: true });
            } else {
                importanceContainer.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">No significant driving cell types found.</p>';
            }
        }
    },

    // Backward compatibility stubs - all functionality now in updateCellDrivers
    updateDriversBar() {
        this.updateCellDrivers();
    },

    updateDriversHeatmap() {
        // Handled by updateCellDrivers
    },

    updateDriversImportance() {
        // Handled by updateCellDrivers
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
            case 'tissue-atlas':
                await this.loadScatlasTissueAtlas(content);
                break;
            case 'differential-analysis':
                await this.loadScatlasDifferentialAnalysis(content);
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

    // scAtlas Tissue Atlas state (combined normal organs + cancer types)
    scatlasOrganData: null,
    tissueViewMode: 'normal',

    async loadScatlasTissueAtlas(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Tissue Atlas</h3>
                <p>Compare cytokine activity across 35 normal human organs and 13 cancer types from scAtlas.</p>
            </div>

            <div class="controls" style="display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem;">
                <div class="control-group">
                    <label>View Mode</label>
                    <select id="tissue-mode" class="filter-select" onchange="AtlasDetailPage.updateTissueAtlas()">
                        <option value="normal">Normal Tissues</option>
                        <option value="cancer">Cancer Tissues</option>
                        <option value="comparison">Cancer vs Normal (matched)</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Select ${this.signatureType === 'CytoSig' ? 'Cytokine' : 'Protein'}</label>
                    <select id="tissue-signature-dropdown" class="filter-select" style="width: 150px;" onchange="AtlasDetailPage.updateTissueBoxplot()">
                        <option value="IFNG">IFNG</option>
                    </select>
                </div>
                <div class="control-group" style="position: relative;">
                    <label>Or Search</label>
                    <input type="text" id="tissue-signature-search" class="filter-select"
                           placeholder="Search..." style="width: 120px;" autocomplete="off" value=""
                           oninput="AtlasDetailPage.showTissueSignatureSuggestions()"
                           onkeyup="if(event.key==='Enter') AtlasDetailPage.updateTissueBoxplot()"
                           onblur="setTimeout(() => document.getElementById('tissue-signature-suggestions').style.display = 'none', 200)">
                    <div id="tissue-signature-suggestions" style="position: absolute; top: 100%; left: 0; width: 150px; max-height: 200px; overflow-y: auto; background: white; border: 1px solid #ddd; border-radius: 4px; display: none; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"></div>
                </div>
            </div>

            <div class="card" style="margin-bottom: 1rem; padding: 1rem;">
                <strong id="tissue-atlas-description">Tissue Atlas:</strong>
                <span id="tissue-atlas-desc-text">Compare cytokine activity across 35 normal human organs from scAtlas.</span>
            </div>

            <!-- Activity Boxplot (Top) -->
            <div class="viz-container" style="max-height: 650px; overflow-y: auto;">
                <div class="viz-title" id="tissue-boxplot-title" style="font-weight: 600; font-size: 14px; margin-bottom: 4px;">Activity Distribution by Tissue</div>
                <div class="viz-subtitle" id="tissue-boxplot-subtitle" style="color: #666; font-size: 12px; margin-bottom: 8px;">Cell-type agnostic mean activity per tissue</div>
                <div id="tissue-boxplot" class="plot-container" style="min-height: 360px;">Loading...</div>
            </div>

            <!-- Heatmap (Bottom) -->
            <div class="viz-container" style="margin-top: 1.5rem; max-height: 520px; overflow: hidden;">
                <div class="viz-title" id="tissue-heatmap-title" style="font-weight: 600; font-size: 14px; margin-bottom: 4px;">Signature × Tissue Heatmap</div>
                <div class="viz-subtitle" id="tissue-heatmap-subtitle" style="color: #666; font-size: 12px; margin-bottom: 8px;">Activity patterns across tissues</div>
                <div id="tissue-heatmap" class="plot-container" style="height: 480px; max-height: 480px; overflow: hidden;">Loading...</div>
            </div>
        `;

        // Load both normal organ and cancer data (with error handling)
        try {
            this.scatlasOrganData = await API.get('/scatlas/organ-signatures', { signature_type: this.signatureType });
        } catch (e) {
            console.warn('Failed to load organ signatures:', e);
            this.scatlasOrganData = null;
        }

        try {
            this.cancerTypesData = await API.get('/scatlas/cancer-types-signatures', { signature_type: this.signatureType });
        } catch (e) {
            console.warn('Failed to load cancer types signatures:', e);
            this.cancerTypesData = null;
        }

        // Get signatures from available data
        if (this.scatlasOrganData && this.scatlasOrganData.length > 0) {
            this.tissueSignatures = [...new Set(this.scatlasOrganData.map(d => d.signature))].sort();
        } else if (this.cancerTypesData?.cytosig_signatures || this.cancerTypesData?.secact_signatures) {
            const sigs = this.signatureType === 'CytoSig'
                ? this.cancerTypesData.cytosig_signatures
                : this.cancerTypesData.secact_signatures;
            this.tissueSignatures = sigs || [];
        } else {
            this.tissueSignatures = [];
        }

        // Populate dropdown
        const dropdown = document.getElementById('tissue-signature-dropdown');
        if (dropdown && this.tissueSignatures.length > 0) {
            dropdown.innerHTML = this.tissueSignatures.map(s => `<option value="${s}">${s}</option>`).join('');
            dropdown.value = this.tissueSignatures.includes('IFNG') ? 'IFNG' : this.tissueSignatures[0];
        }

        // Render visualizations
        this.updateTissueAtlas();
    },

    tissueSignatures: [],

    showTissueSignatureSuggestions() {
        const input = document.getElementById('tissue-signature-search');
        const div = document.getElementById('tissue-signature-suggestions');
        const dropdown = document.getElementById('tissue-signature-dropdown');
        if (!input || !div || !this.tissueSignatures) return;

        const query = input.value.toLowerCase();
        if (!query) {
            div.style.display = 'none';
            return;
        }

        const filtered = this.tissueSignatures.filter(s => s.toLowerCase().includes(query));

        // Auto-update dropdown
        if (dropdown && query) {
            dropdown.innerHTML = filtered.map(s => `<option value="${s}">${s}</option>`).join('');
            const exactMatch = filtered.find(s => s.toLowerCase() === query);
            if (exactMatch) {
                dropdown.value = exactMatch;
            } else if (filtered.length === 1) {
                dropdown.value = filtered[0];
            }
        }

        // Show suggestions
        const suggestions = filtered.slice(0, 15);
        if (suggestions.length === 0) {
            div.style.display = 'none';
            return;
        }

        div.innerHTML = suggestions.map(s =>
            `<div style="padding:6px 10px;cursor:pointer;border-bottom:1px solid #eee"
                 onmouseover="this.style.background='#f0f0f0'" onmouseout="this.style.background='white'"
                 onclick="AtlasDetailPage.selectTissueSignature('${s}')">${s}</div>`
        ).join('');
        div.style.display = 'block';
    },

    selectTissueSignature(sig) {
        const input = document.getElementById('tissue-signature-search');
        const dropdown = document.getElementById('tissue-signature-dropdown');
        const div = document.getElementById('tissue-signature-suggestions');
        if (input) input.value = sig;
        if (dropdown) dropdown.value = sig;
        if (div) div.style.display = 'none';
        this.updateTissueBoxplot();
    },

    updateTissueAtlas() {
        const mode = document.getElementById('tissue-mode')?.value || 'normal';
        this.tissueViewMode = mode;

        // Count actual data available
        const organCount = this.scatlasOrganData ? [...new Set(this.scatlasOrganData.map(d => d.organ))].length : 0;
        const cancerCount = this.cancerTypesData?.cancer_types?.length || 0;

        // Update description text with actual counts
        const descText = document.getElementById('tissue-atlas-desc-text');
        if (descText) {
            if (mode === 'normal') {
                descText.textContent = `Compare cytokine activity across ${organCount} normal human organs from scAtlas.`;
            } else if (mode === 'cancer') {
                descText.innerHTML = `Compare cytokine activity across ${cancerCount} cancer types. <em style="color:#666;font-size:0.9em;">(Cancer types with sufficient sample size for reliable analysis)</em>`;
            } else {
                descText.textContent = 'Compare cytokine activity between cancer types and their matched normal organs.';
            }
        }

        this.updateTissueBoxplot();
        this.updateTissueHeatmap();
    },

    updateTissueBoxplot() {
        const container = document.getElementById('tissue-boxplot');
        if (!container) return;

        const mode = this.tissueViewMode || 'normal';
        const searchInput = document.getElementById('tissue-signature-search')?.value?.trim();
        const signature = searchInput || document.getElementById('tissue-signature-dropdown')?.value || 'IFNG';

        // Update titles
        const titleEl = document.getElementById('tissue-boxplot-title');
        const subtitleEl = document.getElementById('tissue-boxplot-subtitle');

        if (mode === 'normal') {
            if (!this.scatlasOrganData || this.scatlasOrganData.length === 0) {
                container.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">Organ data not available.</p>';
                return;
            }

            const data = this.scatlasOrganData.filter(d => d.signature === signature);
            if (data.length === 0) {
                container.innerHTML = `<p style="text-align:center; color:#666; padding:2rem;">No data for ${signature}.</p>`;
                return;
            }

            data.sort((a, b) => b.mean_activity - a.mean_activity);
            const organs = data.map(d => d.organ);
            const values = data.map(d => d.mean_activity);

            if (titleEl) titleEl.textContent = `${signature} Activity Across Normal Organs`;
            if (subtitleEl) subtitleEl.textContent = `${this.signatureType} activity z-scores (${organs.length} organs)`;

            // Dynamic height based on number of organs (min 15px per bar)
            const barHeight = Math.max(360, organs.length * 15 + 50);

            Plotly.purge(container);
            Plotly.newPlot(container, [{
                y: organs,
                x: values,
                type: 'bar',
                orientation: 'h',
                marker: {
                    color: values,
                    colorscale: [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
                    cmid: 0
                },
                hovertemplate: '<b>%{y}</b><br>Activity: %{x:.3f}<extra></extra>'
            }], {
                margin: { l: 120, r: 30, t: 10, b: 40 },
                xaxis: { title: 'Mean Activity (z-score)' },
                yaxis: { tickfont: { size: 10 } },
                height: barHeight,
                font: { family: 'Inter, sans-serif' }
            }, { responsive: true });

        } else if (mode === 'cancer') {
            if (!this.cancerTypesData?.data) {
                container.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">Cancer data not available.</p>';
                return;
            }

            const labels = this.cancerTypesData.cancer_labels || {};
            const data = this.cancerTypesData.data.filter(d => d.signature === signature);
            if (data.length === 0) {
                container.innerHTML = `<p style="text-align:center; color:#666; padding:2rem;">No data for ${signature}.</p>`;
                return;
            }

            data.sort((a, b) => b.mean_activity - a.mean_activity);
            const cancers = data.map(d => labels[d.cancer_type] || d.cancer_type);
            const values = data.map(d => d.mean_activity);

            if (titleEl) titleEl.textContent = `${signature} Activity Across Cancer Types`;
            if (subtitleEl) subtitleEl.textContent = `${this.signatureType} activity z-scores (${cancers.length} cancer types)`;

            Plotly.purge(container);
            Plotly.newPlot(container, [{
                y: cancers,
                x: values,
                type: 'bar',
                orientation: 'h',
                marker: {
                    color: values,
                    colorscale: [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
                    cmid: 0
                },
                hovertemplate: '<b>%{y}</b><br>Activity: %{x:.3f}<extra></extra>'
            }], {
                margin: { l: 150, r: 30, t: 10, b: 40 },
                xaxis: { title: 'Mean Activity (z-score)' },
                height: 360
            }, { responsive: true });

        } else {
            // Comparison mode: organ-matched grouped bar chart (side by side)
            if (!this.scatlasOrganData || !this.cancerTypesData?.data) {
                container.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">Data not available for comparison.</p>';
                return;
            }

            const normalData = this.scatlasOrganData.filter(d => d.signature === signature);
            const cancerData = this.cancerTypesData.data.filter(d => d.signature === signature);

            if (normalData.length === 0 && cancerData.length === 0) {
                container.innerHTML = `<p style="text-align:center; color:#666; padding:2rem;">No data for ${signature}.</p>`;
                return;
            }

            if (titleEl) titleEl.textContent = `${signature} Activity: Cancer vs Matched Normal Tissue`;
            if (subtitleEl) subtitleEl.textContent = 'Paired comparison of cancer types with corresponding normal organs';

            // Cancer to organ mapping
            const cancerToOrgan = this.cancerTypesData.cancer_to_organ || {
                'BRCA': 'Breast', 'CRC': 'Colon', 'ESCA': 'Esophagus', 'HCC': 'Liver',
                'HNSC': 'Oral', 'ICC': 'Liver', 'KIRC': 'Kidney', 'LUAD': 'Lung',
                'LYM': 'LymphNode', 'PAAD': 'Pancreas', 'STAD': 'Stomach', 'cSCC': 'Skin'
            };
            const labels = this.cancerTypesData.cancer_labels || {};

            // Build paired data: cancer type + matched normal organ
            const pairedData = [];
            for (const cancer of cancerData) {
                const matchedOrgan = cancerToOrgan[cancer.cancer_type];
                const normalMatch = normalData.find(d => d.organ === matchedOrgan);
                pairedData.push({
                    cancer_type: cancer.cancer_type,
                    cancer_label: labels[cancer.cancer_type] || cancer.cancer_type,
                    organ: matchedOrgan,
                    cancer_activity: cancer.mean_activity,
                    normal_activity: normalMatch ? normalMatch.mean_activity : null
                });
            }

            // Sort by cancer activity (descending)
            pairedData.sort((a, b) => b.cancer_activity - a.cancer_activity);

            // Create grouped bar chart
            const categories = pairedData.map(d => d.cancer_label);
            const cancerValues = pairedData.map(d => d.cancer_activity);
            const normalValues = pairedData.map(d => d.normal_activity);
            const organLabels = pairedData.map(d => d.organ || 'N/A');

            Plotly.purge(container);
            Plotly.newPlot(container, [
                {
                    name: 'Cancer',
                    x: categories,
                    y: cancerValues,
                    type: 'bar',
                    marker: { color: '#b2182b' },
                    hovertemplate: '<b>%{x}</b><br>Cancer: %{y:.3f}<extra></extra>'
                },
                {
                    name: 'Normal',
                    x: categories,
                    y: normalValues,
                    type: 'bar',
                    marker: { color: '#2166ac' },
                    text: organLabels,
                    hovertemplate: '<b>%{x}</b><br>Normal (%{text}): %{y:.3f}<extra></extra>'
                }
            ], {
                margin: { l: 60, r: 30, t: 30, b: 100 },
                xaxis: { title: '', tickangle: -45, tickfont: { size: 10 } },
                yaxis: { title: 'Activity (z-score)' },
                height: 360,
                barmode: 'group',
                showlegend: true,
                legend: { x: 0.5, y: 1.08, orientation: 'h', xanchor: 'center' },
                font: { family: 'Inter, sans-serif' }
            }, { responsive: true });
        }
    },

    updateTissueHeatmap() {
        const container = document.getElementById('tissue-heatmap');
        if (!container) return;

        const mode = this.tissueViewMode || 'normal';
        const titleEl = document.getElementById('tissue-heatmap-title');
        const subtitleEl = document.getElementById('tissue-heatmap-subtitle');

        if (mode === 'normal') {
            if (!this.scatlasOrganData || this.scatlasOrganData.length === 0) {
                container.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">Organ data not available.</p>';
                return;
            }

            const organs = [...new Set(this.scatlasOrganData.map(d => d.organ))].sort();
            const signatures = [...new Set(this.scatlasOrganData.map(d => d.signature))].sort();

            const dataMap = {};
            this.scatlasOrganData.forEach(d => {
                dataMap[`${d.signature}|${d.organ}`] = d.mean_activity;
            });

            const zValues = signatures.map(sig =>
                organs.map(organ => dataMap[`${sig}|${organ}`] ?? null)
            );

            if (titleEl) titleEl.textContent = 'Signature × Organ Heatmap';
            if (subtitleEl) subtitleEl.textContent = `${signatures.length} signatures × ${organs.length} organs`;

            Plotly.purge(container);
            Plotly.newPlot(container, [{
                z: zValues,
                x: organs,
                y: signatures,
                type: 'heatmap',
                colorscale: [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
                zmid: 0,
                colorbar: { title: 'Activity', titleside: 'right', len: 0.8 },
                hovertemplate: '<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>'
            }], {
                margin: { l: 80, r: 60, t: 10, b: 80 },
                height: 480,
                xaxis: { title: 'Organ', tickangle: -45, tickfont: { size: 10 } },
                yaxis: { title: '', autorange: 'reversed', tickfont: { size: 9 } }
            }, { responsive: true });

            // Add click handler
            container.on('plotly_click', (clickData) => {
                const sig = clickData.points[0].y;
                const input = document.getElementById('tissue-signature-search');
                const dropdown = document.getElementById('tissue-signature-dropdown');
                if (input) input.value = sig;
                if (dropdown) dropdown.value = sig;
                this.updateTissueBoxplot();
            });

        } else if (mode === 'cancer') {
            if (!this.cancerTypesData?.data) {
                container.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">Cancer data not available.</p>';
                return;
            }

            const labels = this.cancerTypesData.cancer_labels || {};
            const signatures = this.signatureType === 'CytoSig'
                ? this.cancerTypesData.cytosig_signatures
                : this.cancerTypesData.secact_signatures?.slice(0, 50);
            const cancerTypes = this.cancerTypesData.cancer_types;

            if (!signatures || !cancerTypes) {
                container.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">No data available.</p>';
                return;
            }

            const z = cancerTypes.map(ct =>
                signatures.map(sig => {
                    const item = this.cancerTypesData.data.find(d => d.cancer_type === ct && d.signature === sig);
                    return item ? item.mean_activity : 0;
                })
            );

            if (titleEl) titleEl.textContent = 'Signature × Cancer Type Heatmap';
            if (subtitleEl) subtitleEl.textContent = `${cancerTypes.length} cancer types × ${signatures.length} signatures`;

            Plotly.purge(container);
            Plotly.newPlot(container, [{
                z: z,
                x: signatures,
                y: cancerTypes.map(ct => labels[ct] || ct),
                type: 'heatmap',
                colorscale: [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
                zmid: 0,
                colorbar: { title: 'Activity', titleside: 'right', len: 0.8 },
                hovertemplate: '<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>'
            }], {
                margin: { l: 150, r: 60, t: 10, b: 80 },
                height: 430,
                xaxis: { title: this.signatureType === 'CytoSig' ? 'Cytokine' : 'Protein', tickangle: 45, tickfont: { size: 10 } },
                yaxis: { title: '', automargin: true, tickfont: { size: 10 } }
            }, { responsive: true });

            // Add click handler
            container.on('plotly_click', (clickData) => {
                const sig = clickData.points[0].x;
                const input = document.getElementById('tissue-signature-search');
                const dropdown = document.getElementById('tissue-signature-dropdown');
                if (input) input.value = sig;
                if (dropdown) dropdown.value = sig;
                this.updateTissueBoxplot();
            });

        } else {
            // Comparison mode: difference heatmap (Cancer - Normal)
            if (!this.scatlasOrganData || !this.cancerTypesData?.data) {
                container.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">Data not available for comparison.</p>';
                return;
            }

            const labels = this.cancerTypesData.cancer_labels || {};
            const cancerToOrgan = this.cancerTypesData.cancer_to_organ || {
                'BRCA': 'Breast', 'CRC': 'Colon', 'ESCA': 'Esophagus', 'HCC': 'Liver',
                'HNSC': 'Oral', 'ICC': 'Liver', 'KIRC': 'Kidney', 'LUAD': 'Lung',
                'LYM': 'LymphNode', 'PAAD': 'Pancreas', 'STAD': 'Stomach', 'cSCC': 'Skin'
            };
            const signatures = this.signatureType === 'CytoSig'
                ? this.cancerTypesData.cytosig_signatures
                : this.cancerTypesData.secact_signatures?.slice(0, 50);
            const cancerTypes = this.cancerTypesData.cancer_types;

            if (!signatures || !cancerTypes) {
                container.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">No data available.</p>';
                return;
            }

            // Build normal lookup map
            const normalMap = {};
            this.scatlasOrganData.forEach(d => {
                normalMap[`${d.organ}|${d.signature}`] = d.mean_activity;
            });

            // Build cancer lookup map
            const cancerMap = {};
            this.cancerTypesData.data.forEach(d => {
                cancerMap[`${d.cancer_type}|${d.signature}`] = d.mean_activity;
            });

            // Calculate difference: Cancer - Normal (for matched organ)
            const z = cancerTypes.map(ct => {
                const matchedOrgan = cancerToOrgan[ct];
                return signatures.map(sig => {
                    const cancerVal = cancerMap[`${ct}|${sig}`] || 0;
                    const normalVal = matchedOrgan ? (normalMap[`${matchedOrgan}|${sig}`] || 0) : 0;
                    return cancerVal - normalVal;
                });
            });

            if (titleEl) titleEl.textContent = 'Cancer vs Normal Difference Heatmap';
            if (subtitleEl) subtitleEl.textContent = `Activity difference (Cancer - matched Normal) for ${cancerTypes.length} cancer types`;

            Plotly.purge(container);
            Plotly.newPlot(container, [{
                z: z,
                x: signatures,
                y: cancerTypes.map(ct => `${labels[ct] || ct} (vs ${cancerToOrgan[ct] || 'N/A'})`),
                type: 'heatmap',
                colorscale: [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
                zmid: 0,
                colorbar: { title: 'Δ Activity', titleside: 'right', len: 0.8 },
                hovertemplate: '<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>'
            }], {
                margin: { l: 200, r: 60, t: 10, b: 80 },
                height: 430,
                xaxis: { title: this.signatureType === 'CytoSig' ? 'Cytokine' : 'Protein', tickangle: 45, tickfont: { size: 10 } },
                yaxis: { title: '', automargin: true, tickfont: { size: 9 } },
                font: { family: 'Inter, sans-serif' }
            }, { responsive: true });

            // Add click handler
            container.on('plotly_click', (clickData) => {
                const sig = clickData.points[0].x;
                const input = document.getElementById('tissue-signature-search');
                const dropdown = document.getElementById('tissue-signature-dropdown');
                if (input) input.value = sig;
                if (dropdown) dropdown.value = sig;
                this.updateTissueBoxplot();
            });
        }
    },

    // scAtlas Cell Types state
    scatlasCelltypeData: null,

    async loadScatlasCelltypes(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Cell Type Activity</h3>
                <p>Cytokine activity patterns across cell types in normal human organs</p>
            </div>

            <div class="controls" style="display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem;">
                <div class="control-group">
                    <label>Filter by Organ</label>
                    <select id="scatlas-organ-filter" class="filter-select" onchange="AtlasDetailPage.updateScatlasCelltypeBar(); AtlasDetailPage.updateScatlasCelltypeHeatmap();">
                        <option value="">All Organs</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Select ${this.signatureType === 'CytoSig' ? 'Cytokine' : 'Protein'}</label>
                    <select id="scatlas-ct-protein-dropdown" class="filter-select" style="width: 150px;" onchange="AtlasDetailPage.updateScatlasCelltypeBar();">
                        <option value="IFNG">IFNG</option>
                    </select>
                </div>
                <div class="control-group" style="position: relative;">
                    <label>Or Search</label>
                    <input type="text" id="scatlas-ct-protein-search" class="filter-select"
                           placeholder="Search..." style="width: 120px;" autocomplete="off" value="IFNG"
                           oninput="AtlasDetailPage.showScatlasCelltypeSuggestions()"
                           onkeyup="if(event.key==='Enter') AtlasDetailPage.updateScatlasCelltypeBar()">
                    <div id="scatlas-ct-suggestions" style="position: absolute; top: 100%; left: 0; width: 120px; max-height: 200px; overflow-y: auto; background: white; border: 1px solid #ddd; border-radius: 4px; display: none; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"></div>
                </div>
            </div>

            <div class="two-col" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div class="viz-container" style="max-height: 520px; overflow: hidden;">
                    <div class="viz-title" style="font-weight: 600; font-size: 14px; margin-bottom: 4px;">Cell Type Activity Profile</div>
                    <div class="viz-subtitle" id="scatlas-celltype-bar-subtitle" style="color: #666; font-size: 12px; margin-bottom: 8px;">Mean activity across cell types</div>
                    <div id="scatlas-celltype-bar" class="plot-container" style="height: 460px; max-height: 460px; overflow: hidden;">Loading...</div>
                </div>
                <div class="viz-container" style="max-height: 520px; overflow: hidden;">
                    <div class="viz-title" style="font-weight: 600; font-size: 14px; margin-bottom: 4px;">Activity Heatmap</div>
                    <div class="viz-subtitle" id="scatlas-celltype-heatmap-subtitle" style="color: #666; font-size: 12px; margin-bottom: 8px;">Top variable cell types × signatures</div>
                    <div id="scatlas-celltype-heatmap" class="plot-container" style="height: 460px; max-height: 460px; overflow: hidden;">Loading...</div>
                </div>
            </div>
        `;

        // Load cell type signatures data
        this.scatlasCelltypeData = await API.get('/scatlas/celltype-signatures', { signature_type: this.signatureType });

        if (this.scatlasCelltypeData) {
            // Populate organ dropdown
            const organSelect = document.getElementById('scatlas-organ-filter');
            if (organSelect && this.scatlasCelltypeData.organs) {
                organSelect.innerHTML = '<option value="">All Organs</option>' +
                    this.scatlasCelltypeData.organs.map(o => `<option value="${o}">${o}</option>`).join('');
            }

            // Populate protein dropdown
            const proteinSelect = document.getElementById('scatlas-ct-protein-dropdown');
            const signatures = this.signatureType === 'CytoSig'
                ? this.scatlasCelltypeData.cytosig_signatures
                : this.scatlasCelltypeData.secact_signatures;
            if (proteinSelect && signatures) {
                proteinSelect.innerHTML = signatures.map(s => `<option value="${s}">${s}</option>`).join('');
            }

            // Render visualizations
            this.updateScatlasCelltypeBar();
            this.updateScatlasCelltypeHeatmap();
        } else {
            document.getElementById('scatlas-celltype-bar').innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">Cell type data not available.</p>';
            document.getElementById('scatlas-celltype-heatmap').innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">Cell type data not available.</p>';
        }
    },

    showScatlasCelltypeSuggestions() {
        const input = document.getElementById('scatlas-ct-protein-search');
        const div = document.getElementById('scatlas-ct-suggestions');
        const dropdown = document.getElementById('scatlas-ct-protein-dropdown');
        if (!input || !div || !this.scatlasCelltypeData) return;

        const query = input.value.toLowerCase();
        const signatures = this.signatureType === 'CytoSig'
            ? this.scatlasCelltypeData.cytosig_signatures
            : this.scatlasCelltypeData.secact_signatures;

        if (!signatures) return;

        const filtered = signatures.filter(s => s.toLowerCase().includes(query));

        // Auto-update dropdown to show filtered options
        if (dropdown && query) {
            dropdown.innerHTML = filtered.map(s => `<option value="${s}">${s}</option>`).join('');

            // Auto-select if exact match or single result
            const exactMatch = filtered.find(s => s.toLowerCase() === query);
            if (exactMatch) {
                dropdown.value = exactMatch;
            } else if (filtered.length === 1) {
                dropdown.value = filtered[0];
            }
        }

        // Show suggestions dropdown
        const suggestions = filtered.slice(0, 15);
        if (suggestions.length === 0 || !query) {
            div.style.display = 'none';
            return;
        }

        div.innerHTML = suggestions.map(s =>
            `<div style="padding:6px 10px;cursor:pointer;border-bottom:1px solid #eee"
                 onmouseover="this.style.background='#f0f0f0'" onmouseout="this.style.background='white'"
                 onclick="AtlasDetailPage.selectScatlasCelltypeSig('${s}')">${s}</div>`
        ).join('');
        div.style.display = 'block';
    },

    selectScatlasCelltypeSig(sig) {
        const input = document.getElementById('scatlas-ct-protein-search');
        const dropdown = document.getElementById('scatlas-ct-protein-dropdown');
        const div = document.getElementById('scatlas-ct-suggestions');
        if (input) input.value = sig;
        if (dropdown) dropdown.value = sig;
        if (div) div.style.display = 'none';
        this.updateScatlasCelltypeBar();
    },

    updateScatlasCelltypeBar() {
        const container = document.getElementById('scatlas-celltype-bar');
        if (!container) return;

        const ctData = this.scatlasCelltypeData;
        if (!ctData?.data) {
            container.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">Cell type data not available.</p>';
            return;
        }

        const organFilter = document.getElementById('scatlas-organ-filter')?.value || '';
        const searchInput = document.getElementById('scatlas-ct-protein-search')?.value?.trim();
        const signature = searchInput || document.getElementById('scatlas-ct-protein-dropdown')?.value || 'IFNG';

        // Filter data by signature type and signature
        let data = ctData.data.filter(d => d.signature_type === this.signatureType && d.signature === signature);

        // Filter by organ if selected
        if (organFilter) {
            data = data.filter(d => d.organ === organFilter);
        }

        if (data.length === 0) {
            container.innerHTML = `<p style="text-align:center; color:#666; padding:2rem;">No data for ${signature} in selected organ.</p>`;
            return;
        }

        // Aggregate by cell type (average across organs if not filtered)
        const cellTypeMap = {};
        data.forEach(d => {
            if (!cellTypeMap[d.cell_type]) {
                cellTypeMap[d.cell_type] = { sum: 0, count: 0 };
            }
            cellTypeMap[d.cell_type].sum += d.mean_activity;
            cellTypeMap[d.cell_type].count += 1;
        });

        const barData = Object.entries(cellTypeMap)
            .map(([ct, v]) => ({ cell_type: ct, mean: v.sum / v.count }))
            .sort((a, b) => b.mean - a.mean)
            .slice(0, 25);

        // Update subtitle
        const subtitle = document.getElementById('scatlas-celltype-bar-subtitle');
        if (subtitle) {
            const organText = organFilter ? ` in ${organFilter}` : ' across all organs';
            subtitle.textContent = `${signature} activity${organText} (top 25 cell types)`;
        }

        Plotly.purge(container);

        Plotly.newPlot(container, [{
            type: 'bar',
            orientation: 'h',
            y: barData.map(d => d.cell_type),
            x: barData.map(d => d.mean),
            marker: {
                color: barData.map(d => d.mean >= 0 ? '#1f77b4' : '#d62728')
            },
            hovertemplate: '<b>%{y}</b><br>Activity: %{x:.3f}<extra></extra>'
        }], {
            xaxis: { title: `${signature} Activity (z-score)`, zeroline: true },
            yaxis: { automargin: true, tickfont: { size: 10 } },
            margin: { l: 150, r: 30, t: 20, b: 50 },
            height: 460,
            font: { family: 'Inter, sans-serif' }
        }, { responsive: true });
    },

    updateScatlasCelltypeHeatmap() {
        const container = document.getElementById('scatlas-celltype-heatmap');
        if (!container) return;

        const ctData = this.scatlasCelltypeData;
        if (!ctData?.data) {
            container.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">Cell type data not available.</p>';
            return;
        }

        const organFilter = document.getElementById('scatlas-organ-filter')?.value || '';
        const sigType = this.signatureType;

        // Filter data by signature type
        let data = ctData.data.filter(d => d.signature_type === sigType);

        // Filter by organ if selected
        if (organFilter) {
            data = data.filter(d => d.organ === organFilter);
        }

        // Get signatures for this type - limit SecAct to 50 for heatmap performance
        let signatures = sigType === 'CytoSig'
            ? (ctData.cytosig_signatures || [...new Set(data.map(d => d.signature))].sort())
            : (ctData.secact_signatures || [...new Set(data.map(d => d.signature))].sort()).slice(0, 50);

        // Get unique cell types from filtered data - limit to 50 for performance
        let cellTypes = [...new Set(data.map(d => d.cell_type))].slice(0, 50);

        // Update subtitle
        const subtitle = document.getElementById('scatlas-celltype-heatmap-subtitle');
        if (subtitle) {
            const organText = organFilter ? ` in ${organFilter}` : '';
            const sigCount = sigType === 'SecAct' ? ' (top 50 signatures)' : '';
            subtitle.textContent = `${sigType} activity${sigCount} across cell types${organText} (${cellTypes.length} cell types)`;
        }

        Plotly.purge(container);

        if (cellTypes.length === 0 || signatures.length === 0) {
            container.innerHTML = '<p style="text-align: center; color: #666; padding: 2rem;">No data available for this selection</p>';
            return;
        }

        // Build lookup map for O(1) access instead of O(n) filtering
        const dataMap = {};
        data.forEach(d => {
            const key = `${d.cell_type}|${d.signature}`;
            if (!dataMap[key]) dataMap[key] = [];
            dataMap[key].push(d.mean_activity);
        });

        // Create matrix using lookup map
        const zData = cellTypes.map(ct => {
            return signatures.map(sig => {
                const key = `${ct}|${sig}`;
                const values = dataMap[key];
                if (!values || values.length === 0) return 0;
                return values.reduce((a, b) => a + b, 0) / values.length;
            });
        });

        Plotly.newPlot(container, [{
            z: zData,
            x: signatures,
            y: cellTypes,
            type: 'heatmap',
            colorscale: [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
            zmid: 0,
            colorbar: {
                title: 'Activity',
                titleside: 'right',
                len: 0.8
            },
            hovertemplate: '<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>'
        }], {
            margin: { l: 180, r: 50, t: 20, b: 80 },
            xaxis: { tickangle: 45, tickfont: { size: 9 } },
            yaxis: { tickfont: { size: 9 } },
            height: 460,
            font: { family: 'Inter, sans-serif' }
        }, { responsive: true });
    },

    // Differential Analysis state
    diffAnalysisData: null,        // Raw comparison data from /scatlas/cancer-comparison
    diffOrganData: null,           // Normal organ data
    diffCancerTypesData: null,     // Cancer type-specific data

    async loadScatlasDifferentialAnalysis(content) {
        content.innerHTML = `
            <div class="panel-header">
                <h3>Differential Analysis</h3>
                <p>Compare cytokine activity between Tumor and Adjacent tissue (paired samples from cancer patients).</p>
            </div>

            <div class="controls" style="display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem;">
                <div class="control-group">
                    <label>Cancer Type</label>
                    <select id="diff-cancer-dropdown" class="filter-select" style="width: 160px;" onchange="AtlasDetailPage.updateDiffAnalysis()">
                        <option value="all">All Cancers (Pan-Cancer)</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Signature Type</label>
                    <select id="diff-sig-type" class="filter-select" style="width: 150px;" onchange="AtlasDetailPage.updateDiffAnalysis()">
                        <option value="CytoSig">CytoSig (43 cytokines)</option>
                        <option value="SecAct">SecAct (1,170 proteins)</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Select Signature</label>
                    <select id="diff-signature-dropdown" class="filter-select" style="width: 150px;" onchange="AtlasDetailPage.syncDiffSearch(); AtlasDetailPage.updateDiffBoxplot()">
                        <option value="IFNG">IFNG</option>
                    </select>
                </div>
                <div class="control-group" style="position: relative;">
                    <label>Or Search</label>
                    <input type="text" id="diff-signature-search" class="filter-select"
                           placeholder="Search..." style="width: 120px;" autocomplete="off" value="IFNG"
                           oninput="AtlasDetailPage.showDiffSignatureSuggestions()"
                           onkeyup="if(event.key==='Enter') { AtlasDetailPage.syncDiffDropdown(); AtlasDetailPage.updateDiffBoxplot(); }"
                           onblur="setTimeout(() => document.getElementById('diff-signature-suggestions').style.display = 'none', 200)">
                    <div id="diff-signature-suggestions" style="position: absolute; top: 100%; left: 0; width: 150px; max-height: 200px; overflow-y: auto; background: white; border: 1px solid #ddd; border-radius: 4px; display: none; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"></div>
                </div>
            </div>

            <!-- Dynamic description card -->
            <div class="card" id="diff-description-card" style="margin-bottom: 1rem; padding: 1rem; background: #f9fafb; border-radius: 8px; border: 1px solid #e5e7eb;">
                <strong>Differential Analysis:</strong> Compare cytokine activity across tissue types (Pan-Cancer):
                <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                    <li><strong style="color:#d62728;">Tumor:</strong> Cancer tissue from pan-cancer scAtlas (147 cell type samples)</li>
                    <li><strong style="color:#ff7f0e;">Adjacent:</strong> Tumor-adjacent normal tissue - sample-matched controls (105 samples)</li>
                    <li><strong style="color:#2ca02c;">Normal:</strong> Healthy tissue from scAtlas normal organs - independent healthy donors</li>
                </ul>
            </div>

            <div class="two-col" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <!-- Volcano plot -->
                <div class="viz-container" style="min-height: 400px;">
                    <div class="viz-title" style="font-weight: 600; font-size: 14px; margin-bottom: 4px;">Differential Volcano Plot</div>
                    <div class="viz-subtitle" id="diff-volcano-subtitle" style="color: #666; font-size: 12px; margin-bottom: 8px;">Tumor vs Adjacent: log2FC vs significance</div>
                    <div id="diff-volcano" class="plot-container" style="height: 380px;">Loading...</div>
                </div>
                <!-- Top differential signatures -->
                <div class="viz-container" style="min-height: 400px;">
                    <div class="viz-title" style="font-weight: 600; font-size: 14px; margin-bottom: 4px;">Top Differential Signatures</div>
                    <div class="viz-subtitle" style="color: #666; font-size: 12px; margin-bottom: 8px;">Ranked by absolute log2 fold change</div>
                    <div id="diff-top-bar" class="plot-container" style="height: 380px;">Loading...</div>
                </div>
            </div>

            <!-- Boxplot for Tumor vs Adjacent vs Normal -->
            <div class="viz-container" style="margin-top: 1rem; min-height: 350px;">
                <div class="viz-title" id="diff-boxplot-title" style="font-weight: 600; font-size: 14px; margin-bottom: 4px;">Activity Comparison: Tumor vs Adjacent Normal</div>
                <div class="viz-subtitle" id="diff-boxplot-subtitle" style="color: #666; font-size: 12px; margin-bottom: 8px;">Distribution of activity across cell types for selected signature</div>
                <div id="diff-boxplot" class="plot-container" style="height: 300px;">Loading...</div>
            </div>
        `;

        // Load all data for differential analysis
        await this.loadDiffData();

        // Populate dropdowns
        this.populateDiffDropdowns();

        // Render visualizations
        this.updateDiffAnalysis();
    },

    async loadDiffData() {
        const sigType = document.getElementById('diff-sig-type')?.value || 'CytoSig';

        // Load adjacent tissue boxplot data (with proper statistics including by_cancer_type)
        try {
            this.diffBoxplotData = await API.get('/scatlas/adjacent-tissue-boxplots', { signature_type: sigType });
            console.log('Loaded boxplot data:', this.diffBoxplotData?.boxplot_data?.length || 0, 'records');
        } catch (e) {
            console.warn('Failed to load boxplot data:', e);
            this.diffBoxplotData = null;
        }

        // Load organ data for normal comparison
        try {
            this.diffOrganData = await API.get('/scatlas/organ-signatures', { signature_type: sigType });
            console.log('Loaded organ data:', this.diffOrganData?.length || 0, 'records');
        } catch (e) {
            console.warn('Failed to load organ signatures:', e);
            this.diffOrganData = null;
        }

        // Load cancer types data
        try {
            this.diffCancerTypesData = await API.get('/scatlas/cancer-types-signatures', { signature_type: sigType });
            console.log('Loaded cancer types data:', this.diffCancerTypesData?.data?.length || 0, 'records');
        } catch (e) {
            console.warn('Failed to load cancer types signatures:', e);
            this.diffCancerTypesData = null;
        }
    },

    populateDiffDropdowns() {
        // Get cancer types
        const cancerTypes = this.diffBoxplotData?.cancer_types || this.diffCancerTypesData?.cancer_types || [];
        const cancerDropdown = document.getElementById('diff-cancer-dropdown');
        if (cancerDropdown && cancerTypes.length > 0) {
            const currentVal = cancerDropdown.value;
            cancerDropdown.innerHTML = '<option value="all">All Cancers (Pan-Cancer)</option>' +
                cancerTypes.map(ct => `<option value="${ct}">${ct}</option>`).join('');
            if (currentVal && (currentVal === 'all' || cancerTypes.includes(currentVal))) {
                cancerDropdown.value = currentVal;
            }
        }

        // Get signatures from boxplot data
        const sigType = document.getElementById('diff-sig-type')?.value || 'CytoSig';
        let signatures = [];
        if (this.diffBoxplotData?.tumor_vs_adjacent?.length > 0) {
            signatures = [...new Set(this.diffBoxplotData.tumor_vs_adjacent
                .filter(d => d.signature_type === sigType)
                .map(d => d.signature))].sort();
        } else if (this.diffOrganData?.length > 0) {
            signatures = [...new Set(this.diffOrganData.map(d => d.signature))].sort();
        }
        this.diffSignatures = signatures;

        const sigDropdown = document.getElementById('diff-signature-dropdown');
        if (sigDropdown && signatures.length > 0) {
            const currentVal = sigDropdown.value;
            sigDropdown.innerHTML = signatures.slice(0, 100).map(s => `<option value="${s}">${s}</option>`).join('');
            if (signatures.includes(currentVal)) {
                sigDropdown.value = currentVal;
            } else if (signatures.includes('IFNG')) {
                sigDropdown.value = 'IFNG';
            }
        }
    },

    diffSignatures: [],

    async updateDiffAnalysis() {
        // Reload data if signature type changed
        const sigType = document.getElementById('diff-sig-type')?.value || 'CytoSig';
        if (this.lastDiffSigType !== sigType) {
            this.lastDiffSigType = sigType;
            await this.loadDiffData();
            this.populateDiffDropdowns();
        }

        // Update description card
        this.updateDiffDescriptionCard();

        // Update all visualizations
        this.updateDiffVolcano();
        this.updateDiffTopBar();
        this.updateDiffBoxplot();
    },

    updateDiffDescriptionCard() {
        const descCard = document.getElementById('diff-description-card');
        if (!descCard) return;

        const selectedCancer = document.getElementById('diff-cancer-dropdown')?.value || 'all';
        const sigType = document.getElementById('diff-sig-type')?.value || 'CytoSig';

        if (selectedCancer === 'all') {
            const summary = this.diffBoxplotData?.summary || {};
            descCard.innerHTML = `<strong>Differential Analysis:</strong> Compare cytokine activity across tissue types (Pan-Cancer):
            <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                <li><strong style="color:#d62728;">Tumor:</strong> Cancer tissue from pan-cancer scAtlas (${summary.n_tumor_samples || 147} cell type samples)</li>
                <li><strong style="color:#ff7f0e;">Adjacent:</strong> Tumor-adjacent normal tissue - sample-matched controls (${summary.n_adjacent_samples || 105} samples)</li>
                <li><strong style="color:#2ca02c;">Normal:</strong> Healthy tissue from scAtlas normal organs - independent healthy donors</li>
            </ul>`;
        } else {
            // Get cancer-specific sample counts
            const cancerStats = this.diffBoxplotData?.by_cancer_type?.find(
                d => d.cancer_type === selectedCancer && d.signature_type === sigType
            );
            const nTumor = cancerStats?.tumor?.n || 'N/A';
            const nAdj = cancerStats?.adjacent?.n || 'N/A';
            const organMap = this.diffCancerTypesData?.cancer_to_organ || {};
            const matchedOrgan = organMap[selectedCancer] || 'matching organ';

            descCard.innerHTML = `<strong>Differential Analysis:</strong> Compare cytokine activity for <strong>${selectedCancer}</strong>:
            <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                <li><strong style="color:#d62728;">Tumor:</strong> ${selectedCancer} tumor tissue (${nTumor} samples)</li>
                <li><strong style="color:#ff7f0e;">Adjacent:</strong> ${selectedCancer} tumor-adjacent normal tissue (${nAdj} samples)</li>
            </ul>
            <small style="color:#666;">Note: Normal tissue reference (${matchedOrgan}) available in Pan-Cancer view.</small>`;
        }
    },

    syncDiffSearch() {
        const dropdown = document.getElementById('diff-signature-dropdown');
        const searchInput = document.getElementById('diff-signature-search');
        if (dropdown && searchInput) {
            searchInput.value = dropdown.value;
        }
    },

    syncDiffDropdown() {
        const dropdown = document.getElementById('diff-signature-dropdown');
        const searchInput = document.getElementById('diff-signature-search');
        if (dropdown && searchInput) {
            const val = searchInput.value.trim();
            if (this.diffSignatures.includes(val)) {
                dropdown.value = val;
            }
        }
    },

    showDiffSignatureSuggestions() {
        const input = document.getElementById('diff-signature-search');
        const div = document.getElementById('diff-signature-suggestions');
        const dropdown = document.getElementById('diff-signature-dropdown');
        if (!input || !div || !this.diffSignatures) return;

        const query = input.value.toLowerCase();
        if (!query) {
            div.style.display = 'none';
            return;
        }

        const filtered = this.diffSignatures.filter(s => s.toLowerCase().includes(query));

        // Auto-update dropdown
        if (dropdown && query) {
            dropdown.innerHTML = filtered.map(s => `<option value="${s}">${s}</option>`).join('');
            const exactMatch = filtered.find(s => s.toLowerCase() === query);
            if (exactMatch) {
                dropdown.value = exactMatch;
            } else if (filtered.length === 1) {
                dropdown.value = filtered[0];
            }
        }

        // Show suggestions
        const suggestions = filtered.slice(0, 15);
        if (suggestions.length === 0) {
            div.style.display = 'none';
            return;
        }

        div.innerHTML = suggestions.map(s =>
            `<div style="padding:6px 10px;cursor:pointer;border-bottom:1px solid #eee"
                 onmouseover="this.style.background='#f0f0f0'" onmouseout="this.style.background='white'"
                 onclick="AtlasDetailPage.selectDiffSignature('${s}')">${s}</div>`
        ).join('');
        div.style.display = 'block';
    },

    selectDiffSignature(sig) {
        const input = document.getElementById('diff-signature-search');
        const dropdown = document.getElementById('diff-signature-dropdown');
        const div = document.getElementById('diff-signature-suggestions');
        if (input) input.value = sig;
        if (dropdown) dropdown.value = sig;
        if (div) div.style.display = 'none';
        this.updateDiffBoxplot();
    },

    updateDiffVolcano() {
        const container = document.getElementById('diff-volcano');
        if (!container) return;

        const selectedCancer = document.getElementById('diff-cancer-dropdown')?.value || 'all';
        const sigType = document.getElementById('diff-sig-type')?.value || 'CytoSig';

        // Get filtered data based on cancer selection
        let filtered;
        if (selectedCancer === 'all') {
            filtered = (this.diffBoxplotData?.tumor_vs_adjacent || []).filter(d => d.signature_type === sigType);
        } else {
            filtered = (this.diffBoxplotData?.by_cancer_type || []).filter(
                d => d.signature_type === sigType && d.cancer_type === selectedCancer
            );
        }

        if (!filtered || filtered.length === 0) {
            container.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">Differential data not available.</p>';
            return;
        }

        // Process data for volcano plot
        const volcanoData = filtered.map(d => ({
            signature: d.signature,
            log2fc: d.log2fc || 0,
            pvalue: d.pvalue || 1,
            negLogP: d.neg_log10_pval || -Math.log10(Math.max(d.pvalue || 1, 1e-100))
        }));

        // Separate by significance
        const fcThreshold = 0.3;
        const pThreshold = 0.05;

        const significantUp = volcanoData.filter(d => d.log2fc > fcThreshold && d.pvalue < pThreshold);
        const significantDown = volcanoData.filter(d => d.log2fc < -fcThreshold && d.pvalue < pThreshold);
        const notSig = volcanoData.filter(d => Math.abs(d.log2fc) <= fcThreshold || d.pvalue >= pThreshold);

        // Update subtitle with sample counts
        const volcanoSubtitle = document.getElementById('diff-volcano-subtitle');
        if (volcanoSubtitle) {
            const nSig = significantUp.length + significantDown.length;
            if (selectedCancer === 'all') {
                const summary = this.diffBoxplotData?.summary || {};
                volcanoSubtitle.textContent = `${nSig} significant (p<0.05) | Pan-Cancer: ${summary.n_tumor_samples || 147} tumor vs ${summary.n_adjacent_samples || 105} adjacent`;
            } else {
                const sampleEntry = filtered[0];
                const nTumor = sampleEntry?.tumor?.n || sampleEntry?.n_tumor || 'N/A';
                const nAdj = sampleEntry?.adjacent?.n || sampleEntry?.n_adjacent || 'N/A';
                volcanoSubtitle.textContent = `${nSig} significant (p<0.05) | ${selectedCancer}: ${nTumor} tumor vs ${nAdj} adjacent`;
            }
        }

        Plotly.purge(container);
        Plotly.newPlot(container, [
            {
                type: 'scatter', mode: 'markers+text', name: 'Higher in Adjacent',
                x: significantUp.map(d => d.log2fc), y: significantUp.map(d => d.negLogP),
                text: significantUp.map(d => d.signature), textposition: 'top center', textfont: { size: 9 },
                marker: { color: '#ff7f0e', size: 10 },
                hovertemplate: '<b>%{text}</b><br>log2FC: %{x:.3f}<br>-log10(p): %{y:.2f}<extra></extra>'
            },
            {
                type: 'scatter', mode: 'markers+text', name: 'Higher in Tumor',
                x: significantDown.map(d => d.log2fc), y: significantDown.map(d => d.negLogP),
                text: significantDown.map(d => d.signature), textposition: 'top center', textfont: { size: 9 },
                marker: { color: '#d62728', size: 10 },
                hovertemplate: '<b>%{text}</b><br>log2FC: %{x:.3f}<br>-log10(p): %{y:.2f}<extra></extra>'
            },
            {
                type: 'scatter', mode: 'markers', name: 'Not significant',
                x: notSig.map(d => d.log2fc), y: notSig.map(d => d.negLogP),
                text: notSig.map(d => d.signature), marker: { color: '#ccc', size: 6 },
                hovertemplate: '<b>%{text}</b><br>log2FC: %{x:.3f}<br>-log10(p): %{y:.2f}<extra></extra>'
            }
        ], {
            xaxis: { title: 'log2(Adjacent / Tumor)', zeroline: true },
            yaxis: { title: '-log10(p-value)' },
            shapes: [
                { type: 'line', x0: fcThreshold, y0: 0, x1: fcThreshold, y1: 10, line: { dash: 'dot', color: '#999' } },
                { type: 'line', x0: -fcThreshold, y0: 0, x1: -fcThreshold, y1: 10, line: { dash: 'dot', color: '#999' } },
                { type: 'line', x0: -3, y0: -Math.log10(pThreshold), x1: 3, y1: -Math.log10(pThreshold), line: { dash: 'dot', color: '#999' } }
            ],
            margin: { l: 60, r: 30, t: 30, b: 50 },
            legend: { orientation: 'h', y: -0.15 },
            height: 380,
            font: { family: 'Inter, sans-serif' }
        }, { responsive: true });
    },

    updateDiffTopBar() {
        const container = document.getElementById('diff-top-bar');
        if (!container) return;

        const selectedCancer = document.getElementById('diff-cancer-dropdown')?.value || 'all';
        const sigType = document.getElementById('diff-sig-type')?.value || 'CytoSig';

        // Get filtered data based on cancer selection
        let filtered;
        if (selectedCancer === 'all') {
            filtered = (this.diffBoxplotData?.tumor_vs_adjacent || []).filter(d => d.signature_type === sigType);
        } else {
            filtered = (this.diffBoxplotData?.by_cancer_type || []).filter(
                d => d.signature_type === sigType && d.cancer_type === selectedCancer
            );
        }

        if (!filtered || filtered.length === 0) {
            container.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">Top differential data not available.</p>';
            return;
        }

        // Sort by absolute log2FC
        const sorted = [...filtered].sort((a, b) => Math.abs(b.log2fc || 0) - Math.abs(a.log2fc || 0));
        const top15 = sorted.slice(0, 15);

        Plotly.purge(container);
        Plotly.newPlot(container, [{
            type: 'bar',
            orientation: 'h',
            y: top15.map(d => d.signature),
            x: top15.map(d => d.log2fc || 0),
            marker: {
                color: top15.map(d => (d.pvalue || 1) < 0.05 ? ((d.log2fc || 0) > 0 ? '#ff7f0e' : '#d62728') : '#ccc')
            },
            text: top15.map(d => (d.log2fc || 0).toFixed(3)),
            textposition: 'auto',
            hovertemplate: '<b>%{y}</b><br>log2FC: %{x:.3f}<br>p = %{customdata:.2e}<extra></extra>',
            customdata: top15.map(d => d.pvalue || 1)
        }], {
            xaxis: { title: 'log2(Adjacent / Tumor)', zeroline: true },
            yaxis: { automargin: true, tickfont: { size: 10 } },
            margin: { l: 100, r: 50, t: 30, b: 50 },
            height: 380,
            font: { family: 'Inter, sans-serif' }
        }, { responsive: true });

        // Add click handler to update boxplot
        container.on('plotly_click', (clickData) => {
            const sig = clickData.points[0].y;
            const input = document.getElementById('diff-signature-search');
            const dropdown = document.getElementById('diff-signature-dropdown');
            if (input) input.value = sig;
            if (dropdown) dropdown.value = sig;
            this.updateDiffBoxplot();
        });
    },

    updateDiffBoxplot() {
        const container = document.getElementById('diff-boxplot');
        if (!container) return;

        const selectedCancer = document.getElementById('diff-cancer-dropdown')?.value || 'all';
        const sigType = document.getElementById('diff-sig-type')?.value || 'CytoSig';
        const searchVal = document.getElementById('diff-signature-search')?.value?.trim();
        const selectedSig = searchVal || document.getElementById('diff-signature-dropdown')?.value || 'IFNG';

        const cancerLabel = selectedCancer === 'all' ? 'Pan-Cancer' : selectedCancer;

        // Get data based on cancer selection
        let filtered, boxplotFiltered;
        if (selectedCancer === 'all') {
            filtered = (this.diffBoxplotData?.tumor_vs_adjacent || []).filter(d => d.signature_type === sigType);
            boxplotFiltered = (this.diffBoxplotData?.boxplot_data || []).filter(d => d.signature_type === sigType);
        } else {
            filtered = (this.diffBoxplotData?.by_cancer_type || []).filter(
                d => d.signature_type === sigType && d.cancer_type === selectedCancer
            );
            boxplotFiltered = filtered; // by_cancer_type has boxplot stats
        }

        // Get boxplot data for selected signature
        const sigBoxData = boxplotFiltered.find(d => d.signature === selectedSig);
        const sigEntry = filtered.find(d => d.signature === selectedSig);

        // Update title and subtitle
        const boxTitle = document.getElementById('diff-boxplot-title');
        const boxSubtitle = document.getElementById('diff-boxplot-subtitle');

        // Colors matching reference: Red for Tumor, Orange for Adjacent, Green for Normal
        const colors = {
            Tumor: '#d62728',
            Adjacent: '#ff7f0e',
            Normal: '#2ca02c'
        };

        if (sigBoxData?.tumor && sigBoxData?.adjacent) {
            const tumorStats = sigBoxData.tumor;
            const adjStats = sigBoxData.adjacent;

            // Check for Normal data (only in Pan-Cancer view)
            let normalStats = null;
            if (selectedCancer === 'all' && this.diffOrganData) {
                const normalData = this.diffOrganData.filter(d => d.signature === selectedSig);
                if (normalData.length > 0) {
                    const values = normalData.map(d => d.mean_activity).sort((a, b) => a - b);
                    const n = values.length;
                    normalStats = {
                        min: values[0],
                        q1: values[Math.floor(n * 0.25)] || values[0],
                        median: values[Math.floor(n * 0.5)] || values[0],
                        q3: values[Math.floor(n * 0.75)] || values[n - 1],
                        max: values[n - 1],
                        mean: values.reduce((a, b) => a + b, 0) / n,
                        n: n
                    };
                }
            }

            const hasNormal = normalStats && selectedCancer === 'all';

            // Update titles
            if (boxTitle) {
                boxTitle.textContent = hasNormal
                    ? `Activity Comparison: Tumor vs Adjacent vs Normal (${cancerLabel})`
                    : `Activity Comparison: Tumor vs Adjacent Normal (${cancerLabel})`;
            }
            if (boxSubtitle) {
                boxSubtitle.textContent = hasNormal
                    ? `${selectedSig}: Tumor (n=${tumorStats.n}) | Adjacent (n=${adjStats.n}) | Normal (n=${normalStats.n} organs)`
                    : `${selectedSig}: Tumor (n=${tumorStats.n}) vs Adjacent Normal (n=${adjStats.n})`;
            }

            // Create traces
            const traces = [];

            // Tumor trace
            traces.push({
                type: 'box',
                name: 'Tumor',
                x: ['Tumor'],
                lowerfence: [tumorStats.min],
                q1: [tumorStats.q1],
                median: [tumorStats.median],
                q3: [tumorStats.q3],
                upperfence: [tumorStats.max],
                mean: [tumorStats.mean],
                boxpoints: false,
                boxmean: true,
                marker: { color: colors.Tumor },
                fillcolor: 'rgba(214, 39, 40, 0.5)',
                line: { color: colors.Tumor, width: 2 },
                hoverinfo: 'text',
                hovertext: `<b>Tumor (Cancer tissue)</b><br>Max: ${tumorStats.max.toFixed(3)}<br>Q3: ${tumorStats.q3.toFixed(3)}<br>Median: ${tumorStats.median.toFixed(3)}<br>Q1: ${tumorStats.q1.toFixed(3)}<br>Min: ${tumorStats.min.toFixed(3)}<br>n=${tumorStats.n}`
            });

            // Adjacent trace
            traces.push({
                type: 'box',
                name: 'Adjacent',
                x: ['Adjacent'],
                lowerfence: [adjStats.min],
                q1: [adjStats.q1],
                median: [adjStats.median],
                q3: [adjStats.q3],
                upperfence: [adjStats.max],
                mean: [adjStats.mean],
                boxpoints: false,
                boxmean: true,
                marker: { color: colors.Adjacent },
                fillcolor: 'rgba(255, 127, 14, 0.5)',
                line: { color: colors.Adjacent, width: 2 },
                hoverinfo: 'text',
                hovertext: `<b>Adjacent (Tumor-matched normal)</b><br>Max: ${adjStats.max.toFixed(3)}<br>Q3: ${adjStats.q3.toFixed(3)}<br>Median: ${adjStats.median.toFixed(3)}<br>Q1: ${adjStats.q1.toFixed(3)}<br>Min: ${adjStats.min.toFixed(3)}<br>n=${adjStats.n}`
            });

            // Normal trace (if available)
            if (hasNormal) {
                traces.push({
                    type: 'box',
                    name: 'Normal',
                    x: ['Normal'],
                    lowerfence: [normalStats.min],
                    q1: [normalStats.q1],
                    median: [normalStats.median],
                    q3: [normalStats.q3],
                    upperfence: [normalStats.max],
                    mean: [normalStats.mean],
                    boxpoints: false,
                    boxmean: true,
                    marker: { color: colors.Normal },
                    fillcolor: 'rgba(44, 160, 44, 0.5)',
                    line: { color: colors.Normal, width: 2 },
                    hoverinfo: 'text',
                    hovertext: `<b>Normal (scAtlas healthy tissue)</b><br>Max: ${normalStats.max.toFixed(3)}<br>Q3: ${normalStats.q3.toFixed(3)}<br>Median: ${normalStats.median.toFixed(3)}<br>Q1: ${normalStats.q1.toFixed(3)}<br>Min: ${normalStats.min.toFixed(3)}<br>n=${normalStats.n} organs`
                });
            }

            // Get p-value for annotation
            const pval = sigEntry?.pvalue || 1;
            const pvalText = pval < 0.001 ? 'p < 0.001' : `p = ${pval.toFixed(3)}`;
            const direction = sigEntry?.direction === 'up_in_tumor' ? 'Higher in Tumor' : 'Higher in Adjacent';

            Plotly.purge(container);
            Plotly.newPlot(container, traces, {
                title: { text: `${selectedSig}: Tissue Comparison (${cancerLabel})`, font: { size: 14 } },
                yaxis: { title: 'Activity (z-score)', zeroline: true },
                xaxis: { title: '' },
                annotations: [{
                    x: 0.5,
                    y: 1.08,
                    xref: 'paper',
                    yref: 'paper',
                    text: `Tumor vs Adjacent: ${pvalText} | ${direction}`,
                    showarrow: false,
                    font: { size: 11, color: pval < 0.05 ? '#d62728' : '#666' }
                }],
                margin: { l: 60, r: 30, t: 70, b: 50 },
                height: 300,
                showlegend: false,
                font: { family: 'Inter, sans-serif' }
            }, { responsive: true });

        } else {
            // Fallback - no boxplot data
            if (boxTitle) boxTitle.textContent = `Activity Comparison (${cancerLabel})`;
            if (boxSubtitle) boxSubtitle.textContent = `Data not available for ${selectedSig}`;
            container.innerHTML = `<p style="text-align:center; color:#666; padding:2rem;">No boxplot data available for ${selectedSig}.</p>`;
        }
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
            reversescale: true,
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
        const volcanoContainer = document.getElementById('inflam-volcano');
        const barContainer = document.getElementById('inflam-diff-bar');
        if (!volcanoContainer) return;

        const sigType = this.signatureType;
        const diseaseFilter = document.getElementById('inflam-diff-disease')?.value || 'all';

        // Get differential data (use raw data matching index.html format)
        const diffData = this.inflamDifferentialRaw;

        if (!diffData || diffData.length === 0) {
            volcanoContainer.innerHTML = '<p style="text-align:center;color:#666;padding:2rem;">No differential data available</p>';
            if (barContainer) barContainer.innerHTML = '<p style="text-align:center;color:#666;padding:2rem;">No differential data available</p>';
            return;
        }

        // Filter by signature type (d.signature is the type: CytoSig/SecAct)
        let filteredData = diffData.filter(d => d.signature === sigType);

        if (diseaseFilter !== 'all') {
            filteredData = filteredData.filter(d => d.disease === diseaseFilter);
        }

        // For SecAct (many proteins), limit to top 200 by significance score
        if (sigType === 'SecAct' && filteredData.length > 200) {
            filteredData = [...filteredData].sort((a, b) => {
                const scoreA = a.neg_log10_pval * Math.abs(a.log2fc);
                const scoreB = b.neg_log10_pval * Math.abs(b.log2fc);
                return scoreB - scoreA;
            }).slice(0, 200);
        }

        // Update titles with comparison type info
        const diseaseLabel = diseaseFilter === 'all' ? 'All Diseases' : diseaseFilter;
        const volcanoTitle = document.getElementById('inflam-diff-volcano-title');
        const volcanoSubtitle = document.getElementById('inflam-diff-volcano-subtitle');
        const barTitle = document.getElementById('inflam-diff-bar-title');
        const barSubtitle = document.getElementById('inflam-diff-bar-subtitle');

        // Get comparison type for this disease
        let comparisonNote = '';
        if (diseaseFilter !== 'all' && filteredData.length > 0) {
            const compType = filteredData[0].comparison;
            const nHealthy = filteredData[0].n_g2;
            if (compType === 'study_matched') {
                comparisonNote = ` (study-matched healthy, n=${nHealthy})`;
            } else {
                comparisonNote = ` (pooled healthy - no matched controls, n=${nHealthy})`;
            }
        }

        if (volcanoTitle) volcanoTitle.textContent = `Volcano Plot: ${diseaseLabel} vs Healthy`;
        if (volcanoSubtitle) {
            const noteColor = comparisonNote.includes('pooled') ? '#d62728' : '#666';
            volcanoSubtitle.innerHTML = `Positive = higher in ${diseaseLabel}, Negative = higher in Healthy<br><span style="font-size: 0.85em; color: ${noteColor};">${comparisonNote}</span>`;
        }
        if (barTitle) barTitle.textContent = `Top Differential Signatures`;
        if (barSubtitle) barSubtitle.textContent = `Sorted by significance (|effect| × -log10 p)`;

        // Handle empty data
        if (filteredData.length === 0) {
            volcanoContainer.innerHTML = `<p style="text-align:center;color:#666;padding:2rem;">No data available for ${diseaseLabel} (${sigType})</p>`;
            if (barContainer) barContainer.innerHTML = '<p style="text-align:center;color:#666;padding:2rem;">No data available</p>';
            return;
        }

        // 1. Volcano plot
        Plotly.purge(volcanoContainer);

        // Color by significance (matching index.html style)
        const colors = filteredData.map(d => {
            if (d.qvalue < 0.05 && Math.abs(d.log2fc) > 0.5) {
                return d.log2fc > 0 ? '#f4a6a6' : '#a8d4e6';
            }
            return '#cccccc';
        });

        // Only show text for significant points
        const textLabels = filteredData.map(d => {
            if (d.qvalue < 0.05 && Math.abs(d.log2fc) > 0.5) {
                return d.protein;  // Use protein field for signature name
            }
            return '';
        });

        // Dynamic x-axis range based on data
        const maxAbsFC = Math.max(3, Math.ceil(Math.max(...filteredData.map(d => Math.abs(d.log2fc)))));
        const maxY = Math.max(4, Math.ceil(Math.max(...filteredData.map(d => d.neg_log10_pval))));

        Plotly.newPlot(volcanoContainer, [{
            x: filteredData.map(d => d.log2fc),
            y: filteredData.map(d => d.neg_log10_pval),
            text: textLabels,
            customdata: filteredData.map(d => d.protein),  // Use protein for hover
            mode: 'markers+text',
            type: 'scatter',
            marker: {
                color: colors,
                size: 10,
                opacity: 0.7
            },
            textposition: 'top center',
            textfont: { size: 10 },
            hovertemplate: '<b>%{customdata}</b><br>Effect: %{x:.2f}<br>-log10(p): %{y:.2f}<extra></extra>'
        }], {
            xaxis: {
                title: `Activity Difference (${diseaseLabel} - Healthy)`,
                zeroline: true,
                zerolinecolor: '#ccc',
                range: [-maxAbsFC, maxAbsFC]
            },
            yaxis: {
                title: '-log10(p-value)',
                range: [0, maxY * 1.1]
            },
            shapes: [
                // Horizontal line at p=0.05
                { type: 'line', x0: -maxAbsFC, x1: maxAbsFC, y0: -Math.log10(0.05), y1: -Math.log10(0.05), line: { color: '#999', dash: 'dash', width: 1 } },
                // Vertical lines at effect thresholds
                { type: 'line', x0: -0.5, x1: -0.5, y0: 0, y1: maxY * 1.1, line: { color: '#999', dash: 'dash', width: 1 } },
                { type: 'line', x0: 0.5, x1: 0.5, y0: 0, y1: maxY * 1.1, line: { color: '#999', dash: 'dash', width: 1 } }
            ],
            margin: { l: 60, r: 30, t: 30, b: 60 },
            height: 450,
            font: { family: 'Inter, sans-serif' },
            annotations: [{
                x: -maxAbsFC * 0.8,
                y: -0.08,
                xref: 'x',
                yref: 'paper',
                text: `← Higher in Healthy`,
                showarrow: false,
                font: { size: 11, color: '#a8d4e6' }
            }, {
                x: maxAbsFC * 0.8,
                y: -0.08,
                xref: 'x',
                yref: 'paper',
                text: `Higher in ${diseaseLabel} →`,
                showarrow: false,
                font: { size: 11, color: '#f4a6a6' }
            }]
        }, { responsive: true });

        // 2. Top differential bar chart - sorted by significance score
        if (barContainer) {
            Plotly.purge(barContainer);

            // Calculate significance score and sort
            const scoredData = filteredData.map(d => ({
                ...d,
                score: Math.abs(d.log2fc) * d.neg_log10_pval
            }));

            // Sort by score descending and take top 20
            const sorted = [...scoredData].sort((a, b) => b.score - a.score);
            const top20 = sorted.slice(0, 20).reverse();  // Reverse for horizontal bar (top at top)

            Plotly.newPlot(barContainer, [{
                type: 'bar',
                orientation: 'h',
                y: top20.map(d => d.protein),  // Use protein field for signature name
                x: top20.map(d => d.log2fc),
                marker: {
                    color: top20.map(d => d.log2fc > 0 ? '#f4a6a6' : '#a8d4e6')
                },
                text: top20.map(d => d.log2fc.toFixed(2)),
                textposition: 'outside',
                textfont: { size: 9 },
                hovertemplate: '<b>%{y}</b><br>Effect: %{x:.3f}<br>q = %{customdata}<extra></extra>',
                customdata: top20.map(d => d.qvalue?.toExponential(2) || 'N/A')
            }], {
                xaxis: { title: 'Activity Difference', zeroline: true, zerolinecolor: '#ccc' },
                yaxis: { automargin: true, tickfont: { size: 10 } },
                margin: { l: 120, r: 50, t: 30, b: 50 },
                height: 500,
                font: { family: 'Inter, sans-serif' }
            }, { responsive: true });
        }
    },

    async updateTreatmentResponse() {
        const container1 = document.getElementById('treatment-roc');
        const container2 = document.getElementById('treatment-importance');
        const container3 = document.getElementById('treatment-violin');

        const data = this.treatmentResponseRaw;

        // Get filter values (use global signature type)
        const selectedDisease = document.getElementById('treatment-disease')?.value || 'all';
        const selectedSigType = this.signatureType;
        const selectedModel = document.getElementById('treatment-model')?.value || 'all';

        if (!data || (!data.roc_curves?.length && !data.feature_importance?.length)) {
            [container1, container2, container3].forEach(c => {
                if (c) {
                    c.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">Treatment response data will be available after running the analysis pipeline.</p>';
                }
            });
            return;
        }

        // 1. ROC Curves visualization (actual ROC curves)
        if (container1) {
            Plotly.purge(container1);

            let rocData = data.roc_curves || [];
            // Filter by signature type
            rocData = rocData.filter(d => d.signature_type === selectedSigType);
            // Handle "All Diseases" - maps to "All Diseases" in data
            if (selectedDisease !== 'all') {
                rocData = rocData.filter(d => d.disease === selectedDisease);
            } else {
                // Show "All Diseases" entry when all is selected
                rocData = rocData.filter(d => d.disease === 'All Diseases');
            }
            if (selectedModel !== 'all') {
                rocData = rocData.filter(d => d.model === selectedModel);
            }

            if (rocData.length === 0) {
                container1.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">No ROC data for selected filters</p>';
            } else {
                // Color palette for different curves
                const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'];

                // Create traces for each ROC curve
                const traces = rocData.map((d, i) => ({
                    type: 'scatter',
                    mode: 'lines',
                    name: `${d.disease} - ${d.model} (AUC=${d.auc.toFixed(2)})`,
                    x: d.fpr || [0, 1],
                    y: d.tpr || [0, 1],
                    line: { color: colors[i % colors.length], width: 2 },
                    hovertemplate: `<b>${d.disease}</b><br>${d.model}<br>FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>`
                }));

                // Add diagonal reference line
                traces.push({
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Random (AUC=0.50)',
                    x: [0, 1],
                    y: [0, 1],
                    line: { color: '#ccc', width: 1, dash: 'dash' },
                    hoverinfo: 'skip'
                });

                Plotly.newPlot(container1, traces, {
                    xaxis: { title: 'False Positive Rate', range: [0, 1] },
                    yaxis: { title: 'True Positive Rate', range: [0, 1] },
                    margin: { l: 60, r: 30, t: 30, b: 50 },
                    legend: { orientation: 'h', y: -0.25, font: { size: 10 } },
                    height: 400,
                    font: { family: 'Inter, sans-serif' }
                }, { responsive: true });
            }
        }

        // 2. Feature Importance visualization
        if (container2) {
            Plotly.purge(container2);

            let impData = data.feature_importance || [];
            // Filter by signature type
            impData = impData.filter(d => d.signature_type === selectedSigType);
            // Filter by model
            if (selectedModel !== 'all') {
                impData = impData.filter(d => d.model === selectedModel);
            }
            // Handle disease filter
            if (selectedDisease !== 'all') {
                impData = impData.filter(d => d.disease === selectedDisease);
            } else {
                impData = impData.filter(d => d.disease === 'All Diseases');
            }

            if (impData.length === 0) {
                container2.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">No feature importance data for selected filters</p>';
            } else {
                const isLR = selectedModel === 'Logistic Regression';

                // Aggregate importance across records
                const featureMap = {};
                impData.forEach(d => {
                    if (!featureMap[d.feature]) {
                        featureMap[d.feature] = { importance: [], coefficient: [] };
                    }
                    featureMap[d.feature].importance.push(d.importance);
                    if (d.coefficient !== undefined) {
                        featureMap[d.feature].coefficient.push(d.coefficient);
                    }
                });

                const aggregated = Object.entries(featureMap)
                    .map(([feature, vals]) => ({
                        feature,
                        importance: vals.importance.reduce((a, b) => a + b, 0) / vals.importance.length,
                        coefficient: vals.coefficient.length > 0
                            ? vals.coefficient.reduce((a, b) => a + b, 0) / vals.coefficient.length
                            : null
                    }))
                    .sort((a, b) => b.importance - a.importance)
                    .slice(0, 15);

                // For LR, color by coefficient direction (green = positive, red = negative)
                const colors = isLR
                    ? aggregated.map(d => d.coefficient !== null
                        ? (d.coefficient >= 0 ? '#2ca02c' : '#d62728')
                        : '#1f77b4')
                    : aggregated.map(() => '#2ca02c');

                const hoverText = isLR
                    ? aggregated.map(d => d.coefficient !== null
                        ? `${d.feature}<br>Importance: ${d.importance.toFixed(3)}<br>Coef: ${d.coefficient.toFixed(3)} (${d.coefficient >= 0 ? '+' : '-'})`
                        : `${d.feature}<br>Importance: ${d.importance.toFixed(3)}`)
                    : aggregated.map(d => `${d.feature}<br>Importance: ${d.importance.toFixed(3)}`);

                Plotly.newPlot(container2, [{
                    type: 'bar',
                    orientation: 'h',
                    y: aggregated.map(d => d.feature),
                    x: aggregated.map(d => d.importance),
                    marker: { color: colors },
                    text: aggregated.map(d => d.importance.toFixed(3)),
                    textposition: 'auto',
                    hovertext: hoverText,
                    hoverinfo: 'text'
                }], {
                    xaxis: { title: isLR ? 'Normalized |Coefficient|' : 'Importance Score' },
                    yaxis: { automargin: true },
                    margin: { l: 80, r: 30, t: 30, b: 50 },
                    height: 400,
                    font: { family: 'Inter, sans-serif' },
                    annotations: isLR ? [{
                        x: 0.95, y: 1.05, xref: 'paper', yref: 'paper',
                        text: '<span style="color:#2ca02c">■</span> Positive  <span style="color:#d62728">■</span> Negative',
                        showarrow: false, font: { size: 11 }
                    }] : []
                }, { responsive: true });
            }
        }

        // 3. Prediction Violin/Box plots
        if (container3) {
            Plotly.purge(container3);

            let predData = data.predictions || [];
            // Filter by signature type
            predData = predData.filter(d => d.signature_type === selectedSigType);
            // Handle disease filter
            if (selectedDisease !== 'all') {
                predData = predData.filter(d => d.disease === selectedDisease);
            } else {
                predData = predData.filter(d => d.disease === 'All Diseases');
            }

            if (predData.length === 0) {
                container3.innerHTML = '<p style="text-align:center; color:#666; padding:2rem;">No prediction data for selected filters</p>';
            } else {
                const responders = predData.filter(d => d.response === 'Responder').map(d => d.probability);
                const nonResponders = predData.filter(d => d.response === 'Non-responder').map(d => d.probability);

                Plotly.newPlot(container3, [
                    {
                        type: 'violin',
                        y: responders,
                        name: 'Responder',
                        box: { visible: true },
                        meanline: { visible: true },
                        fillcolor: '#2ca02c',
                        line: { color: '#2ca02c' }
                    },
                    {
                        type: 'violin',
                        y: nonResponders,
                        name: 'Non-responder',
                        box: { visible: true },
                        meanline: { visible: true },
                        fillcolor: '#d62728',
                        line: { color: '#d62728' }
                    }
                ], {
                    yaxis: { title: 'Predicted Probability', range: [0, 1] },
                    margin: { l: 50, r: 30, t: 30, b: 50 },
                    showlegend: true,
                    height: 350,
                    font: { family: 'Inter, sans-serif' }
                }, { responsive: true });
            }
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
