/**
 * Atlas Detail Page Handler
 * Shows atlas analysis panels with tabs
 */

const AtlasDetailPage = {
    currentAtlas: null,
    currentTab: null,
    signatureType: 'CytoSig',

    /**
     * Atlas configurations with available analysis tabs
     */
    atlasConfigs: {
        cima: {
            displayName: 'CIMA',
            description: 'Chinese Immune Multi-omics Atlas - 6.5M cells from 421 healthy adults',
            tabs: [
                { id: 'activity', label: 'Activity Heatmap', icon: '&#128202;' },
                { id: 'age-bmi', label: 'Age/BMI Stratified', icon: '&#128200;' },
                { id: 'biochem', label: 'Biochemistry', icon: '&#129514;' },
                { id: 'metabolites', label: 'Metabolites', icon: '&#9879;' },
                { id: 'differential', label: 'Differential', icon: '&#128209;' },
            ],
        },
        inflammation: {
            displayName: 'Inflammation Atlas',
            description: 'Pan-disease immune profiling - 4.9M cells across 12+ diseases',
            tabs: [
                { id: 'activity', label: 'Activity Heatmap', icon: '&#128202;' },
                { id: 'disease', label: 'Disease Differential', icon: '&#128209;' },
                { id: 'age-bmi', label: 'Age/BMI Stratified', icon: '&#128200;' },
                { id: 'treatment', label: 'Treatment Prediction', icon: '&#128137;' },
            ],
        },
        scatlas: {
            displayName: 'scAtlas',
            description: 'Human tissue reference atlas - 6.4M cells across 30 organs',
            tabs: [
                { id: 'activity', label: 'Activity Heatmap', icon: '&#128202;' },
                { id: 'organs', label: 'Organ Signatures', icon: '&#128149;' },
                { id: 'celltypes', label: 'Cell Type Signatures', icon: '&#128300;' },
            ],
        },
    },

    /**
     * Initialize the atlas detail page
     */
    async init(params) {
        this.currentAtlas = params.name;
        this.currentTab = 'activity';

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
            tabs: [{ id: 'activity', label: 'Activity Heatmap', icon: '&#128202;' }],
        };

        // Render header
        this.renderHeader(config);

        // Render tabs
        this.renderTabs(config.tabs);

        // Load default tab content
        await this.loadTabContent('activity');
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
            switch (tabId) {
                case 'activity':
                    await this.loadActivityHeatmap(content);
                    break;
                case 'age-bmi':
                    await this.loadAgeBmiStratified(content);
                    break;
                case 'biochem':
                    await this.loadBiochemistry(content);
                    break;
                case 'metabolites':
                    await this.loadMetabolites(content);
                    break;
                case 'differential':
                    await this.loadDifferential(content);
                    break;
                case 'disease':
                    await this.loadDiseaseDifferential(content);
                    break;
                case 'treatment':
                    await this.loadTreatmentPrediction(content);
                    break;
                case 'organs':
                    await this.loadOrganSignatures(content);
                    break;
                case 'celltypes':
                    await this.loadCellTypeSignatures(content);
                    break;
                default:
                    content.innerHTML = '<p>Content not available</p>';
            }
        } catch (error) {
            console.error('Failed to load tab content:', error);
            content.innerHTML = `<p class="loading">Failed to load data: ${error.message}</p>`;
        }
    },

    /**
     * Load activity heatmap
     */
    async loadActivityHeatmap(container) {
        container.innerHTML = `
            <div class="panel-header">
                <h3>Cytokine Activity Heatmap</h3>
                <p>Mean activity z-scores across cell types</p>
            </div>
            <div id="activity-heatmap" class="plot-container"></div>
        `;

        const data = await API.getAtlasActivity(this.currentAtlas, {
            signature_type: this.signatureType,
        });

        if (data && data.z) {
            Heatmap.createActivityHeatmap('activity-heatmap', data, {
                title: `${this.signatureType} Activity`,
            });
        } else {
            document.getElementById('activity-heatmap').innerHTML = '<p class="loading">No activity data available</p>';
        }
    },

    /**
     * Load age/BMI stratified analysis
     */
    async loadAgeBmiStratified(container) {
        container.innerHTML = `
            <div class="panel-header">
                <h3>Age & BMI Stratified Activity</h3>
                <p>Activity patterns across age groups and BMI categories</p>
            </div>
            <div class="stratified-controls">
                <select id="stratified-variable" class="filter-select" onchange="AtlasDetailPage.updateStratifiedPlot()">
                    <option value="age">Age Groups</option>
                    <option value="bmi">BMI Categories</option>
                </select>
                <select id="stratified-celltype" class="filter-select">
                    <option value="">All Cell Types</option>
                </select>
                <select id="stratified-signature" class="filter-select">
                    <option value="">Select Signature</option>
                </select>
            </div>
            <div id="stratified-plot" class="plot-container"></div>
        `;

        // Load cell types and signatures for dropdowns
        try {
            const cellTypes = await API.getAtlasCellTypes(this.currentAtlas);
            const signatures = await API.getAtlasSignatures(this.currentAtlas, this.signatureType);

            const cellTypeSelect = document.getElementById('stratified-celltype');
            const sigSelect = document.getElementById('stratified-signature');

            if (cellTypes && cellTypeSelect) {
                cellTypes.forEach(ct => {
                    cellTypeSelect.innerHTML += `<option value="${ct}">${ct}</option>`;
                });
            }

            if (signatures && sigSelect) {
                signatures.forEach(sig => {
                    sigSelect.innerHTML += `<option value="${sig}">${sig}</option>`;
                });
            }
        } catch (e) {
            console.warn('Failed to load filter options:', e);
        }

        // Load initial data
        await this.updateStratifiedPlot();
    },

    /**
     * Update stratified plot based on selections
     */
    async updateStratifiedPlot() {
        const variable = document.getElementById('stratified-variable')?.value || 'age';
        const cellType = document.getElementById('stratified-celltype')?.value;
        const signature = document.getElementById('stratified-signature')?.value;

        const plotContainer = document.getElementById('stratified-plot');
        if (!plotContainer) return;

        plotContainer.innerHTML = '<div class="loading"><div class="spinner"></div>Loading...</div>';

        try {
            let data;
            if (this.currentAtlas === 'cima') {
                data = await API.getCimaAgeBmiStratified({
                    variable,
                    cell_type: cellType,
                    signature,
                    signature_type: this.signatureType,
                });
            } else if (this.currentAtlas === 'inflammation') {
                data = await API.getInflammationAgeBmiStratified({
                    variable,
                    cell_type: cellType,
                    signature,
                    signature_type: this.signatureType,
                });
            }

            if (data && data.groups && data.values) {
                Scatter.createBoxPlot('stratified-plot', data, {
                    title: variable === 'age' ? 'Activity by Age Group' : 'Activity by BMI Category',
                    yLabel: 'Activity (z-score)',
                    showPoints: true,
                });
            } else {
                plotContainer.innerHTML = '<p class="loading">No stratified data available</p>';
            }
        } catch (error) {
            plotContainer.innerHTML = `<p class="loading">Error loading data: ${error.message}</p>`;
        }
    },

    /**
     * Load biochemistry correlations (CIMA)
     */
    async loadBiochemistry(container) {
        container.innerHTML = `
            <div class="panel-header">
                <h3>Biochemistry Correlations</h3>
                <p>Correlation between cytokine activity and blood biochemistry markers</p>
            </div>
            <div id="biochem-heatmap" class="plot-container"></div>
        `;

        try {
            const data = await API.getCimaCorrelations('biochemistry', {
                signature_type: this.signatureType,
            });

            if (data && data.z) {
                Heatmap.createCorrelationHeatmap('biochem-heatmap', data, {
                    title: 'Biochemistry Correlations',
                    xLabel: 'Biochemistry Marker',
                    yLabel: 'Signature',
                });
            } else {
                document.getElementById('biochem-heatmap').innerHTML = '<p class="loading">No biochemistry data available</p>';
            }
        } catch (error) {
            document.getElementById('biochem-heatmap').innerHTML = '<p class="loading">Failed to load biochemistry data</p>';
        }
    },

    /**
     * Load metabolite correlations (CIMA)
     */
    async loadMetabolites(container) {
        container.innerHTML = `
            <div class="panel-header">
                <h3>Metabolite Correlations</h3>
                <p>Top correlations with plasma metabolites and lipids</p>
            </div>
            <div id="metabolite-heatmap" class="plot-container"></div>
        `;

        try {
            const data = await API.getCimaMetabolites({
                signature_type: this.signatureType,
                top_n: 50,
            });

            if (data && data.z) {
                Heatmap.createCorrelationHeatmap('metabolite-heatmap', data, {
                    title: 'Metabolite Correlations',
                });
            } else {
                document.getElementById('metabolite-heatmap').innerHTML = '<p class="loading">No metabolite data available</p>';
            }
        } catch (error) {
            document.getElementById('metabolite-heatmap').innerHTML = '<p class="loading">Failed to load metabolite data</p>';
        }
    },

    /**
     * Load differential analysis (CIMA)
     */
    async loadDifferential(container) {
        container.innerHTML = `
            <div class="panel-header">
                <h3>Differential Analysis</h3>
                <p>Activity differences by sex, smoking status, and blood type</p>
            </div>
            <div class="differential-controls">
                <select id="diff-variable" class="filter-select" onchange="AtlasDetailPage.updateDifferentialPlot()">
                    <option value="sex">Sex</option>
                    <option value="smoking">Smoking Status</option>
                    <option value="blood_type">Blood Type</option>
                </select>
            </div>
            <div id="differential-heatmap" class="plot-container"></div>
        `;

        await this.updateDifferentialPlot();
    },

    async updateDifferentialPlot() {
        const variable = document.getElementById('diff-variable')?.value || 'sex';

        try {
            const data = await API.getCimaDifferential({
                variable,
                signature_type: this.signatureType,
            });

            if (data && data.z) {
                Heatmap.create('differential-heatmap', data, {
                    title: `Differential by ${variable}`,
                    colorbarTitle: 'Effect Size',
                    symmetric: true,
                });
            } else {
                document.getElementById('differential-heatmap').innerHTML = '<p class="loading">No differential data available</p>';
            }
        } catch (error) {
            document.getElementById('differential-heatmap').innerHTML = '<p class="loading">Failed to load differential data</p>';
        }
    },

    /**
     * Load disease differential (Inflammation)
     */
    async loadDiseaseDifferential(container) {
        container.innerHTML = `
            <div class="panel-header">
                <h3>Disease Differential</h3>
                <p>Activity differences in disease vs healthy</p>
            </div>
            <div id="disease-heatmap" class="plot-container"></div>
        `;

        try {
            const data = await API.getInflammationDifferential({
                signature_type: this.signatureType,
            });

            if (data && data.z) {
                Heatmap.create('disease-heatmap', data, {
                    title: 'Disease vs Healthy',
                    colorbarTitle: 'Log2 Fold Change',
                    symmetric: true,
                });
            } else {
                document.getElementById('disease-heatmap').innerHTML = '<p class="loading">No disease differential data available</p>';
            }
        } catch (error) {
            document.getElementById('disease-heatmap').innerHTML = '<p class="loading">Failed to load disease data</p>';
        }
    },

    /**
     * Load treatment prediction (Inflammation)
     */
    async loadTreatmentPrediction(container) {
        container.innerHTML = `
            <div class="panel-header">
                <h3>Treatment Response Prediction</h3>
                <p>Predictive performance for treatment response</p>
            </div>
            <div id="treatment-plot" class="plot-container"></div>
        `;

        try {
            const data = await API.getInflammationPrediction({
                signature_type: this.signatureType,
            });

            if (data) {
                // Render as bar chart of AUC values
                const diseases = Object.keys(data);
                const aucs = diseases.map(d => data[d]?.auc || 0);

                Plotly.newPlot('treatment-plot', [{
                    x: diseases,
                    y: aucs,
                    type: 'bar',
                    marker: { color: '#2563eb' },
                }], {
                    title: 'Treatment Response Prediction (AUC)',
                    xaxis: { title: 'Disease' },
                    yaxis: { title: 'AUC', range: [0, 1] },
                    margin: { l: 60, r: 30, t: 50, b: 100 },
                });
            } else {
                document.getElementById('treatment-plot').innerHTML = '<p class="loading">No treatment prediction data available</p>';
            }
        } catch (error) {
            document.getElementById('treatment-plot').innerHTML = '<p class="loading">Failed to load treatment data</p>';
        }
    },

    /**
     * Load organ signatures (scAtlas)
     */
    async loadOrganSignatures(container) {
        container.innerHTML = `
            <div class="panel-header">
                <h3>Organ-Specific Signatures</h3>
                <p>Top cytokine signatures by organ</p>
            </div>
            <div id="organ-heatmap" class="plot-container"></div>
        `;

        try {
            const data = await API.getScatlasActivity({
                signature_type: this.signatureType,
                group_by: 'organ',
            });

            if (data && data.z) {
                Heatmap.createActivityHeatmap('organ-heatmap', data, {
                    title: 'Organ Signatures',
                    yLabel: 'Organ',
                });
            } else {
                document.getElementById('organ-heatmap').innerHTML = '<p class="loading">No organ data available</p>';
            }
        } catch (error) {
            document.getElementById('organ-heatmap').innerHTML = '<p class="loading">Failed to load organ data</p>';
        }
    },

    /**
     * Load cell type signatures (scAtlas)
     */
    async loadCellTypeSignatures(container) {
        container.innerHTML = `
            <div class="panel-header">
                <h3>Cell Type Signatures</h3>
                <p>Activity patterns across cell types</p>
            </div>
            <div id="celltype-heatmap" class="plot-container"></div>
        `;

        try {
            const data = await API.getScatlasActivity({
                signature_type: this.signatureType,
                group_by: 'cell_type',
                top_n: 50,
            });

            if (data && data.z) {
                Heatmap.createActivityHeatmap('celltype-heatmap', data, {
                    title: 'Cell Type Signatures',
                });
            } else {
                document.getElementById('celltype-heatmap').innerHTML = '<p class="loading">No cell type data available</p>';
            }
        } catch (error) {
            document.getElementById('celltype-heatmap').innerHTML = '<p class="loading">Failed to load cell type data</p>';
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
