/**
 * Atlas Detail Page - Example Implementation
 * Demonstrates how to use the new chart components, TabPanel, FilterBar, etc.
 * This is a REFERENCE IMPLEMENTATION - not production code
 */

const AtlasDetailPageExample = {
    currentAtlas: null,
    tabPanel: null,
    filterBar: null,

    /**
     * Initialize the page
     */
    async init(params) {
        this.currentAtlas = params.name; // 'cima', 'inflammation', or 'scatlas'

        // Render page structure
        this.renderPage();

        // Set up filter bar
        this.setupFilterBar();

        // Set up tab panel
        this.setupTabPanel();

        // Initialize tab panel
        this.tabPanel.init();
    },

    /**
     * Render page structure
     */
    renderPage() {
        const app = document.getElementById('app');
        app.innerHTML = `
            <div class="page atlas-detail-page">
                <div class="page-header">
                    <h1>${this.getAtlasDisplayName()}</h1>
                    <p>${this.getAtlasDescription()}</p>
                </div>

                <!-- Filter Bar -->
                <div id="atlas-filter-bar"></div>

                <!-- Tab Panel -->
                <div id="atlas-tabs"></div>
            </div>
        `;
    },

    /**
     * Set up filter bar
     */
    setupFilterBar() {
        this.filterBar = new FilterBar('atlas-filter-bar', { syncState: true });

        // Add signature type toggle
        this.filterBar.addToggle('signatureType', [
            { value: 'CytoSig', label: 'CytoSig (43 cytokines)' },
            { value: 'SecAct', label: 'SecAct (1,170 proteins)' }
        ], 'CytoSig');

        // Add cell type dropdown (will be populated after loading data)
        this.filterBar.addDropdown('cellType', 'Cell Type', ['All'], 'All');

        // Listen for filter changes
        this.filterBar.onChange((filterId, value) => {
            console.log(`Filter ${filterId} changed to ${value}`);

            // Reload active tab when filters change
            if (this.tabPanel) {
                const activeTab = this.tabPanel.getActiveTab();
                this.tabPanel.reloadTab(activeTab);
            }
        });
    },

    /**
     * Set up tab panel with lazy-loaded tabs
     */
    setupTabPanel() {
        this.tabPanel = new TabPanel('atlas-tabs');

        if (this.currentAtlas === 'cima') {
            this.setupCimaTabs();
        } else if (this.currentAtlas === 'inflammation') {
            this.setupInflammationTabs();
        } else if (this.currentAtlas === 'scatlas') {
            this.setupScatlasTabs();
        }
    },

    /**
     * Set up CIMA tabs (example)
     */
    setupCimaTabs() {
        // Tab 1: Cell Types (Heatmap)
        this.tabPanel.addTab('celltypes', 'Cell Types', async () => {
            const signatureType = this.filterBar.getValue('signatureType');

            // Load data
            const data = await dataLoader.load('/atlases/cima/activity', {
                signature_type: signatureType
            });

            // Create container HTML
            const html = `
                <div class="tab-content-section">
                    <div class="section-header">
                        <h2>Cytokine Activity by Cell Type</h2>
                        <div id="export-celltypes"></div>
                    </div>
                    <div id="heatmap-celltypes" class="chart-container"></div>
                </div>
            `;

            // Return HTML (will be rendered before chart)
            setTimeout(() => {
                // Render heatmap
                const chart = new HeatmapChart('heatmap-celltypes', {
                    title: '',
                    xLabel: 'Signature',
                    yLabel: 'Cell Type',
                    colorbarTitle: 'Activity (z-score)',
                    symmetric: true
                });
                chart.render(data);

                // Add export button
                const exportBtn = new ExportButton('export-celltypes', {
                    formats: ['CSV', 'PNG']
                });
                exportBtn.setChart(chart);
            }, 0);

            return html;
        }, { icon: '📊' });

        // Tab 2: Age & BMI Correlations (Heatmap + Scatter)
        this.tabPanel.addTab('age-bmi', 'Age & BMI', async () => {
            const signatureType = this.filterBar.getValue('signatureType');

            // Load both datasets in parallel
            const [ageData, bmiData] = await Promise.all([
                dataLoader.load('/cima/correlations/age', { signature_type: signatureType }),
                dataLoader.load('/cima/correlations/bmi', { signature_type: signatureType })
            ]);

            const html = `
                <div class="tab-content-section">
                    <h2>Age Correlations</h2>
                    <div id="heatmap-age" class="chart-container"></div>
                </div>
                <div class="tab-content-section">
                    <h2>BMI Correlations</h2>
                    <div id="heatmap-bmi" class="chart-container"></div>
                </div>
            `;

            setTimeout(() => {
                // Age heatmap
                const ageChart = new HeatmapChart('heatmap-age', {
                    xLabel: 'Signature',
                    yLabel: 'Cell Type',
                    colorbarTitle: 'Spearman ρ',
                    symmetric: true
                });
                ageChart.render(ageData);

                // BMI heatmap
                const bmiChart = new HeatmapChart('heatmap-bmi', {
                    xLabel: 'Signature',
                    yLabel: 'Cell Type',
                    colorbarTitle: 'Spearman ρ',
                    symmetric: true
                });
                bmiChart.render(bmiData);
            }, 0);

            return html;
        }, { icon: '📈' });

        // Tab 3: Boxplots (Age/BMI stratified)
        this.tabPanel.addTab('boxplots', 'Boxplots', async () => {
            const signatureType = this.filterBar.getValue('signatureType');

            // For demo, pick a signature (in real implementation, add signature selector)
            const signature = 'IFNG';

            const data = await dataLoader.load(`/cima/boxplots/age/${signature}`, {
                signature_type: signatureType
            });

            const html = `
                <div class="tab-content-section">
                    <h2>Activity by Age Group - ${signature}</h2>
                    <div id="boxplot-age" class="chart-container"></div>
                </div>
            `;

            setTimeout(() => {
                const chart = new BoxplotChart('boxplot-age', {
                    showPoints: true,
                    yLabel: 'Activity (z-score)'
                });
                chart.render(data);
            }, 0);

            return html;
        }, { icon: '📦' });

        // Tab 4: Differential Analysis (Volcano + Bar)
        this.tabPanel.addTab('differential', 'Differential', async () => {
            const signatureType = this.filterBar.getValue('signatureType');

            const data = await dataLoader.load('/cima/differential', {
                signature_type: signatureType,
                comparison: 'male_vs_female' // Example comparison
            });

            const html = `
                <div class="tab-content-section">
                    <h2>Differential Activity - Male vs Female</h2>
                    <div id="volcano-diff" class="chart-container"></div>
                </div>
                <div class="tab-content-section">
                    <h2>Top Differential Signatures</h2>
                    <div id="bar-top-diff" class="chart-container"></div>
                </div>
            `;

            setTimeout(() => {
                // Volcano plot
                const volcanoChart = new VolcanoChart('volcano-diff', {
                    fdrThreshold: 0.05,
                    activityThreshold: 0.5
                });
                volcanoChart.render(data);

                // Bar chart of top signatures
                const topSigs = data.points
                    .filter(p => p.fdr < 0.05 && Math.abs(p.activity_diff) > 0.5)
                    .sort((a, b) => Math.abs(b.activity_diff) - Math.abs(a.activity_diff))
                    .slice(0, 20);

                const barData = {
                    categories: topSigs.map(p => p.signature),
                    values: topSigs.map(p => p.activity_diff),
                    colors: topSigs.map(p => p.activity_diff > 0 ? '#dc2626' : '#2563eb')
                };

                const barChart = new BarChart('bar-top-diff', {
                    orientation: 'h',
                    yLabel: 'Δ Activity'
                });
                barChart.render(barData);
            }, 0);

            return html;
        }, { icon: '🌋' });

        // Tab 5: Biochemistry Correlations
        this.tabPanel.addTab('biochemistry', 'Biochemistry', async () => {
            const signatureType = this.filterBar.getValue('signatureType');

            const data = await dataLoader.load('/cima/correlations/biochemistry', {
                signature_type: signatureType
            });

            const html = `
                <div class="tab-content-section">
                    <h2>Biochemistry Correlations</h2>
                    <p>Correlations between cytokine activity and blood biochemistry markers</p>
                    <div id="heatmap-biochem" class="chart-container"></div>
                </div>
            `;

            setTimeout(() => {
                const chart = new HeatmapChart('heatmap-biochem', {
                    xLabel: 'Signature',
                    yLabel: 'Biochemistry Marker',
                    colorbarTitle: 'Spearman ρ',
                    symmetric: true
                });
                chart.render(data);
            }, 0);

            return html;
        }, { icon: '🧪' });

        // Add more tabs as needed...
    },

    /**
     * Set up Inflammation tabs (placeholder)
     */
    setupInflammationTabs() {
        this.tabPanel.addTab('celltypes', 'Cell Types', async () => {
            return '<p>Inflammation Cell Types - To be implemented</p>';
        }, { icon: '📊' });
        // ... add more tabs
    },

    /**
     * Set up scAtlas tabs (placeholder)
     */
    setupScatlasTabs() {
        this.tabPanel.addTab('tissue', 'Tissue Atlas', async () => {
            return '<p>Tissue Atlas - To be implemented</p>';
        }, { icon: '🫀' });
        // ... add more tabs
    },

    /**
     * Helper methods
     */
    getAtlasDisplayName() {
        const names = {
            cima: 'CIMA - Chinese Immune Multi-omics Atlas',
            inflammation: 'Inflammation Atlas',
            scatlas: 'scAtlas - Human Tissue Reference'
        };
        return names[this.currentAtlas] || this.currentAtlas;
    },

    getAtlasDescription() {
        const descriptions = {
            cima: '6.5M cells from 421 healthy adults with matched biochemistry and metabolomics',
            inflammation: '4.9M cells across 20 inflammatory diseases with treatment response data',
            scatlas: '6.4M cells across 35 organs with pan-cancer immune profiling'
        };
        return descriptions[this.currentAtlas] || '';
    }
};

// Make available globally (for reference only)
window.AtlasDetailPageExample = AtlasDetailPageExample;
