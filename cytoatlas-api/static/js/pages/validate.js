/**
 * Validation Page Handler
 * 5-type credibility assessment dashboard
 */

const ValidatePage = {
    currentAtlas: null,
    currentSignature: null,
    signatureType: 'CytoSig',

    /**
     * Initialize the validation page
     */
    async init(params, query) {
        // Get atlas from query params if provided
        if (query.atlas) {
            this.currentAtlas = query.atlas;
        }
        if (query.signature) {
            this.currentSignature = query.signature;
        }
        if (query.type) {
            this.signatureType = query.type;
        }

        // Render template
        this.render();

        // Load atlases for dropdown
        await this.loadAtlases();

        // If atlas is set, load validation data
        if (this.currentAtlas) {
            await this.loadSignatures();
            if (this.currentSignature) {
                await this.loadValidationPanels();
            } else {
                await this.loadValidationSummary();
            }
        }
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
    },

    /**
     * Load available atlases
     */
    async loadAtlases() {
        const select = document.getElementById('val-atlas');
        if (!select) return;

        try {
            const atlases = await API.getValidationAtlases();

            select.innerHTML = '<option value="">Select Atlas</option>';
            atlases.forEach(atlas => {
                const selected = atlas === this.currentAtlas ? 'selected' : '';
                select.innerHTML += `<option value="${atlas}" ${selected}>${atlas}</option>`;
            });

            select.addEventListener('change', async (e) => {
                this.currentAtlas = e.target.value;
                this.currentSignature = null;
                await this.loadSignatures();
                await this.loadValidationSummary();
            });
        } catch (error) {
            // Fallback to hardcoded atlases
            select.innerHTML = `
                <option value="">Select Atlas</option>
                <option value="cima" ${this.currentAtlas === 'cima' ? 'selected' : ''}>CIMA</option>
                <option value="inflammation" ${this.currentAtlas === 'inflammation' ? 'selected' : ''}>Inflammation</option>
                <option value="scatlas" ${this.currentAtlas === 'scatlas' ? 'selected' : ''}>scAtlas</option>
            `;
        }

        // Set up type selector
        const typeSelect = document.getElementById('val-type');
        if (typeSelect) {
            typeSelect.value = this.signatureType;
            typeSelect.addEventListener('change', async (e) => {
                this.signatureType = e.target.value;
                await this.loadSignatures();
                if (this.currentAtlas) {
                    await this.loadValidationSummary();
                }
            });
        }
    },

    /**
     * Load available signatures for the selected atlas
     */
    async loadSignatures() {
        const select = document.getElementById('val-signature');
        if (!select || !this.currentAtlas) {
            if (select) select.innerHTML = '<option value="">Select Signature</option>';
            return;
        }

        try {
            const signatures = await API.getValidationSignatures(this.currentAtlas, this.signatureType);

            select.innerHTML = '<option value="">All Signatures (Summary)</option>';
            signatures.forEach(sig => {
                const selected = sig === this.currentSignature ? 'selected' : '';
                select.innerHTML += `<option value="${sig}" ${selected}>${sig}</option>`;
            });

            select.addEventListener('change', async (e) => {
                this.currentSignature = e.target.value;
                if (this.currentSignature) {
                    await this.loadValidationPanels();
                } else {
                    await this.loadValidationSummary();
                }
            });
        } catch (error) {
            select.innerHTML = '<option value="">No signatures available</option>';
        }
    },

    /**
     * Load validation summary (overall quality)
     */
    async loadValidationSummary() {
        const summaryContainer = document.getElementById('validation-summary');
        const panelsContainer = document.getElementById('validation-panels');

        if (!summaryContainer || !this.currentAtlas) return;

        summaryContainer.innerHTML = '<div class="loading"><div class="spinner"></div>Loading summary...</div>';
        panelsContainer.innerHTML = '';

        try {
            const summary = await API.getValidationSummary(this.currentAtlas, this.signatureType);

            const gradeClass = `grade-${summary.quality_grade?.toLowerCase() || 'c'}`;

            summaryContainer.innerHTML = `
                <h3>Overall Quality: ${summary.quality_score}/100 (Grade ${summary.quality_grade})</h3>
                <div class="quality-bar">
                    <div class="quality-bar-fill ${gradeClass}" style="width: ${summary.quality_score}%"></div>
                </div>
                <div class="summary-metrics">
                    <div class="metric">
                        <span class="metric-label">Sample-Level Correlation</span>
                        <span class="metric-value">r = ${summary.sample_level_mean_r?.toFixed(3) || 'N/A'}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Cell Type-Level Correlation</span>
                        <span class="metric-value">r = ${summary.celltype_level_mean_r?.toFixed(3) || 'N/A'}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Gene Coverage</span>
                        <span class="metric-value">${(summary.mean_gene_coverage * 100)?.toFixed(1) || 'N/A'}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Biological Validation</span>
                        <span class="metric-value">${(summary.biological_validation_rate * 100)?.toFixed(1) || 'N/A'}%</span>
                    </div>
                </div>
                <p class="interpretation">${summary.interpretation || ''}</p>
            `;

            // Add recommendations if present
            if (summary.recommendations && summary.recommendations.length > 0) {
                summaryContainer.innerHTML += `
                    <div class="recommendations">
                        <h4>Recommendations</h4>
                        <ul>
                            ${summary.recommendations.map(r => `<li>${r}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }

            // Load biological associations table
            await this.loadBiologicalAssociations(panelsContainer);

        } catch (error) {
            summaryContainer.innerHTML = `<p class="loading">Failed to load summary: ${error.message}</p>`;
        }
    },

    /**
     * Load biological associations table
     */
    async loadBiologicalAssociations(container) {
        try {
            const data = await API.getBiologicalAssociations(this.currentAtlas, this.signatureType);

            if (data && data.associations && data.associations.length > 0) {
                const tableRows = data.associations.map(assoc => `
                    <tr class="${assoc.validated ? 'validated' : 'not-validated'}">
                        <td>${assoc.signature}</td>
                        <td>${assoc.expected_cell_type}</td>
                        <td>${assoc.observed_top_cell_type || 'N/A'}</td>
                        <td>${assoc.rank || 'N/A'}</td>
                        <td>${assoc.validated ? '&#9989;' : '&#10060;'}</td>
                    </tr>
                `).join('');

                container.innerHTML = `
                    <div class="validation-panel" style="grid-column: 1 / -1;">
                        <h4>&#128300; Biological Associations</h4>
                        <p>Validation rate: ${(data.validation_rate * 100).toFixed(1)}%</p>
                        <table class="validation-table">
                            <thead>
                                <tr>
                                    <th>Signature</th>
                                    <th>Expected Cell Type</th>
                                    <th>Observed Top</th>
                                    <th>Rank</th>
                                    <th>Valid</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${tableRows}
                            </tbody>
                        </table>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Failed to load biological associations:', error);
        }
    },

    /**
     * Load validation panels for a specific signature
     */
    async loadValidationPanels() {
        const summaryContainer = document.getElementById('validation-summary');
        const panelsContainer = document.getElementById('validation-panels');

        if (!panelsContainer || !this.currentAtlas || !this.currentSignature) return;

        summaryContainer.innerHTML = `<h3>Validation: ${this.currentSignature}</h3>`;
        panelsContainer.innerHTML = '<div class="loading"><div class="spinner"></div>Loading validation panels...</div>';

        // Load all validation types in parallel
        const panels = await Promise.allSettled([
            this.loadSampleLevelPanel(),
            this.loadCellTypeLevelPanel(),
            this.loadSingleCellPanel(),
            this.loadGeneCoveragePanel(),
        ]);

        panelsContainer.innerHTML = panels
            .filter(p => p.status === 'fulfilled' && p.value)
            .map(p => p.value)
            .join('');

        // Initialize plots
        await this.initializePlots();
    },

    /**
     * Load sample-level validation panel
     */
    async loadSampleLevelPanel() {
        try {
            const data = await API.getSampleLevelValidation(
                this.currentAtlas,
                this.currentSignature,
                this.signatureType
            );

            if (!data) return null;

            return `
                <div class="validation-panel">
                    <h4>&#128202; Sample-Level Validation</h4>
                    <p>Expression vs Activity across samples</p>
                    <div id="sample-level-plot" class="plot-container" style="height: 300px;"
                         data-validation-type="sample-level"></div>
                    <div class="panel-stats">
                        <span>r = ${data.stats?.pearson_r?.toFixed(3) || 'N/A'}</span>
                        <span>p = ${data.stats?.p_value?.toExponential(2) || 'N/A'}</span>
                    </div>
                </div>
            `;
        } catch (error) {
            return null;
        }
    },

    /**
     * Load cell type-level validation panel
     */
    async loadCellTypeLevelPanel() {
        try {
            const data = await API.getCellTypeLevelValidation(
                this.currentAtlas,
                this.currentSignature,
                this.signatureType
            );

            if (!data) return null;

            return `
                <div class="validation-panel">
                    <h4>&#128300; Cell Type-Level Validation</h4>
                    <p>Expression vs Activity across cell types</p>
                    <div id="celltype-level-plot" class="plot-container" style="height: 300px;"
                         data-validation-type="celltype-level"></div>
                    <div class="panel-stats">
                        <span>r = ${data.stats?.pearson_r?.toFixed(3) || 'N/A'}</span>
                        <span>Concordance: ${(data.biological_concordance * 100)?.toFixed(1) || 'N/A'}%</span>
                    </div>
                </div>
            `;
        } catch (error) {
            return null;
        }
    },

    /**
     * Load single-cell validation panel
     */
    async loadSingleCellPanel() {
        try {
            const data = await API.getSingleCellDirect(
                this.currentAtlas,
                this.currentSignature,
                this.signatureType
            );

            if (!data) return null;

            return `
                <div class="validation-panel">
                    <h4>&#128302; Single-Cell Direct Validation</h4>
                    <p>Activity in expressing vs non-expressing cells</p>
                    <div id="singlecell-plot" class="plot-container" style="height: 300px;"
                         data-validation-type="singlecell"></div>
                    <div class="panel-stats">
                        <span>Fold Change: ${data.fold_change?.toFixed(2) || 'N/A'}x</span>
                        <span>p = ${data.p_value?.toExponential(2) || 'N/A'}</span>
                    </div>
                </div>
            `;
        } catch (error) {
            return null;
        }
    },

    /**
     * Load gene coverage panel
     */
    async loadGeneCoveragePanel() {
        try {
            const data = await API.getGeneCoverage(
                this.currentAtlas,
                this.currentSignature,
                this.signatureType
            );

            if (!data) return null;

            const coveragePercent = (data.coverage * 100).toFixed(1);
            const gradeClass = data.quality === 'excellent' ? 'grade-a' :
                              data.quality === 'good' ? 'grade-b' :
                              data.quality === 'moderate' ? 'grade-c' : 'grade-d';

            return `
                <div class="validation-panel">
                    <h4>&#128269; Gene Coverage</h4>
                    <p>Signature genes detected in atlas</p>
                    <div class="coverage-display">
                        <div class="coverage-value">${coveragePercent}%</div>
                        <div class="coverage-details">
                            <span>${data.n_detected} / ${data.n_total} genes</span>
                            <span class="atlas-badge ${gradeClass}">${data.quality}</span>
                        </div>
                    </div>
                    <div class="quality-bar">
                        <div class="quality-bar-fill ${gradeClass}" style="width: ${coveragePercent}%"></div>
                    </div>
                </div>
            `;
        } catch (error) {
            return null;
        }
    },

    /**
     * Initialize plots after HTML is rendered
     */
    async initializePlots() {
        // Sample-level plot
        const samplePlot = document.getElementById('sample-level-plot');
        if (samplePlot) {
            try {
                const data = await API.getSampleLevelValidation(
                    this.currentAtlas,
                    this.currentSignature,
                    this.signatureType
                );
                if (data) {
                    Scatter.createCorrelationScatter('sample-level-plot', data, {
                        xLabel: 'Expression',
                        yLabel: 'Activity',
                    });
                }
            } catch (e) {
                samplePlot.innerHTML = '<p class="loading">No data</p>';
            }
        }

        // Cell type-level plot
        const celltypePlot = document.getElementById('celltype-level-plot');
        if (celltypePlot) {
            try {
                const data = await API.getCellTypeLevelValidation(
                    this.currentAtlas,
                    this.currentSignature,
                    this.signatureType
                );
                if (data) {
                    Scatter.createCorrelationScatter('celltype-level-plot', data, {
                        xLabel: 'Expression',
                        yLabel: 'Activity',
                    });
                }
            } catch (e) {
                celltypePlot.innerHTML = '<p class="loading">No data</p>';
            }
        }

        // Single-cell distribution plot
        const scPlot = document.getElementById('singlecell-plot');
        if (scPlot) {
            try {
                const data = await API.getSingleCellDistribution(
                    this.currentAtlas,
                    this.currentSignature,
                    this.signatureType
                );
                if (data) {
                    Scatter.createViolinPlot('singlecell-plot', data, {
                        title: '',
                    });
                }
            } catch (e) {
                scPlot.innerHTML = '<p class="loading">No data</p>';
            }
        }
    },
};

// Make available globally
window.ValidatePage = ValidatePage;
