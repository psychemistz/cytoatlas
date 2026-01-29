/**
 * Compare Page Handler
 * Cross-atlas comparison views
 */

const ComparePage = {
    signatureType: 'CytoSig',

    /**
     * Initialize the compare page
     */
    async init(params, query) {
        if (query.type) {
            this.signatureType = query.type;
        }

        // Render template
        this.render();

        // Load comparison data
        await this.loadComparison();
    },

    /**
     * Render the page template
     */
    render() {
        const app = document.getElementById('app');
        const template = document.getElementById('compare-template');

        if (app && template) {
            app.innerHTML = template.innerHTML;
        }
    },

    /**
     * Load cross-atlas comparison
     */
    async loadComparison() {
        const content = document.getElementById('compare-content');
        if (!content) return;

        content.innerHTML = `
            <div class="compare-controls">
                <select id="compare-type" class="filter-select" onchange="ComparePage.changeSignatureType(this.value)">
                    <option value="CytoSig" ${this.signatureType === 'CytoSig' ? 'selected' : ''}>CytoSig</option>
                    <option value="SecAct" ${this.signatureType === 'SecAct' ? 'selected' : ''}>SecAct</option>
                </select>
            </div>

            <div class="compare-grid">
                <div class="compare-panel">
                    <h3>&#128202; Validation Quality Comparison</h3>
                    <div id="quality-comparison" class="plot-container"></div>
                </div>

                <div class="compare-panel">
                    <h3>&#128200; Cross-Atlas Correlation</h3>
                    <div id="cross-correlation" class="plot-container"></div>
                </div>

                <div class="compare-panel full-width">
                    <h3>&#128209; Consistency Heatmap</h3>
                    <p>Signature activity correlation across atlases</p>
                    <div id="consistency-heatmap" class="plot-container"></div>
                </div>
            </div>
        `;

        // Load all comparison data
        await Promise.all([
            this.loadQualityComparison(),
            this.loadCrossCorrelation(),
            this.loadConsistencyHeatmap(),
        ]);
    },

    /**
     * Load quality comparison across atlases
     */
    async loadQualityComparison() {
        const container = document.getElementById('quality-comparison');
        if (!container) return;

        try {
            const data = await API.compareAtlasValidation(this.signatureType);

            if (data && data.comparison && Object.keys(data.comparison).length > 0) {
                const atlases = Object.keys(data.comparison);
                const scores = atlases.map(a => data.comparison[a].quality_score);
                const colors = atlases.map(a => {
                    const grade = data.comparison[a].quality_grade;
                    return grade === 'A' ? '#10b981' :
                           grade === 'B' ? '#2563eb' :
                           grade === 'C' ? '#f59e0b' : '#ef4444';
                });

                Plotly.newPlot('quality-comparison', [{
                    x: atlases,
                    y: scores,
                    type: 'bar',
                    marker: { color: colors },
                    text: atlases.map(a => `Grade ${data.comparison[a].quality_grade}`),
                    textposition: 'outside',
                }], {
                    title: '',
                    xaxis: { title: 'Atlas' },
                    yaxis: { title: 'Quality Score', range: [0, 100] },
                    margin: { l: 60, r: 30, t: 30, b: 60 },
                }, { responsive: true });
            } else {
                container.innerHTML = '<p class="loading">No comparison data available</p>';
            }
        } catch (error) {
            container.innerHTML = `<p class="loading">Failed to load comparison: ${error.message}</p>`;
        }
    },

    /**
     * Load cross-atlas correlation scatter
     */
    async loadCrossCorrelation() {
        const container = document.getElementById('cross-correlation');
        if (!container) return;

        try {
            const data = await API.getCrossAtlasComparison({
                signature_type: this.signatureType,
            });

            if (data && data.correlations) {
                // Create scatter matrix or pairwise comparison
                const pairs = [];
                const atlases = Object.keys(data.correlations);

                for (let i = 0; i < atlases.length; i++) {
                    for (let j = i + 1; j < atlases.length; j++) {
                        const corr = data.correlations[atlases[i]]?.[atlases[j]];
                        if (corr !== undefined) {
                            pairs.push({
                                pair: `${atlases[i]} vs ${atlases[j]}`,
                                correlation: corr,
                            });
                        }
                    }
                }

                if (pairs.length > 0) {
                    Plotly.newPlot('cross-correlation', [{
                        x: pairs.map(p => p.pair),
                        y: pairs.map(p => p.correlation),
                        type: 'bar',
                        marker: {
                            color: pairs.map(p =>
                                p.correlation > 0.8 ? '#10b981' :
                                p.correlation > 0.5 ? '#2563eb' :
                                p.correlation > 0.3 ? '#f59e0b' : '#ef4444'
                            ),
                        },
                    }], {
                        title: '',
                        xaxis: { title: 'Atlas Pair' },
                        yaxis: { title: 'Correlation (r)', range: [0, 1] },
                        margin: { l: 60, r: 30, t: 30, b: 100 },
                    }, { responsive: true });
                } else {
                    container.innerHTML = '<p class="loading">No cross-correlation data</p>';
                }
            } else {
                container.innerHTML = '<p class="loading">No comparison data available</p>';
            }
        } catch (error) {
            container.innerHTML = `<p class="loading">Failed to load: ${error.message}</p>`;
        }
    },

    /**
     * Load consistency heatmap
     */
    async loadConsistencyHeatmap() {
        const container = document.getElementById('consistency-heatmap');
        if (!container) return;

        try {
            const data = await API.getCrossAtlasConsistency({
                signature_type: this.signatureType,
            });

            if (data && data.z) {
                Heatmap.createCorrelationHeatmap('consistency-heatmap', data, {
                    title: '',
                    xLabel: 'Signature',
                    yLabel: 'Atlas',
                });
            } else {
                container.innerHTML = '<p class="loading">No consistency data available</p>';
            }
        } catch (error) {
            container.innerHTML = `<p class="loading">Failed to load: ${error.message}</p>`;
        }
    },

    /**
     * Change signature type
     */
    changeSignatureType(type) {
        this.signatureType = type;
        this.loadComparison();
    },
};

// Make available globally
window.ComparePage = ComparePage;
