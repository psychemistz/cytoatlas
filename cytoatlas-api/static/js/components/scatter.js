/**
 * Scatter Plot Component
 * Wrapper for Plotly scatter plot visualizations
 */

const Scatter = {
    /**
     * Default configuration
     */
    defaultConfig: {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    },

    /**
     * Default layout settings
     */
    defaultLayout: {
        margin: { l: 60, r: 30, t: 50, b: 60 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { family: 'Inter, sans-serif' },
        hovermode: 'closest',
    },

    /**
     * Create a scatter plot
     * @param {string} containerId - Container element ID
     * @param {Object} data - Scatter data { x, y, labels, colors, sizes }
     * @param {Object} options - Additional options
     */
    create(containerId, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Container ${containerId} not found`);
            return;
        }

        const { x, y, labels, colors, sizes } = data;

        const trace = {
            x: x,
            y: y,
            mode: options.mode || 'markers',
            type: 'scatter',
            text: labels,
            marker: {
                size: sizes || 8,
                color: colors || '#2563eb',
                opacity: options.opacity || 0.7,
                line: {
                    color: 'white',
                    width: 1,
                },
            },
            hovertemplate: options.hoverTemplate || '%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>',
        };

        // Add trend line if requested
        const traces = [trace];
        if (options.showTrendLine && x.length > 1) {
            const trendLine = this.calculateTrendLine(x, y);
            traces.push({
                x: trendLine.x,
                y: trendLine.y,
                mode: 'lines',
                type: 'scatter',
                name: 'Trend',
                line: { color: '#ef4444', width: 2, dash: 'dash' },
                hoverinfo: 'skip',
            });
        }

        const layout = {
            ...this.defaultLayout,
            title: options.title || '',
            xaxis: {
                title: options.xLabel || 'X',
                zeroline: true,
                zerolinecolor: '#e2e8f0',
                gridcolor: '#f1f5f9',
            },
            yaxis: {
                title: options.yLabel || 'Y',
                zeroline: true,
                zerolinecolor: '#e2e8f0',
                gridcolor: '#f1f5f9',
            },
            showlegend: traces.length > 1,
            ...options.layout,
        };

        Plotly.newPlot(containerId, traces, layout, this.defaultConfig);
    },

    /**
     * Calculate trend line using linear regression
     * @param {Array} x - X values
     * @param {Array} y - Y values
     * @returns {Object} Trend line { x, y }
     */
    calculateTrendLine(x, y) {
        const n = x.length;
        let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

        for (let i = 0; i < n; i++) {
            if (x[i] != null && y[i] != null && !isNaN(x[i]) && !isNaN(y[i])) {
                sumX += x[i];
                sumY += y[i];
                sumXY += x[i] * y[i];
                sumX2 += x[i] * x[i];
            }
        }

        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;

        const validX = x.filter(v => v != null && !isNaN(v));
        const minX = Math.min(...validX);
        const maxX = Math.max(...validX);

        return {
            x: [minX, maxX],
            y: [slope * minX + intercept, slope * maxX + intercept],
        };
    },

    /**
     * Create a correlation scatter plot (expression vs activity)
     * @param {string} containerId - Container element ID
     * @param {Object} data - Correlation data from API
     * @param {Object} options - Options
     */
    createCorrelationScatter(containerId, data, options = {}) {
        if (!data || !data.points || data.points.length === 0) {
            document.getElementById(containerId).innerHTML = '<p class="loading">No data available</p>';
            return;
        }

        const x = data.points.map(p => p.expression || p.x);
        const y = data.points.map(p => p.activity || p.y);
        const labels = data.points.map(p => p.sample_id || p.cell_type || p.label || '');

        // Color by group if available
        const colors = data.points.map(p => p.color || '#2563eb');
        const sizes = data.points.map(p => p.n_cells ? Math.sqrt(p.n_cells) / 10 + 5 : 8);

        this.create(containerId, {
            x, y, labels, colors, sizes,
        }, {
            title: options.title || `${data.signature || 'Signature'} Validation`,
            xLabel: options.xLabel || 'Expression',
            yLabel: options.yLabel || 'Activity',
            showTrendLine: true,
            hoverTemplate: '%{text}<br>Expression: %{x:.3f}<br>Activity: %{y:.3f}<extra></extra>',
            ...options,
        });

        // Add annotation with stats
        if (data.stats) {
            Plotly.relayout(containerId, {
                annotations: [{
                    x: 0.02,
                    y: 0.98,
                    xref: 'paper',
                    yref: 'paper',
                    text: `r = ${data.stats.pearson_r?.toFixed(3) || 'N/A'}<br>p = ${data.stats.p_value?.toExponential(2) || 'N/A'}`,
                    showarrow: false,
                    font: { size: 12 },
                    bgcolor: 'white',
                    bordercolor: '#e2e8f0',
                    borderwidth: 1,
                    borderpad: 4,
                }],
            });
        }
    },

    /**
     * Create a box plot
     * @param {string} containerId - Container element ID
     * @param {Object} data - Box plot data { groups, values, labels }
     * @param {Object} options - Options
     */
    createBoxPlot(containerId, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const traces = data.groups.map((group, i) => ({
            y: data.values[i],
            name: group,
            type: 'box',
            boxpoints: options.showPoints ? 'all' : false,
            jitter: 0.3,
            pointpos: -1.8,
            marker: { opacity: 0.5 },
        }));

        const layout = {
            ...this.defaultLayout,
            title: options.title || '',
            yaxis: {
                title: options.yLabel || 'Value',
                zeroline: true,
            },
            showlegend: false,
            ...options.layout,
        };

        Plotly.newPlot(containerId, traces, layout, this.defaultConfig);
    },

    /**
     * Create a violin plot (for single-cell distributions)
     * @param {string} containerId - Container element ID
     * @param {Object} data - Distribution data { expressing, non_expressing }
     * @param {Object} options - Options
     */
    createViolinPlot(containerId, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const traces = [
            {
                y: data.expressing_activities || [],
                name: 'Expressing',
                type: 'violin',
                box: { visible: true },
                meanline: { visible: true },
                fillcolor: 'rgba(37, 99, 235, 0.5)',
                line: { color: '#2563eb' },
            },
            {
                y: data.non_expressing_activities || [],
                name: 'Non-expressing',
                type: 'violin',
                box: { visible: true },
                meanline: { visible: true },
                fillcolor: 'rgba(100, 116, 139, 0.5)',
                line: { color: '#64748b' },
            },
        ];

        const layout = {
            ...this.defaultLayout,
            title: options.title || 'Activity Distribution',
            yaxis: {
                title: 'Activity (z-score)',
                zeroline: true,
            },
            violinmode: 'group',
            ...options.layout,
        };

        Plotly.newPlot(containerId, traces, layout, this.defaultConfig);
    },

    /**
     * Update scatter plot data
     * @param {string} containerId - Container element ID
     * @param {Object} data - New data
     */
    update(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container) return;

        Plotly.restyle(containerId, {
            x: [data.x],
            y: [data.y],
            text: [data.labels],
        });
    },

    /**
     * Destroy a scatter plot
     * @param {string} containerId - Container element ID
     */
    destroy(containerId) {
        const container = document.getElementById(containerId);
        if (container) {
            Plotly.purge(containerId);
        }
    },
};

// Make available globally
window.Scatter = Scatter;
