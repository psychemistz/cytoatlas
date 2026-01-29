/**
 * Heatmap Component
 * Wrapper for Plotly heatmap visualizations
 */

const Heatmap = {
    /**
     * Default configuration for heatmaps
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
        margin: { l: 150, r: 50, t: 50, b: 100 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { family: 'Inter, sans-serif' },
    },

    /**
     * Create a heatmap
     * @param {string} containerId - Container element ID
     * @param {Object} data - Heatmap data { z, x, y }
     * @param {Object} options - Additional options
     */
    create(containerId, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Container ${containerId} not found`);
            return;
        }

        const { z, x, y, colorscale = 'RdBu', zmin, zmax, reversescale = true } = data;

        // Calculate zmin/zmax if not provided
        const flatZ = z.flat().filter(v => v != null && !isNaN(v));
        const calculatedZmin = zmin ?? Math.min(...flatZ);
        const calculatedZmax = zmax ?? Math.max(...flatZ);

        // Symmetric colorscale for activity data
        const absMax = Math.max(Math.abs(calculatedZmin), Math.abs(calculatedZmax));
        const symmetricZmin = options.symmetric ? -absMax : calculatedZmin;
        const symmetricZmax = options.symmetric ? absMax : calculatedZmax;

        const trace = {
            z: z,
            x: x,
            y: y,
            type: 'heatmap',
            colorscale: colorscale,
            reversescale: reversescale,
            zmin: symmetricZmin,
            zmax: symmetricZmax,
            hoverongaps: false,
            hovertemplate: `${options.xLabel || 'X'}: %{x}<br>${options.yLabel || 'Y'}: %{y}<br>Value: %{z:.3f}<extra></extra>`,
            colorbar: {
                title: options.colorbarTitle || 'Value',
                titleside: 'right',
            },
        };

        const layout = {
            ...this.defaultLayout,
            title: options.title || '',
            xaxis: {
                title: options.xLabel || '',
                tickangle: options.xTickAngle || -45,
                tickfont: { size: 10 },
            },
            yaxis: {
                title: options.yLabel || '',
                tickfont: { size: 10 },
            },
            ...options.layout,
        };

        Plotly.newPlot(containerId, [trace], layout, this.defaultConfig);
    },

    /**
     * Create a clustered heatmap with dendrograms (simplified - uses ordering)
     * @param {string} containerId - Container element ID
     * @param {Object} data - Heatmap data with optional row/col order
     * @param {Object} options - Additional options
     */
    createClustered(containerId, data, options = {}) {
        // For now, just use provided order or default order
        // Full dendrogram support would require additional clustering library
        this.create(containerId, data, { ...options, symmetric: true });
    },

    /**
     * Update heatmap data
     * @param {string} containerId - Container element ID
     * @param {Object} data - New data { z, x, y }
     */
    update(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container) return;

        Plotly.restyle(containerId, {
            z: [data.z],
            x: [data.x],
            y: [data.y],
        });
    },

    /**
     * Create activity heatmap (cell types x signatures)
     * @param {string} containerId - Container element ID
     * @param {Object} activityData - Activity data from API
     * @param {Object} options - Options
     */
    createActivityHeatmap(containerId, activityData, options = {}) {
        if (!activityData || !activityData.z || activityData.z.length === 0) {
            document.getElementById(containerId).innerHTML = '<p class="loading">No data available</p>';
            return;
        }

        this.create(containerId, {
            z: activityData.z,
            x: activityData.signatures || activityData.x,
            y: activityData.cell_types || activityData.y,
            colorscale: 'RdBu',
            reversescale: true,
        }, {
            title: options.title || 'Cytokine Activity',
            xLabel: 'Signature',
            yLabel: 'Cell Type',
            colorbarTitle: 'Activity (z-score)',
            symmetric: true,
            xTickAngle: -45,
            ...options,
        });
    },

    /**
     * Create correlation heatmap
     * @param {string} containerId - Container element ID
     * @param {Object} corrData - Correlation data
     * @param {Object} options - Options
     */
    createCorrelationHeatmap(containerId, corrData, options = {}) {
        if (!corrData || !corrData.z) {
            document.getElementById(containerId).innerHTML = '<p class="loading">No data available</p>';
            return;
        }

        this.create(containerId, {
            z: corrData.z,
            x: corrData.x,
            y: corrData.y,
            colorscale: 'RdBu',
            reversescale: true,
            zmin: -1,
            zmax: 1,
        }, {
            title: options.title || 'Correlation',
            xLabel: options.xLabel || '',
            yLabel: options.yLabel || '',
            colorbarTitle: 'Correlation (r)',
            symmetric: true,
            ...options,
        });
    },

    /**
     * Destroy a heatmap
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
window.Heatmap = Heatmap;
