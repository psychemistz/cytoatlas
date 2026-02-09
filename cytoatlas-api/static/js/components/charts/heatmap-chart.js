/**
 * HeatmapChart Component
 * Standardized Plotly heatmap with clustering and annotations
 */

class HeatmapChart {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container ${containerId} not found`);
        }
        this.containerId = containerId;
        this.options = { ...HeatmapChart.DEFAULTS, ...options };
        this.plot = null;
        this.data = null;
    }

    static DEFAULTS = {
        colorscale: [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
        symmetric: true,
        reversescale: false,
        showscale: true,
        title: '',
        xLabel: '',
        yLabel: '',
        colorbarTitle: 'Value',
        xTickAngle: -45,
        yTickAngle: 0,
    };

    static PLOTLY_CONFIG = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        toImageButtonOptions: {
            format: 'png',
            filename: 'cytoatlas_heatmap',
            scale: 2
        }
    };

    static PLOTLY_LAYOUT = {
        font: { family: 'Inter, system-ui, sans-serif', size: 12 },
        margin: { l: 150, r: 50, t: 50, b: 150 },
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
    };

    render(data) {
        if (!data || !data.z || data.z.length === 0) {
            this.container.innerHTML = '<p class="loading">No data available</p>';
            return;
        }

        this.data = data;

        // Calculate z-range
        const flatZ = data.z.flat().filter(v => v != null && !isNaN(v));
        if (flatZ.length === 0) {
            this.container.innerHTML = '<p class="loading">No valid data</p>';
            return;
        }

        let zmin = data.zmin ?? Math.min(...flatZ);
        let zmax = data.zmax ?? Math.max(...flatZ);

        // Symmetric range for activity data
        if (this.options.symmetric) {
            const absMax = Math.max(Math.abs(zmin), Math.abs(zmax));
            zmin = -absMax;
            zmax = absMax;
        }

        const trace = {
            z: data.z,
            x: data.x || data.signatures || [],
            y: data.y || data.cell_types || [],
            type: 'heatmap',
            colorscale: data.colorscale || this.options.colorscale,
            reversescale: this.options.reversescale,
            zmin: zmin,
            zmax: zmax,
            hoverongaps: false,
            hovertemplate: `${this.options.xLabel || 'X'}: %{x}<br>${this.options.yLabel || 'Y'}: %{y}<br>Value: %{z:.3f}<extra></extra>`,
            showscale: this.options.showscale,
            colorbar: {
                title: this.options.colorbarTitle,
                titleside: 'right',
                len: 0.9,
            },
        };

        const layout = {
            ...HeatmapChart.PLOTLY_LAYOUT,
            title: this.options.title,
            xaxis: {
                title: this.options.xLabel,
                tickangle: this.options.xTickAngle,
                tickfont: { size: 10 },
                side: 'bottom',
            },
            yaxis: {
                title: this.options.yLabel,
                tickangle: this.options.yTickAngle,
                tickfont: { size: 10 },
            },
        };

        this.plot = Plotly.newPlot(this.containerId, [trace], layout, HeatmapChart.PLOTLY_CONFIG);
    }

    update(data) {
        if (!this.plot) {
            this.render(data);
            return;
        }

        this.data = data;
        Plotly.restyle(this.containerId, {
            z: [data.z],
            x: [data.x || data.signatures],
            y: [data.y || data.cell_types],
        });
    }

    resize() {
        if (this.plot && this.container) {
            Plotly.Plots.resize(this.containerId);
        }
    }

    destroy() {
        if (this.plot) {
            Plotly.purge(this.containerId);
            this.plot = null;
        }
    }

    exportCSV() {
        if (!this.data) return '';

        const rows = [];
        const headers = ['', ...(this.data.x || this.data.signatures || [])];
        rows.push(headers.join(','));

        const yLabels = this.data.y || this.data.cell_types || [];
        this.data.z.forEach((row, i) => {
            const rowData = [yLabels[i] || `Row${i}`, ...row];
            rows.push(rowData.join(','));
        });

        return rows.join('\n');
    }

    exportPNG() {
        if (this.plot) {
            Plotly.downloadImage(this.containerId, {
                format: 'png',
                filename: this.options.title ? this.options.title.replace(/\s/g, '_') : 'heatmap',
                width: 1200,
                height: 800,
                scale: 2
            });
        }
    }
}

// Make available globally
window.HeatmapChart = HeatmapChart;
