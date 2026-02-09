/**
 * BoxplotChart Component
 * Plotly boxplot with group comparison and jitter points
 */

class BoxplotChart {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container ${containerId} not found`);
        }
        this.containerId = containerId;
        this.options = { ...BoxplotChart.DEFAULTS, ...options };
        this.plot = null;
        this.data = null;
    }

    static DEFAULTS = {
        showPoints: true,
        jitter: 0.3,
        pointpos: -1.8,
        title: '',
        xLabel: '',
        yLabel: 'Value',
        colors: null,
        orientation: 'v', // 'v' or 'h'
    };

    static PLOTLY_CONFIG = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        toImageButtonOptions: {
            format: 'png',
            filename: 'cytoatlas_boxplot',
            scale: 2
        }
    };

    static PLOTLY_LAYOUT = {
        font: { family: 'Inter, system-ui, sans-serif', size: 12 },
        margin: { l: 80, r: 40, t: 40, b: 80 },
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
    };

    render(data) {
        if (!data || !data.groups || data.groups.length === 0) {
            this.container.innerHTML = '<p class="loading">No data available</p>';
            return;
        }

        this.data = data;

        const traces = data.groups.map((group, i) => {
            const trace = {
                name: group,
                type: 'box',
                boxpoints: this.options.showPoints ? 'all' : false,
                jitter: this.options.jitter,
                pointpos: this.options.pointpos,
                marker: {
                    color: this.options.colors ? this.options.colors[i] : undefined,
                    opacity: 0.5,
                    size: 4,
                },
                line: {
                    color: this.options.colors ? this.options.colors[i] : undefined,
                },
            };

            if (this.options.orientation === 'v') {
                trace.y = data.values[i];
                trace.x = Array(data.values[i].length).fill(group);
            } else {
                trace.x = data.values[i];
                trace.y = Array(data.values[i].length).fill(group);
            }

            return trace;
        });

        const layout = {
            ...BoxplotChart.PLOTLY_LAYOUT,
            title: this.options.title,
            showlegend: false,
        };

        if (this.options.orientation === 'v') {
            layout.yaxis = {
                title: this.options.yLabel,
                zeroline: true,
                zerolinecolor: '#e2e8f0',
                gridcolor: '#f1f5f9',
            };
            layout.xaxis = {
                title: this.options.xLabel,
            };
        } else {
            layout.xaxis = {
                title: this.options.yLabel,
                zeroline: true,
                zerolinecolor: '#e2e8f0',
                gridcolor: '#f1f5f9',
            };
            layout.yaxis = {
                title: this.options.xLabel,
            };
        }

        this.plot = Plotly.newPlot(this.containerId, traces, layout, BoxplotChart.PLOTLY_CONFIG);
    }

    update(data) {
        if (!this.plot) {
            this.render(data);
            return;
        }
        this.data = data;
        // Re-render for simplicity
        this.destroy();
        this.render(data);
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

        const rows = [['Group', 'Value']];
        this.data.groups.forEach((group, i) => {
            this.data.values[i].forEach(value => {
                rows.push([group, value]);
            });
        });

        return rows.map(row => row.join(',')).join('\n');
    }

    exportPNG() {
        if (this.plot) {
            Plotly.downloadImage(this.containerId, {
                format: 'png',
                filename: this.options.title ? this.options.title.replace(/\s/g, '_') : 'boxplot',
                width: 800,
                height: 600,
                scale: 2
            });
        }
    }
}

// Make available globally
window.BoxplotChart = BoxplotChart;
