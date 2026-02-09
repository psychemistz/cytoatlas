/**
 * BarChart Component
 * Plotly bar chart (horizontal/vertical), grouped/stacked
 */

class BarChart {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container ${containerId} not found`);
        }
        this.containerId = containerId;
        this.options = { ...BarChart.DEFAULTS, ...options };
        this.plot = null;
        this.data = null;
    }

    static DEFAULTS = {
        orientation: 'v', // 'v' or 'h'
        barmode: 'group', // 'group', 'stack', 'relative'
        title: '',
        xLabel: '',
        yLabel: 'Value',
        colors: null,
        showValues: false,
    };

    static PLOTLY_CONFIG = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        toImageButtonOptions: {
            format: 'png',
            filename: 'cytoatlas_bar',
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
        if (!data || !data.categories || data.categories.length === 0) {
            this.container.innerHTML = '<p class="loading">No data available</p>';
            return;
        }

        this.data = data;

        // Support multiple series
        const traces = [];

        if (data.series && Array.isArray(data.series)) {
            // Multiple series (grouped/stacked bars)
            data.series.forEach((s, i) => {
                const trace = {
                    name: s.name || `Series ${i + 1}`,
                    type: 'bar',
                    orientation: this.options.orientation,
                    marker: {
                        color: this.options.colors ? this.options.colors[i] : undefined,
                    },
                };

                if (this.options.orientation === 'v') {
                    trace.x = data.categories;
                    trace.y = s.values;
                } else {
                    trace.y = data.categories;
                    trace.x = s.values;
                }

                if (this.options.showValues) {
                    trace.text = s.values.map(v => v.toFixed(2));
                    trace.textposition = 'auto';
                }

                traces.push(trace);
            });
        } else {
            // Single series
            const trace = {
                type: 'bar',
                orientation: this.options.orientation,
                marker: {
                    color: data.colors || this.options.colors || '#2563eb',
                },
            };

            if (this.options.orientation === 'v') {
                trace.x = data.categories;
                trace.y = data.values;
            } else {
                trace.y = data.categories;
                trace.x = data.values;
            }

            if (this.options.showValues) {
                trace.text = data.values.map(v => v.toFixed(2));
                trace.textposition = 'auto';
            }

            traces.push(trace);
        }

        const layout = {
            ...BarChart.PLOTLY_LAYOUT,
            title: this.options.title,
            barmode: this.options.barmode,
            showlegend: traces.length > 1,
        };

        if (this.options.orientation === 'v') {
            layout.xaxis = {
                title: this.options.xLabel,
                tickangle: -45,
            };
            layout.yaxis = {
                title: this.options.yLabel,
                zeroline: true,
                zerolinecolor: '#e2e8f0',
                gridcolor: '#f1f5f9',
            };
        } else {
            layout.yaxis = {
                title: this.options.xLabel,
            };
            layout.xaxis = {
                title: this.options.yLabel,
                zeroline: true,
                zerolinecolor: '#e2e8f0',
                gridcolor: '#f1f5f9',
            };
        }

        this.plot = Plotly.newPlot(this.containerId, traces, layout, BarChart.PLOTLY_CONFIG);
    }

    update(data) {
        if (!this.plot) {
            this.render(data);
            return;
        }
        this.data = data;
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

        const rows = [];

        if (this.data.series) {
            const headers = ['Category', ...this.data.series.map(s => s.name)];
            rows.push(headers.join(','));

            this.data.categories.forEach((cat, i) => {
                const row = [cat, ...this.data.series.map(s => s.values[i])];
                rows.push(row.join(','));
            });
        } else {
            rows.push(['Category', 'Value']);
            this.data.categories.forEach((cat, i) => {
                rows.push([cat, this.data.values[i]].join(','));
            });
        }

        return rows.join('\n');
    }

    exportPNG() {
        if (this.plot) {
            Plotly.downloadImage(this.containerId, {
                format: 'png',
                filename: this.options.title ? this.options.title.replace(/\s/g, '_') : 'bar',
                width: 800,
                height: 600,
                scale: 2
            });
        }
    }
}

// Make available globally
window.BarChart = BarChart;
