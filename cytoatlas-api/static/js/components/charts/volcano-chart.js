/**
 * VolcanoChart Component
 * Plotly scatter configured for volcano plots (activity_diff vs -log10(p))
 */

class VolcanoChart {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container ${containerId} not found`);
        }
        this.containerId = containerId;
        this.options = { ...VolcanoChart.DEFAULTS, ...options };
        this.plot = null;
        this.data = null;
    }

    static DEFAULTS = {
        title: 'Volcano Plot',
        xLabel: 'Δ Activity',
        yLabel: '-log10(p-value)',
        fdrThreshold: 0.05,
        activityThreshold: 0.5,
        showThresholds: true,
    };

    static PLOTLY_CONFIG = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        toImageButtonOptions: {
            format: 'png',
            filename: 'cytoatlas_volcano',
            scale: 2
        }
    };

    static PLOTLY_LAYOUT = {
        font: { family: 'Inter, system-ui, sans-serif', size: 12 },
        margin: { l: 80, r: 40, t: 40, b: 80 },
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        hovermode: 'closest',
    };

    render(data) {
        if (!data || !data.points || data.points.length === 0) {
            this.container.innerHTML = '<p class="loading">No data available</p>';
            return;
        }

        this.data = data;

        // Extract values
        const x = data.points.map(p => p.activity_diff || p.log2fc || p.x);
        const y = data.points.map(p => {
            const pval = p.p_value || p.pval || p.y;
            return pval > 0 ? -Math.log10(pval) : 0;
        });
        const labels = data.points.map(p => p.signature || p.label || p.gene || '');

        // Color points by significance
        const colors = data.points.map(p => {
            const activityDiff = Math.abs(p.activity_diff || p.log2fc || p.x || 0);
            const pval = p.fdr || p.p_value || p.pval || 1;

            if (pval < this.options.fdrThreshold && activityDiff > this.options.activityThreshold) {
                return (p.activity_diff || p.log2fc || p.x || 0) > 0 ? '#dc2626' : '#2563eb';
            }
            return '#94a3b8';
        });

        const trace = {
            x: x,
            y: y,
            mode: 'markers',
            type: 'scatter',
            text: labels,
            marker: {
                size: 6,
                color: colors,
                opacity: 0.7,
                line: {
                    color: 'white',
                    width: 0.5,
                },
            },
            hovertemplate: '%{text}<br>Δ Activity: %{x:.3f}<br>-log10(p): %{y:.3f}<extra></extra>',
        };

        const layout = {
            ...VolcanoChart.PLOTLY_LAYOUT,
            title: this.options.title,
            xaxis: {
                title: this.options.xLabel,
                zeroline: true,
                zerolinecolor: '#e2e8f0',
                gridcolor: '#f1f5f9',
            },
            yaxis: {
                title: this.options.yLabel,
                zeroline: false,
                gridcolor: '#f1f5f9',
            },
            shapes: [],
        };

        // Add threshold lines
        if (this.options.showThresholds) {
            const pThreshold = -Math.log10(this.options.fdrThreshold);
            const xRange = [Math.min(...x), Math.max(...x)];

            // Horizontal line for p-value threshold
            layout.shapes.push({
                type: 'line',
                x0: xRange[0],
                x1: xRange[1],
                y0: pThreshold,
                y1: pThreshold,
                line: {
                    color: '#94a3b8',
                    width: 1,
                    dash: 'dash',
                },
            });

            // Vertical lines for activity threshold
            layout.shapes.push({
                type: 'line',
                x0: this.options.activityThreshold,
                x1: this.options.activityThreshold,
                y0: 0,
                y1: Math.max(...y),
                line: {
                    color: '#94a3b8',
                    width: 1,
                    dash: 'dash',
                },
            });

            layout.shapes.push({
                type: 'line',
                x0: -this.options.activityThreshold,
                x1: -this.options.activityThreshold,
                y0: 0,
                y1: Math.max(...y),
                line: {
                    color: '#94a3b8',
                    width: 1,
                    dash: 'dash',
                },
            });
        }

        this.plot = Plotly.newPlot(this.containerId, [trace], layout, VolcanoChart.PLOTLY_CONFIG);
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

        const rows = [['Signature', 'Activity_Diff', 'P_Value', 'FDR']];
        this.data.points.forEach(p => {
            rows.push([
                p.signature || p.label || p.gene || '',
                p.activity_diff || p.log2fc || p.x || '',
                p.p_value || p.pval || '',
                p.fdr || ''
            ]);
        });

        return rows.map(row => row.join(',')).join('\n');
    }

    exportPNG() {
        if (this.plot) {
            Plotly.downloadImage(this.containerId, {
                format: 'png',
                filename: this.options.title ? this.options.title.replace(/\s/g, '_') : 'volcano',
                width: 800,
                height: 600,
                scale: 2
            });
        }
    }
}

// Make available globally
window.VolcanoChart = VolcanoChart;
