/**
 * ScatterChart Component
 * Plotly scatter with regression line, tooltips, and color coding
 */

class ScatterChart {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container ${containerId} not found`);
        }
        this.containerId = containerId;
        this.options = { ...ScatterChart.DEFAULTS, ...options };
        this.plot = null;
        this.data = null;
    }

    static DEFAULTS = {
        showTrendLine: false,
        showStats: false,
        mode: 'markers',
        title: '',
        xLabel: 'X',
        yLabel: 'Y',
        colors: null,
        sizes: null,
        opacity: 0.7,
    };

    static PLOTLY_CONFIG = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        toImageButtonOptions: {
            format: 'png',
            filename: 'cytoatlas_scatter',
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
        if (!data || !data.x || data.x.length === 0) {
            this.container.innerHTML = '<p class="loading">No data available</p>';
            return;
        }

        this.data = data;

        const trace = {
            x: data.x,
            y: data.y,
            mode: this.options.mode,
            type: 'scatter',
            text: data.labels || data.text,
            marker: {
                size: data.sizes || this.options.sizes || 8,
                color: data.colors || this.options.colors || '#2563eb',
                opacity: this.options.opacity,
                line: {
                    color: 'white',
                    width: 1,
                },
            },
            hovertemplate: data.hoverTemplate || '%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>',
        };

        const traces = [trace];

        // Add trend line if requested
        if (this.options.showTrendLine && data.x.length > 1) {
            const trendLine = this.calculateTrendLine(data.x, data.y);
            traces.push({
                x: trendLine.x,
                y: trendLine.y,
                mode: 'lines',
                type: 'scatter',
                name: 'Trend',
                line: { color: '#ef4444', width: 2, dash: 'dash' },
                hoverinfo: 'skip',
                showlegend: false,
            });
        }

        const layout = {
            ...ScatterChart.PLOTLY_LAYOUT,
            title: this.options.title,
            xaxis: {
                title: this.options.xLabel,
                zeroline: true,
                zerolinecolor: '#e2e8f0',
                gridcolor: '#f1f5f9',
            },
            yaxis: {
                title: this.options.yLabel,
                zeroline: true,
                zerolinecolor: '#e2e8f0',
                gridcolor: '#f1f5f9',
            },
            showlegend: traces.length > 1,
        };

        this.plot = Plotly.newPlot(this.containerId, traces, layout, ScatterChart.PLOTLY_CONFIG);

        // Add stats annotation if requested
        if (this.options.showStats && data.stats) {
            this.addStatsAnnotation(data.stats);
        }
    }

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
    }

    addStatsAnnotation(stats) {
        const annotations = [];
        let text = '';

        if (stats.pearson_r != null) {
            text += `r = ${stats.pearson_r.toFixed(3)}`;
        }
        if (stats.spearman_r != null) {
            text += (text ? '<br>' : '') + `ρ = ${stats.spearman_r.toFixed(3)}`;
        }
        if (stats.p_value != null) {
            text += (text ? '<br>' : '') + `p = ${stats.p_value.toExponential(2)}`;
        }

        if (text) {
            annotations.push({
                x: 0.02,
                y: 0.98,
                xref: 'paper',
                yref: 'paper',
                text: text,
                showarrow: false,
                font: { size: 12 },
                bgcolor: 'white',
                bordercolor: '#e2e8f0',
                borderwidth: 1,
                borderpad: 4,
                xanchor: 'left',
                yanchor: 'top',
            });

            Plotly.relayout(this.containerId, { annotations });
        }
    }

    update(data) {
        if (!this.plot) {
            this.render(data);
            return;
        }

        this.data = data;
        Plotly.restyle(this.containerId, {
            x: [data.x],
            y: [data.y],
            text: [data.labels || data.text],
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

        const rows = [['Label', 'X', 'Y']];
        for (let i = 0; i < this.data.x.length; i++) {
            rows.push([
                this.data.labels?.[i] || `Point${i}`,
                this.data.x[i],
                this.data.y[i]
            ]);
        }

        return rows.map(row => row.join(',')).join('\n');
    }

    exportPNG() {
        if (this.plot) {
            Plotly.downloadImage(this.containerId, {
                format: 'png',
                filename: this.options.title ? this.options.title.replace(/\s/g, '_') : 'scatter',
                width: 800,
                height: 600,
                scale: 2
            });
        }
    }
}

// Make available globally
window.ScatterChart = ScatterChart;
