/**
 * Forest Plot Component
 * Creates forest plots for meta-analysis visualization
 */

const ForestPlot = {
    /**
     * Create a forest plot
     * @param {string} containerId - DOM element ID for the chart
     * @param {Array} data - Array of signature objects with individual and pooled effects
     * @param {Object} options - Configuration options
     */
    create(containerId, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Container ${containerId} not found`);
            return;
        }

        if (!data || data.length === 0) {
            container.innerHTML = '<p class="loading">No meta-analysis data available</p>';
            return;
        }

        // Limit to top N signatures for readability
        const maxSignatures = options.maxSignatures || 20;
        const displayData = data.slice(0, maxSignatures);

        // Build traces for each atlas (individual effects)
        const atlasColors = {
            'CIMA': '#3b82f6',
            'Inflammation': '#f59e0b',
            'scAtlas': '#10b981',
        };

        const traces = [];
        const atlases = [...new Set(
            displayData.flatMap(d => d.individual_effects.map(e => e.atlas))
        )];

        // Y-axis positions (reversed so first signature is at top)
        const yPositions = displayData.map((_, i) => displayData.length - i);

        // Individual atlas effects (dots with error bars)
        atlases.forEach((atlas, atlasIdx) => {
            const y = [];
            const x = [];
            const errorX = [];
            const text = [];

            displayData.forEach((sig, sigIdx) => {
                const effect = sig.individual_effects.find(e => e.atlas === atlas);
                if (effect && effect.effect !== null) {
                    y.push(yPositions[sigIdx] + (atlasIdx - 1) * 0.15); // Slight vertical offset per atlas
                    x.push(effect.effect);
                    errorX.push(effect.se * 1.96); // 95% CI
                    text.push(`${atlas}: ${effect.effect.toFixed(3)} (n=${effect.n})`);
                }
            });

            if (x.length > 0) {
                traces.push({
                    type: 'scatter',
                    mode: 'markers',
                    name: atlas,
                    x: x,
                    y: y,
                    marker: {
                        color: atlasColors[atlas] || '#6b7280',
                        size: 8,
                        symbol: 'circle',
                    },
                    error_x: {
                        type: 'data',
                        array: errorX,
                        visible: true,
                        color: atlasColors[atlas] || '#6b7280',
                        thickness: 1.5,
                        width: 0,
                    },
                    text: text,
                    hovertemplate: '%{text}<extra></extra>',
                    legendgroup: atlas,
                });
            }
        });

        // Pooled effects (diamonds)
        const pooledX = [];
        const pooledY = [];
        const pooledErrorLow = [];
        const pooledErrorHigh = [];
        const pooledText = [];

        displayData.forEach((sig, sigIdx) => {
            if (sig.pooled_effect !== null) {
                pooledX.push(sig.pooled_effect);
                pooledY.push(yPositions[sigIdx]);
                pooledErrorLow.push(sig.pooled_effect - sig.ci_low);
                pooledErrorHigh.push(sig.ci_high - sig.pooled_effect);
                pooledText.push(
                    `Pooled: ${sig.pooled_effect.toFixed(3)} [${sig.ci_low.toFixed(3)}, ${sig.ci_high.toFixed(3)}]<br>I²: ${(sig.I2 || 0).toFixed(0)}%`
                );
            }
        });

        traces.push({
            type: 'scatter',
            mode: 'markers',
            name: 'Pooled',
            x: pooledX,
            y: pooledY,
            marker: {
                color: '#1e293b',
                size: 12,
                symbol: 'diamond',
            },
            error_x: {
                type: 'data',
                array: pooledErrorHigh,
                arrayminus: pooledErrorLow,
                visible: true,
                color: '#1e293b',
                thickness: 2,
                width: 4,
            },
            text: pooledText,
            hovertemplate: '%{text}<extra></extra>',
        });

        // Reference line at 0
        traces.push({
            type: 'scatter',
            mode: 'lines',
            x: [0, 0],
            y: [0, displayData.length + 1],
            line: {
                color: '#94a3b8',
                width: 1,
                dash: 'dash',
            },
            showlegend: false,
            hoverinfo: 'skip',
        });

        const layout = {
            title: options.title || 'Meta-Analysis Forest Plot',
            font: {
                family: 'Inter, sans-serif',
                size: 12,
            },
            xaxis: {
                title: options.xLabel || 'Effect Size (Correlation)',
                zeroline: true,
                zerolinecolor: '#94a3b8',
                zerolinewidth: 1,
                gridcolor: '#e2e8f0',
            },
            yaxis: {
                title: '',
                tickmode: 'array',
                tickvals: yPositions,
                ticktext: displayData.map(d => d.signature),
                gridcolor: '#e2e8f0',
                autorange: true,
            },
            legend: {
                orientation: 'h',
                y: -0.15,
                x: 0.5,
                xanchor: 'center',
            },
            margin: { l: 120, r: 40, t: 60, b: 80 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            hovermode: 'closest',
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d', 'autoScale2d'],
        };

        Plotly.newPlot(containerId, traces, layout, config);
    },

    /**
     * Create an I² heterogeneity bar chart
     * @param {string} containerId - DOM element ID
     * @param {Array} data - Array of signature objects with I2 values
     * @param {Object} options - Configuration options
     */
    createHeterogeneityChart(containerId, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container || !data || data.length === 0) return;

        // Filter to signatures with valid I2 and sort by I2
        const filteredData = data
            .filter(d => d.I2 !== null && d.I2 !== undefined)
            .sort((a, b) => b.I2 - a.I2)
            .slice(0, options.maxSignatures || 30);

        const colors = filteredData.map(d => {
            const i2 = d.I2 || 0;  // Already a percentage (0-100)
            if (i2 < 25) return '#10b981'; // Low heterogeneity - green
            if (i2 < 50) return '#f59e0b'; // Moderate - yellow
            if (i2 < 75) return '#f97316'; // Substantial - orange
            return '#ef4444'; // Considerable - red
        });

        const trace = {
            type: 'bar',
            orientation: 'h',
            x: filteredData.map(d => d.I2 || 0),  // Already percentage
            y: filteredData.map(d => d.signature),
            marker: { color: colors },
            text: filteredData.map(d => `${(d.I2 || 0).toFixed(0)}%`),
            textposition: 'outside',
            hovertemplate: '%{y}: I² = %{x:.1f}%<extra></extra>',
        };

        const layout = {
            title: options.title || 'Heterogeneity (I²)',
            xaxis: {
                title: 'I² (%)',
                range: [0, 105],
                gridcolor: '#e2e8f0',
            },
            yaxis: {
                title: '',
                automargin: true,
            },
            margin: { l: 120, r: 40, t: 60, b: 60 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            shapes: [
                // Reference lines for heterogeneity thresholds
                { type: 'line', x0: 25, x1: 25, y0: -0.5, y1: filteredData.length - 0.5, line: { dash: 'dot', color: '#94a3b8', width: 1 } },
                { type: 'line', x0: 50, x1: 50, y0: -0.5, y1: filteredData.length - 0.5, line: { dash: 'dot', color: '#94a3b8', width: 1 } },
                { type: 'line', x0: 75, x1: 75, y0: -0.5, y1: filteredData.length - 0.5, line: { dash: 'dot', color: '#94a3b8', width: 1 } },
            ],
        };

        Plotly.newPlot(containerId, [trace], layout, { responsive: true });
    },
};

// Make available globally
window.ForestPlot = ForestPlot;
