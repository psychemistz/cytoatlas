/**
 * Chart Components Index
 * Exports all chart components for the CytoAtlas SPA
 */

// Note: Since the SPA uses global objects (not ES6 modules),
// individual chart files are loaded via script tags and
// expose themselves globally. This file documents the available charts.

/**
 * Available Chart Components:
 *
 * - HeatmapChart: Plotly heatmap with colorscale, annotations, clustering support
 * - BoxplotChart: Plotly boxplot with group comparison, jitter points
 * - ScatterChart: Plotly scatter with regression line, tooltips, color coding
 * - VolcanoChart: Plotly scatter configured for volcano plots (activity_diff vs -log10(p))
 * - BarChart: Plotly bar chart (horizontal/vertical), grouped/stacked
 * - LollipopChart: D3 lollipop chart for ranked signatures
 * - ViolinChart: Plotly violin plot with box overlay
 *
 * All charts follow the same pattern:
 *
 * const chart = new ChartName(containerId, options);
 * chart.render(data);
 * chart.update(data);
 * chart.resize();
 * chart.destroy();
 * chart.exportCSV();
 * chart.exportPNG();
 */

// Standardized Plotly configuration used across all chart components
const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    toImageButtonOptions: { format: 'png', filename: 'cytoatlas_chart', scale: 2 }
};

const PLOTLY_LAYOUT = {
    font: { family: 'Inter, system-ui, sans-serif', size: 12 },
    margin: { l: 80, r: 40, t: 40, b: 80 },
    paper_bgcolor: 'white',
    plot_bgcolor: 'white'
};

// Export for documentation purposes
window.ChartDefaults = {
    PLOTLY_CONFIG,
    PLOTLY_LAYOUT
};
