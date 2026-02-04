/**
 * Sankey Chart Component
 * Wrapper for Plotly Sankey diagrams
 */

const SankeyChart = {
    /**
     * Create a Sankey diagram
     * @param {string} containerId - DOM element ID for the chart
     * @param {Array} nodes - Array of node objects {name, category, color?}
     * @param {Array} links - Array of link objects {source, target, value, label?}
     * @param {Object} options - Configuration options
     */
    create(containerId, nodes, links, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Container ${containerId} not found`);
            return;
        }

        // Default color scheme for categories
        const categoryColors = options.categoryColors || {
            atlas: '#3b82f6',      // Blue
            lineage: '#10b981',    // Green
            cell_type: '#f59e0b',  // Amber
            default: '#6b7280',    // Gray
        };

        // Build node colors and labels
        const nodeColors = nodes.map(n => {
            if (n.color) return n.color;
            return categoryColors[n.category] || categoryColors.default;
        });

        const nodeLabels = nodes.map(n => n.name || n.label || '');

        // Build link colors (lighter versions of source node colors)
        const linkColors = links.map(l => {
            const sourceColor = nodeColors[l.source] || '#6b7280';
            // Make link color semi-transparent
            return sourceColor.replace(')', ', 0.4)').replace('rgb', 'rgba');
        });

        // Build link hover labels
        const linkLabels = links.map(l => {
            if (l.label) return l.label;
            const sourceName = nodeLabels[l.source] || 'Source';
            const targetName = nodeLabels[l.target] || 'Target';
            return `${sourceName} → ${targetName}: ${l.value.toLocaleString()}`;
        });

        const trace = {
            type: 'sankey',
            orientation: 'h',
            node: {
                pad: options.nodePad || 15,
                thickness: options.nodeThickness || 20,
                line: {
                    color: 'white',
                    width: 1,
                },
                label: nodeLabels,
                color: nodeColors,
                hovertemplate: '%{label}<br>Total: %{value:,.0f}<extra></extra>',
            },
            link: {
                source: links.map(l => l.source),
                target: links.map(l => l.target),
                value: links.map(l => l.value),
                color: linkColors.map(c => c.includes('rgba') ? c : `rgba(128, 128, 128, 0.3)`),
                customdata: links.map(l => l.n_types || ''),
                hovertemplate: '%{customdata} cell types<br>%{value:,.0f} cells<extra></extra>',
            },
        };

        const layout = {
            title: options.title || '',
            font: {
                family: 'Inter, sans-serif',
                size: 12,
                color: '#1e293b',
            },
            margin: options.margin || { l: 10, r: 10, t: 40, b: 10 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
        };

        const config = {
            responsive: true,
            displayModeBar: false,
        };

        Plotly.newPlot(containerId, [trace], layout, config);
    },

    /**
     * Create a cell type mapping Sankey diagram
     * @param {string} containerId - DOM element ID
     * @param {Object} data - Data from getCelltypeSankey API
     * @param {Object} options - Configuration options
     */
    createCelltypeMapping(containerId, data, options = {}) {
        if (!data || !data.nodes || !data.links) {
            const container = document.getElementById(containerId);
            if (container) {
                container.innerHTML = '<p class="loading">No cell type mapping data available</p>';
            }
            return;
        }

        // Assign colors based on node category
        const atlasColors = {
            'CIMA': '#3b82f6',
            'Inflammation': '#f59e0b',
            'scAtlas': '#10b981',
        };

        const lineageColors = {
            'T': '#ef4444',
            'B': '#8b5cf6',
            'NK': '#ec4899',
            'Myeloid': '#f97316',
            'DC': '#14b8a6',
            'Other': '#6b7280',
        };

        const nodes = data.nodes.map(n => ({
            ...n,
            color: atlasColors[n.name] || lineageColors[n.name] || '#6b7280',
        }));

        this.create(containerId, nodes, data.links, {
            ...options,
            title: options.title || 'Cell Type Harmonization Across Atlases',
        });
    },
};

// Make available globally
window.SankeyChart = SankeyChart;
