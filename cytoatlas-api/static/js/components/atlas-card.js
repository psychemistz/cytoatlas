/**
 * Atlas Card Component
 * Renders an atlas card with summary information
 */

const AtlasCard = {
    /**
     * Render an atlas card
     * @param {Object} atlas - Atlas data
     * @returns {string} HTML string
     */
    render(atlas) {
        const gradeClass = atlas.validation_grade ? `grade-${atlas.validation_grade.toLowerCase()}` : '';

        return `
            <div class="atlas-card" data-atlas="${atlas.name}">
                <div class="atlas-card-header">
                    <div class="atlas-card-title">
                        <h3>${atlas.display_name || atlas.name}</h3>
                        ${atlas.source_type === 'builtin' ? '<span class="atlas-badge">Core</span>' : ''}
                        ${atlas.validation_grade ? `<span class="atlas-badge ${gradeClass}">Grade ${atlas.validation_grade}</span>` : ''}
                    </div>
                    <p class="atlas-card-desc">${atlas.description || ''}</p>
                </div>
                <div class="atlas-card-stats">
                    <div class="atlas-stat">
                        <div class="atlas-stat-value">${this.formatNumber(atlas.n_cells)}</div>
                        <div class="atlas-stat-label">Cells</div>
                    </div>
                    <div class="atlas-stat">
                        <div class="atlas-stat-value">${atlas.n_samples || '-'}</div>
                        <div class="atlas-stat-label">Samples</div>
                    </div>
                    <div class="atlas-stat">
                        <div class="atlas-stat-value">${atlas.n_cell_types || '-'}</div>
                        <div class="atlas-stat-label">Cell Types</div>
                    </div>
                </div>
                <div class="atlas-card-actions">
                    <a href="/atlas/${atlas.name}" class="btn btn-primary">Explore</a>
                    <a href="/validate?atlas=${atlas.name}" class="btn btn-secondary">Validate</a>
                </div>
            </div>
        `;
    },

    /**
     * Render multiple atlas cards
     * @param {Array} atlases - Array of atlas data
     * @returns {string} HTML string
     */
    renderList(atlases) {
        if (!atlases || atlases.length === 0) {
            return '<p class="loading">No atlases available</p>';
        }
        return atlases.map(atlas => this.render(atlas)).join('');
    },

    /**
     * Format large numbers (e.g., 6500000 -> "6.5M")
     * @param {number} num - Number to format
     * @returns {string} Formatted string
     */
    formatNumber(num) {
        if (!num) return '-';

        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        }
        if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    },

    /**
     * Render atlas cards into a container
     * @param {string} containerId - Container element ID
     * @param {Array} atlases - Array of atlas data
     */
    renderInto(containerId, atlases) {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = this.renderList(atlases);
        }
    },
};

// Make available globally
window.AtlasCard = AtlasCard;
