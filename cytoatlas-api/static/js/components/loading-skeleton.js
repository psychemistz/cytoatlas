/**
 * LoadingSkeleton Component
 * Simple skeleton loading animation
 */

const LoadingSkeleton = {
    /**
     * Show loading skeleton in a container
     * @param {string} containerId - Container element ID
     * @param {Object} options - Options (lines, height, etc.)
     */
    show(containerId, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Container ${containerId} not found`);
            return;
        }

        const lines = options.lines || 3;
        const height = options.height || '20px';
        const spacing = options.spacing || '10px';

        const skeletonHTML = `
            <div class="loading-skeleton" data-skeleton="true">
                ${Array(lines).fill(0).map((_, i) => `
                    <div class="skeleton-line" style="height: ${height}; margin-bottom: ${spacing};"></div>
                `).join('')}
                ${options.message ? `<p class="loading-text">${options.message}</p>` : ''}
            </div>
        `;

        container.innerHTML = skeletonHTML;
    },

    /**
     * Show spinner loading indicator
     * @param {string} containerId - Container element ID
     * @param {string} message - Loading message
     */
    showSpinner(containerId, message = 'Loading...') {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Container ${containerId} not found`);
            return;
        }

        container.innerHTML = `
            <div class="loading" data-skeleton="true">
                <div class="spinner"></div>
                <p>${message}</p>
            </div>
        `;
    },

    /**
     * Hide loading skeleton
     * @param {string} containerId - Container element ID
     */
    hide(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const skeleton = container.querySelector('[data-skeleton="true"]');
        if (skeleton) {
            skeleton.remove();
        }
    },

    /**
     * Show error state
     * @param {string} containerId - Container element ID
     * @param {string} message - Error message
     */
    showError(containerId, message = 'Failed to load data') {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = `
            <div class="error-container" data-skeleton="true">
                <p class="error-text">${message}</p>
                <button class="btn btn-secondary" onclick="window.location.reload()">Reload Page</button>
            </div>
        `;
    },

    /**
     * Show empty state
     * @param {string} containerId - Container element ID
     * @param {string} message - Empty state message
     */
    showEmpty(containerId, message = 'No data available') {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = `
            <div class="empty-state" data-skeleton="true">
                <p class="empty-text">${message}</p>
            </div>
        `;
    },
};

// Make available globally
window.LoadingSkeleton = LoadingSkeleton;
