/**
 * TabPanel Component
 * Lazy-loaded tab panel with content management
 */

class TabPanel {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container ${containerId} not found`);
        }
        this.containerId = containerId;
        this.options = { ...TabPanel.DEFAULTS, ...options };

        this.tabs = [];
        this.activeTab = null;
        this.loadedTabs = new Set();
        this.tabData = new Map();

        this.render();
    }

    static DEFAULTS = {
        orientation: 'horizontal', // 'horizontal' or 'vertical'
        defaultTab: 0,
    };

    /**
     * Add a tab with lazy loader function
     * @param {string} id - Tab ID
     * @param {string} label - Tab label
     * @param {Function} loaderFn - Async function to load tab content
     * @param {Object} options - Tab options (icon, etc.)
     */
    addTab(id, label, loaderFn, options = {}) {
        this.tabs.push({
            id,
            label,
            loaderFn,
            icon: options.icon || null,
            enabled: options.enabled !== false,
        });
    }

    /**
     * Render the tab panel structure
     */
    render() {
        this.container.innerHTML = `
            <div class="tab-panel ${this.options.orientation}">
                <div class="tab-headers" id="${this.containerId}-headers"></div>
                <div class="tab-content" id="${this.containerId}-content"></div>
            </div>
        `;

        this.headersContainer = document.getElementById(`${this.containerId}-headers`);
        this.contentContainer = document.getElementById(`${this.containerId}-content`);
    }

    /**
     * Render tab headers
     */
    renderHeaders() {
        if (!this.headersContainer) return;

        this.headersContainer.innerHTML = this.tabs.map(tab => `
            <button
                class="tab-header ${tab.id === this.activeTab ? 'active' : ''} ${!tab.enabled ? 'disabled' : ''}"
                data-tab-id="${tab.id}"
                ${!tab.enabled ? 'disabled' : ''}
            >
                ${tab.icon ? `<span class="tab-icon">${tab.icon}</span>` : ''}
                <span class="tab-label">${tab.label}</span>
            </button>
        `).join('');

        // Add click handlers
        this.headersContainer.querySelectorAll('.tab-header').forEach(header => {
            header.addEventListener('click', () => {
                const tabId = header.dataset.tabId;
                this.activateTab(tabId);
            });
        });
    }

    /**
     * Initialize the tab panel (call after adding all tabs)
     */
    init() {
        this.renderHeaders();

        // Activate default tab
        if (this.tabs.length > 0) {
            const defaultTab = this.tabs[this.options.defaultTab] || this.tabs[0];
            this.activateTab(defaultTab.id);
        }
    }

    /**
     * Activate a tab
     * @param {string} tabId - Tab ID to activate
     */
    async activateTab(tabId) {
        const tab = this.tabs.find(t => t.id === tabId);
        if (!tab || !tab.enabled) return;

        this.activeTab = tabId;

        // Update header states
        this.headersContainer.querySelectorAll('.tab-header').forEach(header => {
            header.classList.toggle('active', header.dataset.tabId === tabId);
        });

        // Load content if not already loaded
        if (!this.loadedTabs.has(tabId)) {
            await this.loadTabContent(tab);
        } else {
            // Show cached content
            this.showCachedContent(tabId);
        }
    }

    /**
     * Load tab content using loader function
     * @param {Object} tab - Tab object
     */
    async loadTabContent(tab) {
        // Show loading skeleton
        this.contentContainer.innerHTML = `
            <div class="loading-container">
                <div class="loading-skeleton">
                    <div class="skeleton-line"></div>
                    <div class="skeleton-line"></div>
                    <div class="skeleton-line"></div>
                </div>
                <p class="loading-text">Loading ${tab.label}...</p>
            </div>
        `;

        try {
            // Call loader function
            const content = await tab.loaderFn();

            // Store content
            this.tabData.set(tab.id, content);
            this.loadedTabs.add(tab.id);

            // Render content
            this.contentContainer.innerHTML = content;
        } catch (error) {
            console.error(`Failed to load tab ${tab.id}:`, error);
            this.contentContainer.innerHTML = `
                <div class="error-container">
                    <p class="error-text">Failed to load content: ${error.message}</p>
                    <button class="btn btn-secondary" onclick="window.location.reload()">Reload Page</button>
                </div>
            `;
        }
    }

    /**
     * Show cached content for a tab
     * @param {string} tabId - Tab ID
     */
    showCachedContent(tabId) {
        const content = this.tabData.get(tabId);
        if (content) {
            this.contentContainer.innerHTML = content;
        }
    }

    /**
     * Reload a specific tab (clear cache and reload)
     * @param {string} tabId - Tab ID
     */
    async reloadTab(tabId) {
        this.loadedTabs.delete(tabId);
        this.tabData.delete(tabId);

        if (this.activeTab === tabId) {
            await this.activateTab(tabId);
        }
    }

    /**
     * Get active tab ID
     * @returns {string} Active tab ID
     */
    getActiveTab() {
        return this.activeTab;
    }

    /**
     * Enable/disable a tab
     * @param {string} tabId - Tab ID
     * @param {boolean} enabled - Enable or disable
     */
    setTabEnabled(tabId, enabled) {
        const tab = this.tabs.find(t => t.id === tabId);
        if (tab) {
            tab.enabled = enabled;
            this.renderHeaders();
        }
    }

    /**
     * Destroy the tab panel
     */
    destroy() {
        this.tabs = [];
        this.loadedTabs.clear();
        this.tabData.clear();
        this.container.innerHTML = '';
    }
}

// Make available globally
window.TabPanel = TabPanel;
