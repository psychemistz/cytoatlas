/**
 * FilterBar Component
 * Dynamic filter bar with toggles, dropdowns, and state sync
 */

class FilterBar {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container ${containerId} not found`);
        }
        this.containerId = containerId;
        this.options = { ...FilterBar.DEFAULTS, ...options };

        this.filters = [];
        this.values = {};
        this.callbacks = [];

        this.render();
    }

    static DEFAULTS = {
        syncState: true, // Sync with appState
        layout: 'horizontal', // 'horizontal' or 'vertical'
    };

    /**
     * Render the filter bar container
     */
    render() {
        this.container.innerHTML = `
            <div class="filter-bar ${this.options.layout}">
                <div class="filter-controls" id="${this.containerId}-controls"></div>
            </div>
        `;
        this.controlsContainer = document.getElementById(`${this.containerId}-controls`);
    }

    /**
     * Add a toggle filter (e.g., CytoSig/SecAct)
     * @param {string} id - Filter ID
     * @param {Array} options - Array of {value, label} objects
     * @param {string} defaultValue - Default selected value
     */
    addToggle(id, options, defaultValue = null) {
        const value = defaultValue || options[0].value;
        this.values[id] = value;

        this.filters.push({
            id,
            type: 'toggle',
            options,
            value,
        });

        this.renderFilters();
    }

    /**
     * Add a dropdown filter
     * @param {string} id - Filter ID
     * @param {string} label - Filter label
     * @param {Array} options - Array of option values or {value, label} objects
     * @param {string} defaultValue - Default selected value
     */
    addDropdown(id, label, options, defaultValue = null) {
        const value = defaultValue || (options[0]?.value || options[0]);
        this.values[id] = value;

        this.filters.push({
            id,
            type: 'dropdown',
            label,
            options,
            value,
        });

        this.renderFilters();
    }

    /**
     * Add a search input filter
     * @param {string} id - Filter ID
     * @param {string} placeholder - Placeholder text
     */
    addSearch(id, placeholder = 'Search...') {
        this.values[id] = '';

        this.filters.push({
            id,
            type: 'search',
            placeholder,
            value: '',
        });

        this.renderFilters();
    }

    /**
     * Render all filters
     */
    renderFilters() {
        this.controlsContainer.innerHTML = this.filters.map(filter => {
            switch (filter.type) {
                case 'toggle':
                    return this.renderToggle(filter);
                case 'dropdown':
                    return this.renderDropdown(filter);
                case 'search':
                    return this.renderSearch(filter);
                default:
                    return '';
            }
        }).join('');

        // Add event listeners
        this.attachEventListeners();
    }

    /**
     * Render a toggle filter
     */
    renderToggle(filter) {
        return `
            <div class="filter-item filter-toggle" data-filter-id="${filter.id}">
                <div class="toggle-group">
                    ${filter.options.map(opt => `
                        <button
                            class="toggle-btn ${opt.value === filter.value ? 'active' : ''}"
                            data-value="${opt.value}"
                        >
                            ${opt.label}
                        </button>
                    `).join('')}
                </div>
            </div>
        `;
    }

    /**
     * Render a dropdown filter
     */
    renderDropdown(filter) {
        const normalizedOptions = filter.options.map(opt =>
            typeof opt === 'string' ? { value: opt, label: opt } : opt
        );

        return `
            <div class="filter-item filter-dropdown" data-filter-id="${filter.id}">
                <label class="filter-label">${filter.label}:</label>
                <select class="filter-select">
                    ${normalizedOptions.map(opt => `
                        <option value="${opt.value}" ${opt.value === filter.value ? 'selected' : ''}>
                            ${opt.label}
                        </option>
                    `).join('')}
                </select>
            </div>
        `;
    }

    /**
     * Render a search filter
     */
    renderSearch(filter) {
        return `
            <div class="filter-item filter-search" data-filter-id="${filter.id}">
                <input
                    type="text"
                    class="filter-search-input"
                    placeholder="${filter.placeholder}"
                    value="${filter.value}"
                />
            </div>
        `;
    }

    /**
     * Attach event listeners to filters
     */
    attachEventListeners() {
        // Toggle buttons
        this.controlsContainer.querySelectorAll('.filter-toggle').forEach(toggle => {
            const filterId = toggle.dataset.filterId;
            toggle.querySelectorAll('.toggle-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    const value = btn.dataset.value;
                    this.setValue(filterId, value);
                });
            });
        });

        // Dropdowns
        this.controlsContainer.querySelectorAll('.filter-dropdown').forEach(dropdown => {
            const filterId = dropdown.dataset.filterId;
            const select = dropdown.querySelector('select');
            select.addEventListener('change', () => {
                this.setValue(filterId, select.value);
            });
        });

        // Search inputs
        this.controlsContainer.querySelectorAll('.filter-search').forEach(search => {
            const filterId = search.dataset.filterId;
            const input = search.querySelector('input');
            input.addEventListener('input', () => {
                this.setValue(filterId, input.value);
            });
        });
    }

    /**
     * Set a filter value
     * @param {string} id - Filter ID
     * @param {*} value - New value
     */
    setValue(id, value) {
        const oldValue = this.values[id];
        this.values[id] = value;

        // Update filter object
        const filter = this.filters.find(f => f.id === id);
        if (filter) {
            filter.value = value;
        }

        // Sync with appState if enabled
        if (this.options.syncState) {
            window.appState.set(id, value);
        }

        // Notify callbacks
        this.callbacks.forEach(callback => {
            try {
                callback(id, value, oldValue);
            } catch (error) {
                console.error('Error in filter callback:', error);
            }
        });

        // Re-render to update UI
        this.renderFilters();
    }

    /**
     * Get a filter value
     * @param {string} id - Filter ID
     * @returns {*} Filter value
     */
    getValue(id) {
        return this.values[id];
    }

    /**
     * Get all filter values
     * @returns {Object} All filter values
     */
    getValues() {
        return { ...this.values };
    }

    /**
     * Register onChange callback
     * @param {Function} callback - Callback function(filterId, value, oldValue)
     */
    onChange(callback) {
        this.callbacks.push(callback);
    }

    /**
     * Update dropdown options
     * @param {string} id - Filter ID
     * @param {Array} options - New options
     */
    updateOptions(id, options) {
        const filter = this.filters.find(f => f.id === id);
        if (filter && filter.type === 'dropdown') {
            filter.options = options;
            this.renderFilters();
        }
    }

    /**
     * Reset all filters to default values
     */
    reset() {
        this.filters.forEach(filter => {
            const defaultValue = filter.type === 'search' ? '' :
                filter.options[0]?.value || filter.options[0];
            this.setValue(filter.id, defaultValue);
        });
    }

    /**
     * Destroy the filter bar
     */
    destroy() {
        this.filters = [];
        this.values = {};
        this.callbacks = [];
        this.container.innerHTML = '';
    }
}

// Make available globally
window.FilterBar = FilterBar;
