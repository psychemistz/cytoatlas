/**
 * Search Page Handler
 * Gene/Cytokine/Protein-centric search and discovery
 */

const SearchPage = {
    searchTypes: [],
    currentResults: [],
    currentQuery: '',
    currentType: 'all',
    offset: 0,
    limit: 20,
    debounceTimer: null,

    /**
     * Initialize the search page
     */
    async init(params, query) {
        this.render();

        // Load search types
        await this.loadSearchTypes();

        // Check for initial query
        if (query.q) {
            this.currentQuery = query.q;
            document.getElementById('search-input').value = query.q;
            await this.performSearch();
        }

        if (query.type) {
            this.currentType = query.type;
            document.getElementById('type-filter').value = query.type;
        }

        this.setupEventHandlers();
    },

    /**
     * Render the search page template
     */
    render() {
        const app = document.getElementById('app');
        app.innerHTML = `
            <div class="search-page">
                <header class="search-header">
                    <h1>Search CytoAtlas</h1>
                    <p class="subtitle">
                        Search across 17+ million immune cells for cytokines, secreted proteins,
                        cell types, diseases, and organs.
                    </p>
                </header>

                <div class="search-controls">
                    <div class="search-bar">
                        <input type="text"
                               id="search-input"
                               placeholder="Search for IFNG, CD8 T cells, TNF, liver..."
                               autocomplete="off">
                        <button id="search-btn" class="btn btn-primary">
                            Search
                        </button>
                    </div>

                    <div class="search-filters">
                        <label>
                            Type:
                            <select id="type-filter">
                                <option value="all">All Types</option>
                                <option value="cytokine">Cytokines (CytoSig)</option>
                                <option value="protein">Secreted Proteins (SecAct)</option>
                                <option value="cell_type">Cell Types</option>
                                <option value="disease">Diseases</option>
                                <option value="organ">Organs</option>
                            </select>
                        </label>
                    </div>
                </div>

                <div id="autocomplete-dropdown" class="autocomplete-dropdown hidden"></div>

                <div id="search-stats" class="search-stats"></div>

                <div id="search-results" class="search-results">
                    <div class="search-intro">
                        <h3>Popular Signatures</h3>
                        <div class="popular-signatures">
                            <a href="/gene/IFNG" class="signature-link">
                                <span class="sig-name">IFNG</span>
                                <span class="sig-type cytosig">CytoSig</span>
                            </a>
                            <a href="/gene/TNF" class="signature-link">
                                <span class="sig-name">TNF</span>
                                <span class="sig-type cytosig">CytoSig</span>
                            </a>
                            <a href="/gene/IL6" class="signature-link">
                                <span class="sig-name">IL6</span>
                                <span class="sig-type cytosig">CytoSig</span>
                            </a>
                            <a href="/gene/IL17A" class="signature-link">
                                <span class="sig-name">IL17A</span>
                                <span class="sig-type cytosig">CytoSig</span>
                            </a>
                            <a href="/gene/TGFB1" class="signature-link">
                                <span class="sig-name">TGFB1</span>
                                <span class="sig-type cytosig">CytoSig</span>
                            </a>
                            <a href="/gene/CCL2?type=SecAct" class="signature-link">
                                <span class="sig-name">CCL2</span>
                                <span class="sig-type secact">SecAct</span>
                            </a>
                        </div>

                        <h3>Quick Search</h3>
                        <div class="example-chips">
                            <button class="chip" data-query="IFNG">IFNG</button>
                            <button class="chip" data-query="TNF">TNF</button>
                            <button class="chip" data-query="IL-17">IL-17</button>
                            <button class="chip" data-query="CD8 T cell">CD8 T cell</button>
                            <button class="chip" data-query="liver">liver</button>
                            <button class="chip" data-query="COVID-19">COVID-19</button>
                        </div>

                        <h3>Browse by Type</h3>
                        <div class="search-type-cards" id="search-type-cards"></div>
                    </div>
                </div>

                <div id="load-more-container" class="load-more-container hidden">
                    <button id="load-more-btn" class="btn btn-secondary">
                        Load More Results
                    </button>
                </div>
            </div>
        `;
    },

    /**
     * Load available search types
     */
    async loadSearchTypes() {
        try {
            const response = await fetch('/api/v1/search/types');
            this.searchTypes = await response.json();
            this.renderSearchTypeCards();
        } catch (error) {
            console.error('Failed to load search types:', error);
        }
    },

    /**
     * Render search type cards
     */
    renderSearchTypeCards() {
        const container = document.getElementById('search-type-cards');
        if (!container || !this.searchTypes.length) return;

        container.innerHTML = this.searchTypes.map(type => `
            <div class="search-type-card" data-type="${type.type}">
                <div class="type-icon">${this.getTypeIcon(type.type)}</div>
                <h4>${type.name}</h4>
                <p>${type.description}</p>
                ${type.count ? `<span class="count">${type.count.toLocaleString()} items</span>` : ''}
            </div>
        `).join('');
    },

    /**
     * Get icon for search type
     */
    getTypeIcon(type) {
        const icons = {
            gene: '🧬',
            cytokine: '💠',
            protein: '⚪',
            cell_type: '🔬',
            disease: '🩺',
            organ: '❤️',
        };
        return icons[type] || '📋';
    },

    /**
     * Set up event handlers
     */
    setupEventHandlers() {
        const searchInput = document.getElementById('search-input');
        const searchBtn = document.getElementById('search-btn');
        const typeFilter = document.getElementById('type-filter');
        const loadMoreBtn = document.getElementById('load-more-btn');

        // Search input with debounced autocomplete
        searchInput?.addEventListener('input', (e) => {
            clearTimeout(this.debounceTimer);
            this.debounceTimer = setTimeout(() => {
                this.handleAutocomplete(e.target.value);
            }, 200);
        });

        // Enter key to search
        searchInput?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                this.hideAutocomplete();
                this.currentQuery = searchInput.value;
                this.offset = 0;
                this.performSearch();
            } else if (e.key === 'Escape') {
                this.hideAutocomplete();
            }
        });

        // Search button
        searchBtn?.addEventListener('click', () => {
            this.hideAutocomplete();
            this.currentQuery = searchInput.value;
            this.offset = 0;
            this.performSearch();
        });

        // Type filter
        typeFilter?.addEventListener('change', (e) => {
            this.currentType = e.target.value;
            if (this.currentQuery) {
                this.offset = 0;
                this.performSearch();
            }
        });

        // Load more
        loadMoreBtn?.addEventListener('click', () => {
            this.loadMore();
        });

        // Example chips
        document.querySelectorAll('.chip[data-query]').forEach(chip => {
            chip.addEventListener('click', () => {
                const query = chip.dataset.query;
                searchInput.value = query;
                this.currentQuery = query;
                this.offset = 0;
                this.performSearch();
            });
        });

        // Search type cards
        document.querySelectorAll('.search-type-card[data-type]').forEach(card => {
            card.addEventListener('click', () => {
                this.currentType = card.dataset.type;
                typeFilter.value = card.dataset.type;
                searchInput.focus();
            });
        });

        // Hide autocomplete on outside click
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.search-bar') && !e.target.closest('.autocomplete-dropdown')) {
                this.hideAutocomplete();
            }
        });
    },

    /**
     * Handle autocomplete
     */
    async handleAutocomplete(query) {
        if (!query || query.length < 2) {
            this.hideAutocomplete();
            return;
        }

        try {
            const response = await fetch(`/api/v1/search/autocomplete?q=${encodeURIComponent(query)}&limit=8`);
            const data = await response.json();
            this.showAutocomplete(data.suggestions);
        } catch (error) {
            console.error('Autocomplete error:', error);
            this.hideAutocomplete();
        }
    },

    /**
     * Show autocomplete dropdown
     */
    showAutocomplete(suggestions) {
        const dropdown = document.getElementById('autocomplete-dropdown');
        if (!dropdown || !suggestions.length) {
            this.hideAutocomplete();
            return;
        }

        dropdown.innerHTML = suggestions.map(s => `
            <div class="autocomplete-item" data-text="${s.text}" data-type="${s.type}">
                <span class="type-badge ${s.type}">${s.type}</span>
                <span class="text">${s.highlight}</span>
                ${(s.type === 'cytokine' || s.type === 'protein') ?
                    `<a href="/gene/${encodeURIComponent(s.text)}?type=${s.type === 'protein' ? 'SecAct' : 'CytoSig'}"
                        class="autocomplete-action"
                        onclick="event.stopPropagation()">View →</a>` : ''}
            </div>
        `).join('');

        dropdown.classList.remove('hidden');

        // Add click handlers
        dropdown.querySelectorAll('.autocomplete-item').forEach(item => {
            item.addEventListener('click', (e) => {
                // Don't trigger if clicking the action link
                if (e.target.classList.contains('autocomplete-action')) return;

                const text = item.dataset.text;
                document.getElementById('search-input').value = text;
                this.currentQuery = text;
                this.hideAutocomplete();
                this.offset = 0;
                this.performSearch();
            });
        });
    },

    /**
     * Hide autocomplete dropdown
     */
    hideAutocomplete() {
        const dropdown = document.getElementById('autocomplete-dropdown');
        dropdown?.classList.add('hidden');
    },

    /**
     * Perform search
     */
    async performSearch() {
        if (!this.currentQuery) return;

        const resultsContainer = document.getElementById('search-results');
        const statsContainer = document.getElementById('search-stats');
        const loadMoreContainer = document.getElementById('load-more-container');

        // Show loading state
        if (this.offset === 0) {
            resultsContainer.innerHTML = '<div class="loading"><div class="spinner"></div>Searching...</div>';
        }

        try {
            const params = new URLSearchParams({
                q: this.currentQuery,
                type: this.currentType,
                offset: this.offset,
                limit: this.limit,
            });

            const response = await fetch(`/api/v1/search?${params}`);
            const data = await response.json();

            // Update stats
            statsContainer.innerHTML = `
                Found <strong>${data.total_results.toLocaleString()}</strong> results
                for "<strong>${data.query}</strong>"
                ${data.type_filter !== 'all' ? `in <strong>${data.type_filter}</strong>` : ''}
            `;

            // Render results
            if (this.offset === 0) {
                this.currentResults = data.results;
            } else {
                this.currentResults = [...this.currentResults, ...data.results];
            }

            this.renderResults();

            // Show/hide load more
            if (data.has_more) {
                loadMoreContainer.classList.remove('hidden');
            } else {
                loadMoreContainer.classList.add('hidden');
            }

            // Update URL
            const url = new URL(window.location);
            url.searchParams.set('q', this.currentQuery);
            url.searchParams.set('type', this.currentType);
            window.history.replaceState({}, '', url);

        } catch (error) {
            console.error('Search error:', error);
            resultsContainer.innerHTML = `
                <div class="error-message">
                    <h3>Search Failed</h3>
                    <p>Unable to search. Please try again.</p>
                </div>
            `;
        }
    },

    /**
     * Load more results
     */
    async loadMore() {
        this.offset += this.limit;
        await this.performSearch();
    },

    /**
     * Render search results
     */
    renderResults() {
        const container = document.getElementById('search-results');

        if (!this.currentResults.length) {
            container.innerHTML = `
                <div class="no-results">
                    <h3>No results found</h3>
                    <p>Try different search terms or remove filters.</p>
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <div class="results-list">
                ${this.currentResults.map(result => this.renderResultCard(result)).join('')}
            </div>
        `;

        // Use event delegation for all result card actions
        container.addEventListener('click', (e) => this.handleResultClick(e));
    },

    /**
     * Handle clicks on result cards using event delegation
     */
    handleResultClick(e) {
        const card = e.target.closest('.result-card');
        if (!card) return;

        // Check if clicking a specific action button
        const viewDetailsBtn = e.target.closest('.view-details-btn');
        const quickViewBtn = e.target.closest('.quick-view-btn');
        const exploreBtn = e.target.closest('.explore-btn');

        if (viewDetailsBtn) {
            e.preventDefault();
            const name = viewDetailsBtn.dataset.name;
            const sigType = viewDetailsBtn.dataset.sigtype;
            router.navigate(`/gene/${encodeURIComponent(name)}?type=${sigType}`);
            return;
        }

        if (quickViewBtn) {
            e.preventDefault();
            const entityId = card.dataset.entityId;
            this.showQuickView(entityId);
            return;
        }

        if (exploreBtn) {
            e.preventDefault();
            const atlas = exploreBtn.dataset.atlas;
            router.navigate(`/atlas/${atlas}`);
            return;
        }

        // Default: clicking card opens quick view
        if (!e.target.closest('a') && !e.target.closest('button')) {
            const entityId = card.dataset.entityId;
            this.showQuickView(entityId);
        }
    },

    /**
     * Render individual result card
     */
    renderResultCard(result) {
        const isSignature = result.type === 'cytokine' || result.type === 'protein' || result.type === 'gene';

        // For signatures, show both CytoSig and SecAct options
        let signatureLinks = '';
        if (isSignature) {
            signatureLinks = `
                <div class="signature-links">
                    <a href="/gene/${encodeURIComponent(result.name)}?type=CytoSig"
                       class="sig-link cytosig"
                       onclick="event.stopPropagation()">
                        <span class="sig-icon">&#128300;</span>
                        <span class="sig-label">CytoSig</span>
                    </a>
                    <a href="/gene/${encodeURIComponent(result.name)}?type=SecAct"
                       class="sig-link secact"
                       onclick="event.stopPropagation()">
                        <span class="sig-icon">&#9898;</span>
                        <span class="sig-label">SecAct</span>
                    </a>
                </div>
            `;
        }

        return `
            <div class="result-card" data-entity-id="${result.id}" data-type="${result.type}">
                <div class="result-header">
                    <span class="type-badge ${result.type}">${this.formatTypeName(result.type)}</span>
                    <h3 class="result-name">${result.name}</h3>
                </div>
                ${result.description ? `<p class="result-description">${result.description}</p>` : ''}
                <div class="result-meta">
                    <span class="atlas-badges">
                        ${result.atlases.map(a => `<span class="atlas-badge">${a}</span>`).join('')}
                    </span>
                </div>
                ${isSignature ? signatureLinks : ''}
                <div class="result-actions">
                    <button class="btn btn-sm btn-outline quick-view-btn">
                        Quick View
                    </button>
                    ${!isSignature && result.atlases.length === 1 ? `
                        <button class="btn btn-sm btn-outline explore-btn" data-atlas="${result.atlases[0].toLowerCase()}">
                            Explore in ${result.atlases[0]}
                        </button>
                    ` : ''}
                    ${result.type === 'disease' ? `
                        <a href="/atlas/inflammation" class="btn btn-sm btn-outline" onclick="event.stopPropagation()">
                            Inflammation Atlas
                        </a>
                    ` : ''}
                    ${result.type === 'organ' ? `
                        <a href="/atlas/scatlas" class="btn btn-sm btn-outline" onclick="event.stopPropagation()">
                            scAtlas
                        </a>
                    ` : ''}
                </div>
            </div>
        `;
    },

    /**
     * Format type name for display
     */
    formatTypeName(type) {
        const names = {
            cytokine: 'Cytokine',
            protein: 'Protein',
            cell_type: 'Cell Type',
            disease: 'Disease',
            organ: 'Organ',
            gene: 'Gene',
        };
        return names[type] || type;
    },

    /**
     * Show quick view panel for an entity
     */
    async showQuickView(entityId) {
        // Parse entity type and name from ID (format: "type:name")
        const [entityType, entityName] = entityId.includes(':') ? entityId.split(':') : ['unknown', entityId];

        try {
            const response = await fetch(`/api/v1/search/${encodeURIComponent(entityId)}/activity`);

            if (!response.ok) {
                throw new Error('Failed to load activity data');
            }

            const data = await response.json();
            this.renderQuickViewModal(data);

        } catch (error) {
            console.error('Failed to load entity detail:', error);
            // Show a simplified modal for entities without activity data
            this.renderSimpleInfoModal(entityType, entityName);
        }
    },

    /**
     * Render quick view modal with activity data
     */
    renderQuickViewModal(data) {
        // Remove any existing modal
        document.querySelector('.search-modal')?.remove();

        const isSignature = data.entity_type === 'cytokine' || data.entity_type === 'protein';
        const sigType = data.entity_type === 'protein' ? 'SecAct' : 'CytoSig';

        const modal = document.createElement('div');
        modal.className = 'search-modal';
        modal.innerHTML = `
            <div class="modal-backdrop"></div>
            <div class="modal-content">
                <div class="modal-header">
                    <div class="modal-title">
                        <h2>${data.entity_name}</h2>
                        <span class="type-badge ${data.entity_type}">${this.formatTypeName(data.entity_type)}</span>
                    </div>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="activity-summary">
                        <h3>Activity Summary</h3>
                        <div class="stats-row">
                            <div class="stat-item">
                                <span class="stat-value">${data.mean_activity?.toFixed(3) || 'N/A'}</span>
                                <span class="stat-label">Mean</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-value">${data.std_activity?.toFixed(3) || 'N/A'}</span>
                                <span class="stat-label">Std Dev</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-value">${data.min_activity?.toFixed(3) || 'N/A'}</span>
                                <span class="stat-label">Min</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-value">${data.max_activity?.toFixed(3) || 'N/A'}</span>
                                <span class="stat-label">Max</span>
                            </div>
                        </div>
                    </div>

                    ${data.top_positive_cell_types?.length ? `
                        <div class="cell-type-section">
                            <h3>Highest Activity</h3>
                            <div class="cell-type-list">
                                ${data.top_positive_cell_types.slice(0, 5).map(ct => `
                                    <div class="cell-type-row">
                                        <span class="cell-name">${ct.cell_type}</span>
                                        <span class="cell-atlas">${ct.atlas}</span>
                                        <span class="cell-activity positive">${ct.activity.toFixed(3)}</span>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    ` : ''}

                    ${data.top_negative_cell_types?.length ? `
                        <div class="cell-type-section">
                            <h3>Lowest Activity</h3>
                            <div class="cell-type-list">
                                ${data.top_negative_cell_types.slice(0, 5).map(ct => `
                                    <div class="cell-type-row">
                                        <span class="cell-name">${ct.cell_type}</span>
                                        <span class="cell-atlas">${ct.atlas}</span>
                                        <span class="cell-activity negative">${ct.activity.toFixed(3)}</span>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    ` : ''}
                </div>
                <div class="modal-footer">
                    ${isSignature ? `
                        <div class="modal-sig-links">
                            <a href="/gene/${encodeURIComponent(data.entity_name)}?type=CytoSig" class="btn btn-primary">
                                View CytoSig
                            </a>
                            <a href="/gene/${encodeURIComponent(data.entity_name)}?type=SecAct" class="btn btn-outline">
                                View SecAct
                            </a>
                        </div>
                    ` : ''}
                    <button class="btn btn-secondary close-modal-btn">Close</button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Close handlers
        modal.querySelector('.modal-close').addEventListener('click', () => modal.remove());
        modal.querySelector('.modal-backdrop').addEventListener('click', () => modal.remove());
        modal.querySelector('.close-modal-btn')?.addEventListener('click', () => modal.remove());

        // Close on escape
        const escHandler = (e) => {
            if (e.key === 'Escape') {
                modal.remove();
                document.removeEventListener('keydown', escHandler);
            }
        };
        document.addEventListener('keydown', escHandler);
    },

    /**
     * Render simple info modal for entities without activity data
     */
    renderSimpleInfoModal(type, name) {
        document.querySelector('.search-modal')?.remove();

        const modal = document.createElement('div');
        modal.className = 'search-modal';
        modal.innerHTML = `
            <div class="modal-backdrop"></div>
            <div class="modal-content modal-small">
                <div class="modal-header">
                    <div class="modal-title">
                        <h2>${name}</h2>
                        <span class="type-badge ${type}">${this.formatTypeName(type)}</span>
                    </div>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <p class="info-text">
                        ${this.getEntityDescription(type, name)}
                    </p>
                </div>
                <div class="modal-footer">
                    ${type === 'disease' ? `
                        <a href="/atlas/inflammation" class="btn btn-primary">View in Inflammation Atlas</a>
                    ` : type === 'organ' ? `
                        <a href="/atlas/scatlas" class="btn btn-primary">View in scAtlas</a>
                    ` : type === 'cell_type' ? `
                        <a href="/explore" class="btn btn-primary">Explore Cell Types</a>
                    ` : ''}
                    <button class="btn btn-secondary close-modal-btn">Close</button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        modal.querySelector('.modal-close').addEventListener('click', () => modal.remove());
        modal.querySelector('.modal-backdrop').addEventListener('click', () => modal.remove());
        modal.querySelector('.close-modal-btn')?.addEventListener('click', () => modal.remove());
    },

    /**
     * Get description for entity type
     */
    getEntityDescription(type, name) {
        const descriptions = {
            disease: `<strong>${name}</strong> is a disease condition in the Inflammation Atlas. View the atlas to explore cytokine activity patterns associated with this disease.`,
            organ: `<strong>${name}</strong> is an organ/tissue in scAtlas. View the atlas to explore cell type-specific cytokine activities in this tissue.`,
            cell_type: `<strong>${name}</strong> is an immune cell type. Explore the atlases to see cytokine activity patterns in this cell population.`,
        };
        return descriptions[type] || `<strong>${name}</strong> - ${type}`;
    },

    /**
     * Navigate to gene detail page (for modal button)
     */
    navigateToGeneDetail(name, type) {
        document.querySelector('.search-modal')?.remove();
        const signatureType = type === 'protein' ? 'SecAct' : 'CytoSig';
        router.navigate(`/gene/${encodeURIComponent(name)}?type=${signatureType}`);
    },
};

// Make available globally
window.SearchPage = SearchPage;
