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
                        <h3>Quick Search Examples</h3>
                        <div class="example-chips">
                            <button class="chip" data-query="IFNG">IFNG</button>
                            <button class="chip" data-query="TNF">TNF</button>
                            <button class="chip" data-query="IL-17">IL-17</button>
                            <button class="chip" data-query="CD8 T cell">CD8 T cell</button>
                            <button class="chip" data-query="liver">liver</button>
                            <button class="chip" data-query="COVID-19">COVID-19</button>
                        </div>

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
            </div>
        `).join('');

        dropdown.classList.remove('hidden');

        // Add click handlers
        dropdown.querySelectorAll('.autocomplete-item').forEach(item => {
            item.addEventListener('click', () => {
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
            resultsContainer.innerHTML = '<div class="loading">Searching...</div>';
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
                for "${data.query}"
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
                    Search failed. Please try again.
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

        // Add click handlers for result cards
        container.querySelectorAll('.result-card').forEach(card => {
            card.addEventListener('click', () => {
                const entityId = card.dataset.entityId;
                this.showEntityDetail(entityId);
            });
        });
    },

    /**
     * Render individual result card
     */
    renderResultCard(result) {
        return `
            <div class="result-card" data-entity-id="${result.id}">
                <div class="result-header">
                    <span class="type-badge ${result.type}">${result.type}</span>
                    <h3 class="result-name">${result.name}</h3>
                    <span class="score">${Math.round(result.score)}</span>
                </div>
                ${result.description ? `<p class="result-description">${result.description}</p>` : ''}
                <div class="result-meta">
                    <span class="atlas-count">
                        ${result.atlas_count} ${result.atlas_count === 1 ? 'atlas' : 'atlases'}
                    </span>
                    <span class="atlases">${result.atlases.join(', ')}</span>
                </div>
                <div class="result-actions">
                    <button class="btn btn-sm btn-outline view-activity">View Activity</button>
                    <button class="btn btn-sm btn-outline view-correlations">Correlations</button>
                </div>
            </div>
        `;
    },

    /**
     * Show entity detail modal/panel
     */
    async showEntityDetail(entityId) {
        // Load activity data
        try {
            const response = await fetch(`/api/v1/search/${encodeURIComponent(entityId)}/activity`);
            const data = await response.json();

            // Create modal
            const modal = document.createElement('div');
            modal.className = 'modal entity-detail-modal';
            modal.innerHTML = `
                <div class="modal-backdrop"></div>
                <div class="modal-content">
                    <div class="modal-header">
                        <h2>${data.entity_name}</h2>
                        <span class="type-badge ${data.entity_type}">${data.entity_type}</span>
                        <button class="modal-close">&times;</button>
                    </div>
                    <div class="modal-body">
                        <div class="activity-summary">
                            <h3>Activity Summary</h3>
                            <div class="stats-grid">
                                <div class="stat">
                                    <span class="label">Mean</span>
                                    <span class="value">${data.mean_activity.toFixed(3)}</span>
                                </div>
                                <div class="stat">
                                    <span class="label">Std Dev</span>
                                    <span class="value">${data.std_activity.toFixed(3)}</span>
                                </div>
                                <div class="stat">
                                    <span class="label">Min</span>
                                    <span class="value">${data.min_activity.toFixed(3)}</span>
                                </div>
                                <div class="stat">
                                    <span class="label">Max</span>
                                    <span class="value">${data.max_activity.toFixed(3)}</span>
                                </div>
                            </div>
                        </div>

                        <div class="top-cell-types">
                            <h3>Top Positive Activity</h3>
                            <div class="cell-type-list">
                                ${data.top_positive_cell_types.slice(0, 5).map(ct => `
                                    <div class="cell-type-item">
                                        <span class="cell-type-name">${ct.cell_type}</span>
                                        <span class="atlas">${ct.atlas}</span>
                                        <span class="activity positive">${ct.activity.toFixed(3)}</span>
                                    </div>
                                `).join('')}
                            </div>

                            <h3>Top Negative Activity</h3>
                            <div class="cell-type-list">
                                ${data.top_negative_cell_types.slice(0, 5).map(ct => `
                                    <div class="cell-type-item">
                                        <span class="cell-type-name">${ct.cell_type}</span>
                                        <span class="atlas">${ct.atlas}</span>
                                        <span class="activity negative">${ct.activity.toFixed(3)}</span>
                                    </div>
                                `).join('')}
                            </div>
                        </div>

                        <div class="modal-actions">
                            <button class="btn btn-primary" onclick="Router.navigate('/explore')">
                                Explore in Atlas
                            </button>
                            <button class="btn btn-secondary" onclick="Router.navigate('/compare')">
                                Compare Across Atlases
                            </button>
                        </div>
                    </div>
                </div>
            `;

            document.body.appendChild(modal);

            // Close handlers
            modal.querySelector('.modal-close').addEventListener('click', () => modal.remove());
            modal.querySelector('.modal-backdrop').addEventListener('click', () => modal.remove());

        } catch (error) {
            console.error('Failed to load entity detail:', error);
            alert('Failed to load entity details. Please try again.');
        }
    },
};

// Make available globally
window.SearchPage = SearchPage;
