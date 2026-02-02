/**
 * Explore Page Handler
 * Browse and filter atlases
 */

const ExplorePage = {
    atlases: [],
    filteredAtlases: [],

    /**
     * Initialize the explore page
     */
    async init(params, query) {
        // Render template
        this.render();

        // Load atlases
        await this.loadAtlases();

        // Apply search query if present
        if (query.q) {
            this.applySearch(query.q);
        }

        // Set up filter handlers
        this.setupFilters();
    },

    /**
     * Render the explore page template
     */
    render() {
        const app = document.getElementById('app');
        const template = document.getElementById('explore-template');

        if (app && template) {
            app.innerHTML = template.innerHTML;
        }
    },

    /**
     * Load atlas data
     */
    async loadAtlases() {
        const container = document.getElementById('atlas-list');
        if (!container) return;

        try {
            this.atlases = await API.getAtlases();
            this.filteredAtlases = [...this.atlases];
            this.renderAtlases();
        } catch (error) {
            console.error('Failed to load atlases:', error);
            // Show placeholder atlases
            this.atlases = this.getPlaceholderAtlases();
            this.filteredAtlases = [...this.atlases];
            this.renderAtlases();
        }
    },

    /**
     * Get placeholder atlases when API unavailable
     */
    getPlaceholderAtlases() {
        return [
            {
                name: 'cima',
                display_name: 'CIMA',
                description: 'Chinese Immune Multi-omics Atlas',
                n_cells: 6484974,
                n_samples: 421,
                n_cell_types: 39,
                source_type: 'builtin',
                validation_grade: 'A',
                tissues: ['PBMC'],
            },
            {
                name: 'inflammation',
                display_name: 'Inflammation Atlas',
                description: 'Pan-disease inflammatory conditions',
                n_cells: 4900000,
                n_samples: 817,
                n_cell_types: 43,
                source_type: 'builtin',
                validation_grade: 'B',
                tissues: ['PBMC'],
            },
            {
                name: 'scatlas',
                display_name: 'scAtlas',
                description: 'Human tissue reference atlas',
                n_cells: 6400000,
                n_samples: 781,  // 317 normal + 464 cancer donors
                n_cell_types: 213,
                source_type: 'builtin',
                validation_grade: 'B',
                tissues: ['tissue'],
            },
        ];
    },

    /**
     * Render atlas cards
     */
    renderAtlases() {
        const container = document.getElementById('atlas-list');
        if (!container) return;

        if (this.filteredAtlases.length === 0) {
            container.innerHTML = '<p class="loading">No atlases match your filters</p>';
            return;
        }

        AtlasCard.renderInto('atlas-list', this.filteredAtlases);
    },

    /**
     * Set up filter handlers
     */
    setupFilters() {
        const typeFilter = document.getElementById('filter-type');
        const tissueFilter = document.getElementById('filter-tissue');
        const sortBy = document.getElementById('sort-by');

        if (typeFilter) {
            typeFilter.addEventListener('change', () => this.applyFilters());
        }
        if (tissueFilter) {
            tissueFilter.addEventListener('change', () => this.applyFilters());
        }
        if (sortBy) {
            sortBy.addEventListener('change', () => this.applyFilters());
        }
    },

    /**
     * Apply filters and sorting
     */
    applyFilters() {
        const typeFilter = document.getElementById('filter-type')?.value || 'all';
        const tissueFilter = document.getElementById('filter-tissue')?.value || 'all';
        const sortBy = document.getElementById('sort-by')?.value || 'cells';

        // Filter
        this.filteredAtlases = this.atlases.filter(atlas => {
            // Type filter
            if (typeFilter !== 'all') {
                if (typeFilter === 'builtin' && atlas.source_type !== 'builtin') return false;
                if (typeFilter === 'published' && atlas.source_type !== 'published') return false;
                if (typeFilter === 'user' && atlas.source_type !== 'user') return false;
            }

            // Tissue filter
            if (tissueFilter !== 'all' && atlas.tissues) {
                const hasMatchingTissue = atlas.tissues.some(t =>
                    t.toLowerCase().includes(tissueFilter.toLowerCase())
                );
                if (!hasMatchingTissue) return false;
            }

            return true;
        });

        // Sort
        this.filteredAtlases.sort((a, b) => {
            switch (sortBy) {
                case 'cells':
                    return (b.n_cells || 0) - (a.n_cells || 0);
                case 'name':
                    return (a.display_name || a.name).localeCompare(b.display_name || b.name);
                case 'grade':
                    const gradeOrder = { 'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4 };
                    return (gradeOrder[a.validation_grade] || 5) - (gradeOrder[b.validation_grade] || 5);
                default:
                    return 0;
            }
        });

        this.renderAtlases();
    },

    /**
     * Apply search query
     * @param {string} query - Search query
     */
    applySearch(query) {
        if (!query || !query.trim()) {
            this.filteredAtlases = [...this.atlases];
            this.renderAtlases();
            return;
        }

        const searchTerms = query.toLowerCase().split(/\s+/);

        this.filteredAtlases = this.atlases.filter(atlas => {
            const searchText = [
                atlas.name,
                atlas.display_name,
                atlas.description,
                ...(atlas.diseases || []),
                ...(atlas.tissues || []),
                ...(atlas.cell_types || []),
            ].join(' ').toLowerCase();

            return searchTerms.every(term => searchText.includes(term));
        });

        this.renderAtlases();
    },
};

// Make available globally
window.ExplorePage = ExplorePage;
