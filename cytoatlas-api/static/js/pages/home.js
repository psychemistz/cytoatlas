/**
 * Home Page Handler
 */

const HomePage = {
    /**
     * Initialize the home page
     */
    async init() {
        // Render template
        this.render();

        // Load atlas cards
        await this.loadAtlases();

        // Set up search handlers
        this.setupSearch();
    },

    /**
     * Render the home page template
     */
    render() {
        const app = document.getElementById('app');
        const template = document.getElementById('home-template');

        if (app && template) {
            app.innerHTML = template.innerHTML;
        }
    },

    /**
     * Load atlas data and render cards
     */
    async loadAtlases() {
        const container = document.getElementById('atlas-cards');
        if (!container) return;

        try {
            const atlases = await API.getAtlases();

            // Filter to core atlases for home page
            const coreAtlases = atlases.filter(a => a.source_type === 'builtin') || atlases.slice(0, 3);

            if (coreAtlases.length > 0) {
                AtlasCard.renderInto('atlas-cards', coreAtlases);
            } else {
                // Show placeholder cards if API not available
                container.innerHTML = this.getPlaceholderCards();
            }
        } catch (error) {
            console.error('Failed to load atlases:', error);
            // Show placeholder cards on error
            container.innerHTML = this.getPlaceholderCards();
        }
    },

    /**
     * Get placeholder atlas cards (when API unavailable)
     */
    getPlaceholderCards() {
        const placeholders = [
            {
                name: 'cima',
                display_name: 'CIMA',
                description: 'Chinese Immune Multi-omics Atlas - Healthy adult immune profiling with biochemistry and metabolomics',
                n_cells: 6484974,
                n_samples: 421,
                n_cell_types: 39,
                source_type: 'builtin',
                validation_grade: 'A',
            },
            {
                name: 'inflammation',
                display_name: 'Inflammation Atlas',
                description: 'Pan-disease immune profiling across multiple inflammatory conditions with treatment response data',
                n_cells: 4900000,
                n_samples: 817,
                n_cell_types: 43,
                source_type: 'builtin',
                validation_grade: 'B',
            },
            {
                name: 'scatlas',
                display_name: 'scAtlas',
                description: 'Human tissue reference atlas with normal organs and pan-cancer immune profiling',
                n_cells: 6400000,
                n_samples: null,
                n_cell_types: 213,
                source_type: 'builtin',
                validation_grade: 'B',
            },
        ];

        return AtlasCard.renderList(placeholders);
    },

    /**
     * Set up search functionality
     */
    setupSearch() {
        const heroInput = document.getElementById('hero-search-input');

        if (heroInput) {
            heroInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.handleSearch(heroInput.value);
                }
            });
        }
    },

    /**
     * Handle search - go directly to gene page
     * @param {string} query - Search query (gene symbol)
     */
    handleSearch(query) {
        if (!query || !query.trim()) return;

        // Navigate directly to gene page
        router.navigate(`/gene/${encodeURIComponent(query.trim().toUpperCase())}`);
    },
};

// Global search handlers (called from HTML)
window.handleHeroSearch = function() {
    const input = document.getElementById('hero-search-input');
    if (input) {
        HomePage.handleSearch(input.value);
    }
};

window.handleSearch = function() {
    const input = document.getElementById('global-search');
    if (input) {
        HomePage.handleSearch(input.value);
    }
};

window.searchExample = function(term) {
    router.navigate(`/gene/${encodeURIComponent(term)}`);
};

// Make available globally
window.HomePage = HomePage;
