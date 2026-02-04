/**
 * Search Page Handler
 * Simple gene search - navigates to gene detail page
 */

const SearchPage = {
    /**
     * Initialize the search page
     */
    async init(params, query) {
        this.render();
        this.setupEventHandlers();

        // If query provided, search immediately
        if (query.q) {
            document.getElementById('search-input').value = query.q;
            this.performSearch(query.q);
        }
    },

    /**
     * Render the search page template
     */
    render() {
        const app = document.getElementById('app');
        app.innerHTML = `
            <div class="search-page">
                <header class="search-header">
                    <h1>Search</h1>
                    <p class="subtitle">
                        Enter a gene symbol to view expression, cytokine/protein activity,
                        cell type specificity, disease associations, and organ/tissue patterns.
                    </p>
                </header>

                <div class="search-controls">
                    <div class="search-bar search-bar-large">
                        <input type="text"
                               id="search-input"
                               placeholder="Enter gene symbol (e.g., IFNG, TNF, IL6...)"
                               autocomplete="off"
                               autofocus>
                        <button id="search-btn" class="btn btn-primary btn-large">
                            Search
                        </button>
                    </div>
                </div>

                <div class="search-examples">
                    <p>Popular genes:</p>
                    <div class="example-genes">
                        <a href="/gene/IFNG" class="gene-link" title="CytoSig, SecAct">
                            IFNG <span class="avail-badges"><span class="mini-badge cs">CS</span><span class="mini-badge sa">SA</span></span>
                        </a>
                        <a href="/gene/TNF" class="gene-link" title="CytoSig, SecAct">
                            TNF <span class="avail-badges"><span class="mini-badge cs">CS</span><span class="mini-badge sa">SA</span></span>
                        </a>
                        <a href="/gene/IL6" class="gene-link" title="CytoSig, SecAct">
                            IL6 <span class="avail-badges"><span class="mini-badge cs">CS</span><span class="mini-badge sa">SA</span></span>
                        </a>
                        <a href="/gene/IL17A" class="gene-link" title="CytoSig only">
                            IL17A <span class="avail-badges"><span class="mini-badge cs">CS</span><span class="mini-badge unavail">—</span></span>
                        </a>
                        <a href="/gene/IL10" class="gene-link" title="CytoSig, SecAct">
                            IL10 <span class="avail-badges"><span class="mini-badge cs">CS</span><span class="mini-badge sa">SA</span></span>
                        </a>
                        <a href="/gene/TGFB1" class="gene-link" title="CytoSig, SecAct">
                            TGFB1 <span class="avail-badges"><span class="mini-badge cs">CS</span><span class="mini-badge sa">SA</span></span>
                        </a>
                        <a href="/gene/IL1B" class="gene-link" title="CytoSig, SecAct">
                            IL1B <span class="avail-badges"><span class="mini-badge cs">CS</span><span class="mini-badge sa">SA</span></span>
                        </a>
                        <a href="/gene/CCL2" class="gene-link" title="SecAct only">
                            CCL2 <span class="avail-badges"><span class="mini-badge unavail">—</span><span class="mini-badge sa">SA</span></span>
                        </a>
                        <a href="/gene/CXCL10" class="gene-link" title="SecAct only">
                            CXCL10 <span class="avail-badges"><span class="mini-badge unavail">—</span><span class="mini-badge sa">SA</span></span>
                        </a>
                        <a href="/gene/IL2" class="gene-link" title="CytoSig only">
                            IL2 <span class="avail-badges"><span class="mini-badge cs">CS</span><span class="mini-badge unavail">—</span></span>
                        </a>
                    </div>
                    <div class="legend">
                        <span class="legend-item"><span class="mini-badge cs">CS</span> CytoSig (43 cytokines)</span>
                        <span class="legend-item"><span class="mini-badge sa">SA</span> SecAct (1,170 proteins)</span>
                        <span class="legend-item"><span class="mini-badge unavail">—</span> Not available</span>
                    </div>
                </div>

                <div class="search-info">
                    <div class="info-card">
                        <h3>What you'll see</h3>
                        <ul>
                            <li><strong>Gene Expression</strong> - Expression levels by cell type across atlases</li>
                            <li><strong>CytoSig Activity</strong> - Cytokine signaling activity (44 cytokines)</li>
                            <li><strong>SecAct Activity</strong> - Secreted protein activity (1,170 proteins)</li>
                            <li><strong>Disease Associations</strong> - Differential activity in diseases vs healthy</li>
                            <li><strong>Correlations</strong> - Age, BMI, and biochemistry correlations</li>
                        </ul>
                    </div>
                </div>
            </div>
        `;
    },

    /**
     * Set up event handlers
     */
    setupEventHandlers() {
        const searchInput = document.getElementById('search-input');
        const searchBtn = document.getElementById('search-btn');

        // Enter key to search
        searchInput?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                this.performSearch(searchInput.value);
            }
        });

        // Search button
        searchBtn?.addEventListener('click', () => {
            this.performSearch(searchInput.value);
        });
    },

    /**
     * Perform search - navigate directly to gene page
     */
    performSearch(query) {
        if (!query || !query.trim()) return;

        // Navigate directly to gene page
        const geneSymbol = query.trim().toUpperCase();
        router.navigate(`/gene/${encodeURIComponent(geneSymbol)}`);
    },
};

// Make available globally
window.SearchPage = SearchPage;
