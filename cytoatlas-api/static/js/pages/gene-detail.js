/**
 * Gene Detail Page Handler
 * Displays gene expression + CytoSig + SecAct data across all atlases
 */

const GeneDetailPage = {
    gene: null,
    activeTab: 'expression',
    data: null,
    geneInfo: null,  // Loaded from gene_info.json

    tabs: [
        { id: 'expression', label: 'Expression', icon: '&#129516;' },
        { id: 'cytosig', label: 'CytoSig', icon: '&#128300;' },
        { id: 'secact', label: 'SecAct', icon: '&#9898;' },
        { id: 'diseases', label: 'Diseases', icon: '&#129658;' },
        { id: 'correlations', label: 'Correlations', icon: '&#128200;' },
    ],

    /**
     * Initialize the gene detail page
     */
    async init(params, query) {
        this.gene = decodeURIComponent(params.signature);
        this.activeTab = query.tab || 'expression';

        // Load gene info database
        await this.loadGeneInfo();

        this.render();
        await this.loadGeneData();
    },

    /**
     * Load gene info database (descriptions, HGNC symbols)
     */
    async loadGeneInfo() {
        try {
            const response = await fetch('/static/data/gene_info.json');
            this.geneInfo = await response.json();
        } catch (error) {
            console.warn('Failed to load gene info:', error);
            this.geneInfo = null;
        }
    },

    /**
     * Get gene description from the database
     */
    getGeneDescription(gene) {
        if (!this.geneInfo) return null;

        // Check CytoSig signatures (may have non-standard names)
        if (this.geneInfo.cytosig && this.geneInfo.cytosig[gene]) {
            return this.geneInfo.cytosig[gene].description;
        }

        // Check SecAct common genes
        if (this.geneInfo.secact && this.geneInfo.secact[gene]) {
            return this.geneInfo.secact[gene].description;
        }

        // Check if gene matches any CytoSig HGNC symbol
        if (this.geneInfo.cytosig) {
            for (const [sigName, info] of Object.entries(this.geneInfo.cytosig)) {
                if (info.hgnc_symbol === gene) {
                    return info.description;
                }
            }
        }

        // Check if gene matches any SecAct HGNC symbol
        if (this.geneInfo.secact) {
            for (const [sigName, info] of Object.entries(this.geneInfo.secact)) {
                if (info.hgnc_symbol === gene) {
                    return info.description;
                }
            }
        }

        return null;
    },

    /**
     * Get HGNC symbol for a gene (handles CytoSig non-standard names)
     */
    getHGNCSymbol(gene) {
        if (!this.geneInfo) return gene;

        // Check CytoSig signatures
        if (this.geneInfo.cytosig && this.geneInfo.cytosig[gene]) {
            return this.geneInfo.cytosig[gene].hgnc_symbol;
        }

        // Check SecAct common genes
        if (this.geneInfo.secact && this.geneInfo.secact[gene]) {
            return this.geneInfo.secact[gene].hgnc_symbol;
        }

        // Already a standard symbol
        return gene;
    },

    /**
     * Get gene info object (with IDs) for a gene
     */
    getGeneInfoObject(gene) {
        if (!this.geneInfo) return null;

        // Check CytoSig signatures
        if (this.geneInfo.cytosig && this.geneInfo.cytosig[gene]) {
            return this.geneInfo.cytosig[gene];
        }

        // Check SecAct common genes
        if (this.geneInfo.secact && this.geneInfo.secact[gene]) {
            return this.geneInfo.secact[gene];
        }

        // Check if gene matches any CytoSig HGNC symbol
        if (this.geneInfo.cytosig) {
            for (const [sigName, info] of Object.entries(this.geneInfo.cytosig)) {
                if (info.hgnc_symbol === gene) {
                    return info;
                }
            }
        }

        // Check if gene matches any SecAct HGNC symbol
        if (this.geneInfo.secact) {
            for (const [sigName, info] of Object.entries(this.geneInfo.secact)) {
                if (info.hgnc_symbol === gene) {
                    return info;
                }
            }
        }

        return null;
    },

    /**
     * Get external database links for a gene
     */
    getExternalLinks(gene) {
        const info = this.getGeneInfoObject(gene);
        const hgncSymbol = this.getHGNCSymbol(gene);

        // If we have specific IDs, use direct links
        if (info) {
            return [
                {
                    name: 'NCBI',
                    url: info.entrez_id ? `https://www.ncbi.nlm.nih.gov/gene/${info.entrez_id}` : `https://www.ncbi.nlm.nih.gov/gene/?term=${encodeURIComponent(hgncSymbol)}[sym]+AND+human[orgn]`,
                    icon: '🔬',
                },
                {
                    name: 'UniProt',
                    url: info.uniprot_id ? `https://www.uniprot.org/uniprotkb/${info.uniprot_id}` : `https://www.uniprot.org/uniprotkb?query=${encodeURIComponent(hgncSymbol)}+AND+organism_id:9606`,
                    icon: '🧬',
                },
                {
                    name: 'GeneCards',
                    url: `https://www.genecards.org/cgi-bin/carddisp.pl?gene=${encodeURIComponent(hgncSymbol)}`,
                    icon: '📋',
                },
                {
                    name: 'Ensembl',
                    url: info.ensembl_id ? `https://useast.ensembl.org/Homo_sapiens/Gene/Summary?g=${info.ensembl_id}` : `https://www.ensembl.org/Human/Search/Results?q=${encodeURIComponent(hgncSymbol)};site=ensembl;facet_species=Human`,
                    icon: '🌐',
                },
                {
                    name: 'HGNC',
                    url: info.hgnc_id ? `https://www.genenames.org/data/gene-symbol-report/#!/hgnc_id/${info.hgnc_id}` : `https://www.genenames.org/tools/search/#!/?query=${encodeURIComponent(hgncSymbol)}`,
                    icon: '🏷️',
                },
                {
                    name: 'GO',
                    url: info.uniprot_id ? `https://amigo.geneontology.org/amigo/gene_product/UniProtKB:${info.uniprot_id}` : `https://amigo.geneontology.org/amigo/search/bioentity?q=${encodeURIComponent(hgncSymbol)}`,
                    icon: '🔗',
                },
            ];
        }

        // Fallback to search-based links
        return [
            {
                name: 'NCBI',
                url: `https://www.ncbi.nlm.nih.gov/gene/?term=${encodeURIComponent(hgncSymbol)}[sym]+AND+human[orgn]`,
                icon: '🔬',
            },
            {
                name: 'UniProt',
                url: `https://www.uniprot.org/uniprotkb?query=${encodeURIComponent(hgncSymbol)}+AND+organism_id:9606`,
                icon: '🧬',
            },
            {
                name: 'GeneCards',
                url: `https://www.genecards.org/cgi-bin/carddisp.pl?gene=${encodeURIComponent(hgncSymbol)}`,
                icon: '📋',
            },
            {
                name: 'Ensembl',
                url: `https://www.ensembl.org/Human/Search/Results?q=${encodeURIComponent(hgncSymbol)};site=ensembl;facet_species=Human`,
                icon: '🌐',
            },
            {
                name: 'HGNC',
                url: `https://www.genenames.org/tools/search/#!/?query=${encodeURIComponent(hgncSymbol)}`,
                icon: '🏷️',
            },
            {
                name: 'GO',
                url: `https://amigo.geneontology.org/amigo/search/bioentity?q=${encodeURIComponent(hgncSymbol)}`,
                icon: '🔗',
            },
        ];
    },

    /**
     * Render the page template
     */
    render() {
        const app = document.getElementById('app');
        const description = this.getGeneDescription(this.gene);
        const hgncSymbol = this.getHGNCSymbol(this.gene);
        const externalLinks = this.getExternalLinks(this.gene);
        const showHGNCNote = hgncSymbol !== this.gene;

        app.innerHTML = `
            <div class="gene-detail-page">
                <header class="gene-header">
                    <div class="header-top">
                        <a href="/search" class="back-link">&#8592; Back to Search</a>
                    </div>
                    <div id="gene-title" class="gene-title">
                        <h1>${this.gene}</h1>
                        ${showHGNCNote ? `<span class="hgnc-symbol-note">HGNC: ${hgncSymbol}</span>` : ''}
                        <div class="external-links">
                            ${externalLinks.map(link => `
                                <a href="${link.url}" target="_blank" rel="noopener noreferrer"
                                   class="external-link" title="View in ${link.name}">
                                    <span class="link-icon">${link.icon}</span>
                                    <span class="link-name">${link.name}</span>
                                </a>
                            `).join('')}
                        </div>
                    </div>
                    ${description ? `
                        <div class="gene-description">
                            <p>${description}</p>
                        </div>
                    ` : ''}
                    <div id="gene-summary" class="gene-summary">
                        <div class="loading"><div class="spinner"></div>Loading gene data...</div>
                    </div>
                </header>

                <nav class="gene-tabs" id="gene-tabs">
                    ${this.tabs.map(tab => `
                        <button class="gene-tab ${tab.id === this.activeTab ? 'active' : ''}"
                                data-tab="${tab.id}"
                                onclick="GeneDetailPage.switchTab('${tab.id}')">
                            <span>${tab.icon}</span> ${tab.label}
                        </button>
                    `).join('')}
                </nav>

                <main class="gene-content" id="gene-content">
                    <div class="loading"><div class="spinner"></div>Loading...</div>
                </main>
            </div>
        `;
    },

    /**
     * Load complete gene data
     */
    async loadGeneData() {
        try {
            // Load complete gene page data
            this.data = await API.get(`/gene/${encodeURIComponent(this.gene)}/full`);

            // Handle redirect to canonical HGNC name
            if (this.data.redirect_to && this.data.redirect_to !== this.gene) {
                // Update URL without reloading page
                const newUrl = `/gene/${encodeURIComponent(this.data.redirect_to)}`;
                history.replaceState({}, '', newUrl);

                // Update gene name
                this.gene = this.data.redirect_to;

                // Update page title
                const h1 = document.querySelector('.gene-title h1');
                if (h1) h1.textContent = this.gene;
            }

            this.computeUnifiedCellTypes();
            this.updateSummary();
            this.updateTabs();
            await this.loadTabContent(this.activeTab);

        } catch (error) {
            console.error('Failed to load gene data:', error);
            const content = document.getElementById('gene-content');
            if (content) {
                content.innerHTML = `
                    <div class="error-message">
                        <h3>Gene Not Found</h3>
                        <p>No data available for <strong>${this.gene}</strong>.</p>
                        <p>Try searching for a different gene or check the spelling.</p>
                        <a href="/search" class="btn btn-primary">Back to Search</a>
                    </div>
                `;
            }
        }
    },

    /**
     * Compute unified cell type list across expression and activity data
     */
    computeUnifiedCellTypes() {
        if (!this.data) return;

        // Collect all cell types with their max values from all data sources
        // For scAtlas, aggregate by base cell type (without organ suffix)
        const cellTypeScores = {};

        // Helper to get cell type key (normalized name for scAtlas, original for others)
        const getCellTypeKey = (d) => {
            if (d.atlas === 'scAtlas' || d.atlas === 'scAtlas_Normal') {
                return this.normalizeCellTypeName(d.cell_type);
            }
            return d.cell_type;
        };

        // From expression data
        if (this.data.expression && this.data.expression.data) {
            this.data.expression.data.forEach(d => {
                const ct = getCellTypeKey(d);
                if (!cellTypeScores[ct]) {
                    cellTypeScores[ct] = { expression: 0, cytosig: 0, secact: 0, atlas: d.atlas };
                }
                cellTypeScores[ct].expression = Math.max(
                    cellTypeScores[ct].expression,
                    d.mean_expression || 0
                );
            });
        }

        // From CytoSig activity
        if (this.data.cytosig_activity) {
            this.data.cytosig_activity.forEach(d => {
                const ct = getCellTypeKey(d);
                // Normalize atlas name for scAtlas variants
                const atlas = (d.atlas === 'scAtlas_Normal' || d.atlas === 'scAtlas_Cancer') ? 'scAtlas' : d.atlas;
                if (!cellTypeScores[ct]) {
                    cellTypeScores[ct] = { expression: 0, cytosig: 0, secact: 0, atlas: atlas };
                }
                cellTypeScores[ct].cytosig = Math.max(
                    cellTypeScores[ct].cytosig,
                    Math.abs(d.mean_activity) || 0
                );
                cellTypeScores[ct].atlas = atlas;
            });
        }

        // From SecAct activity
        if (this.data.secact_activity) {
            this.data.secact_activity.forEach(d => {
                const ct = getCellTypeKey(d);
                if (!cellTypeScores[ct]) {
                    const atlas = (d.atlas === 'scAtlas_Normal' || d.atlas === 'scAtlas_Cancer') ? 'scAtlas' : d.atlas;
                    cellTypeScores[ct] = { expression: 0, cytosig: 0, secact: 0, atlas: atlas };
                }
                cellTypeScores[ct].secact = Math.max(
                    cellTypeScores[ct].secact,
                    Math.abs(d.mean_activity) || 0
                );
            });
        }

        // Group cell types by atlas and get top from each for balanced representation
        const atlasCellTypes = {};

        // Determine which atlas each cell type belongs to
        Object.entries(cellTypeScores).forEach(([ct, scores]) => {
            const atlas = scores.atlas || 'Unknown';
            if (!atlasCellTypes[atlas]) atlasCellTypes[atlas] = new Set();
            atlasCellTypes[atlas].add(ct);
        });

        // Sort cell types within each atlas by activity score
        const maxCytosig = Math.max(...Object.keys(cellTypeScores).map(ct => cellTypeScores[ct].cytosig)) || 1;
        const maxSecact = Math.max(...Object.keys(cellTypeScores).map(ct => cellTypeScores[ct].secact)) || 1;

        const scoreCell = (ct) => {
            const s = cellTypeScores[ct] || { cytosig: 0, secact: 0 };
            return (s.cytosig / maxCytosig) + (s.secact / maxSecact);
        };

        // Get top cell types from each atlas
        const atlases = Object.keys(atlasCellTypes);
        const perAtlas = Math.ceil(30 / Math.max(atlases.length, 1));

        const selectedCellTypes = [];
        atlases.forEach(atlas => {
            const cts = Array.from(atlasCellTypes[atlas])
                .filter(ct => cellTypeScores[ct])
                .sort((a, b) => scoreCell(b) - scoreCell(a))
                .slice(0, perAtlas);
            selectedCellTypes.push(...cts);
        });

        // Sort final list by score and take top 30
        const sortedCellTypes = selectedCellTypes
            .sort((a, b) => scoreCell(b) - scoreCell(a))
            .slice(0, 30);

        // Store unified cell types (balanced across atlases)
        this._unifiedCellTypes = sortedCellTypes;
    },

    /**
     * Update the summary section
     */
    updateSummary() {
        const summaryEl = document.getElementById('gene-summary');
        if (!summaryEl || !this.data) return;

        // Update title with HGNC symbol if different from query
        const titleEl = document.getElementById('gene-title');
        if (titleEl && this.data.hgnc_symbol && this.data.hgnc_symbol !== this.gene) {
            const hgncNote = titleEl.querySelector('.hgnc-symbol-note');
            if (hgncNote) {
                hgncNote.textContent = `HGNC: ${this.data.hgnc_symbol}`;
            } else {
                const h1 = titleEl.querySelector('h1');
                if (h1) {
                    h1.insertAdjacentHTML('afterend', `<span class="hgnc-symbol-note">HGNC: ${this.data.hgnc_symbol}</span>`);
                }
            }
        }

        // Update description from API if available
        const descEl = document.querySelector('.gene-description p');
        if (this.data.description) {
            if (descEl) {
                descEl.textContent = this.data.description;
            } else {
                const headerEl = document.querySelector('.gene-header');
                const geneTitleEl = document.getElementById('gene-title');
                if (headerEl && geneTitleEl) {
                    geneTitleEl.insertAdjacentHTML('afterend', `
                        <div class="gene-description">
                            <p>${this.data.description}</p>
                        </div>
                    `);
                }
            }
        }

        const badges = [];
        if (this.data.has_expression) badges.push('<span class="data-badge expression">Expression</span>');
        if (this.data.has_cytosig) badges.push('<span class="data-badge cytosig">CytoSig</span>');
        if (this.data.has_secact) {
            badges.push('<span class="data-badge secact">SecAct</span>');
        } else if (this.data.is_cytosig_only) {
            // Show that SecAct is unavailable with reason
            const reason = this.data.cytosig_only_reason || 'Not included in SecAct';
            badges.push(`<span class="data-badge secact unavailable" title="${reason}">SecAct ✗</span>`);
        }

        // Add CytoSig-only indicator if applicable
        const cytosigOnlyNote = this.data.is_cytosig_only
            ? `<span class="cytosig-only-note" title="${this.data.cytosig_only_reason || ''}">CytoSig only</span>`
            : '';

        summaryEl.innerHTML = `
            <div class="data-availability">
                <span class="label">Available data:</span>
                ${badges.length ? badges.join(' ') : '<span class="no-data">No data available</span>'}
                ${cytosigOnlyNote}
            </div>
            <div class="atlas-info">
                <span class="label">Atlases:</span>
                ${this.data.atlases.map(a => `<span class="atlas-badge">${a}</span>`).join('')}
            </div>
        `;
    },

    /**
     * Update tab states based on data availability
     */
    updateTabs() {
        const tabsContainer = document.getElementById('gene-tabs');
        if (!tabsContainer || !this.data) return;

        // Disable tabs with no data
        tabsContainer.querySelectorAll('.gene-tab').forEach(tab => {
            const tabId = tab.dataset.tab;
            let hasData = true;

            if (tabId === 'expression' && !this.data.has_expression) hasData = false;
            if (tabId === 'cytosig' && !this.data.has_cytosig) hasData = false;
            if (tabId === 'secact' && !this.data.has_secact) hasData = false;

            if (!hasData) {
                tab.classList.add('disabled');
                tab.title = 'No data available';
            }
        });

        // If current tab has no data, switch to first available
        if (this.activeTab === 'expression' && !this.data.has_expression) {
            if (this.data.has_cytosig) this.switchTab('cytosig');
            else if (this.data.has_secact) this.switchTab('secact');
        }
    },

    /**
     * Switch to a different tab
     */
    async switchTab(tabId) {
        // Don't switch to disabled tabs
        const tab = document.querySelector(`.gene-tab[data-tab="${tabId}"]`);
        if (tab?.classList.contains('disabled')) return;

        // Update active state
        document.querySelectorAll('.gene-tab').forEach(t => {
            t.classList.toggle('active', t.dataset.tab === tabId);
        });

        this.activeTab = tabId;

        // Update URL
        const url = new URL(window.location);
        url.searchParams.set('tab', tabId);
        window.history.replaceState({}, '', url);

        await this.loadTabContent(tabId);
    },

    /**
     * Load tab content
     */
    async loadTabContent(tabId) {
        const content = document.getElementById('gene-content');
        if (!content) return;

        content.innerHTML = '<div class="loading"><div class="spinner"></div>Loading...</div>';

        try {
            switch (tabId) {
                case 'expression':
                    await this.loadExpressionTab(content);
                    break;
                case 'cytosig':
                    await this.loadActivityTab(content, 'CytoSig', this.data.cytosig_activity);
                    break;
                case 'secact':
                    await this.loadActivityTab(content, 'SecAct', this.data.secact_activity);
                    break;
                case 'diseases':
                    await this.loadDiseasesTab(content);
                    break;
                case 'correlations':
                    await this.loadCorrelationsTab(content);
                    break;
                default:
                    content.innerHTML = '<p>Tab not found</p>';
            }
        } catch (error) {
            console.error(`Failed to load ${tabId} tab:`, error);
            content.innerHTML = `<div class="error-message">Failed to load data: ${error.message}</div>`;
        }
    },

    // ==================== Expression Tab ====================

    async loadExpressionTab(content) {
        if (!this.data.expression || !this.data.expression.data.length) {
            content.innerHTML = `
                <div class="no-data-panel">
                    <h3>No Expression Data</h3>
                    <p>Gene expression data is not available for <strong>${this.gene}</strong>.</p>
                    <p>Try viewing the CytoSig or SecAct activity tabs instead.</p>
                </div>
            `;
            return;
        }

        const expr = this.data.expression;

        content.innerHTML = `
            <div class="tab-header">
                <h2>${this.gene} Gene Expression</h2>
                <p>Mean log-normalized expression by cell type across atlases</p>
                <div class="stats-inline">
                    <span><strong>${expr.n_cell_types}</strong> cell types</span>
                    <span><strong>${expr.atlases.length}</strong> atlases</span>
                    <span>Top: <strong>${expr.top_cell_type}</strong></span>
                </div>
                <div class="tab-filters">
                    <select id="expr-atlas-filter" onchange="GeneDetailPage.filterExpressionByAtlas()">
                        <option value="all">All Atlases</option>
                        ${expr.atlases.map(a => `<option value="${a}">${a}</option>`).join('')}
                    </select>
                </div>
            </div>

            <div class="method-explanation">
                <details>
                    <summary><strong>Expression Aggregation</strong> - How expression is computed</summary>
                    <div class="method-content">
                        <p><strong>Cell-type level expression</strong> is computed by aggregating single-cell data within each cell type.</p>

                        <h4>Aggregation Process (no resampling)</h4>
                        <ol>
                            <li><strong>Group cells</strong> by unique (cell_type, sample_id) combinations</li>
                            <li><strong>Filter groups</strong> with &lt;10 cells (minimum threshold for statistical reliability)</li>
                            <li><strong>Compute mean expression</strong> of log-normalized counts across all cells in each group (direct averaging, no bootstrap/resampling)</li>
                            <li><strong>Calculate % expressed</strong> as fraction of cells with non-zero expression</li>
                        </ol>

                        <h4>Normalization</h4>
                        <p>Expression values are log-normalized (log1p of library-size normalized counts) to account for sequencing depth differences between cells.</p>

                        <p><em>Best for:</em> Identifying gene-expressing cell populations, comparing expression levels across atlases.</p>
                    </div>
                </details>
            </div>

            <div class="viz-container">
                <div id="expression-chart" class="chart-container"></div>
            </div>
            <div class="data-table-container">
                <table class="data-table" id="expression-table">
                    <thead>
                        <tr>
                            <th>Cell Type</th>
                            <th>Atlas</th>
                            <th>Mean Expression</th>
                            <th>% Expressed</th>
                            <th>N Cells</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${expr.data.slice(0, 50).map(d => `
                            <tr>
                                <td>${d.cell_type}</td>
                                <td><span class="atlas-badge">${d.atlas}</span></td>
                                <td class="expression-value">${d.mean_expression.toFixed(4)}</td>
                                <td>${d.pct_expressed.toFixed(1)}%</td>
                                <td>${d.n_cells ? d.n_cells.toLocaleString() : '-'}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;

        this._expressionData = expr.data;
        this.createExpressionChart(expr.data, this.data.expression_boxplot);
    },

    createExpressionChart(data, boxplotData = null) {
        // Check if showing single atlas (filtered) or all atlases
        const atlasesInData = [...new Set(data.map(d => d.atlas))];
        const isSingleAtlas = atlasesInData.length === 1;

        // For single atlas view, derive cell types from filtered data (not unified)
        // For all atlases view, use unified cell types for consistency
        const topCellTypes = isSingleAtlas
            ? [...new Set(data.map(d => d.cell_type))].slice(0, 30)
            : (this._unifiedCellTypes || [...new Set(data.map(d => d.cell_type))].slice(0, 30));

        // Filter to only cell types present in this data
        const availableCellTypes = topCellTypes.filter(ct => data.some(d => d.cell_type === ct));

        const colors = {
            'CIMA': '#3498db',
            'Inflammation': '#e74c3c',
            'scAtlas': '#2ecc71',
            'scAtlas_Normal': '#2ecc71',
            'scAtlas_Cancer': '#9b59b6',
        };

        // Check if we have box plot data
        if (boxplotData && boxplotData.data && boxplotData.data.length > 0) {
            // Filter to only cell types that have boxplot data
            const boxplotCellTypes = new Set(boxplotData.data.map(d => d.cell_type));
            const cellTypesWithBoxplot = availableCellTypes.filter(ct => boxplotCellTypes.has(ct));

            if (cellTypesWithBoxplot.length > 0) {
                this.createBoxPlotChart('expression-chart', boxplotData.data, cellTypesWithBoxplot, colors,
                    `${this.gene} Expression by Cell Type`, 'Expression (log-normalized)');
                return;
            }
        }

        // Fall back to bar chart
        const atlasOrder = ['CIMA', 'Inflammation', 'scAtlas'];
        const atlases = atlasOrder.filter(a => data.some(d => d.atlas === a));

        // Sort cell types based on atlas count
        let orderedCellTypes;
        if (atlases.length === 1) {
            // Single atlas: sort by mean expression (descending)
            const atlas = atlases[0];
            orderedCellTypes = [...availableCellTypes].sort((a, b) => {
                const aData = data.find(d => d.cell_type === a && d.atlas === atlas);
                const bData = data.find(d => d.cell_type === b && d.atlas === atlas);
                return (bData?.mean_expression || 0) - (aData?.mean_expression || 0);
            });
        } else {
            // Multiple atlases: group by atlas, sort within each group
            orderedCellTypes = [];
            atlases.forEach(atlas => {
                const atlasCellTypes = availableCellTypes
                    .filter(ct => data.some(d => d.cell_type === ct && d.atlas === atlas))
                    .sort((a, b) => {
                        const aData = data.find(d => d.cell_type === a && d.atlas === atlas);
                        const bData = data.find(d => d.cell_type === b && d.atlas === atlas);
                        return (bData?.mean_expression || 0) - (aData?.mean_expression || 0);
                    });
                orderedCellTypes.push(...atlasCellTypes);
            });
            orderedCellTypes = [...new Set(orderedCellTypes)];
        }

        // Create horizontal grouped bar chart
        const traces = [];
        atlases.forEach(atlas => {
            const yValues = [];
            const xValues = [];
            orderedCellTypes.forEach(ct => {
                const match = data.find(d => d.cell_type === ct && d.atlas === atlas);
                yValues.push(ct);
                xValues.push(match ? match.mean_expression : null);
            });
            traces.push({
                type: 'bar',
                name: atlas,
                x: xValues,
                y: yValues,
                orientation: 'h',
                marker: { color: colors[atlas] || '#95a5a6' },
                hovertemplate: '<b>%{y}</b><br>%{x:.3f}<extra>' + atlas + '</extra>',
            });
        });

        const layout = {
            title: `${this.gene} Expression by Cell Type`,
            xaxis: {
                title: 'Mean Expression (log-normalized)',
                zeroline: true,
            },
            yaxis: {
                title: '',
                automargin: true,
                categoryorder: 'array',
                categoryarray: [...orderedCellTypes].reverse(),
            },
            height: Math.max(400, orderedCellTypes.length * 25),
            margin: { l: 180, r: 50, t: 50, b: 50 },
            barmode: 'group',
            legend: { orientation: 'h', y: 1.1 },
        };

        Plotly.newPlot('expression-chart', traces, layout, { responsive: true });
    },

    /**
     * Create a box plot chart using Plotly with pre-computed statistics
     * Uses shapes to draw boxes since Plotly doesn't directly support pre-computed quartiles
     */
    createBoxPlotChart(containerId, boxData, cellTypes, colors, title, xAxisTitle) {
        // Filter boxData to only include specified cell types
        const filteredData = boxData.filter(d => cellTypes.includes(d.cell_type));

        // Get unique atlases in preferred order
        const atlasOrder = ['CIMA', 'Inflammation', 'scAtlas'];
        const atlases = atlasOrder.filter(a => filteredData.some(d => d.atlas === a));

        // Sort cell types based on atlas count
        let orderedCellTypes;
        if (atlases.length === 1) {
            // Single atlas: sort by median activity (descending)
            const atlas = atlases[0];
            orderedCellTypes = [...cellTypes].sort((a, b) => {
                const aData = filteredData.find(d => d.cell_type === a && d.atlas === atlas);
                const bData = filteredData.find(d => d.cell_type === b && d.atlas === atlas);
                return (bData?.median || 0) - (aData?.median || 0);
            });
        } else {
            // Multiple atlases: group by atlas, sort within each group by median
            orderedCellTypes = [];
            atlases.forEach(atlas => {
                const atlasCellTypes = cellTypes
                    .filter(ct => filteredData.some(d => d.cell_type === ct && d.atlas === atlas))
                    .sort((a, b) => {
                        const aData = filteredData.find(d => d.cell_type === a && d.atlas === atlas);
                        const bData = filteredData.find(d => d.cell_type === b && d.atlas === atlas);
                        return (bData?.median || 0) - (aData?.median || 0);
                    });
                orderedCellTypes.push(...atlasCellTypes);
            });
            // Remove duplicates while preserving order
            orderedCellTypes = [...new Set(orderedCellTypes)];
        }

        // Create traces using scatter for box visualization
        const traces = [];
        const boxWidth = 0.3;
        const atlasOffsets = {};
        atlases.forEach((atlas, i) => {
            atlasOffsets[atlas] = (i - (atlases.length - 1) / 2) * boxWidth;
        });

        // Create y-axis positions for cell types (using ordered list)
        const yPositions = {};
        orderedCellTypes.forEach((ct, i) => {
            yPositions[ct] = i;
        });

        atlases.forEach(atlas => {
            const atlasData = filteredData.filter(d => d.atlas === atlas);
            const color = colors[atlas] || '#95a5a6';

            // Arrays for box elements
            const boxX = [];  // x positions for boxes (median line)
            const boxY = [];  // y positions
            const q1Values = [];
            const q3Values = [];
            const medianValues = [];
            const minValues = [];
            const maxValues = [];
            const hoverTexts = [];

            atlasData.forEach(d => {
                if (!(d.cell_type in yPositions)) return;

                const yPos = yPositions[d.cell_type] + atlasOffsets[atlas];
                boxY.push(yPos);
                q1Values.push(d.q1);
                q3Values.push(d.q3);
                medianValues.push(d.median);
                minValues.push(d.min);
                maxValues.push(d.max);
                hoverTexts.push(
                    `<b>${d.cell_type}</b><br>` +
                    `Atlas: ${atlas}<br>` +
                    `Median: ${d.median.toFixed(3)}<br>` +
                    `IQR: ${d.q1.toFixed(3)} - ${d.q3.toFixed(3)}<br>` +
                    `Range: ${d.min.toFixed(3)} - ${d.max.toFixed(3)}<br>` +
                    `n=${d.n}`
                );
            });

            // Draw whiskers (min to max line)
            traces.push({
                type: 'scatter',
                mode: 'lines',
                x: minValues.flatMap((min, i) => [min, maxValues[i], null]),
                y: boxY.flatMap(y => [y, y, null]),
                line: { color: color, width: 1 },
                showlegend: false,
                legendgroup: atlas,
                hoverinfo: 'skip',
            });

            // Draw boxes (IQR rectangles) using filled area
            boxY.forEach((y, i) => {
                traces.push({
                    type: 'scatter',
                    mode: 'lines',
                    x: [q1Values[i], q3Values[i], q3Values[i], q1Values[i], q1Values[i]],
                    y: [y - boxWidth/3, y - boxWidth/3, y + boxWidth/3, y + boxWidth/3, y - boxWidth/3],
                    fill: 'toself',
                    fillcolor: color + '80',  // Add transparency
                    line: { color: color, width: 1 },
                    showlegend: i === 0,
                    legendgroup: atlas,
                    name: atlas,
                    text: hoverTexts[i],
                    hoverinfo: 'text',
                });
            });

            // Draw median lines
            traces.push({
                type: 'scatter',
                mode: 'lines',
                x: medianValues.flatMap((med, i) => [med, med, null]),
                y: boxY.flatMap(y => [y - boxWidth/3, y + boxWidth/3, null]),
                line: { color: '#000', width: 2 },
                showlegend: false,
                legendgroup: atlas,
                hoverinfo: 'skip',
            });

            // Draw whisker caps
            traces.push({
                type: 'scatter',
                mode: 'lines',
                x: minValues.flatMap((min, i) => [min, min, null, maxValues[i], maxValues[i], null]),
                y: boxY.flatMap(y => [y - boxWidth/4, y + boxWidth/4, null, y - boxWidth/4, y + boxWidth/4, null]),
                line: { color: color, width: 1 },
                showlegend: false,
                legendgroup: atlas,
                hoverinfo: 'skip',
            });
        });

        const layout = {
            title: title,
            xaxis: {
                title: xAxisTitle,
                zeroline: true,
                zerolinecolor: '#ccc',
            },
            yaxis: {
                title: '',
                tickmode: 'array',
                tickvals: orderedCellTypes.map((ct, i) => i),
                ticktext: orderedCellTypes,
                automargin: true,
            },
            height: Math.max(400, orderedCellTypes.length * 35),
            margin: { l: 180, r: 50, t: 50, b: 50 },
            legend: { orientation: 'h', y: 1.1 },
            hovermode: 'closest',
        };

        Plotly.newPlot(containerId, traces, layout, { responsive: true });
    },

    filterExpressionByAtlas() {
        const atlas = document.getElementById('expr-atlas-filter').value;
        let filtered = this._expressionData;
        let filteredBoxplot = this.data.expression_boxplot;

        if (atlas !== 'all') {
            filtered = filtered.filter(d => d.atlas === atlas);
            if (filteredBoxplot && filteredBoxplot.data) {
                filteredBoxplot = {
                    ...filteredBoxplot,
                    data: filteredBoxplot.data.filter(d => d.atlas === atlas),
                };
            }
        }

        this.createExpressionChart(filtered, filteredBoxplot);

        const tbody = document.querySelector('#expression-table tbody');
        if (tbody) {
            tbody.innerHTML = filtered.slice(0, 50).map(d => `
                <tr>
                    <td>${d.cell_type}</td>
                    <td><span class="atlas-badge">${d.atlas}</span></td>
                    <td class="expression-value">${d.mean_expression.toFixed(4)}</td>
                    <td>${d.pct_expressed.toFixed(1)}%</td>
                    <td>${d.n_cells ? d.n_cells.toLocaleString() : '-'}</td>
                </tr>
            `).join('');
        }
    },

    // ==================== Activity Tab (CytoSig/SecAct) ====================

    async loadActivityTab(content, sigType, activityData) {
        if (!activityData || !activityData.length) {
            // Check if this is a CytoSig-only signature (for SecAct tab)
            if (sigType === 'SecAct' && this.data && this.data.is_cytosig_only) {
                const reason = this.data.cytosig_only_reason || 'Not included in SecAct signature matrix';
                content.innerHTML = `
                    <div class="no-data-panel cytosig-only-notice">
                        <div class="notice-icon">&#9432;</div>
                        <h3>${this.gene} is available in CytoSig only</h3>
                        <p class="reason"><strong>Reason:</strong> ${reason}</p>
                        <p class="explanation">
                            SecAct contains ~1,170 secreted proteins, but <strong>${this.gene}</strong> is not included.
                            This is one of 11 CytoSig cytokines without a SecAct equivalent.
                        </p>
                        <div class="cytosig-only-list">
                            <details>
                                <summary>View all CytoSig-only signatures (11)</summary>
                                <ul>
                                    <li><strong>EGF</strong> - Not in SecAct</li>
                                    <li><strong>IFN1</strong> - Type I IFN (composite)</li>
                                    <li><strong>IL12</strong> - Heterodimer subunit</li>
                                    <li><strong>IL13</strong> - Not in SecAct</li>
                                    <li><strong>IL17A</strong> - Not in SecAct</li>
                                    <li><strong>IL2</strong> - Not in SecAct</li>
                                    <li><strong>IL22</strong> - Not in SecAct</li>
                                    <li><strong>IL3</strong> - Not in SecAct</li>
                                    <li><strong>IL4</strong> - Not in SecAct</li>
                                    <li><strong>NO</strong> - Nitric oxide (not a protein)</li>
                                    <li><strong>WNT3A</strong> - Not in SecAct</li>
                                </ul>
                            </details>
                        </div>
                        <p class="suggestion">
                            <a href="#" onclick="GeneDetailPage.switchTab('cytosig'); return false;">
                                &#8592; View CytoSig activity instead
                            </a>
                        </p>
                    </div>
                `;
            } else {
                content.innerHTML = `
                    <div class="no-data-panel">
                        <h3>No ${sigType} Activity</h3>
                        <p><strong>${this.gene}</strong> is not available as a ${sigType} signature.</p>
                        <p>${sigType === 'CytoSig' ? 'CytoSig contains 43 major cytokines.' : 'SecAct contains ~1,170 secreted proteins.'}</p>
                    </div>
                `;
            }
            return;
        }

        // Store raw activity data for filtering
        this._activityData = activityData;
        this._activitySigType = sigType;

        // Aggregate scAtlas data by cell type for initial "All Atlases" view
        // Handle both 'scAtlas' and 'scAtlas_Normal' atlas names
        const scAtlasData = activityData.filter(d => d.atlas === 'scAtlas' || d.atlas === 'scAtlas_Normal');
        const nonScAtlasData = activityData.filter(d => d.atlas !== 'scAtlas' && d.atlas !== 'scAtlas_Normal');
        const scAtlasAggregated = this.aggregateByBaseCellType(scAtlasData);

        // Combined data: non-scAtlas as-is + aggregated scAtlas
        const displayData = [...nonScAtlasData, ...scAtlasAggregated]
            .sort((a, b) => b.mean_activity - a.mean_activity);

        // Group by atlas for dropdown
        const byAtlas = {};
        activityData.forEach(d => {
            if (!byAtlas[d.atlas]) byAtlas[d.atlas] = [];
            byAtlas[d.atlas].push(d);
        });

        // Prepare boxplot data with aggregated scAtlas
        let boxplotData = sigType === 'CytoSig' ? this.data.cytosig_boxplot : this.data.secact_boxplot;
        let displayBoxplotData = null;
        if (boxplotData && boxplotData.data) {
            const scAtlasBp = boxplotData.data.filter(d => d.atlas === 'scAtlas' || d.atlas === 'scAtlas_Normal');
            const nonScAtlasBp = boxplotData.data.filter(d => d.atlas !== 'scAtlas' && d.atlas !== 'scAtlas_Normal');
            const scAtlasBpAggregated = this.aggregateBoxplotByBaseCellType(scAtlasBp);
            displayBoxplotData = {
                ...boxplotData,
                data: [...nonScAtlasBp, ...scAtlasBpAggregated],
            };
        }

        content.innerHTML = `
            <div class="tab-header">
                <h2>${this.gene} ${sigType} Activity</h2>
                <p>Pseudobulk-aggregated ${sigType === 'CytoSig' ? 'cytokine' : 'secreted protein'} activity by cell type (ridge regression z-scores)</p>
                <div class="stats-inline">
                    <span><strong>${displayData.length}</strong> cell types</span>
                    <span><strong>${Object.keys(byAtlas).length}</strong> atlases</span>
                </div>
                <div class="tab-filters">
                    <select id="activity-atlas-filter" onchange="GeneDetailPage.filterActivityByAtlas('${sigType}')">
                        <option value="all">All Atlases</option>
                        ${Object.keys(byAtlas).map(a => `<option value="${a}">${a}</option>`).join('')}
                    </select>
                    <select id="activity-organ-filter" class="hidden" onchange="GeneDetailPage.filterActivityByOrgan('${sigType}')">
                        <option value="all">All Organs</option>
                    </select>
                </div>
            </div>

            <div class="method-explanation">
                <details>
                    <summary><strong>Pseudobulk Method</strong> - How activity is computed</summary>
                    <div class="method-content">
                        <p><strong>Pseudobulk aggregation</strong> combines cells by cell type and sample to create robust activity estimates.
                        This approach reduces noise from single-cell dropout and technical variation.</p>

                        <h4>Aggregation Process (no resampling)</h4>
                        <ol>
                            <li><strong>Group cells</strong> by unique (cell_type, sample_id) combinations</li>
                            <li><strong>Filter groups</strong> with &lt;10 cells (minimum threshold for statistical reliability)</li>
                            <li><strong>Sum raw counts</strong> across all cells within each group (direct summation, no bootstrap/resampling)</li>
                            <li><strong>Normalize</strong> to CPM (counts per million) and apply log1p transformation</li>
                        </ol>

                        <h4>${sigType} Activity Inference</h4>
                        <ol>
                            <li><strong>Align genes</strong> between expression matrix and ${sigType} signature matrix</li>
                            <li><strong>Ridge regression</strong> (α=1e-4) to infer activity from gene expression</li>
                            <li><strong>Z-score normalization</strong> across all samples for comparability</li>
                        </ol>

                        <h4>scAtlas Organ Aggregation</h4>
                        <p>scAtlas contains cell type activity computed separately for each organ. Display mode depends on your selection:</p>
                        <ul>
                            <li><strong>All Atlases / All Organs:</strong> Activities are aggregated across organs using <em>cell-weighted mean</em>:
                                <code style="display:block;margin:0.5em 0;padding:0.5em;background:#f5f5f5;border-radius:4px;">
                                    Aggregated Activity = Σ(activity<sub>organ</sub> × n_cells<sub>organ</sub>) / Σ(n_cells<sub>organ</sub>)
                                </code>
                                This weights each organ's contribution by its cell count, giving more influence to organs with larger cell populations.
                            </li>
                            <li><strong>Specific Organ:</strong> Shows activity values computed only from cells of that organ, without cross-organ aggregation.</li>
                        </ul>
                        <p><em>Note:</em> Box plot statistics (quartiles) for aggregated view are approximated from organ-level statistics using cell-count weighting.</p>

                        <h4>Box Plot Availability</h4>
                        <p>Box plots require ≥3 biological samples per (cell_type, organ) combination to compute valid quartile statistics (min, Q1, median, Q3, max).
                        Cell types with fewer samples are shown as bar charts instead. For specific organs in scAtlas, box plots are only available for cell types
                        with sufficient sample coverage in that organ.</p>

                        <p><em>Signature matrix:</em> ${sigType === 'CytoSig' ? '43 cytokines' : '1,170 secreted proteins'} with curated gene weights from literature.</p>
                        <p><em>Best for:</em> Comparing cell types, identifying signature-producing cell populations, cross-atlas comparisons.</p>
                    </div>
                </details>
            </div>

            <div class="viz-container">
                <div id="activity-chart" class="chart-container"></div>
            </div>
            <div class="data-table-container">
                <table class="data-table" id="activity-table">
                    <thead>
                        <tr>
                            <th>Cell Type</th>
                            <th>Atlas</th>
                            <th>Mean Activity (z-score)</th>
                            <th>N Cells</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${displayData.slice(0, 50).map(d => `
                            <tr>
                                <td>${d.cell_type}</td>
                                <td><span class="atlas-badge">${d.atlas}</span></td>
                                <td class="${d.mean_activity >= 0 ? 'positive' : 'negative'}">${d.mean_activity.toFixed(4)}</td>
                                <td>${d.n_cells ? d.n_cells.toLocaleString() : '-'}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;

        this.createActivityChart(displayData, sigType, displayBoxplotData);
    },

    createActivityChart(data, sigType, boxplotData = null) {
        // Check if showing single atlas (filtered) or all atlases
        const atlasesInData = [...new Set(data.map(d => d.atlas))];
        const isSingleAtlas = atlasesInData.length === 1;

        // For single atlas view, derive cell types from filtered data (not unified)
        // For all atlases view, use unified cell types for consistency
        const topCellTypes = isSingleAtlas
            ? [...new Set(data.map(d => d.cell_type))].slice(0, 30)
            : (this._unifiedCellTypes || [...new Set(data.map(d => d.cell_type))].slice(0, 30));

        // Filter to only cell types present in this data
        const availableCellTypes = topCellTypes.filter(ct => data.some(d => d.cell_type === ct));

        const colors = {
            'CIMA': '#3498db',
            'Inflammation': '#e74c3c',
            'scAtlas': '#2ecc71',
            'scAtlas_Normal': '#2ecc71',
            'scAtlas_Cancer': '#9b59b6',
        };

        // Check if we have box plot data
        if (boxplotData && boxplotData.data && boxplotData.data.length > 0) {
            // Filter to only cell types that have boxplot data
            const boxplotCellTypes = new Set(boxplotData.data.map(d => d.cell_type));
            const cellTypesWithBoxplot = availableCellTypes.filter(ct => boxplotCellTypes.has(ct));
            const cellTypesWithoutBoxplot = availableCellTypes.filter(ct => !boxplotCellTypes.has(ct));

            if (cellTypesWithBoxplot.length > 0) {
                this.createBoxPlotChart('activity-chart', boxplotData.data, cellTypesWithBoxplot, colors,
                    `${this.gene} ${sigType} Activity by Cell Type`, 'Activity (z-score)');

                // Show notification if some cell types don't have boxplot data
                if (cellTypesWithoutBoxplot.length > 0) {
                    const chartContainer = document.getElementById('activity-chart');
                    const notice = document.createElement('div');
                    notice.className = 'boxplot-notice';
                    notice.innerHTML = `<small>📊 Showing ${cellTypesWithBoxplot.length} of ${availableCellTypes.length} cell types. ` +
                        `${cellTypesWithoutBoxplot.length} cell type(s) have &lt;3 samples and are shown in table only.</small>`;
                    chartContainer.parentElement.insertBefore(notice, chartContainer.nextSibling);
                }
                return;
            }
        }

        // Fall back to bar chart
        const atlasOrder = ['CIMA', 'Inflammation', 'scAtlas'];
        const atlases = atlasOrder.filter(a => data.some(d => d.atlas === a));

        // Sort cell types based on atlas count
        let orderedCellTypes;
        if (atlases.length === 1) {
            // Single atlas: sort by mean activity (descending)
            const atlas = atlases[0];
            orderedCellTypes = [...availableCellTypes].sort((a, b) => {
                const aData = data.find(d => d.cell_type === a && d.atlas === atlas);
                const bData = data.find(d => d.cell_type === b && d.atlas === atlas);
                return (bData?.mean_activity || 0) - (aData?.mean_activity || 0);
            });
        } else {
            // Multiple atlases: group by atlas, sort within each group
            orderedCellTypes = [];
            atlases.forEach(atlas => {
                const atlasCellTypes = availableCellTypes
                    .filter(ct => data.some(d => d.cell_type === ct && d.atlas === atlas))
                    .sort((a, b) => {
                        const aData = data.find(d => d.cell_type === a && d.atlas === atlas);
                        const bData = data.find(d => d.cell_type === b && d.atlas === atlas);
                        return (bData?.mean_activity || 0) - (aData?.mean_activity || 0);
                    });
                orderedCellTypes.push(...atlasCellTypes);
            });
            orderedCellTypes = [...new Set(orderedCellTypes)];
        }

        // Create horizontal grouped bar chart
        const traces = [];
        atlases.forEach(atlas => {
            const yValues = [];
            const xValues = [];
            orderedCellTypes.forEach(ct => {
                const match = data.find(d => d.cell_type === ct && d.atlas === atlas);
                yValues.push(ct);
                xValues.push(match ? match.mean_activity : null);
            });
            traces.push({
                type: 'bar',
                name: atlas,
                x: xValues,
                y: yValues,
                orientation: 'h',
                marker: { color: colors[atlas] || '#95a5a6' },
                hovertemplate: '<b>%{y}</b><br>%{x:.4f}<extra>' + atlas + '</extra>',
            });
        });

        const layout = {
            title: `${this.gene} ${sigType} Activity by Cell Type`,
            xaxis: {
                title: 'Mean Activity (z-score)',
                zeroline: true,
                zerolinecolor: '#999',
            },
            yaxis: {
                title: '',
                automargin: true,
                categoryorder: 'array',
                categoryarray: [...orderedCellTypes].reverse(),
            },
            height: Math.max(400, orderedCellTypes.length * 25),
            margin: { l: 180, r: 50, t: 50, b: 50 },
            barmode: 'group',
            legend: { orientation: 'h', y: 1.1 },
        };

        Plotly.newPlot('activity-chart', traces, layout, { responsive: true });
    },

    /**
     * Extract organ from cell type name like "Macrophage (Spleen)" -> "Spleen"
     */
    extractOrganFromCellType(cellType) {
        const match = cellType.match(/\(([^)]+)\)$/);
        return match ? match[1] : null;
    },

    /**
     * Extract base cell type from name like "Macrophage (Spleen)" -> "Macrophage"
     */
    extractBaseCellType(cellType) {
        return cellType.replace(/\s*\([^)]+\)$/, '');
    },

    /**
     * Normalize cell type name for consistent aggregation
     * Handles case variations and common plural forms
     */
    normalizeCellTypeName(cellType) {
        // First extract base (remove organ suffix)
        let name = this.extractBaseCellType(cellType);

        // Normalize case - use lowercase for matching
        const lowerName = name.toLowerCase().trim();

        // Comprehensive normalization map (lowercase -> canonical form)
        // Includes all observed variations from scAtlas data
        const normalizationMap = {
            // B cells
            'b cell': 'B Cell',
            'b cells': 'B Cell',
            // T cells
            't cell': 'T Cell',
            't cells': 'T Cell',
            // NK cells
            'nk cell': 'NK Cell',
            'nk cells': 'NK Cell',
            'nkt cell': 'NKT Cell',
            // Macrophages
            'macrophage': 'Macrophage',
            'macrophages': 'Macrophage',
            // Monocytes
            'monocyte': 'Monocyte',
            'monocytes': 'Monocyte',
            // Fibroblasts
            'fibroblast': 'Fibroblast',
            'fibroblasts': 'Fibroblast',
            'fb': 'Fibroblast',
            // Endothelial cells - aggregate all vascular/blood vessel variants
            'vascular': 'Endothelial Cell',
            'endothelia': 'Endothelial Cell',
            'endothelial': 'Endothelial Cell',
            'endothelial cell': 'Endothelial Cell',
            'endothelial cells': 'Endothelial Cell',
            'vascular endothelial cells': 'Endothelial Cell',
            'blood vessels': 'Endothelial Cell',
            'blood_vessel': 'Endothelial Cell',
            'artery endothelial cell': 'Endothelial Cell',
            'endothelial cell of artery': 'Endothelial Cell',
            'vein endothelial cell': 'Endothelial Cell',
            'capillary endothelial cell': 'Endothelial Cell',
            'endothelial cell of vascular tree': 'Endothelial Cell',
            'cardiac endothelial cell': 'Endothelial Cell',
            'portal_endothelial_cells': 'Endothelial Cell',
            'bronchial vessel endothelial cell': 'Endothelial Cell',
            'lung microvascular endothelial cell': 'Endothelial Cell',
            'retinal blood vessel endothelial cell': 'Endothelial Cell',
            'ascending vasa recta endothelium': 'Endothelial Cell',
            'peritubular capillary endothelium': 'Endothelial Cell',
            'endothelial cell of hepatic sinusoid': 'Endothelial Cell',
            // Vascular smooth muscle - keep separate from endothelial
            'vascular associated smooth muscle cell': 'Vascular Smooth Muscle Cell',
            // Lymphatic endothelial - aggregate all variants
            'lymphatic endothelial cell': 'Lymphatic Endothelial Cell',
            'lymphatic endothelial cells': 'Lymphatic Endothelial Cell',
            'endothelial cell of lymphatic vessel': 'Lymphatic Endothelial Cell',
            'lymphatic ec': 'Lymphatic Endothelial Cell',
            'lymph_vessel': 'Lymphatic Endothelial Cell',
            'lymphatic': 'Lymphatic Endothelial Cell',
            // Lymphoid cells
            'lymphoid': 'Lymphoid',
            'lymphocytes': 'Lymphoid',
            'innate_lymphoid': 'Innate Lymphoid Cell',
            'innate lymphoid cell': 'Innate Lymphoid Cell',
            // Epithelial cells
            'epithelial cell': 'Epithelial Cell',
            'epithelial cells': 'Epithelial Cell',
            'basal cell': 'Basal Cell',
            'basal cells': 'Basal Cell',
            'secretory cell': 'Secretory Cell',
            'umbrella cell': 'Umbrella Cell',
            'umbrella cells': 'Umbrella Cell',
            // Other immune cells
            'myeloid': 'Myeloid',
            'granulocyte': 'Granulocyte',
            'basophil': 'Basophil',
            'basophils': 'Basophil',
            'neutrophil': 'Neutrophil',
            'neutrophils': 'Neutrophil',
            'mast cell': 'Mast Cell',
            'mast cells': 'Mast Cell',
            'plasma cell': 'Plasma Cell',
            'plasma cells': 'Plasma Cell',
            // DCs
            'dc': 'DC',
            'cdc2': 'cDC2',
            'cdc2s': 'cDC2',
            'pdc': 'pDC',
            'pdcs': 'pDC',
            // Other cell types
            'hepatocyte': 'Hepatocyte',
            'cholangiocyte': 'Cholangiocyte',
            'cholangiocytes': 'Cholangiocyte',
            'melanocyte': 'Melanocyte',
            'melanocytes': 'Melanocyte',
            'keratinocyte': 'Keratinocyte',
            'keratinocytes': 'Keratinocyte',
            'pericyte': 'Pericyte',
            'pericytes': 'Pericyte',
            'myoepithelial cell': 'Myoepithelial Cell',
            'myoepithelial cells': 'Myoepithelial Cell',
            // Smooth muscle
            'smooth muscle': 'Smooth Muscle',
            'smooth muscle cell': 'Smooth Muscle Cell',
            'smooth muscle cells': 'Smooth Muscle Cell',
        };

        // Check for exact match in normalization map
        if (normalizationMap[lowerName]) {
            return normalizationMap[lowerName];
        }

        // For compound names like "Macrophage C1QB", normalize the base part
        // Sort by key length (descending) to match longer patterns first
        const sortedKeys = Object.keys(normalizationMap).sort((a, b) => b.length - a.length);
        for (const key of sortedKeys) {
            if (lowerName.startsWith(key + ' ')) {
                // Replace the base with normalized version, keep the rest
                const rest = name.substring(key.length);
                return normalizationMap[key] + rest;
            }
        }

        // Default: return original with trimming
        return name.trim();
    },

    /**
     * Aggregate scAtlas data by cell type (weighted mean across organs)
     * Also normalizes cell type names (case, plurals)
     */
    aggregateByBaseCellType(data) {
        const aggregates = {};

        data.forEach(d => {
            // Use normalized cell type name for consistent aggregation
            const normalizedCellType = this.normalizeCellTypeName(d.cell_type);
            const nCells = d.n_cells || 1;

            if (!aggregates[normalizedCellType]) {
                aggregates[normalizedCellType] = {
                    cell_type: normalizedCellType,
                    atlas: 'scAtlas',  // Normalize to 'scAtlas' for aggregated view
                    signature: d.signature,
                    signature_type: d.signature_type,
                    sum_weighted: 0,
                    total_cells: 0,
                    organs: new Set(),
                };
            }

            aggregates[normalizedCellType].sum_weighted += d.mean_activity * nCells;
            aggregates[normalizedCellType].total_cells += nCells;
            const organ = this.extractOrganFromCellType(d.cell_type);
            if (organ) aggregates[normalizedCellType].organs.add(organ);
        });

        // Convert to array with weighted mean
        return Object.values(aggregates).map(agg => ({
            cell_type: agg.cell_type,
            atlas: agg.atlas,
            signature: agg.signature,
            signature_type: agg.signature_type,
            mean_activity: agg.total_cells > 0 ? agg.sum_weighted / agg.total_cells : 0,
            n_cells: agg.total_cells,
            n_organs: agg.organs.size,
        })).sort((a, b) => b.mean_activity - a.mean_activity);
    },

    /**
     * Aggregate boxplot data by cell type (combine samples across organs)
     * Also normalizes cell type names (case, plurals)
     */
    aggregateBoxplotByBaseCellType(data) {
        const aggregates = {};

        data.forEach(d => {
            // Use normalized cell type name for consistent aggregation
            const normalizedCellType = this.normalizeCellTypeName(d.cell_type);

            if (!aggregates[normalizedCellType]) {
                aggregates[normalizedCellType] = {
                    cell_type: normalizedCellType,
                    atlas: 'scAtlas',  // Normalize to 'scAtlas' for aggregated view
                    samples: [],  // Collect all sample values to recompute quartiles
                };
            }

            // We don't have raw samples, so we approximate by using the existing stats
            // weighted by n samples. This is an approximation.
            aggregates[normalizedCellType].samples.push({
                min: d.min,
                q1: d.q1,
                median: d.median,
                q3: d.q3,
                max: d.max,
                n: d.n || 1,
            });
        });

        // Convert to array with approximate combined stats
        return Object.values(aggregates).map(agg => {
            // Weighted average of quartiles (approximation)
            let totalN = 0;
            let sumMin = 0, sumQ1 = 0, sumMedian = 0, sumQ3 = 0, sumMax = 0;
            let globalMin = Infinity, globalMax = -Infinity;

            agg.samples.forEach(s => {
                const n = s.n || 1;
                totalN += n;
                sumMin += s.min * n;
                sumQ1 += s.q1 * n;
                sumMedian += s.median * n;
                sumQ3 += s.q3 * n;
                sumMax += s.max * n;
                globalMin = Math.min(globalMin, s.min);
                globalMax = Math.max(globalMax, s.max);
            });

            return {
                cell_type: agg.cell_type,
                atlas: agg.atlas,
                min: globalMin,
                q1: totalN > 0 ? sumQ1 / totalN : 0,
                median: totalN > 0 ? sumMedian / totalN : 0,
                q3: totalN > 0 ? sumQ3 / totalN : 0,
                max: globalMax,
                n: totalN,
            };
        }).sort((a, b) => b.median - a.median);
    },

    filterActivityByAtlas(sigType) {
        const atlas = document.getElementById('activity-atlas-filter').value;
        const organFilter = document.getElementById('activity-organ-filter');
        let filtered = [];
        let boxplotData = sigType === 'CytoSig' ? this.data.cytosig_boxplot : this.data.secact_boxplot;
        let boxplotFiltered = [];

        // Always aggregate scAtlas data by cell type (for both "all" and "scAtlas" views)
        // Handle both 'scAtlas' and 'scAtlas_Normal' atlas names
        const scAtlasData = this._activityData.filter(d => d.atlas === 'scAtlas' || d.atlas === 'scAtlas_Normal');
        const scAtlasAggregated = this.aggregateByBaseCellType(scAtlasData);

        // Aggregate scAtlas boxplot data
        let scAtlasBpAggregated = [];
        if (boxplotData && boxplotData.data) {
            const scAtlasBp = boxplotData.data.filter(d => d.atlas === 'scAtlas' || d.atlas === 'scAtlas_Normal');
            scAtlasBpAggregated = this.aggregateBoxplotByBaseCellType(scAtlasBp);
        }

        // Show/hide organ filter based on atlas selection
        if (atlas === 'scAtlas') {
            // Extract unique organs from scAtlas data
            const organs = new Set();
            scAtlasData.forEach(d => {
                const organ = this.extractOrganFromCellType(d.cell_type);
                if (organ) organs.add(organ);
            });

            // Populate organ dropdown
            const sortedOrgans = [...organs].sort();
            organFilter.innerHTML = `
                <option value="all">All Organs - Aggregated (${sortedOrgans.length} organs)</option>
                ${sortedOrgans.map(o => `<option value="${o}">${o}</option>`).join('')}
            `;
            organFilter.classList.remove('hidden');
            organFilter.value = 'all';

            // Use aggregated scAtlas data
            filtered = scAtlasAggregated;
            boxplotFiltered = scAtlasBpAggregated;
        } else {
            organFilter.classList.add('hidden');
            organFilter.value = 'all';

            if (atlas === 'all') {
                // For "All Atlases": use CIMA + Inflammation as-is, plus aggregated scAtlas
                const nonScAtlasData = this._activityData.filter(d => d.atlas !== 'scAtlas');
                filtered = [...nonScAtlasData, ...scAtlasAggregated];

                // Combine boxplot data: non-scAtlas as-is + aggregated scAtlas
                if (boxplotData && boxplotData.data) {
                    const nonScAtlasBp = boxplotData.data.filter(d => d.atlas !== 'scAtlas' && d.atlas !== 'scAtlas_Normal');
                    boxplotFiltered = [...nonScAtlasBp, ...scAtlasBpAggregated];
                }
            } else {
                // Specific atlas (CIMA or Inflammation)
                filtered = this._activityData.filter(d => d.atlas === atlas);
                if (boxplotData && boxplotData.data) {
                    boxplotFiltered = boxplotData.data.filter(d => d.atlas === atlas);
                }
            }
        }

        // Sort by mean activity
        filtered.sort((a, b) => b.mean_activity - a.mean_activity);

        // Update boxplot data object
        if (boxplotData) {
            boxplotData = { ...boxplotData, data: boxplotFiltered };
        }

        this._currentAtlasFilter = atlas;
        this._currentOrganFilter = 'all';
        this.createActivityChart(filtered, sigType, boxplotData);
        this.updateActivityTable(filtered, atlas === 'scAtlas', false);
    },

    filterActivityByOrgan(sigType) {
        const atlas = document.getElementById('activity-atlas-filter').value;
        const organ = document.getElementById('activity-organ-filter').value;
        let boxplotData = sigType === 'CytoSig' ? this.data.cytosig_boxplot : this.data.secact_boxplot;

        // Start with scAtlas data (handle both 'scAtlas' and 'scAtlas_Normal')
        const scAtlasData = this._activityData.filter(d => d.atlas === 'scAtlas' || d.atlas === 'scAtlas_Normal');

        let filtered;
        if (organ === 'all') {
            // Aggregate by base cell type
            filtered = this.aggregateByBaseCellType(scAtlasData);

            if (boxplotData && boxplotData.data) {
                const scAtlasBp = boxplotData.data.filter(d => d.atlas === 'scAtlas' || d.atlas === 'scAtlas_Normal');
                boxplotData = {
                    ...boxplotData,
                    data: this.aggregateBoxplotByBaseCellType(scAtlasBp),
                };
            }
        } else {
            // Filter to specific organ and show base cell type names
            filtered = scAtlasData
                .filter(d => {
                    const cellOrgan = this.extractOrganFromCellType(d.cell_type);
                    return cellOrgan === organ;
                })
                .map(d => ({
                    ...d,
                    cell_type: this.extractBaseCellType(d.cell_type),  // Show base name
                    organ: organ,
                }))
                .sort((a, b) => b.mean_activity - a.mean_activity);

            // Filter boxplot data to specific organ
            if (boxplotData && boxplotData.data) {
                const organBp = boxplotData.data
                    .filter(d => {
                        if (d.atlas !== 'scAtlas' && d.atlas !== 'scAtlas_Normal') return false;
                        const cellOrgan = this.extractOrganFromCellType(d.cell_type);
                        return cellOrgan === organ;
                    })
                    .map(d => ({
                        ...d,
                        cell_type: this.extractBaseCellType(d.cell_type),
                    }));
                boxplotData = { ...boxplotData, data: organBp };
            }
        }

        this._currentOrganFilter = organ;
        this.createActivityChart(filtered, sigType, boxplotData);
        this.updateActivityTable(filtered, true, organ !== 'all', organ);
    },

    updateActivityTable(data, isScAtlas = false, isOrganSpecific = false, organName = null) {
        const tbody = document.querySelector('#activity-table tbody');
        const thead = document.querySelector('#activity-table thead tr');

        if (!tbody) return;

        // Update table header based on view type
        if (thead) {
            if (isScAtlas && !isOrganSpecific) {
                // Aggregated view - show N Organs column
                thead.innerHTML = `
                    <th>Cell Type</th>
                    <th>N Organs</th>
                    <th>Mean Activity (z-score)</th>
                    <th>N Cells</th>
                `;
            } else if (isScAtlas && isOrganSpecific) {
                // Organ-specific view
                thead.innerHTML = `
                    <th>Cell Type</th>
                    <th>Organ</th>
                    <th>Mean Activity (z-score)</th>
                    <th>N Cells</th>
                `;
            } else {
                thead.innerHTML = `
                    <th>Cell Type</th>
                    <th>Atlas</th>
                    <th>Mean Activity (z-score)</th>
                    <th>N Cells</th>
                `;
            }
        }

        // Update table body
        tbody.innerHTML = data.slice(0, 50).map(d => {
            if (isScAtlas && !isOrganSpecific) {
                // Aggregated view
                return `
                    <tr>
                        <td>${d.cell_type}</td>
                        <td><span class="organ-badge">${d.n_organs || '-'}</span></td>
                        <td class="${d.mean_activity >= 0 ? 'positive' : 'negative'}">${d.mean_activity.toFixed(4)}</td>
                        <td>${d.n_cells ? d.n_cells.toLocaleString() : '-'}</td>
                    </tr>
                `;
            } else if (isScAtlas && isOrganSpecific) {
                // Organ-specific view
                return `
                    <tr>
                        <td>${d.cell_type}</td>
                        <td><span class="organ-badge">${organName}</span></td>
                        <td class="${d.mean_activity >= 0 ? 'positive' : 'negative'}">${d.mean_activity.toFixed(4)}</td>
                        <td>${d.n_cells ? d.n_cells.toLocaleString() : '-'}</td>
                    </tr>
                `;
            } else {
                return `
                    <tr>
                        <td>${d.cell_type}</td>
                        <td><span class="atlas-badge">${d.atlas}</span></td>
                        <td class="${d.mean_activity >= 0 ? 'positive' : 'negative'}">${d.mean_activity.toFixed(4)}</td>
                        <td>${d.n_cells ? d.n_cells.toLocaleString() : '-'}</td>
                    </tr>
                `;
            }
        }).join('');
    },

    // ==================== Diseases Tab ====================

    async loadDiseasesTab(content) {
        try {
            // Try CytoSig first, then SecAct
            let response = await API.get(`/gene/${encodeURIComponent(this.gene)}/diseases`, {
                signature_type: 'CytoSig',
            });

            if (!response.data || !response.data.length) {
                response = await API.get(`/gene/${encodeURIComponent(this.gene)}/diseases`, {
                    signature_type: 'SecAct',
                });
            }

            const data = response.data || [];

            if (!data.length) {
                content.innerHTML = `
                    <div class="no-data-panel">
                        <h3>No Disease Data</h3>
                        <p>Disease differential data is not available for <strong>${this.gene}</strong>.</p>
                    </div>
                `;
                return;
            }

            content.innerHTML = `
                <div class="tab-header">
                    <h2>${this.gene} Disease Associations</h2>
                    <p>Differential activity (disease vs healthy) from the Inflammation Atlas</p>
                    <div class="stats-inline">
                        <span><strong>${response.n_diseases}</strong> diseases</span>
                        <span><strong>${response.n_significant}</strong> significant (FDR < 0.05)</span>
                    </div>
                </div>
                <div class="viz-container">
                    <div id="volcano-chart" class="chart-container"></div>
                </div>
                <div class="data-table-container">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Disease</th>
                                <th>Group</th>
                                <th>&Delta; Activity</th>
                                <th>P-value</th>
                                <th>Significant</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${data.map(d => `
                                <tr class="${d.is_significant ? 'significant' : ''}">
                                    <td>${d.disease}</td>
                                    <td>${d.disease_group}</td>
                                    <td class="${d.activity_diff >= 0 ? 'positive' : 'negative'}">${d.activity_diff.toFixed(4)}</td>
                                    <td>${d.pvalue < 0.001 ? d.pvalue.toExponential(2) : d.pvalue.toFixed(4)}</td>
                                    <td>${d.is_significant ? '&#10003;' : ''}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `;

            // Create volcano plot
            this.createVolcanoPlot(data);

        } catch (error) {
            content.innerHTML = `
                <div class="no-data-panel">
                    <h3>No Disease Data</h3>
                    <p>Disease differential data is not available for <strong>${this.gene}</strong>.</p>
                </div>
            `;
        }
    },

    createVolcanoPlot(data) {
        const sigData = data.filter(d => d.is_significant);
        const nsData = data.filter(d => !d.is_significant);

        const traces = [];

        if (nsData.length) {
            traces.push({
                type: 'scatter',
                mode: 'markers',
                name: 'Not Significant',
                x: nsData.map(d => d.activity_diff),
                y: nsData.map(d => d.neg_log10_pval || -Math.log10(d.pvalue)),
                text: nsData.map(d => d.disease),
                marker: { color: '#999', size: 8 },
                hovertemplate: '<b>%{text}</b><br>&Delta; Activity: %{x:.3f}<extra></extra>',
            });
        }

        if (sigData.length) {
            traces.push({
                type: 'scatter',
                mode: 'markers+text',
                name: 'Significant (FDR < 0.05)',
                x: sigData.map(d => d.activity_diff),
                y: sigData.map(d => d.neg_log10_pval || -Math.log10(d.pvalue)),
                text: sigData.map(d => d.disease),
                textposition: 'top center',
                textfont: { size: 10 },
                marker: {
                    color: sigData.map(d => d.activity_diff >= 0 ? '#b2182b' : '#2166ac'),
                    size: 12,
                },
                hovertemplate: '<b>%{text}</b><br>&Delta; Activity: %{x:.3f}<extra></extra>',
            });
        }

        const layout = {
            title: `${this.gene} Disease Volcano Plot`,
            xaxis: { title: '&Delta; Activity (disease - healthy)', zeroline: true },
            yaxis: { title: '-log10(p-value)' },
            height: 500,
            showlegend: true,
            legend: { orientation: 'h', y: -0.15 },
        };

        Plotly.newPlot('volcano-chart', traces, layout, { responsive: true });
    },

    // ==================== Correlations Tab ====================

    async loadCorrelationsTab(content) {
        try {
            const data = await API.get(`/gene/${encodeURIComponent(this.gene)}/correlations`, {
                signature_type: this.data.has_cytosig ? 'CytoSig' : 'SecAct',
            });

            // Combine all correlations for visualization
            const allCorrelations = [
                ...(data.age || []).map(c => ({ ...c, category: 'Age' })),
                ...(data.bmi || []).map(c => ({ ...c, category: 'BMI' })),
                ...(data.biochemistry || []).map(c => ({ ...c, category: 'Biochemistry' })),
                ...(data.metabolites || []).map(c => ({ ...c, category: 'Metabolites' })),
            ];

            // Calculate significant count from actual data (handles both q_value and qvalue field names)
            const totalSignificant = allCorrelations.filter(c => {
                const qval = c.q_value ?? c.qvalue;
                return qval != null && qval < 0.05;
            }).length;

            content.innerHTML = `
                <div class="tab-header">
                    <h2>${this.gene} Correlations</h2>
                    <p>Correlations with clinical and molecular variables (Spearman rho, from CIMA atlas)</p>
                    <div class="stats-inline">
                        <span><strong>${allCorrelations.length}</strong> total correlations</span>
                        <span><strong>${totalSignificant}</strong> significant (FDR < 0.05)</span>
                    </div>
                    <div class="tab-filters">
                        <select id="corr-category-filter" onchange="GeneDetailPage.filterCorrelationsByCategory()">
                            <option value="all">All Categories</option>
                            <option value="Age">Age</option>
                            <option value="BMI">BMI</option>
                            <option value="Biochemistry">Biochemistry</option>
                            <option value="Metabolites">Metabolites</option>
                        </select>
                        <label class="filter-checkbox">
                            <input type="checkbox" id="corr-sig-only" onchange="GeneDetailPage.filterCorrelationsByCategory()">
                            Show significant only
                        </label>
                    </div>
                </div>
                <div class="viz-container">
                    <div id="correlation-chart" class="chart-container"></div>
                </div>
                <div class="data-table-container">
                    <table class="data-table" id="correlation-table">
                        <thead>
                            <tr>
                                <th>Variable</th>
                                <th>Category</th>
                                <th>Cell Type</th>
                                <th>Spearman &rho;</th>
                                <th>P-value</th>
                                <th>Q-value (FDR)</th>
                                <th>Sig</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${allCorrelations
                                .sort((a, b) => {
                                    // Sort by significance first, then by absolute rho
                                    const qvalA = a.q_value ?? a.qvalue ?? 1;
                                    const qvalB = b.q_value ?? b.qvalue ?? 1;
                                    const sigA = qvalA < 0.05 ? 1 : 0;
                                    const sigB = qvalB < 0.05 ? 1 : 0;
                                    if (sigA !== sigB) return sigB - sigA;
                                    return Math.abs(b.rho || 0) - Math.abs(a.rho || 0);
                                })
                                .slice(0, 50).map(c => {
                                const qval = c.q_value ?? c.qvalue;
                                const isSignificant = qval != null && qval < 0.05;
                                const pval = c.p_value || c.pvalue;
                                return `
                                    <tr class="${isSignificant ? 'significant' : ''}">
                                        <td>${c.variable || c.measure || '-'}</td>
                                        <td>${c.category}</td>
                                        <td>${c.cell_type || 'All'}</td>
                                        <td class="${c.rho >= 0 ? 'positive' : 'negative'}">${c.rho ? c.rho.toFixed(4) : '-'}</td>
                                        <td>${pval ? (pval < 0.001 ? pval.toExponential(2) : pval.toFixed(4)) : '-'}</td>
                                        <td>${qval != null ? (qval < 0.001 ? qval.toExponential(2) : qval.toFixed(4)) : '-'}</td>
                                        <td>${isSignificant ? '&#10003;' : ''}</td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                </div>
            `;

            this._correlationData = allCorrelations;
            this.createCorrelationChart(allCorrelations);

        } catch (error) {
            content.innerHTML = `
                <div class="no-data-panel">
                    <h3>No Correlation Data</h3>
                    <p>Correlation data is not available for <strong>${this.gene}</strong>.</p>
                </div>
            `;
        }
    },

    createCorrelationChart(data) {
        const colors = {
            'Age': '#3498db',
            'BMI': '#e74c3c',
            'Biochemistry': '#2ecc71',
            'Metabolites': '#9b59b6',
        };

        // Sort by absolute rho and get top variables
        const sortedData = [...data].sort((a, b) => Math.abs(b.rho || 0) - Math.abs(a.rho || 0));
        const topData = sortedData.slice(0, 30);

        // Create horizontal bar chart with color by category
        const trace = {
            type: 'bar',
            x: topData.map(d => d.rho || 0),
            y: topData.map(d => `${d.variable || d.measure}${d.cell_type && d.cell_type !== 'All' ? ` (${d.cell_type})` : ''}`),
            orientation: 'h',
            marker: {
                color: topData.map(d => colors[d.category] || '#95a5a6'),
            },
            text: topData.map(d => d.category),
            hovertemplate: '<b>%{y}</b><br>ρ = %{x:.4f}<br>%{text}<extra></extra>',
        };

        // Create legend entries for categories
        const legendTraces = Object.keys(colors).map(cat => ({
            type: 'bar',
            x: [null],
            y: [null],
            name: cat,
            marker: { color: colors[cat] },
            showlegend: true,
        }));

        const layout = {
            title: `${this.gene} Top Correlations`,
            xaxis: {
                title: 'Spearman ρ',
                zeroline: true,
                zerolinecolor: '#999',
                range: [-1, 1],
            },
            yaxis: {
                title: '',
                automargin: true,
            },
            height: Math.max(400, topData.length * 22),
            margin: { l: 200, r: 50, t: 50, b: 50 },
            legend: { orientation: 'h', y: 1.1 },
            showlegend: true,
        };

        Plotly.newPlot('correlation-chart', [trace, ...legendTraces], layout, { responsive: true });
    },

    filterCorrelationsByCategory() {
        const category = document.getElementById('corr-category-filter').value;
        const sigOnly = document.getElementById('corr-sig-only').checked;

        let filtered = this._correlationData;

        if (category !== 'all') {
            filtered = filtered.filter(c => c.category === category);
        }

        if (sigOnly) {
            filtered = filtered.filter(c => {
                const qval = c.q_value ?? c.qvalue;
                return qval != null && qval < 0.05;
            });
        }

        this.createCorrelationChart(filtered);

        // Update table
        const tbody = document.querySelector('#correlation-table tbody');
        if (tbody) {
            // Sort by significance first, then by absolute rho
            const sorted = [...filtered].sort((a, b) => {
                const qvalA = a.q_value ?? a.qvalue ?? 1;
                const qvalB = b.q_value ?? b.qvalue ?? 1;
                const sigA = qvalA < 0.05 ? 1 : 0;
                const sigB = qvalB < 0.05 ? 1 : 0;
                if (sigA !== sigB) return sigB - sigA;
                return Math.abs(b.rho || 0) - Math.abs(a.rho || 0);
            });

            tbody.innerHTML = sorted.slice(0, 50).map(c => {
                const qval = c.q_value ?? c.qvalue;
                const isSignificant = qval != null && qval < 0.05;
                const pval = c.p_value || c.pvalue;
                return `
                    <tr class="${isSignificant ? 'significant' : ''}">
                        <td>${c.variable || c.measure || '-'}</td>
                        <td>${c.category}</td>
                        <td>${c.cell_type || 'All'}</td>
                        <td class="${c.rho >= 0 ? 'positive' : 'negative'}">${c.rho ? c.rho.toFixed(4) : '-'}</td>
                        <td>${pval ? (pval < 0.001 ? pval.toExponential(2) : pval.toFixed(4)) : '-'}</td>
                        <td>${qval != null ? (qval < 0.001 ? qval.toExponential(2) : qval.toFixed(4)) : '-'}</td>
                        <td>${isSignificant ? '&#10003;' : ''}</td>
                    </tr>
                `;
            }).join('');
        }
    },
};

// Make available globally
window.GeneDetailPage = GeneDetailPage;
