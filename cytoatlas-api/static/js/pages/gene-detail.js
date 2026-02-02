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
        const cellTypeScores = {};

        // From expression data
        if (this.data.expression && this.data.expression.data) {
            this.data.expression.data.forEach(d => {
                if (!cellTypeScores[d.cell_type]) {
                    cellTypeScores[d.cell_type] = { expression: 0, cytosig: 0, secact: 0 };
                }
                cellTypeScores[d.cell_type].expression = Math.max(
                    cellTypeScores[d.cell_type].expression,
                    d.mean_expression || 0
                );
            });
        }

        // From CytoSig activity
        if (this.data.cytosig_activity) {
            this.data.cytosig_activity.forEach(d => {
                if (!cellTypeScores[d.cell_type]) {
                    cellTypeScores[d.cell_type] = { expression: 0, cytosig: 0, secact: 0 };
                }
                cellTypeScores[d.cell_type].cytosig = Math.max(
                    cellTypeScores[d.cell_type].cytosig,
                    Math.abs(d.mean_activity) || 0
                );
            });
        }

        // From SecAct activity
        if (this.data.secact_activity) {
            this.data.secact_activity.forEach(d => {
                if (!cellTypeScores[d.cell_type]) {
                    cellTypeScores[d.cell_type] = { expression: 0, cytosig: 0, secact: 0 };
                }
                cellTypeScores[d.cell_type].secact = Math.max(
                    cellTypeScores[d.cell_type].secact,
                    Math.abs(d.mean_activity) || 0
                );
            });
        }

        // Group cell types by atlas and get top from each for balanced representation
        const atlasCellTypes = {};

        // Determine which atlas each cell type belongs to (from activity data)
        if (this.data.cytosig_activity) {
            this.data.cytosig_activity.forEach(d => {
                const atlas = d.atlas;
                if (!atlasCellTypes[atlas]) atlasCellTypes[atlas] = new Set();
                atlasCellTypes[atlas].add(d.cell_type);
            });
        }

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
        if (this.data.has_secact) badges.push('<span class="data-badge secact">SecAct</span>');

        summaryEl.innerHTML = `
            <div class="data-availability">
                <span class="label">Available data:</span>
                ${badges.length ? badges.join(', ') : '<span class="no-data">No data available</span>'}
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
            content.innerHTML = `
                <div class="no-data-panel">
                    <h3>No ${sigType} Activity</h3>
                    <p><strong>${this.gene}</strong> is not available as a ${sigType} signature.</p>
                    <p>${sigType === 'CytoSig' ? 'CytoSig contains 43 major cytokines.' : 'SecAct contains ~1,170 secreted proteins.'}</p>
                </div>
            `;
            return;
        }

        // Group by atlas
        const byAtlas = {};
        activityData.forEach(d => {
            if (!byAtlas[d.atlas]) byAtlas[d.atlas] = [];
            byAtlas[d.atlas].push(d);
        });

        content.innerHTML = `
            <div class="tab-header">
                <h2>${this.gene} ${sigType} Activity</h2>
                <p>Pseudobulk-aggregated ${sigType === 'CytoSig' ? 'cytokine' : 'secreted protein'} activity by cell type (ridge regression z-scores)</p>
                <div class="stats-inline">
                    <span><strong>${activityData.length}</strong> cell types</span>
                    <span><strong>${Object.keys(byAtlas).length}</strong> atlases</span>
                </div>
                <div class="tab-filters">
                    <select id="activity-atlas-filter" onchange="GeneDetailPage.filterActivityByAtlas('${sigType}')">
                        <option value="all">All Atlases</option>
                        ${Object.keys(byAtlas).map(a => `<option value="${a}">${a}</option>`).join('')}
                    </select>
                </div>
            </div>

            <div class="method-explanation">
                <details>
                    <summary><strong>Pseudobulk Method</strong> - How activity is computed</summary>
                    <div class="method-content">
                        <strong>Pseudobulk</strong> aggregates cells by cell type and sample before computing activity.
                        This reduces noise from technical variation and dropout, providing robust cell-type-level estimates.
                        <ul>
                            <li><strong>Step 1:</strong> Group cells by (cell_type, sample_id)</li>
                            <li><strong>Step 2:</strong> Sum raw counts within each group</li>
                            <li><strong>Step 3:</strong> Normalize to CPM and log-transform</li>
                            <li><strong>Step 4:</strong> Apply ridge regression with ${sigType} signature matrix</li>
                            <li><strong>Step 5:</strong> Z-score normalize activity values</li>
                        </ul>
                        <em>Best for:</em> Comparing cell types, identifying signature-producing cell populations.
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
                        ${activityData.slice(0, 50).map(d => `
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

        this._activityData = activityData;
        this._activitySigType = sigType;
        const boxplotData = sigType === 'CytoSig' ? this.data.cytosig_boxplot : this.data.secact_boxplot;
        this.createActivityChart(activityData, sigType, boxplotData);
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

            if (cellTypesWithBoxplot.length > 0) {
                this.createBoxPlotChart('activity-chart', boxplotData.data, cellTypesWithBoxplot, colors,
                    `${this.gene} ${sigType} Activity by Cell Type`, 'Activity (z-score)');
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

    filterActivityByAtlas(sigType) {
        const atlas = document.getElementById('activity-atlas-filter').value;
        let filtered = this._activityData;
        let boxplotData = sigType === 'CytoSig' ? this.data.cytosig_boxplot : this.data.secact_boxplot;

        if (atlas !== 'all') {
            filtered = filtered.filter(d => d.atlas === atlas);
            if (boxplotData && boxplotData.data) {
                boxplotData = {
                    ...boxplotData,
                    data: boxplotData.data.filter(d => d.atlas === atlas),
                };
            }
        }

        this.createActivityChart(filtered, sigType, boxplotData);

        const tbody = document.querySelector('#activity-table tbody');
        if (tbody) {
            tbody.innerHTML = filtered.slice(0, 50).map(d => `
                <tr>
                    <td>${d.cell_type}</td>
                    <td><span class="atlas-badge">${d.atlas}</span></td>
                    <td class="${d.mean_activity >= 0 ? 'positive' : 'negative'}">${d.mean_activity.toFixed(4)}</td>
                    <td>${d.n_cells ? d.n_cells.toLocaleString() : '-'}</td>
                </tr>
            `).join('');
        }
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
                    color: sigData.map(d => d.activity_diff >= 0 ? '#2ecc71' : '#e74c3c'),
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
