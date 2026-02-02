/**
 * Gene Detail Page Handler
 * Displays gene expression + CytoSig + SecAct data across all atlases
 */

const GeneDetailPage = {
    gene: null,
    activeTab: 'expression',
    data: null,

    // Brief gene descriptions for common cytokines and immune genes
    geneDescriptions: {
        'IFNG': 'Interferon gamma (IFN-γ) is a key cytokine for innate and adaptive immunity. It activates macrophages, enhances antigen presentation, and promotes Th1 responses. Critical for defense against viral and intracellular bacterial infections.',
        'TNF': 'Tumor necrosis factor (TNF-α) is a pro-inflammatory cytokine produced mainly by macrophages. It regulates immune cells, induces fever, apoptosis, and inflammation. Implicated in autoimmune diseases like rheumatoid arthritis.',
        'IL6': 'Interleukin-6 is a pleiotropic cytokine with both pro- and anti-inflammatory effects. It regulates immune responses, acute phase reactions, and hematopoiesis. Elevated in inflammatory conditions and COVID-19.',
        'IL17A': 'Interleukin-17A is the signature cytokine of Th17 cells. It promotes neutrophil recruitment and antimicrobial defense at mucosal surfaces. Implicated in autoimmune diseases like psoriasis and multiple sclerosis.',
        'IL10': 'Interleukin-10 is an anti-inflammatory cytokine that limits immune responses. It inhibits pro-inflammatory cytokine production and antigen presentation. Important for preventing autoimmunity and tissue damage.',
        'TGFB1': 'Transforming growth factor beta 1 regulates cell growth, differentiation, and immune function. It promotes regulatory T cell development and has both immunosuppressive and pro-fibrotic effects.',
        'IL1B': 'Interleukin-1 beta is a potent pro-inflammatory cytokine. It induces fever, activates endothelium, and promotes inflammation. Key mediator in autoinflammatory diseases and inflammasome activation.',
        'IL2': 'Interleukin-2 is essential for T cell proliferation and survival. It promotes both effector and regulatory T cell responses. Used therapeutically in cancer immunotherapy.',
        'IL4': 'Interleukin-4 drives Th2 differentiation and IgE class switching. It promotes alternative macrophage activation and is central to allergic responses and helminth immunity.',
        'IL12A': 'Interleukin-12 (p35 subunit) promotes Th1 differentiation and IFN-γ production. Critical for cell-mediated immunity against intracellular pathogens.',
        'IL12B': 'Interleukin-12 (p40 subunit) shared with IL-23. Promotes Th1 responses and NK cell activation. Target for treatment of psoriasis and inflammatory bowel disease.',
        'IL23A': 'Interleukin-23 (p19 subunit) maintains Th17 cell responses. Important in mucosal immunity and implicated in autoimmune diseases like psoriasis and Crohn\'s disease.',
        'CCL2': 'C-C motif chemokine ligand 2 (MCP-1) recruits monocytes, memory T cells, and dendritic cells. Key mediator of inflammation and monocyte trafficking to tissues.',
        'CXCL10': 'C-X-C motif chemokine ligand 10 (IP-10) attracts activated T cells and NK cells. Induced by IFN-γ, it promotes Th1 responses and antiviral immunity.',
        'CXCL8': 'Interleukin-8 (CXCL8) is a major neutrophil chemoattractant. It promotes neutrophil recruitment, activation, and degranulation during acute inflammation.',
        'IL18': 'Interleukin-18 synergizes with IL-12 to induce IFN-γ production. It activates NK cells and Th1 responses. Involved in inflammasome-mediated inflammation.',
        'IL21': 'Interleukin-21 regulates B cell differentiation, germinal center formation, and plasma cell development. It also enhances NK cell and CD8 T cell cytotoxicity.',
        'IL22': 'Interleukin-22 promotes epithelial barrier function and antimicrobial defense. Produced by Th17/Th22 cells, it protects mucosal surfaces but can drive pathology in psoriasis.',
        'VEGFA': 'Vascular endothelial growth factor A promotes angiogenesis and vascular permeability. Key regulator of blood vessel formation in development, wound healing, and tumors.',
        'CSF2': 'Granulocyte-macrophage colony-stimulating factor (GM-CSF) promotes myeloid cell differentiation and activation. Implicated in inflammatory diseases and used therapeutically.',
    },

    tabs: [
        { id: 'expression', label: 'Gene Expression', icon: '&#129516;' },
        { id: 'cytosig', label: 'CytoSig Activity', icon: '&#128300;' },
        { id: 'secact', label: 'SecAct Activity', icon: '&#9898;' },
        { id: 'diseases', label: 'Diseases', icon: '&#129658;' },
        { id: 'correlations', label: 'Correlations', icon: '&#128200;' },
    ],

    /**
     * Initialize the gene detail page
     */
    async init(params, query) {
        this.gene = decodeURIComponent(params.signature);
        this.activeTab = query.tab || 'expression';

        this.render();
        await this.loadGeneData();
    },

    /**
     * Get external database links for a gene
     */
    getExternalLinks(gene) {
        return [
            {
                name: 'NCBI Gene',
                url: `https://www.ncbi.nlm.nih.gov/gene/?term=${encodeURIComponent(gene)}[sym]+AND+human[orgn]`,
                icon: '🔬',
            },
            {
                name: 'UniProt',
                url: `https://www.uniprot.org/uniprotkb?query=${encodeURIComponent(gene)}+AND+organism_id:9606`,
                icon: '🧬',
            },
            {
                name: 'GeneCards',
                url: `https://www.genecards.org/cgi-bin/carddisp.pl?gene=${encodeURIComponent(gene)}`,
                icon: '📋',
            },
            {
                name: 'Ensembl',
                url: `https://www.ensembl.org/Human/Search/Results?q=${encodeURIComponent(gene)};site=ensembl;facet_species=Human`,
                icon: '🌐',
            },
            {
                name: 'HGNC',
                url: `https://www.genenames.org/tools/search/#!/?query=${encodeURIComponent(gene)}`,
                icon: '🏷️',
            },
            {
                name: 'GO',
                url: `https://amigo.geneontology.org/amigo/search/bioentity?q=${encodeURIComponent(gene)}`,
                icon: '🔗',
            },
        ];
    },

    /**
     * Render the page template
     */
    render() {
        const app = document.getElementById('app');
        const description = this.geneDescriptions[this.gene] || null;
        const externalLinks = this.getExternalLinks(this.gene);

        app.innerHTML = `
            <div class="gene-detail-page">
                <header class="gene-header">
                    <div class="header-top">
                        <a href="/search" class="back-link">&#8592; Back to Search</a>
                    </div>
                    <div id="gene-title" class="gene-title">
                        <h1>${this.gene}</h1>
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
     * Update the summary section
     */
    updateSummary() {
        const summaryEl = document.getElementById('gene-summary');
        if (!summaryEl || !this.data) return;

        const badges = [];
        if (this.data.has_expression) badges.push('<span class="data-badge expression">Expression</span>');
        if (this.data.has_cytosig) badges.push('<span class="data-badge cytosig">CytoSig</span>');
        if (this.data.has_secact) badges.push('<span class="data-badge secact">SecAct</span>');

        summaryEl.innerHTML = `
            <div class="data-availability">
                <span class="label">Available data:</span>
                ${badges.length ? badges.join('') : '<span class="no-data">No data available</span>'}
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
        this.createExpressionChart(expr.data);
    },

    createExpressionChart(data) {
        const topData = data.slice(0, 30);

        const traces = [];
        const atlases = [...new Set(topData.map(d => d.atlas))];

        const colors = {
            'CIMA': '#3498db',
            'Inflammation': '#e74c3c',
            'scAtlas_Normal': '#2ecc71',
            'scAtlas_Cancer': '#9b59b6',
        };

        atlases.forEach(atlas => {
            const atlasData = topData.filter(d => d.atlas === atlas);
            traces.push({
                type: 'bar',
                name: atlas,
                y: atlasData.map(d => d.cell_type),
                x: atlasData.map(d => d.mean_expression),
                orientation: 'h',
                marker: { color: colors[atlas] || '#95a5a6' },
                text: atlasData.map(d => `${d.mean_expression.toFixed(2)} (${d.pct_expressed.toFixed(0)}%)`),
                hovertemplate: '<b>%{y}</b><br>Expression: %{x:.3f}<br>%{text}<extra></extra>',
            });
        });

        const layout = {
            title: `${this.gene} Expression by Cell Type`,
            barmode: 'group',
            xaxis: {
                title: 'Mean Expression (log-normalized)',
            },
            yaxis: {
                title: '',
                automargin: true,
            },
            height: Math.max(400, topData.length * 20),
            margin: { l: 200, r: 50, t: 50, b: 50 },
            legend: { orientation: 'h', y: 1.1 },
        };

        Plotly.newPlot('expression-chart', traces, layout, { responsive: true });
    },

    filterExpressionByAtlas() {
        const atlas = document.getElementById('expr-atlas-filter').value;
        let filtered = this._expressionData;

        if (atlas !== 'all') {
            filtered = filtered.filter(d => d.atlas === atlas);
        }

        this.createExpressionChart(filtered);

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
                <p>Inferred ${sigType === 'CytoSig' ? 'cytokine' : 'secreted protein'} activity by cell type</p>
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
        this.createActivityChart(activityData, sigType);
    },

    createActivityChart(data, sigType) {
        const topData = data.slice(0, 30);

        const traces = [];
        const atlases = [...new Set(topData.map(d => d.atlas))];

        atlases.forEach(atlas => {
            const atlasData = topData.filter(d => d.atlas === atlas);
            traces.push({
                type: 'bar',
                name: atlas,
                y: atlasData.map(d => d.cell_type),
                x: atlasData.map(d => d.mean_activity),
                orientation: 'h',
                text: atlasData.map(d => d.mean_activity.toFixed(3)),
                textposition: 'auto',
                hovertemplate: '<b>%{y}</b><br>Activity: %{x:.4f}<extra></extra>',
            });
        });

        const layout = {
            title: `${this.gene} ${sigType} Activity by Cell Type`,
            barmode: 'group',
            xaxis: {
                title: 'Mean Activity (z-score)',
                zeroline: true,
                zerolinecolor: '#999',
            },
            yaxis: {
                title: '',
                automargin: true,
            },
            height: Math.max(400, topData.length * 20),
            margin: { l: 200, r: 50, t: 50, b: 50 },
            legend: { orientation: 'h', y: 1.1 },
        };

        Plotly.newPlot('activity-chart', traces, layout, { responsive: true });
    },

    filterActivityByAtlas(sigType) {
        const atlas = document.getElementById('activity-atlas-filter').value;
        let filtered = this._activityData;

        if (atlas !== 'all') {
            filtered = filtered.filter(d => d.atlas === atlas);
        }

        this.createActivityChart(filtered, sigType);

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
                                    <td>${d.p_value < 0.001 ? d.p_value.toExponential(2) : d.p_value.toFixed(4)}</td>
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
                y: nsData.map(d => d.neg_log10_pval || -Math.log10(d.p_value)),
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
                y: sigData.map(d => d.neg_log10_pval || -Math.log10(d.p_value)),
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

            content.innerHTML = `
                <div class="tab-header">
                    <h2>${this.gene} Correlations</h2>
                    <p>Correlations with clinical and molecular variables (from CIMA)</p>
                </div>
                <div class="correlation-sections">
                    <div class="correlation-section">
                        <h3>&#128197; Age (${data.n_significant_age} significant)</h3>
                        ${this.renderCorrelationList(data.age)}
                    </div>
                    <div class="correlation-section">
                        <h3>&#9878; BMI (${data.n_significant_bmi} significant)</h3>
                        ${this.renderCorrelationList(data.bmi)}
                    </div>
                    <div class="correlation-section">
                        <h3>&#129514; Biochemistry (${data.n_significant_biochem} significant)</h3>
                        ${this.renderCorrelationList(data.biochemistry)}
                    </div>
                    <div class="correlation-section">
                        <h3>&#9879; Metabolites (${data.n_significant_metabol} significant)</h3>
                        ${this.renderCorrelationList(data.metabolites)}
                    </div>
                </div>
            `;
        } catch (error) {
            content.innerHTML = `
                <div class="no-data-panel">
                    <h3>No Correlation Data</h3>
                    <p>Correlation data is not available for <strong>${this.gene}</strong>.</p>
                </div>
            `;
        }
    },

    renderCorrelationList(correlations) {
        if (!correlations || !correlations.length) {
            return '<p class="no-data">No data available.</p>';
        }

        const topCorr = correlations.slice(0, 10);

        return `
            <div class="correlation-list">
                ${topCorr.map(c => {
                    const isSignificant = c.q_value && c.q_value < 0.05;
                    return `
                        <div class="correlation-item ${isSignificant ? 'significant' : ''}">
                            <span class="variable">${c.variable}${c.cell_type && c.cell_type !== 'All' ? ` (${c.cell_type})` : ''}</span>
                            <span class="rho ${c.rho >= 0 ? 'positive' : 'negative'}">&rho; = ${c.rho.toFixed(3)}</span>
                            ${isSignificant ? '<span class="sig-badge">*</span>' : ''}
                        </div>
                    `;
                }).join('')}
            </div>
            ${correlations.length > 10 ? `<p class="more-link">...and ${correlations.length - 10} more</p>` : ''}
        `;
    },
};

// Make available globally
window.GeneDetailPage = GeneDetailPage;
