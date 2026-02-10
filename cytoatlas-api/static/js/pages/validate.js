/**
 * Validation Page Handler — 5-Tab Implementation
 *
 * Tab 1: Summary — Boxplots of per-entity rho (from validation_corr_boxplot.json)
 * Tab 2: Bulk RNA-seq — GTEx/TCGA scatter with tissue/cancer colors
 * Tab 3: Donor Level — Cross-sample correlations per atlas
 * Tab 4: Cell Type Level — Celltype-stratified correlations
 * Tab 5: Single-Cell — Density heatmap (ALL cells) + 50K scatter overlay
 */

const ValidatePage = {
    // Global state
    activeTab: 'summary',

    // Client-side response cache
    _cache: new Map(),
    _cacheMaxSize: 200,

    // Debounce timer
    _debounceTimer: null,

    // Level configs matching index.html
    DONOR_LEVEL_CONFIG: {
        cima: {
            levels: { donor_only: 'Donor Only', donor_l1: 'L1', donor_l2: 'L2', donor_l3: 'L3', donor_l4: 'L4' },
            groupLabel: { donor_only: '', donor_l1: 'Cell Type', donor_l2: 'Cell Type', donor_l3: 'Cell Type', donor_l4: 'Cell Type' },
        },
        inflammation_main: {
            levels: { donor_only: 'Donor Only', donor_l1: 'L1', donor_l2: 'L2' },
            groupLabel: { donor_only: '', donor_l1: 'Cell Type', donor_l2: 'Cell Type' },
        },
        inflammation_val: {
            levels: { donor_only: 'Donor Only', donor_l1: 'L1', donor_l2: 'L2' },
            groupLabel: { donor_only: '', donor_l1: 'Cell Type', donor_l2: 'Cell Type' },
        },
        inflammation_ext: {
            levels: { donor_only: 'Donor Only', donor_l1: 'L1', donor_l2: 'L2' },
            groupLabel: { donor_only: '', donor_l1: 'Cell Type', donor_l2: 'Cell Type' },
        },
        scatlas_normal: {
            levels: { donor_organ: 'By Organ' },
            groupLabel: 'Organ',
        },
        scatlas_cancer: {
            levels: { donor_organ: 'By Sample Type' },
            groupLabel: 'Sample Type',
        },
    },

    CELLTYPE_LEVEL_CONFIG: {
        cima: {
            levels: { l1: 'L1', l2: 'L2', l3: 'L3', l4: 'L4' },
            groupLabel: 'Cell Type',
        },
        inflammation_main: {
            levels: { l1: 'L1', l2: 'L2' },
            groupLabel: 'Cell Type',
        },
        inflammation_val: {
            levels: { l1: 'L1', l2: 'L2' },
            groupLabel: 'Cell Type',
        },
        inflammation_ext: {
            levels: { l1: 'L1', l2: 'L2' },
            groupLabel: 'Cell Type',
        },
        scatlas_normal: {
            levels: { celltype: 'Cell Type', organ_celltype: 'Organ/Cell Type' },
            groupLabel: { celltype: 'Cell Type', organ_celltype: 'Organ/Cell Type' },
        },
        scatlas_cancer: {
            levels: { celltype: 'Cell Type', organ_celltype: 'Sample/Cell Type' },
            groupLabel: { celltype: 'Cell Type', organ_celltype: 'Sample/Cell Type' },
        },
    },

    // Per-tab atlas options
    SC_ATLAS_OPTIONS: [
        { value: 'cima', label: 'CIMA (6.5M cells)' },
        { value: 'inflammation_main', label: 'Inflammation Main (4.9M)' },
        { value: 'inflammation_val', label: 'Inflammation Val' },
        { value: 'inflammation_ext', label: 'Inflammation Ext' },
        { value: 'scatlas_normal', label: 'scAtlas Normal (6.4M)' },
        { value: 'scatlas_cancer', label: 'scAtlas Cancer (6.4M)' },
    ],
    SC_FULL_ATLAS_OPTIONS: [
        { value: 'cima', label: 'CIMA (6.5M cells)' },
        { value: 'scatlas', label: 'scAtlas (12.8M cells)' },
        { value: 'inflammation', label: 'Inflammation (6.3M cells)' },
    ],

    // Per-tab signature type options
    ALL_SIG_OPTIONS: [
        { value: 'cytosig', label: 'CytoSig (43)' },
        { value: 'lincytosig', label: 'LinCytoSig' },
        { value: 'secact', label: 'SecAct (1,249)' },
    ],
    BULK_SIG_OPTIONS: [
        { value: 'cytosig', label: 'CytoSig (43)' },
        { value: 'secact', label: 'SecAct (1,249)' },
    ],

    // Tab-specific state (each tab tracks its own atlas + sigtype)
    summary: { sigtype: 'cytosig' },
    bulkRnaseq: { sigtype: 'cytosig', dataset: 'gtex', group: '', target: null, targets: [], hideNonExpr: false },
    donorLevel: { atlas: 'cima', sigtype: 'cytosig', level: null, group: '', target: null, targets: [], hideNonExpr: false },
    celltypeLevel: { atlas: 'cima', sigtype: 'cytosig', level: null, group: '', target: null, targets: [], hideNonExpr: false },
    singleCell: { atlas: 'cima', sigtype: 'cytosig', target: null, targets: [], donor: '', celltype: '', hideNonExpr: false },

    // ==================== Initialization ====================

    async init(params, query) {
        if (query.tab) this.activeTab = query.tab;

        this.render();
        this.setupEventListeners();
        await this.loadTab(this.activeTab);
    },

    render() {
        const app = document.getElementById('app');
        const template = document.getElementById('validate-template');
        if (app && template) app.innerHTML = template.innerHTML;

        // Activate correct tab button
        const tabBtns = document.querySelectorAll('#validation-tabs .tab-btn');
        tabBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === this.activeTab);
        });
    },

    setupEventListeners() {
        const tabBtns = document.querySelectorAll('#validation-tabs .tab-btn');
        tabBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });
    },

    async switchTab(tab) {
        this.activeTab = tab;
        const tabBtns = document.querySelectorAll('#validation-tabs .tab-btn');
        tabBtns.forEach(btn => btn.classList.toggle('active', btn.dataset.tab === tab));
        await this.loadTab(tab);
    },

    async loadTab(tab) {
        const content = document.getElementById('validation-content');
        if (!content) return;
        content.innerHTML = '<div class="loading"><div class="spinner"></div>Loading...</div>';

        try {
            switch (tab) {
                case 'summary': await this.loadSummaryTab(); break;
                case 'bulk-rnaseq': await this.loadBulkRnaseqTab(); break;
                case 'donor-level': await this.loadDonorLevelTab(); break;
                case 'celltype-level': await this.loadCelltypeLevelTab(); break;
                case 'singlecell': await this.loadSingleCellTab(); break;
            }
        } catch (err) {
            content.innerHTML = `<div class="tab-panel"><p class="error">Error loading tab: ${err.message}</p></div>`;
            console.error('Tab load error:', err);
        }
    },

    // ==================== Cached API Fetch ====================

    async cachedFetch(fetchFn, cacheKey) {
        if (this._cache.has(cacheKey)) return this._cache.get(cacheKey);
        const data = await fetchFn();
        if (this._cache.size > this._cacheMaxSize) {
            const firstKey = this._cache.keys().next().value;
            this._cache.delete(firstKey);
        }
        this._cache.set(cacheKey, data);
        return data;
    },

    debounced(fn, delay = 300) {
        clearTimeout(this._debounceTimer);
        this._debounceTimer = setTimeout(fn, delay);
    },

    _atlasSelectHTML(id, options, selected) {
        return `<label>Atlas: <select id="${id}">` +
            options.map(o => `<option value="${o.value}"${o.value === selected ? ' selected' : ''}>${o.label}</option>`).join('') +
            `</select></label>`;
    },

    _sigtypeSelectHTML(id, options, selected) {
        return `<label>Signature: <select id="${id}">` +
            options.map(o => `<option value="${o.value}"${o.value === selected ? ' selected' : ''}>${o.label}</option>`).join('') +
            `</select></label>`;
    },

    // ==================== Tab 1: Summary ====================

    async loadSummaryTab() {
        const content = document.getElementById('validation-content');
        content.innerHTML = `
            <div class="tab-panel">
                <h3>Validation Summary</h3>
                <p>Distribution of Spearman correlations (rho) between signature gene expression and predicted cytokine/secreted protein activity. Each box shows the per-target rho distribution across entities (donors, tissues, or cell types) at a given aggregation level. Higher rho indicates stronger agreement between expression and inferred activity. CytoSig (43 cytokines, blue) and SecAct (1,249 secreted proteins, orange) are shown side-by-side.</p>

                <div class="filter-bar">
                    <label>Target:
                        <select id="val-summary-target">
                            <option value="_all">All Targets</option>
                        </select>
                    </label>
                    <label>
                        <input type="text" id="val-summary-search" placeholder="Search target...">
                    </label>
                </div>

                <div class="panel-grid">
                    <div class="panel" style="grid-column: span 2;">
                        <div class="viz-title">Correlation Distributions — CytoSig vs SecAct</div>
                        <div class="viz-subtitle">Per-entity Spearman rho distribution</div>
                        <div id="val-summary-box" class="plot-container" style="height: 500px;"></div>
                    </div>
                </div>

                <div class="panel-grid">
                    <div class="panel" style="grid-column: span 2;">
                        <div class="viz-title">Method Comparison — CytoSig vs LinCytoSig vs SecAct</div>
                        <div class="viz-subtitle">Cell-type-level Spearman rho (all donors pooled) across single-cell atlases</div>
                        <div id="val-summary-method-comparison" class="plot-container" style="height: 450px;"></div>
                    </div>
                </div>
            </div>
        `;

        // Load boxplot data for both signature types + method comparison
        const [cytosigData, secactData, methodData] = await Promise.all([
            this.cachedFetch(() => API.getSummaryBoxplot('cytosig'), 'summary-boxplot-cytosig'),
            this.cachedFetch(() => API.getSummaryBoxplot('secact'), 'summary-boxplot-secact'),
            this.cachedFetch(() => API.getMethodComparison(), 'method-comparison').catch(() => null),
        ]);

        // Populate target dropdown
        const targetSelect = document.getElementById('val-summary-target');
        if (targetSelect) {
            const targets = new Set([
                ...(cytosigData.targets || []),
                ...(secactData.targets || []),
            ]);
            [...targets].sort().forEach(t => {
                const opt = document.createElement('option');
                opt.value = t;
                opt.textContent = t;
                targetSelect.appendChild(opt);
            });

            targetSelect.addEventListener('change', () => {
                this._renderSummaryCombinedBox(cytosigData, secactData, targetSelect.value);
            });
        }

        const searchInput = document.getElementById('val-summary-search');
        if (searchInput && targetSelect) {
            searchInput.addEventListener('input', () => {
                this._filterSelect(targetSelect, searchInput.value);
            });
        }

        this._renderSummaryCombinedBox(cytosigData, secactData, '_all');
        if (methodData) {
            this._renderMethodComparison('val-summary-method-comparison', methodData);
        }
    },

    _renderSummaryCombinedBox(cytosigData, secactData, target) {
        const div = document.getElementById('val-summary-box');
        if (!div) return;

        const cytosigCats = cytosigData.categories || [];
        const secactCats = secactData.categories || [];
        const cytosigRhos = cytosigData.rhos || {};
        const secactRhos = secactData.rhos || {};

        if (!cytosigCats.length && !secactCats.length) {
            div.innerHTML = '<p style="color:#888;text-align:center;margin-top:40px;">No data available</p>';
            return;
        }

        // Collect rho values for a category key from rhos dict.
        // rhos structure: { targetName: { catKey: [rho_values] } }
        const collectVals = (rhos, catKey) => {
            let yVals = [];
            if (target === '_all') {
                for (const tgtRhos of Object.values(rhos)) {
                    const vals = tgtRhos[catKey];
                    if (vals !== null && vals !== undefined) {
                        if (Array.isArray(vals)) {
                            yVals.push(...vals);
                        } else {
                            yVals.push(vals);
                        }
                    }
                }
            } else {
                const tgtRhos = rhos[target];
                if (tgtRhos) {
                    const vals = tgtRhos[catKey];
                    if (vals !== null && vals !== undefined) {
                        yVals = Array.isArray(vals) ? vals : [vals];
                    }
                }
            }
            return yVals;
        };

        // Build unified category list from whichever dataset has more categories
        const refCats = cytosigCats.length >= secactCats.length ? cytosigCats : secactCats;
        const traces = [];

        // CytoSig traces (one box per category)
        refCats.forEach((cat, i) => {
            const catKey = typeof cat === 'string' ? cat : (cat.key || String(cat));
            const catLabel = typeof cat === 'string' ? cat.replace(/_/g, ' ') : (cat.label || cat.key || String(cat));
            const yVals = collectVals(cytosigRhos, catKey);
            traces.push({
                type: 'box',
                y: yVals,
                name: 'CytoSig',
                x: yVals.map(() => catLabel),
                legendgroup: 'CytoSig',
                marker: { color: '#1f77b4' },
                boxpoints: yVals.length < 50 ? 'all' : 'outliers',
                jitter: 0.3,
                offsetgroup: 'cytosig',
                showlegend: i === 0,
            });
        });

        // SecAct traces (one box per category, side-by-side)
        const secRefCats = secactCats.length > 0 ? secactCats : refCats;
        secRefCats.forEach((cat, i) => {
            const catKey = typeof cat === 'string' ? cat : (cat.key || String(cat));
            const catLabel = typeof cat === 'string' ? cat.replace(/_/g, ' ') : (cat.label || cat.key || String(cat));
            const yVals = collectVals(secactRhos, catKey);
            traces.push({
                type: 'box',
                y: yVals,
                name: 'SecAct',
                x: yVals.map(() => catLabel),
                legendgroup: 'SecAct',
                marker: { color: '#ff7f0e' },
                boxpoints: yVals.length < 50 ? 'all' : 'outliers',
                jitter: 0.3,
                offsetgroup: 'secact',
                showlegend: i === 0,
            });
        });

        Plotly.newPlot(div, traces, {
            boxmode: 'group',
            yaxis: { title: 'Spearman rho', zeroline: true },
            xaxis: { tickangle: -30 },
            margin: { l: 60, r: 20, t: 40, b: 120 },
            legend: { orientation: 'h', x: 0.3, y: 1.05 },
            title: { text: target === '_all' ? 'All Targets' : target, font: { size: 14 } },
        }, { responsive: true });
    },

    /**
     * Render method comparison: CytoSig vs LinCytoSig vs SecAct boxplots
     * across SC atlases at celltype level.
     */
    _renderMethodComparison(divId, methodData) {
        const div = document.getElementById(divId);
        if (!div) return;

        const categories = methodData.categories || [];
        if (!categories.length) {
            div.innerHTML = '<p class="no-data">No method comparison data available</p>';
            return;
        }

        const sigColors = {
            cytosig: '#1f77b4',
            lincytosig: '#2ca02c',
            secact: '#ff7f0e',
        };
        const sigLabels = {
            cytosig: 'CytoSig',
            lincytosig: 'LinCytoSig',
            secact: 'SecAct',
        };

        const traces = [];
        const sigs = ['cytosig', 'lincytosig', 'secact'];

        sigs.forEach((sig, si) => {
            const sigData = methodData[sig];
            if (!sigData || !sigData.rhos) return;

            categories.forEach((cat, ci) => {
                const catKey = cat.key;
                const catLabel = cat.label;

                // Collect all rho values for this sig × category
                const rhos = [];
                for (const [target, targetRhos] of Object.entries(sigData.rhos)) {
                    const val = targetRhos[catKey];
                    if (val != null && isFinite(val)) {
                        rhos.push(val);
                    }
                }

                traces.push({
                    type: 'box',
                    y: rhos,
                    name: sigLabels[sig],
                    x: rhos.map(() => catLabel),
                    legendgroup: sig,
                    marker: { color: sigColors[sig] },
                    boxpoints: rhos.length < 50 ? 'all' : 'outliers',
                    jitter: 0.3,
                    offsetgroup: sig,
                    showlegend: ci === 0,
                });
            });
        });

        if (!traces.length) {
            div.innerHTML = '<p class="no-data">No comparison data</p>';
            return;
        }

        Plotly.newPlot(div, traces, {
            boxmode: 'group',
            yaxis: { title: 'Spearman rho', zeroline: true },
            xaxis: { tickangle: -20 },
            margin: { l: 60, r: 20, t: 40, b: 130 },
            legend: { orientation: 'h', x: 0.15, y: 1.05 },
        }, { responsive: true });
    },

    // ==================== Tab 2: Bulk RNA-seq ====================

    async loadBulkRnaseqTab() {
        const content = document.getElementById('validation-content');
        content.innerHTML = `
            <div class="tab-panel">
                <h3>Bulk RNA-seq Validation</h3>
                <p>Validation against independent bulk RNA-seq cohorts. For each target, mean expression of signature genes is correlated with predicted activity across samples. GTEx provides normal tissue validation; TCGA provides cancer-type validation. Points are colored by tissue/cancer type. Spearman rho and Pearson r are computed on visible points.</p>

                <div class="filter-bar">
                    ${this._sigtypeSelectHTML('val-bulk-sigtype', this.BULK_SIG_OPTIONS, this.bulkRnaseq.sigtype)}
                    <label>Dataset:
                        <select id="val-bulk-dataset">
                            <option value="gtex"${this.bulkRnaseq.dataset === 'gtex' ? ' selected' : ''}>GTEx (normal tissues)</option>
                            <option value="tcga"${this.bulkRnaseq.dataset === 'tcga' ? ' selected' : ''}>TCGA (cancer types)</option>
                        </select>
                    </label>
                    <label>Target:
                        <select id="val-bulk-target"><option value="">Loading...</option></select>
                    </label>
                    <label>${this.bulkRnaseq.dataset === 'gtex' ? 'Tissue' : 'Cancer Type'}:
                        <select id="val-bulk-group"><option value="">All</option></select>
                    </label>
                    <label>
                        <input type="checkbox" id="val-bulk-hide-nonexpr"> Hide non-expressing
                    </label>
                    <label>
                        <input type="text" id="val-bulk-search" placeholder="Search target...">
                    </label>
                </div>

                <div class="panel-grid">
                    <div class="panel" style="grid-column: span 2;">
                        <div class="viz-title">Expression vs Activity</div>
                        <div class="viz-subtitle">Each point = one tissue/cancer sample</div>
                        <div id="val-bulk-scatter" class="plot-container" style="height: 450px;"></div>
                    </div>
                </div>
                <div class="panel-grid">
                    <div class="panel">
                        <div class="viz-title">Per-Group Correlation</div>
                        <div id="val-bulk-bar" class="plot-container" style="height: 350px;"></div>
                    </div>
                    <div class="panel">
                        <div class="viz-title">Activity by Group</div>
                        <div id="val-bulk-box" class="plot-container" style="height: 350px;"></div>
                    </div>
                </div>
            </div>
        `;

        const sigtypeSel = document.getElementById('val-bulk-sigtype');
        const datasetSel = document.getElementById('val-bulk-dataset');
        const targetSel = document.getElementById('val-bulk-target');
        const groupSel = document.getElementById('val-bulk-group');
        const hideNonExpr = document.getElementById('val-bulk-hide-nonexpr');
        const searchInput = document.getElementById('val-bulk-search');

        if (sigtypeSel) {
            sigtypeSel.addEventListener('change', () => {
                this.bulkRnaseq.sigtype = sigtypeSel.value;
                // Don't reset target — try to keep current selection
                this._loadBulkTargets();
            });
        }
        if (datasetSel) {
            datasetSel.addEventListener('change', () => {
                this.bulkRnaseq.dataset = datasetSel.value;
                this.bulkRnaseq.group = '';
                // Don't reset target — try to keep current selection
                this.loadBulkRnaseqTab();
            });
        }
        if (groupSel) {
            groupSel.addEventListener('change', () => {
                this.bulkRnaseq.group = groupSel.value;
                this._renderBulkScatter();
            });
        }
        if (hideNonExpr) {
            hideNonExpr.checked = this.bulkRnaseq.hideNonExpr;
            hideNonExpr.addEventListener('change', () => {
                this.bulkRnaseq.hideNonExpr = hideNonExpr.checked;
                this._renderBulkScatter();
            });
        }
        if (targetSel) {
            targetSel.addEventListener('change', () => {
                this.bulkRnaseq.target = targetSel.value;
                this._renderBulkScatter();
            });
        }
        if (searchInput) {
            searchInput.addEventListener('input', () => {
                this._filterSelect(targetSel, searchInput.value);
            });
        }

        await this._loadBulkTargets();
    },

    async _loadBulkTargets() {
        const targetSel = document.getElementById('val-bulk-target');
        if (!targetSel) return;
        targetSel.innerHTML = '<option value="">Loading...</option>';

        const ds = this.bulkRnaseq.dataset;
        const sig = this.bulkRnaseq.sigtype;
        const targets = await this.cachedFetch(
            () => API.getBulkRnaseqTargets(ds, sig),
            `bulk-targets-${ds}-${sig}`
        );

        this.bulkRnaseq.targets = targets;
        targets.sort((a, b) => Math.abs(b.rho || 0) - Math.abs(a.rho || 0));

        targetSel.innerHTML = targets.map(t =>
            `<option value="${t.target}">${t.target} (${t.gene || ''}, r=${(t.rho || 0).toFixed(3)})</option>`
        ).join('');

        // Preserve current target: exact match, then fuzzy match (cytokine name)
        const currentTarget = this.bulkRnaseq.target;
        let matched = currentTarget && targets.find(t => t.target === currentTarget);
        if (!matched && currentTarget && targets.length) {
            const base = currentTarget.includes('__') ? currentTarget.split('__').pop() : currentTarget;
            matched = targets.find(t => t.target === base)
                || targets.find(t => t.target.endsWith('__' + base))
                || targets.find(t => t.target.toLowerCase().includes(base.toLowerCase()));
        }
        if (matched) {
            this.bulkRnaseq.target = matched.target;
        } else if (targets.length) {
            this.bulkRnaseq.target = targets[0].target;
        }
        if (this.bulkRnaseq.target) {
            targetSel.value = this.bulkRnaseq.target;
        }

        // Reapply search filter if search text is present
        const searchInput = document.getElementById('val-bulk-search');
        if (searchInput && searchInput.value) {
            this._filterSelect(targetSel, searchInput.value);
        }

        await this._renderBulkScatter();
    },

    async _renderBulkScatter() {
        if (!this.bulkRnaseq.target) return;
        const ds = this.bulkRnaseq.dataset;
        const target = this.bulkRnaseq.target;
        const sig = this.bulkRnaseq.sigtype;
        const cacheKey = `bulk-scatter-${ds}-${target}-${sig}`;

        const data = await this.cachedFetch(
            () => API.getBulkRnaseqScatter(ds, target, sig),
            cacheKey
        );

        // Stale check: target may have changed during async fetch
        if (this.bulkRnaseq.target !== target) return;

        if (!data || !data.points) {
            document.getElementById('val-bulk-scatter').innerHTML = '<p class="no-data">No scatter data available</p>';
            return;
        }

        // Populate group selector (tissue/cancer type)
        const groupSel = document.getElementById('val-bulk-group');
        if (groupSel && data.groups) {
            const currentGroup = groupSel.value;
            groupSel.innerHTML = '<option value="">All</option>' +
                data.groups.map(g => `<option value="${g}">${g.replace(/_/g, ' ')}</option>`).join('');
            if (currentGroup) groupSel.value = currentGroup;
        }

        this._renderValidationScatter('val-bulk-scatter', data, target, {
            hideNonExpr: this.bulkRnaseq.hideNonExpr,
            celltypeFilter: this.bulkRnaseq.group,
            unitLabel: 'samples',
            filterLabel: ds === 'gtex' ? 'Tissue' : 'Cancer Type',
            atlasLabel: ds === 'gtex' ? 'GTEx' : 'TCGA',
        });

        this._renderGroupBar('val-bulk-bar', data);
        this._renderGroupBox('val-bulk-box', data);
    },

    // ==================== Tab 3: Donor Level ====================

    async loadDonorLevelTab() {
        const atlas = this.donorLevel.atlas;
        const config = this.DONOR_LEVEL_CONFIG[atlas];
        const levels = config ? Object.keys(config.levels) : [];
        if (!this.donorLevel.level && levels.length) {
            this.donorLevel.level = levels[0];
        }

        const content = document.getElementById('validation-content');
        content.innerHTML = `
            <div class="tab-panel">
                <h3>Donor-Level Validation</h3>
                <p>Cross-sample validation at the donor level. For each target, pseudobulk mean expression is correlated with predicted activity across donors. Points are colored by cell type. A positive Spearman rho confirms that donors with higher signature gene expression also have higher inferred activity.</p>

                <div class="filter-bar">
                    ${this._atlasSelectHTML('val-donor-atlas', this.SC_ATLAS_OPTIONS, atlas)}
                    ${this._sigtypeSelectHTML('val-donor-sigtype', this.ALL_SIG_OPTIONS, this.donorLevel.sigtype)}
                    <label>Level:
                        <select id="val-donor-level">
                            ${levels.map(l => `<option value="${l}" ${l === this.donorLevel.level ? 'selected' : ''}>${config ? config.levels[l] : l}</option>`).join('')}
                        </select>
                    </label>
                    <label>Target:
                        <select id="val-donor-target"><option value="">Loading...</option></select>
                    </label>
                    <label>${config && config.groupLabel ? (typeof config.groupLabel === 'string' ? config.groupLabel : config.groupLabel[this.donorLevel.level] || 'Group') : 'Group'}:
                        <select id="val-donor-group"><option value="">All</option></select>
                    </label>
                    <label>
                        <input type="checkbox" id="val-donor-hide-nonexpr"> Hide non-expressing
                    </label>
                    <label>
                        <input type="text" id="val-donor-search" placeholder="Search target...">
                    </label>
                </div>

                <div class="panel-grid">
                    <div class="panel" style="grid-column: span 2;">
                        <div class="viz-title">Expression vs Activity</div>
                        <div class="viz-subtitle" id="val-donor-subtitle">Each point = one donor pseudobulk sample</div>
                        <div id="val-donor-scatter" class="plot-container" style="height: 450px;"></div>
                    </div>
                </div>
                <div class="panel-grid">
                    <div class="panel">
                        <div class="viz-title">Per-Group Correlation</div>
                        <div id="val-donor-bar" class="plot-container" style="height: 350px;"></div>
                    </div>
                    <div class="panel">
                        <div class="viz-title">Activity by Group</div>
                        <div id="val-donor-box" class="plot-container" style="height: 350px;"></div>
                    </div>
                </div>
            </div>
        `;

        const atlasSel = document.getElementById('val-donor-atlas');
        const sigtypeSel = document.getElementById('val-donor-sigtype');
        const levelSel = document.getElementById('val-donor-level');
        const targetSel = document.getElementById('val-donor-target');
        const groupSel = document.getElementById('val-donor-group');
        const hideNonExpr = document.getElementById('val-donor-hide-nonexpr');
        const searchInput = document.getElementById('val-donor-search');

        if (atlasSel) {
            atlasSel.addEventListener('change', () => {
                this.donorLevel.atlas = atlasSel.value;
                this.donorLevel.level = null;
                this.donorLevel.target = null;
                this.loadDonorLevelTab();
            });
        }
        if (sigtypeSel) {
            sigtypeSel.addEventListener('change', () => {
                this.donorLevel.sigtype = sigtypeSel.value;
                // Don't reset target — preserve current selection across sigtype changes
                this._loadDonorTargets();
            });
        }
        if (levelSel) {
            levelSel.addEventListener('change', () => {
                this.donorLevel.level = levelSel.value;
                // Don't reset target — preserve current selection across level changes
                this._loadDonorTargets();
            });
        }
        if (targetSel) {
            targetSel.addEventListener('change', () => {
                this.donorLevel.target = targetSel.value;
                this._renderDonorScatter();
            });
        }
        if (groupSel) {
            groupSel.addEventListener('change', () => {
                this.donorLevel.group = groupSel.value;
                this._renderDonorScatter();
            });
        }
        if (hideNonExpr) {
            hideNonExpr.checked = this.donorLevel.hideNonExpr;
            hideNonExpr.addEventListener('change', () => {
                this.donorLevel.hideNonExpr = hideNonExpr.checked;
                this._renderDonorScatter();
            });
        }
        if (searchInput) {
            searchInput.addEventListener('input', () => {
                this._filterSelect(targetSel, searchInput.value);
            });
        }

        await this._loadDonorTargets();
    },

    async _loadDonorTargets() {
        const targetSel = document.getElementById('val-donor-target');
        if (!targetSel) return;
        targetSel.innerHTML = '<option value="">Loading...</option>';

        const atlas = this.donorLevel.atlas;
        const sig = this.donorLevel.sigtype;
        const level = this.donorLevel.level || 'donor_only';

        let targets;
        if (level === 'donor_only') {
            // Use donor API (pure donor-level, no celltype stratification)
            targets = await this.cachedFetch(
                () => API.getDonorTargets(atlas, sig),
                `donor-targets-only-${atlas}-${sig}`
            );
        } else {
            // Use celltype API (has level dimension + groups)
            targets = await this.cachedFetch(
                () => API.getCelltypeTargets(atlas, sig, level),
                `donor-targets-${atlas}-${sig}-${level}`
            );
        }

        this.donorLevel.targets = targets;
        targets.sort((a, b) => Math.abs(b.rho || 0) - Math.abs(a.rho || 0));

        targetSel.innerHTML = targets.map(t =>
            `<option value="${t.target}">${t.target} (${t.gene || ''}, r=${Number(t.rho || 0).toFixed(3)}, n=${t.n || ''})</option>`
        ).join('');

        // Preserve current target: exact match, then fuzzy match (cytokine name)
        const currentTarget = this.donorLevel.target;
        let matched = currentTarget && targets.find(t => t.target === currentTarget);
        if (!matched && currentTarget && targets.length) {
            // Extract base cytokine name (e.g., "IFNG" from "B_Cell__IFNG" or just "IFNG")
            const base = currentTarget.includes('__') ? currentTarget.split('__').pop() : currentTarget;
            matched = targets.find(t => t.target === base)
                || targets.find(t => t.target.endsWith('__' + base))
                || targets.find(t => t.target.toLowerCase().includes(base.toLowerCase()));
        }
        if (matched) {
            this.donorLevel.target = matched.target;
        } else if (targets.length) {
            this.donorLevel.target = targets[0].target;
        }
        if (this.donorLevel.target) targetSel.value = this.donorLevel.target;

        // Reapply search filter if search text is present
        const searchInput = document.getElementById('val-donor-search');
        if (searchInput && searchInput.value) {
            this._filterSelect(targetSel, searchInput.value);
        }

        await this._renderDonorScatter();
    },

    async _renderDonorScatter() {
        if (!this.donorLevel.target) return;
        const atlas = this.donorLevel.atlas;
        const target = this.donorLevel.target;
        const sig = this.donorLevel.sigtype;
        const level = this.donorLevel.level || 'donor_only';
        const cacheKey = `donor-scatter-${atlas}-${level}-${target}-${sig}`;

        let data;
        if (level === 'donor_only') {
            // Use donor API (pure donor-level)
            data = await this.cachedFetch(
                () => API.getDonorScatter(atlas, target, sig),
                cacheKey
            );
        } else {
            // Use celltype API (has level + groups) for proper group coloring
            data = await this.cachedFetch(
                () => API.getCelltypeScatter(atlas, target, sig, level),
                cacheKey
            );
        }

        if (this.donorLevel.target !== target) return;

        if (!data || !data.points) {
            document.getElementById('val-donor-scatter').innerHTML = '<p class="no-data">No scatter data</p>';
            return;
        }

        // Update subtitle
        const subtitle = document.getElementById('val-donor-subtitle');
        if (subtitle) {
            const nPts = data.points ? data.points.length : (data.n || 0);
            if (level === 'donor_only') {
                subtitle.textContent = `Each point = one donor (${nPts} donors, all cell types pooled)`;
            } else {
                subtitle.textContent = `Each point = one donor \u00D7 cell type (${nPts} observations)`;
            }
        }

        // Populate group selector
        const groupSel = document.getElementById('val-donor-group');
        if (groupSel && data.groups) {
            const currentGroup = groupSel.value;
            groupSel.innerHTML = '<option value="">All</option>' +
                data.groups.map(g => `<option value="${g}">${g.replace(/_/g, ' ')}</option>`).join('');
            if (currentGroup) groupSel.value = currentGroup;
        } else if (groupSel) {
            groupSel.innerHTML = '<option value="">All</option>';
        }

        const donorConfig = this.DONOR_LEVEL_CONFIG[atlas];
        const donorGroupLabel = donorConfig && donorConfig.groupLabel
            ? (typeof donorConfig.groupLabel === 'string' ? donorConfig.groupLabel : donorConfig.groupLabel[level] || 'Group')
            : 'Cell Type';

        this._renderValidationScatter('val-donor-scatter', data, target, {
            hideNonExpr: this.donorLevel.hideNonExpr,
            celltypeFilter: this.donorLevel.group,
            unitLabel: 'donors',
            filterLabel: donorGroupLabel,
            atlasLabel: this.donorLevel.atlas,
        });

        this._renderGroupBar('val-donor-bar', data);
        this._renderGroupBox('val-donor-box', data);
    },

    // ==================== Tab 4: Cell Type Level ====================

    async loadCelltypeLevelTab() {
        const atlas = this.celltypeLevel.atlas;
        const config = this.CELLTYPE_LEVEL_CONFIG[atlas];
        const levels = config ? Object.keys(config.levels) : [];
        if (!this.celltypeLevel.level && levels.length) {
            this.celltypeLevel.level = levels[0];
        }

        const content = document.getElementById('validation-content');
        content.innerHTML = `
            <div class="tab-panel">
                <h3>Cell Type Level Validation</h3>
                <p>Cell-type-level validation. Each point represents one cell type with all donors pooled together. Expression and activity are averaged across all cells of each type, testing whether predicted activity tracks gene expression across cell types independent of donor variation.</p>

                <div class="filter-bar">
                    ${this._atlasSelectHTML('val-ct-atlas', this.SC_ATLAS_OPTIONS, atlas)}
                    ${this._sigtypeSelectHTML('val-ct-sigtype', this.ALL_SIG_OPTIONS, this.celltypeLevel.sigtype)}
                    <label>Level:
                        <select id="val-ct-level">
                            ${levels.map(l => `<option value="${l}" ${l === this.celltypeLevel.level ? 'selected' : ''}>${config ? config.levels[l] : l}</option>`).join('')}
                        </select>
                    </label>
                    <label>Target:
                        <select id="val-ct-target"><option value="">Loading...</option></select>
                    </label>
                    <label>
                        <input type="checkbox" id="val-ct-hide-nonexpr"> Hide non-expressing
                    </label>
                    <label>
                        <input type="text" id="val-ct-search" placeholder="Search target...">
                    </label>
                </div>

                <div class="panel-grid">
                    <div class="panel" style="grid-column: span 2;">
                        <div class="viz-title">Expression vs Activity</div>
                        <div class="viz-subtitle">Each point = one cell type (all donors pooled)</div>
                        <div id="val-ct-scatter" class="plot-container" style="height: 450px;"></div>
                    </div>
                </div>
                <div class="panel-grid">
                    <div class="panel">
                        <div class="viz-title">Top Targets by |rho|</div>
                        <div class="viz-subtitle">Targets ranked by absolute Spearman correlation</div>
                        <div id="val-ct-ranking" class="plot-container" style="height: 350px;"></div>
                    </div>
                    <div class="panel">
                        <div class="viz-title">Activity by Cell Type</div>
                        <div class="viz-subtitle">Predicted activity per cell type for selected target</div>
                        <div id="val-ct-activity" class="plot-container" style="height: 350px;"></div>
                    </div>
                </div>
            </div>
        `;

        const ctAtlasSel = document.getElementById('val-ct-atlas');
        const ctSigtypeSel = document.getElementById('val-ct-sigtype');
        const levelSel = document.getElementById('val-ct-level');
        const targetSel = document.getElementById('val-ct-target');
        const hideNonExpr = document.getElementById('val-ct-hide-nonexpr');
        const searchInput = document.getElementById('val-ct-search');

        if (ctAtlasSel) {
            ctAtlasSel.addEventListener('change', () => {
                this.celltypeLevel.atlas = ctAtlasSel.value;
                this.celltypeLevel.level = null;
                // Don't reset target
                this.loadCelltypeLevelTab();
            });
        }
        if (ctSigtypeSel) {
            ctSigtypeSel.addEventListener('change', () => {
                this.celltypeLevel.sigtype = ctSigtypeSel.value;
                // Don't reset target — preserve current selection
                this._loadCelltypeTargets();
            });
        }
        if (levelSel) {
            levelSel.addEventListener('change', () => {
                this.celltypeLevel.level = levelSel.value;
                // Don't reset target — preserve current selection
                this._loadCelltypeTargets();
            });
        }
        if (targetSel) {
            targetSel.addEventListener('change', () => {
                this.celltypeLevel.target = targetSel.value;
                this._renderCelltypeScatter();
            });
        }
        if (hideNonExpr) {
            hideNonExpr.checked = this.celltypeLevel.hideNonExpr;
            hideNonExpr.addEventListener('change', () => {
                this.celltypeLevel.hideNonExpr = hideNonExpr.checked;
                this._renderCelltypeScatter();
            });
        }
        if (searchInput) {
            searchInput.addEventListener('input', () => {
                this._filterSelect(targetSel, searchInput.value);
            });
        }

        await this._loadCelltypeTargets();
    },

    async _loadCelltypeTargets() {
        const targetSel = document.getElementById('val-ct-target');
        if (!targetSel) return;
        targetSel.innerHTML = '<option value="">Loading...</option>';

        const atlas = this.celltypeLevel.atlas;
        const sig = this.celltypeLevel.sigtype;
        const level = this.celltypeLevel.level || 'l1';
        const targets = await this.cachedFetch(
            () => API.getCelltypeTargets(atlas, sig, level),
            `ct-targets-${atlas}-${sig}-${level}`
        );

        this.celltypeLevel.targets = targets;
        targets.sort((a, b) => Math.abs(b.rho || 0) - Math.abs(a.rho || 0));

        targetSel.innerHTML = targets.map(t =>
            `<option value="${t.target}">${t.target} (${t.gene || ''}, r=${(t.rho || 0).toFixed(3)})</option>`
        ).join('');

        // Preserve current target: exact match, then fuzzy match (cytokine name)
        const currentTarget = this.celltypeLevel.target;
        let matched = currentTarget && targets.find(t => t.target === currentTarget);
        if (!matched && currentTarget && targets.length) {
            const base = currentTarget.includes('__') ? currentTarget.split('__').pop() : currentTarget;
            matched = targets.find(t => t.target === base)
                || targets.find(t => t.target.endsWith('__' + base))
                || targets.find(t => t.target.toLowerCase().includes(base.toLowerCase()));
        }
        if (matched) {
            this.celltypeLevel.target = matched.target;
        } else if (targets.length) {
            this.celltypeLevel.target = targets[0].target;
        }
        if (this.celltypeLevel.target) targetSel.value = this.celltypeLevel.target;

        // Reapply search filter if search text is present
        const searchInput = document.getElementById('val-ct-search');
        if (searchInput && searchInput.value) {
            this._filterSelect(targetSel, searchInput.value);
        }

        // Render target ranking bar
        this._renderTargetRanking('val-ct-ranking', targets);

        await this._renderCelltypeScatter();
    },

    async _renderCelltypeScatter() {
        if (!this.celltypeLevel.target) return;
        const atlas = this.celltypeLevel.atlas;
        const target = this.celltypeLevel.target;
        const sig = this.celltypeLevel.sigtype;
        const level = this.celltypeLevel.level || 'l1';
        const cacheKey = `ct-scatter-${atlas}-${level}-${target}-${sig}`;

        const data = await this.cachedFetch(
            () => API.getCelltypeScatter(atlas, target, sig, level),
            cacheKey
        );

        if (this.celltypeLevel.target !== target) return;

        if (!data || !data.points) {
            document.getElementById('val-ct-scatter').innerHTML = '<p class="no-data">No scatter data</p>';
            return;
        }

        this._renderValidationScatter('val-ct-scatter', data, target, {
            hideNonExpr: this.celltypeLevel.hideNonExpr,
            celltypeFilter: '',
            unitLabel: 'cell types',
            filterLabel: 'Cell Type',
            atlasLabel: this.celltypeLevel.atlas,
        });

        // Render per-celltype activity bar
        this._renderCelltypeActivityBar('val-ct-activity', data);
    },

    /**
     * Render top targets ranked by |rho| as horizontal bar chart.
     */
    _renderTargetRanking(divId, targets) {
        const div = document.getElementById(divId);
        if (!div) return;

        if (!targets || !targets.length) {
            div.innerHTML = '<p class="no-data">No targets available</p>';
            return;
        }

        const top20 = targets.slice(0, 20); // Already sorted by |rho|
        const getColor = (r) => {
            const ar = Math.abs(r);
            if (ar > 0.5) return '#1a9850';
            if (ar > 0.3) return '#91cf60';
            if (ar > 0.1) return '#fee08b';
            return '#d73027';
        };

        Plotly.newPlot(div, [{
            type: 'bar',
            y: top20.map(t => t.target),
            x: top20.map(t => t.rho || 0),
            orientation: 'h',
            marker: { color: top20.map(t => getColor(t.rho || 0)) },
            text: top20.map(t => `r=${(t.rho || 0).toFixed(3)}`),
            textposition: 'outside',
            hovertemplate: '%{y}<br>rho = %{x:.3f}<extra></extra>',
        }], {
            xaxis: { title: 'Spearman rho' },
            yaxis: { automargin: true, autorange: 'reversed' },
            margin: { l: 120, r: 60, t: 30, b: 50 },
        }, { responsive: true });
    },

    /**
     * Render per-celltype activity as horizontal bar chart for celltype-only data.
     * Each point in the scatter IS a celltype, so we show activity per celltype.
     */
    _renderCelltypeActivityBar(divId, data) {
        const div = document.getElementById(divId);
        if (!div) return;

        const points = data.points || [];
        const groups = data.groups || data.celltypes || [];
        if (!groups.length || !points.length) {
            div.innerHTML = '<p class="no-data">No activity data</p>';
            return;
        }

        // Build celltype -> activity mapping
        const ctActivities = [];
        points.forEach(p => {
            const gIdx = p[2] !== undefined ? p[2] : 0;
            const gName = gIdx >= 0 && gIdx < groups.length ? groups[gIdx] : 'Unknown';
            ctActivities.push({ name: gName, activity: p[1] });
        });

        // Sort by activity descending
        ctActivities.sort((a, b) => b.activity - a.activity);
        const top20 = ctActivities.slice(0, 20);

        const getColor = (a) => {
            if (a > 1) return '#1a9850';
            if (a > 0) return '#91cf60';
            if (a > -1) return '#fee08b';
            return '#d73027';
        };

        Plotly.newPlot(div, [{
            type: 'bar',
            y: top20.map(c => c.name.length > 30 ? c.name.substring(0, 27) + '...' : c.name),
            x: top20.map(c => c.activity),
            orientation: 'h',
            marker: { color: top20.map(c => getColor(c.activity)) },
            text: top20.map(c => c.activity.toFixed(2)),
            textposition: 'outside',
            hovertemplate: '%{y}<br>Activity: %{x:.3f}<extra></extra>',
        }], {
            xaxis: { title: 'Activity (z-score)' },
            yaxis: { automargin: true, autorange: 'reversed' },
            margin: { l: 150, r: 60, t: 30, b: 50 },
        }, { responsive: true });
    },

    // ==================== Tab 5: Single-Cell ====================

    async loadSingleCellTab() {
        const content = document.getElementById('validation-content');
        content.innerHTML = `
            <div class="tab-panel">
                <h3>Single-Cell Validation (All Cells)</h3>
                <p>Single-cell level validation using all cells (no aggregation). The density heatmap shows the joint distribution of expression and activity across millions of cells; the scatter overlay shows a 50K stratified sample colored by cell type. Per-cell-type bar charts show how correlation varies across populations.</p>

                <div class="filter-bar">
                    ${this._atlasSelectHTML('val-sc-atlas', this.SC_FULL_ATLAS_OPTIONS, this.singleCell.atlas)}
                    ${this._sigtypeSelectHTML('val-sc-sigtype', this.ALL_SIG_OPTIONS, this.singleCell.sigtype)}
                    <label>Target:
                        <select id="val-sc-target"><option value="">Loading...</option></select>
                    </label>
                    <label>Cell Type:
                        <select id="val-sc-celltype"><option value="">All</option></select>
                    </label>
                    <label>
                        <input type="checkbox" id="val-sc-hide-nonexpr"> Hide non-expressing
                    </label>
                    <label>
                        <input type="text" id="val-sc-search" placeholder="Search target...">
                    </label>
                </div>

                <div class="panel-grid">
                    <div class="panel" style="grid-column: span 2;">
                        <div class="viz-title">Expression vs Activity</div>
                        <div class="viz-subtitle" id="val-sc-subtitle">Density from all cells + 50K scatter overlay</div>
                        <div id="val-sc-scatter" class="plot-container" style="height: 500px;"></div>
                    </div>
                </div>
                <div class="panel-grid">
                    <div class="panel">
                        <div class="viz-title">Per-Cell-Type Correlation</div>
                        <div class="viz-subtitle">Spearman rho from ALL cells</div>
                        <div id="val-sc-bar" class="plot-container" style="height: 350px;"></div>
                    </div>
                    <div class="panel">
                        <div class="viz-title">Activity: Expressing vs Non-Expressing</div>
                        <div class="viz-subtitle">Mean activity comparison from ALL cells</div>
                        <div id="val-sc-box" class="plot-container" style="height: 350px;"></div>
                    </div>
                </div>
            </div>
        `;

        const scAtlasSel = document.getElementById('val-sc-atlas');
        const scSigtypeSel = document.getElementById('val-sc-sigtype');
        const targetSel = document.getElementById('val-sc-target');
        const ctSel = document.getElementById('val-sc-celltype');
        const hideNonExpr = document.getElementById('val-sc-hide-nonexpr');
        const searchInput = document.getElementById('val-sc-search');

        if (scAtlasSel) {
            scAtlasSel.addEventListener('change', () => {
                this.singleCell.atlas = scAtlasSel.value;
                // Don't reset target — preserve current selection
                this._loadSingleCellTargets();
            });
        }
        if (scSigtypeSel) {
            scSigtypeSel.addEventListener('change', () => {
                this.singleCell.sigtype = scSigtypeSel.value;
                // Don't reset target — preserve current selection
                this._loadSingleCellTargets();
            });
        }
        if (targetSel) {
            targetSel.addEventListener('change', () => {
                this.singleCell.target = targetSel.value;
                this._renderSingleCellPlots();
            });
        }
        if (ctSel) {
            ctSel.addEventListener('change', () => {
                this.singleCell.celltype = ctSel.value;
                this._renderSingleCellPlots();
            });
        }
        if (hideNonExpr) {
            hideNonExpr.checked = this.singleCell.hideNonExpr;
            hideNonExpr.addEventListener('change', () => {
                this.singleCell.hideNonExpr = hideNonExpr.checked;
                this._renderSingleCellPlots();
            });
        }
        if (searchInput) {
            searchInput.addEventListener('input', () => {
                this._filterSelect(targetSel, searchInput.value);
            });
        }

        await this._loadSingleCellTargets();
    },

    async _loadSingleCellTargets() {
        const targetSel = document.getElementById('val-sc-target');
        if (!targetSel) return;
        targetSel.innerHTML = '<option value="">Loading...</option>';

        const atlas = this.singleCell.atlas;
        const sig = this.singleCell.sigtype;

        let targets;
        try {
            targets = await this.cachedFetch(
                () => API.getSingleCellFullSignatures(atlas, sig),
                `sc-targets-${atlas}-${sig}`
            );
        } catch (e) {
            // Fallback to legacy endpoint
            try {
                targets = await this.cachedFetch(
                    () => API.getValidationSignatures(atlas, sig === 'cytosig' ? 'CytoSig' : 'SecAct'),
                    `sc-targets-legacy-${atlas}-${sig}`
                );
                // Normalize to list of objects
                if (targets.length && typeof targets[0] === 'string') {
                    targets = targets.map(t => ({ target: t, gene: t, rho: null }));
                }
            } catch (e2) {
                targets = [];
            }
        }

        this.singleCell.targets = targets;
        targets.sort((a, b) => Math.abs(b.rho || 0) - Math.abs(a.rho || 0));

        targetSel.innerHTML = targets.map(t => {
            const rhoStr = t.rho != null ? `, r=${Number(t.rho).toFixed(3)}` : '';
            const nStr = t.n_total ? `, n=${(t.n_total/1e6).toFixed(1)}M` : '';
            return `<option value="${t.target}">${t.target} (${t.gene || ''}${rhoStr}${nStr})</option>`;
        }).join('');

        // Preserve current target: exact match, then fuzzy match (cytokine name)
        const currentTarget = this.singleCell.target;
        let matched = currentTarget && targets.find(t => t.target === currentTarget);
        if (!matched && currentTarget && targets.length) {
            const base = currentTarget.includes('__') ? currentTarget.split('__').pop() : currentTarget;
            matched = targets.find(t => t.target === base)
                || targets.find(t => t.target.endsWith('__' + base))
                || targets.find(t => t.target.toLowerCase().includes(base.toLowerCase()));
        }
        if (matched) {
            this.singleCell.target = matched.target;
        } else if (targets.length) {
            this.singleCell.target = targets[0].target;
        }
        if (this.singleCell.target) targetSel.value = this.singleCell.target;

        // Reapply search filter if search text is present
        const searchInput = document.getElementById('val-sc-search');
        if (searchInput && searchInput.value) {
            this._filterSelect(targetSel, searchInput.value);
        }

        await this._renderSingleCellPlots();
    },

    async _renderSingleCellPlots() {
        if (!this.singleCell.target) return;
        const atlas = this.singleCell.atlas;
        const target = this.singleCell.target;
        const sig = this.singleCell.sigtype;
        const cacheKey = `sc-scatter-${atlas}-${target}-${sig}`;

        let data;
        try {
            data = await this.cachedFetch(
                () => API.getSingleCellFullScatter(atlas, target, sig),
                cacheKey
            );
        } catch (e) {
            if (this.singleCell.target === target) {
                document.getElementById('val-sc-scatter').innerHTML =
                    `<p class="no-data">Single-cell data not available for this atlas/target.</p>`;
            }
            return;
        }

        if (this.singleCell.target !== target) return;

        if (!data) {
            document.getElementById('val-sc-scatter').innerHTML = '<p class="no-data">No data available</p>';
            return;
        }

        // Populate celltype filter
        const ctSel = document.getElementById('val-sc-celltype');
        if (ctSel && data.sampled && data.sampled.celltypes) {
            const currentCt = ctSel.value;
            ctSel.innerHTML = '<option value="">All</option>' +
                data.sampled.celltypes.map(ct => `<option value="${ct}">${ct.replace(/_/g, ' ')}</option>`).join('');
            if (currentCt) ctSel.value = currentCt;
        }

        // Update subtitle with cell count
        const subtitle = document.getElementById('val-sc-subtitle');
        if (subtitle && data.n_total) {
            subtitle.textContent = `Density from ${data.n_total.toLocaleString()} cells + 50K scatter overlay`;
        }

        // Render density + scatter
        this._renderDensityScatter('val-sc-scatter', data);

        // Render per-celltype bar chart
        this._renderCelltypeBar('val-sc-bar', data);

        // Render expressing vs non-expressing comparison
        this._renderExprVsNonExprBox('val-sc-box', data);
    },

    _renderDensityScatter(divId, data) {
        const div = document.getElementById(divId);
        if (!div) return;

        const traces = [];
        const hideNonExpr = this.singleCell.hideNonExpr;
        const ctFilter = this.singleCell.celltype;

        // Layer 1: Density heatmap from ALL cells
        if (data.density) {
            const d = data.density;
            const nBins = d.n_bins || 100;
            const counts = d.counts || [];

            // Reshape flat array to 2D
            const z = [];
            for (let i = 0; i < nBins; i++) {
                z.push(counts.slice(i * nBins, (i + 1) * nBins));
            }

            // Apply log transform for better visualization
            const zLog = z.map(row => row.map(v => v > 0 ? Math.log10(v + 1) : 0));

            traces.push({
                type: 'heatmap',
                z: zLog,
                x0: d.expr_range[0],
                dx: (d.expr_range[1] - d.expr_range[0]) / nBins,
                y0: d.act_range[0],
                dy: (d.act_range[1] - d.act_range[0]) / nBins,
                colorscale: [
                    [0, 'rgba(255,255,255,0)'],
                    [0.01, 'rgba(240,249,255,0.3)'],
                    [0.1, 'rgba(107,174,214,0.5)'],
                    [0.3, 'rgba(49,130,189,0.6)'],
                    [0.6, 'rgba(8,81,156,0.7)'],
                    [1, 'rgba(8,48,107,0.9)'],
                ],
                showscale: true,
                colorbar: { title: 'log10(n+1)', len: 0.5, y: 0.8 },
                hovertemplate: 'Expr: %{x:.2f}<br>Activity: %{y:.2f}<br>Count: %{z:.0f}<extra>density</extra>',
            });
        }

        // Layer 2: 50K scatter overlay
        if (data.sampled && data.sampled.points) {
            let points = data.sampled.points;
            const celltypes = data.sampled.celltypes || [];

            // Filter by celltype
            if (ctFilter) {
                const ctIdx = celltypes.indexOf(ctFilter);
                if (ctIdx >= 0) {
                    points = points.filter(p => p[2] === ctIdx);
                }
            }

            // Filter non-expressing
            if (hideNonExpr) {
                points = points.filter(p => p[4] === 1); // is_expressing flag
            }

            if (points.length > 0) {
                const colors = [
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                ];

                traces.push({
                    type: 'scattergl',
                    mode: 'markers',
                    x: points.map(p => p[0]),
                    y: points.map(p => p[1]),
                    marker: {
                        size: 2,
                        opacity: 0.3,
                        color: points.map(p => colors[p[2] % colors.length]),
                    },
                    text: points.map(p => celltypes[p[2]] || 'unknown'),
                    hovertemplate: '%{text}<br>Expr: %{x:.2f}<br>Activity: %{y:.2f}<extra></extra>',
                    showlegend: false,
                });
            }
        }

        // Stats annotation
        const rhoVal = data.rho != null ? Number(data.rho) : 0;
        let annoText = `Spearman rho = ${rhoVal.toFixed(3)}`;
        if (data.pval != null) {
            annoText += `<br>p = ${Number(data.pval).toExponential(2)}`;
        }
        annoText += `<br>n = ${(data.n_total || 0).toLocaleString()} cells`;
        if (hideNonExpr) annoText += '<br>(non-expressing hidden)';
        if (ctFilter) annoText += `<br>Filter: ${ctFilter}`;

        const layout = {
            xaxis: { title: `${data.gene || data.target} Expression` },
            yaxis: { title: `${data.target} Activity (z-score)` },
            margin: { l: 60, r: 30, t: 50, b: 60 },
            annotations: [{
                x: 0.02, y: 0.98, xref: 'paper', yref: 'paper',
                text: annoText,
                showarrow: false, font: { size: 11 },
                bgcolor: 'rgba(255,255,255,0.9)', borderpad: 6,
            }],
            title: { text: `${data.target} — ${this.singleCell.atlas}`, font: { size: 14 } },
        };

        Plotly.newPlot(div, traces, layout, { responsive: true });
    },

    _renderCelltypeBar(divId, data) {
        const div = document.getElementById(divId);
        if (!div) return;

        const stats = data.celltype_stats || [];
        if (!stats.length) {
            div.innerHTML = '<p class="no-data">No per-celltype data</p>';
            return;
        }

        const sorted = [...stats].sort((a, b) => (b.rho || 0) - (a.rho || 0));
        const top20 = sorted.slice(0, 20);

        const getColor = (r) => {
            if (r > 0.3) return '#1a9850';
            if (r > 0.1) return '#91cf60';
            if (r > 0) return '#fee08b';
            return '#d73027';
        };

        Plotly.newPlot(div, [{
            type: 'bar',
            y: top20.map(s => s.celltype.replace(/_/g, ' ')),
            x: top20.map(s => s.rho || 0),
            orientation: 'h',
            marker: { color: top20.map(s => getColor(s.rho || 0)) },
            text: top20.map(s => `r=${(s.rho || 0).toFixed(3)}, n=${(s.n || 0).toLocaleString()}`),
            textposition: 'outside',
            hovertemplate: '%{y}<br>rho = %{x:.3f}<extra></extra>',
        }], {
            xaxis: { title: 'Spearman rho' },
            yaxis: { automargin: true, autorange: 'reversed' },
            margin: { l: 150, r: 80, t: 30, b: 50 },
        }, { responsive: true });
    },

    _renderExprVsNonExprBox(divId, data) {
        const div = document.getElementById(divId);
        if (!div) return;

        // Support both old field names and new API names
        const meanExpr = data.mean_activity_expressing ?? data.mean_act_expressing ?? null;
        const meanNonExpr = data.mean_activity_non_expressing ?? data.mean_act_non_expressing ?? null;

        if (meanExpr == null && meanNonExpr == null) {
            div.innerHTML = '<p class="no-data">No expressing/non-expressing data</p>';
            return;
        }

        const exprVal = meanExpr != null ? Number(meanExpr) : 0;
        const nonExprVal = meanNonExpr != null ? Number(meanNonExpr) : 0;

        // Compute activity difference (not ratio — z-scores can be negative)
        const actDiff = data.activity_diff != null ? Number(data.activity_diff) : (exprVal - nonExprVal);

        Plotly.newPlot(div, [{
            type: 'bar',
            x: ['Expressing', 'Non-Expressing'],
            y: [exprVal, nonExprVal],
            marker: { color: ['#2ca02c', '#d62728'] },
            text: [exprVal.toFixed(3), nonExprVal.toFixed(3)],
            textposition: 'outside',
        }], {
            yaxis: { title: 'Mean Activity (z-score)' },
            margin: { l: 60, r: 30, t: 60, b: 50 },
            annotations: [{
                x: 0.5, y: 1.1, xref: 'paper', yref: 'paper',
                text: `\u0394 Activity = ${actDiff.toFixed(3)}` +
                      `<br>n_expr = ${(data.n_expressing || 0).toLocaleString()}, n_total = ${(data.n_total || 0).toLocaleString()}`,
                showarrow: false, font: { size: 11 },
            }],
        }, { responsive: true });
    },

    // ==================== Shared Rendering Utilities ====================

    /**
     * Render scatter plot for donor/celltype/bulk tabs.
     * data.points: [[expr, act, group_idx], ...]
     * data.groups: ["group1", "group2", ...]
     */
    _renderValidationScatter(divId, data, target, opts = {}) {
        const div = document.getElementById(divId);
        if (!div) return;

        let points = data.points || [];
        const groups = data.groups || data.celltypes || [];
        const hideNonExpr = opts.hideNonExpr || false;
        const celltypeFilter = opts.celltypeFilter || '';
        const unitLabel = opts.unitLabel || 'samples';
        const filterLabel = opts.filterLabel || 'Group';
        const atlasLabel = opts.atlasLabel || '';

        // Filter by group
        if (celltypeFilter && groups.length) {
            const gIdx = groups.indexOf(celltypeFilter);
            if (gIdx >= 0) {
                points = points.filter(p => p[2] === gIdx);
            }
        }

        // Hide non-expressing: detect mode spike
        if (hideNonExpr && points.length > 0) {
            const xRounded = points.map(p => Math.round(p[0] * 100) / 100);
            const freq = {};
            xRounded.forEach(v => { freq[v] = (freq[v] || 0) + 1; });
            const entries = Object.entries(freq).sort((a, b) => b[1] - a[1]);
            if (entries.length > 0) {
                const modeVal = parseFloat(entries[0][0]);
                const modeCount = entries[0][1];
                if (modeCount > points.length * 0.1 && modeVal < 0) {
                    points = points.filter(p => Math.round(p[0] * 100) / 100 !== modeVal);
                }
            }
        }

        if (points.length === 0) {
            div.innerHTML = '<p class="no-data">No data after filtering</p>';
            return;
        }

        // Color by group
        const colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        ];

        const xArr = points.map(p => p[0]);
        const yArr = points.map(p => p[1]);
        const n = xArr.length;

        // Use pre-computed rho from metadata (exact); fall back to re-computation for filtered views
        const isFiltered = hideNonExpr || celltypeFilter;
        const spearmanR = isFiltered ? this._spearmanRho(xArr, yArr) : (data.rho != null ? Number(data.rho) : this._spearmanRho(xArr, yArr));

        const traces = [];
        const useWebGL = points.length > 1000;
        const scatterType = useWebGL ? 'scattergl' : 'scatter';

        if (groups.length > 0) {
            // One trace per group for legend
            const groupIndices = {};
            points.forEach((p, i) => {
                const gIdx = p[2] !== undefined ? p[2] : -1;
                if (!groupIndices[gIdx]) groupIndices[gIdx] = [];
                groupIndices[gIdx].push(i);
            });

            for (const [gIdxStr, indices] of Object.entries(groupIndices)) {
                const gIdx = parseInt(gIdxStr);
                const gName = gIdx >= 0 && gIdx < groups.length ? groups[gIdx] : 'Other';
                traces.push({
                    type: scatterType,
                    mode: 'markers',
                    name: gName.replace(/_/g, ' '),
                    x: indices.map(i => points[i][0]),
                    y: indices.map(i => points[i][1]),
                    marker: {
                        size: 5,
                        color: colors[gIdx % colors.length],
                        opacity: 0.5,
                    },
                    hovertemplate: `${gName}<br>Expr: %{x:.2f}<br>Activity: %{y:.2f}<extra></extra>`,
                });
            }
        } else {
            traces.push({
                type: scatterType,
                mode: 'markers',
                x: xArr,
                y: yArr,
                marker: { size: 5, color: '#1a5f7a', opacity: 0.5 },
                hovertemplate: 'Expr: %{x:.2f}<br>Activity: %{y:.2f}<extra></extra>',
            });
        }

        // Trendline
        const trendline = this.calculateTrendline(xArr, yArr);
        if (trendline.x) traces.push(trendline);

        // Annotation — use pre-computed p-value when unfiltered
        const pval = isFiltered ? null : (data.pval != null ? Number(data.pval) : null);
        let annoText = `Spearman rho = ${spearmanR.toFixed(3)}`;
        if (pval != null) {
            annoText += `<br>p = ${pval.toExponential(2)}`;
        }
        if (data.rho_ci) {
            annoText += `<br>95% CI [${data.rho_ci[0]}, ${data.rho_ci[1]}]`;
        }
        annoText += `<br>n = ${n} ${unitLabel}`;
        if (hideNonExpr) annoText += '<br>(non-expressing hidden)';
        if (celltypeFilter) annoText += `<br>${filterLabel}: ${celltypeFilter}`;

        const layout = {
            xaxis: { title: `${data.gene || target} Expression (z-score)` },
            yaxis: { title: `${target} Activity (z-score)` },
            margin: { l: 60, r: 30, t: 50, b: 60 },
            legend: { orientation: 'v', x: 1.02, y: 1, font: { size: 9 } },
            annotations: [{
                x: 0.02, y: 0.98, xref: 'paper', yref: 'paper',
                text: annoText,
                showarrow: false, font: { size: 11 },
                bgcolor: 'rgba(255,255,255,0.9)', borderpad: 6,
            }],
            title: { text: `${target}${atlasLabel ? ' — ' + atlasLabel : ''}`, font: { size: 14 } },
        };

        Plotly.newPlot(div, traces, layout, { responsive: true });
    },

    /**
     * Render per-group Pearson r bar chart.
     */
    _renderGroupBar(divId, data) {
        const div = document.getElementById(divId);
        if (!div) return;

        const points = data.points || [];
        const groups = data.groups || data.celltypes || [];
        if (!groups.length) {
            div.innerHTML = '<p class="no-data">No group data</p>';
            return;
        }

        // Compute per-group Pearson r
        const groupData = {};
        points.forEach(p => {
            const gIdx = p[2] !== undefined ? p[2] : 0;
            if (!groupData[gIdx]) groupData[gIdx] = { x: [], y: [] };
            groupData[gIdx].x.push(p[0]);
            groupData[gIdx].y.push(p[1]);
        });

        const bars = [];
        for (const [gIdxStr, gd] of Object.entries(groupData)) {
            const gIdx = parseInt(gIdxStr);
            const gName = gIdx >= 0 && gIdx < groups.length ? groups[gIdx] : 'Other';
            if (gd.x.length < 5) continue;

            const n = gd.x.length;
            const sX = gd.x.reduce((a, b) => a + b, 0);
            const sY = gd.y.reduce((a, b) => a + b, 0);
            const sXY = gd.x.reduce((a, x, i) => a + x * gd.y[i], 0);
            const sX2 = gd.x.reduce((a, x) => a + x * x, 0);
            const sY2 = gd.y.reduce((a, y) => a + y * y, 0);
            const d = Math.sqrt((n * sX2 - sX * sX) * (n * sY2 - sY * sY));
            const r = d > 0 ? (n * sXY - sX * sY) / d : 0;

            bars.push({ name: gName, r, n });
        }

        bars.sort((a, b) => b.r - a.r);
        const top15 = bars.slice(0, 15);

        const getColor = (r) => {
            if (r > 0.5) return '#1a9850';
            if (r > 0.2) return '#91cf60';
            if (r > 0) return '#fee08b';
            return '#d73027';
        };

        Plotly.newPlot(div, [{
            type: 'bar',
            y: top15.map(b => b.name.replace(/_/g, ' ')),
            x: top15.map(b => b.r),
            orientation: 'h',
            marker: { color: top15.map(b => getColor(b.r)) },
            text: top15.map(b => `r=${b.r.toFixed(2)}, n=${b.n}`),
            textposition: 'outside',
            hovertemplate: '%{y}<br>r = %{x:.3f}<extra></extra>',
        }], {
            xaxis: { title: 'Pearson r' },
            yaxis: { automargin: true, autorange: 'reversed' },
            margin: { l: 150, r: 80, t: 30, b: 50 },
        }, { responsive: true });
    },

    /**
     * Render activity boxplot per group.
     */
    _renderGroupBox(divId, data) {
        const div = document.getElementById(divId);
        if (!div) return;

        const points = data.points || [];
        const groups = data.groups || data.celltypes || [];
        if (!groups.length) {
            div.innerHTML = '<p class="no-data">No group data</p>';
            return;
        }

        const colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        ];

        // Build per-group activity arrays
        const groupActs = {};
        points.forEach(p => {
            const gIdx = p[2] !== undefined ? p[2] : 0;
            if (!groupActs[gIdx]) groupActs[gIdx] = [];
            groupActs[gIdx].push(p[1]);
        });

        const traces = [];
        for (const [gIdxStr, acts] of Object.entries(groupActs)) {
            const gIdx = parseInt(gIdxStr);
            if (acts.length < 3) continue;
            const gName = gIdx >= 0 && gIdx < groups.length ? groups[gIdx] : 'Other';
            traces.push({
                type: 'box',
                y: acts,
                name: gName.replace(/_/g, ' '),
                marker: { color: colors[gIdx % colors.length] },
                boxpoints: false,
            });
        }

        if (traces.length === 0) {
            div.innerHTML = '<p class="no-data">No activity data</p>';
            return;
        }

        // Limit to 15 groups for readability
        traces.sort((a, b) => {
            const medA = a.y.sort((x, y) => x - y)[Math.floor(a.y.length / 2)];
            const medB = b.y.sort((x, y) => x - y)[Math.floor(b.y.length / 2)];
            return medB - medA;
        });
        const topTraces = traces.slice(0, 15);

        Plotly.newPlot(div, topTraces, {
            yaxis: { title: 'Activity (z-score)' },
            margin: { l: 60, r: 20, t: 30, b: 100 },
            showlegend: false,
            xaxis: { tickangle: -45 },
        }, { responsive: true });
    },

    // ==================== Utility Functions ====================

    calculateTrendline(x, y) {
        if (!x.length || !y.length || x.length < 2) return {};
        const n = x.length;
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
        const sumX2 = x.reduce((acc, xi) => acc + xi * xi, 0);
        const denom = n * sumX2 - sumX * sumX;
        if (Math.abs(denom) < 1e-10) return {};
        const slope = (n * sumXY - sumX * sumY) / denom;
        const intercept = (sumY - slope * sumX) / n;
        const minX = Math.min(...x);
        const maxX = Math.max(...x);
        return {
            x: [minX, maxX],
            y: [slope * minX + intercept, slope * maxX + intercept],
            mode: 'lines',
            type: 'scatter',
            line: { color: 'red', dash: 'dash', width: 1 },
            showlegend: false,
            hoverinfo: 'skip',
        };
    },

    /**
     * Compute Spearman rank correlation (Pearson on ranks).
     */
    _spearmanRho(x, y) {
        const n = x.length;
        if (n < 3) return 0;
        const rank = (arr) => {
            const sorted = arr.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
            const ranks = new Array(n);
            let i = 0;
            while (i < n) {
                let j = i;
                while (j < n - 1 && sorted[j + 1][0] === sorted[j][0]) j++;
                const avgRank = (i + j) / 2 + 1;
                for (let k = i; k <= j; k++) ranks[sorted[k][1]] = avgRank;
                i = j + 1;
            }
            return ranks;
        };
        const rx = rank(x);
        const ry = rank(y);
        const sumRx = rx.reduce((a, b) => a + b, 0);
        const sumRy = ry.reduce((a, b) => a + b, 0);
        const sumRxRy = rx.reduce((a, v, i) => a + v * ry[i], 0);
        const sumRx2 = rx.reduce((a, v) => a + v * v, 0);
        const sumRy2 = ry.reduce((a, v) => a + v * v, 0);
        const d = Math.sqrt((n * sumRx2 - sumRx * sumRx) * (n * sumRy2 - sumRy * sumRy));
        return d > 0 ? (n * sumRxRy - sumRx * sumRy) / d : 0;
    },

    /**
     * Filter select options by search query.
     */
    _filterSelect(selectEl, query) {
        if (!selectEl) return;
        const lowerQ = query.toLowerCase();
        const options = selectEl.options;
        let firstVisible = null;
        for (let i = 0; i < options.length; i++) {
            const match = options[i].textContent.toLowerCase().includes(lowerQ) || !lowerQ;
            options[i].style.display = match ? '' : 'none';
            if (match && firstVisible === null) firstVisible = options[i];
        }
        // Auto-select first visible option and trigger change
        if (firstVisible && selectEl.value !== firstVisible.value) {
            selectEl.value = firstVisible.value;
            selectEl.dispatchEvent(new Event('change'));
        }
    },
};

// Make available globally
window.ValidatePage = ValidatePage;
