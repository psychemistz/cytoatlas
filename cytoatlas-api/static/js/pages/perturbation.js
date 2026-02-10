/**
 * Perturbation Page
 * Displays cytokine perturbation (parse_10M) and drug perturbation (Tahoe-100M) data.
 *
 * Tabs:
 * 1. Cytokine Response - Heatmap of 90 cytokines x 18 cell types (parse_10M)
 * 2. Ground Truth - Scatter of predicted vs actual CytoSig activity
 * 3. Drug Sensitivity - Heatmap of 95 drugs x 50 cell lines (Tahoe)
 * 4. Dose-Response - Line charts for Plate 13 drugs
 * 5. Pathway Activation - Drug -> cytokine pathway mapping
 */

window.PerturbationPage = (function () {
    'use strict';

    const API_BASE = '/api/v1/perturbation';
    let tabPanel = null;

    function init(params, query) {
        const app = document.getElementById('app');
        if (!app) return;

        app.innerHTML = `
            <div class="page-header">
                <h1>Perturbation Analysis</h1>
                <p class="page-subtitle">
                    Cytokine perturbation ground truth (parse_10M, 9.7M cells) and
                    drug sensitivity profiling (Tahoe-100M, 100.6M cells)
                </p>
            </div>
            <div id="perturbation-tabs" class="tab-container"></div>
            <div id="perturbation-content" class="tab-content-area"></div>
        `;

        initTabs();
    }

    function initTabs() {
        tabPanel = new TabPanel('perturbation-tabs', { orientation: 'horizontal' });

        tabPanel.addTab('cytokine-response', 'Cytokine Response', loadCytokineResponse);
        tabPanel.addTab('ground-truth', 'Ground Truth', loadGroundTruth);
        tabPanel.addTab('drug-sensitivity', 'Drug Sensitivity', loadDrugSensitivity);
        tabPanel.addTab('dose-response', 'Dose-Response', loadDoseResponse);
        tabPanel.addTab('pathway-activation', 'Pathway Activation', loadPathwayActivation);

        tabPanel.init();
    }

    // =========================================================================
    // Tab 1: Cytokine Response (parse_10M)
    // =========================================================================
    async function loadCytokineResponse() {
        const container = document.getElementById('perturbation-content');
        container.innerHTML = '<div class="loading-skeleton"></div>';

        try {
            const data = await API.get(`${API_BASE}/parse10m/heatmap`);

            container.innerHTML = `
                <div class="panel">
                    <div class="panel-header">
                        <h3>Cytokine Treatment Response Heatmap</h3>
                        <p>Activity z-scores for 90 cytokine treatments across 18 PBMC cell types (parse_10M)</p>
                        <div class="panel-controls">
                            <select id="cytokine-sig-type" class="select-control">
                                <option value="CytoSig">CytoSig (44 cytokines)</option>
                                <option value="SecAct">SecAct (1,249 proteins)</option>
                            </select>
                        </div>
                    </div>
                    <div id="cytokine-heatmap" class="chart-container" style="height: 600px;"></div>
                </div>
            `;

            renderCytokineHeatmap(data);

            document.getElementById('cytokine-sig-type')?.addEventListener('change', async (e) => {
                const newData = await API.get(`${API_BASE}/parse10m/heatmap?signature_type=${e.target.value}`);
                renderCytokineHeatmap(newData);
            });
        } catch (error) {
            container.innerHTML = `<div class="error-message">Failed to load cytokine response data: ${error.message}</div>`;
        }
    }

    function renderCytokineHeatmap(data) {
        if (!data || !Array.isArray(data) || data.length === 0) {
            document.getElementById('cytokine-heatmap').innerHTML =
                '<p class="no-data">No cytokine response data available yet. Run scripts/18_parse10m_activity.py first.</p>';
            return;
        }

        const cytokines = [...new Set(data.map(d => d.cytokine))];
        const cellTypes = [...new Set(data.map(d => d.cell_type))];

        const zValues = cytokines.map(cyt =>
            cellTypes.map(ct => {
                const record = data.find(d => d.cytokine === cyt && d.cell_type === ct);
                return record ? record.activity_diff : 0;
            })
        );

        Plotly.newPlot('cytokine-heatmap', [{
            type: 'heatmap',
            z: zValues,
            x: cellTypes,
            y: cytokines,
            colorscale: 'RdBu',
            reversescale: true,
            colorbar: { title: { text: '\u0394 Activity' } },
        }], {
            margin: { l: 120, r: 40, t: 20, b: 100 },
            xaxis: { title: 'Cell Type', tickangle: -45 },
            yaxis: { title: 'Cytokine Treatment' },
        }, { responsive: true });
    }

    // =========================================================================
    // Tab 2: Ground Truth Validation
    // =========================================================================
    async function loadGroundTruth() {
        const container = document.getElementById('perturbation-content');
        container.innerHTML = '<div class="loading-skeleton"></div>';

        try {
            const data = await API.get(`${API_BASE}/parse10m/ground-truth`);

            container.innerHTML = `
                <div class="panel">
                    <div class="panel-header">
                        <h3>CytoSig Ground Truth Validation</h3>
                        <p>Predicted activity vs actual cytokine treatment response.
                           Points above the diagonal indicate CytoSig correctly predicts the treated cytokine.</p>
                    </div>
                    <div id="ground-truth-scatter" class="chart-container" style="height: 500px;"></div>
                </div>
            `;

            renderGroundTruthScatter(data);
        } catch (error) {
            container.innerHTML = `<div class="error-message">Failed to load ground truth data: ${error.message}</div>`;
        }
    }

    function renderGroundTruthScatter(data) {
        if (!data || !Array.isArray(data) || data.length === 0) {
            document.getElementById('ground-truth-scatter').innerHTML =
                '<p class="no-data">No ground truth data available yet. Run scripts/21_parse10m_ground_truth.py first.</p>';
            return;
        }

        const selfSig = data.filter(d => d.is_self_signature);
        const otherSig = data.filter(d => !d.is_self_signature);

        Plotly.newPlot('ground-truth-scatter', [
            {
                type: 'scatter',
                mode: 'markers',
                x: selfSig.map(d => d.predicted_activity),
                y: selfSig.map(d => d.actual_response),
                text: selfSig.map(d => `${d.cytokine} (${d.cell_type})`),
                name: 'Self-signature (expected match)',
                marker: { color: '#e74c3c', size: 8, opacity: 0.7 },
            },
            {
                type: 'scatter',
                mode: 'markers',
                x: otherSig.map(d => d.predicted_activity),
                y: otherSig.map(d => d.actual_response),
                text: otherSig.map(d => `${d.cytokine} (${d.cell_type})`),
                name: 'Non-self signature',
                marker: { color: '#95a5a6', size: 4, opacity: 0.3 },
            },
        ], {
            xaxis: { title: 'Predicted CytoSig Activity' },
            yaxis: { title: 'Actual Treatment Response' },
            margin: { l: 60, r: 20, t: 20, b: 60 },
            showlegend: true,
            legend: { x: 0.02, y: 0.98 },
        }, { responsive: true });
    }

    // =========================================================================
    // Tab 3: Drug Sensitivity (Tahoe)
    // =========================================================================
    async function loadDrugSensitivity() {
        const container = document.getElementById('perturbation-content');
        container.innerHTML = '<div class="loading-skeleton"></div>';

        try {
            const data = await API.get(`${API_BASE}/tahoe/sensitivity-matrix`);

            container.innerHTML = `
                <div class="panel">
                    <div class="panel-header">
                        <h3>Drug Sensitivity Matrix</h3>
                        <p>Activity changes across 95 drugs and 50 cancer cell lines (Tahoe-100M)</p>
                        <div class="panel-controls">
                            <select id="drug-sig-type" class="select-control">
                                <option value="CytoSig">CytoSig</option>
                                <option value="SecAct">SecAct</option>
                            </select>
                        </div>
                    </div>
                    <div id="drug-sensitivity-heatmap" class="chart-container" style="height: 600px;"></div>
                </div>
            `;

            renderDrugSensitivityHeatmap(data);
        } catch (error) {
            container.innerHTML = `<div class="error-message">Failed to load drug sensitivity data: ${error.message}</div>`;
        }
    }

    function renderDrugSensitivityHeatmap(data) {
        if (!data || !Array.isArray(data) || data.length === 0) {
            document.getElementById('drug-sensitivity-heatmap').innerHTML =
                '<p class="no-data">No drug sensitivity data available yet. Run scripts/19_tahoe_activity.py first.</p>';
            return;
        }

        const drugs = [...new Set(data.map(d => d.drug))].slice(0, 50);
        const cellLines = [...new Set(data.map(d => d.cell_line))].slice(0, 30);

        const zValues = drugs.map(drug =>
            cellLines.map(cl => {
                const record = data.find(d => d.drug === drug && d.cell_line === cl);
                return record ? record.activity_diff : 0;
            })
        );

        Plotly.newPlot('drug-sensitivity-heatmap', [{
            type: 'heatmap',
            z: zValues,
            x: cellLines,
            y: drugs,
            colorscale: 'RdBu',
            reversescale: true,
            colorbar: { title: { text: '\u0394 Activity' } },
        }], {
            margin: { l: 120, r: 40, t: 20, b: 100 },
            xaxis: { title: 'Cell Line', tickangle: -45 },
            yaxis: { title: 'Drug' },
        }, { responsive: true });
    }

    // =========================================================================
    // Tab 4: Dose-Response
    // =========================================================================
    async function loadDoseResponse() {
        const container = document.getElementById('perturbation-content');
        container.innerHTML = '<div class="loading-skeleton"></div>';

        try {
            const data = await API.get(`${API_BASE}/tahoe/dose-response`);

            container.innerHTML = `
                <div class="panel">
                    <div class="panel-header">
                        <h3>Dose-Response Curves (Plate 13)</h3>
                        <p>Activity changes across 3 dose levels for 25 drugs in 50 cell lines</p>
                        <div class="panel-controls">
                            <select id="dose-drug-select" class="select-control">
                                <option value="">Select a drug...</option>
                            </select>
                        </div>
                    </div>
                    <div id="dose-response-chart" class="chart-container" style="height: 500px;"></div>
                </div>
            `;

            if (data && Array.isArray(data) && data.length > 0) {
                const drugs = [...new Set(data.map(d => d.drug))].sort();
                const select = document.getElementById('dose-drug-select');
                drugs.forEach(drug => {
                    const opt = document.createElement('option');
                    opt.value = drug;
                    opt.textContent = drug;
                    select.appendChild(opt);
                });

                select.addEventListener('change', () => renderDoseResponse(data, select.value));

                if (drugs.length > 0) {
                    select.value = drugs[0];
                    renderDoseResponse(data, drugs[0]);
                }
            } else {
                document.getElementById('dose-response-chart').innerHTML =
                    '<p class="no-data">No dose-response data available yet. Run scripts/19_tahoe_activity.py first.</p>';
            }
        } catch (error) {
            container.innerHTML = `<div class="error-message">Failed to load dose-response data: ${error.message}</div>`;
        }
    }

    function renderDoseResponse(data, drug) {
        const drugData = data.filter(d => d.drug === drug);
        const cellLines = [...new Set(drugData.map(d => d.cell_line))].slice(0, 10);

        const traces = cellLines.map(cl => {
            const clData = drugData.filter(d => d.cell_line === cl).sort((a, b) => a.dose - b.dose);
            return {
                type: 'scatter',
                mode: 'lines+markers',
                x: clData.map(d => d.dose),
                y: clData.map(d => d.activity_diff),
                name: cl,
            };
        });

        Plotly.newPlot('dose-response-chart', traces, {
            xaxis: { title: 'Dose', type: 'log' },
            yaxis: { title: '\u0394 Activity' },
            margin: { l: 60, r: 20, t: 20, b: 60 },
            showlegend: true,
        }, { responsive: true });
    }

    // =========================================================================
    // Tab 5: Pathway Activation
    // =========================================================================
    async function loadPathwayActivation() {
        const container = document.getElementById('perturbation-content');
        container.innerHTML = '<div class="loading-skeleton"></div>';

        try {
            const data = await API.get(`${API_BASE}/tahoe/pathway-activation`);

            container.innerHTML = `
                <div class="panel">
                    <div class="panel-header">
                        <h3>Drug-Induced Pathway Activation</h3>
                        <p>Which cytokine/secreted protein pathways each drug activates or suppresses</p>
                    </div>
                    <div id="pathway-heatmap" class="chart-container" style="height: 500px;"></div>
                </div>
            `;

            renderPathwayHeatmap(data);
        } catch (error) {
            container.innerHTML = `<div class="error-message">Failed to load pathway data: ${error.message}</div>`;
        }
    }

    function renderPathwayHeatmap(data) {
        if (!data || !Array.isArray(data) || data.length === 0) {
            document.getElementById('pathway-heatmap').innerHTML =
                '<p class="no-data">No pathway activation data available yet. Run scripts/22_tahoe_drug_signatures.py first.</p>';
            return;
        }

        const drugs = [...new Set(data.map(d => d.drug))].slice(0, 30);
        const pathways = [...new Set(data.map(d => d.pathway))];

        const zValues = drugs.map(drug =>
            pathways.map(pw => {
                const record = data.find(d => d.drug === drug && d.pathway === pw);
                return record ? record.activation_score : 0;
            })
        );

        Plotly.newPlot('pathway-heatmap', [{
            type: 'heatmap',
            z: zValues,
            x: pathways,
            y: drugs,
            colorscale: 'RdBu',
            reversescale: true,
            colorbar: { title: { text: 'Activation Score' } },
        }], {
            margin: { l: 120, r: 40, t: 20, b: 100 },
            xaxis: { title: 'Cytokine Pathway', tickangle: -45 },
            yaxis: { title: 'Drug' },
        }, { responsive: true });
    }

    return { init };
})();
