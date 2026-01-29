/**
 * CytoAtlas API Client
 * Provides methods for interacting with the CytoAtlas REST API
 */

const API = {
    BASE_URL: '/api/v1',

    /**
     * Make an API request
     * @param {string} endpoint - API endpoint (without base URL)
     * @param {Object} options - Fetch options
     * @returns {Promise<Object>} Response data
     */
    async request(endpoint, options = {}) {
        const url = `${this.BASE_URL}${endpoint}`;

        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };

        const mergedOptions = { ...defaultOptions, ...options };

        try {
            const response = await fetch(url, mergedOptions);

            if (!response.ok) {
                const error = await response.json().catch(() => ({ error: response.statusText }));
                throw new Error(error.detail || error.error || 'API request failed');
            }

            return await response.json();
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            throw error;
        }
    },

    /**
     * GET request
     */
    async get(endpoint, params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const url = queryString ? `${endpoint}?${queryString}` : endpoint;
        return this.request(url, { method: 'GET' });
    },

    /**
     * POST request
     */
    async post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data),
        });
    },

    // ==================== Health ====================

    async health() {
        return this.get('/health');
    },

    // ==================== Atlases ====================

    async getAtlases() {
        return this.get('/atlases');
    },

    async getAtlas(atlasName) {
        return this.get(`/atlases/${atlasName}`);
    },

    async getAtlasSummary(atlasName) {
        return this.get(`/atlases/${atlasName}/summary`);
    },

    async getAtlasActivity(atlasName, params = {}) {
        return this.get(`/atlases/${atlasName}/activity`, params);
    },

    async getAtlasCellTypes(atlasName) {
        return this.get(`/atlases/${atlasName}/cell-types`);
    },

    async getAtlasSignatures(atlasName, type = 'CytoSig') {
        return this.get(`/atlases/${atlasName}/signatures`, { signature_type: type });
    },

    // ==================== CIMA ====================

    async getCimaSummary() {
        return this.get('/cima/summary');
    },

    async getCimaActivity(params = {}) {
        return this.get('/cima/activity', params);
    },

    async getCimaCorrelations(variable, params = {}) {
        return this.get(`/cima/correlations/${variable}`, params);
    },

    async getCimaAgeBmiStratified(params = {}) {
        return this.get('/cima/age-bmi-stratified', params);
    },

    async getCimaDifferential(params = {}) {
        return this.get('/cima/differential', params);
    },

    async getCimaMetabolites(params = {}) {
        return this.get('/cima/metabolites', params);
    },

    // ==================== Inflammation ====================

    async getInflammationSummary() {
        return this.get('/inflammation/summary');
    },

    async getInflammationActivity(params = {}) {
        return this.get('/inflammation/activity', params);
    },

    async getInflammationDifferential(params = {}) {
        return this.get('/inflammation/disease-differential', params);
    },

    async getInflammationAgeBmiStratified(params = {}) {
        return this.get('/inflammation/age-bmi-stratified', params);
    },

    async getInflammationPrediction(params = {}) {
        return this.get('/inflammation/treatment-prediction', params);
    },

    // ==================== scAtlas ====================

    async getScatlasSummary() {
        return this.get('/scatlas/summary');
    },

    async getScatlasActivity(params = {}) {
        return this.get('/scatlas/activity', params);
    },

    async getScatlasOrgans() {
        return this.get('/scatlas/organs');
    },

    async getScatlasCellTypes() {
        return this.get('/scatlas/cell-types');
    },

    // ==================== Cross-Atlas ====================

    async getCrossAtlasComparison(params = {}) {
        return this.get('/cross-atlas/comparison', params);
    },

    async getCrossAtlasConsistency(params = {}) {
        return this.get('/cross-atlas/consistency', params);
    },

    // ==================== Validation ====================

    async getValidationAtlases() {
        return this.get('/validation/atlases');
    },

    async getValidationSummary(atlas, signatureType = 'CytoSig') {
        return this.get(`/validation/summary/${atlas}`, { signature_type: signatureType });
    },

    async getValidationSignatures(atlas, signatureType = 'CytoSig') {
        return this.get(`/validation/signatures/${atlas}`, { signature_type: signatureType });
    },

    async getSampleLevelValidation(atlas, signature, signatureType = 'CytoSig', cellType = null) {
        const params = { signature_type: signatureType };
        if (cellType) params.cell_type = cellType;
        return this.get(`/validation/sample-level/${atlas}/${signature}`, params);
    },

    async getCellTypeLevelValidation(atlas, signature, signatureType = 'CytoSig') {
        return this.get(`/validation/celltype-level/${atlas}/${signature}`, { signature_type: signatureType });
    },

    async getPseudobulkVsSingleCell(atlas, signature, signatureType = 'CytoSig') {
        return this.get(`/validation/pseudobulk-vs-singlecell/${atlas}/${signature}`, { signature_type: signatureType });
    },

    async getSingleCellDirect(atlas, signature, signatureType = 'CytoSig', cellType = null) {
        const params = { signature_type: signatureType };
        if (cellType) params.cell_type = cellType;
        return this.get(`/validation/singlecell-direct/${atlas}/${signature}`, params);
    },

    async getSingleCellDistribution(atlas, signature, signatureType = 'CytoSig', cellType = null) {
        const params = { signature_type: signatureType };
        if (cellType) params.cell_type = cellType;
        return this.get(`/validation/singlecell-distribution/${atlas}/${signature}`, params);
    },

    async getBiologicalAssociations(atlas, signatureType = 'CytoSig') {
        return this.get(`/validation/biological-associations/${atlas}`, { signature_type: signatureType });
    },

    async getGeneCoverage(atlas, signature, signatureType = 'CytoSig') {
        return this.get(`/validation/gene-coverage/${atlas}/${signature}`, { signature_type: signatureType });
    },

    async getCVStability(atlas, signatureType = 'CytoSig', cellType = null) {
        const params = { signature_type: signatureType };
        if (cellType) params.cell_type = cellType;
        return this.get(`/validation/cv-stability/${atlas}`, params);
    },

    async compareAtlasValidation(signatureType = 'CytoSig') {
        return this.get('/validation/compare-atlases', { signature_type: signatureType });
    },

    // ==================== Export ====================

    getExportUrl(atlas, format = 'csv', type = 'activity') {
        return `${this.BASE_URL}/export/${atlas}/${type}.${format}`;
    },
};

// Make API available globally
window.API = API;
