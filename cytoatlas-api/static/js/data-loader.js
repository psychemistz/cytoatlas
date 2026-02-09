/**
 * DataLoader - Lazy Data Loading with Caching
 * Handles API requests with caching and request deduplication
 */

class DataLoader {
    constructor(apiClient) {
        this.api = apiClient || window.API;
        this._cache = new Map();
        this._pending = new Map(); // For request deduplication
        this.cacheTimeout = 5 * 60 * 1000; // 5 minutes default cache timeout
    }

    /**
     * Load data from API with caching
     * @param {string} endpoint - API endpoint
     * @param {Object} params - Query parameters
     * @param {Object} options - Options (force, cache, etc.)
     * @returns {Promise<Object>} Data
     */
    async load(endpoint, params = {}, options = {}) {
        const key = this._cacheKey(endpoint, params);

        // Force reload if requested
        if (options.force) {
            this._cache.delete(key);
        }

        // Return cached data if available and not expired
        if (this._cache.has(key)) {
            const cached = this._cache.get(key);
            const now = Date.now();

            if (!cached.expiry || now < cached.expiry) {
                console.log(`[DataLoader] Cache hit: ${endpoint}`);
                return cached.data;
            } else {
                // Expired - remove from cache
                this._cache.delete(key);
            }
        }

        // Deduplicate concurrent requests
        if (this._pending.has(key)) {
            console.log(`[DataLoader] Deduplicating request: ${endpoint}`);
            return this._pending.get(key);
        }

        // Make API request
        console.log(`[DataLoader] Loading: ${endpoint}`, params);

        const promise = this._makeRequest(endpoint, params)
            .then(data => {
                // Cache the result
                const expiry = options.cache === false ? null : Date.now() + this.cacheTimeout;
                this._cache.set(key, { data, expiry });
                return data;
            })
            .catch(error => {
                console.error(`[DataLoader] Error loading ${endpoint}:`, error);
                throw error;
            })
            .finally(() => {
                // Remove from pending
                this._pending.delete(key);
            });

        this._pending.set(key, promise);
        return promise;
    }

    /**
     * Make API request (uses API client method if available, otherwise generic get)
     * @param {string} endpoint - API endpoint
     * @param {Object} params - Query parameters
     * @returns {Promise<Object>} Data
     */
    async _makeRequest(endpoint, params) {
        // Use API client's get method
        return this.api.get(endpoint, params);
    }

    /**
     * Invalidate cache entries matching a pattern
     * @param {string|RegExp} pattern - Pattern to match (string or regex)
     */
    invalidate(pattern) {
        const regex = typeof pattern === 'string'
            ? new RegExp(pattern.replace(/\*/g, '.*'))
            : pattern;

        const keysToDelete = [];
        for (const key of this._cache.keys()) {
            if (regex.test(key)) {
                keysToDelete.push(key);
            }
        }

        keysToDelete.forEach(key => this._cache.delete(key));

        if (keysToDelete.length > 0) {
            console.log(`[DataLoader] Invalidated ${keysToDelete.length} cache entries`);
        }
    }

    /**
     * Clear all cached data
     */
    clearCache() {
        this._cache.clear();
        console.log('[DataLoader] Cache cleared');
    }

    /**
     * Get cache statistics
     * @returns {Object} Cache stats
     */
    getCacheStats() {
        return {
            size: this._cache.size,
            pending: this._pending.size,
            entries: Array.from(this._cache.keys())
        };
    }

    /**
     * Generate cache key from endpoint and params
     * @param {string} endpoint - API endpoint
     * @param {Object} params - Query parameters
     * @returns {string} Cache key
     */
    _cacheKey(endpoint, params) {
        const sortedParams = Object.keys(params || {})
            .sort()
            .reduce((acc, key) => {
                acc[key] = params[key];
                return acc;
            }, {});

        const queryString = new URLSearchParams(sortedParams).toString();
        return queryString ? `${endpoint}?${queryString}` : endpoint;
    }

    /**
     * Preload data into cache
     * @param {string} endpoint - API endpoint
     * @param {Object} params - Query parameters
     * @returns {Promise<void>}
     */
    async preload(endpoint, params = {}) {
        try {
            await this.load(endpoint, params);
            console.log(`[DataLoader] Preloaded: ${endpoint}`);
        } catch (error) {
            console.warn(`[DataLoader] Preload failed: ${endpoint}`, error);
        }
    }

    /**
     * Batch preload multiple endpoints
     * @param {Array} endpoints - Array of {endpoint, params} objects
     * @returns {Promise<void>}
     */
    async batchPreload(endpoints) {
        const promises = endpoints.map(({ endpoint, params }) =>
            this.preload(endpoint, params)
        );
        await Promise.all(promises);
    }
}

// Create and export global instance
const dataLoader = new DataLoader(window.API);
window.dataLoader = dataLoader;
