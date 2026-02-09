/**
 * AppState - Application State Management
 * Simple reactive state store with subscriber pattern
 */

class AppState {
    constructor() {
        this._state = {
            signatureType: 'CytoSig',
            atlas: null,
            cellType: null,
            disease: null,
            organ: null,
            signature: null,
            filters: {}
        };
        this._subscribers = new Map();
    }

    /**
     * Get a state value
     * @param {string} key - State key
     * @returns {*} State value
     */
    get(key) {
        return this._state[key];
    }

    /**
     * Set a state value and notify subscribers
     * @param {string} key - State key
     * @param {*} value - New value
     */
    set(key, value) {
        const oldValue = this._state[key];

        if (oldValue === value) {
            return; // No change
        }

        this._state[key] = value;

        // Notify subscribers for this key
        if (this._subscribers.has(key)) {
            this._subscribers.get(key).forEach(callback => {
                try {
                    callback(value, oldValue);
                } catch (error) {
                    console.error(`Error in state subscriber for ${key}:`, error);
                }
            });
        }

        // Notify wildcard subscribers
        if (this._subscribers.has('*')) {
            this._subscribers.get('*').forEach(callback => {
                try {
                    callback(key, value, oldValue);
                } catch (error) {
                    console.error('Error in wildcard state subscriber:', error);
                }
            });
        }
    }

    /**
     * Subscribe to state changes
     * @param {string} key - State key to watch (or '*' for all changes)
     * @param {Function} callback - Callback function
     * @returns {Function} Unsubscribe function
     */
    subscribe(key, callback) {
        if (!this._subscribers.has(key)) {
            this._subscribers.set(key, new Set());
        }

        this._subscribers.get(key).add(callback);

        // Return unsubscribe function
        return () => {
            this.unsubscribe(key, callback);
        };
    }

    /**
     * Unsubscribe from state changes
     * @param {string} key - State key
     * @param {Function} callback - Callback function to remove
     */
    unsubscribe(key, callback) {
        if (this._subscribers.has(key)) {
            this._subscribers.get(key).delete(callback);
        }
    }

    /**
     * Get all state
     * @returns {Object} Copy of current state
     */
    getAll() {
        return { ...this._state };
    }

    /**
     * Update multiple state values at once
     * @param {Object} updates - Object with key-value pairs to update
     */
    setMany(updates) {
        Object.entries(updates).forEach(([key, value]) => {
            this.set(key, value);
        });
    }

    /**
     * Reset state to initial values
     */
    reset() {
        this.setMany({
            signatureType: 'CytoSig',
            atlas: null,
            cellType: null,
            disease: null,
            organ: null,
            signature: null,
            filters: {}
        });
    }

    /**
     * Clear all subscribers
     */
    clearSubscribers() {
        this._subscribers.clear();
    }
}

// Create and export global instance
const appState = new AppState();
window.appState = appState;
