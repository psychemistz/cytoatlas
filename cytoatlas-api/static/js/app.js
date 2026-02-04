/**
 * CytoAtlas Main Application
 * Initializes routing and global handlers
 */

(function() {
    'use strict';

    /**
     * Initialize the application
     */
    function init() {
        // Register routes
        router.register('/', (params, query) => HomePage.init());
        router.register('/search', (params, query) => SearchPage.init(params, query));
        router.register('/gene/:signature', (params, query) => GeneDetailPage.init(params, query));
        router.register('/explore', (params, query) => ExplorePage.init(params, query));
        router.register('/atlas/:name', (params, query) => AtlasDetailPage.init(params));
        router.register('/validate', (params, query) => ValidatePage.init(params, query));
        router.register('/compare', (params, query) => ComparePage.init(params, query));
        router.register('/methods', (params, query) => MethodsPage.init(params, query));
        router.register('/submit', (params, query) => SubmitPage.init(params, query));
        router.register('/chat', (params, query) => ChatPage.init(params, query));
        router.register('/chat/:conversationId', (params, query) => ChatPage.init(params, query));

        // Initialize router
        router.init();

        // Check API health
        checkApiHealth();

        console.log('CytoAtlas initialized');
    }

    /**
     * Check API health and show status
     */
    async function checkApiHealth() {
        try {
            const health = await API.health();
            if (health.status !== 'healthy') {
                console.warn('API health check:', health);
            }
        } catch (error) {
            console.warn('API not available, using offline mode:', error.message);
        }
    }

    /**
     * Submit page handler (placeholder)
     */
    window.SubmitPage = {
        init(params, query) {
            const app = document.getElementById('app');
            const template = document.getElementById('submit-template');

            if (app && template) {
                app.innerHTML = template.innerHTML;
            }

            // Set up upload zone
            this.setupUploadZone();
        },

        setupUploadZone() {
            const uploadZone = document.getElementById('upload-zone');
            const fileInput = document.getElementById('file-input');

            if (!uploadZone || !fileInput) return;

            // Click to browse
            uploadZone.addEventListener('click', () => fileInput.click());

            // Drag and drop
            uploadZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadZone.classList.add('dragover');
            });

            uploadZone.addEventListener('dragleave', () => {
                uploadZone.classList.remove('dragover');
            });

            uploadZone.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadZone.classList.remove('dragover');

                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.handleFile(files[0]);
                }
            });

            // File input change
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.handleFile(e.target.files[0]);
                }
            });
        },

        handleFile(file) {
            if (!file.name.endsWith('.h5ad')) {
                alert('Please upload an H5AD file');
                return;
            }

            const uploadZone = document.getElementById('upload-zone');
            uploadZone.innerHTML = `
                <div class="upload-icon">&#128196;</div>
                <p><strong>${file.name}</strong></p>
                <p class="upload-hint">${(file.size / 1024 / 1024).toFixed(1)} MB</p>
            `;

            // Enable submit button
            const submitBtn = document.querySelector('.submit-actions .btn-primary');
            if (submitBtn) {
                submitBtn.disabled = false;
            }
        },
    };

    /**
     * Global submit handler
     */
    window.submitDataset = async function() {
        const name = document.getElementById('dataset-name')?.value;
        const desc = document.getElementById('dataset-desc')?.value;

        if (!name) {
            alert('Please enter a dataset name');
            return;
        }

        alert('Dataset submission will be available in a future update. This is a demo.');
    };

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
