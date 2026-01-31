/**
 * Submit Page Handler
 * H5AD file upload and processing
 */

const SubmitPage = {
    uploadSession: null,
    currentFile: null,
    uploadProgress: 0,
    jobId: null,
    wsConnection: null,

    /**
     * Initialize the submit page
     */
    async init(params, query) {
        this.render();
        this.setupEventHandlers();

        // Check authentication
        const user = await this.checkAuth();
        if (!user) {
            this.showLoginRequired();
            return;
        }

        // Load existing jobs
        await this.loadJobs();
    },

    /**
     * Render the submit page template
     */
    render() {
        const app = document.getElementById('app');
        app.innerHTML = `
            <div class="submit-page">
                <header class="submit-header">
                    <h1>Submit Your Data</h1>
                    <p class="subtitle">
                        Upload your H5AD single-cell data to compute CytoSig and SecAct
                        activity signatures. Your atlas will be available for exploration
                        and comparison.
                    </p>
                </header>

                <div id="auth-required" class="auth-required hidden">
                    <div class="auth-card">
                        <h2>Login Required</h2>
                        <p>You need to be logged in to submit data.</p>
                        <button class="btn btn-primary" onclick="Router.navigate('/login')">
                            Log In
                        </button>
                        <p class="secondary">
                            Don't have an account?
                            <a href="/register">Register</a>
                        </p>
                    </div>
                </div>

                <div id="submit-content" class="submit-content hidden">
                    <div class="upload-section">
                        <h2>Upload H5AD File</h2>

                        <div class="upload-dropzone" id="upload-dropzone">
                            <div class="dropzone-content">
                                <div class="dropzone-icon">📁</div>
                                <p>Drag and drop your H5AD file here</p>
                                <p class="secondary">or</p>
                                <button class="btn btn-primary" id="file-select-btn">
                                    Select File
                                </button>
                                <input type="file" id="file-input" accept=".h5ad" hidden>
                            </div>
                            <p class="file-info">Maximum file size: 50GB</p>
                        </div>

                        <div id="selected-file" class="selected-file hidden">
                            <div class="file-details">
                                <span class="filename" id="selected-filename"></span>
                                <span class="filesize" id="selected-filesize"></span>
                            </div>
                            <button class="btn btn-sm btn-outline" id="clear-file-btn">
                                Clear
                            </button>
                        </div>
                    </div>

                    <div class="options-section hidden" id="options-section">
                        <h2>Atlas Options</h2>

                        <div class="form-group">
                            <label for="atlas-name">Atlas Name *</label>
                            <input type="text" id="atlas-name"
                                   placeholder="My Single-Cell Atlas"
                                   maxlength="100" required>
                        </div>

                        <div class="form-group">
                            <label for="atlas-description">Description</label>
                            <textarea id="atlas-description"
                                      placeholder="Brief description of your dataset..."
                                      rows="3"></textarea>
                        </div>

                        <div class="form-group">
                            <label>Signature Types</label>
                            <div class="checkbox-group">
                                <label class="checkbox-label">
                                    <input type="checkbox" id="sig-cytosig" checked>
                                    <span>CytoSig (43 cytokines)</span>
                                </label>
                                <label class="checkbox-label">
                                    <input type="checkbox" id="sig-secact" checked>
                                    <span>SecAct (1,170 secreted proteins)</span>
                                </label>
                            </div>
                        </div>

                        <div class="submit-actions">
                            <button class="btn btn-primary btn-lg" id="start-upload-btn">
                                Upload & Process
                            </button>
                        </div>
                    </div>

                    <div id="upload-progress" class="upload-progress hidden">
                        <h2>Upload Progress</h2>
                        <div class="progress-container">
                            <div class="progress-bar">
                                <div class="progress-fill" id="progress-fill"></div>
                            </div>
                            <div class="progress-text">
                                <span id="progress-percent">0%</span>
                                <span id="progress-status">Preparing...</span>
                            </div>
                        </div>
                        <button class="btn btn-outline" id="cancel-upload-btn">
                            Cancel
                        </button>
                    </div>

                    <div id="processing-status" class="processing-status hidden">
                        <h2>Processing Status</h2>
                        <div class="status-card">
                            <div class="status-header">
                                <span class="status-badge" id="job-status-badge">Pending</span>
                                <span class="job-id" id="job-id-display"></span>
                            </div>
                            <div class="progress-container">
                                <div class="progress-bar">
                                    <div class="progress-fill" id="job-progress-fill"></div>
                                </div>
                                <div class="progress-text">
                                    <span id="job-progress-percent">0%</span>
                                    <span id="job-current-step">Queued</span>
                                </div>
                            </div>
                            <div id="job-stats" class="job-stats hidden">
                                <div class="stat">
                                    <span class="label">Cells</span>
                                    <span class="value" id="job-n-cells">-</span>
                                </div>
                                <div class="stat">
                                    <span class="label">Samples</span>
                                    <span class="value" id="job-n-samples">-</span>
                                </div>
                                <div class="stat">
                                    <span class="label">Cell Types</span>
                                    <span class="value" id="job-n-cell-types">-</span>
                                </div>
                            </div>
                            <div id="job-error" class="job-error hidden"></div>
                            <div id="job-complete" class="job-complete hidden">
                                <p>Your atlas is ready!</p>
                                <button class="btn btn-primary" id="view-atlas-btn">
                                    View Atlas
                                </button>
                            </div>
                        </div>
                    </div>

                    <div class="jobs-section">
                        <h2>Your Jobs</h2>
                        <div id="jobs-list" class="jobs-list">
                            <p class="loading">Loading jobs...</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    },

    /**
     * Check authentication
     */
    async checkAuth() {
        try {
            const response = await fetch('/api/v1/auth/me', {
                credentials: 'include',
            });
            if (response.ok) {
                return await response.json();
            }
        } catch (error) {
            console.error('Auth check failed:', error);
        }
        return null;
    },

    /**
     * Show login required message
     */
    showLoginRequired() {
        document.getElementById('auth-required')?.classList.remove('hidden');
        document.getElementById('submit-content')?.classList.add('hidden');
    },

    /**
     * Set up event handlers
     */
    setupEventHandlers() {
        const dropzone = document.getElementById('upload-dropzone');
        const fileInput = document.getElementById('file-input');
        const fileSelectBtn = document.getElementById('file-select-btn');
        const clearFileBtn = document.getElementById('clear-file-btn');
        const startUploadBtn = document.getElementById('start-upload-btn');
        const cancelUploadBtn = document.getElementById('cancel-upload-btn');

        // File select button
        fileSelectBtn?.addEventListener('click', () => fileInput?.click());

        // File input change
        fileInput?.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });

        // Drag and drop
        dropzone?.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });

        dropzone?.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });

        dropzone?.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                this.handleFileSelect(e.dataTransfer.files[0]);
            }
        });

        // Clear file
        clearFileBtn?.addEventListener('click', () => this.clearFile());

        // Start upload
        startUploadBtn?.addEventListener('click', () => this.startUpload());

        // Cancel upload
        cancelUploadBtn?.addEventListener('click', () => this.cancelUpload());

        // Show content after auth check
        document.getElementById('submit-content')?.classList.remove('hidden');
    },

    /**
     * Handle file selection
     */
    handleFileSelect(file) {
        if (!file.name.endsWith('.h5ad')) {
            alert('Please select an H5AD file.');
            return;
        }

        const maxSize = 50 * 1024 * 1024 * 1024; // 50GB
        if (file.size > maxSize) {
            alert('File size exceeds 50GB limit.');
            return;
        }

        this.currentFile = file;

        // Update UI
        document.getElementById('upload-dropzone')?.classList.add('hidden');
        document.getElementById('selected-file')?.classList.remove('hidden');
        document.getElementById('selected-filename').textContent = file.name;
        document.getElementById('selected-filesize').textContent = this.formatFileSize(file.size);
        document.getElementById('options-section')?.classList.remove('hidden');

        // Pre-fill atlas name from filename
        const atlasNameInput = document.getElementById('atlas-name');
        if (atlasNameInput && !atlasNameInput.value) {
            atlasNameInput.value = file.name.replace('.h5ad', '');
        }
    },

    /**
     * Clear selected file
     */
    clearFile() {
        this.currentFile = null;
        document.getElementById('file-input').value = '';
        document.getElementById('upload-dropzone')?.classList.remove('hidden');
        document.getElementById('selected-file')?.classList.add('hidden');
        document.getElementById('options-section')?.classList.add('hidden');
    },

    /**
     * Format file size
     */
    formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
    },

    /**
     * Start upload process
     */
    async startUpload() {
        if (!this.currentFile) return;

        const atlasName = document.getElementById('atlas-name')?.value?.trim();
        if (!atlasName) {
            alert('Please enter an atlas name.');
            return;
        }

        const atlasDescription = document.getElementById('atlas-description')?.value?.trim();
        const useCytoSig = document.getElementById('sig-cytosig')?.checked;
        const useSecAct = document.getElementById('sig-secact')?.checked;

        if (!useCytoSig && !useSecAct) {
            alert('Please select at least one signature type.');
            return;
        }

        // Show progress UI
        document.getElementById('options-section')?.classList.add('hidden');
        document.getElementById('upload-progress')?.classList.remove('hidden');

        try {
            // Initialize upload
            this.updateProgress(0, 'Initializing upload...');

            const initResponse = await fetch('/api/v1/submit/upload/init', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({
                    filename: this.currentFile.name,
                    file_size: this.currentFile.size,
                    atlas_name: atlasName,
                    atlas_description: atlasDescription,
                }),
            });

            if (!initResponse.ok) {
                const error = await initResponse.json();
                throw new Error(error.detail || 'Failed to initialize upload');
            }

            this.uploadSession = await initResponse.json();

            // Upload chunks
            await this.uploadChunks();

            // Complete upload
            this.updateProgress(95, 'Finalizing upload...');

            const completeResponse = await fetch('/api/v1/submit/upload/complete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({
                    upload_id: this.uploadSession.upload_id,
                }),
            });

            if (!completeResponse.ok) {
                const error = await completeResponse.json();
                throw new Error(error.detail || 'Failed to complete upload');
            }

            const uploadResult = await completeResponse.json();

            // Start processing
            this.updateProgress(100, 'Starting processing...');

            const signatureTypes = [];
            if (useCytoSig) signatureTypes.push('CytoSig');
            if (useSecAct) signatureTypes.push('SecAct');

            const processResponse = await fetch('/api/v1/submit/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({
                    file_path: uploadResult.file_path,
                    atlas_name: atlasName,
                    atlas_description: atlasDescription,
                    signature_types: signatureTypes,
                }),
            });

            if (!processResponse.ok) {
                const error = await processResponse.json();
                throw new Error(error.detail || 'Failed to start processing');
            }

            const processResult = await processResponse.json();
            this.jobId = processResult.job_id;

            // Switch to processing status view
            document.getElementById('upload-progress')?.classList.add('hidden');
            document.getElementById('processing-status')?.classList.remove('hidden');
            document.getElementById('job-id-display').textContent = `Job #${this.jobId}`;

            // Start WebSocket monitoring
            this.connectJobWebSocket();

        } catch (error) {
            console.error('Upload failed:', error);
            alert(`Upload failed: ${error.message}`);
            this.resetUpload();
        }
    },

    /**
     * Upload file chunks
     */
    async uploadChunks() {
        const chunkSize = this.uploadSession.chunk_size;
        const totalChunks = this.uploadSession.total_chunks;
        const file = this.currentFile;

        for (let i = 0; i < totalChunks; i++) {
            const start = i * chunkSize;
            const end = Math.min(start + chunkSize, file.size);
            const chunk = file.slice(start, end);

            const formData = new FormData();
            formData.append('upload_id', this.uploadSession.upload_id);
            formData.append('chunk_index', i);
            formData.append('chunk', chunk);

            const response = await fetch('/api/v1/submit/upload/chunk', {
                method: 'POST',
                credentials: 'include',
                body: formData,
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || `Failed to upload chunk ${i}`);
            }

            const progress = Math.round(((i + 1) / totalChunks) * 90);
            this.updateProgress(progress, `Uploading... ${i + 1}/${totalChunks} chunks`);
        }
    },

    /**
     * Update progress display
     */
    updateProgress(percent, status) {
        this.uploadProgress = percent;
        document.getElementById('progress-fill').style.width = `${percent}%`;
        document.getElementById('progress-percent').textContent = `${percent}%`;
        document.getElementById('progress-status').textContent = status;
    },

    /**
     * Cancel upload
     */
    cancelUpload() {
        // TODO: Implement upload cancellation
        this.resetUpload();
    },

    /**
     * Reset upload state
     */
    resetUpload() {
        this.uploadSession = null;
        this.uploadProgress = 0;
        document.getElementById('upload-progress')?.classList.add('hidden');
        document.getElementById('options-section')?.classList.remove('hidden');
    },

    /**
     * Connect WebSocket for job monitoring
     */
    connectJobWebSocket() {
        if (this.wsConnection) {
            this.wsConnection.close();
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/v1/ws/jobs/${this.jobId}`;

        this.wsConnection = new WebSocket(wsUrl);

        this.wsConnection.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.updateJobStatus(data);
        };

        this.wsConnection.onerror = (error) => {
            console.error('WebSocket error:', error);
            // Fall back to polling
            this.startJobPolling();
        };

        this.wsConnection.onclose = () => {
            // Check if job is complete, otherwise reconnect
            if (this.jobId) {
                setTimeout(() => this.connectJobWebSocket(), 5000);
            }
        };
    },

    /**
     * Start polling for job status (fallback)
     */
    startJobPolling() {
        const pollInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/v1/submit/jobs/${this.jobId}`, {
                    credentials: 'include',
                });
                if (response.ok) {
                    const job = await response.json();
                    this.updateJobStatus(job);

                    if (['completed', 'failed', 'cancelled'].includes(job.status)) {
                        clearInterval(pollInterval);
                    }
                }
            } catch (error) {
                console.error('Polling error:', error);
            }
        }, 3000);
    },

    /**
     * Update job status display
     */
    updateJobStatus(data) {
        const statusBadge = document.getElementById('job-status-badge');
        const progressFill = document.getElementById('job-progress-fill');
        const progressPercent = document.getElementById('job-progress-percent');
        const currentStep = document.getElementById('job-current-step');
        const statsSection = document.getElementById('job-stats');
        const errorSection = document.getElementById('job-error');
        const completeSection = document.getElementById('job-complete');

        // Update status badge
        statusBadge.textContent = data.status || data.step;
        statusBadge.className = `status-badge ${data.status}`;

        // Update progress
        const progress = data.progress || 0;
        progressFill.style.width = `${progress}%`;
        progressPercent.textContent = `${progress}%`;
        currentStep.textContent = data.current_step || data.step || '';

        // Update stats if available
        if (data.n_cells || data.n_samples || data.n_cell_types) {
            statsSection.classList.remove('hidden');
            if (data.n_cells) {
                document.getElementById('job-n-cells').textContent = data.n_cells.toLocaleString();
            }
            if (data.n_samples) {
                document.getElementById('job-n-samples').textContent = data.n_samples.toLocaleString();
            }
            if (data.n_cell_types) {
                document.getElementById('job-n-cell-types').textContent = data.n_cell_types.toLocaleString();
            }
        }

        // Handle completion states
        if (data.status === 'completed') {
            completeSection.classList.remove('hidden');
            errorSection.classList.add('hidden');
            document.getElementById('view-atlas-btn')?.addEventListener('click', () => {
                // Navigate to the new atlas
                Router.navigate('/explore');
            });
        } else if (data.status === 'failed') {
            errorSection.classList.remove('hidden');
            errorSection.textContent = data.error_message || data.error || 'Processing failed';
            completeSection.classList.add('hidden');
        }
    },

    /**
     * Load existing jobs
     */
    async loadJobs() {
        const container = document.getElementById('jobs-list');

        try {
            const response = await fetch('/api/v1/submit/jobs', {
                credentials: 'include',
            });

            if (!response.ok) {
                container.innerHTML = '<p class="error">Failed to load jobs</p>';
                return;
            }

            const data = await response.json();

            if (data.jobs.length === 0) {
                container.innerHTML = '<p class="empty">No jobs yet. Upload your first H5AD file!</p>';
                return;
            }

            container.innerHTML = data.jobs.map(job => this.renderJobCard(job)).join('');

        } catch (error) {
            console.error('Failed to load jobs:', error);
            container.innerHTML = '<p class="error">Failed to load jobs</p>';
        }
    },

    /**
     * Render a job card
     */
    renderJobCard(job) {
        const duration = job.duration_seconds
            ? this.formatDuration(job.duration_seconds)
            : '-';

        return `
            <div class="job-card ${job.status}">
                <div class="job-header">
                    <span class="job-name">${job.atlas_name}</span>
                    <span class="status-badge ${job.status}">${job.status}</span>
                </div>
                <div class="job-meta">
                    <span>Created: ${new Date(job.created_at).toLocaleDateString()}</span>
                    <span>Duration: ${duration}</span>
                </div>
                ${job.status === 'processing' ? `
                    <div class="progress-bar small">
                        <div class="progress-fill" style="width: ${job.progress}%"></div>
                    </div>
                    <span class="progress-text">${job.progress}% - ${job.current_step}</span>
                ` : ''}
                ${job.status === 'completed' ? `
                    <div class="job-stats-mini">
                        ${job.n_cells ? `<span>${job.n_cells.toLocaleString()} cells</span>` : ''}
                        ${job.n_cell_types ? `<span>${job.n_cell_types} cell types</span>` : ''}
                    </div>
                    <button class="btn btn-sm btn-primary"
                            onclick="Router.navigate('/atlas/${job.atlas_name}')">
                        View Atlas
                    </button>
                ` : ''}
                ${job.status === 'failed' ? `
                    <p class="error-message">${job.error_message || 'Unknown error'}</p>
                ` : ''}
            </div>
        `;
    },

    /**
     * Format duration in seconds
     */
    formatDuration(seconds) {
        if (seconds < 60) return `${seconds}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
        const hours = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${mins}m`;
    },
};

// Make available globally
window.SubmitPage = SubmitPage;
