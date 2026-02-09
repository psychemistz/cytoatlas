/**
 * ExportButton Component
 * Export button with CSV, PNG, and JSON support
 */

class ExportButton {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container ${containerId} not found`);
        }
        this.containerId = containerId;
        this.options = { ...ExportButton.DEFAULTS, ...options };

        this.data = null;
        this.chart = null;

        this.render();
    }

    static DEFAULTS = {
        formats: ['CSV', 'PNG'], // Available export formats
        label: 'Export',
        showDropdown: true,
    };

    /**
     * Render the export button
     */
    render() {
        if (this.options.showDropdown && this.options.formats.length > 1) {
            this.container.innerHTML = `
                <div class="export-button dropdown">
                    <button class="btn btn-secondary dropdown-toggle">
                        ${this.options.label}
                    </button>
                    <div class="dropdown-menu" id="${this.containerId}-menu">
                        ${this.options.formats.map(format => `
                            <button class="dropdown-item" data-format="${format}">
                                Export as ${format}
                            </button>
                        `).join('')}
                    </div>
                </div>
            `;

            this.attachDropdownListeners();
        } else {
            // Single button for one format
            const format = this.options.formats[0];
            this.container.innerHTML = `
                <button class="btn btn-secondary export-btn" data-format="${format}">
                    ${this.options.label} ${format}
                </button>
            `;

            this.attachButtonListener();
        }
    }

    /**
     * Attach dropdown listeners
     */
    attachDropdownListeners() {
        const toggle = this.container.querySelector('.dropdown-toggle');
        const menu = this.container.querySelector('.dropdown-menu');

        toggle.addEventListener('click', (e) => {
            e.stopPropagation();
            menu.classList.toggle('show');
        });

        document.addEventListener('click', () => {
            menu.classList.remove('show');
        });

        this.container.querySelectorAll('.dropdown-item').forEach(item => {
            item.addEventListener('click', () => {
                const format = item.dataset.format;
                this.export(format);
                menu.classList.remove('show');
            });
        });
    }

    /**
     * Attach button listener
     */
    attachButtonListener() {
        const button = this.container.querySelector('.export-btn');
        button.addEventListener('click', () => {
            const format = button.dataset.format;
            this.export(format);
        });
    }

    /**
     * Set data for CSV/JSON export
     * @param {*} data - Data to export
     */
    setData(data) {
        this.data = data;
    }

    /**
     * Set chart for PNG export (Plotly div ID or chart object)
     * @param {string|Object} chart - Plotly div ID or chart object with exportPNG method
     */
    setChart(chart) {
        this.chart = chart;
    }

    /**
     * Export data in specified format
     * @param {string} format - Export format (CSV, PNG, JSON)
     */
    export(format) {
        switch (format.toUpperCase()) {
            case 'CSV':
                this.exportCSV();
                break;
            case 'PNG':
                this.exportPNG();
                break;
            case 'JSON':
                this.exportJSON();
                break;
            default:
                console.warn(`Unknown export format: ${format}`);
        }
    }

    /**
     * Export as CSV
     */
    exportCSV() {
        let csvContent = '';

        if (this.chart && typeof this.chart.exportCSV === 'function') {
            // Chart object with exportCSV method
            csvContent = this.chart.exportCSV();
        } else if (this.data) {
            // Convert data to CSV
            csvContent = this.dataToCSV(this.data);
        } else {
            console.warn('No data available for CSV export');
            return;
        }

        this.downloadFile(csvContent, 'export.csv', 'text/csv');
    }

    /**
     * Export as PNG
     */
    exportPNG() {
        if (this.chart) {
            if (typeof this.chart.exportPNG === 'function') {
                // Chart object with exportPNG method
                this.chart.exportPNG();
            } else if (typeof this.chart === 'string') {
                // Plotly div ID
                Plotly.downloadImage(this.chart, {
                    format: 'png',
                    filename: 'export',
                    width: 1200,
                    height: 800,
                    scale: 2
                });
            }
        } else {
            console.warn('No chart available for PNG export');
        }
    }

    /**
     * Export as JSON
     */
    exportJSON() {
        if (!this.data) {
            console.warn('No data available for JSON export');
            return;
        }

        const jsonContent = JSON.stringify(this.data, null, 2);
        this.downloadFile(jsonContent, 'export.json', 'application/json');
    }

    /**
     * Convert data to CSV format
     * @param {*} data - Data to convert
     * @returns {string} CSV string
     */
    dataToCSV(data) {
        if (Array.isArray(data)) {
            // Array of objects
            if (data.length === 0) return '';

            const headers = Object.keys(data[0]);
            const rows = [headers.join(',')];

            data.forEach(row => {
                const values = headers.map(header => {
                    const value = row[header];
                    // Escape commas and quotes
                    if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
                        return `"${value.replace(/"/g, '""')}"`;
                    }
                    return value;
                });
                rows.push(values.join(','));
            });

            return rows.join('\n');
        } else if (data.z && data.x && data.y) {
            // Heatmap-style data
            const rows = [['', ...data.x]];
            data.z.forEach((row, i) => {
                rows.push([data.y[i], ...row]);
            });
            return rows.map(row => row.join(',')).join('\n');
        } else {
            // Generic object - convert to key-value CSV
            const rows = [['Key', 'Value']];
            Object.entries(data).forEach(([key, value]) => {
                rows.push([key, JSON.stringify(value)]);
            });
            return rows.map(row => row.join(',')).join('\n');
        }
    }

    /**
     * Download file
     * @param {string} content - File content
     * @param {string} filename - File name
     * @param {string} mimeType - MIME type
     */
    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.click();
        URL.revokeObjectURL(url);
    }

    /**
     * Destroy the export button
     */
    destroy() {
        this.container.innerHTML = '';
        this.data = null;
        this.chart = null;
    }
}

// Make available globally
window.ExportButton = ExportButton;
