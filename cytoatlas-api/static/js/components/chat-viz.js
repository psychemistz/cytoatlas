/**
 * Chat Visualization Component
 * Renders inline visualizations in chat messages using Plotly
 */

const ChatViz = {
    /**
     * Render a visualization into a container
     * @param {HTMLElement} container - Container element
     * @param {Object} viz - Visualization config
     */
    render(container, viz) {
        if (!container) return;

        switch (viz.type) {
            case 'heatmap':
                this.renderHeatmap(container, viz);
                break;
            case 'bar_chart':
                this.renderBarChart(container, viz);
                break;
            case 'scatter':
                this.renderScatter(container, viz);
                break;
            case 'box_plot':
                this.renderBoxPlot(container, viz);
                break;
            case 'line_chart':
                this.renderLineChart(container, viz);
                break;
            case 'table':
                this.renderTable(container, viz);
                break;
            default:
                container.innerHTML = `<p>Unknown visualization type: ${viz.type}</p>`;
        }
    },

    /**
     * Render a heatmap
     */
    renderHeatmap(container, viz) {
        const { data, title, config } = viz;
        const { x_labels, y_labels, values } = data;

        const trace = {
            type: 'heatmap',
            x: x_labels,
            y: y_labels,
            z: values,
            colorscale: config.colorscale || 'RdBu',
            reversescale: config.reversescale !== false,
            hoverongaps: false,
            colorbar: {
                title: config.colorbar_title || 'Activity',
                thickness: 15,
            },
        };

        const layout = {
            title: title,
            xaxis: {
                title: config.x_title || '',
                tickangle: -45,
            },
            yaxis: {
                title: config.y_title || '',
            },
            margin: { l: 100, r: 50, t: 50, b: 100 },
            height: config.height || 400,
        };

        Plotly.newPlot(container, [trace], layout, { responsive: true });
    },

    /**
     * Render a bar chart
     */
    renderBarChart(container, viz) {
        const { data, title, config } = viz;
        const { labels, values, colors } = data;

        const trace = {
            type: 'bar',
            x: config.horizontal ? values : labels,
            y: config.horizontal ? labels : values,
            orientation: config.horizontal ? 'h' : 'v',
            marker: {
                color: colors || values.map(v => v >= 0 ? '#2563eb' : '#dc2626'),
            },
            text: values.map(v => v.toFixed(2)),
            textposition: 'auto',
        };

        const layout = {
            title: title,
            xaxis: {
                title: config.x_title || '',
                tickangle: config.horizontal ? 0 : -45,
            },
            yaxis: {
                title: config.y_title || '',
            },
            margin: { l: config.horizontal ? 150 : 60, r: 20, t: 50, b: config.horizontal ? 40 : 100 },
            height: config.height || 400,
        };

        Plotly.newPlot(container, [trace], layout, { responsive: true });
    },

    /**
     * Render a scatter plot
     */
    renderScatter(container, viz) {
        const { data, title, config } = viz;
        const { x, y, labels, colors } = data;

        const trace = {
            type: 'scatter',
            mode: 'markers',
            x: x,
            y: y,
            text: labels,
            marker: {
                size: config.marker_size || 8,
                color: colors || '#2563eb',
                colorscale: config.colorscale,
                showscale: Boolean(config.colorscale),
            },
            hovertemplate: config.hover_template || '%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>',
        };

        // Add trendline if specified
        const traces = [trace];
        if (config.trendline && x.length > 1) {
            const { slope, intercept } = this.linearRegression(x, y);
            const xMin = Math.min(...x);
            const xMax = Math.max(...x);
            traces.push({
                type: 'scatter',
                mode: 'lines',
                x: [xMin, xMax],
                y: [slope * xMin + intercept, slope * xMax + intercept],
                line: { color: '#94a3b8', dash: 'dash' },
                showlegend: false,
            });
        }

        const layout = {
            title: title,
            xaxis: { title: config.x_title || '' },
            yaxis: { title: config.y_title || '' },
            margin: { l: 60, r: 20, t: 50, b: 60 },
            height: config.height || 400,
            hovermode: 'closest',
        };

        Plotly.newPlot(container, traces, layout, { responsive: true });
    },

    /**
     * Render a box plot
     */
    renderBoxPlot(container, viz) {
        const { data, title, config } = viz;
        const { labels, values } = data;

        const traces = labels.map((label, i) => ({
            type: 'box',
            y: values[i],
            name: label,
            boxpoints: config.show_points ? 'all' : false,
            jitter: 0.3,
            pointpos: -1.8,
        }));

        const layout = {
            title: title,
            yaxis: { title: config.y_title || 'Value' },
            margin: { l: 60, r: 20, t: 50, b: 80 },
            height: config.height || 400,
        };

        Plotly.newPlot(container, traces, layout, { responsive: true });
    },

    /**
     * Render a line chart
     */
    renderLineChart(container, viz) {
        const { data, title, config } = viz;
        const { x, y, labels, series } = data;

        let traces;
        if (series) {
            // Multiple series
            traces = series.map(s => ({
                type: 'scatter',
                mode: 'lines+markers',
                x: s.x || x,
                y: s.y,
                name: s.name,
                line: { width: 2 },
            }));
        } else {
            // Single series
            traces = [{
                type: 'scatter',
                mode: 'lines+markers',
                x: x,
                y: y,
                text: labels,
                line: { width: 2, color: '#2563eb' },
            }];
        }

        const layout = {
            title: title,
            xaxis: { title: config.x_title || '' },
            yaxis: { title: config.y_title || '' },
            margin: { l: 60, r: 20, t: 50, b: 60 },
            height: config.height || 400,
            hovermode: 'x unified',
        };

        Plotly.newPlot(container, traces, layout, { responsive: true });
    },

    /**
     * Render a data table
     */
    renderTable(container, viz) {
        const { data, title, config } = viz;
        const { headers, rows } = data;

        const table = document.createElement('div');
        table.className = 'chat-viz-table';

        if (title) {
            const titleEl = document.createElement('h4');
            titleEl.textContent = title;
            table.appendChild(titleEl);
        }

        const tableEl = document.createElement('table');
        tableEl.className = 'viz-table';

        // Header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        headers.forEach(h => {
            const th = document.createElement('th');
            th.textContent = h;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        tableEl.appendChild(thead);

        // Body
        const tbody = document.createElement('tbody');
        const maxRows = config.max_rows || 20;
        rows.slice(0, maxRows).forEach(row => {
            const tr = document.createElement('tr');
            row.forEach((cell, i) => {
                const td = document.createElement('td');
                // Format numbers
                if (typeof cell === 'number') {
                    td.textContent = Number.isInteger(cell) ? cell : cell.toFixed(3);
                    td.className = 'numeric';
                } else {
                    td.textContent = cell;
                }
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        tableEl.appendChild(tbody);

        table.appendChild(tableEl);

        if (rows.length > maxRows) {
            const more = document.createElement('p');
            more.className = 'table-more';
            more.textContent = `Showing ${maxRows} of ${rows.length} rows`;
            table.appendChild(more);
        }

        container.appendChild(table);
    },

    /**
     * Simple linear regression
     */
    linearRegression(x, y) {
        const n = x.length;
        let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;

        for (let i = 0; i < n; i++) {
            sumX += x[i];
            sumY += y[i];
            sumXY += x[i] * y[i];
            sumXX += x[i] * x[i];
        }

        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;

        return { slope, intercept };
    },
};

// Make available globally
window.ChatViz = ChatViz;
