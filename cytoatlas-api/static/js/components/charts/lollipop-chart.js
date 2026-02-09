/**
 * LollipopChart Component
 * D3 lollipop chart for ranked signatures
 */

class LollipopChart {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container ${containerId} not found`);
        }
        this.containerId = containerId;
        this.options = { ...LollipopChart.DEFAULTS, ...options };
        this.svg = null;
        this.data = null;
    }

    static DEFAULTS = {
        title: '',
        xLabel: 'Value',
        yLabel: '',
        color: '#2563eb',
        circleRadius: 5,
        orientation: 'h', // 'h' or 'v'
        margin: { top: 40, right: 40, bottom: 80, left: 150 },
    };

    render(data) {
        if (!data || !data.categories || data.categories.length === 0) {
            this.container.innerHTML = '<p class="loading">No data available</p>';
            return;
        }

        this.data = data;
        this.container.innerHTML = ''; // Clear container

        const margin = this.options.margin;
        const width = this.container.clientWidth || 800;
        const height = Math.max(400, data.categories.length * 25);
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = height - margin.top - margin.bottom;

        // Create SVG
        this.svg = d3.select(`#${this.containerId}`)
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        const g = this.svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Scales
        const x = d3.scaleLinear()
            .domain([0, d3.max(data.values)])
            .range([0, plotWidth])
            .nice();

        const y = d3.scaleBand()
            .domain(data.categories)
            .range([0, plotHeight])
            .padding(0.2);

        // Axes
        g.append('g')
            .attr('transform', `translate(0,${plotHeight})`)
            .call(d3.axisBottom(x))
            .selectAll('text')
            .style('font-family', 'Inter, system-ui, sans-serif');

        g.append('g')
            .call(d3.axisLeft(y))
            .selectAll('text')
            .style('font-family', 'Inter, system-ui, sans-serif');

        // Lines
        g.selectAll('.lollipop-line')
            .data(data.categories)
            .enter()
            .append('line')
            .attr('class', 'lollipop-line')
            .attr('x1', 0)
            .attr('x2', (d, i) => x(data.values[i]))
            .attr('y1', d => y(d) + y.bandwidth() / 2)
            .attr('y2', d => y(d) + y.bandwidth() / 2)
            .attr('stroke', this.options.color)
            .attr('stroke-width', 2);

        // Circles
        g.selectAll('.lollipop-circle')
            .data(data.categories)
            .enter()
            .append('circle')
            .attr('class', 'lollipop-circle')
            .attr('cx', (d, i) => x(data.values[i]))
            .attr('cy', d => y(d) + y.bandwidth() / 2)
            .attr('r', this.options.circleRadius)
            .attr('fill', this.options.color)
            .attr('stroke', 'white')
            .attr('stroke-width', 2);

        // Title
        if (this.options.title) {
            this.svg.append('text')
                .attr('x', width / 2)
                .attr('y', 20)
                .attr('text-anchor', 'middle')
                .style('font-family', 'Inter, system-ui, sans-serif')
                .style('font-size', '16px')
                .style('font-weight', '600')
                .text(this.options.title);
        }

        // Axis labels
        if (this.options.xLabel) {
            this.svg.append('text')
                .attr('x', width / 2)
                .attr('y', height - 10)
                .attr('text-anchor', 'middle')
                .style('font-family', 'Inter, system-ui, sans-serif')
                .text(this.options.xLabel);
        }
    }

    update(data) {
        this.destroy();
        this.render(data);
    }

    resize() {
        if (this.svg) {
            this.update(this.data);
        }
    }

    destroy() {
        if (this.svg) {
            d3.select(`#${this.containerId}`).selectAll('*').remove();
            this.svg = null;
        }
    }

    exportCSV() {
        if (!this.data) return '';

        const rows = [['Category', 'Value']];
        this.data.categories.forEach((cat, i) => {
            rows.push([cat, this.data.values[i]]);
        });

        return rows.map(row => row.join(',')).join('\n');
    }

    exportPNG() {
        if (this.svg) {
            const svgNode = this.svg.node();
            const serializer = new XMLSerializer();
            const svgString = serializer.serializeToString(svgNode);
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();

            img.onload = function() {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                const pngUrl = canvas.toDataURL('image/png');
                const downloadLink = document.createElement('a');
                downloadLink.href = pngUrl;
                downloadLink.download = 'lollipop.png';
                downloadLink.click();
            };

            img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgString)));
        }
    }
}

// Make available globally
window.LollipopChart = LollipopChart;
