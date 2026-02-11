import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { COLORS } from './chart-defaults';

interface LollipopChartProps {
  categories: string[];
  values: number[];
  title?: string;
  xTitle?: string;
  color?: string;
  circleRadius?: number;
  className?: string;
}

export function LollipopChart({
  categories,
  values,
  title,
  xTitle,
  color = COLORS.primary,
  circleRadius = 5,
  className,
}: LollipopChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !categories.length) return;

    const margin = { top: 40, right: 40, bottom: 80, left: 150 };
    const containerWidth = svgRef.current.parentElement?.clientWidth ?? 600;
    const width = containerWidth - margin.left - margin.right;
    const height = Math.max(400, categories.length * 25);

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    svg.attr('width', containerWidth).attr('height', height + margin.top + margin.bottom);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear().domain([0, Math.max(...values) * 1.1]).range([0, width]);

    const y = d3
      .scaleBand()
      .domain(categories)
      .range([0, height])
      .padding(0.2);

    // X axis
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x).ticks(6))
      .selectAll('text')
      .style('font-family', 'Inter, system-ui, sans-serif')
      .style('font-size', '11px');

    // Y axis
    g.append('g')
      .call(d3.axisLeft(y))
      .selectAll('text')
      .style('font-family', 'Inter, system-ui, sans-serif')
      .style('font-size', '11px');

    // Lines
    g.selectAll('.lollipop-line')
      .data(categories)
      .join('line')
      .attr('x1', 0)
      .attr('x2', (_, i) => x(values[i]))
      .attr('y1', (d) => (y(d) ?? 0) + y.bandwidth() / 2)
      .attr('y2', (d) => (y(d) ?? 0) + y.bandwidth() / 2)
      .attr('stroke', color)
      .attr('stroke-width', 2);

    // Circles
    g.selectAll('.lollipop-circle')
      .data(categories)
      .join('circle')
      .attr('cx', (_, i) => x(values[i]))
      .attr('cy', (d) => (y(d) ?? 0) + y.bandwidth() / 2)
      .attr('r', circleRadius)
      .attr('fill', color)
      .attr('stroke', 'white')
      .attr('stroke-width', 2);

    // Title
    if (title) {
      svg
        .append('text')
        .attr('x', containerWidth / 2)
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .style('font-family', 'Inter, system-ui, sans-serif')
        .style('font-size', '14px')
        .style('font-weight', '600')
        .text(title);
    }

    // X-axis title
    if (xTitle) {
      svg
        .append('text')
        .attr('x', margin.left + width / 2)
        .attr('y', height + margin.top + margin.bottom - 10)
        .attr('text-anchor', 'middle')
        .style('font-family', 'Inter, system-ui, sans-serif')
        .style('font-size', '12px')
        .text(xTitle);
    }
  }, [categories, values, title, xTitle, color, circleRadius]);

  return (
    <div className={className}>
      <svg ref={svgRef} />
    </div>
  );
}
