import { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js-dist-min';
import { PlotlyChart } from './plotly-chart';

interface SankeyNode {
  name: string;
  category?: string;
  color?: string;
}

interface SankeyLink {
  source: number;
  target: number;
  value: number;
  label?: string;
}

interface SankeyChartProps {
  nodes: SankeyNode[];
  links: SankeyLink[];
  title?: string;
  className?: string;
  height?: number;
}

const CATEGORY_COLORS: Record<string, string> = {
  atlas: '#3b82f6',
  lineage: '#10b981',
  cell_type: '#f59e0b',
};

function toRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

export function SankeyChart({ nodes, links, title, className, height = 600 }: SankeyChartProps) {
  const { data, layout } = useMemo(() => {
    const nodeColors = nodes.map(
      (n) => n.color ?? CATEGORY_COLORS[n.category ?? ''] ?? '#6b7280',
    );

    const linkColors = links.map((l) => toRgba(nodeColors[l.source] ?? '#6b7280', 0.4));

    const traces: Data[] = [
      {
        type: 'sankey',
        orientation: 'h',
        node: {
          pad: 15,
          thickness: 20,
          line: { color: 'white', width: 1 },
          label: nodes.map((n) => n.name),
          color: nodeColors,
          hovertemplate: '%{label}<br>Total: %{value:,.0f}<extra></extra>',
        },
        link: {
          source: links.map((l) => l.source),
          target: links.map((l) => l.target),
          value: links.map((l) => l.value),
          color: linkColors,
          hovertemplate: '%{source.label} \u2192 %{target.label}<br>%{value:,.0f}<extra></extra>',
        },
      } as Data,
    ];

    const chartLayout: Partial<Layout> = {
      title: title ? { text: title, font: { size: 14 } } : undefined,
      margin: { l: 20, r: 20, t: title ? 60 : 20, b: 20 },
      height,
    };

    return { data: traces, layout: chartLayout };
  }, [nodes, links, title, height]);

  return <PlotlyChart data={data} layout={layout} config={{ displayModeBar: false }} className={className} />;
}
