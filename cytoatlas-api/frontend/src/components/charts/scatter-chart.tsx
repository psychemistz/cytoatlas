import { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js-dist-min';
import { PlotlyChart } from './plotly-chart';
import { COLORS, title as t } from './chart-defaults';

const GROUP_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
];

interface ScatterChartProps {
  x: number[];
  y: number[];
  labels?: string[];
  text?: string[];
  /** When provided (parallel to x/y), renders one trace per unique group with distinct colors */
  groups?: string[];
  colors?: string | string[];
  sizes?: number | number[];
  xTitle?: string;
  yTitle?: string;
  title?: string;
  showTrendLine?: boolean;
  stats?: { r?: number; p?: number; rho?: number; spearman_r?: number; pearson_r?: number; p_value?: number; fisherpval?: number; n?: number };
  hoverTemplate?: string;
  className?: string;
  height?: number;
  /** Use WebGL for large datasets (>1000 points) */
  useWebGL?: boolean;
}

export function ScatterChart({
  x,
  y,
  labels,
  text: textProp,
  groups: groupsProp,
  colors = COLORS.primary,
  sizes = 8,
  xTitle,
  yTitle,
  title,
  showTrendLine = false,
  stats: rawStats,
  hoverTemplate,
  className,
  height = 500,
  useWebGL: useWebGLProp,
}: ScatterChartProps) {
  const { data, layout } = useMemo(() => {
    const resolvedLabels = labels ?? textProp;
    const autoWebGL = useWebGLProp ?? x.length > 1000;
    const scatterType = autoWebGL ? 'scattergl' : 'scatter';

    // Normalize stats aliases
    const stats = rawStats
      ? {
          rho: rawStats.rho ?? rawStats.spearman_r,
          r: rawStats.r ?? rawStats.pearson_r,
          p: rawStats.p ?? rawStats.p_value ?? rawStats.fisherpval,
          n: rawStats.n,
        }
      : undefined;

    const traces: Data[] = [];

    if (groupsProp && groupsProp.length === x.length) {
      // Per-group coloring: one trace per unique group
      const groupMap = new Map<string, number[]>();
      groupsProp.forEach((g, i) => {
        const key = g || 'Other';
        if (!groupMap.has(key)) groupMap.set(key, []);
        groupMap.get(key)!.push(i);
      });
      let colorIdx = 0;
      for (const [groupName, indices] of groupMap) {
        traces.push({
          type: scatterType as 'scatter',
          mode: 'markers',
          name: groupName.replace(/_/g, ' '),
          x: indices.map((i) => x[i]),
          y: indices.map((i) => y[i]),
          marker: {
            size: typeof sizes === 'number' ? Math.max(4, sizes - 2) : sizes,
            color: GROUP_COLORS[colorIdx % GROUP_COLORS.length],
            opacity: 0.5,
          },
          hovertemplate: `${groupName}<br>Expr: %{x:.2f}<br>Activity: %{y:.2f}<extra></extra>`,
        });
        colorIdx++;
      }
    } else {
      // Single trace
      traces.push({
        type: scatterType as 'scatter',
        mode: 'markers',
        x,
        y,
        text: resolvedLabels,
        marker: {
          color: colors,
          size: sizes,
          opacity: 0.7,
          line: { color: 'white', width: 1 },
        },
        hovertemplate: hoverTemplate ?? '%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>',
      });
    }

    if (showTrendLine && x.length >= 2) {
      const n = x.length;
      const sumX = x.reduce((a, b) => a + b, 0);
      const sumY = y.reduce((a, b) => a + b, 0);
      const sumXY = x.reduce((a, xi, i) => a + xi * y[i], 0);
      const sumX2 = x.reduce((a, xi) => a + xi * xi, 0);
      const denom = n * sumX2 - sumX * sumX;
      if (Math.abs(denom) > 1e-10) {
        const slope = (n * sumXY - sumX * sumY) / denom;
        const intercept = (sumY - slope * sumX) / n;
        const xMin = Math.min(...x);
        const xMax = Math.max(...x);
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: [xMin, xMax],
          y: [slope * xMin + intercept, slope * xMax + intercept],
          line: { color: COLORS.red, width: 2, dash: 'dash' },
          showlegend: false,
          hoverinfo: 'skip',
        });
      }
    }

    const annoLines: string[] = [];
    if (stats) {
      if (stats.rho !== undefined) annoLines.push(`\u03C1 = ${stats.rho.toFixed(3)}`);
      else if (stats.r !== undefined) annoLines.push(`r = ${stats.r.toFixed(3)}`);
      if (stats.p !== undefined) annoLines.push(`p = ${stats.p.toExponential(2)}`);
      if (stats.n !== undefined) annoLines.push(`n = ${stats.n.toLocaleString()}`);
    }

    const annotations = annoLines.length > 0
      ? [
          {
            x: 0.02,
            y: 0.98,
            xref: 'paper' as const,
            yref: 'paper' as const,
            text: annoLines.join('<br>'),
            showarrow: false,
            xanchor: 'left' as const,
            yanchor: 'top' as const,
            font: { size: 12 },
            bgcolor: 'white',
            bordercolor: COLORS.zeroline,
            borderwidth: 1,
            borderpad: 4,
          },
        ]
      : undefined;

    const chartLayout: Partial<Layout> = {
      title: title ? { text: title, font: { size: 14 } } : undefined,
      xaxis: {
        title: t(xTitle),
        gridcolor: COLORS.gridline,
        zerolinecolor: COLORS.zeroline,
      },
      yaxis: {
        title: t(yTitle),
        gridcolor: COLORS.gridline,
        zerolinecolor: COLORS.zeroline,
      },
      height,
      annotations,
      legend: groupsProp ? { orientation: 'v' as const, x: 1.02, y: 1, font: { size: 9 } } : undefined,
    };

    return { data: traces, layout: chartLayout };
  }, [x, y, labels, textProp, groupsProp, colors, sizes, xTitle, yTitle, title, showTrendLine, rawStats, hoverTemplate, height, useWebGLProp]);

  return <PlotlyChart data={data} layout={layout} className={className} />;
}
