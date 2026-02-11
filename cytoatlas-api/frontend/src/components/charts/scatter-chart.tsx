import { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js-dist-min';
import { PlotlyChart } from './plotly-chart';
import { COLORS, title as t } from './chart-defaults';

interface ScatterChartProps {
  x: number[];
  y: number[];
  labels?: string[];
  colors?: string | string[];
  sizes?: number | number[];
  xTitle?: string;
  yTitle?: string;
  title?: string;
  showTrendLine?: boolean;
  stats?: { r?: number; p?: number; rho?: number };
  className?: string;
  height?: number;
}

export function ScatterChart({
  x,
  y,
  labels,
  colors = COLORS.primary,
  sizes = 8,
  xTitle,
  yTitle,
  title,
  showTrendLine = false,
  stats,
  className,
  height = 500,
}: ScatterChartProps) {
  const { data, layout } = useMemo(() => {
    const traces: Data[] = [
      {
        type: 'scatter',
        mode: 'markers',
        x,
        y,
        text: labels,
        marker: {
          color: colors,
          size: sizes,
          opacity: 0.7,
          line: { color: 'white', width: 1 },
        },
        hovertemplate: '%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>',
      },
    ];

    if (showTrendLine && x.length >= 2) {
      const n = x.length;
      const sumX = x.reduce((a, b) => a + b, 0);
      const sumY = y.reduce((a, b) => a + b, 0);
      const sumXY = x.reduce((a, xi, i) => a + xi * y[i], 0);
      const sumX2 = x.reduce((a, xi) => a + xi * xi, 0);
      const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
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

    const annotations = stats
      ? [
          {
            x: 0.02,
            y: 0.98,
            xref: 'paper' as const,
            yref: 'paper' as const,
            text: [
              stats.rho !== undefined ? `\u03C1 = ${stats.rho.toFixed(3)}` : stats.r !== undefined ? `r = ${stats.r.toFixed(3)}` : '',
              stats.p !== undefined ? `p = ${stats.p.toExponential(2)}` : '',
            ]
              .filter(Boolean)
              .join('<br>'),
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
    };

    return { data: traces, layout: chartLayout };
  }, [x, y, labels, colors, sizes, xTitle, yTitle, title, showTrendLine, stats, height]);

  return <PlotlyChart data={data} layout={layout} className={className} />;
}
