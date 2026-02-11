import { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js-dist-min';
import { PlotlyChart } from './plotly-chart';
import { HEATMAP_COLORSCALE, title as t } from './chart-defaults';

interface HeatmapChartProps {
  z: number[][];
  x?: string[];
  y?: string[];
  signatures?: string[];
  cell_types?: string[];
  title?: string;
  xTitle?: string;
  yTitle?: string;
  xLabel?: string;
  yLabel?: string;
  colorscale?: [number, string][];
  symmetric?: boolean;
  zmin?: number;
  zmax?: number;
  colorbarTitle?: string;
  className?: string;
  height?: number;
}

export function HeatmapChart({
  z,
  x: xProp,
  y: yProp,
  signatures,
  cell_types,
  title,
  xTitle,
  yTitle,
  xLabel = 'X',
  yLabel = 'Y',
  colorscale = HEATMAP_COLORSCALE,
  symmetric = true,
  zmin: zminProp,
  zmax: zmaxProp,
  colorbarTitle = 'Value',
  className,
  height,
}: HeatmapChartProps) {
  const { data, layout } = useMemo(() => {
    const x = xProp ?? signatures ?? [];
    const y = yProp ?? cell_types ?? [];
    let zmin = zminProp;
    let zmax = zmaxProp;

    if (symmetric && zmin === undefined && zmax === undefined) {
      const flat = z.flat();
      const absMax = Math.max(...flat.map(Math.abs));
      zmin = -absMax;
      zmax = absMax;
    }

    const dynamicHeight = height ?? Math.max(400, y.length * 22 + 200);

    const traces: Data[] = [
      {
        type: 'heatmap',
        z,
        x,
        y,
        colorscale,
        zmin,
        zmax,
        hoverongaps: false,
        hovertemplate: `${xLabel}: %{x}<br>${yLabel}: %{y}<br>Value: %{z:.3f}<extra></extra>`,
        colorbar: {
          title: { text: colorbarTitle, side: 'right' },
          len: 0.9,
        },
      },
    ];

    const chartLayout: Partial<Layout> = {
      title: title ? { text: title, font: { size: 14 } } : undefined,
      margin: { l: 150, r: 50, t: title ? 60 : 30, b: 150 },
      xaxis: {
        title: t(xTitle),
        tickangle: -45,
        tickfont: { size: 10 },
      },
      yaxis: {
        title: t(yTitle),
        tickangle: 0,
        tickfont: { size: 10 },
      },
      height: dynamicHeight,
    };

    return { data: traces, layout: chartLayout };
  }, [z, xProp, yProp, signatures, cell_types, title, xTitle, yTitle, xLabel, yLabel, colorscale, symmetric, zminProp, zmaxProp, colorbarTitle, height]);

  return <PlotlyChart data={data} layout={layout} className={className} />;
}
