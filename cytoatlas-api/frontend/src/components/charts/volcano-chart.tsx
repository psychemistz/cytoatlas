import { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js-dist-min';
import { PlotlyChart } from './plotly-chart';
import { COLORS, title as t } from './chart-defaults';

interface VolcanoPoint {
  signature: string;
  activity_diff: number;
  p_value: number;
  fdr: number;
}

interface VolcanoChartProps {
  points: VolcanoPoint[];
  title?: string;
  fdrThreshold?: number;
  activityThreshold?: number;
  className?: string;
  height?: number;
  onClick?: (signature: string) => void;
}

export function VolcanoChart({
  points,
  title,
  fdrThreshold = 0.05,
  activityThreshold = 0.5,
  className,
  height = 500,
  onClick,
}: VolcanoChartProps) {
  const { data, layout } = useMemo(() => {
    const x = points.map((p) => p.activity_diff);
    const y = points.map((p) => -Math.log10(Math.max(p.p_value, 1e-300)));
    const text = points.map((p) => p.signature);

    const colors = points.map((p) => {
      if (p.fdr < fdrThreshold && Math.abs(p.activity_diff) > activityThreshold) {
        return p.activity_diff > 0 ? COLORS.red : COLORS.primary;
      }
      return COLORS.gray;
    });

    const traces: Data[] = [
      {
        type: 'scatter',
        mode: 'markers',
        x,
        y,
        text,
        marker: { color: colors, size: 8, opacity: 0.7, line: { color: 'white', width: 1 } },
        hovertemplate: '%{text}<br>\u0394 Activity: %{x:.3f}<br>-log10(p): %{y:.2f}<extra></extra>',
      },
    ];

    const yThreshold = -Math.log10(fdrThreshold);
    const shapes = [
      // Vertical lines at ±activityThreshold
      { type: 'line' as const, x0: -activityThreshold, x1: -activityThreshold, y0: 0, y1: 1, yref: 'paper' as const, line: { dash: 'dash' as const, color: COLORS.gray, width: 1 } },
      { type: 'line' as const, x0: activityThreshold, x1: activityThreshold, y0: 0, y1: 1, yref: 'paper' as const, line: { dash: 'dash' as const, color: COLORS.gray, width: 1 } },
      // Horizontal line at -log10(fdrThreshold)
      { type: 'line' as const, x0: 0, x1: 1, xref: 'paper' as const, y0: yThreshold, y1: yThreshold, line: { dash: 'dash' as const, color: COLORS.gray, width: 1 } },
    ];

    const chartLayout: Partial<Layout> = {
      title: title ? { text: title, font: { size: 14 } } : undefined,
      xaxis: { title: t('\u0394 Activity'), gridcolor: COLORS.gridline, zerolinecolor: COLORS.zeroline },
      yaxis: { title: t('-log10(p-value)'), gridcolor: COLORS.gridline },
      shapes,
      height,
    };

    return { data: traces, layout: chartLayout };
  }, [points, title, fdrThreshold, activityThreshold, height]);

  return (
    <PlotlyChart
      data={data}
      layout={layout}
      className={className}
      onClick={onClick ? (e) => {
        const idx = (e as { points: { pointIndex: number }[] }).points[0]?.pointIndex;
        if (idx !== undefined) onClick(points[idx].signature);
      } : undefined}
    />
  );
}
