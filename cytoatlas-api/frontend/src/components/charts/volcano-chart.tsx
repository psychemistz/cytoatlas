import { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js-dist-min';
import { PlotlyChart } from './plotly-chart';
import { COLORS, title as t } from './chart-defaults';

interface VolcanoPoint {
  signature?: string;
  label?: string;
  gene?: string;
  activity_diff?: number;
  log2fc?: number;
  x?: number;
  p_value?: number;
  pval?: number;
  y?: number;
  fdr?: number;
}

interface VolcanoChartProps {
  points: VolcanoPoint[];
  title?: string;
  fdrThreshold?: number;
  activityThreshold?: number;
  className?: string;
  height?: number;
  onClick?: (signature: string) => void;
  leftLabel?: string;
  rightLabel?: string;
  maxPoints?: number;
}

export function VolcanoChart({
  points,
  title,
  fdrThreshold = 0.05,
  activityThreshold = 0.5,
  className,
  height = 500,
  onClick,
  leftLabel = '← Lower',
  rightLabel = 'Higher →',
  maxPoints,
}: VolcanoChartProps) {
  // Truncate to top N by significance score when maxPoints is set (e.g. SecAct 1,249 → 200)
  const filteredPoints = useMemo(() => {
    if (!maxPoints || points.length <= maxPoints) return points;
    return [...points]
      .map((p) => ({
        ...p,
        _score: Math.abs(p.activity_diff ?? p.log2fc ?? p.x ?? 0) *
          -Math.log10(Math.max(p.p_value ?? p.pval ?? p.y ?? 1, 1e-300)),
      }))
      .sort((a, b) => b._score - a._score)
      .slice(0, maxPoints);
  }, [points, maxPoints]);

  const { data, layout } = useMemo(() => {
    const x = filteredPoints.map((p) => p.activity_diff ?? p.log2fc ?? p.x ?? 0);
    const y = filteredPoints.map((p) => {
      const pval = p.p_value ?? p.pval ?? p.y ?? 1;
      return pval > 0 ? -Math.log10(pval) : 0;
    });
    const text = filteredPoints.map((p) => p.signature ?? p.label ?? p.gene ?? '');

    const colors = filteredPoints.map((p) => {
      const actDiff = p.activity_diff ?? p.log2fc ?? p.x ?? 0;
      const fdr = p.fdr ?? p.p_value ?? p.pval ?? 1;
      if (fdr < fdrThreshold && Math.abs(actDiff) > activityThreshold) {
        return actDiff > 0 ? COLORS.red : COLORS.green;
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
        marker: { color: colors, size: 10, opacity: 0.7, line: { color: 'white', width: 0.5 } },
        hovertemplate: '%{text}<br>\u0394 Activity: %{x:.3f}<br>-log10(p): %{y:.3f}<extra></extra>',
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

    // Dynamic symmetric x-axis range
    const maxAbsFC = Math.ceil(Math.max(...x.map(Math.abs), 1));

    const annotations = [
      {
        x: -activityThreshold * 1.5,
        y: 1.02,
        yref: 'paper' as const,
        xref: 'x' as const,
        text: leftLabel,
        showarrow: false,
        font: { size: 10, color: '#6b7280' },
      },
      {
        x: activityThreshold * 1.5,
        y: 1.02,
        yref: 'paper' as const,
        xref: 'x' as const,
        text: rightLabel,
        showarrow: false,
        font: { size: 10, color: '#6b7280' },
      },
    ];

    const chartLayout: Partial<Layout> = {
      title: title ? { text: title, font: { size: 14 } } : undefined,
      xaxis: { title: t('\u0394 Activity'), gridcolor: COLORS.gridline, zerolinecolor: COLORS.zeroline, range: [-maxAbsFC, maxAbsFC] },
      yaxis: { title: t('-log10(p-value)'), gridcolor: COLORS.gridline, zeroline: false },
      shapes,
      annotations,
      height,
    };

    return { data: traces, layout: chartLayout };
  }, [filteredPoints, title, fdrThreshold, activityThreshold, height, leftLabel, rightLabel]);

  return (
    <PlotlyChart
      data={data}
      layout={layout}
      className={className}
      onClick={onClick ? (e) => {
        const idx = (e as { points: { pointIndex: number }[] }).points[0]?.pointIndex;
        if (idx !== undefined) onClick(filteredPoints[idx].signature ?? filteredPoints[idx].label ?? filteredPoints[idx].gene ?? '');
      } : undefined}
    />
  );
}
