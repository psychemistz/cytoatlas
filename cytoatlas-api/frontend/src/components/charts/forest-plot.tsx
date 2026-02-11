import { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js-dist-min';
import { PlotlyChart } from './plotly-chart';
import { COLORS, title as t } from './chart-defaults';

interface IndividualEffect {
  atlas: string;
  effect: number;
  se: number;
  n: number;
}

interface ForestPlotItem {
  signature: string;
  individual_effects: IndividualEffect[];
  pooled_effect: number;
  ci_low: number;
  ci_high: number;
  I2: number;
}

interface ForestPlotProps {
  items: ForestPlotItem[];
  title?: string;
  className?: string;
  height?: number;
}

const ATLAS_COLORS: Record<string, string> = {
  CIMA: '#3b82f6',
  cima: '#3b82f6',
  Inflammation: '#f59e0b',
  inflammation: '#f59e0b',
  scAtlas: '#10b981',
  scatlas: '#10b981',
};

export function ForestPlot({ items, title, className, height }: ForestPlotProps) {
  const { data, layout } = useMemo(() => {
    const traces: Data[] = [];
    const tickvals: number[] = [];
    const ticktext: string[] = [];

    // Group traces by atlas for legend
    const atlasTraces: Record<string, { x: number[]; y: number[]; error_x: number[]; customdata: number[] }> = {};

    items.forEach((item, idx) => {
      const baseY = idx;
      tickvals.push(baseY);
      ticktext.push(item.signature);

      // Individual effects
      item.individual_effects.forEach((effect, aIdx) => {
        const atlasName = effect.atlas;
        if (!atlasTraces[atlasName]) {
          atlasTraces[atlasName] = { x: [], y: [], error_x: [], customdata: [] };
        }
        atlasTraces[atlasName].x.push(effect.effect);
        atlasTraces[atlasName].y.push(baseY + (aIdx - 1) * 0.15);
        atlasTraces[atlasName].error_x.push(effect.se * 1.96);
        atlasTraces[atlasName].customdata.push(effect.n);
      });
    });

    // Atlas effect traces
    Object.entries(atlasTraces).forEach(([atlas, d]) => {
      traces.push({
        type: 'scatter',
        mode: 'markers',
        name: atlas,
        x: d.x,
        y: d.y,
        customdata: d.customdata,
        error_x: {
          type: 'data',
          array: d.error_x,
          visible: true,
          color: ATLAS_COLORS[atlas] ?? COLORS.gray,
          width: 0,
          thickness: 1.5,
        },
        marker: { color: ATLAS_COLORS[atlas] ?? COLORS.gray, size: 8 },
        hovertemplate: `${atlas}: %{x:.3f} (n=%{customdata})<extra></extra>`,
      });
    });

    // Pooled effects (diamonds)
    const pooledCustomdata = items.map((i) => [i.ci_low.toFixed(3), i.ci_high.toFixed(3), i.I2.toFixed(1)]);
    traces.push({
      type: 'scatter',
      mode: 'markers',
      name: 'Pooled',
      x: items.map((i) => i.pooled_effect),
      y: items.map((_, idx) => idx),
      customdata: pooledCustomdata,
      error_x: {
        type: 'data',
        array: items.map((i) => i.ci_high - i.pooled_effect),
        arrayminus: items.map((i) => i.pooled_effect - i.ci_low),
        visible: true,
        color: COLORS.darkSlate,
        width: 4,
        thickness: 2,
      },
      marker: { color: COLORS.darkSlate, size: 12, symbol: 'diamond' },
      hovertemplate: 'Pooled: %{x:.3f} [%{customdata[0]}, %{customdata[1]}] I\u00b2: %{customdata[2]}%<extra></extra>',
    });

    const dynamicHeight = height ?? Math.max(400, items.length * 60 + 120);

    const chartLayout: Partial<Layout> = {
      title: title ? { text: title, font: { size: 14 } } : undefined,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      margin: { l: 120, r: 40, t: 60, b: 80 },
      xaxis: { title: t('Effect Size (Correlation)'), zeroline: true, zerolinecolor: COLORS.zeroline, gridcolor: COLORS.zeroline },
      yaxis: { tickmode: 'array' as const, tickvals, ticktext, autorange: 'reversed' as const, gridcolor: COLORS.zeroline },
      shapes: [
        { type: 'line', x0: 0, x1: 0, y0: 0, y1: 1, yref: 'paper', line: { dash: 'dash', color: COLORS.gray, width: 1 } },
      ],
      legend: { orientation: 'h' as const, y: -0.15, x: 0.5, xanchor: 'center' as const },
      height: dynamicHeight,
    };

    return { data: traces, layout: chartLayout };
  }, [items, title, height]);

  return <PlotlyChart data={data} layout={layout} className={className} />;
}
