import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import type { Data, Layout, Config } from 'plotly.js-dist-min';
import { PLOTLY_LAYOUT, PLOTLY_CONFIG } from './chart-defaults';

interface PlotlyChartProps {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  className?: string;
  style?: React.CSSProperties;
  id?: string;
  onClick?: (event: Plotly.PlotMouseEvent) => void;
  onHover?: (event: Plotly.PlotMouseEvent) => void;
}

// Plotly.PlotMouseEvent type stub
declare namespace Plotly {
  interface PlotMouseEvent {
    points: { pointIndex: number; data: Data; x: number; y: number; text: string }[];
  }
}

export function PlotlyChart({
  data,
  layout: layoutOverride,
  config: configOverride,
  className = '',
  style,
  id,
  onClick,
  onHover,
}: PlotlyChartProps) {
  const mergedLayout = useMemo(
    () => ({
      ...PLOTLY_LAYOUT,
      ...layoutOverride,
      xaxis: { ...PLOTLY_LAYOUT.xaxis, ...layoutOverride?.xaxis },
      yaxis: { ...PLOTLY_LAYOUT.yaxis, ...layoutOverride?.yaxis },
    }),
    [layoutOverride],
  );

  const mergedConfig = useMemo(
    () => ({
      ...PLOTLY_CONFIG,
      ...configOverride,
    }),
    [configOverride],
  );

  return (
    <div className={`w-full ${className}`} style={style}>
      <Plot
        divId={id}
        data={data}
        layout={mergedLayout as Layout}
        config={mergedConfig as Config}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
        onClick={onClick as never}
        onHover={onHover as never}
      />
    </div>
  );
}
