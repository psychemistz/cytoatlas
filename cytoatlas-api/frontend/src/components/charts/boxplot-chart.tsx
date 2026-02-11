import { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js-dist-min';
import { PlotlyChart } from './plotly-chart';
import { COLORS, title as t } from './chart-defaults';

interface BoxplotChartProps {
  groups: string[];
  values: number[][];
  title?: string;
  xTitle?: string;
  yTitle?: string;
  orientation?: 'v' | 'h';
  showPoints?: boolean;
  colors?: string[];
  className?: string;
  height?: number;
}

const DEFAULT_COLORS = [COLORS.primary, COLORS.green, COLORS.amber, COLORS.purple, COLORS.red];

export function BoxplotChart({
  groups,
  values,
  title,
  xTitle,
  yTitle,
  orientation = 'v',
  showPoints = true,
  colors = DEFAULT_COLORS,
  className,
  height = 500,
}: BoxplotChartProps) {
  const { data, layout } = useMemo(() => {
    const traces: Data[] = groups.map((group, i) => ({
      type: 'box' as const,
      name: group,
      ...(orientation === 'v' ? { y: values[i] } : { x: values[i] }),
      boxpoints: showPoints ? ('all' as const) : (false as const),
      jitter: 0.3,
      pointpos: -1.8,
      marker: { color: colors[i % colors.length], opacity: 0.5, size: 4 },
      line: { color: colors[i % colors.length] },
    }));

    const chartLayout: Partial<Layout> = {
      title: title ? { text: title, font: { size: 14 } } : undefined,
      xaxis: { title: orientation === 'v' ? undefined : t(xTitle) },
      yaxis: { title: orientation === 'v' ? t(yTitle) : undefined },
      showlegend: false,
      height,
    };

    return { data: traces, layout: chartLayout };
  }, [groups, values, title, xTitle, yTitle, orientation, showPoints, colors, height]);

  return <PlotlyChart data={data} layout={layout} className={className} />;
}
