import { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js-dist-min';
import { PlotlyChart } from './plotly-chart';
import { COLORS, title as t } from './chart-defaults';

type AxisConfig = {
  title?: ReturnType<typeof t>;
  zeroline?: boolean;
  zerolinecolor?: string;
  gridcolor?: string;
};

interface ViolinChartProps {
  groups: string[];
  values: number[][];
  title?: string;
  xTitle?: string;
  yTitle?: string;
  orientation?: 'v' | 'h';
  showBox?: boolean;
  showMeanLine?: boolean;
  colors?: string[];
  className?: string;
  height?: number;
}

const DEFAULT_COLORS = [COLORS.primary, COLORS.green, COLORS.amber, COLORS.purple, COLORS.red];

export function ViolinChart({
  groups,
  values,
  title,
  xTitle,
  yTitle,
  orientation = 'v',
  showBox = true,
  showMeanLine = true,
  colors = DEFAULT_COLORS,
  className,
  height = 500,
}: ViolinChartProps) {
  const { data, layout } = useMemo(() => {
    const traces: Data[] = groups.map((group, i) => ({
      type: 'violin' as const,
      name: group,
      ...(orientation === 'v' ? { y: values[i] } : { x: values[i] }),
      box: { visible: showBox },
      meanline: { visible: showMeanLine },
      fillcolor: colors[i % colors.length],
      opacity: 0.6,
      line: { color: colors[i % colors.length] },
    }));

    const valueAxisConfig: AxisConfig = {
      zeroline: true,
      zerolinecolor: COLORS.zeroline,
      gridcolor: COLORS.gridline,
    };

    const xaxis: AxisConfig = orientation === 'v'
      ? { title: t(xTitle) }
      : { title: t(yTitle), ...valueAxisConfig };

    const yaxis: AxisConfig = orientation === 'v'
      ? { title: t(yTitle), ...valueAxisConfig }
      : { title: t(xTitle) };

    const chartLayout: Partial<Layout> & { violinmode?: string } = {
      title: title ? { text: title, font: { size: 14 } } : undefined,
      xaxis,
      yaxis,
      violinmode: 'group',
      showlegend: false,
      height,
    };

    return { data: traces, layout: chartLayout };
  }, [groups, values, title, xTitle, yTitle, orientation, showBox, showMeanLine, colors, height]);

  return <PlotlyChart data={data} layout={layout} className={className} />;
}
