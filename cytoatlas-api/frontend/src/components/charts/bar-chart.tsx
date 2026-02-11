import { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js-dist-min';
import { PlotlyChart } from './plotly-chart';
import { COLORS, title as t } from './chart-defaults';

interface BarSeries {
  name: string;
  values: number[];
}

interface BarChartProps {
  categories: string[];
  values?: number[];
  series?: BarSeries[];
  title?: string;
  xTitle?: string;
  yTitle?: string;
  orientation?: 'v' | 'h';
  barmode?: 'group' | 'stack' | 'relative';
  colors?: string[];
  className?: string;
  height?: number;
}

const DEFAULT_COLORS = [COLORS.primary, COLORS.green, COLORS.amber, COLORS.purple, COLORS.red];

export function BarChart({
  categories,
  values,
  series,
  title,
  xTitle,
  yTitle,
  orientation = 'v',
  barmode = 'group',
  colors = DEFAULT_COLORS,
  className,
  height = 500,
}: BarChartProps) {
  const { data, layout } = useMemo(() => {
    let traces: Data[];

    if (series) {
      traces = series.map((s, i) => ({
        type: 'bar' as const,
        name: s.name,
        ...(orientation === 'v'
          ? { x: categories, y: s.values }
          : { y: categories, x: s.values }),
        marker: { color: colors[i % colors.length] },
        orientation,
      }));
    } else {
      traces = [
        {
          type: 'bar' as const,
          ...(orientation === 'v'
            ? { x: categories, y: values }
            : { y: categories, x: values }),
          marker: { color: colors[0] },
          orientation,
        },
      ];
    }

    const chartLayout: Partial<Layout> = {
      title: title ? { text: title, font: { size: 14 } } : undefined,
      barmode,
      xaxis: { title: t(xTitle) },
      yaxis: { title: t(yTitle) },
      showlegend: !!series && series.length > 1,
      height,
    };

    return { data: traces, layout: chartLayout };
  }, [categories, values, series, title, xTitle, yTitle, orientation, barmode, colors, height]);

  return <PlotlyChart data={data} layout={layout} className={className} />;
}
