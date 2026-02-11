import { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js-dist-min';
import { useTechnologyComparison } from '@/api/hooks/use-spatial';
import { PlotlyChart } from '@/components/charts/plotly-chart';
import { COLORS } from '@/components/charts/chart-defaults';
import { Spinner } from '@/components/ui/loading-skeleton';

interface TechComparisonTabProps {
  signatureType: string;
}

const PAIR_COLORS = [
  COLORS.primary,
  COLORS.green,
  COLORS.amber,
  COLORS.purple,
  COLORS.red,
  COLORS.gray,
  COLORS.darkSlate,
];

export default function TechComparisonTab({ signatureType }: TechComparisonTabProps) {
  const { data, isLoading, error } = useTechnologyComparison(signatureType);

  const { traces, layout } = useMemo(() => {
    if (!data || data.length === 0) return { traces: [] as Data[], layout: {} as Partial<Layout> };

    // Group by technology pair
    const groups = new Map<string, { x: number[]; y: number[]; labels: string[] }>();
    for (const row of data) {
      const pairKey = `${row.technology_1} vs ${row.technology_2}`;
      const group = groups.get(pairKey) ?? { x: [], y: [], labels: [] };
      group.x.push(row.activity_tech1);
      group.y.push(row.activity_tech2);
      group.labels.push(row.tissue);
      groups.set(pairKey, group);
    }

    const plotTraces: Data[] = Array.from(groups.entries()).map(
      ([pairName, group], i) => ({
        type: 'scatter' as const,
        mode: 'markers' as const,
        name: pairName,
        x: group.x,
        y: group.y,
        text: group.labels,
        marker: {
          color: PAIR_COLORS[i % PAIR_COLORS.length],
          size: 8,
          opacity: 0.7,
          line: { color: 'white', width: 1 },
        },
        hovertemplate: '%{text}<br>Tech 1: %{x:.3f}<br>Tech 2: %{y:.3f}<extra>%{fullData.name}</extra>',
      }),
    );

    // Add identity line
    const allX = data.map((d) => d.activity_tech1);
    const allY = data.map((d) => d.activity_tech2);
    const allVals = [...allX, ...allY];
    const minVal = Math.min(...allVals);
    const maxVal = Math.max(...allVals);

    plotTraces.push({
      type: 'scatter' as const,
      mode: 'lines' as const,
      x: [minVal, maxVal],
      y: [minVal, maxVal],
      line: { color: COLORS.gray, width: 1, dash: 'dash' },
      showlegend: false,
      hoverinfo: 'skip' as const,
    });

    const plotLayout: Partial<Layout> = {
      xaxis: {
        title: { text: 'Technology 1 Activity' },
      },
      yaxis: {
        title: { text: 'Technology 2 Activity' },
      },
      height: 550,
      showlegend: true,
      legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.8)' },
    };

    return { traces: plotTraces, layout: plotLayout };
  }, [data]);

  if (isLoading) return <Spinner message="Loading technology comparison..." />;
  if (error) {
    return (
      <p className="py-8 text-center text-red-600">
        Failed to load technology comparison: {(error as Error).message}
      </p>
    );
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="mb-2 text-sm font-semibold text-text-secondary">
          Cross-Technology Activity Comparison
        </h3>
        <p className="mb-3 text-xs text-text-muted">
          Scatter plot comparing activity z-scores between technology pairs across
          shared tissues. Points near the diagonal indicate high cross-technology
          agreement. Signatures from the {signatureType} matrix.
        </p>
      </div>

      {traces.length > 0 ? (
        <PlotlyChart data={traces} layout={layout} />
      ) : (
        <p className="py-8 text-center text-text-muted">No technology comparison data available</p>
      )}
    </div>
  );
}
