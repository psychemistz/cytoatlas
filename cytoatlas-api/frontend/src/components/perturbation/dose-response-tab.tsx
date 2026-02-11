import { useState, useMemo } from 'react';
import type { Data, Layout } from 'plotly.js-dist-min';
import { useDrugList, useDoseResponse } from '@/api/hooks/use-perturbation';
import { PlotlyChart } from '@/components/charts/plotly-chart';
import { COLORS, title as t } from '@/components/charts/chart-defaults';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';

// Palette for distinguishing cell lines
const LINE_COLORS = [
  COLORS.primary,
  COLORS.red,
  COLORS.green,
  COLORS.amber,
  COLORS.purple,
  COLORS.darkSlate,
  '#0891b2',
  '#be185d',
  '#4338ca',
  '#15803d',
];

export default function DoseResponseTab() {
  const { data: drugs, isLoading: drugsLoading } = useDrugList();
  const [selectedDrug, setSelectedDrug] = useState<string>('');
  const { data: doseData, isLoading: doseLoading, error } = useDoseResponse(
    selectedDrug || undefined,
  );

  // Auto-select first drug when list loads
  useMemo(() => {
    if (drugs && drugs.length > 0 && !selectedDrug) {
      setSelectedDrug(drugs[0]);
    }
  }, [drugs, selectedDrug]);

  const { traces, layout } = useMemo(() => {
    if (!doseData || doseData.length === 0) {
      return { traces: [] as Data[], layout: {} as Partial<Layout> };
    }

    // Group by cell line
    const byLine = new Map<string, { dose: number; activity_diff: number }[]>();
    for (const d of doseData) {
      const arr = byLine.get(d.cell_line) ?? [];
      arr.push({ dose: d.dose, activity_diff: d.activity_diff });
      byLine.set(d.cell_line, arr);
    }

    const plotTraces: Data[] = [...byLine.entries()].map(([cellLine, points], i) => {
      const sorted = [...points].sort((a, b) => a.dose - b.dose);
      return {
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        name: cellLine,
        x: sorted.map((p) => p.dose),
        y: sorted.map((p) => p.activity_diff),
        line: { color: LINE_COLORS[i % LINE_COLORS.length], width: 2 },
        marker: { size: 6 },
        hovertemplate: `${cellLine}<br>Dose: %{x}<br>Delta Activity: %{y:.3f}<extra></extra>`,
      };
    });

    const chartLayout: Partial<Layout> = {
      title: selectedDrug
        ? { text: `Dose-Response: ${selectedDrug}`, font: { size: 14 } }
        : undefined,
      xaxis: {
        title: t('Dose'),
        type: 'log',
      },
      yaxis: {
        title: t('Delta Activity'),
      },
      height: 500,
      showlegend: true,
      legend: { orientation: 'h' as const, y: -0.2 },
    };

    return { traces: plotTraces, layout: chartLayout };
  }, [doseData, selectedDrug]);

  const drugOptions = (drugs || []).map((d) => ({ value: d, label: d }));

  if (drugsLoading) return <Spinner message="Loading drug list..." />;

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold">Dose-Response Curves</h3>
        <p className="text-sm text-text-secondary">
          Activity change as a function of drug concentration across cell lines (Tahoe Plate 13)
        </p>
      </div>

      {drugOptions.length > 0 && (
        <FilterBar>
          <SelectFilter
            label="Drug"
            options={drugOptions}
            value={selectedDrug}
            onChange={setSelectedDrug}
          />
        </FilterBar>
      )}

      {doseLoading && <Spinner message="Loading dose-response data..." />}

      {error && (
        <div className="rounded-lg border border-danger/20 bg-danger/5 p-6 text-center">
          <p className="text-sm text-danger">Failed to load dose-response data</p>
        </div>
      )}

      {!doseLoading && !error && traces.length === 0 && selectedDrug && (
        <div className="rounded-lg border border-border-light bg-bg-secondary p-8 text-center text-text-muted">
          No dose-response data available for {selectedDrug}
        </div>
      )}

      {traces.length > 0 && <PlotlyChart data={traces} layout={layout} />}
    </div>
  );
}
