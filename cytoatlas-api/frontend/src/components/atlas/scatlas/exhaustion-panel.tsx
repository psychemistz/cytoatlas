import { useState, useMemo } from 'react';
import type { ExhaustionDiff } from '@/api/types/activity';
import { useScatlasExhaustion } from '@/api/hooks/use-scatlas';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { VolcanoChart } from '@/components/charts/volcano-chart';
import { BarChart } from '@/components/charts/bar-chart';

interface ExhaustionPanelProps {
  signatureType: string;
}

export default function ExhaustionPanel({
  signatureType,
}: ExhaustionPanelProps) {
  const [cancerType, setCancerType] = useState('All');

  const { data, isLoading, error } = useScatlasExhaustion(signatureType);

  const cancerTypes = useMemo(() => {
    if (!data) return [];
    return [...new Set(data.map((d) => d.cancer_type))].sort();
  }, [data]);

  const cancerOptions = useMemo(
    () => [
      { value: 'All', label: 'All Cancer Types' },
      ...cancerTypes.map((ct) => ({ value: ct, label: ct })),
    ],
    [cancerTypes],
  );

  const filtered = useMemo((): ExhaustionDiff[] => {
    if (!data) return [];
    if (cancerType === 'All') return data;
    return data.filter((d) => d.cancer_type === cancerType);
  }, [data, cancerType]);

  const volcanoPoints = useMemo(() => {
    return filtered.map((d) => ({
      signature: d.signature,
      activity_diff: d.activity_diff,
      p_value: d.p_value,
      fdr: d.fdr,
    }));
  }, [filtered]);

  const barData = useMemo(() => {
    if (!filtered.length) return null;

    const scored = filtered.map((d) => ({
      signature: d.signature,
      diff: d.activity_diff,
      score:
        Math.abs(d.activity_diff) *
        -Math.log10(Math.max(d.p_value, 1e-300)),
    }));

    const sorted = [...scored].sort((a, b) => b.score - a.score).slice(0, 20);
    return {
      categories: sorted.map((d) => d.signature),
      values: sorted.map((d) => d.diff),
    };
  }, [filtered]);

  if (isLoading)
    return <Spinner message="Loading exhaustion analysis data..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load exhaustion data: {(error as Error).message}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <p className="py-8 text-center text-text-muted">
        No exhaustion analysis data available
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <FilterBar>
        <SelectFilter
          label="Cancer Type"
          options={cancerOptions}
          value={cancerType}
          onChange={setCancerType}
        />
      </FilterBar>

      {volcanoPoints.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Exhausted vs Non-Exhausted T Cells -- Volcano Plot
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Positive {'\u0394'} Activity = higher in exhausted T cells
          </p>
          <VolcanoChart
            points={volcanoPoints}
            title={`T Cell Exhaustion: ${cancerType === 'All' ? 'All Cancer Types' : cancerType}`}
            fdrThreshold={0.05}
            activityThreshold={0.5}
          />
        </div>
      )}

      {barData && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Top 20 Exhaustion-Associated Signatures
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Ranked by significance score (|{'\u0394'} Activity| x -log10(p-value))
          </p>
          <BarChart
            categories={barData.categories}
            values={barData.values}
            orientation="h"
            xTitle={'\u0394 Activity (Exhausted - Non-Exhausted)'}
            yTitle="Signature"
            title="Top 20 Exhaustion-Associated Signatures"
            height={Math.max(500, barData.categories.length * 24 + 150)}
          />
        </div>
      )}
    </div>
  );
}
