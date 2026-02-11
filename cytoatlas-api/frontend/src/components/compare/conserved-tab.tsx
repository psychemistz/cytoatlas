import { useState, useMemo } from 'react';
import { useConservedSignatures, useConsistencyHeatmap } from '@/api/hooks/use-cross-atlas';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { BarChart } from '@/components/charts/bar-chart';
import { HeatmapChart } from '@/components/charts/heatmap-chart';

interface ConservedTabProps {
  signatureType: string;
}

export default function ConservedTab({ signatureType }: ConservedTabProps) {
  const [minAtlases, setMinAtlases] = useState(2);

  const { data: conserved, isLoading: consLoading } = useConservedSignatures(
    signatureType,
    minAtlases,
  );
  const { data: heatmap, isLoading: hmLoading } = useConsistencyHeatmap(signatureType);

  const barData = useMemo(() => {
    if (!conserved) return null;
    const sorted = [...conserved]
      .sort((a, b) => b.conservation_score - a.conservation_score)
      .slice(0, 30);
    return {
      categories: sorted.map((s) => s.signature),
      values: sorted.map((s) => s.conservation_score),
    };
  }, [conserved]);

  if (consLoading || hmLoading) return <Spinner message="Loading conserved signatures..." />;

  return (
    <div className="space-y-6">
      <FilterBar>
        <SelectFilter
          label="Min Atlases"
          options={[
            { value: '2', label: '2+ Atlases' },
            { value: '3', label: '3 Atlases (all)' },
          ]}
          value={String(minAtlases)}
          onChange={(v) => setMinAtlases(Number(v))}
        />
      </FilterBar>

      {barData && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Top 30 Conserved Signatures
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Signatures with consistent activity patterns across {minAtlases}+ atlases.
          </p>
          <BarChart
            categories={barData.categories}
            values={barData.values}
            orientation="h"
            xTitle="Conservation Score"
            yTitle="Signature"
            title="Conserved Signatures"
            height={Math.max(400, barData.categories.length * 24 + 150)}
          />
        </div>
      )}

      {heatmap && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Activity Consistency Heatmap
          </h3>
          <HeatmapChart
            z={heatmap.z}
            x={heatmap.x}
            y={heatmap.y}
            title="Signature Consistency Across Atlases"
            xTitle="Atlas"
            yTitle="Signature"
            colorbarTitle="Mean Activity"
            symmetric
          />
        </div>
      )}
    </div>
  );
}
