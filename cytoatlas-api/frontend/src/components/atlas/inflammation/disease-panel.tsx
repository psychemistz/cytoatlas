import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import type { ActivityData } from '@/api/types/activity';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { HeatmapChart } from '@/components/charts/heatmap-chart';
import { BarChart } from '@/components/charts/bar-chart';

interface DiseasePanelProps {
  signatureType: string;
}

export default function DiseasePanel({ signatureType }: DiseasePanelProps) {
  const [selectedDisease, setSelectedDisease] = useState<string>('');

  const { data, isLoading, error } = useQuery({
    queryKey: ['inflammation', 'disease-activity', signatureType],
    queryFn: () =>
      get<ActivityData[]>('/atlases/inflammation/disease-activity', {
        signature_type: signatureType,
      }),
  });

  const diseases = useMemo(() => {
    if (!data) return [];
    return [...new Set(data.map((d) => d.cell_type))].sort();
  }, [data]);

  // Auto-select first disease when data loads
  const activeDisease = selectedDisease || diseases[0] || '';

  const diseaseOptions = useMemo(() => {
    return diseases.map((d) => ({ value: d, label: d }));
  }, [diseases]);

  const heatmap = useMemo(() => {
    if (!data) return null;
    const diseaseList = [...new Set(data.map((d) => d.cell_type))].sort();
    const signatures = [...new Set(data.map((d) => d.signature))].sort();
    const lookup = new Map(
      data.map((d) => [`${d.cell_type}||${d.signature}`, d.mean_activity]),
    );
    const z = diseaseList.map((disease) =>
      signatures.map((sig) => lookup.get(`${disease}||${sig}`) ?? 0),
    );
    return { z, x: signatures, y: diseaseList };
  }, [data]);

  const barData = useMemo(() => {
    if (!data || !activeDisease) return null;
    const rows = data
      .filter((d) => d.cell_type === activeDisease)
      .sort((a, b) => b.mean_activity - a.mean_activity)
      .slice(0, 15);
    return {
      categories: rows.map((d) => d.signature),
      values: rows.map((d) => d.mean_activity),
    };
  }, [data, activeDisease]);

  if (isLoading) return <Spinner message="Loading disease activity data..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load disease activity data: {(error as Error).message}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <p className="py-8 text-center text-text-muted">
        No disease activity data available
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <FilterBar>
        <SelectFilter
          label="Disease"
          options={diseaseOptions}
          value={activeDisease}
          onChange={setSelectedDisease}
        />
      </FilterBar>

      {heatmap && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Disease x Signature Activity Matrix
          </h3>
          <HeatmapChart
            z={heatmap.z}
            x={heatmap.x}
            y={heatmap.y}
            title="Disease Activity Heatmap"
            xTitle="Signature"
            yTitle="Disease"
            colorbarTitle="Mean Activity"
            symmetric
          />
        </div>
      )}

      {barData && barData.categories.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Top 15 Signatures for {activeDisease}
          </h3>
          <BarChart
            categories={barData.categories}
            values={barData.values}
            orientation="h"
            xTitle="Mean Activity (z-score)"
            yTitle="Signature"
            title={`Top 15 Signatures: ${activeDisease}`}
            height={Math.max(400, barData.categories.length * 24 + 150)}
          />
        </div>
      )}
    </div>
  );
}
