import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import type { ActivityData } from '@/api/types/activity';
import { Spinner } from '@/components/ui/loading-skeleton';
import { SearchInput } from '@/components/ui/search-input';
import { BarChart } from '@/components/charts/bar-chart';
import { HeatmapChart } from '@/components/charts/heatmap-chart';

interface CelltypePanelProps {
  signatureType: string;
  atlasName: string;
}

export default function CelltypePanel({ signatureType, atlasName }: CelltypePanelProps) {
  const [search, setSearch] = useState('');
  const [selectedSignature, setSelectedSignature] = useState<string | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ['atlas-activity', atlasName, signatureType],
    queryFn: () =>
      get<ActivityData[]>(`/atlases/${atlasName}/activity`, { signature_type: signatureType }),
  });

  const signatures = useMemo(() => {
    if (!data) return [];
    return [...new Set(data.map((d) => d.signature))].sort();
  }, [data]);

  const filteredSignatures = useMemo(() => {
    if (!search) return signatures;
    const q = search.toLowerCase();
    return signatures.filter((s) => s.toLowerCase().includes(q));
  }, [signatures, search]);

  const barData = useMemo(() => {
    if (!data || !selectedSignature) return null;
    const rows = data
      .filter((d) => d.signature === selectedSignature)
      .sort((a, b) => b.mean_activity - a.mean_activity);
    const top15 = rows.slice(0, 15);
    const bottom15 = rows.slice(-15).reverse();
    const combined = [...top15, ...bottom15];
    const unique = Array.from(new Map(combined.map((d) => [d.cell_type, d])).values());
    return {
      categories: unique.map((d) => d.cell_type),
      values: unique.map((d) => d.mean_activity),
    };
  }, [data, selectedSignature]);

  const heatmap = useMemo(() => {
    if (!data) return null;
    const cellTypes = [...new Set(data.map((d) => d.cell_type))].sort();
    const sigs = [...new Set(data.map((d) => d.signature))].sort();
    const lookup = new Map(data.map((d) => [`${d.cell_type}||${d.signature}`, d.mean_activity]));
    const z = cellTypes.map((ct) => sigs.map((sig) => lookup.get(`${ct}||${sig}`) ?? 0));
    return { z, x: sigs, y: cellTypes };
  }, [data]);

  if (isLoading) return <Spinner message="Loading activity data..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load activity data: {(error as Error).message}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return <p className="py-8 text-center text-text-muted">No activity data available</p>;
  }

  return (
    <div className="space-y-6">
      <div>
        <label className="mb-1 block text-sm font-medium text-text-secondary">Search Signature</label>
        <SearchInput
          value={search}
          onChange={setSearch}
          onSubmit={(v) => {
            if (signatures.includes(v)) setSelectedSignature(v);
          }}
          suggestions={filteredSignatures}
          placeholder="Type to search signatures..."
          className="max-w-sm"
        />
      </div>

      {selectedSignature && barData && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Top / Bottom 15 Cell Types for {selectedSignature}
          </h3>
          <BarChart
            categories={barData.categories}
            values={barData.values}
            orientation="h"
            xTitle="Mean Activity (z-score)"
            yTitle="Cell Type"
            title={`${selectedSignature} Activity by Cell Type`}
            height={Math.max(500, barData.categories.length * 22 + 150)}
          />
        </div>
      )}

      {!selectedSignature && (
        <p className="text-sm text-text-muted">
          Select a signature above to see the top and bottom cell types by activity.
        </p>
      )}

      {heatmap && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">Activity Heatmap</h3>
          <HeatmapChart
            z={heatmap.z}
            x={heatmap.x}
            y={heatmap.y}
            title="Cell Type x Signature Activity"
            xTitle="Signature"
            yTitle="Cell Type"
            colorbarTitle="Mean Activity"
            symmetric
          />
        </div>
      )}
    </div>
  );
}
