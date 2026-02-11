import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { HeatmapChart } from '@/components/charts/heatmap-chart';

interface DriversPanelProps {
  signatureType: string;
}

interface DriverEntry {
  cell_type: string;
  signature: string;
  driver_score: number;
  disease: string;
}

export default function DriversPanel({ signatureType }: DriversPanelProps) {
  const [selectedDisease, setSelectedDisease] = useState<string>('');

  const { data, isLoading, error } = useQuery({
    queryKey: ['inflammation', 'cell-type-drivers', signatureType],
    queryFn: () =>
      get<DriverEntry[]>('/atlases/inflammation/cell-type-drivers', {
        signature_type: signatureType,
      }),
  });

  const diseases = useMemo(() => {
    if (!data) return [];
    return [...new Set(data.map((d) => d.disease))].sort();
  }, [data]);

  const activeDisease = selectedDisease || diseases[0] || '';

  const diseaseOptions = useMemo(() => {
    return diseases.map((d) => ({ value: d, label: d }));
  }, [diseases]);

  const heatmap = useMemo(() => {
    if (!data || !activeDisease) return null;

    const filtered = data.filter((d) => d.disease === activeDisease);
    if (filtered.length === 0) return null;

    const cellTypes = [...new Set(filtered.map((d) => d.cell_type))].sort();
    const signatures = [...new Set(filtered.map((d) => d.signature))].sort();

    const lookup = new Map(
      filtered.map((d) => [
        `${d.cell_type}||${d.signature}`,
        d.driver_score,
      ]),
    );

    const z = cellTypes.map((ct) =>
      signatures.map((sig) => lookup.get(`${ct}||${sig}`) ?? 0),
    );

    return { z, x: signatures, y: cellTypes };
  }, [data, activeDisease]);

  if (isLoading) return <Spinner message="Loading driver data..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load driver data: {(error as Error).message}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <p className="py-8 text-center text-text-muted">
        No cell-type driver data available
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
            Cell-Type Driver Scores for {activeDisease}
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Driver scores indicate which cell types contribute most to each
            signature in {activeDisease}.
          </p>
          <HeatmapChart
            z={heatmap.z}
            x={heatmap.x}
            y={heatmap.y}
            title={`Cell-Type Drivers: ${activeDisease}`}
            xTitle="Signature"
            yTitle="Cell Type"
            colorbarTitle="Driver Score"
            symmetric={false}
            zmin={0}
          />
        </div>
      )}

      {activeDisease && !heatmap && (
        <p className="py-4 text-center text-sm text-text-muted">
          No driver data available for {activeDisease}.
        </p>
      )}
    </div>
  );
}
