import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { BarChart } from '@/components/charts/bar-chart';
import { HeatmapChart } from '@/components/charts/heatmap-chart';

interface InfiltrationData {
  cancer_type: string;
  cell_type: string;
  fraction: number;
  mean_activity: number;
  signature: string;
}

interface ImmuneInfiltrationProps {
  signatureType: string;
}

export default function ImmuneInfiltration({
  signatureType,
}: ImmuneInfiltrationProps) {
  const [cancerType, setCancerType] = useState('All');

  const { data, isLoading, error } = useQuery({
    queryKey: ['scatlas', 'immune-infiltration', signatureType],
    queryFn: () =>
      get<InfiltrationData[]>('/atlases/scatlas/immune-infiltration', {
        signature_type: signatureType,
      }),
  });

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

  const filtered = useMemo(() => {
    if (!data) return [];
    if (cancerType === 'All') return data;
    return data.filter((d) => d.cancer_type === cancerType);
  }, [data, cancerType]);

  const stackedBar = useMemo(() => {
    if (!filtered.length) return null;

    const ctSet = [...new Set(filtered.map((d) => d.cancer_type))].sort();
    const cellTypes = [...new Set(filtered.map((d) => d.cell_type))].sort();

    const fractionLookup = new Map(
      filtered.map((d) => [`${d.cancer_type}||${d.cell_type}`, d.fraction]),
    );

    const series = cellTypes.map((ct) => ({
      name: ct,
      values: ctSet.map(
        (cancer) => fractionLookup.get(`${cancer}||${ct}`) ?? 0,
      ),
    }));

    return { categories: ctSet, series };
  }, [filtered]);

  const heatmap = useMemo(() => {
    if (!filtered.length) return null;

    const cellTypes = [...new Set(filtered.map((d) => d.cell_type))].sort();
    const signatures = [...new Set(filtered.map((d) => d.signature))].sort();

    const lookup = new Map<string, number[]>();
    for (const d of filtered) {
      const key = `${d.cell_type}||${d.signature}`;
      if (!lookup.has(key)) lookup.set(key, []);
      lookup.get(key)!.push(d.mean_activity);
    }

    const z = cellTypes.map((ct) =>
      signatures.map((sig) => {
        const vals = lookup.get(`${ct}||${sig}`);
        if (!vals || vals.length === 0) return 0;
        return vals.reduce((a, b) => a + b, 0) / vals.length;
      }),
    );

    return { z, x: signatures, y: cellTypes };
  }, [filtered]);

  if (isLoading)
    return <Spinner message="Loading immune infiltration data..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load immune infiltration data: {(error as Error).message}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <p className="py-8 text-center text-text-muted">
        No immune infiltration data available
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

      {stackedBar && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Immune Cell Composition per Cancer Type
          </h3>
          <BarChart
            categories={stackedBar.categories}
            series={stackedBar.series}
            barmode="stack"
            orientation="v"
            xTitle="Cancer Type"
            yTitle="Fraction"
            title="Immune Cell Type Composition in TME"
            height={Math.max(500, stackedBar.categories.length * 30 + 150)}
          />
        </div>
      )}

      {heatmap && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Immune Signature Activity in TME Cells
          </h3>
          <HeatmapChart
            z={heatmap.z}
            x={heatmap.x}
            y={heatmap.y}
            title="Immune Cell Type x Signature Activity"
            xTitle="Signature"
            yTitle="Immune Cell Type"
            colorbarTitle="Mean Activity"
            symmetric
          />
        </div>
      )}
    </div>
  );
}
