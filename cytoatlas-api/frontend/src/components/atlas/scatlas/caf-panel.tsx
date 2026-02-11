import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { HeatmapChart } from '@/components/charts/heatmap-chart';
import { BarChart } from '@/components/charts/bar-chart';

interface CafData {
  cancer_type: string;
  subtype: string;
  signature: string;
  mean_activity: number;
}

interface CafPanelProps {
  signatureType: string;
}

export default function CafPanel({ signatureType }: CafPanelProps) {
  const [cancerType, setCancerType] = useState('All');

  const { data, isLoading, error } = useQuery({
    queryKey: ['scatlas', 'caf-subtypes', signatureType],
    queryFn: () =>
      get<CafData[]>('/atlases/scatlas/caf-subtypes', {
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

  const heatmap = useMemo(() => {
    if (!filtered.length) return null;

    const subtypes = [...new Set(filtered.map((d) => d.subtype))].sort();
    const signatures = [...new Set(filtered.map((d) => d.signature))].sort();

    const lookup = new Map<string, number[]>();
    for (const d of filtered) {
      const key = `${d.subtype}||${d.signature}`;
      if (!lookup.has(key)) lookup.set(key, []);
      lookup.get(key)!.push(d.mean_activity);
    }

    const z = subtypes.map((st) =>
      signatures.map((sig) => {
        const vals = lookup.get(`${st}||${sig}`);
        if (!vals || vals.length === 0) return 0;
        return vals.reduce((a, b) => a + b, 0) / vals.length;
      }),
    );

    return { z, x: signatures, y: subtypes };
  }, [filtered]);

  const barData = useMemo(() => {
    if (!filtered.length) return null;

    const subtypeActivity = new Map<string, Map<string, number[]>>();
    for (const d of filtered) {
      if (!subtypeActivity.has(d.subtype))
        subtypeActivity.set(d.subtype, new Map());
      const sigMap = subtypeActivity.get(d.subtype)!;
      if (!sigMap.has(d.signature)) sigMap.set(d.signature, []);
      sigMap.get(d.signature)!.push(d.mean_activity);
    }

    const enriched: { signature: string; subtype: string; activity: number }[] =
      [];
    for (const [subtype, sigMap] of subtypeActivity) {
      for (const [signature, vals] of sigMap) {
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        enriched.push({ signature, subtype, activity: mean });
      }
    }

    const sorted = [...enriched]
      .sort((a, b) => Math.abs(b.activity) - Math.abs(a.activity))
      .slice(0, 20);

    return {
      categories: sorted.map((d) => `${d.signature} (${d.subtype})`),
      values: sorted.map((d) => d.activity),
    };
  }, [filtered]);

  if (isLoading) return <Spinner message="Loading CAF subtype data..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load CAF subtype data: {(error as Error).message}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <p className="py-8 text-center text-text-muted">
        No CAF subtype data available
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

      {heatmap && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            CAF Subtype x Signature Activity
          </h3>
          <HeatmapChart
            z={heatmap.z}
            x={heatmap.x}
            y={heatmap.y}
            title="CAF Subtype x Signature Activity"
            xTitle="Signature"
            yTitle="CAF Subtype"
            colorbarTitle="Mean Activity"
            symmetric
          />
        </div>
      )}

      {barData && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Top 20 Subtype-Enriched Signatures
          </h3>
          <BarChart
            categories={barData.categories}
            values={barData.values}
            orientation="h"
            xTitle="Mean Activity (z-score)"
            yTitle="Signature (Subtype)"
            title="Top Subtype-Enriched Signatures"
            height={Math.max(500, barData.categories.length * 24 + 150)}
          />
        </div>
      )}
    </div>
  );
}
