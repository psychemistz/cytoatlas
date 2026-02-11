import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { HeatmapChart } from '@/components/charts/heatmap-chart';
import { BoxplotChart } from '@/components/charts/boxplot-chart';

interface TcellStateData {
  cancer_type: string;
  state: string;
  signature: string;
  mean_activity: number;
}

interface TcellStatePanelProps {
  signatureType: string;
}

export default function TcellStatePanel({
  signatureType,
}: TcellStatePanelProps) {
  const [cancerType, setCancerType] = useState('All');
  const [selectedSignature, setSelectedSignature] = useState<string | null>(
    null,
  );

  const { data, isLoading, error } = useQuery({
    queryKey: ['scatlas', 'tcell-states', signatureType],
    queryFn: () =>
      get<TcellStateData[]>('/atlases/scatlas/tcell-states', {
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

  const signatures = useMemo(() => {
    if (!filtered.length) return [];
    return [...new Set(filtered.map((d) => d.signature))].sort();
  }, [filtered]);

  const heatmap = useMemo(() => {
    if (!filtered.length) return null;

    const states = [...new Set(filtered.map((d) => d.state))].sort();
    const sigs = [...new Set(filtered.map((d) => d.signature))].sort();

    const lookup = new Map<string, number[]>();
    for (const d of filtered) {
      const key = `${d.state}||${d.signature}`;
      if (!lookup.has(key)) lookup.set(key, []);
      lookup.get(key)!.push(d.mean_activity);
    }

    const z = states.map((st) =>
      sigs.map((sig) => {
        const vals = lookup.get(`${st}||${sig}`);
        if (!vals || vals.length === 0) return 0;
        return vals.reduce((a, b) => a + b, 0) / vals.length;
      }),
    );

    return { z, x: sigs, y: states };
  }, [filtered]);

  const boxplot = useMemo(() => {
    if (!filtered.length || !selectedSignature) return null;

    const sigRows = filtered.filter((d) => d.signature === selectedSignature);
    const stateMap = new Map<string, number[]>();
    for (const d of sigRows) {
      if (!stateMap.has(d.state)) stateMap.set(d.state, []);
      stateMap.get(d.state)!.push(d.mean_activity);
    }

    const states = [...stateMap.keys()].sort();
    return {
      groups: states,
      values: states.map((s) => stateMap.get(s)!),
    };
  }, [filtered, selectedSignature]);

  if (isLoading) return <Spinner message="Loading T cell state data..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load T cell state data: {(error as Error).message}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <p className="py-8 text-center text-text-muted">
        No T cell state data available
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
        {signatures.length > 0 && (
          <SelectFilter
            label="Signature"
            options={[
              { value: '', label: 'Select signature...' },
              ...signatures.map((s) => ({ value: s, label: s })),
            ]}
            value={selectedSignature ?? ''}
            onChange={(v) => setSelectedSignature(v || null)}
          />
        )}
      </FilterBar>

      {heatmap && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            T Cell State x Signature Activity
          </h3>
          <HeatmapChart
            z={heatmap.z}
            x={heatmap.x}
            y={heatmap.y}
            title="T Cell State x Signature Activity"
            xTitle="Signature"
            yTitle="T Cell State"
            colorbarTitle="Mean Activity"
            symmetric
          />
        </div>
      )}

      {boxplot && selectedSignature && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            {selectedSignature} Activity Distribution per T Cell State
          </h3>
          <BoxplotChart
            groups={boxplot.groups}
            values={boxplot.values}
            title={`${selectedSignature}: Activity by T Cell State`}
            yTitle="Mean Activity (z-score)"
            showPoints
          />
        </div>
      )}

      {!selectedSignature && (
        <p className="text-sm text-text-muted">
          Select a signature above to see the activity distribution per T cell
          state.
        </p>
      )}
    </div>
  );
}
