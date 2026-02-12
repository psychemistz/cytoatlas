import { useMemo } from 'react';
import { useCimaBiochemistry } from '@/api/hooks/use-cima';
import type { BiochemCorrelation } from '@/api/types/activity';
import { Spinner } from '@/components/ui/loading-skeleton';
import { HeatmapChart } from '@/components/charts/heatmap-chart';
import { CORRELATION_COLORSCALE } from '@/components/charts/chart-defaults';

interface BiochemistryPanelProps {
  signatureType: string;
}

function buildHeatmapFromFlat(rows: BiochemCorrelation[], topN: number) {
  // Rank signatures by their maximum absolute rho across all markers
  const sigMaxRho = new Map<string, number>();
  for (const row of rows) {
    const current = sigMaxRho.get(row.signature) ?? 0;
    const absRho = Math.abs(row.rho);
    if (absRho > current) {
      sigMaxRho.set(row.signature, absRho);
    }
  }

  const rankedSigs = [...sigMaxRho.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .map(([sig]) => sig);

  const sigSet = new Set(rankedSigs);
  const markers = [...new Set(rows.map((r) => r.marker))].sort();

  // Build lookup for fast access
  const lookup = new Map<string, number>();
  for (const row of rows) {
    if (sigSet.has(row.signature)) {
      lookup.set(`${row.signature}||${row.marker}`, row.rho);
    }
  }

  // Build z-matrix: rows = signatures, cols = markers
  const z = rankedSigs.map((sig) =>
    markers.map((marker) => lookup.get(`${sig}||${marker}`) ?? 0),
  );

  return { z, x: markers, y: rankedSigs };
}

export default function BiochemistryPanel({ signatureType }: BiochemistryPanelProps) {
  const { data, isLoading, error } = useCimaBiochemistry(signatureType);

  const heatmap = useMemo(() => {
    if (!data || data.length === 0) return null;
    return buildHeatmapFromFlat(data, 30);
  }, [data]);

  if (isLoading) return <Spinner message="Loading biochemistry correlations..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load biochemistry data: {(error as Error).message}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <p className="py-8 text-center text-text-muted">
        No biochemistry correlation data available
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="mb-1 text-sm font-semibold text-text-secondary">
          Biochemistry Marker Correlations
        </h3>
        <p className="mb-3 text-xs text-text-muted">
          Top 30 signatures by max |Spearman rho| across all biochemistry markers.
          {data?.length ?? 0} total correlation pairs.
        </p>
      </div>

      {heatmap && heatmap.y.length > 0 && (
        <HeatmapChart
          z={heatmap.z}
          x={heatmap.x}
          y={heatmap.y}
          title="Signature x Biochemistry Marker Correlation"
          xTitle="Biochemistry Marker"
          yTitle="Signature"
          colorbarTitle="Spearman rho"
          colorscale={CORRELATION_COLORSCALE}
          symmetric
          height={Math.max(500, heatmap.y.length * 22 + 200)}
        />
      )}
    </div>
  );
}
