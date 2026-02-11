import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import type { CorrelationData } from '@/api/types/activity';
import { Spinner } from '@/components/ui/loading-skeleton';
import { HeatmapChart } from '@/components/charts/heatmap-chart';

interface SeverityPanelProps {
  signatureType: string;
}

const SIGNIFICANCE_THRESHOLD = 0.05;

export default function SeverityPanel({ signatureType }: SeverityPanelProps) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['inflammation', 'severity', signatureType],
    queryFn: () =>
      get<CorrelationData[]>('/atlases/inflammation/correlations/severity', {
        signature_type: signatureType,
      }),
  });

  const heatmap = useMemo(() => {
    if (!data) return null;

    const diseases = [
      ...new Set(data.filter((d) => d.cell_type).map((d) => d.cell_type!)),
    ].sort();
    const signatures = [...new Set(data.map((d) => d.signature))].sort();

    const lookup = new Map(
      data
        .filter((d) => d.cell_type)
        .map((d) => [
          `${d.cell_type}||${d.signature}`,
          { rho: d.rho, p: d.p_value },
        ]),
    );

    // Null out non-significant values (set to null for display)
    const z = diseases.map((disease) =>
      signatures.map((sig) => {
        const entry = lookup.get(`${disease}||${sig}`);
        if (!entry) return 0;
        return entry.p <= SIGNIFICANCE_THRESHOLD ? entry.rho : 0;
      }),
    );

    return { z, x: signatures, y: diseases };
  }, [data]);

  if (isLoading) return <Spinner message="Loading severity correlations..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load severity data: {(error as Error).message}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <p className="py-8 text-center text-text-muted">
        No severity correlation data available
      </p>
    );
  }

  return (
    <div className="space-y-6">
      {heatmap && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Disease Severity Correlation
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Spearman correlation between signature activity and disease severity.
            Non-significant correlations (p &gt; {SIGNIFICANCE_THRESHOLD}) are
            zeroed out.
          </p>
          <HeatmapChart
            z={heatmap.z}
            x={heatmap.x}
            y={heatmap.y}
            title="Severity Correlation (Spearman rho)"
            xTitle="Signature"
            yTitle="Disease"
            colorbarTitle="Spearman rho"
            symmetric
          />
        </div>
      )}
    </div>
  );
}
