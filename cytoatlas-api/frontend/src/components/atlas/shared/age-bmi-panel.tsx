import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import type { CorrelationData } from '@/api/types/activity';
import { Spinner } from '@/components/ui/loading-skeleton';
import { LollipopChart } from '@/components/charts/lollipop-chart';
import { HeatmapChart } from '@/components/charts/heatmap-chart';

interface AgeBmiPanelProps {
  signatureType: string;
  atlasName: string;
}

function buildLollipopData(rows: CorrelationData[], n: number) {
  const sorted = [...rows].sort((a, b) => b.rho - a.rho);
  const topN = sorted.slice(0, n);
  const bottomN = sorted.slice(-n).reverse();
  const combined = [...topN, ...bottomN];
  const unique = Array.from(new Map(combined.map((d) => [d.signature, d])).values());
  return {
    categories: unique.map((d) => d.signature),
    values: unique.map((d) => d.rho),
  };
}

function buildCorrelationHeatmap(rows: CorrelationData[]) {
  const cellTypes = [...new Set(rows.filter((d) => d.cell_type).map((d) => d.cell_type!))].sort();
  const sigs = [...new Set(rows.map((d) => d.signature))].sort();
  const lookup = new Map(
    rows.filter((d) => d.cell_type).map((d) => [`${d.cell_type}||${d.signature}`, d.rho]),
  );
  const z = cellTypes.map((ct) => sigs.map((sig) => lookup.get(`${ct}||${sig}`) ?? 0));
  return { z, x: sigs, y: cellTypes };
}

export default function AgeBmiPanel({ signatureType, atlasName }: AgeBmiPanelProps) {
  const ageQuery = useQuery({
    queryKey: ['atlas-correlations-age', atlasName, signatureType],
    queryFn: () =>
      get<CorrelationData[]>(`/atlases/${atlasName}/correlations/age`, {
        signature_type: signatureType,
      }),
  });

  const bmiQuery = useQuery({
    queryKey: ['atlas-correlations-bmi', atlasName, signatureType],
    queryFn: () =>
      get<CorrelationData[]>(`/atlases/${atlasName}/correlations/bmi`, {
        signature_type: signatureType,
      }),
  });

  const ageLollipop = useMemo(() => {
    if (!ageQuery.data) return null;
    return buildLollipopData(ageQuery.data, 10);
  }, [ageQuery.data]);

  const bmiLollipop = useMemo(() => {
    if (!bmiQuery.data) return null;
    return buildLollipopData(bmiQuery.data, 10);
  }, [bmiQuery.data]);

  const ageHeatmap = useMemo(() => {
    if (!ageQuery.data) return null;
    return buildCorrelationHeatmap(ageQuery.data);
  }, [ageQuery.data]);

  const isLoading = ageQuery.isLoading || bmiQuery.isLoading;
  const error = ageQuery.error || bmiQuery.error;

  if (isLoading) return <Spinner message="Loading correlation data..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load correlation data: {(error as Error).message}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Age Correlations (top 10 positive + bottom 10 negative)
          </h3>
          {ageLollipop && ageLollipop.categories.length > 0 ? (
            <LollipopChart
              categories={ageLollipop.categories}
              values={ageLollipop.values}
              title="Age Correlation (Spearman rho)"
              xTitle="rho"
            />
          ) : (
            <p className="py-4 text-sm text-text-muted">No age correlation data available</p>
          )}
        </div>

        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            BMI Correlations (top 10 positive + bottom 10 negative)
          </h3>
          {bmiLollipop && bmiLollipop.categories.length > 0 ? (
            <LollipopChart
              categories={bmiLollipop.categories}
              values={bmiLollipop.values}
              title="BMI Correlation (Spearman rho)"
              xTitle="rho"
            />
          ) : (
            <p className="py-4 text-sm text-text-muted">No BMI correlation data available</p>
          )}
        </div>
      </div>

      {ageHeatmap && ageHeatmap.y.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Correlation Matrix (Cell Type x Signature)
          </h3>
          <HeatmapChart
            z={ageHeatmap.z}
            x={ageHeatmap.x}
            y={ageHeatmap.y}
            title="Age Correlation by Cell Type"
            xTitle="Signature"
            yTitle="Cell Type"
            colorbarTitle="Spearman rho"
            symmetric
          />
        </div>
      )}
    </div>
  );
}
