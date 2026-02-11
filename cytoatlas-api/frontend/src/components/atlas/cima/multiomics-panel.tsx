import { useMemo } from 'react';
import { useCimaBiochemistry, useCimaMetabolites } from '@/api/hooks/use-cima';
import { Spinner } from '@/components/ui/loading-skeleton';
import { BarChart } from '@/components/charts/bar-chart';

interface MultiomicsPanelProps {
  signatureType: string;
}

const FDR_THRESHOLD = 0.05;

export default function MultiomicsPanel({ signatureType }: MultiomicsPanelProps) {
  const biochemQuery = useCimaBiochemistry(signatureType);
  const metabQuery = useCimaMetabolites(signatureType);

  const barData = useMemo(() => {
    const counts = new Map<string, number>();

    if (biochemQuery.data) {
      const sigCount = biochemQuery.data.filter(
        (d) => (d.q_value ?? d.p_value) < FDR_THRESHOLD,
      ).length;
      if (sigCount > 0) counts.set('Biochemistry', sigCount);
    }

    if (metabQuery.data) {
      const byCat = new Map<string, number>();
      for (const row of metabQuery.data) {
        if ((row.q_value ?? row.p_value) < FDR_THRESHOLD) {
          const cat = row.category || 'Other';
          byCat.set(cat, (byCat.get(cat) ?? 0) + 1);
        }
      }
      for (const [cat, n] of byCat) {
        counts.set(cat, n);
      }
    }

    if (counts.size === 0) return null;

    const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    return {
      categories: sorted.map(([cat]) => cat),
      values: sorted.map(([, n]) => n),
    };
  }, [biochemQuery.data, metabQuery.data]);

  const totalSig = barData
    ? barData.values.reduce((a, b) => a + b, 0)
    : 0;

  const isLoading = biochemQuery.isLoading || metabQuery.isLoading;
  const error = biochemQuery.error || metabQuery.error;

  if (isLoading) return <Spinner message="Loading multi-omics data..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load multi-omics data: {(error as Error).message}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="mb-1 text-sm font-semibold text-text-secondary">
          Multi-omics Integration
        </h3>
        <p className="mb-3 text-xs text-text-muted">
          Multi-omics integration of biochemistry and metabolite correlations.
          A force-directed network visualization is planned for a future release.
        </p>
      </div>

      <div className="rounded-lg border border-border-light bg-bg-secondary p-4">
        <h4 className="mb-2 text-sm font-semibold text-text-secondary">
          Significant Correlations by Category (FDR &lt; 0.05)
        </h4>
        <p className="mb-3 text-xs text-text-muted">
          {totalSig} significant correlation pairs across all omics layers.
        </p>

        {barData && barData.categories.length > 0 ? (
          <BarChart
            categories={barData.categories}
            values={barData.values}
            orientation="h"
            xTitle="Number of Significant Pairs"
            yTitle="Category"
            title="Significant Correlations by Category"
            height={Math.max(350, barData.categories.length * 40 + 150)}
          />
        ) : (
          <p className="py-6 text-center text-text-muted">
            No significant correlations found at FDR &lt; 0.05
          </p>
        )}
      </div>
    </div>
  );
}
