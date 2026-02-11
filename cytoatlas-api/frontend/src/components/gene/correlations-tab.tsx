import { useMemo } from 'react';
import { useGeneCorrelations } from '@/api/hooks/use-gene';
import { Spinner } from '@/components/ui/loading-skeleton';
import { BarChart } from '@/components/charts/bar-chart';

interface CorrelationsTabProps {
  gene: string;
  signatureType: string;
}

export default function CorrelationsTab({
  gene,
  signatureType,
}: CorrelationsTabProps) {
  const { data, isLoading, error } = useGeneCorrelations(gene, signatureType);

  const barData = useMemo(() => {
    if (!data || data.length === 0) return null;
    const sorted = [...data].sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho));
    return {
      categories: sorted.map((d) => `${d.variable} (${d.type})`),
      values: sorted.map((d) => d.rho),
    };
  }, [data]);

  if (isLoading) return <Spinner message="Loading correlations..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load correlations: {(error as Error).message}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <p className="py-8 text-center text-text-muted">
        No correlation data for {gene}
      </p>
    );
  }

  return (
    <div className="space-y-6">
      {barData && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Phenotype Correlations
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Spearman correlation between {gene} activity and phenotypic variables.
          </p>
          <BarChart
            categories={barData.categories}
            values={barData.values}
            orientation="h"
            xTitle="Spearman rho"
            yTitle="Variable"
            title={`${gene}: Correlations`}
            height={Math.max(400, barData.categories.length * 24 + 150)}
          />
        </div>
      )}

      <div className="overflow-x-auto rounded-lg border border-border-light">
        <table className="w-full text-left text-sm">
          <thead className="bg-bg-secondary">
            <tr>
              <th className="px-3 py-2 font-medium">Variable</th>
              <th className="px-3 py-2 font-medium">Type</th>
              <th className="px-3 py-2 font-medium text-right">Spearman rho</th>
              <th className="px-3 py-2 font-medium text-right">p-value</th>
              <th className="px-3 py-2 font-medium text-right">N</th>
            </tr>
          </thead>
          <tbody>
            {data.map((d, i) => (
              <tr key={i} className="border-t border-border-light">
                <td className="px-3 py-1.5">{d.variable}</td>
                <td className="px-3 py-1.5">
                  <span className="rounded bg-bg-secondary px-1.5 py-0.5 text-xs">
                    {d.type}
                  </span>
                </td>
                <td className="px-3 py-1.5 text-right font-mono">
                  {d.rho.toFixed(4)}
                </td>
                <td className="px-3 py-1.5 text-right font-mono">
                  {d.p_value.toExponential(2)}
                </td>
                <td className="px-3 py-1.5 text-right">{d.n}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
