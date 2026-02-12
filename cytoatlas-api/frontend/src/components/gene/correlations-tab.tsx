import { useMemo } from 'react';
import { useGeneCorrelations } from '@/api/hooks/use-gene';
import { Spinner } from '@/components/ui/loading-skeleton';
import { BarChart } from '@/components/charts/bar-chart';

function rhoColor(rho: number): string {
  const abs = Math.abs(rho);
  if (abs > 0.5) return rho > 0 ? 'text-red-600' : 'text-blue-600';
  if (abs > 0.3) return rho > 0 ? 'text-orange-500' : 'text-sky-500';
  return 'text-text-primary';
}

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

  const groupedData = useMemo(() => {
    if (!data || data.length === 0) return null;
    const groups = new Map<string, typeof data>();
    for (const d of data) {
      const key = d.type;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(d);
    }
    // Sort each group by |rho| descending
    for (const [, items] of groups) {
      items.sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho));
    }
    // Order categories: age, bmi, biochemistry, metabolites, then rest
    const order = ['age', 'bmi', 'biochemistry', 'metabolites'];
    const sorted = [...groups.entries()].sort((a, b) => {
      const ia = order.indexOf(a[0].toLowerCase());
      const ib = order.indexOf(b[0].toLowerCase());
      return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
    });
    return sorted;
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

      {groupedData && groupedData.map(([category, items]) => (
        <div key={category}>
          <h3 className="mb-2 text-sm font-semibold capitalize text-text-secondary">
            {category} ({items.length})
          </h3>
          <div className="overflow-x-auto rounded-lg border border-border-light">
            <table className="w-full text-left text-sm">
              <thead className="bg-bg-secondary">
                <tr>
                  <th className="px-3 py-2 font-medium">Variable</th>
                  <th className="px-3 py-2 font-medium text-right">Spearman rho</th>
                  <th className="px-3 py-2 font-medium text-right">p-value</th>
                  <th className="px-3 py-2 font-medium text-right">N</th>
                </tr>
              </thead>
              <tbody>
                {items.map((d, i) => (
                  <tr key={i} className="border-t border-border-light hover:bg-bg-tertiary">
                    <td className="px-3 py-1.5">{d.variable}</td>
                    <td className={`px-3 py-1.5 text-right font-mono ${rhoColor(d.rho)}`}>
                      {d.rho.toFixed(4)}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono">
                      {d.p_value != null ? d.p_value.toExponential(2) : '-'}
                    </td>
                    <td className="px-3 py-1.5 text-right">{d.n}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}
    </div>
  );
}
