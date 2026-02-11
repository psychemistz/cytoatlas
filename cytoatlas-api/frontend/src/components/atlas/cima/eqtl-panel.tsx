import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import { Spinner } from '@/components/ui/loading-skeleton';
import { ScatterChart } from '@/components/charts/scatter-chart';
import { FilterBar, SearchFilter } from '@/components/ui/filter-bar';

interface EqtlPanelProps {
  signatureType: string;
}

interface EqtlResult {
  gene: string;
  snp: string;
  beta: number;
  p_value: number;
  fdr: number;
}

export default function EqtlPanel({ signatureType }: EqtlPanelProps) {
  const [search, setSearch] = useState('');

  const { data, isLoading, error } = useQuery({
    queryKey: ['cima', 'eqtl', signatureType],
    queryFn: () =>
      get<EqtlResult[]>('/atlases/cima/eqtl', {
        signature_type: signatureType,
      }),
  });

  const filtered = useMemo(() => {
    if (!data) return [];
    if (!search.trim()) return data;
    const q = search.toLowerCase();
    return data.filter(
      (d) =>
        d.gene.toLowerCase().includes(q) ||
        d.snp.toLowerCase().includes(q),
    );
  }, [data, search]);

  const scatterData = useMemo(() => {
    if (filtered.length === 0) return null;

    const x = filtered.map((d) => d.beta);
    const y = filtered.map((d) => -Math.log10(Math.max(d.p_value, 1e-300)));
    const labels = filtered.map((d) => `${d.gene} (${d.snp})`);
    const colors = filtered.map((d) =>
      d.fdr < 0.05
        ? d.beta > 0
          ? '#ef4444'
          : '#3b82f6'
        : '#9ca3af',
    );

    return { x, y, labels, colors };
  }, [filtered]);

  const topResults = useMemo(() => {
    if (!filtered || filtered.length === 0) return [];
    return [...filtered]
      .sort((a, b) => a.p_value - b.p_value)
      .slice(0, 20);
  }, [filtered]);

  if (isLoading) return <Spinner message="Loading eQTL results..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load eQTL data: {(error as Error).message}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <p className="py-8 text-center text-text-muted">
        No eQTL data available
      </p>
    );
  }

  const sigCount = data.filter((d) => d.fdr < 0.05).length;

  return (
    <div className="space-y-6">
      <div>
        <h3 className="mb-1 text-sm font-semibold text-text-secondary">
          eQTL Analysis
        </h3>
        <p className="mb-3 text-xs text-text-muted">
          {data.length} eQTL associations tested, {sigCount} significant at FDR &lt; 0.05.
          {search && ` Showing ${filtered.length} matches for "${search}".`}
        </p>
      </div>

      <FilterBar>
        <SearchFilter
          value={search}
          onChange={setSearch}
          placeholder="Search gene or SNP..."
        />
      </FilterBar>

      {scatterData && scatterData.x.length > 0 && (
        <ScatterChart
          x={scatterData.x}
          y={scatterData.y}
          labels={scatterData.labels}
          colors={scatterData.colors}
          xTitle="Effect Size (beta)"
          yTitle="-log10(p-value)"
          title="eQTL Effect Size vs Significance"
          height={500}
        />
      )}

      {scatterData && scatterData.x.length === 0 && (
        <p className="py-4 text-center text-text-muted">
          No eQTL results match the search filter
        </p>
      )}

      {topResults.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-semibold text-text-secondary">
            Top 20 eQTL Results by Significance
          </h4>
          <div className="overflow-x-auto rounded-lg border border-border-light">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-border-light bg-bg-secondary">
                  <th className="px-3 py-2 font-medium text-text-secondary">Gene</th>
                  <th className="px-3 py-2 font-medium text-text-secondary">SNP</th>
                  <th className="px-3 py-2 text-right font-medium text-text-secondary">Beta</th>
                  <th className="px-3 py-2 text-right font-medium text-text-secondary">p-value</th>
                  <th className="px-3 py-2 text-right font-medium text-text-secondary">FDR</th>
                </tr>
              </thead>
              <tbody>
                {topResults.map((row, i) => (
                  <tr
                    key={`${row.gene}-${row.snp}-${i}`}
                    className="border-b border-border-light last:border-b-0 hover:bg-bg-tertiary"
                  >
                    <td className="px-3 py-2 font-medium text-text-primary">
                      {row.gene}
                    </td>
                    <td className="px-3 py-2 font-mono text-xs text-text-secondary">
                      {row.snp}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      <span className={row.beta > 0 ? 'text-red-600' : 'text-blue-600'}>
                        {row.beta.toFixed(4)}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-text-secondary">
                      {row.p_value.toExponential(2)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      <span className={row.fdr < 0.05 ? 'font-semibold text-text-primary' : 'text-text-muted'}>
                        {row.fdr.toExponential(2)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
