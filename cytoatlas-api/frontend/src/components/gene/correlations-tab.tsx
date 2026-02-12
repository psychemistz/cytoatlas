import { useState, useMemo } from 'react';
import { useGeneCorrelations } from '@/api/hooks/use-gene';
import { Spinner } from '@/components/ui/loading-skeleton';
import { BarChart } from '@/components/charts/bar-chart';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';

const CATEGORY_COLORS: Record<string, string> = {
  age: '#2166ac',
  bmi: '#b2182b',
  biochemistry: '#1b7837',
  metabolite: '#762a83',
  metabolites: '#762a83',
};

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
  const [category, setCategory] = useState('all');
  const [sigOnly, setSigOnly] = useState(false);

  const categoryOptions = useMemo(() => {
    const opts = [{ value: 'all', label: 'All Categories' }];
    if (!data) return opts;
    const cats = [...new Set(data.map((d) => d.type))].sort();
    for (const c of cats) {
      opts.push({ value: c, label: c.charAt(0).toUpperCase() + c.slice(1) });
    }
    return opts;
  }, [data]);

  const filtered = useMemo(() => {
    if (!data) return [];
    let items = data;
    if (category !== 'all') items = items.filter((d) => d.type === category);
    if (sigOnly) items = items.filter((d) => d.q_value != null && d.q_value < 0.05);
    return items;
  }, [data, category, sigOnly]);

  const barData = useMemo(() => {
    if (filtered.length === 0) return null;
    const sorted = [...filtered].sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho));
    return {
      categories: sorted.map((d) => `${d.variable} (${d.type})`),
      values: sorted.map((d) => d.rho),
      colors: sorted.map((d) => CATEGORY_COLORS[d.type.toLowerCase()] ?? '#6b7280'),
    };
  }, [filtered]);

  const groupedData = useMemo(() => {
    if (filtered.length === 0) return null;
    const groups = new Map<string, typeof filtered>();
    for (const d of filtered) {
      const key = d.type;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(d);
    }
    for (const [, items] of groups) {
      items.sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho));
    }
    const order = ['age', 'bmi', 'biochemistry', 'metabolites', 'metabolite'];
    const sorted = [...groups.entries()].sort((a, b) => {
      const ia = order.indexOf(a[0].toLowerCase());
      const ib = order.indexOf(b[0].toLowerCase());
      return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
    });
    return sorted;
  }, [filtered]);

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
      <FilterBar>
        <SelectFilter
          label="Category"
          options={categoryOptions}
          value={category}
          onChange={setCategory}
        />
        <label className="flex items-center gap-2 text-sm text-text-secondary">
          <input
            type="checkbox"
            checked={sigOnly}
            onChange={(e) => setSigOnly(e.target.checked)}
            className="rounded border-border-primary"
          />
          Show significant only
        </label>
      </FilterBar>

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
            colors={barData.colors}
          />
        </div>
      )}

      {groupedData && groupedData.map(([cat, items]) => (
        <div key={cat}>
          <h3 className="mb-2 text-sm font-semibold capitalize text-text-secondary">
            {cat} ({items.length})
          </h3>
          <div className="overflow-x-auto rounded-lg border border-border-light">
            <table className="w-full text-left text-sm">
              <thead className="bg-bg-secondary">
                <tr>
                  <th className="px-3 py-2 font-medium">Variable</th>
                  <th className="px-3 py-2 font-medium">Cell Type</th>
                  <th className="px-3 py-2 font-medium text-right">Spearman rho</th>
                  <th className="px-3 py-2 font-medium text-right">p-value</th>
                  <th className="px-3 py-2 font-medium text-right">Q-value</th>
                  <th className="px-3 py-2 font-medium text-right">N</th>
                  <th className="px-3 py-2 font-medium text-center">Sig</th>
                </tr>
              </thead>
              <tbody>
                {items.map((d, i) => (
                  <tr key={i} className="border-t border-border-light hover:bg-bg-tertiary">
                    <td className="px-3 py-1.5">{d.variable}</td>
                    <td className="px-3 py-1.5 text-text-muted">{d.cell_type ?? '-'}</td>
                    <td className={`px-3 py-1.5 text-right font-mono ${rhoColor(d.rho)}`}>
                      {d.rho.toFixed(4)}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono">
                      {d.p_value != null ? d.p_value.toExponential(2) : '-'}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono">
                      {d.q_value != null ? d.q_value.toExponential(2) : '-'}
                    </td>
                    <td className="px-3 py-1.5 text-right">{d.n}</td>
                    <td className="px-3 py-1.5 text-center">
                      {d.q_value != null && d.q_value < 0.05 ? (
                        <span className="text-green-600" title="FDR < 0.05">{'\u2713'}</span>
                      ) : ''}
                    </td>
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
