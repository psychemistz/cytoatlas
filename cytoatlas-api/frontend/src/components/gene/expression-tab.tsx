import { useState, useMemo } from 'react';
import { useGeneExpression } from '@/api/hooks/use-gene';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { BarChart } from '@/components/charts/bar-chart';

interface ExpressionTabProps {
  gene: string;
}

export default function ExpressionTab({ gene }: ExpressionTabProps) {
  const [atlas, setAtlas] = useState('all');
  const { data, isLoading, error } = useGeneExpression(gene);

  const atlases = useMemo(() => {
    if (!data?.data) return [];
    return [...new Set(data.data.map((d) => d.atlas))].sort();
  }, [data]);

  const atlasOptions = useMemo(
    () => [
      { value: 'all', label: 'All Atlases' },
      ...atlases.map((a) => ({ value: a, label: a })),
    ],
    [atlases],
  );

  const filtered = useMemo(() => {
    if (!data?.data) return [];
    let items = data.data;
    if (atlas !== 'all') items = items.filter((d) => d.atlas === atlas);
    return items.sort((a, b) => b.mean_expression - a.mean_expression);
  }, [data, atlas]);

  const barData = useMemo(() => {
    const top = filtered.slice(0, 30);
    if (!top.length) return null;
    return {
      categories: top.map((d) => `${d.cell_type} (${d.atlas})`),
      values: top.map((d) => d.mean_expression),
    };
  }, [filtered]);

  if (isLoading) return <Spinner message="Loading expression data..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load expression: {(error as Error).message}
      </div>
    );
  }

  if (!data?.data || data.data.length === 0) {
    return (
      <p className="py-8 text-center text-text-muted">
        No expression data available for {gene}
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <FilterBar>
        <SelectFilter
          label="Atlas"
          options={atlasOptions}
          value={atlas}
          onChange={setAtlas}
        />
      </FilterBar>

      <div className="text-sm text-text-secondary">
        {filtered.length} cell types, {atlases.length} atlases
      </div>

      {barData && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Top 30 Cell Types by Expression
          </h3>
          <BarChart
            categories={barData.categories}
            values={barData.values}
            orientation="h"
            xTitle="Mean Expression (log-normalized)"
            yTitle="Cell Type"
            title={`${gene} Expression`}
            height={Math.max(400, barData.categories.length * 24 + 150)}
          />
        </div>
      )}

      <div className="overflow-x-auto rounded-lg border border-border-light">
        <table className="w-full text-left text-sm">
          <thead className="bg-bg-secondary">
            <tr>
              <th className="px-3 py-2 font-medium">Cell Type</th>
              <th className="px-3 py-2 font-medium">Atlas</th>
              <th className="px-3 py-2 font-medium text-right">Mean Expr</th>
              <th className="px-3 py-2 font-medium text-right">% Expressed</th>
              <th className="px-3 py-2 font-medium text-right">N Cells</th>
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0, 50).map((d, i) => (
              <tr key={i} className="border-t border-border-light">
                <td className="px-3 py-1.5">{d.cell_type}</td>
                <td className="px-3 py-1.5">
                  <span className="rounded bg-bg-secondary px-1.5 py-0.5 text-xs">
                    {d.atlas}
                  </span>
                </td>
                <td className="px-3 py-1.5 text-right font-mono">
                  {d.mean_expression.toFixed(4)}
                </td>
                <td className="px-3 py-1.5 text-right">
                  {d.pct_expressed != null ? `${d.pct_expressed.toFixed(1)}%` : '-'}
                </td>
                <td className="px-3 py-1.5 text-right">
                  {d.n_cells?.toLocaleString() ?? '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
