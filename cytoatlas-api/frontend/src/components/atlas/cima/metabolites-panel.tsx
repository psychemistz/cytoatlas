import { useState, useMemo } from 'react';
import { useCimaMetabolites } from '@/api/hooks/use-cima';
import type { MetaboliteCorrelation } from '@/api/types/activity';
import { Spinner } from '@/components/ui/loading-skeleton';
import { LollipopChart } from '@/components/charts/lollipop-chart';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';

interface MetabolitesPanelProps {
  signatureType: string;
}

const CATEGORY_OPTIONS = [
  { value: 'All', label: 'All Categories' },
  { value: 'Lipid', label: 'Lipid' },
  { value: 'Amino Acid', label: 'Amino Acid' },
  { value: 'Carbohydrate', label: 'Carbohydrate' },
  { value: 'Nucleotide', label: 'Nucleotide' },
  { value: 'Cofactor', label: 'Cofactor' },
];

function filterAndRank(
  rows: MetaboliteCorrelation[],
  category: string,
  topN: number,
) {
  const filtered = category === 'All'
    ? rows
    : rows.filter((r) => r.category === category);

  const sorted = [...filtered].sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho));
  const top = sorted.slice(0, topN);

  return {
    categories: top.map((d) => `${d.signature} \u2194 ${d.metabolite}`),
    values: top.map((d) => Math.abs(d.rho)),
    rhoValues: top.map((d) => d.rho),
    categoryLabels: top.map((d) => d.category),
  };
}

export default function MetabolitesPanel({ signatureType }: MetabolitesPanelProps) {
  const { data, isLoading, error } = useCimaMetabolites(signatureType);
  const [category, setCategory] = useState('All');

  const availableCategories = useMemo(() => {
    if (!data) return CATEGORY_OPTIONS;
    const present = new Set(data.map((d) => d.category));
    return CATEGORY_OPTIONS.filter(
      (opt) => opt.value === 'All' || present.has(opt.value),
    );
  }, [data]);

  const lollipopData = useMemo(() => {
    if (!data || data.length === 0) return null;
    return filterAndRank(data, category, 30);
  }, [data, category]);

  const summaryText = useMemo(() => {
    if (!data) return '';
    const total = data.length;
    const catCounts = new Map<string, number>();
    for (const row of data) {
      catCounts.set(row.category, (catCounts.get(row.category) ?? 0) + 1);
    }
    const parts = [...catCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([cat, n]) => `${cat}: ${n}`)
      .join(', ');
    return `${total} total pairs (${parts})`;
  }, [data]);

  if (isLoading) return <Spinner message="Loading metabolite correlations..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load metabolite data: {(error as Error).message}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <p className="py-8 text-center text-text-muted">
        No metabolite correlation data available
      </p>
    );
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="mb-1 text-sm font-semibold text-text-secondary">
          Metabolite-Signature Correlations
        </h3>
        <p className="mb-3 text-xs text-text-muted">{summaryText}</p>
      </div>

      <FilterBar>
        <SelectFilter
          label="Category"
          options={availableCategories}
          value={category}
          onChange={setCategory}
        />
      </FilterBar>

      {lollipopData && lollipopData.categories.length > 0 ? (
        <LollipopChart
          categories={lollipopData.categories}
          values={lollipopData.values}
          title={`Top 30 Metabolite-Signature Pairs by |rho|${category !== 'All' ? ` (${category})` : ''}`}
          xTitle="|Spearman rho|"
        />
      ) : (
        <p className="py-8 text-center text-text-muted">
          No correlations found for the selected category
        </p>
      )}
    </div>
  );
}
