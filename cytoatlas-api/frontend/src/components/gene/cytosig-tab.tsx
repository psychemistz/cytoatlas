import { useState, useMemo } from 'react';
import { useGeneCellTypes } from '@/api/hooks/use-gene';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { BarChart } from '@/components/charts/bar-chart';
import { ViolinChart } from '@/components/charts/violin-chart';

interface CytosigTabProps {
  gene: string;
  signatureType: string;
}

export default function CytosigTab({ gene, signatureType }: CytosigTabProps) {
  const [atlas, setAtlas] = useState('all');
  const { data, isLoading, error } = useGeneCellTypes(gene, signatureType);

  const atlases = useMemo(() => {
    if (!data) return [];
    return [...new Set(data.map((d) => d.atlas))].sort();
  }, [data]);

  const atlasOptions = useMemo(
    () => [
      { value: 'all', label: 'All Atlases' },
      ...atlases.map((a) => ({ value: a, label: a })),
    ],
    [atlases],
  );

  const filtered = useMemo(() => {
    if (!data) return [];
    let items = data;
    if (atlas !== 'all') items = items.filter((d) => d.atlas === atlas);
    return items.sort((a, b) => Math.abs(b.mean_activity) - Math.abs(a.mean_activity));
  }, [data, atlas]);

  const barData = useMemo(() => {
    const top = filtered.slice(0, 30);
    if (!top.length) return null;
    return {
      categories: top.map((d) => `${d.cell_type} (${d.atlas})`),
      values: top.map((d) => d.mean_activity),
    };
  }, [filtered]);

  const nCellTypes = useMemo(() => {
    if (!data) return 0;
    return new Set(data.map((d) => d.cell_type)).size;
  }, [data]);

  if (isLoading) return <Spinner message="Loading CytoSig activity..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load activity: {(error as Error).message}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <p className="py-8 text-center text-text-muted">
        No CytoSig activity data for {gene}
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <p className="text-xs text-text-muted">
          Activity = Z-score from ridge regression against 44-signature matrix
        </p>
        <p className="mt-1 text-sm text-text-secondary">
          {nCellTypes} cell types across {atlases.length} atlas{atlases.length !== 1 ? 'es' : ''}
        </p>
      </div>

      <FilterBar>
        <SelectFilter
          label="Atlas"
          options={atlasOptions}
          value={atlas}
          onChange={setAtlas}
        />
      </FilterBar>

      {barData && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Top 30 Cell Types by |Activity|
          </h3>
          <BarChart
            categories={barData.categories}
            values={barData.values}
            orientation="h"
            xTitle="Mean Activity (z-score)"
            yTitle="Cell Type"
            title={`${gene}: CytoSig Activity`}
            height={Math.max(400, barData.categories.length * 24 + 150)}
          />
        </div>
      )}

      {atlases.length > 1 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Activity Distribution by Atlas
          </h3>
          <ViolinChart
            groups={atlases}
            values={atlases.map((a) =>
              (data || []).filter((d) => d.atlas === a).map((d) => d.mean_activity),
            )}
            title={`${gene}: Activity by Atlas`}
            yTitle="Mean Activity (z-score)"
          />
        </div>
      )}
    </div>
  );
}
