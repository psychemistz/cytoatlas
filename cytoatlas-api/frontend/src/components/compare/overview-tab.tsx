import { useMemo } from 'react';
import { useCrossAtlasSummary } from '@/api/hooks/use-cross-atlas';
import { Spinner } from '@/components/ui/loading-skeleton';
import { StatCard } from '@/components/ui/stat-card';
import { BarChart } from '@/components/charts/bar-chart';
import { formatNumber } from '@/lib/utils';

interface OverviewTabProps {
  signatureType: string;
}

export default function OverviewTab({ signatureType: _signatureType }: OverviewTabProps) {
  const { data, isLoading, error } = useCrossAtlasSummary();

  const summary = data as
    | {
        total_cells?: number;
        total_samples?: number;
        total_cell_types?: number;
        n_signatures?: number;
        atlases?: Record<
          string,
          { cells: number; samples: number; cell_types: number; focus: string }
        >;
      }
    | undefined;

  const atlasBarData = useMemo(() => {
    if (!summary?.atlases) return null;
    const names = Object.keys(summary.atlases);
    return {
      cells: {
        categories: names,
        values: names.map((n) => summary.atlases![n].cells),
      },
      samples: {
        categories: names,
        values: names.map((n) => summary.atlases![n].samples),
      },
    };
  }, [summary]);

  if (isLoading) return <Spinner message="Loading cross-atlas summary..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load summary: {(error as Error).message}
      </div>
    );
  }

  if (!summary) {
    return <p className="py-8 text-center text-text-muted">No summary data available</p>;
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard label="Total Cells" value={formatNumber(summary.total_cells ?? 0)} />
        <StatCard label="Samples" value={String(summary.total_samples ?? 0)} />
        <StatCard label="Cell Types" value={String(summary.total_cell_types ?? 0)} />
        <StatCard label="Signatures" value={String(summary.n_signatures ?? 0)} />
      </div>

      {atlasBarData && (
        <div className="grid gap-6 md:grid-cols-2">
          <div>
            <h3 className="mb-2 text-sm font-semibold text-text-secondary">
              Cells per Atlas
            </h3>
            <BarChart
              categories={atlasBarData.cells.categories}
              values={atlasBarData.cells.values}
              title="Cells per Atlas"
              yTitle="Number of Cells"
            />
          </div>
          <div>
            <h3 className="mb-2 text-sm font-semibold text-text-secondary">
              Samples per Atlas
            </h3>
            <BarChart
              categories={atlasBarData.samples.categories}
              values={atlasBarData.samples.values}
              title="Samples per Atlas"
              yTitle="Number of Samples"
            />
          </div>
        </div>
      )}
    </div>
  );
}
