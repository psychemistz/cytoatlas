import { useMemo } from 'react';
import { useCytokineHeatmap } from '@/api/hooks/use-perturbation';
import { HeatmapChart } from '@/components/charts/heatmap-chart';
import { Spinner } from '@/components/ui/loading-skeleton';

interface CytokineResponseTabProps {
  signatureType: string;
}

/**
 * Pivot an array of {cytokine, cell_type, activity_diff} into heatmap z/x/y.
 * x = cell types, y = cytokines, z[row][col] = activity_diff
 */
function pivotToHeatmap(data: { cytokine: string; cell_type: string; activity_diff: number }[]) {
  const cellTypes = [...new Set(data.map((d) => d.cell_type))].sort();
  const cytokines = [...new Set(data.map((d) => d.cytokine))].sort();

  const lookup = new Map<string, number>();
  for (const d of data) {
    lookup.set(`${d.cytokine}||${d.cell_type}`, d.activity_diff);
  }

  const z: number[][] = cytokines.map((cyto) =>
    cellTypes.map((ct) => lookup.get(`${cyto}||${ct}`) ?? 0),
  );

  return { z, x: cellTypes, y: cytokines };
}

export default function CytokineResponseTab({ signatureType }: CytokineResponseTabProps) {
  const { data, isLoading, error } = useCytokineHeatmap(signatureType);

  const heatmap = useMemo(() => {
    if (!data || data.length === 0) return null;
    return pivotToHeatmap(data);
  }, [data]);

  if (isLoading) return <Spinner message="Loading cytokine response heatmap..." />;

  if (error) {
    return (
      <div className="rounded-lg border border-danger/20 bg-danger/5 p-6 text-center">
        <p className="text-sm text-danger">Failed to load cytokine response data</p>
      </div>
    );
  }

  if (!heatmap) {
    return (
      <div className="rounded-lg border border-border-light bg-bg-secondary p-8 text-center text-text-muted">
        No cytokine response data available
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold">Cytokine Response Heatmap</h3>
        <p className="text-sm text-text-secondary">
          Activity difference between cytokine-treated and control cells across cell types (parse_10M)
        </p>
      </div>
      <HeatmapChart
        z={heatmap.z}
        x={heatmap.x}
        y={heatmap.y}
        xTitle="Cell Type"
        yTitle="Cytokine"
        colorbarTitle="Delta Activity"
        symmetric
      />
    </div>
  );
}
