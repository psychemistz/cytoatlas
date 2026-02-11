import { useMemo } from 'react';
import { usePathwayActivation } from '@/api/hooks/use-perturbation';
import { HeatmapChart } from '@/components/charts/heatmap-chart';
import { Spinner } from '@/components/ui/loading-skeleton';
import type { PathwayActivation } from '@/api/types/perturbation';

const MAX_DRUGS = 30;

/**
 * Filter to top drugs by mean absolute activation score,
 * then pivot into heatmap z/x/y format.
 */
function pivotPathways(data: PathwayActivation[]) {
  // Compute mean absolute activation per drug
  const drugScores = new Map<string, { sum: number; count: number }>();
  for (const d of data) {
    const entry = drugScores.get(d.drug) ?? { sum: 0, count: 0 };
    entry.sum += Math.abs(d.activation_score);
    entry.count += 1;
    drugScores.set(d.drug, entry);
  }
  const topDrugs = [...drugScores.entries()]
    .sort((a, b) => b[1].sum / b[1].count - a[1].sum / a[1].count)
    .slice(0, MAX_DRUGS)
    .map(([drug]) => drug);

  const drugSet = new Set(topDrugs);
  const pathways = [...new Set(data.map((d) => d.pathway))].sort();

  const lookup = new Map<string, number>();
  for (const d of data) {
    if (drugSet.has(d.drug)) {
      lookup.set(`${d.drug}||${d.pathway}`, d.activation_score);
    }
  }

  const z: number[][] = topDrugs.map((drug) =>
    pathways.map((pw) => lookup.get(`${drug}||${pw}`) ?? 0),
  );

  return { z, x: pathways, y: topDrugs };
}

export default function PathwayActivationTab() {
  const { data, isLoading, error } = usePathwayActivation();

  const heatmap = useMemo(() => {
    if (!data || data.length === 0) return null;
    return pivotPathways(data);
  }, [data]);

  if (isLoading) return <Spinner message="Loading pathway activation data..." />;

  if (error) {
    return (
      <div className="rounded-lg border border-danger/20 bg-danger/5 p-6 text-center">
        <p className="text-sm text-danger">Failed to load pathway activation data</p>
      </div>
    );
  }

  if (!heatmap) {
    return (
      <div className="rounded-lg border border-border-light bg-bg-secondary p-8 text-center text-text-muted">
        No pathway activation data available
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold">Pathway Activation</h3>
        <p className="text-sm text-text-secondary">
          Drug-induced pathway activation scores across top {heatmap.y.length} drugs (Tahoe-100M)
        </p>
      </div>
      <HeatmapChart
        z={heatmap.z}
        x={heatmap.x}
        y={heatmap.y}
        xTitle="Pathway"
        yTitle="Drug"
        colorbarTitle="Activation Score"
        symmetric
      />
    </div>
  );
}
