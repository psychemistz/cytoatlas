import { useMemo } from 'react';
import { useSensitivityMatrix } from '@/api/hooks/use-perturbation';
import { HeatmapChart } from '@/components/charts/heatmap-chart';
import { Spinner } from '@/components/ui/loading-skeleton';
import type { DrugSensitivity } from '@/api/types/perturbation';

interface DrugSensitivityTabProps {
  signatureType: string;
}

const MAX_DRUGS = 50;
const MAX_CELL_LINES = 30;

/**
 * Filter to top drugs and cell lines by mean absolute activity,
 * then pivot into heatmap z/x/y format.
 */
function pivotSensitivity(data: DrugSensitivity[]) {
  // Compute mean absolute activity per drug
  const drugScores = new Map<string, { sum: number; count: number }>();
  for (const d of data) {
    const entry = drugScores.get(d.drug) ?? { sum: 0, count: 0 };
    entry.sum += Math.abs(d.activity_diff);
    entry.count += 1;
    drugScores.set(d.drug, entry);
  }
  const sortedDrugs = [...drugScores.entries()]
    .sort((a, b) => b[1].sum / b[1].count - a[1].sum / a[1].count)
    .slice(0, MAX_DRUGS)
    .map(([drug]) => drug);

  // Compute mean absolute activity per cell line
  const lineScores = new Map<string, { sum: number; count: number }>();
  for (const d of data) {
    const entry = lineScores.get(d.cell_line) ?? { sum: 0, count: 0 };
    entry.sum += Math.abs(d.activity_diff);
    entry.count += 1;
    lineScores.set(d.cell_line, entry);
  }
  const sortedLines = [...lineScores.entries()]
    .sort((a, b) => b[1].sum / b[1].count - a[1].sum / a[1].count)
    .slice(0, MAX_CELL_LINES)
    .map(([line]) => line);

  const drugSet = new Set(sortedDrugs);
  const lineSet = new Set(sortedLines);

  const lookup = new Map<string, number>();
  for (const d of data) {
    if (drugSet.has(d.drug) && lineSet.has(d.cell_line)) {
      lookup.set(`${d.drug}||${d.cell_line}`, d.activity_diff);
    }
  }

  const z: number[][] = sortedDrugs.map((drug) =>
    sortedLines.map((line) => lookup.get(`${drug}||${line}`) ?? 0),
  );

  return { z, x: sortedLines, y: sortedDrugs };
}

export default function DrugSensitivityTab({ signatureType }: DrugSensitivityTabProps) {
  const { data, isLoading, error } = useSensitivityMatrix(signatureType);

  const heatmap = useMemo(() => {
    if (!data || data.length === 0) return null;
    return pivotSensitivity(data);
  }, [data]);

  if (isLoading) return <Spinner message="Loading drug sensitivity matrix..." />;

  if (error) {
    return (
      <div className="rounded-lg border border-danger/20 bg-danger/5 p-6 text-center">
        <p className="text-sm text-danger">Failed to load drug sensitivity data</p>
      </div>
    );
  }

  if (!heatmap) {
    return (
      <div className="rounded-lg border border-border-light bg-bg-secondary p-8 text-center text-text-muted">
        No drug sensitivity data available
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold">Drug Sensitivity Matrix</h3>
        <p className="text-sm text-text-secondary">
          Activity difference between drug-treated and control cells (Tahoe-100M).
          Showing top {heatmap.y.length} drugs by {heatmap.x.length} cell lines ranked by mean absolute activity.
        </p>
      </div>
      <HeatmapChart
        z={heatmap.z}
        x={heatmap.x}
        y={heatmap.y}
        xTitle="Cell Line"
        yTitle="Drug"
        colorbarTitle="Delta Activity"
        symmetric
      />
    </div>
  );
}
