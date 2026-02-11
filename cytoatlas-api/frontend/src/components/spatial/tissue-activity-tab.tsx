import { useMemo } from 'react';
import { useTissueActivity } from '@/api/hooks/use-spatial';
import { HeatmapChart } from '@/components/charts/heatmap-chart';
import { Spinner } from '@/components/ui/loading-skeleton';

interface TissueActivityTabProps {
  signatureType: string;
}

export default function TissueActivityTab({ signatureType }: TissueActivityTabProps) {
  const { data, isLoading, error } = useTissueActivity(signatureType);

  const heatmapData = useMemo(() => {
    if (!data || data.length === 0) return null;

    // Compute mean absolute activity per signature for ranking
    const sigAbsMean = new Map<string, { sum: number; count: number }>();
    for (const row of data) {
      const entry = sigAbsMean.get(row.signature) ?? { sum: 0, count: 0 };
      entry.sum += Math.abs(row.mean_activity);
      entry.count += 1;
      sigAbsMean.set(row.signature, entry);
    }

    // Sort signatures by mean absolute activity and take top 30
    const rankedSignatures = Array.from(sigAbsMean.entries())
      .map(([sig, { sum, count }]) => ({ sig, avg: sum / count }))
      .sort((a, b) => b.avg - a.avg)
      .slice(0, 30)
      .map((s) => s.sig);

    const signatureSet = new Set(rankedSignatures);

    // Collect unique tissues
    const tissueSet = new Set<string>();
    for (const row of data) {
      tissueSet.add(row.tissue);
    }
    const tissues = Array.from(tissueSet).sort();

    // Build lookup map for fast access
    const lookup = new Map<string, number>();
    for (const row of data) {
      if (signatureSet.has(row.signature)) {
        lookup.set(`${row.tissue}__${row.signature}`, row.mean_activity);
      }
    }

    // Build z matrix: rows = tissues, cols = signatures
    const z: number[][] = tissues.map((tissue) =>
      rankedSignatures.map((sig) => lookup.get(`${tissue}__${sig}`) ?? 0),
    );

    return { z, x: rankedSignatures, y: tissues };
  }, [data]);

  if (isLoading) return <Spinner message="Loading tissue activity..." />;
  if (error) {
    return (
      <p className="py-8 text-center text-red-600">
        Failed to load tissue activity data: {(error as Error).message}
      </p>
    );
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="mb-2 text-sm font-semibold text-text-secondary">
          Tissue-Level Activity Heatmap
        </h3>
        <p className="mb-3 text-xs text-text-muted">
          Mean activity z-scores across tissues for the top 30 signatures ranked by
          absolute activity. Signatures from the {signatureType} matrix.
        </p>
      </div>

      {heatmapData ? (
        <HeatmapChart
          z={heatmapData.z}
          x={heatmapData.x}
          y={heatmapData.y}
          xTitle="Signature"
          yTitle="Tissue"
          colorbarTitle="Mean Activity"
          symmetric
        />
      ) : (
        <p className="py-8 text-center text-text-muted">No tissue activity data available</p>
      )}
    </div>
  );
}
