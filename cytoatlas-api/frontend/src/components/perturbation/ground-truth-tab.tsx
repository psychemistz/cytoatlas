import { useMemo } from 'react';
import { useGroundTruth } from '@/api/hooks/use-perturbation';
import { ScatterChart } from '@/components/charts/scatter-chart';
import { Spinner } from '@/components/ui/loading-skeleton';
import { COLORS } from '@/components/charts/chart-defaults';

interface GroundTruthTabProps {
  signatureType: string;
}

function computeSpearman(x: number[], y: number[]): { rho: number; n: number } {
  const n = x.length;
  if (n < 3) return { rho: 0, n };

  function rank(arr: number[]): number[] {
    const sorted = arr.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
    const ranks = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      ranks[sorted[i].i] = i + 1;
    }
    return ranks;
  }

  const rx = rank(x);
  const ry = rank(y);
  const d2 = rx.reduce((sum, r, i) => sum + (r - ry[i]) ** 2, 0);
  const rho = 1 - (6 * d2) / (n * (n * n - 1));
  return { rho, n };
}

export default function GroundTruthTab({ signatureType }: GroundTruthTabProps) {
  const { data, isLoading, error } = useGroundTruth(signatureType);

  const { selfData, otherData, stats } = useMemo(() => {
    if (!data || data.length === 0) {
      return { selfData: null, otherData: null, stats: null };
    }

    const selfPoints = data.filter((d) => d.is_self_signature);
    const otherPoints = data.filter((d) => !d.is_self_signature);

    const allX = data.map((d) => d.actual_response);
    const allY = data.map((d) => d.predicted_activity);
    const { rho } = computeSpearman(allX, allY);

    return {
      selfData: {
        x: selfPoints.map((d) => d.actual_response),
        y: selfPoints.map((d) => d.predicted_activity),
        labels: selfPoints.map((d) => `${d.cytokine} - ${d.cell_type}`),
      },
      otherData: {
        x: otherPoints.map((d) => d.actual_response),
        y: otherPoints.map((d) => d.predicted_activity),
        labels: otherPoints.map((d) => `${d.cytokine} - ${d.cell_type}`),
      },
      stats: { rho, n: data.length },
    };
  }, [data]);

  if (isLoading) return <Spinner message="Loading ground truth data..." />;

  if (error) {
    return (
      <div className="rounded-lg border border-danger/20 bg-danger/5 p-6 text-center">
        <p className="text-sm text-danger">Failed to load ground truth data</p>
      </div>
    );
  }

  if (!selfData || !otherData || !stats) {
    return (
      <div className="rounded-lg border border-border-light bg-bg-secondary p-8 text-center text-text-muted">
        No ground truth data available
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold">Ground Truth Validation</h3>
        <p className="text-sm text-text-secondary">
          Predicted activity vs actual treatment response. Red points indicate self-signature
          matches (e.g., IL-6 treatment predicted by IL-6 signature). N = {stats.n}
        </p>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <ScatterChart
          x={selfData.x}
          y={selfData.y}
          labels={selfData.labels}
          colors={COLORS.red}
          xTitle="Actual Response"
          yTitle="Predicted Activity"
          title="Self-Signature Matches"
          showTrendLine
          stats={{ rho: stats.rho }}
        />
        <ScatterChart
          x={otherData.x}
          y={otherData.y}
          labels={otherData.labels}
          colors={COLORS.primary}
          xTitle="Actual Response"
          yTitle="Predicted Activity"
          title="Non-Self Signatures"
          showTrendLine
        />
      </div>
    </div>
  );
}
