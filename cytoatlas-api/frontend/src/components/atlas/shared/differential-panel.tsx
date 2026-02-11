import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import type { DifferentialData } from '@/api/types/activity';
import { Spinner } from '@/components/ui/loading-skeleton';
import { VolcanoChart } from '@/components/charts/volcano-chart';
import { BarChart } from '@/components/charts/bar-chart';

interface DifferentialPanelProps {
  signatureType: string;
  atlasName: string;
  context: 'population' | 'disease' | 'cancer';
}

const ENDPOINT_MAP: Record<DifferentialPanelProps['context'], string> = {
  population: '/atlases/cima/population-stratification',
  disease: '/atlases/inflammation/differential',
  cancer: '/atlases/scatlas/differential',
};

const TITLE_MAP: Record<DifferentialPanelProps['context'], string> = {
  population: 'Population Stratification',
  disease: 'Disease Differential Activity',
  cancer: 'Cancer Differential Activity',
};

export default function DifferentialPanel({
  signatureType,
  context,
}: DifferentialPanelProps) {
  const endpoint = ENDPOINT_MAP[context];

  const { data, isLoading, error } = useQuery({
    queryKey: ['differential', context, signatureType],
    queryFn: () =>
      get<DifferentialData[]>(endpoint, { signature_type: signatureType }),
  });

  const volcanoPoints = useMemo(() => {
    if (!data) return [];
    return data.map((d) => ({
      signature: d.signature,
      activity_diff: d.activity_diff,
      p_value: d.p_value,
      fdr: d.fdr,
    }));
  }, [data]);

  const barData = useMemo(() => {
    if (!data) return null;
    const scored = data.map((d) => ({
      signature: d.signature,
      score: Math.abs(d.activity_diff) * -Math.log10(Math.max(d.p_value, 1e-300)),
    }));
    const sorted = [...scored].sort((a, b) => b.score - a.score).slice(0, 20);
    return {
      categories: sorted.map((d) => d.signature),
      values: sorted.map((d) => d.score),
    };
  }, [data]);

  if (isLoading) return <Spinner message="Loading differential data..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load differential data: {(error as Error).message}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return <p className="py-8 text-center text-text-muted">No differential data available</p>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="mb-2 text-sm font-semibold text-text-secondary">
          {TITLE_MAP[context]} -- Volcano Plot
        </h3>
        <VolcanoChart
          points={volcanoPoints}
          title={`${TITLE_MAP[context]}: \u0394 Activity vs Significance`}
          fdrThreshold={0.05}
          activityThreshold={0.5}
        />
      </div>

      {barData && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Top 20 Signatures by Significance Score
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Score = |{'\u0394'} Activity| x -log10(p-value)
          </p>
          <BarChart
            categories={barData.categories}
            values={barData.values}
            orientation="h"
            xTitle="Significance Score"
            yTitle="Signature"
            title="Top 20 Significant Signatures"
            height={Math.max(500, barData.categories.length * 24 + 150)}
          />
        </div>
      )}
    </div>
  );
}
