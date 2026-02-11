import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import type { TreatmentPrediction } from '@/api/types/activity';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { BarChart } from '@/components/charts/bar-chart';

interface TreatmentPanelProps {
  signatureType: string;
}

export default function TreatmentPanel({ signatureType }: TreatmentPanelProps) {
  const [selectedTreatment, setSelectedTreatment] = useState<string>('');

  const { data, isLoading, error } = useQuery({
    queryKey: ['inflammation', 'treatment-predictions', signatureType],
    queryFn: () =>
      get<TreatmentPrediction[]>(
        '/atlases/inflammation/treatment-predictions',
        { signature_type: signatureType },
      ),
  });

  const sortedTreatments = useMemo(() => {
    if (!data) return [];
    return [...data].sort((a, b) => b.auroc - a.auroc);
  }, [data]);

  const treatmentOptions = useMemo(() => {
    return sortedTreatments.map((t) => ({
      value: t.treatment,
      label: `${t.treatment} (AUROC: ${t.auroc.toFixed(3)})`,
    }));
  }, [sortedTreatments]);

  const aurocBarData = useMemo(() => {
    if (!sortedTreatments.length) return null;
    return {
      categories: sortedTreatments.map((t) => t.treatment),
      values: sortedTreatments.map((t) => t.auroc),
    };
  }, [sortedTreatments]);

  const activeTreatment = selectedTreatment || sortedTreatments[0]?.treatment || '';

  const featureData = useMemo(() => {
    if (!data || !activeTreatment) return null;
    const pred = data.find((t) => t.treatment === activeTreatment);
    if (!pred || !pred.features || pred.features.length === 0) return null;

    const sorted = [...pred.features]
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 10);

    return {
      categories: sorted.map((f) => f.name),
      values: sorted.map((f) => f.importance),
    };
  }, [data, activeTreatment]);

  if (isLoading) return <Spinner message="Loading treatment predictions..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load treatment data: {(error as Error).message}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <p className="py-8 text-center text-text-muted">
        No treatment prediction data available
      </p>
    );
  }

  return (
    <div className="space-y-6">
      {aurocBarData && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Treatment Response Prediction (AUROC)
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            AUROC scores for predicting treatment response using cytokine activity
            signatures.
          </p>
          <BarChart
            categories={aurocBarData.categories}
            values={aurocBarData.values}
            orientation="h"
            xTitle="AUROC"
            yTitle="Treatment"
            title="Treatment Response Prediction"
            height={Math.max(400, aurocBarData.categories.length * 28 + 150)}
          />
        </div>
      )}

      <FilterBar>
        <SelectFilter
          label="Treatment"
          options={treatmentOptions}
          value={activeTreatment}
          onChange={setSelectedTreatment}
        />
      </FilterBar>

      {featureData && featureData.categories.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Top 10 Feature Importances for {activeTreatment}
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Signatures most predictive of response to {activeTreatment}.
          </p>
          <BarChart
            categories={featureData.categories}
            values={featureData.values}
            orientation="h"
            xTitle="Feature Importance"
            yTitle="Signature"
            title={`Feature Importance: ${activeTreatment}`}
            height={Math.max(350, featureData.categories.length * 28 + 150)}
          />
        </div>
      )}

      {activeTreatment && !featureData && (
        <p className="py-4 text-center text-sm text-text-muted">
          No feature importance data available for {activeTreatment}.
        </p>
      )}
    </div>
  );
}
