import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import { Spinner } from '@/components/ui/loading-skeleton';
import { ScatterChart } from '@/components/charts/scatter-chart';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';

interface ValidationPanelProps {
  signatureType: string;
}

interface CrossCohortValidation {
  main: number[];
  validation: number[];
  signatures: string[];
  rho: number;
  p_value: number;
}

const COMPARISON_OPTIONS = [
  { value: 'main_vs_validation', label: 'Main vs Validation' },
  { value: 'main_vs_external', label: 'Main vs External' },
  { value: 'validation_vs_external', label: 'Validation vs External' },
];

const COMPARISON_LABELS: Record<string, { x: string; y: string }> = {
  main_vs_validation: { x: 'Main Cohort', y: 'Validation Cohort' },
  main_vs_external: { x: 'Main Cohort', y: 'External Cohort' },
  validation_vs_external: { x: 'Validation Cohort', y: 'External Cohort' },
};

export default function ValidationPanel({ signatureType }: ValidationPanelProps) {
  const [comparison, setComparison] = useState('main_vs_validation');

  const { data, isLoading, error } = useQuery({
    queryKey: ['inflammation', 'cross-cohort-validation', signatureType, comparison],
    queryFn: () =>
      get<CrossCohortValidation>(
        '/atlases/inflammation/cross-cohort-validation',
        { signature_type: signatureType, comparison },
      ),
  });

  if (isLoading) return <Spinner message="Loading validation data..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load validation data: {(error as Error).message}
      </div>
    );
  }

  if (!data || !data.main || data.main.length === 0) {
    return (
      <p className="py-8 text-center text-text-muted">
        No cross-cohort validation data available
      </p>
    );
  }

  const axisLabels = COMPARISON_LABELS[comparison] ?? COMPARISON_LABELS.main_vs_validation;

  return (
    <div className="space-y-6">
      <FilterBar>
        <SelectFilter
          label="Cohort Comparison"
          options={COMPARISON_OPTIONS}
          value={comparison}
          onChange={setComparison}
        />
      </FilterBar>

      <div>
        <h3 className="mb-2 text-sm font-semibold text-text-secondary">
          Cross-Cohort Validation
        </h3>
        <p className="mb-2 text-xs text-text-muted">
          Activity in the {axisLabels.x.toLowerCase()} (x-axis) versus{' '}
          {axisLabels.y.toLowerCase()} (y-axis). Each point represents a
          signature. The trend line shows the linear fit.
        </p>
        <ScatterChart
          x={data.main}
          y={data.validation}
          labels={data.signatures}
          xTitle={`${axisLabels.x} Activity`}
          yTitle={`${axisLabels.y} Activity`}
          title="Cross-Cohort Validation"
          showTrendLine
          stats={{ rho: data.rho, p: data.p_value }}
          height={550}
        />
      </div>

      <div className="rounded-md border border-border-light bg-bg-secondary p-4">
        <h4 className="mb-2 text-sm font-semibold text-text-secondary">
          Validation Statistics
        </h4>
        <dl className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <dt className="text-text-muted">Spearman rho</dt>
            <dd className="font-mono font-semibold text-text-primary">
              {data.rho.toFixed(4)}
            </dd>
          </div>
          <div>
            <dt className="text-text-muted">p-value</dt>
            <dd className="font-mono font-semibold text-text-primary">
              {data.p_value.toExponential(2)}
            </dd>
          </div>
          <div>
            <dt className="text-text-muted">Signatures</dt>
            <dd className="font-mono font-semibold text-text-primary">
              {data.signatures.length}
            </dd>
          </div>
        </dl>
      </div>
    </div>
  );
}
