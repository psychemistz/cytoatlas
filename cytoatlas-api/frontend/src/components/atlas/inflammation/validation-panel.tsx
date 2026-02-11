import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import { Spinner } from '@/components/ui/loading-skeleton';
import { ScatterChart } from '@/components/charts/scatter-chart';

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

export default function ValidationPanel({ signatureType }: ValidationPanelProps) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['inflammation', 'cross-cohort-validation', signatureType],
    queryFn: () =>
      get<CrossCohortValidation>(
        '/atlases/inflammation/cross-cohort-validation',
        { signature_type: signatureType },
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

  return (
    <div className="space-y-6">
      <div>
        <h3 className="mb-2 text-sm font-semibold text-text-secondary">
          Cross-Cohort Validation
        </h3>
        <p className="mb-2 text-xs text-text-muted">
          Activity in the main cohort (x-axis) versus validation cohort
          (y-axis). Each point represents a signature. The trend line shows the
          linear fit.
        </p>
        <ScatterChart
          x={data.main}
          y={data.validation}
          labels={data.signatures}
          xTitle="Main Cohort Activity"
          yTitle="Validation Cohort Activity"
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
