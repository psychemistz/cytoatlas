import { useMemo } from 'react';
import { useGeneDiseases } from '@/api/hooks/use-gene';
import { Spinner } from '@/components/ui/loading-skeleton';
import { BarChart } from '@/components/charts/bar-chart';
import { VolcanoChart } from '@/components/charts/volcano-chart';

interface DiseasesTabProps {
  gene: string;
  signatureType: string;
}

export default function DiseasesTab({ gene, signatureType }: DiseasesTabProps) {
  const { data, isLoading, error } = useGeneDiseases(gene, signatureType);

  const nSignificant = useMemo(() => {
    if (!data) return 0;
    return data.filter((d) => d.fdr != null && d.fdr < 0.05).length;
  }, [data]);

  const volcanoPoints = useMemo(() => {
    if (!data || data.length === 0) return null;
    return data.map((d) => ({
      label: d.disease,
      activity_diff: d.activity_diff,
      p_value: d.p_value,
      fdr: d.fdr,
    }));
  }, [data]);

  const barData = useMemo(() => {
    if (!data || data.length === 0) return null;
    const sorted = [...data].sort(
      (a, b) => Math.abs(b.activity_diff) - Math.abs(a.activity_diff),
    );
    return {
      categories: sorted.map((d) => d.disease),
      values: sorted.map((d) => d.activity_diff),
    };
  }, [data]);

  if (isLoading) return <Spinner message="Loading disease data..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load diseases: {(error as Error).message}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <p className="py-8 text-center text-text-muted">
        No disease association data for {gene}
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <p className="text-sm text-text-secondary">
        {data.length} diseases, {nSignificant} significant (FDR &lt; 0.05)
      </p>

      {volcanoPoints && volcanoPoints.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Volcano Plot
          </h3>
          <VolcanoChart
            points={volcanoPoints}
            title={`${gene}: Disease Volcano`}
            leftLabel={'\u2190 Lower in disease'}
            rightLabel={'Higher in disease \u2192'}
            height={450}
          />
        </div>
      )}

      {barData && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Disease Associations
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            {'\u0394'} Activity between disease and healthy groups.
          </p>
          <BarChart
            categories={barData.categories}
            values={barData.values}
            orientation="h"
            xTitle={'\u0394 Activity'}
            yTitle="Disease"
            title={`${gene}: Disease Associations`}
            height={Math.max(400, barData.categories.length * 28 + 150)}
            colors={barData.values.map(v => v >= 0 ? '#b2182b' : '#2166ac')}
          />
        </div>
      )}

      <div className="overflow-x-auto rounded-lg border border-border-light">
        <table className="w-full text-left text-sm">
          <thead className="bg-bg-secondary">
            <tr>
              <th className="px-3 py-2 font-medium">Disease</th>
              <th className="px-3 py-2 font-medium">Cohort</th>
              <th className="px-3 py-2 font-medium text-right">{'\u0394'} Activity</th>
              <th className="px-3 py-2 font-medium text-right">p-value</th>
              <th className="px-3 py-2 font-medium text-right">FDR</th>
              <th className="px-3 py-2 font-medium text-center">Sig</th>
            </tr>
          </thead>
          <tbody>
            {data.map((d, i) => (
              <tr key={i} className="border-t border-border-light">
                <td className="px-3 py-1.5">{d.disease}</td>
                <td className="px-3 py-1.5">{d.cohort}</td>
                <td className={`px-3 py-1.5 text-right font-mono ${d.activity_diff >= 0 ? 'text-red-600' : 'text-blue-600'}`}>
                  {d.activity_diff.toFixed(3)}
                </td>
                <td className="px-3 py-1.5 text-right font-mono">
                  {d.p_value != null ? d.p_value.toExponential(2) : '-'}
                </td>
                <td className="px-3 py-1.5 text-right font-mono">
                  {d.fdr != null ? (
                    <span className={d.fdr < 0.05 ? 'font-semibold' : ''}>
                      {d.fdr.toExponential(2)} {d.fdr < 0.01 ? '**' : d.fdr < 0.05 ? '*' : ''}
                    </span>
                  ) : '-'}
                </td>
                <td className="px-3 py-1.5 text-center">
                  {d.fdr != null && d.fdr < 0.05 ? (
                    <span className="text-green-600" title="FDR < 0.05">{'\u2713'}</span>
                  ) : ''}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
