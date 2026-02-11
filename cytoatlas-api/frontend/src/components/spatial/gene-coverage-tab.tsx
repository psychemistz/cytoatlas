import { useMemo } from 'react';
import { useGeneCoverage } from '@/api/hooks/use-spatial';
import { BarChart } from '@/components/charts/bar-chart';
import { Spinner } from '@/components/ui/loading-skeleton';

interface GeneCoverageTabProps {
  signatureType: string;
}

export default function GeneCoverageTab({ signatureType: _signatureType }: GeneCoverageTabProps) {
  const { data, isLoading, error } = useGeneCoverage();

  const chartData = useMemo(() => {
    if (!data || data.length === 0) return null;

    // Sort by CytoSig coverage descending
    const sorted = [...data].sort((a, b) => b.cytosig_coverage - a.cytosig_coverage);

    const technologies = sorted.map((d) => d.technology);
    const cytosigValues = sorted.map((d) => d.cytosig_coverage * 100);
    const secactValues = sorted.map((d) => d.secact_coverage * 100);

    return {
      categories: technologies,
      series: [
        { name: 'CytoSig Coverage', values: cytosigValues },
        { name: 'SecAct Coverage', values: secactValues },
      ],
    };
  }, [data]);

  if (isLoading) return <Spinner message="Loading gene coverage..." />;
  if (error) {
    return (
      <p className="py-8 text-center text-red-600">
        Failed to load gene coverage data: {(error as Error).message}
      </p>
    );
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="mb-2 text-sm font-semibold text-text-secondary">
          Signature Gene Panel Coverage by Technology
        </h3>
        <p className="mb-3 text-xs text-text-muted">
          Fraction of CytoSig (44 cytokines) and SecAct (1,249 secreted proteins)
          signature genes present in each spatial technology's gene panel. Higher
          coverage enables more reliable activity inference.
        </p>
      </div>

      {chartData ? (
        <BarChart
          categories={chartData.categories}
          series={chartData.series}
          xTitle="Technology"
          yTitle="Gene Panel Coverage (%)"
          barmode="group"
          height={500}
        />
      ) : (
        <p className="py-8 text-center text-text-muted">No gene coverage data available</p>
      )}
    </div>
  );
}
