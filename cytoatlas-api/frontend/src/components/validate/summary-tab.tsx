import { useMemo } from 'react';
import { useSummaryBoxplot, useMethodComparison } from '@/api/hooks/use-validation';
import { Spinner } from '@/components/ui/loading-skeleton';
import { BoxplotChart } from '@/components/charts/boxplot-chart';

interface SummaryTabProps {
  sigtype: string;
}

export default function SummaryTab({ sigtype }: SummaryTabProps) {
  const { data: boxplotData, isLoading: boxLoading } = useSummaryBoxplot(sigtype);
  const { data: methodData, isLoading: methodLoading } = useMethodComparison();

  const boxplotGroups = useMemo(() => {
    if (!boxplotData?.rhos) return null;
    const groups: string[] = [];
    const values: number[][] = [];
    for (const [target, catMap] of Object.entries(boxplotData.rhos)) {
      for (const [cat, rhos] of Object.entries(catMap)) {
        groups.push(`${target} (${cat})`);
        values.push(rhos);
      }
    }
    return { groups, values };
  }, [boxplotData]);

  const methodGroups = useMemo(() => {
    if (!methodData?.rhos) return null;
    const groups: string[] = [];
    const values: number[][] = [];
    for (const [method, catMap] of Object.entries(methodData.rhos)) {
      for (const [cat, rhos] of Object.entries(catMap)) {
        groups.push(`${method} (${cat})`);
        values.push(rhos);
      }
    }
    return { groups, values };
  }, [methodData]);

  if (boxLoading || methodLoading) return <Spinner message="Loading summary..." />;

  return (
    <div className="space-y-8">
      <div>
        <h3 className="mb-2 text-sm font-semibold text-text-secondary">
          Validation Correlation Distribution
        </h3>
        <p className="mb-3 text-xs text-text-muted">
          Distribution of Spearman rho values between predicted activity and
          signature gene expression across validation targets and levels.
        </p>
        {boxplotGroups ? (
          <BoxplotChart
            groups={boxplotGroups.groups}
            values={boxplotGroups.values}
            title="Validation Correlations"
            yTitle="Spearman rho"
          />
        ) : (
          <p className="py-8 text-center text-text-muted">No summary data available</p>
        )}
      </div>

      <div>
        <h3 className="mb-2 text-sm font-semibold text-text-secondary">
          Method Comparison
        </h3>
        <p className="mb-3 text-xs text-text-muted">
          Comparison of CytoSig, LinCytoSig, and SecAct validation performance.
        </p>
        {methodGroups ? (
          <BoxplotChart
            groups={methodGroups.groups}
            values={methodGroups.values}
            title="Method Comparison"
            yTitle="Spearman rho"
          />
        ) : (
          <p className="py-8 text-center text-text-muted">No method comparison data available</p>
        )}
      </div>
    </div>
  );
}
