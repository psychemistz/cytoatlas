import { useState, useMemo } from 'react';
import { useSummaryBoxplot, useMethodComparison } from '@/api/hooks/use-validation';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { BoxplotChart } from '@/components/charts/boxplot-chart';

/**
 * Summary tab — loads BOTH CytoSig and SecAct data for side-by-side comparison.
 * Method Comparison shows CytoSig vs LinCytoSig vs SecAct.
 */
export default function SummaryTab() {
  const [selectedTarget, setSelectedTarget] = useState('_all');
  const { data: cytosigData, isLoading: csLoading } = useSummaryBoxplot('cytosig');
  const { data: secactData, isLoading: saLoading } = useSummaryBoxplot('secact');
  const { data: methodData, isLoading: methodLoading } = useMethodComparison();

  const targetOptions = useMemo(() => {
    const targets = new Set<string>();
    if (cytosigData?.targets) cytosigData.targets.forEach((t) => targets.add(t));
    if (secactData?.targets) secactData.targets.forEach((t) => targets.add(t));
    return [
      { value: '_all', label: 'All Targets' },
      ...[...targets].sort().map((t) => ({ value: t, label: t })),
    ];
  }, [cytosigData, secactData]);

  // Combined CytoSig + SecAct boxplot (side-by-side)
  const combinedBoxplot = useMemo(() => {
    if (!cytosigData?.rhos && !secactData?.rhos) return null;

    const groups: string[] = [];
    const values: number[][] = [];

    // Collect category keys from whichever dataset has more
    const csRhos = cytosigData?.rhos ?? {};
    const saRhos = secactData?.rhos ?? {};
    const allCats = new Set<string>();
    for (const catMap of Object.values(csRhos)) {
      for (const cat of Object.keys(catMap)) allCats.add(cat);
    }
    for (const catMap of Object.values(saRhos)) {
      for (const cat of Object.keys(catMap)) allCats.add(cat);
    }
    const catKeys = [...allCats].sort();

    for (const cat of catKeys) {
      const catLabel = cat.replace(/_/g, ' ');

      // CytoSig values for this category
      const csVals: number[] = [];
      if (selectedTarget === '_all') {
        for (const tgtRhos of Object.values(csRhos)) {
          const vals = tgtRhos[cat];
          if (Array.isArray(vals)) csVals.push(...vals);
          else if (vals != null) csVals.push(vals);
        }
      } else {
        const tgtRhos = csRhos[selectedTarget];
        if (tgtRhos) {
          const vals = tgtRhos[cat];
          if (Array.isArray(vals)) csVals.push(...vals);
          else if (vals != null) csVals.push(vals);
        }
      }

      // SecAct values for this category
      const saVals: number[] = [];
      if (selectedTarget === '_all') {
        for (const tgtRhos of Object.values(saRhos)) {
          const vals = tgtRhos[cat];
          if (Array.isArray(vals)) saVals.push(...vals);
          else if (vals != null) saVals.push(vals);
        }
      } else {
        const tgtRhos = saRhos[selectedTarget];
        if (tgtRhos) {
          const vals = tgtRhos[cat];
          if (Array.isArray(vals)) saVals.push(...vals);
          else if (vals != null) saVals.push(vals);
        }
      }

      if (csVals.length > 0) {
        groups.push(`CytoSig — ${catLabel}`);
        values.push(csVals);
      }
      if (saVals.length > 0) {
        groups.push(`SecAct — ${catLabel}`);
        values.push(saVals);
      }
    }

    return groups.length > 0 ? { groups, values } : null;
  }, [cytosigData, secactData, selectedTarget]);

  // Method comparison: CytoSig vs LinCytoSig vs SecAct
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
    return groups.length > 0 ? { groups, values } : null;
  }, [methodData]);

  if (csLoading || saLoading || methodLoading) return <Spinner message="Loading summary..." />;

  return (
    <div className="space-y-8">
      <div>
        <h3 className="mb-2 text-sm font-semibold text-text-secondary">
          Correlation Distributions — CytoSig vs SecAct
        </h3>
        <p className="mb-3 text-xs text-text-muted">
          Distribution of Spearman rho values between predicted activity and
          signature gene expression. CytoSig (43 cytokines) and SecAct (1,249
          secreted proteins) are shown side-by-side.
        </p>

        {targetOptions.length > 1 && (
          <FilterBar className="mb-4">
            <SelectFilter
              label="Target"
              options={targetOptions}
              value={selectedTarget}
              onChange={setSelectedTarget}
            />
          </FilterBar>
        )}

        {combinedBoxplot ? (
          <BoxplotChart
            groups={combinedBoxplot.groups}
            values={combinedBoxplot.values}
            title={selectedTarget === '_all' ? 'All Targets' : selectedTarget}
            yTitle="Spearman rho"
          />
        ) : (
          <p className="py-8 text-center text-text-muted">No summary data available</p>
        )}
      </div>

      <div>
        <h3 className="mb-2 text-sm font-semibold text-text-secondary">
          Method Comparison — CytoSig vs LinCytoSig vs SecAct
        </h3>
        <p className="mb-3 text-xs text-text-muted">
          Cell-type-level Spearman rho (all donors pooled) across single-cell
          atlases.
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
