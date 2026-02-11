import { useState, useMemo } from 'react';
import { useDonorTargets, useDonorScatter } from '@/api/hooks/use-validation';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { ScatterChart } from '@/components/charts/scatter-chart';
import { BarChart } from '@/components/charts/bar-chart';

interface DonorLevelTabProps {
  atlas: string;
  sigtype: string;
}

export default function DonorLevelTab({ atlas, sigtype }: DonorLevelTabProps) {
  const [target, setTarget] = useState('');
  const { data: targets, isLoading: tgLoading } = useDonorTargets(atlas, sigtype);
  const { data: scatter, isLoading: scLoading } = useDonorScatter(atlas, target, sigtype);

  const targetOptions = useMemo(() => {
    if (!targets) return [];
    return targets
      .sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho))
      .slice(0, 100)
      .map((t) => ({
        value: t.target,
        label: `${t.target} (rho=${t.rho.toFixed(3)})`,
      }));
  }, [targets]);

  const barData = useMemo(() => {
    if (!targets) return null;
    const sorted = [...targets].sort((a, b) => b.rho - a.rho).slice(0, 30);
    return {
      categories: sorted.map((t) => t.target),
      values: sorted.map((t) => t.rho),
    };
  }, [targets]);

  if (!atlas) {
    return (
      <p className="py-8 text-center text-text-muted">
        Select an atlas above to view donor-level validation.
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <FilterBar>
        {targetOptions.length > 0 && (
          <SelectFilter
            label="Target"
            options={targetOptions}
            value={target}
            onChange={setTarget}
          />
        )}
      </FilterBar>

      {tgLoading && <Spinner message="Loading targets..." />}

      {barData && !target && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Top 30 Donor-Level Correlations ({atlas})
          </h3>
          <BarChart
            categories={barData.categories}
            values={barData.values}
            orientation="h"
            xTitle="Spearman rho"
            yTitle="Target"
            title={`Donor-Level Validation: ${atlas}`}
            height={Math.max(400, barData.categories.length * 24 + 150)}
          />
        </div>
      )}

      {target && scLoading && <Spinner message="Loading scatter..." />}

      {scatter && target && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            {target} — Donor Level ({atlas})
          </h3>
          <ScatterChart
            x={scatter.points.map((p) => p.x)}
            y={scatter.points.map((p) => p.y)}
            labels={scatter.points.map((p) => p.label || p.cell_type || '')}
            xTitle="Signature Gene Expression"
            yTitle="Predicted Activity"
            title={`${target}: Donor-Level Validation`}
            showTrendLine
            stats={{ rho: scatter.rho, p: scatter.p_value }}
            height={500}
          />
        </div>
      )}
    </div>
  );
}
