import { useState, useMemo, useEffect } from 'react';
import {
  useCelltypeLevels,
  useCelltypeTargets,
  useCelltypeScatter,
} from '@/api/hooks/use-validation';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { ScatterChart } from '@/components/charts/scatter-chart';
import { BarChart } from '@/components/charts/bar-chart';

interface CelltypeLevelTabProps {
  atlas: string;
  sigtype: string;
}

export default function CelltypeLevelTab({ atlas, sigtype }: CelltypeLevelTabProps) {
  const [level, setLevel] = useState('');
  const [target, setTarget] = useState('');

  const { data: levels } = useCelltypeLevels(atlas);
  const { data: targets, isLoading: tgLoading } = useCelltypeTargets(atlas, sigtype, level);
  const { data: scatter, isLoading: scLoading } = useCelltypeScatter(
    atlas,
    target,
    sigtype,
    level,
  );

  useEffect(() => {
    if (levels && levels.length > 0 && !level) {
      setLevel(levels[0]);
    }
  }, [levels, level]);

  const levelOptions = (levels || []).map((l) => ({ value: l, label: l }));

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
        Select an atlas above to view cell-type level validation.
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <FilterBar>
        {levelOptions.length > 0 && (
          <SelectFilter
            label="Level"
            options={levelOptions}
            value={level}
            onChange={(v) => {
              setLevel(v);
              setTarget('');
            }}
          />
        )}
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
            Top 30 Cell-Type Level Correlations ({atlas}, {level})
          </h3>
          <BarChart
            categories={barData.categories}
            values={barData.values}
            orientation="h"
            xTitle="Spearman rho"
            yTitle="Target"
            title={`Cell-Type Validation: ${atlas} (${level})`}
            height={Math.max(400, barData.categories.length * 24 + 150)}
          />
        </div>
      )}

      {target && scLoading && <Spinner message="Loading scatter..." />}

      {scatter && target && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            {target} — Cell-Type Level ({atlas}, {level})
          </h3>
          <ScatterChart
            x={scatter.points.map((p) => p.x)}
            y={scatter.points.map((p) => p.y)}
            labels={scatter.points.map((p) => p.label || p.cell_type || '')}
            xTitle="Signature Gene Expression"
            yTitle="Predicted Activity"
            title={`${target}: Cell-Type Level (${level})`}
            showTrendLine
            stats={{ rho: scatter.rho, p: scatter.p_value }}
            height={500}
          />
        </div>
      )}
    </div>
  );
}
