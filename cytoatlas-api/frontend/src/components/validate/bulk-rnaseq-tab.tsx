import { useState, useMemo } from 'react';
import {
  useBulkRnaseqDatasets,
  useBulkRnaseqTargets,
  useBulkRnaseqScatter,
} from '@/api/hooks/use-validation';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { ScatterChart } from '@/components/charts/scatter-chart';
import { BarChart } from '@/components/charts/bar-chart';

interface BulkRnaseqTabProps {
  sigtype: string;
}

export default function BulkRnaseqTab({ sigtype }: BulkRnaseqTabProps) {
  const [dataset, setDataset] = useState('');
  const [target, setTarget] = useState('');

  const { data: datasets, isLoading: dsLoading } = useBulkRnaseqDatasets();
  const activeDataset = dataset || (datasets?.[0] ?? '');
  const { data: targets, isLoading: tgLoading } = useBulkRnaseqTargets(activeDataset, sigtype);
  const { data: scatter, isLoading: scLoading } = useBulkRnaseqScatter(
    activeDataset,
    target,
    sigtype,
  );

  const dsOptions = (datasets || []).map((d) => ({ value: d, label: d.toUpperCase() }));

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

  if (dsLoading) return <Spinner message="Loading datasets..." />;

  return (
    <div className="space-y-6">
      <FilterBar>
        <SelectFilter
          label="Dataset"
          options={dsOptions}
          value={activeDataset}
          onChange={(v) => {
            setDataset(v);
            setTarget('');
          }}
        />
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
            Top 30 Targets by Correlation ({activeDataset.toUpperCase()})
          </h3>
          <BarChart
            categories={barData.categories}
            values={barData.values}
            orientation="h"
            xTitle="Spearman rho"
            yTitle="Target"
            title={`Validation Correlations: ${activeDataset.toUpperCase()}`}
            height={Math.max(400, barData.categories.length * 24 + 150)}
          />
        </div>
      )}

      {target && scLoading && <Spinner message="Loading scatter..." />}

      {scatter && target && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            {target} — {activeDataset.toUpperCase()}
          </h3>
          <ScatterChart
            x={scatter.points.map((p) => p.x)}
            y={scatter.points.map((p) => p.y)}
            labels={scatter.points.map((p) => p.label || p.cell_type || '')}
            xTitle="Signature Gene Expression"
            yTitle="Predicted Activity"
            title={`${target}: Expression vs Activity`}
            showTrendLine
            stats={{ rho: scatter.rho, p: scatter.p_value }}
            height={500}
          />
        </div>
      )}

      {!target && !tgLoading && targets && targets.length === 0 && (
        <p className="py-8 text-center text-text-muted">
          No validation targets available for {activeDataset.toUpperCase()}
        </p>
      )}
    </div>
  );
}
