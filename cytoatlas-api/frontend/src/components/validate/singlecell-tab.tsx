import { useState, useMemo } from 'react';
import {
  useSingleCellSignatures,
  useSingleCellScatter,
  useSingleCellCelltypes,
} from '@/api/hooks/use-validation';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { ScatterChart } from '@/components/charts/scatter-chart';
import { BarChart } from '@/components/charts/bar-chart';

interface SinglecellTabProps {
  atlas: string;
  sigtype: string;
}

export default function SinglecellTab({ atlas, sigtype }: SinglecellTabProps) {
  const [target, setTarget] = useState('');

  const { data: signatures, isLoading: sigLoading } = useSingleCellSignatures(atlas, sigtype);
  const { data: scatter, isLoading: scLoading } = useSingleCellScatter(atlas, target, sigtype);
  const { data: celltypes, isLoading: ctLoading } = useSingleCellCelltypes(atlas, target, sigtype);

  const sigOptions = useMemo(() => {
    if (!signatures) return [];
    return signatures.map((s) => ({
      value: s.signature,
      label: s.rho != null ? `${s.signature} (rho=${s.rho.toFixed(3)})` : s.signature,
    }));
  }, [signatures]);

  const ctBarData = useMemo(() => {
    if (!celltypes) return null;
    const sorted = [...celltypes].sort((a, b) => (b.rho ?? 0) - (a.rho ?? 0));
    return {
      categories: sorted.map((c) => c.cell_type),
      values: sorted.map((c) => c.rho ?? 0),
    };
  }, [celltypes]);

  if (!atlas) {
    return (
      <p className="py-8 text-center text-text-muted">
        Select an atlas above to view single-cell validation.
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <FilterBar>
        {sigOptions.length > 0 && (
          <SelectFilter
            label="Signature"
            options={sigOptions}
            value={target}
            onChange={setTarget}
          />
        )}
      </FilterBar>

      {sigLoading && <Spinner message="Loading signatures..." />}

      {!target && signatures && signatures.length > 0 && (
        <p className="py-4 text-center text-sm text-text-muted">
          Select a signature above to view single-cell validation
        </p>
      )}

      {target && scLoading && <Spinner message="Loading scatter..." />}

      {scatter && target && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            {target} — Single-Cell Expression vs Activity
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Each point is a sampled single cell. Colored by cell type.
          </p>
          <ScatterChart
            x={scatter.points.map((p) => p.x)}
            y={scatter.points.map((p) => p.y)}
            labels={scatter.points.map((p) => p.cell_type || '')}
            xTitle="Expression"
            yTitle="Predicted Activity"
            title={`${target}: Single-Cell Validation`}
            showTrendLine
            stats={{ rho: scatter.rho, p: scatter.p_value }}
            height={500}
          />
        </div>
      )}

      {target && ctLoading && <Spinner message="Loading cell-type stats..." />}

      {ctBarData && target && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Per-Cell-Type Correlation for {target}
          </h3>
          <BarChart
            categories={ctBarData.categories}
            values={ctBarData.values}
            orientation="h"
            xTitle="Spearman rho"
            yTitle="Cell Type"
            title={`Cell-Type Correlations: ${target}`}
            height={Math.max(400, ctBarData.categories.length * 24 + 150)}
          />
        </div>
      )}
    </div>
  );
}
