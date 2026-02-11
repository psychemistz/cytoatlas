import { useState, useMemo, useEffect } from 'react';
import {
  useSingleCellSignatures,
  useSingleCellScatter,
  useSingleCellCelltypes,
} from '@/api/hooks/use-validation';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter, CheckboxFilter } from '@/components/ui/filter-bar';
import { SearchableSelect } from '@/components/ui/searchable-select';
import { ScatterChart } from '@/components/charts/scatter-chart';
import { BarChart } from '@/components/charts/bar-chart';
import { BoxplotChart } from '@/components/charts/boxplot-chart';

const SIG_OPTIONS = [
  { value: 'cytosig', label: 'CytoSig (43)' },
  { value: 'lincytosig', label: 'LinCytoSig' },
  { value: 'secact', label: 'SecAct (1,249)' },
];

interface SinglecellTabProps {
  atlas: string;
}

export default function SinglecellTab({ atlas }: SinglecellTabProps) {
  const [sigtype, setSigtype] = useState('cytosig');
  const [target, setTarget] = useState('');
  const [cellType, setCellType] = useState('');
  const [hideNonExpr, setHideNonExpr] = useState(false);

  const { data: signatures, isLoading: sigLoading } = useSingleCellSignatures(atlas, sigtype);
  const { data: scatter, isLoading: scLoading } = useSingleCellScatter(atlas, target, sigtype);
  const { data: celltypes, isLoading: ctLoading } = useSingleCellCelltypes(atlas, target, sigtype);

  // Auto-select first signature when signatures load
  useEffect(() => {
    if (!signatures || signatures.length === 0) return;
    if (!target) {
      setTarget(signatures[0].signature);
    } else {
      const available = new Set(signatures.map((s) => s.signature));
      if (!available.has(target)) {
        const match = signatures.find((s) => s.signature.toLowerCase() === target.toLowerCase());
        setTarget(match ? match.signature : signatures[0].signature);
        setCellType('');
      }
    }
  }, [signatures]); // eslint-disable-line react-hooks/exhaustive-deps

  const sigOptions = useMemo(() => {
    if (!signatures) return [];
    return signatures.map((s) => ({
      value: s.signature,
      label: s.rho != null ? `${s.signature} (rho=${s.rho.toFixed(3)})` : s.signature,
    }));
  }, [signatures]);

  const cellTypeOptions = useMemo(() => {
    if (!scatter?.points) return [];
    const unique = [...new Set(scatter.points.map((p) => p.cell_type || ''))].filter(Boolean);
    unique.sort((a, b) => a.localeCompare(b));
    return [
      { value: '', label: 'All' },
      ...unique.map((ct) => ({ value: ct, label: ct })),
    ];
  }, [scatter]);

  const filteredPoints = useMemo(() => {
    if (!scatter?.points) return [];
    let pts = scatter.points;
    if (cellType) {
      pts = pts.filter((p) => p.cell_type === cellType);
    }
    if (hideNonExpr) {
      pts = pts.filter((p) => p.x > 0);
    }
    return pts;
  }, [scatter, cellType, hideNonExpr]);

  const boxData = useMemo(() => {
    if (!scatter?.points) return null;
    const expressing = scatter.points.filter((p) => p.x > 0).map((p) => p.y);
    const nonExpressing = scatter.points.filter((p) => p.x <= 0).map((p) => p.y);
    return {
      groups: ['Expressing', 'Non-expressing'],
      values: [expressing, nonExpressing],
    };
  }, [scatter]);

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
        <SelectFilter
          label="Signature"
          options={SIG_OPTIONS}
          value={sigtype}
          onChange={(v) => { setSigtype(v); }}
        />
        {sigOptions.length > 0 && (
          <SearchableSelect
            label="Target"
            options={sigOptions}
            value={target}
            onChange={setTarget}
            placeholder="Search signatures..."
          />
        )}
        {cellTypeOptions.length > 0 && target && (
          <SelectFilter
            label="Cell Type"
            options={cellTypeOptions}
            value={cellType}
            onChange={setCellType}
          />
        )}
        {target && scatter && (
          <CheckboxFilter
            label="Hide non-expressing"
            checked={hideNonExpr}
            onChange={setHideNonExpr}
          />
        )}
      </FilterBar>

      {sigLoading && <Spinner message="Loading signatures..." />}

      {!sigLoading && signatures && signatures.length === 0 && (
        <p className="py-8 text-center text-text-muted">
          No single-cell validation data available for <strong>{atlas}</strong>.
          Try selecting a different atlas (e.g., cima, scatlas, inflammation).
        </p>
      )}

      {target && scLoading && <Spinner message="Loading scatter..." />}

      {scatter && target && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            {target} — Single-Cell Expression vs Activity
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Each point is a sampled single cell ({filteredPoints.length} points).
            {!cellType && ' Colored by cell type.'}
          </p>
          <ScatterChart
            x={filteredPoints.map((p) => p.x)}
            y={filteredPoints.map((p) => p.y)}
            groups={!cellType ? filteredPoints.map((p) => p.cell_type || '') : undefined}
            labels={filteredPoints.map((p) => p.cell_type || '')}
            xTitle="Expression"
            yTitle="Predicted Activity"
            title={`${target}: Single-Cell Validation`}
            showTrendLine
            stats={{ rho: scatter.rho, p: scatter.p_value, n: filteredPoints.length }}
            height={500}
          />
        </div>
      )}

      {boxData && target && !scLoading && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            {target} — Expressing vs Non-expressing Activity Distribution
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Comparison of predicted activity between cells with expression &gt; 0
            (expressing) and expression &le; 0 (non-expressing).
          </p>
          <BoxplotChart
            groups={boxData.groups}
            values={boxData.values}
            xTitle="Group"
            yTitle="Predicted Activity"
            title={`${target}: Expressing vs Non-expressing`}
            showPoints={false}
            height={400}
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
