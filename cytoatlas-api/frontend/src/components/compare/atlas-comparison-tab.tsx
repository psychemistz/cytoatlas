import { useState, useMemo } from 'react';
import { usePairwiseScatter, useCorrelationMatrix } from '@/api/hooks/use-cross-atlas';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { ScatterChart } from '@/components/charts/scatter-chart';
import { HeatmapChart } from '@/components/charts/heatmap-chart';
import { BarChart } from '@/components/charts/bar-chart';
import { CORRELATION_COLORSCALE } from '@/components/charts/chart-defaults';

interface AtlasComparisonTabProps {
  signatureType: string;
}

const ATLAS_OPTIONS = [
  { value: 'cima', label: 'CIMA' },
  { value: 'inflammation', label: 'Inflammation' },
  { value: 'scatlas', label: 'scAtlas' },
];

export default function AtlasComparisonTab({
  signatureType,
}: AtlasComparisonTabProps) {
  const [atlas1, setAtlas1] = useState('cima');
  const [atlas2, setAtlas2] = useState('inflammation');
  const [level, setLevel] = useState('pseudobulk');

  const { data: scatter, isLoading: scLoading } = usePairwiseScatter(
    atlas1,
    atlas2,
    signatureType,
    level,
  );

  const { data: corrMatrix, isLoading: matLoading } =
    useCorrelationMatrix(signatureType);

  const perCelltypeCorr = useMemo(() => {
    if (!scatter?.points || scatter.points.length < 2) return null;
    const groups = new Map<string, { xs: number[]; ys: number[] }>();
    for (const p of scatter.points) {
      const ct = p.cell_type || p.label || 'Unknown';
      if (!groups.has(ct)) groups.set(ct, { xs: [], ys: [] });
      const g = groups.get(ct)!;
      g.xs.push(p.x);
      g.ys.push(p.y);
    }
    const entries = [...groups.entries()]
      .filter(([, g]) => g.xs.length >= 2)
      .map(([ct, g]) => ({
        ct,
        meanX: g.xs.reduce((a, b) => a + b, 0) / g.xs.length,
        meanY: g.ys.reduce((a, b) => a + b, 0) / g.ys.length,
      }))
      .sort((a, b) => b.meanX - a.meanX);
    if (entries.length === 0) return null;
    return {
      categories: entries.map(e => e.ct),
      series: [
        { name: atlas1, values: entries.map(e => e.meanX) },
        { name: atlas2, values: entries.map(e => e.meanY) },
      ],
    };
  }, [scatter, atlas1, atlas2]);

  return (
    <div className="space-y-6">
      <FilterBar>
        <SelectFilter
          label="Atlas 1"
          options={ATLAS_OPTIONS}
          value={atlas1}
          onChange={setAtlas1}
        />
        <SelectFilter
          label="Atlas 2"
          options={ATLAS_OPTIONS.filter((a) => a.value !== atlas1)}
          value={atlas2}
          onChange={setAtlas2}
        />
        <SelectFilter
          label="Level"
          options={[
            { value: 'pseudobulk', label: 'Pseudobulk' },
            { value: 'singlecell', label: 'Single Cell' },
          ]}
          value={level}
          onChange={setLevel}
        />
      </FilterBar>

      {scLoading && <Spinner message="Loading pairwise comparison..." />}

      {scatter && (
        <div className="space-y-4">
          <div>
            <h3 className="mb-2 text-sm font-semibold text-text-secondary">
              {atlas1} vs {atlas2} ({level})
            </h3>
            <ScatterChart
              x={scatter.points.map((p) => p.x)}
              y={scatter.points.map((p) => p.y)}
              labels={scatter.points.map((p) => p.label || p.cell_type || '')}
              xTitle={atlas1}
              yTitle={atlas2}
              title={`Atlas Comparison: ${atlas1} vs ${atlas2}`}
              showTrendLine
              stats={{ rho: scatter.rho, p: scatter.p_value }}
              height={500}
            />
          </div>
          <div className="rounded-md border border-border-light bg-bg-secondary p-4">
            <h4 className="mb-2 text-sm font-semibold text-text-secondary">
              Comparison Statistics
            </h4>
            <dl className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <dt className="text-text-muted">Spearman rho</dt>
                <dd className="font-mono font-semibold">{scatter.rho.toFixed(4)}</dd>
              </div>
              <div>
                <dt className="text-text-muted">p-value</dt>
                <dd className="font-mono font-semibold">{scatter.p_value.toExponential(2)}</dd>
              </div>
              <div>
                <dt className="text-text-muted">Data points</dt>
                <dd className="font-mono font-semibold">
                  {scatter.points.length.toLocaleString()}
                </dd>
              </div>
            </dl>
          </div>
        </div>
      )}

      {perCelltypeCorr && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Mean Activity by Cell Type
          </h3>
          <BarChart
            categories={perCelltypeCorr.categories}
            series={perCelltypeCorr.series}
            orientation="h"
            xTitle="Mean Activity"
            yTitle="Cell Type"
            title={`${atlas1} vs ${atlas2}: Cell Type Comparison`}
            height={Math.max(400, perCelltypeCorr.categories.length * 28 + 150)}
          />
        </div>
      )}

      {matLoading && <Spinner message="Loading correlation matrix..." />}

      {corrMatrix && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Atlas-to-Atlas Correlation Matrix
          </h3>
          <HeatmapChart
            z={corrMatrix.z}
            x={corrMatrix.x}
            y={corrMatrix.y}
            title="Pairwise Atlas Correlations"
            colorbarTitle="Spearman rho"
            colorscale={CORRELATION_COLORSCALE}
            zmin={-1}
            zmax={1}
            symmetric
          />
        </div>
      )}
    </div>
  );
}
