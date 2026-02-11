import { useState } from 'react';
import { usePairwiseScatter, useCorrelationMatrix } from '@/api/hooks/use-cross-atlas';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { ScatterChart } from '@/components/charts/scatter-chart';
import { HeatmapChart } from '@/components/charts/heatmap-chart';

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
            zmin={-1}
            zmax={1}
            symmetric
          />
        </div>
      )}
    </div>
  );
}
