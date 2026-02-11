import { useState, useMemo } from 'react';
import { useCelltypeSankey } from '@/api/hooks/use-cross-atlas';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter, ToggleGroup } from '@/components/ui/filter-bar';
import { SankeyChart } from '@/components/charts/sankey-chart';
import { BarChart } from '@/components/charts/bar-chart';

const LINEAGE_OPTIONS = [
  { value: 'all', label: 'All Lineages' },
  { value: 'T_cell', label: 'T Cell' },
  { value: 'Myeloid', label: 'Myeloid' },
  { value: 'B_cell', label: 'B Cell' },
  { value: 'NK_ILC', label: 'NK/ILC' },
];

export default function CelltypeMappingTab() {
  const [level, setLevel] = useState<'coarse' | 'fine'>('coarse');
  const [lineage, setLineage] = useState('all');

  const { data, isLoading, error } = useCelltypeSankey(level, lineage);

  const annotationBarData = useMemo(() => {
    if (!data?.nodes || !data?.links) return null;
    const atlasGroups = new Map<string, number>();
    for (const node of data.nodes) {
      const parts = node.label.split(':');
      if (parts.length > 1) {
        const atlas = parts[0].trim();
        atlasGroups.set(atlas, (atlasGroups.get(atlas) || 0) + 1);
      }
    }
    if (atlasGroups.size === 0) return null;
    return {
      categories: [...atlasGroups.keys()],
      values: [...atlasGroups.values()],
    };
  }, [data]);

  if (isLoading) return <Spinner message="Loading cell type mapping..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load mapping: {(error as Error).message}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <FilterBar>
        <ToggleGroup
          options={[
            { value: 'coarse', label: 'Coarse (8 lineages)' },
            { value: 'fine', label: 'Fine (~32 types)' },
          ]}
          value={level}
          onChange={(v) => setLevel(v as 'coarse' | 'fine')}
        />
        <SelectFilter
          label="Lineage"
          options={LINEAGE_OPTIONS}
          value={lineage}
          onChange={setLineage}
        />
      </FilterBar>

      {data?.nodes && (
        <div className="rounded-md border border-border-light bg-bg-secondary p-4">
          <dl className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <dt className="text-text-muted">Total Node Types</dt>
              <dd className="font-mono font-semibold">{data.nodes.length}</dd>
            </div>
            <div>
              <dt className="text-text-muted">Mappings</dt>
              <dd className="font-mono font-semibold">{data.links.length}</dd>
            </div>
            <div>
              <dt className="text-text-muted">Level</dt>
              <dd className="font-mono font-semibold capitalize">{level}</dd>
            </div>
          </dl>
        </div>
      )}

      {annotationBarData && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Annotations per Atlas
          </h3>
          <BarChart
            categories={annotationBarData.categories}
            values={annotationBarData.values}
            title="Cell Type Annotations by Atlas"
            yTitle="Count"
          />
        </div>
      )}

      {level === 'coarse' && data?.nodes && data.nodes.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Cell Type Mapping (Coarse Level)
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Flow from atlas-specific annotations to harmonized lineages.
          </p>
          <SankeyChart
            nodes={data.nodes.map((n) => ({ name: n.label, category: n.color }))}
            links={data.links}
            title="Cell Type Harmonization: Coarse"
            height={600}
          />
        </div>
      )}

      {level === 'fine' && data?.nodes && data.nodes.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Cell Type Mapping (Fine Level)
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Fine-grained cell type harmonization across atlases.
          </p>
          <SankeyChart
            nodes={data.nodes.map((n) => ({ name: n.label, category: n.color }))}
            links={data.links}
            title="Cell Type Harmonization: Fine"
            height={700}
          />
        </div>
      )}

      {(!data?.nodes || data.nodes.length === 0) && (
        <p className="py-8 text-center text-text-muted">
          No cell type mapping data available
        </p>
      )}
    </div>
  );
}
