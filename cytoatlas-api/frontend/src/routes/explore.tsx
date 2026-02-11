import { useState, useMemo } from 'react';
import { useAtlases } from '@/api/hooks/use-atlas';
import { PLACEHOLDER_ATLASES } from '@/lib/constants';
import { AtlasCard } from '@/components/ui/atlas-card';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { Spinner } from '@/components/ui/loading-skeleton';

const TYPE_OPTIONS = [
  { value: 'all', label: 'All Types' },
  { value: 'builtin', label: 'Core Atlases' },
  { value: 'published', label: 'Published' },
  { value: 'user', label: 'Community' },
];

const SORT_OPTIONS = [
  { value: 'cells', label: 'Sort by Cells' },
  { value: 'name', label: 'Sort by Name' },
  { value: 'grade', label: 'Sort by Grade' },
];

export default function Explore() {
  const { data: atlases, isLoading } = useAtlases();
  const [typeFilter, setTypeFilter] = useState('all');
  const [sortBy, setSortBy] = useState('cells');

  const allAtlases = atlases ?? PLACEHOLDER_ATLASES;

  const filtered = useMemo(() => {
    let result = allAtlases.filter((a) => {
      if (typeFilter !== 'all' && a.source_type !== typeFilter) return false;
      return true;
    });

    const gradeOrder: Record<string, number> = { A: 0, B: 1, C: 2, D: 3, F: 4 };
    result = [...result].sort((a, b) => {
      switch (sortBy) {
        case 'cells':
          return (b.n_cells ?? 0) - (a.n_cells ?? 0);
        case 'name':
          return (a.display_name ?? a.name).localeCompare(b.display_name ?? b.name);
        case 'grade':
          return (gradeOrder[a.validation_grade ?? ''] ?? 5) - (gradeOrder[b.validation_grade ?? ''] ?? 5);
        default:
          return 0;
      }
    });

    return result;
  }, [allAtlases, typeFilter, sortBy]);

  return (
    <div className="mx-auto max-w-[1400px] px-4 py-12">
      <div className="mb-8">
        <h1 className="mb-2 text-3xl font-bold">Explore Atlases</h1>
        <p className="text-text-secondary">Browse and analyze cytokine activities across single-cell atlases</p>
      </div>

      <FilterBar className="mb-6">
        <SelectFilter label="" options={TYPE_OPTIONS} value={typeFilter} onChange={setTypeFilter} />
        <SelectFilter label="" options={SORT_OPTIONS} value={sortBy} onChange={setSortBy} />
      </FilterBar>

      {isLoading ? (
        <Spinner message="Loading atlases..." />
      ) : filtered.length === 0 ? (
        <p className="py-12 text-center text-text-muted">No atlases match your filters</p>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1.5rem' }}>
          {filtered.map((atlas) => (
            <AtlasCard
              key={atlas.name}
              name={atlas.name}
              displayName={atlas.display_name}
              description={atlas.description}
              nCells={atlas.n_cells}
              nSamples={atlas.n_samples}
              nCellTypes={atlas.n_cell_types}
            />
          ))}
        </div>
      )}
    </div>
  );
}
