import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import type { AtlasSummary } from '@/api/types/atlas';
import { Spinner } from '@/components/ui/loading-skeleton';
import { StatCard } from '@/components/ui/stat-card';

interface OverviewPanelProps {
  signatureType: string;
  atlasName: string;
}

export default function OverviewPanel({ atlasName }: OverviewPanelProps) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['atlas-summary', atlasName],
    queryFn: () => get<AtlasSummary>(`/atlases/${atlasName}/summary`),
  });

  if (isLoading) return <Spinner message="Loading atlas summary..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load atlas summary: {(error as Error).message}
      </div>
    );
  }

  if (!data) return null;

  const formatCount = (n: number) =>
    n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1_000 ? `${(n / 1_000).toFixed(1)}K` : String(n);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">{data.display_name}</h2>
      </div>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-5">
        <StatCard value={formatCount(data.n_cells)} label="Cells" />
        <StatCard value={formatCount(data.n_samples)} label="Samples" />
        <StatCard value={data.n_cell_types} label="Cell Types" />
        <StatCard value={data.n_cytosig_signatures} label="CytoSig Signatures" />
        <StatCard value={data.n_secact_signatures} label="SecAct Signatures" />
      </div>

      {data.diseases && data.diseases.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">Diseases</h3>
          <div className="flex flex-wrap gap-2">
            {data.diseases.map((d) => (
              <span key={d} className="rounded-full bg-red-50 px-3 py-1 text-xs font-medium text-red-700">
                {d}
              </span>
            ))}
          </div>
        </div>
      )}

      {data.tissues && data.tissues.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">Tissues</h3>
          <div className="flex flex-wrap gap-2">
            {data.tissues.map((t) => (
              <span key={t} className="rounded-full bg-green-50 px-3 py-1 text-xs font-medium text-green-700">
                {t}
              </span>
            ))}
          </div>
        </div>
      )}

      <div>
        <h3 className="mb-2 text-sm font-semibold text-text-secondary">
          Cell Types ({data.cell_types.length})
        </h3>
        <div className="grid grid-cols-2 gap-1 md:grid-cols-3 lg:grid-cols-4">
          {data.cell_types.map((ct) => (
            <span key={ct} className="rounded bg-bg-tertiary px-2 py-1 text-xs text-text-primary">
              {ct}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
