import { useSpatialSummary, useSpatialDatasets } from '@/api/hooks/use-spatial';
import { StatCard } from '@/components/ui/stat-card';
import { Spinner } from '@/components/ui/loading-skeleton';

interface OverviewTabProps {
  signatureType: string;
}

const TECHNOLOGY_TIERS = [
  {
    tier: 'A',
    label: 'Full Inference',
    description: '15K+ genes (e.g., Visium). Complete CytoSig/SecAct activity inference.',
  },
  {
    tier: 'B',
    label: 'Targeted Scoring',
    description: '150-1,000 genes (e.g., Xenium, MERFISH, CosMx). Partial signature coverage.',
  },
  {
    tier: 'C',
    label: 'Skip',
    description: '<150 genes. Insufficient gene panel coverage for reliable inference.',
  },
];

export default function OverviewTab({ signatureType: _signatureType }: OverviewTabProps) {
  const { data: summary, isLoading: summaryLoading } = useSpatialSummary();
  const { data: datasets, isLoading: datasetsLoading } = useSpatialDatasets();

  if (summaryLoading || datasetsLoading) {
    return <Spinner message="Loading spatial overview..." />;
  }

  return (
    <div className="space-y-8">
      {/* Summary stats */}
      {summary && (
        <div>
          <h3 className="mb-3 text-sm font-semibold text-text-secondary">
            Corpus Summary
          </h3>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <StatCard
              value={summary.total_cells.toLocaleString()}
              label="Total Cells"
              className="rounded-lg border border-border-light bg-bg-secondary p-4"
            />
            <StatCard
              value={summary.total_datasets.toLocaleString()}
              label="Datasets"
              className="rounded-lg border border-border-light bg-bg-secondary p-4"
            />
            <StatCard
              value={summary.technologies.toLocaleString()}
              label="Technologies"
              className="rounded-lg border border-border-light bg-bg-secondary p-4"
            />
            <StatCard
              value={summary.tissues.toLocaleString()}
              label="Tissues"
              className="rounded-lg border border-border-light bg-bg-secondary p-4"
            />
          </div>
        </div>
      )}

      {/* Technology tiers */}
      <div>
        <h3 className="mb-3 text-sm font-semibold text-text-secondary">
          Technology Tiers
        </h3>
        <p className="mb-3 text-xs text-text-muted">
          Spatial technologies are classified into tiers based on gene panel size,
          which determines the feasibility and reliability of activity inference.
        </p>
        <div className="grid gap-3 sm:grid-cols-3">
          {TECHNOLOGY_TIERS.map((t) => (
            <div
              key={t.tier}
              className="rounded-lg border border-border-light bg-bg-secondary p-4"
            >
              <div className="mb-1 flex items-center gap-2">
                <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-primary text-xs font-bold text-text-inverse">
                  {t.tier}
                </span>
                <span className="text-sm font-semibold">{t.label}</span>
              </div>
              <p className="text-xs text-text-muted">{t.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Dataset table */}
      <div>
        <h3 className="mb-3 text-sm font-semibold text-text-secondary">
          Datasets ({datasets?.length ?? 0})
        </h3>
        {datasets && datasets.length > 0 ? (
          <div className="overflow-x-auto rounded-lg border border-border-light">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border-light bg-bg-secondary text-left">
                  <th className="px-4 py-2 font-medium text-text-secondary">Dataset</th>
                  <th className="px-4 py-2 font-medium text-text-secondary">Technology</th>
                  <th className="px-4 py-2 font-medium text-text-secondary">Tissue</th>
                  <th className="px-4 py-2 text-right font-medium text-text-secondary">Cells</th>
                  <th className="px-4 py-2 text-right font-medium text-text-secondary">Genes</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((ds) => (
                  <tr
                    key={ds.dataset_id}
                    className="border-b border-border-light last:border-b-0 hover:bg-bg-tertiary"
                  >
                    <td className="px-4 py-2 font-mono text-xs">
                      {ds.filename ?? ds.dataset_id}
                    </td>
                    <td className="px-4 py-2">{ds.technology}</td>
                    <td className="px-4 py-2">{ds.tissue}</td>
                    <td className="px-4 py-2 text-right">
                      {ds.n_cells.toLocaleString()}
                    </td>
                    <td className="px-4 py-2 text-right">
                      {ds.n_genes.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="py-8 text-center text-text-muted">No datasets available</p>
        )}
      </div>
    </div>
  );
}
