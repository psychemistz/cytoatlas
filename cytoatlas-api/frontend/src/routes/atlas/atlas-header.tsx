import { Link } from 'react-router';
import { useAppStore } from '@/stores/app-store';
import { SignatureToggle } from '@/components/ui/signature-toggle';
import { formatNumber } from '@/lib/utils';
import type { AtlasSummary } from '@/api/types/atlas';

interface AtlasHeaderProps {
  atlasName: string;
  displayName: string;
  summary?: AtlasSummary;
  signatureType: string;
}

export function AtlasHeader({ atlasName, displayName, summary }: AtlasHeaderProps) {
  return (
    <div className="mb-6">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">{displayName}</h1>
          {summary && (
            <div className="mt-1 flex gap-4 text-sm text-text-secondary">
              <span>{formatNumber(summary.n_cells)} cells</span>
              <span>{summary.n_samples.toLocaleString()} samples</span>
              <span>{summary.n_cell_types} cell types</span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3">
          <SignatureToggle />
          <Link
            to={`/validate?atlas=${atlasName}`}
            className="rounded-md border border-border-light px-3 py-1.5 text-sm font-medium text-text-secondary hover:bg-bg-tertiary"
          >
            Validate
          </Link>
          <a
            href={`/api/v1/export/${atlasName}/activity.csv`}
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-md border border-border-light px-3 py-1.5 text-sm font-medium text-text-secondary hover:bg-bg-tertiary"
          >
            Export CSV
          </a>
        </div>
      </div>
    </div>
  );
}
