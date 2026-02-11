import { Link } from 'react-router';
import { formatNumber, cn } from '@/lib/utils';

interface AtlasCardProps {
  name: string;
  displayName: string;
  description: string;
  nCells: number;
  nSamples: number;
  nCellTypes: number;
  validationGrade?: string;
  sourceType?: string;
  className?: string;
}

const GRADE_COLORS: Record<string, string> = {
  A: 'bg-accent/10 text-accent',
  B: 'bg-primary/10 text-primary',
  C: 'bg-warning/10 text-warning',
};

export function AtlasCard({
  name,
  displayName,
  description,
  nCells,
  nSamples,
  nCellTypes,
  validationGrade,
  sourceType,
  className,
}: AtlasCardProps) {
  return (
    <div
      className={cn(
        'overflow-hidden rounded-xl border border-border-light bg-bg-primary shadow-sm transition-all hover:-translate-y-0.5 hover:shadow-lg',
        className,
      )}
    >
      {/* Header section */}
      <div className="border-b border-border-light p-6">
        <div className="mb-2 flex items-center gap-2">
          <h3 className="text-lg font-semibold">{displayName}</h3>
          {sourceType === 'builtin' && (
            <span className="rounded bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">Core</span>
          )}
          {validationGrade && (
            <span className={cn('rounded px-2 py-0.5 text-xs font-medium', GRADE_COLORS[validationGrade] ?? 'bg-bg-tertiary text-text-muted')}>
              Grade {validationGrade}
            </span>
          )}
        </div>
        <p className="text-sm text-text-secondary">{description}</p>
      </div>

      {/* Stats section */}
      <div className="grid grid-cols-3 gap-2 bg-bg-secondary px-6 py-3 text-center">
        <div>
          <div className="font-semibold text-text-primary">{formatNumber(nCells)}</div>
          <div className="text-xs text-text-muted">Cells</div>
        </div>
        <div>
          <div className="font-semibold text-text-primary">{nSamples.toLocaleString()}</div>
          <div className="text-xs text-text-muted">Samples</div>
        </div>
        <div>
          <div className="font-semibold text-text-primary">{nCellTypes}</div>
          <div className="text-xs text-text-muted">Cell Types</div>
        </div>
      </div>

      {/* Actions section */}
      <div className="flex gap-2 px-6 py-3">
        <Link
          to={`/atlas/${name}`}
          className="flex-1 rounded-md bg-primary py-2 text-center text-sm font-medium text-text-inverse hover:bg-primary-dark"
        >
          Explore
        </Link>
        <Link
          to={`/validate?atlas=${name}`}
          className="flex-1 rounded-md border border-border-light py-2 text-center text-sm font-medium text-text-secondary hover:bg-bg-tertiary"
        >
          Validate
        </Link>
      </div>
    </div>
  );
}
