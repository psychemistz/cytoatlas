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

export function AtlasCard({
  name,
  displayName,
  description,
  nCells,
  nSamples,
  nCellTypes,
  className,
}: AtlasCardProps) {
  return (
    <div
      className={cn(
        'overflow-hidden rounded-xl border border-border-light bg-white shadow-sm transition-all hover:-translate-y-0.5 hover:shadow-lg',
        className,
      )}
    >
      <div className="p-6">
        <div className="mb-3">
          <h3 className="text-lg font-bold text-text-primary">{displayName}</h3>
        </div>
        <p className="mb-5 text-sm leading-relaxed text-text-secondary">{description}</p>

        <div className="mb-5 grid grid-cols-3 gap-3 rounded-lg bg-bg-tertiary p-3 text-center">
          <div>
            <div className="text-lg font-extrabold text-primary">{formatNumber(nCells)}</div>
            <div className="text-xs font-medium text-text-muted">Cells</div>
          </div>
          <div>
            <div className="text-lg font-extrabold text-primary">{nSamples.toLocaleString()}</div>
            <div className="text-xs font-medium text-text-muted">Samples</div>
          </div>
          <div>
            <div className="text-lg font-extrabold text-primary">{nCellTypes}</div>
            <div className="text-xs font-medium text-text-muted">Cell Types</div>
          </div>
        </div>

        <div className="flex gap-3">
          <Link
            to={`/atlas/${name}`}
            className="flex-1 rounded-lg bg-blue-600 py-2.5 text-center text-sm font-bold text-white no-underline shadow-sm transition-all hover:bg-blue-700 hover:text-white hover:shadow-md"
          >
            Explore
          </Link>
          <Link
            to={`/validate?atlas=${name}`}
            className="flex-1 rounded-lg border-2 border-blue-200 bg-blue-50 py-2.5 text-center text-sm font-bold text-blue-700 no-underline transition-all hover:border-blue-300 hover:bg-blue-100"
          >
            Validate
          </Link>
        </div>
      </div>
    </div>
  );
}
