import { cn } from '@/lib/utils';

interface LoadingSkeletonProps {
  lines?: number;
  className?: string;
  message?: string;
}

export function LoadingSkeleton({ lines = 3, className, message }: LoadingSkeletonProps) {
  return (
    <div className={cn('animate-pulse space-y-3 p-4', className)}>
      {message && <p className="text-sm text-text-muted">{message}</p>}
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="h-4 rounded bg-bg-tertiary"
          style={{ width: `${80 - i * 15}%` }}
        />
      ))}
    </div>
  );
}

export function Spinner({ message }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-12">
      <svg className="h-8 w-8 animate-spin text-primary" viewBox="0 0 24 24" fill="none">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
        />
      </svg>
      {message && <p className="text-sm text-text-muted">{message}</p>}
    </div>
  );
}
