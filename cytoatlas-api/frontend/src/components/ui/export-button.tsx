import { useState } from 'react';
import { cn } from '@/lib/utils';

type ExportFormat = 'CSV' | 'PNG' | 'JSON';

interface ExportButtonProps {
  formats?: ExportFormat[];
  onExport: (format: ExportFormat) => void;
  className?: string;
}

export function ExportButton({ formats = ['CSV', 'PNG'], onExport, className }: ExportButtonProps) {
  const [open, setOpen] = useState(false);

  if (formats.length === 1) {
    return (
      <button
        onClick={() => onExport(formats[0])}
        className={cn(
          'rounded-md border border-border-light px-3 py-1.5 text-sm font-medium text-text-secondary hover:bg-bg-tertiary',
          className,
        )}
      >
        Export {formats[0]}
      </button>
    );
  }

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          'flex items-center gap-1 rounded-md border border-border-light px-3 py-1.5 text-sm font-medium text-text-secondary hover:bg-bg-tertiary',
          className,
        )}
      >
        Export
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="m6 9 6 6 6-6" />
        </svg>
      </button>
      {open && (
        <div className="absolute right-0 top-full z-10 mt-1 rounded-md border border-border-light bg-bg-primary py-1 shadow-md">
          {formats.map((fmt) => (
            <button
              key={fmt}
              onClick={() => {
                onExport(fmt);
                setOpen(false);
              }}
              className="block w-full px-4 py-1.5 text-left text-sm text-text-secondary hover:bg-bg-tertiary"
            >
              {fmt}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
