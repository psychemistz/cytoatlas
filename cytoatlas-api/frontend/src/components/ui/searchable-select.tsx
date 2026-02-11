import { useState, useRef, useEffect, useMemo } from 'react';
import { cn } from '@/lib/utils';

interface SearchableSelectProps {
  label: string;
  options: { value: string; label: string }[];
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
}

export function SearchableSelect({
  label,
  options,
  value,
  onChange,
  placeholder = 'Search...',
  className,
}: SearchableSelectProps) {
  const [search, setSearch] = useState('');
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const filtered = useMemo(() => {
    if (!search) return options;
    const q = search.toLowerCase();
    return options.filter((o) => o.label.toLowerCase().includes(q));
  }, [options, search]);

  const selectedLabel = options.find((o) => o.value === value)?.label ?? '';

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
        setSearch('');
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className={cn('relative flex items-center gap-2', className)} ref={ref}>
      <label className="text-sm font-medium text-text-secondary whitespace-nowrap">{label}</label>
      <div className="relative">
        <input
          ref={inputRef}
          type="text"
          value={open ? search : selectedLabel}
          placeholder={open ? placeholder : selectedLabel || placeholder}
          onChange={(e) => setSearch(e.target.value)}
          onFocus={() => {
            setOpen(true);
            setSearch('');
          }}
          className="w-56 rounded-md border border-border-light bg-bg-primary px-3 py-1.5 text-sm outline-none focus:border-primary"
        />
        {open && filtered.length > 0 && (
          <div className="absolute left-0 top-full z-50 mt-1 max-h-60 w-72 overflow-y-auto rounded-md border border-border-light bg-bg-primary shadow-lg">
            {filtered.map((o) => (
              <button
                key={o.value}
                onClick={() => {
                  onChange(o.value);
                  setOpen(false);
                  setSearch('');
                }}
                className={cn(
                  'block w-full px-3 py-1.5 text-left text-sm hover:bg-bg-tertiary',
                  o.value === value && 'bg-primary/10 font-medium text-primary',
                )}
              >
                {o.label}
              </button>
            ))}
          </div>
        )}
        {open && filtered.length === 0 && search && (
          <div className="absolute left-0 top-full z-50 mt-1 w-72 rounded-md border border-border-light bg-bg-primary p-3 text-sm text-text-muted shadow-lg">
            No matches for &quot;{search}&quot;
          </div>
        )}
      </div>
    </div>
  );
}
