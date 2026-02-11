import { cn } from '@/lib/utils';

interface ToggleOption {
  value: string;
  label: string;
}

interface FilterBarProps {
  children: React.ReactNode;
  className?: string;
}

export function FilterBar({ children, className }: FilterBarProps) {
  return (
    <div className={cn('flex flex-wrap items-center gap-3 rounded-lg border border-border-light bg-bg-secondary p-3', className)}>
      {children}
    </div>
  );
}

interface ToggleGroupProps {
  options: ToggleOption[];
  value: string;
  onChange: (value: string) => void;
  label?: string;
}

export function ToggleGroup({ options, value, onChange, label }: ToggleGroupProps) {
  return (
    <div className="flex items-center gap-2">
      {label && <span className="text-sm font-medium text-text-secondary">{label}</span>}
      <div className="flex rounded-md border border-border-light bg-bg-primary">
        {options.map((opt) => (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            className={cn(
              'px-3 py-1.5 text-sm font-medium transition-colors first:rounded-l-md last:rounded-r-md',
              value === opt.value
                ? 'bg-primary text-text-inverse'
                : 'text-text-secondary hover:bg-bg-tertiary',
            )}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}

interface SelectFilterProps {
  label: string;
  options: { value: string; label: string }[];
  value: string;
  onChange: (value: string) => void;
}

export function SelectFilter({ label, options, value, onChange }: SelectFilterProps) {
  return (
    <div className="flex items-center gap-2">
      <label className="text-sm font-medium text-text-secondary">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-border-light bg-bg-primary px-3 py-1.5 text-sm outline-none focus:border-primary"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}

interface SearchFilterProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

export function SearchFilter({ value, onChange, placeholder = 'Search...' }: SearchFilterProps) {
  return (
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className="rounded-md border border-border-light bg-bg-primary px-3 py-1.5 text-sm outline-none focus:border-primary"
    />
  );
}
