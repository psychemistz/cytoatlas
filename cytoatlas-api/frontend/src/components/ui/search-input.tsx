import { useState, useEffect, useRef, type KeyboardEvent } from 'react';
import { cn } from '@/lib/utils';

interface SearchInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit?: (value: string) => void;
  suggestions?: string[];
  placeholder?: string;
  className?: string;
}

export function SearchInput({
  value,
  onChange,
  onSubmit,
  suggestions = [],
  placeholder = 'Search...',
  className,
}: SearchInputProps) {
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedIdx, setSelectedIdx] = useState(-1);
  const ref = useRef<HTMLDivElement>(null);

  const filtered = value
    ? suggestions.filter((s) => s.toLowerCase().includes(value.toLowerCase())).slice(0, 8)
    : [];

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setShowSuggestions(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIdx((i) => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIdx((i) => Math.max(i - 1, -1));
    } else if (e.key === 'Enter') {
      if (selectedIdx >= 0 && filtered[selectedIdx]) {
        onChange(filtered[selectedIdx]);
        onSubmit?.(filtered[selectedIdx]);
      } else {
        onSubmit?.(value);
      }
      setShowSuggestions(false);
    } else if (e.key === 'Escape') {
      setShowSuggestions(false);
    }
  }

  return (
    <div ref={ref} className={cn('relative', className)}>
      <input
        type="text"
        value={value}
        onChange={(e) => {
          onChange(e.target.value);
          setShowSuggestions(true);
          setSelectedIdx(-1);
        }}
        onFocus={() => setShowSuggestions(true)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        className="w-full rounded-md border border-border-light px-3 py-2 text-sm outline-none focus:border-primary"
      />
      {showSuggestions && filtered.length > 0 && (
        <ul className="absolute left-0 right-0 top-full z-20 mt-1 max-h-60 overflow-auto rounded-md border border-border-light bg-bg-primary py-1 shadow-md">
          {filtered.map((item, i) => (
            <li
              key={item}
              onClick={() => {
                onChange(item);
                onSubmit?.(item);
                setShowSuggestions(false);
              }}
              className={cn(
                'cursor-pointer px-3 py-1.5 text-sm',
                i === selectedIdx ? 'bg-primary/10 text-primary' : 'text-text-primary hover:bg-bg-tertiary',
              )}
            >
              {item}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
