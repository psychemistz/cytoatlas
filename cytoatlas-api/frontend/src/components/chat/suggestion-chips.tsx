import type { ChatSuggestion } from '@/api/types/chat';

interface SuggestionChipsProps {
  suggestions: ChatSuggestion[];
  onSelect: (text: string) => void;
}

const CATEGORY_LABELS: Record<string, string> = {
  explore: 'Explore',
  compare: 'Compare',
  analyze: 'Analyze',
  explain: 'Explain',
};

const CATEGORY_COLORS: Record<string, string> = {
  explore: 'bg-blue-500/10 text-blue-400 hover:bg-blue-500/20',
  compare: 'bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20',
  analyze: 'bg-amber-500/10 text-amber-400 hover:bg-amber-500/20',
  explain: 'bg-purple-500/10 text-purple-400 hover:bg-purple-500/20',
};

export function SuggestionChips({ suggestions, onSelect }: SuggestionChipsProps) {
  const grouped = suggestions.reduce<Record<string, ChatSuggestion[]>>((acc, s) => {
    const cat = s.category;
    if (!acc[cat]) acc[cat] = [];
    acc[cat].push(s);
    return acc;
  }, {});

  return (
    <div className="space-y-4">
      {Object.entries(grouped).map(([category, items]) => (
        <div key={category}>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-text-muted">
            {CATEGORY_LABELS[category] || category}
          </h3>
          <div className="flex flex-wrap gap-2">
            {items.map((item, i) => (
              <button
                key={i}
                onClick={() => onSelect(item.text)}
                className={`rounded-lg px-3 py-2 text-sm transition-colors ${CATEGORY_COLORS[category] || 'bg-bg-tertiary text-text-secondary hover:bg-bg-tertiary/80'}`}
              >
                {item.text}
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
