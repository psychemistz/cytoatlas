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
  explore: 'bg-blue-50 text-blue-700 border border-blue-200 hover:bg-blue-100',
  compare: 'bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100',
  analyze: 'bg-amber-50 text-amber-700 border border-amber-200 hover:bg-amber-100',
  explain: 'bg-purple-50 text-purple-700 border border-purple-200 hover:bg-purple-100',
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
          <h3 className="mb-2 text-xs font-bold uppercase tracking-wider text-slate-500">
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
