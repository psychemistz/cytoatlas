import { cn } from '@/lib/utils';
import type { Conversation } from '@/api/types/chat';

interface ChatSidebarProps {
  conversations: Conversation[];
  selectedId: number | null;
  onSelect: (id: number) => void;
  onNewChat: () => void;
}

function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMin = Math.floor(diffMs / 60000);
  const diffHr = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHr / 24);

  if (diffMin < 1) return 'just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  if (diffHr < 24) return `${diffHr}h ago`;
  if (diffDay < 7) return `${diffDay}d ago`;
  return date.toLocaleDateString();
}

export function ChatSidebar({
  conversations,
  selectedId,
  onSelect,
  onNewChat,
}: ChatSidebarProps) {
  return (
    <div className="flex h-full w-64 flex-col border-r border-border-light bg-bg-secondary">
      <div className="border-b border-border-light p-3">
        <button
          onClick={onNewChat}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-primary/90"
        >
          <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 5v14M5 12h14" />
          </svg>
          New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {conversations.length === 0 && (
          <p className="px-3 py-6 text-center text-xs text-text-muted">
            No conversations yet
          </p>
        )}
        {conversations.map((conv) => (
          <button
            key={conv.id}
            onClick={() => onSelect(conv.id)}
            className={cn(
              'mb-1 flex w-full flex-col items-start rounded-lg px-3 py-2.5 text-left transition-colors',
              selectedId === conv.id
                ? 'bg-primary/10 text-primary'
                : 'text-text-secondary hover:bg-bg-tertiary',
            )}
          >
            <span className="w-full truncate text-sm font-medium">{conv.title}</span>
            <span className="mt-0.5 text-xs text-text-muted">
              {formatRelativeTime(conv.updated_at)}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
