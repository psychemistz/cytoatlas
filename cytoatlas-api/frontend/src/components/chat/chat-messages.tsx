import { useRef, useEffect } from 'react';
import { cn } from '@/lib/utils';
import { ChatViz } from '@/components/chat/chat-viz';
import type { ChatMessage } from '@/api/types/chat';

interface ChatMessagesProps {
  messages: ChatMessage[];
  isStreaming: boolean;
  streamedContent: string;
  toolCalls: string[];
}

const TOOL_LABELS: Record<string, string> = {
  search_entity: 'Searching...',
  get_activity_data: 'Fetching activity data...',
  list_cell_types: 'Getting cell types...',
  list_signatures: 'Loading signatures...',
  get_correlations: 'Analyzing correlations...',
  get_disease_activity: 'Getting disease data...',
  compare_atlases: 'Comparing atlases...',
  get_atlas_summary: 'Loading atlas summary...',
  create_visualization: 'Creating visualization...',
  export_data: 'Preparing export...',
};

function formatContent(text: string): React.ReactNode[] {
  const blocks = text.split(/(```[\s\S]*?```)/g);
  const result: React.ReactNode[] = [];

  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i];

    if (block.startsWith('```') && block.endsWith('```')) {
      const inner = block.slice(3, -3);
      const newlineIdx = inner.indexOf('\n');
      const code = newlineIdx >= 0 ? inner.slice(newlineIdx + 1) : inner;
      result.push(
        <pre
          key={`code-${i}`}
          className="my-2 overflow-x-auto rounded-lg bg-bg-tertiary p-3 text-xs leading-relaxed"
        >
          <code>{code}</code>
        </pre>,
      );
      continue;
    }

    const parts = block.split(/(\*\*.*?\*\*|\*.*?\*|`[^`]+`)/g);
    const inlineNodes: React.ReactNode[] = [];

    for (let j = 0; j < parts.length; j++) {
      const part = parts[j];
      if (part.startsWith('**') && part.endsWith('**')) {
        inlineNodes.push(
          <strong key={`b-${i}-${j}`}>{part.slice(2, -2)}</strong>,
        );
      } else if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
        inlineNodes.push(
          <em key={`i-${i}-${j}`}>{part.slice(1, -1)}</em>,
        );
      } else if (part.startsWith('`') && part.endsWith('`')) {
        inlineNodes.push(
          <code
            key={`ic-${i}-${j}`}
            className="rounded bg-bg-tertiary px-1.5 py-0.5 text-xs"
          >
            {part.slice(1, -1)}
          </code>,
        );
      } else if (part) {
        inlineNodes.push(<span key={`t-${i}-${j}`}>{part}</span>);
      }
    }

    if (inlineNodes.length > 0) {
      result.push(<span key={`inline-${i}`}>{inlineNodes}</span>);
    }
  }

  return result;
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';

  return (
    <div className={cn('flex w-full', isUser ? 'justify-end' : 'justify-start')}>
      <div
        className={cn(
          'max-w-[80%] rounded-xl px-4 py-3 text-sm leading-relaxed',
          isUser
            ? 'bg-primary text-white'
            : 'bg-bg-secondary text-text-primary',
        )}
      >
        <div className="whitespace-pre-wrap">{formatContent(message.content)}</div>
        {message.visualizations?.map((viz, i) => (
          <ChatViz key={`${message.id}-viz-${i}`} visualization={viz} />
        ))}
      </div>
    </div>
  );
}

function StreamingBubble({
  content,
  toolCalls,
}: {
  content: string;
  toolCalls: string[];
}) {
  const activeTools = toolCalls.length > 0;
  const lastTool = activeTools ? toolCalls[toolCalls.length - 1] : null;
  const toolLabel = lastTool ? (TOOL_LABELS[lastTool] || `Running ${lastTool}...`) : null;

  return (
    <div className="flex w-full justify-start">
      <div className="max-w-[80%] rounded-xl bg-bg-secondary px-4 py-3 text-sm leading-relaxed text-text-primary">
        {toolLabel && !content && (
          <div className="flex items-center gap-2 text-text-muted">
            <svg
              className="h-4 w-4 animate-spin"
              viewBox="0 0 24 24"
              fill="none"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
              />
            </svg>
            <span>{toolLabel}</span>
          </div>
        )}
        {content && (
          <div className="whitespace-pre-wrap">{formatContent(content)}</div>
        )}
        {toolLabel && content && (
          <div className="mt-2 flex items-center gap-2 text-xs text-text-muted">
            <svg
              className="h-3 w-3 animate-spin"
              viewBox="0 0 24 24"
              fill="none"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
              />
            </svg>
            <span>{toolLabel}</span>
          </div>
        )}
        {!content && !toolLabel && (
          <div className="flex gap-1">
            <span className="h-2 w-2 animate-bounce rounded-full bg-text-muted" style={{ animationDelay: '0ms' }} />
            <span className="h-2 w-2 animate-bounce rounded-full bg-text-muted" style={{ animationDelay: '150ms' }} />
            <span className="h-2 w-2 animate-bounce rounded-full bg-text-muted" style={{ animationDelay: '300ms' }} />
          </div>
        )}
      </div>
    </div>
  );
}

export function ChatMessages({
  messages,
  isStreaming,
  streamedContent,
  toolCalls,
}: ChatMessagesProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, streamedContent, toolCalls]);

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6">
      <div className="mx-auto flex max-w-3xl flex-col gap-4">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {isStreaming && (
          <StreamingBubble content={streamedContent} toolCalls={toolCalls} />
        )}
      </div>
    </div>
  );
}
