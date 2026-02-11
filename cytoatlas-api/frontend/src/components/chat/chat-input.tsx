import { useState, useCallback, useRef, useEffect } from 'react';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled: boolean;
}

const MAX_CHARS = 10000;
const MAX_ROWS = 6;
const LINE_HEIGHT = 24;

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const adjustHeight = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    const maxHeight = LINE_HEIGHT * MAX_ROWS;
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`;
  }, []);

  useEffect(() => {
    adjustHeight();
  }, [value, adjustHeight]);

  const handleSend = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue('');
  }, [value, disabled, onSend]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  const charCount = value.length;
  const isOverLimit = charCount > MAX_CHARS;
  const isEmpty = value.trim().length === 0;

  return (
    <div className="border-t border-border-light bg-bg-primary p-4">
      <div className="mx-auto flex max-w-3xl items-end gap-3">
        <div className="relative flex-1">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about cytokine activity, cell types, diseases..."
            disabled={disabled}
            rows={1}
            className="w-full resize-none rounded-lg border border-border-light bg-bg-secondary px-4 py-3 pr-16 text-sm text-text-primary placeholder-text-muted focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50"
            style={{ lineHeight: `${LINE_HEIGHT}px` }}
          />
          <span
            className={`absolute bottom-2 right-3 text-xs ${isOverLimit ? 'text-red-400' : 'text-text-muted'}`}
          >
            {charCount}/{MAX_CHARS}
          </span>
        </div>
        <button
          onClick={handleSend}
          disabled={disabled || isEmpty || isOverLimit}
          className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-primary text-white transition-colors hover:bg-primary/90 disabled:opacity-40 disabled:hover:bg-primary"
          aria-label="Send message"
        >
          <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M22 2L11 13" />
            <path d="M22 2L15 22L11 13L2 9L22 2Z" />
          </svg>
        </button>
      </div>
    </div>
  );
}
