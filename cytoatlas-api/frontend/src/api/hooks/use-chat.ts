import { useState, useCallback, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import { API_BASE } from '@/lib/constants';
import type {
  ChatSuggestion,
  ChatVisualization,
  Conversation,
  ConversationDetail,
  ChatStatus,
  StreamChunk,
} from '@/api/types/chat';

export function useChatSuggestions() {
  return useQuery({
    queryKey: ['chat', 'suggestions'],
    queryFn: async () => {
      const res = await get<{ suggestions: ChatSuggestion[] }>('/chat/suggestions');
      return res.suggestions;
    },
  });
}

export function useChatConversations() {
  return useQuery({
    queryKey: ['chat', 'conversations'],
    queryFn: async () => {
      const res = await get<{ conversations: Conversation[] }>('/chat/conversations');
      return res.conversations;
    },
  });
}

export function useChatConversation(conversationId: number | null) {
  return useQuery({
    queryKey: ['chat', 'conversation', conversationId],
    queryFn: () => get<ConversationDetail>(`/chat/conversations/${conversationId}`),
    enabled: conversationId !== null,
  });
}

export function useChatStatus() {
  return useQuery({
    queryKey: ['chat', 'status'],
    queryFn: () => get<ChatStatus>('/chat/status'),
  });
}

export function useSendMessage() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamedContent, setStreamedContent] = useState('');
  const [toolCalls, setToolCalls] = useState<string[]>([]);
  const [visualizations, setVisualizations] = useState<ChatVisualization[]>([]);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    setStreamedContent('');
    setToolCalls([]);
    setVisualizations([]);
    setError(null);
  }, []);

  const sendMessage = useCallback(
    async (
      content: string,
      conversationId: number | null,
      sessionId: string,
    ): Promise<number | null> => {
      setIsStreaming(true);
      setStreamedContent('');
      setToolCalls([]);
      setVisualizations([]);
      setError(null);

      const controller = new AbortController();
      abortRef.current = controller;

      let messageId: number | null = null;

      try {
        const response = await fetch(`${API_BASE}/chat/message/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            content,
            conversation_id: conversationId,
            session_id: sessionId,
          }),
          signal: controller.signal,
        });

        if (!response.ok) {
          const body = await response.json().catch(() => ({ detail: 'Stream request failed' }));
          throw new Error(body.detail || 'Stream request failed');
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error('No response body');

        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith('data: ')) continue;

            const jsonStr = trimmed.slice(6);
            if (jsonStr === '[DONE]') continue;

            try {
              const chunk: StreamChunk = JSON.parse(jsonStr);

              switch (chunk.type) {
                case 'text':
                  setStreamedContent((prev) => prev + chunk.content);
                  break;
                case 'tool_call':
                  setToolCalls((prev) => [...prev, chunk.tool_call.name]);
                  break;
                case 'visualization':
                  setVisualizations((prev) => [...prev, chunk.visualization]);
                  break;
                case 'done':
                  messageId = chunk.message_id;
                  break;
                case 'error':
                  setError(chunk.content);
                  break;
              }
            } catch {
              // Skip malformed JSON lines
            }
          }
        }
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          setError((err as Error).message || 'Failed to send message');
        }
      } finally {
        setIsStreaming(false);
        abortRef.current = null;
      }

      return messageId;
    },
    [],
  );

  return { sendMessage, isStreaming, streamedContent, toolCalls, visualizations, error, reset };
}
