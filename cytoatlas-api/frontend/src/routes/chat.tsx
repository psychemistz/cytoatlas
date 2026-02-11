import { useState, useCallback, useEffect } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import {
  useChatSuggestions,
  useChatConversations,
  useChatConversation,
  useChatStatus,
  useSendMessage,
} from '@/api/hooks/use-chat';
import { ChatSidebar } from '@/components/chat/chat-sidebar';
import { ChatMessages } from '@/components/chat/chat-messages';
import { ChatInput } from '@/components/chat/chat-input';
import { SuggestionChips } from '@/components/chat/suggestion-chips';
import { Spinner } from '@/components/ui/loading-skeleton';
import type { ChatMessage, ChatSuggestion } from '@/api/types/chat';

function getSessionId(): string {
  const key = 'cytoatlas_session';
  let id = localStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(key, id);
  }
  return id;
}

export default function Chat() {
  const queryClient = useQueryClient();
  const [selectedConversationId, setSelectedConversationId] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId] = useState(getSessionId);

  const { data: suggestions } = useChatSuggestions();
  const { data: conversations = [], isLoading: conversationsLoading } = useChatConversations();
  const { data: conversationDetail } = useChatConversation(selectedConversationId);
  const { data: status } = useChatStatus();
  const { sendMessage, isStreaming, streamedContent, toolCalls, error, reset } = useSendMessage();

  // Sync messages when a conversation is loaded
  useEffect(() => {
    if (conversationDetail) {
      setMessages(conversationDetail.messages);
    }
  }, [conversationDetail]);

  const handleNewChat = useCallback(() => {
    setSelectedConversationId(null);
    setMessages([]);
    reset();
  }, [reset]);

  const handleSelectConversation = useCallback(
    (id: number) => {
      setSelectedConversationId(id);
      reset();
    },
    [reset],
  );

  const handleSend = useCallback(
    async (content: string) => {
      // Add user message to the local state immediately
      const userMessage: ChatMessage = {
        id: Date.now(),
        role: 'user',
        content,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMessage]);

      const msgId = await sendMessage(content, selectedConversationId, sessionId);

      // After streaming completes, add the assistant message
      // The streamedContent will have the final content at this point
      // We need to read it from the ref after sendMessage resolves
      if (msgId !== null) {
        // Refresh conversations list to get the new/updated conversation
        queryClient.invalidateQueries({ queryKey: ['chat', 'conversations'] });

        // If this was a new conversation, the server assigned an ID
        if (selectedConversationId === null) {
          // Refetch conversations to find the new one
          queryClient.invalidateQueries({ queryKey: ['chat', 'conversations'] });
        }
      }
    },
    [sendMessage, selectedConversationId, sessionId, queryClient],
  );

  // When streaming finishes, append the assistant message to the local messages
  useEffect(() => {
    if (!isStreaming && streamedContent) {
      const assistantMessage: ChatMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: streamedContent,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
      reset();
    }
  }, [isStreaming, streamedContent, reset]);

  const llmConfigured = status?.llm_configured ?? false;
  const showWelcome = selectedConversationId === null && messages.length === 0;

  return (
    <div className="flex h-[calc(100vh-4rem)] overflow-hidden">
      {/* Sidebar */}
      <ChatSidebar
        conversations={conversations}
        selectedId={selectedConversationId}
        onSelect={handleSelectConversation}
        onNewChat={handleNewChat}
      />

      {/* Main chat area */}
      <div className="flex flex-1 flex-col">
        {conversationsLoading ? (
          <Spinner message="Loading conversations..." />
        ) : showWelcome ? (
          <WelcomeScreen
            suggestions={suggestions || []}
            llmConfigured={llmConfigured}
            onSendSuggestion={handleSend}
          />
        ) : (
          <ChatMessages
            messages={messages}
            isStreaming={isStreaming}
            streamedContent={streamedContent}
            toolCalls={toolCalls}
          />
        )}

        {error && (
          <div className="mx-auto max-w-3xl px-4 pb-2">
            <div className="rounded-lg border border-red-500/20 bg-red-500/10 px-4 py-2 text-sm text-red-400">
              {error}
            </div>
          </div>
        )}

        <ChatInput onSend={handleSend} disabled={isStreaming || !llmConfigured} />
      </div>
    </div>
  );
}

function WelcomeScreen({
  suggestions,
  llmConfigured,
  onSendSuggestion,
}: {
  suggestions: ChatSuggestion[];
  llmConfigured: boolean;
  onSendSuggestion: (text: string) => void;
}) {
  return (
    <div className="flex flex-1 items-center justify-center overflow-y-auto px-4 py-12">
      <div className="w-full max-w-2xl">
        <div className="mb-8 text-center">
          <h1 className="mb-2 text-3xl font-bold text-text-primary">CytoAtlas Assistant</h1>
          <p className="text-text-secondary">
            Ask questions about cytokine activity, cell types, diseases, and more.
            Powered by AI with access to the full CytoAtlas dataset.
          </p>
          {!llmConfigured && (
            <div className="mt-4 inline-block rounded-lg border border-amber-500/20 bg-amber-500/10 px-4 py-2 text-sm text-amber-400">
              AI assistant is not configured. Please contact the administrator.
            </div>
          )}
        </div>

        {suggestions.length > 0 && (
          <div>
            <h2 className="mb-4 text-center text-sm font-medium text-text-muted">
              Try asking about...
            </h2>
            <SuggestionChips suggestions={suggestions} onSelect={onSendSuggestion} />
          </div>
        )}

        <div className="mt-8 text-center text-xs text-text-muted">
          <p>
            The assistant can search across all atlases, analyze activity data,
            create visualizations, and export results.
          </p>
        </div>
      </div>
    </div>
  );
}
