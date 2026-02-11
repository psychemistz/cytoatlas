export interface ChatMessage {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  visualizations?: ChatVisualization[];
  downloadable_data?: { format: string; description: string };
  created_at: string;
}

export interface Conversation {
  id: number;
  title: string;
  updated_at: string;
}

export interface ConversationDetail {
  id: number;
  title: string;
  messages: ChatMessage[];
}

export interface ChatSuggestion {
  text: string;
  category: 'explore' | 'compare' | 'analyze' | 'explain';
}

export interface ChatVisualization {
  container_id: string;
  type?: string;
  data?: unknown;
  layout?: unknown;
}

export type StreamChunk =
  | { type: 'text'; content: string }
  | { type: 'tool_call'; tool_call: { name: string; arguments?: unknown } }
  | { type: 'tool_result'; tool_result: unknown }
  | { type: 'visualization'; visualization: ChatVisualization }
  | { type: 'done'; message_id: number }
  | { type: 'error'; content: string };

export interface ChatStatus {
  status: string;
  llm_configured: boolean;
  model?: string;
  tools_available?: string[];
}
