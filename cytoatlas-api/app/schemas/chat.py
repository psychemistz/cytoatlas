"""Chat-related Pydantic schemas."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Chat message roles."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class VisualizationType(str, Enum):
    """Types of inline visualizations."""

    HEATMAP = "heatmap"
    BAR_CHART = "bar_chart"
    SCATTER = "scatter"
    BOX_PLOT = "box_plot"
    LINE_CHART = "line_chart"
    TABLE = "table"


class VisualizationConfig(BaseModel):
    """Configuration for an inline visualization."""

    type: VisualizationType
    title: str | None = None
    data: dict[str, Any]
    config: dict[str, Any] = Field(default_factory=dict)
    container_id: str | None = None


class DownloadableData(BaseModel):
    """Reference to downloadable data from a message."""

    message_id: int
    format: str = "csv"  # csv, json, tsv
    description: str
    size_estimate: str | None = None


class ToolCall(BaseModel):
    """Record of a tool invocation."""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """Result from a tool invocation."""

    tool_call_id: str
    content: str | dict[str, Any]
    is_error: bool = False


class ChatMessageRequest(BaseModel):
    """Request to send a chat message."""

    content: str = Field(..., min_length=1, max_length=10000, description="Message content")
    conversation_id: int | None = Field(None, description="Existing conversation ID")
    session_id: str | None = Field(None, description="Session ID for anonymous users")
    context: dict[str, Any] | None = Field(
        None,
        description="Additional context (current atlas, page, etc.)"
    )


class ChatMessageResponse(BaseModel):
    """Response containing the assistant's message."""

    message_id: int
    conversation_id: int
    role: MessageRole = MessageRole.ASSISTANT
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None
    visualizations: list[VisualizationConfig] | None = None
    downloadable_data: DownloadableData | None = None

    # Token usage
    input_tokens: int | None = None
    output_tokens: int | None = None

    # Timing
    created_at: datetime


class ChatMessage(BaseModel):
    """A chat message for conversation history."""

    id: int
    role: MessageRole
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None
    visualizations: list[VisualizationConfig] | None = None
    downloadable_data: DownloadableData | None = None
    created_at: datetime


class ConversationResponse(BaseModel):
    """A conversation with its messages."""

    id: int
    title: str | None
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessage]
    message_count: int


class ConversationSummary(BaseModel):
    """Summary of a conversation for listing."""

    id: int
    title: str | None
    created_at: datetime
    updated_at: datetime
    message_count: int
    last_message_preview: str | None


class ConversationListResponse(BaseModel):
    """List of conversations."""

    conversations: list[ConversationSummary]
    total: int
    offset: int
    limit: int


class SuggestionCategory(str, Enum):
    """Categories for chat suggestions."""

    EXPLORE = "explore"
    COMPARE = "compare"
    ANALYZE = "analyze"
    EXPLAIN = "explain"


class ChatSuggestion(BaseModel):
    """A suggested query for the chat."""

    text: str
    category: SuggestionCategory
    description: str | None = None


class ChatSuggestionsResponse(BaseModel):
    """Response with suggested queries."""

    suggestions: list[ChatSuggestion]


class StreamChunk(BaseModel):
    """A chunk of streamed response."""

    type: str  # text, tool_call, tool_result, visualization, done
    content: str | dict[str, Any] | None = None
    message_id: int | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    visualization: VisualizationConfig | None = None
