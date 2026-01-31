"""Context manager for chat conversations.

Handles conversation history, token budgeting, and data caching for downloads.
"""

import json
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from app.config import get_settings
from app.schemas.chat import (
    ChatMessage,
    ConversationResponse,
    ConversationSummary,
    DownloadableData,
    MessageRole,
    ToolCall,
    ToolResult,
    VisualizationConfig,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class ConversationContext:
    """Manages context for a single conversation."""

    def __init__(
        self,
        conversation_id: int,
        user_id: int | None,
        session_id: str,
        title: str | None = None,
    ):
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.session_id = session_id
        self.title = title
        self.messages: list[ChatMessage] = []
        self.data_cache: dict[str, Any] = {}
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def add_user_message(self, content: str) -> ChatMessage:
        """Add a user message to the conversation."""
        message = ChatMessage(
            id=len(self.messages) + 1,
            role=MessageRole.USER,
            content=content,
            created_at=datetime.utcnow(),
        )
        self.messages.append(message)
        self.updated_at = datetime.utcnow()

        # Auto-generate title from first message
        if self.title is None and len(self.messages) == 1:
            self.title = content[:50] + ("..." if len(content) > 50 else "")

        return message

    def add_assistant_message(
        self,
        content: str,
        tool_calls: list[ToolCall] | None = None,
        tool_results: list[ToolResult] | None = None,
        visualizations: list[VisualizationConfig] | None = None,
        downloadable_data: DownloadableData | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> ChatMessage:
        """Add an assistant message to the conversation."""
        message = ChatMessage(
            id=len(self.messages) + 1,
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
            tool_results=tool_results,
            visualizations=visualizations,
            downloadable_data=downloadable_data,
            created_at=datetime.utcnow(),
        )
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        return message

    def cache_data(self, key: str, data: Any) -> None:
        """Cache data for potential download."""
        self.data_cache[key] = {
            "data": data,
            "cached_at": datetime.utcnow().isoformat(),
        }

    def get_cached_data(self, key: str) -> Any | None:
        """Retrieve cached data."""
        cached = self.data_cache.get(key)
        return cached.get("data") if cached else None

    def to_messages_for_api(self, max_messages: int = 20) -> list[dict[str, Any]]:
        """Convert conversation to format for Claude API.

        Limits to recent messages to stay within token budget.
        """
        # Take recent messages
        recent = self.messages[-max_messages:]

        api_messages = []
        for msg in recent:
            if msg.role == MessageRole.USER:
                api_messages.append({
                    "role": "user",
                    "content": msg.content,
                })
            elif msg.role == MessageRole.ASSISTANT:
                content = msg.content

                # Include tool results in the message if present
                if msg.tool_results:
                    tool_result_text = "\n\n[Tool Results]\n"
                    for tr in msg.tool_results:
                        tool_result_text += f"{tr.content}\n"
                    content += tool_result_text

                api_messages.append({
                    "role": "assistant",
                    "content": content,
                })

        return api_messages

    def to_response(self) -> ConversationResponse:
        """Convert to API response."""
        return ConversationResponse(
            id=self.conversation_id,
            title=self.title,
            created_at=self.created_at,
            updated_at=self.updated_at,
            messages=self.messages,
            message_count=len(self.messages),
        )

    def to_summary(self) -> ConversationSummary:
        """Convert to summary for listing."""
        last_message = self.messages[-1] if self.messages else None
        preview = None
        if last_message:
            preview = last_message.content[:100] + ("..." if len(last_message.content) > 100 else "")

        return ConversationSummary(
            id=self.conversation_id,
            title=self.title,
            created_at=self.created_at,
            updated_at=self.updated_at,
            message_count=len(self.messages),
            last_message_preview=preview,
        )


class ContextManager:
    """Manages conversation contexts across the application."""

    def __init__(self):
        self._conversations: dict[int, ConversationContext] = {}
        self._session_conversations: dict[str, list[int]] = {}  # session_id -> conversation_ids
        self._user_conversations: dict[int, list[int]] = {}  # user_id -> conversation_ids
        self._next_id = 1

    def create_conversation(
        self,
        user_id: int | None = None,
        session_id: str | None = None,
    ) -> ConversationContext:
        """Create a new conversation."""
        if session_id is None:
            session_id = str(uuid4())

        conversation_id = self._next_id
        self._next_id += 1

        context = ConversationContext(
            conversation_id=conversation_id,
            user_id=user_id,
            session_id=session_id,
        )

        self._conversations[conversation_id] = context

        # Index by session
        if session_id not in self._session_conversations:
            self._session_conversations[session_id] = []
        self._session_conversations[session_id].append(conversation_id)

        # Index by user
        if user_id is not None:
            if user_id not in self._user_conversations:
                self._user_conversations[user_id] = []
            self._user_conversations[user_id].append(conversation_id)

        return context

    def get_conversation(self, conversation_id: int) -> ConversationContext | None:
        """Get a conversation by ID."""
        return self._conversations.get(conversation_id)

    def get_or_create_conversation(
        self,
        conversation_id: int | None,
        user_id: int | None,
        session_id: str | None,
    ) -> ConversationContext:
        """Get existing conversation or create new one."""
        if conversation_id is not None:
            context = self.get_conversation(conversation_id)
            if context is not None:
                # Verify access
                if user_id is not None and context.user_id != user_id:
                    raise PermissionError("Not authorized for this conversation")
                if session_id is not None and context.session_id != session_id:
                    raise PermissionError("Not authorized for this conversation")
                return context

        return self.create_conversation(user_id, session_id)

    def list_conversations(
        self,
        user_id: int | None = None,
        session_id: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[ConversationSummary], int]:
        """List conversations for a user or session."""
        if user_id is not None:
            conv_ids = self._user_conversations.get(user_id, [])
        elif session_id is not None:
            conv_ids = self._session_conversations.get(session_id, [])
        else:
            conv_ids = []

        # Get conversations and sort by updated_at descending
        conversations = [
            self._conversations[cid]
            for cid in conv_ids
            if cid in self._conversations
        ]
        conversations.sort(key=lambda c: c.updated_at, reverse=True)

        total = len(conversations)
        conversations = conversations[offset : offset + limit]

        summaries = [c.to_summary() for c in conversations]
        return summaries, total

    def delete_conversation(
        self,
        conversation_id: int,
        user_id: int | None = None,
        session_id: str | None = None,
    ) -> bool:
        """Delete a conversation."""
        context = self.get_conversation(conversation_id)
        if context is None:
            return False

        # Verify access
        if user_id is not None and context.user_id != user_id:
            raise PermissionError("Not authorized to delete this conversation")
        if session_id is not None and context.session_id != session_id:
            raise PermissionError("Not authorized to delete this conversation")

        # Remove from indexes
        if context.session_id in self._session_conversations:
            self._session_conversations[context.session_id].remove(conversation_id)
        if context.user_id is not None and context.user_id in self._user_conversations:
            self._user_conversations[context.user_id].remove(conversation_id)

        del self._conversations[conversation_id]
        return True


# Singleton instance
_context_manager: ContextManager | None = None


def get_context_manager() -> ContextManager:
    """Get or create the context manager singleton."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager
