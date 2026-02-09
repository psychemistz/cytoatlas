"""Conversation persistence service.

Handles conversation and message storage with SQLAlchemy async or in-memory fallback.
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


class ConversationService:
    """Service for managing conversation persistence.

    Uses SQLAlchemy async if database is configured, otherwise in-memory storage.
    """

    def __init__(self, use_database: bool = False):
        """Initialize conversation service.

        Args:
            use_database: Whether to use database persistence
        """
        self.use_database = use_database and settings.use_database
        self._db_session = None

        # In-memory storage (fallback)
        self._conversations: dict[int, dict[str, Any]] = {}
        self._messages: dict[int, list[ChatMessage]] = {}
        self._data_cache: dict[int, dict[str, Any]] = {}  # conversation_id -> {export_id: data}
        self._session_conversations: dict[str, list[int]] = {}
        self._user_conversations: dict[int, list[int]] = {}
        self._next_id = 1

    async def create_conversation(
        self,
        session_id: str,
        user_id: int | None = None,
        title: str | None = None,
    ) -> int:
        """Create a new conversation.

        Args:
            session_id: Session ID
            user_id: Optional user ID
            title: Optional conversation title

        Returns:
            Conversation ID
        """
        if self.use_database:
            return await self._create_conversation_db(session_id, user_id, title)
        else:
            return self._create_conversation_memory(session_id, user_id, title)

    def _create_conversation_memory(
        self,
        session_id: str,
        user_id: int | None,
        title: str | None,
    ) -> int:
        """Create conversation in memory."""
        conversation_id = self._next_id
        self._next_id += 1

        self._conversations[conversation_id] = {
            "id": conversation_id,
            "session_id": session_id,
            "user_id": user_id,
            "title": title,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        self._messages[conversation_id] = []
        self._data_cache[conversation_id] = {}

        # Index by session and user
        if session_id not in self._session_conversations:
            self._session_conversations[session_id] = []
        self._session_conversations[session_id].append(conversation_id)

        if user_id is not None:
            if user_id not in self._user_conversations:
                self._user_conversations[user_id] = []
            self._user_conversations[user_id].append(conversation_id)

        return conversation_id

    async def _create_conversation_db(
        self,
        session_id: str,
        user_id: int | None,
        title: str | None,
    ) -> int:
        """Create conversation in database."""
        # TODO: Implement database persistence
        # For now, fall back to in-memory
        logger.warning("Database persistence not yet implemented, using in-memory")
        return self._create_conversation_memory(session_id, user_id, title)

    async def add_message(
        self,
        conversation_id: int,
        role: MessageRole,
        content: str,
        tool_calls: list[ToolCall] | None = None,
        tool_results: list[ToolResult] | None = None,
        visualizations: list[VisualizationConfig] | None = None,
        downloadable_data: DownloadableData | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        citations: list[dict[str, Any]] | None = None,
    ) -> ChatMessage:
        """Add a message to a conversation.

        Returns:
            Created message
        """
        if self.use_database:
            return await self._add_message_db(
                conversation_id, role, content, tool_calls, tool_results,
                visualizations, downloadable_data, input_tokens, output_tokens, citations
            )
        else:
            return self._add_message_memory(
                conversation_id, role, content, tool_calls, tool_results,
                visualizations, downloadable_data, input_tokens, output_tokens, citations
            )

    def _add_message_memory(
        self,
        conversation_id: int,
        role: MessageRole,
        content: str,
        tool_calls: list[ToolCall] | None,
        tool_results: list[ToolResult] | None,
        visualizations: list[VisualizationConfig] | None,
        downloadable_data: DownloadableData | None,
        input_tokens: int | None,
        output_tokens: int | None,
        citations: list[dict[str, Any]] | None,
    ) -> ChatMessage:
        """Add message in memory."""
        if conversation_id not in self._messages:
            raise ValueError(f"Conversation {conversation_id} not found")

        message_id = len(self._messages[conversation_id]) + 1

        # Auto-generate title from first user message
        if (
            role == MessageRole.USER
            and len(self._messages[conversation_id]) == 0
            and self._conversations[conversation_id]["title"] is None
        ):
            self._conversations[conversation_id]["title"] = content[:50] + (
                "..." if len(content) > 50 else ""
            )

        message = ChatMessage(
            id=message_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_results=tool_results,
            visualizations=visualizations,
            downloadable_data=downloadable_data,
            created_at=datetime.utcnow(),
        )

        self._messages[conversation_id].append(message)
        self._conversations[conversation_id]["updated_at"] = datetime.utcnow()

        return message

    async def _add_message_db(
        self,
        conversation_id: int,
        role: MessageRole,
        content: str,
        tool_calls: list[ToolCall] | None,
        tool_results: list[ToolResult] | None,
        visualizations: list[VisualizationConfig] | None,
        downloadable_data: DownloadableData | None,
        input_tokens: int | None,
        output_tokens: int | None,
        citations: list[dict[str, Any]] | None,
    ) -> ChatMessage:
        """Add message in database."""
        # TODO: Implement database persistence
        logger.warning("Database persistence not yet implemented, using in-memory")
        return self._add_message_memory(
            conversation_id, role, content, tool_calls, tool_results,
            visualizations, downloadable_data, input_tokens, output_tokens, citations
        )

    async def get_history(
        self,
        conversation_id: int,
        limit: int = 20,
    ) -> list[ChatMessage]:
        """Get conversation history.

        Args:
            conversation_id: Conversation ID
            limit: Maximum messages to return

        Returns:
            List of messages (most recent last)
        """
        if self.use_database:
            return await self._get_history_db(conversation_id, limit)
        else:
            return self._get_history_memory(conversation_id, limit)

    def _get_history_memory(self, conversation_id: int, limit: int) -> list[ChatMessage]:
        """Get history from memory."""
        if conversation_id not in self._messages:
            return []
        return self._messages[conversation_id][-limit:]

    async def _get_history_db(self, conversation_id: int, limit: int) -> list[ChatMessage]:
        """Get history from database."""
        # TODO: Implement database persistence
        logger.warning("Database persistence not yet implemented, using in-memory")
        return self._get_history_memory(conversation_id, limit)

    async def get_or_create_conversation(
        self,
        conversation_id: int | None,
        session_id: str | None,
        user_id: int | None,
    ) -> int:
        """Get existing conversation or create new one.

        Args:
            conversation_id: Optional conversation ID
            session_id: Session ID
            user_id: Optional user ID

        Returns:
            Conversation ID
        """
        if conversation_id is not None:
            # Verify access
            if conversation_id in self._conversations:
                conv = self._conversations[conversation_id]
                if user_id is not None and conv["user_id"] != user_id:
                    raise PermissionError("Not authorized for this conversation")
                if session_id is not None and conv["session_id"] != session_id:
                    raise PermissionError("Not authorized for this conversation")
                return conversation_id

        # Create new conversation
        if session_id is None:
            session_id = str(uuid4())
        return await self.create_conversation(session_id, user_id)

    async def get_conversation_response(self, conversation_id: int) -> ConversationResponse:
        """Get full conversation with messages."""
        if conversation_id not in self._conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        conv = self._conversations[conversation_id]
        messages = self._messages[conversation_id]

        return ConversationResponse(
            id=conv["id"],
            title=conv["title"],
            created_at=conv["created_at"],
            updated_at=conv["updated_at"],
            messages=messages,
            message_count=len(messages),
        )

    async def list_conversations(
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
        conversations.sort(key=lambda c: c["updated_at"], reverse=True)

        total = len(conversations)
        conversations = conversations[offset : offset + limit]

        summaries = []
        for conv in conversations:
            messages = self._messages[conv["id"]]
            last_message = messages[-1] if messages else None
            preview = None
            if last_message:
                preview = last_message.content[:100] + (
                    "..." if len(last_message.content) > 100 else ""
                )

            summaries.append(
                ConversationSummary(
                    id=conv["id"],
                    title=conv["title"],
                    created_at=conv["created_at"],
                    updated_at=conv["updated_at"],
                    message_count=len(messages),
                    last_message_preview=preview,
                )
            )

        return summaries, total

    def cache_data(self, conversation_id: int, key: str, data: Any) -> None:
        """Cache data for download."""
        if conversation_id not in self._data_cache:
            self._data_cache[conversation_id] = {}
        self._data_cache[conversation_id][key] = {
            "data": data,
            "cached_at": datetime.utcnow().isoformat(),
        }

    def get_cached_data(self, conversation_id: int, key: str) -> Any | None:
        """Retrieve cached data."""
        if conversation_id not in self._data_cache:
            return None
        cached = self._data_cache[conversation_id].get(key)
        return cached.get("data") if cached else None


# Singleton
_conversation_service: ConversationService | None = None


def get_conversation_service(use_database: bool | None = None) -> ConversationService:
    """Get or create the conversation service singleton."""
    global _conversation_service
    if _conversation_service is None:
        if use_database is None:
            use_database = settings.use_database
        _conversation_service = ConversationService(use_database=use_database)
    return _conversation_service
