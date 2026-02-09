"""Chat service package for CytoAtlas.

Provides modular, composable chat functionality with:
- Dual LLM backend (vLLM + Anthropic fallback)
- RAG (Retrieval Augmented Generation) using LanceDB
- Tool execution for CytoAtlas queries
- Conversation persistence
"""

from app.services.chat.chat_service import ChatService, get_chat_service

__all__ = ["ChatService", "get_chat_service"]
