"""Unit tests for the chat service and LLM client."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.chat.llm_client import DualLLMClient, VLLMClient
from app.services.chat.chat_service import ChatService
from app.services.chat.conversation_service import ConversationService
from app.schemas.chat import MessageRole


class TestEnsureJsonString:
    """Tests for VLLMClient._ensure_json_string (JSON repair)."""

    def test_valid_json_string(self):
        """Valid JSON string is returned unchanged."""
        args = '{"signature": "IFNG", "atlas": "cima"}'
        result = VLLMClient._ensure_json_string(args)
        assert result == args

    def test_dict_input(self):
        """Dict input is serialized to JSON string."""
        args = {"signature": "IFNG"}
        result = VLLMClient._ensure_json_string(args)
        assert json.loads(result) == args

    def test_garbled_duplicate_json(self):
        """Duplicated/garbled JSON from Mistral is repaired."""
        # Simulates Mistral streaming producing duplicated JSON
        valid_json = '{"signature": "IFNG"}'
        garbled = valid_json + valid_json  # double
        result = VLLMClient._ensure_json_string(garbled)
        parsed = json.loads(result)
        assert parsed == {"signature": "IFNG"}

    def test_partial_then_valid_json(self):
        """Partial JSON prefix followed by valid JSON is repaired."""
        garbled = '{"sig{"signature": "TNF"}'
        result = VLLMClient._ensure_json_string(garbled)
        parsed = json.loads(result)
        assert parsed["signature"] == "TNF"

    def test_completely_invalid_returns_empty(self):
        """Completely invalid input returns empty JSON object."""
        result = VLLMClient._ensure_json_string("not json at all")
        assert result == "{}"

    def test_empty_string_returns_empty(self):
        """Empty string returns empty JSON object."""
        result = VLLMClient._ensure_json_string("")
        assert result == "{}"


class TestDualLLMClient:
    """Tests for DualLLMClient fallback behavior."""

    def test_requires_at_least_one_backend(self):
        """DualLLMClient raises if no backend is configured."""
        with pytest.raises(RuntimeError, match="No LLM backend configured"):
            DualLLMClient(vllm_base_url=None, anthropic_api_key=None)

    def test_vllm_only_client(self):
        """DualLLMClient can be created with only vLLM."""
        client = DualLLMClient(
            vllm_base_url="http://localhost:8001/v1",
            anthropic_api_key=None,
        )
        assert client.vllm_client is not None
        assert client.anthropic_client is None

    def test_anthropic_only_client(self):
        """DualLLMClient can be created with only Anthropic."""
        client = DualLLMClient(
            vllm_base_url=None,
            anthropic_api_key="test-key",
        )
        assert client.vllm_client is None
        assert client.anthropic_client is not None

    def test_dual_client(self):
        """DualLLMClient can be created with both backends."""
        client = DualLLMClient(
            vllm_base_url="http://localhost:8001/v1",
            anthropic_api_key="test-key",
        )
        assert client.vllm_client is not None
        assert client.anthropic_client is not None

    async def test_fallback_on_connection_error(self):
        """DualLLMClient falls back to Anthropic on vLLM connection error."""
        client = DualLLMClient(
            vllm_base_url="http://localhost:8001/v1",
            anthropic_api_key="test-key",
        )

        expected_response = {
            "content": "test response",
            "tool_calls": None,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        client.vllm_client.chat = AsyncMock(side_effect=ConnectionError("refused"))
        client.anthropic_client.chat = AsyncMock(return_value=expected_response)

        result = await client.chat(messages=[{"role": "user", "content": "test"}])

        assert result == expected_response
        client.vllm_client.chat.assert_called_once()
        client.anthropic_client.chat.assert_called_once()

    async def test_raises_when_all_unavailable(self):
        """DualLLMClient raises RuntimeError when all backends fail."""
        client = DualLLMClient(
            vllm_base_url="http://localhost:8001/v1",
            anthropic_api_key=None,
        )
        client.vllm_client.chat = AsyncMock(side_effect=ConnectionError("refused"))

        with pytest.raises(RuntimeError, match="All LLM backends unavailable"):
            await client.chat(messages=[{"role": "user", "content": "test"}])


class TestChatServiceSuggestions:
    """Tests for ChatService suggestion generation."""

    def test_get_suggestions_returns_items(self):
        """get_suggestions returns non-empty list of suggestions."""
        # Create with mock dependencies so no actual LLM is needed
        mock_llm = MagicMock()
        mock_rag = MagicMock()
        mock_conv = MagicMock()
        mock_tool = MagicMock()

        service = ChatService(
            llm_client=mock_llm,
            rag_service=mock_rag,
            conversation_service=mock_conv,
            tool_executor=mock_tool,
            rag_enabled=False,
        )

        suggestions = service.get_suggestions()
        assert len(suggestions.suggestions) > 0

    def test_suggestions_have_categories(self):
        """Each suggestion has a valid category."""
        mock_llm = MagicMock()
        mock_rag = MagicMock()
        mock_conv = MagicMock()
        mock_tool = MagicMock()

        service = ChatService(
            llm_client=mock_llm,
            rag_service=mock_rag,
            conversation_service=mock_conv,
            tool_executor=mock_tool,
            rag_enabled=False,
        )

        suggestions = service.get_suggestions()
        categories = {s.category.value for s in suggestions.suggestions}
        # Should have at least explore and analyze categories
        assert "explore" in categories
        assert "analyze" in categories

    def test_suggestions_have_text_and_description(self):
        """Each suggestion has both text and description."""
        mock_llm = MagicMock()
        mock_rag = MagicMock()
        mock_conv = MagicMock()
        mock_tool = MagicMock()

        service = ChatService(
            llm_client=mock_llm,
            rag_service=mock_rag,
            conversation_service=mock_conv,
            tool_executor=mock_tool,
            rag_enabled=False,
        )

        suggestions = service.get_suggestions()
        for s in suggestions.suggestions:
            assert s.text and len(s.text) > 5
            assert s.description and len(s.description) > 3


class TestConversationService:
    """Tests for ConversationService (in-memory mode)."""

    async def test_create_conversation(self):
        """Creating a conversation returns an integer ID."""
        svc = ConversationService(use_database=False)
        cid = await svc.create_conversation("session-1", user_id=None)
        assert isinstance(cid, int)
        assert cid >= 1

    async def test_add_and_get_messages(self):
        """Messages are stored and retrievable."""
        svc = ConversationService(use_database=False)
        cid = await svc.create_conversation("session-1")

        await svc.add_message(cid, MessageRole.USER, "Hello")
        await svc.add_message(cid, MessageRole.ASSISTANT, "Hi there!")

        history = await svc.get_history(cid)
        assert len(history) == 2
        assert history[0].role == MessageRole.USER
        assert history[1].role == MessageRole.ASSISTANT

    async def test_auto_title_from_first_message(self):
        """Conversation title is auto-generated from first user message."""
        svc = ConversationService(use_database=False)
        cid = await svc.create_conversation("session-1")
        await svc.add_message(cid, MessageRole.USER, "What is IFNG activity?")

        conv = svc._conversations[cid]
        assert conv["title"] is not None
        assert "IFNG" in conv["title"]

    async def test_get_or_create_existing(self):
        """get_or_create returns existing conversation when valid."""
        svc = ConversationService(use_database=False)
        cid = await svc.create_conversation("session-1", user_id=None)

        returned_cid = await svc.get_or_create_conversation(cid, "session-1", None)
        assert returned_cid == cid

    async def test_get_or_create_wrong_session_raises(self):
        """get_or_create raises PermissionError for wrong session."""
        svc = ConversationService(use_database=False)
        cid = await svc.create_conversation("session-1", user_id=None)

        with pytest.raises(PermissionError):
            await svc.get_or_create_conversation(cid, "wrong-session", None)

    async def test_add_message_invalid_conversation(self):
        """Adding message to non-existent conversation raises ValueError."""
        svc = ConversationService(use_database=False)

        with pytest.raises(ValueError, match="not found"):
            svc._add_message_memory(99999, MessageRole.USER, "hello",
                                    None, None, None, None, None, None, None)

    async def test_cache_data(self):
        """cache_data and get_cached_data round-trip correctly."""
        svc = ConversationService(use_database=False)
        cid = await svc.create_conversation("session-1")

        svc.cache_data(cid, "export_123", {"rows": [1, 2, 3]})
        result = svc.get_cached_data(cid, "export_123")
        assert result == {"rows": [1, 2, 3]}

    async def test_get_cached_data_missing_returns_none(self):
        """get_cached_data returns None for unknown keys."""
        svc = ConversationService(use_database=False)
        cid = await svc.create_conversation("session-1")
        assert svc.get_cached_data(cid, "nonexistent") is None

    async def test_list_conversations_by_session(self):
        """Conversations are listed by session ID."""
        svc = ConversationService(use_database=False)
        c1 = await svc.create_conversation("session-A")
        c2 = await svc.create_conversation("session-A")
        c3 = await svc.create_conversation("session-B")

        convos, total = await svc.list_conversations(session_id="session-A")
        assert total == 2
        assert len(convos) == 2
