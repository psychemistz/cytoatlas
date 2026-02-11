"""LLM client abstraction layer for vLLM and Anthropic backends.

Provides unified interface for both OpenAI-compatible (vLLM) and Anthropic Claude APIs.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Send a chat request and return the complete response.

        Returns:
            dict with keys: content, tool_calls, finish_reason, usage
        """
        pass

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream a chat response.

        Yields chunks with keys: type, content, tool_calls, finish_reason
        """
        pass


class VLLMClient(LLMClient):
    """OpenAI-compatible client for vLLM backend."""

    def __init__(self, base_url: str, api_key: str, default_model: str, max_tokens: int):
        self.base_url = base_url
        self.api_key = api_key
        self.default_model = default_model
        self.default_max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        """Lazy-load OpenAI async client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._client

    # Expected keys for each tool — used to score JSON repair candidates
    _TOOL_KEYS = {
        "get_activity_data": {"atlas_name", "signature_type", "signatures"},
        "get_correlations": {"signature", "signature_type", "correlation_type"},
        "get_disease_activity": {"disease", "signature_type"},
        "compare_atlases": {"signature", "signature_type"},
        "get_atlas_summary": {"atlas_name"},
        "list_cell_types": {"atlas_name"},
        "list_signatures": {"signature_type"},
        "search_entity": {"query"},
        "get_validation_metrics": {"atlas_name"},
        "export_data": {"data_type", "format", "parameters"},
        "create_visualization": {"viz_type", "title", "data"},
    }

    @classmethod
    def _ensure_json_string(cls, args: str | dict, tool_name: str = "") -> str:
        """Ensure tool call arguments are a valid JSON string.

        Mistral may produce duplicated/garbled JSON during streaming.
        Strategy: try as-is, then collect all valid JSON substrings
        and pick the one with the most expected keys for the tool.
        """
        if isinstance(args, dict):
            return json.dumps(args)
        try:
            json.loads(args)
            return args
        except (json.JSONDecodeError, TypeError):
            pass

        # Collect all valid JSON candidates starting from each '{'
        candidates = []
        for i in range(len(args)):
            if args[i] == '{':
                candidate = args[i:]
                try:
                    parsed = json.loads(candidate)
                    candidates.append((i, candidate, parsed))
                except (json.JSONDecodeError, TypeError):
                    continue

        if not candidates:
            logger.warning("Could not repair tool args: %s", args)
            return "{}"

        # Score candidates by how many expected keys they contain
        expected_keys = cls._TOOL_KEYS.get(tool_name, set())
        best = candidates[0]
        best_score = -1

        for i, candidate, parsed in candidates:
            if not isinstance(parsed, dict):
                continue
            score = len(expected_keys & set(parsed.keys()))
            # Prefer candidates with more keys overall (breaks ties)
            score = score * 100 + len(parsed)
            if score > best_score:
                best_score = score
                best = (i, candidate, parsed)

        logger.info(
            "Repaired tool args for %s from pos %d: %s (original: %s, candidates: %d)",
            tool_name, best[0], best[1], args, len(candidates),
        )
        return best[1]

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Send a non-streaming chat request."""
        response = await self.client.chat.completions.create(
            model=model or self.default_model,
            max_tokens=max_tokens or self.default_max_tokens,
            tools=tools,
            messages=messages,
        )

        message = response.choices[0].message
        tool_calls = None

        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                repaired = self._ensure_json_string(
                    tc.function.arguments, tool_name=tc.function.name
                )
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(repaired),
                })

        return {
            "content": message.content,
            "tool_calls": tool_calls,
            "finish_reason": response.choices[0].finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        }

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream a chat response."""
        stream = await self.client.chat.completions.create(
            model=model or self.default_model,
            max_tokens=max_tokens or self.default_max_tokens,
            tools=tools,
            messages=messages,
            stream=True,
        )

        collected_tool_calls: dict[int, dict] = {}
        finish_reason = None

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Text content
            if delta and delta.content:
                yield {
                    "type": "content",
                    "content": delta.content,
                }

            # Tool calls
            if delta and delta.tool_calls:
                for tc_chunk in delta.tool_calls:
                    idx = tc_chunk.index
                    if idx not in collected_tool_calls:
                        collected_tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_chunk.id:
                        collected_tool_calls[idx]["id"] = tc_chunk.id
                    if tc_chunk.function:
                        if tc_chunk.function.name:
                            collected_tool_calls[idx]["name"] += tc_chunk.function.name
                        if tc_chunk.function.arguments:
                            collected_tool_calls[idx]["arguments"] += tc_chunk.function.arguments

            # Finish reason
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        # Return accumulated tool calls if any
        if collected_tool_calls:
            tool_calls = []
            for idx in sorted(collected_tool_calls.keys()):
                tc_data = collected_tool_calls[idx]
                repaired = self._ensure_json_string(tc_data["arguments"], tool_name=tc_data["name"])
                tool_calls.append({
                    "id": tc_data["id"],
                    "name": tc_data["name"],
                    "arguments": json.loads(repaired),
                })

            yield {
                "type": "tool_calls",
                "tool_calls": tool_calls,
                "finish_reason": finish_reason,
            }
        else:
            yield {
                "type": "done",
                "finish_reason": finish_reason,
            }


class AnthropicClient(LLMClient):
    """Client for Anthropic Claude API."""

    def __init__(self, api_key: str, default_model: str, max_tokens: int):
        self.api_key = api_key
        self.default_model = default_model
        self.default_max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        """Lazy-load Anthropic async client."""
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._client

    def _convert_messages(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Extract system prompt and convert messages to Anthropic format.

        Handles conversion of OpenAI-format tool calls and tool results:
        - Assistant messages with "tool_calls" → content blocks with type "tool_use"
        - Messages with role "tool" → user messages with type "tool_result" content blocks
        """
        system_prompt = ""
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]

            elif msg["role"] == "assistant":
                if msg.get("tool_calls"):
                    # Convert OpenAI tool_calls to Anthropic tool_use content blocks
                    content_blocks = []
                    if msg.get("content"):
                        content_blocks.append({"type": "text", "text": msg["content"]})
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        tool_input = fn.get("arguments", "{}")
                        if isinstance(tool_input, str):
                            try:
                                tool_input = json.loads(tool_input)
                            except (json.JSONDecodeError, TypeError):
                                tool_input = {}
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": tool_input,
                        })
                    anthropic_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

            elif msg["role"] == "tool":
                # Convert OpenAI tool result to Anthropic tool_result content block
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }
                # Merge into previous user message if it has tool_result blocks
                if (anthropic_messages
                        and anthropic_messages[-1]["role"] == "user"
                        and isinstance(anthropic_messages[-1]["content"], list)):
                    anthropic_messages[-1]["content"].append(tool_result_block)
                else:
                    anthropic_messages.append({
                        "role": "user",
                        "content": [tool_result_block],
                    })

            else:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        return system_prompt, anthropic_messages

    @staticmethod
    def _convert_tools(tools: list[dict] | None) -> list[dict] | None:
        """Convert OpenAI-format tools to Anthropic format if needed."""
        if not tools:
            return tools
        converted = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                fn = tool["function"]
                converted.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })
            elif "name" in tool and "input_schema" in tool:
                converted.append(tool)
            else:
                converted.append(tool)
        return converted

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Send a non-streaming chat request."""
        system_prompt, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)

        response = await self.client.messages.create(
            model=model or self.default_model,
            max_tokens=max_tokens or self.default_max_tokens,
            system=system_prompt,
            tools=anthropic_tools,
            messages=anthropic_messages,
        )

        content = ""
        tool_calls = None

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return {
            "content": content,
            "tool_calls": tool_calls,
            "finish_reason": response.stop_reason,
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
        }

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream a chat response."""
        system_prompt, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)

        async with self.client.messages.stream(
            model=model or self.default_model,
            max_tokens=max_tokens or self.default_max_tokens,
            system=system_prompt,
            tools=anthropic_tools,
            messages=anthropic_messages,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield {
                            "type": "content",
                            "content": event.delta.text,
                        }

            response = await stream.get_final_message()

            # Extract tool calls from final message
            tool_calls = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    })

            if tool_calls:
                yield {
                    "type": "tool_calls",
                    "tool_calls": tool_calls,
                    "finish_reason": response.stop_reason,
                }
            else:
                yield {
                    "type": "done",
                    "finish_reason": response.stop_reason,
                }


def _sanitize_llm_error(error: Exception) -> str:
    """Sanitize LLM error messages to avoid leaking internal details.

    Removes API keys, URLs, file paths, and other sensitive information
    from error messages before they could be exposed to users.
    """
    import re
    msg = str(error)
    # Remove API keys/tokens
    msg = re.sub(r"(api[_-]?key|token|authorization)[=:\s]+\S+", r"\1=[REDACTED]", msg, flags=re.IGNORECASE)
    # Remove URLs (may contain internal hostnames/ports)
    msg = re.sub(r"https?://[^\s]+", "[URL]", msg)
    # Remove file paths
    msg = re.sub(r"(/[^\s:]+)+", "[path]", msg)
    return msg


class DualLLMClient(LLMClient):
    """Client that tries vLLM first, falls back to Anthropic on connection error."""

    def __init__(
        self,
        vllm_base_url: str | None = None,
        vllm_api_key: str = "not-needed",
        anthropic_api_key: str | None = None,
        chat_model: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        anthropic_model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 4096,
    ):
        self.vllm_client = None
        self.anthropic_client = None

        if vllm_base_url:
            self.vllm_client = VLLMClient(
                base_url=vllm_base_url,
                api_key=vllm_api_key,
                default_model=chat_model,
                max_tokens=max_tokens,
            )

        if anthropic_api_key:
            self.anthropic_client = AnthropicClient(
                api_key=anthropic_api_key,
                default_model=anthropic_model,
                max_tokens=max_tokens,
            )

        if not self.vllm_client and not self.anthropic_client:
            raise RuntimeError(
                "No LLM backend configured: set LLM_BASE_URL for vLLM "
                "or ANTHROPIC_API_KEY for Claude"
            )

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Try vLLM first, fall back to Anthropic on connection error."""
        if self.vllm_client:
            try:
                return await self.vllm_client.chat(messages, tools, model, max_tokens)
            except (ConnectionError, OSError) as e:
                logger.warning("vLLM unavailable (%s), falling back to Anthropic", _sanitize_llm_error(e))
            except Exception as e:
                logger.error("vLLM error: %s", _sanitize_llm_error(e))
                if not self.anthropic_client:
                    raise RuntimeError("LLM service error") from None

        if self.anthropic_client:
            try:
                return await self.anthropic_client.chat(messages, tools, model, max_tokens)
            except Exception as e:
                logger.error("Anthropic error: %s", _sanitize_llm_error(e))
                raise RuntimeError("LLM service error") from None

        raise RuntimeError("All LLM backends unavailable")

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Try vLLM streaming first, fall back to Anthropic on connection error."""
        if self.vllm_client:
            try:
                async for chunk in self.vllm_client.chat_stream(messages, tools, model, max_tokens):
                    yield chunk
                return
            except (ConnectionError, OSError) as e:
                logger.warning("vLLM streaming unavailable (%s), falling back to Anthropic", _sanitize_llm_error(e))
            except Exception as e:
                logger.error("vLLM streaming error: %s", _sanitize_llm_error(e))
                if not self.anthropic_client:
                    raise RuntimeError("LLM service error") from None

        if self.anthropic_client:
            try:
                async for chunk in self.anthropic_client.chat_stream(messages, tools, model, max_tokens):
                    yield chunk
                return
            except Exception as e:
                logger.error("Anthropic streaming error: %s", _sanitize_llm_error(e))
                raise RuntimeError("LLM service error") from None

        raise RuntimeError("All LLM backends unavailable")


def get_llm_client() -> DualLLMClient:
    """Get or create the dual LLM client singleton."""
    return DualLLMClient(
        vllm_base_url=settings.llm_base_url,
        vllm_api_key=settings.llm_api_key,
        anthropic_api_key=settings.anthropic_api_key,
        chat_model=settings.chat_model,
        anthropic_model=settings.anthropic_chat_model,
        max_tokens=settings.chat_max_tokens,
    )
