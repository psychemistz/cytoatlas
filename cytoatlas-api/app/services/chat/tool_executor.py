"""Tool execution layer for CytoAtlas chat.

Handles tool invocation, result processing, and format conversion.
"""

import json
import logging
from typing import Any

from app.services.mcp_tools import ToolExecutor as MCPToolExecutor

logger = logging.getLogger(__name__)


class ToolResultChunker:
    """Chunks and truncates tool results that are too large."""

    MAX_RESULT_SIZE = 4000  # characters

    @staticmethod
    def chunk_result(result: dict[str, Any]) -> dict[str, Any]:
        """Truncate large tool results and mark as truncated."""
        result_str = json.dumps(result)

        if len(result_str) <= ToolResultChunker.MAX_RESULT_SIZE:
            return result

        # Truncate and mark
        truncated_result = result.copy()
        truncated_result["_truncated"] = True
        truncated_result["_original_size"] = len(result_str)

        # Try to preserve structure by truncating large arrays/strings
        if isinstance(result.get("results"), list):
            truncated_result["results"] = result["results"][:10]
        if isinstance(result.get("correlations"), list):
            truncated_result["correlations"] = result["correlations"][:20]
        if isinstance(result.get("differential_signatures"), list):
            truncated_result["differential_signatures"] = result["differential_signatures"][:20]

        # Final check
        result_str = json.dumps(truncated_result)
        if len(result_str) > ToolResultChunker.MAX_RESULT_SIZE:
            # Aggressive truncation
            truncated_result = {
                "_truncated": True,
                "_original_size": len(json.dumps(result)),
                "summary": f"Result too large ({len(result_str)} chars). Showing keys: {list(result.keys())}",
            }

        return truncated_result


class ToolCallSerializer:
    """Converts between Anthropic and OpenAI tool call formats."""

    @staticmethod
    def to_anthropic_format(tool_calls: list[dict]) -> list[dict]:
        """Convert OpenAI-style tool calls to Anthropic format.

        OpenAI: {"type": "function", "id": "...", "function": {"name": "...", "arguments": "..."}}
        Anthropic: {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
        """
        return [
            {
                "type": "tool_use",
                "id": tc["id"],
                "name": tc.get("function", {}).get("name") or tc.get("name"),
                "input": (
                    json.loads(tc.get("function", {}).get("arguments", "{}"))
                    if isinstance(tc.get("function", {}).get("arguments"), str)
                    else tc.get("arguments", {})
                ),
            }
            for tc in tool_calls
        ]

    @staticmethod
    def to_openai_format(tool_calls: list[dict]) -> list[dict]:
        """Convert Anthropic-style tool calls to OpenAI format.

        Anthropic: {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
        OpenAI: {"type": "function", "id": "...", "function": {"name": "...", "arguments": "..."}}
        """
        return [
            {
                "type": "function",
                "id": tc["id"],
                "function": {
                    "name": tc.get("name"),
                    "arguments": (
                        json.dumps(tc.get("input", {}))
                        if isinstance(tc.get("input"), dict)
                        else tc.get("input", "{}")
                    ),
                },
            }
            for tc in tool_calls
        ]


class ToolExecutor:
    """Enhanced tool executor with chunking and format conversion."""

    def __init__(self):
        self._mcp_executor = MCPToolExecutor()
        self._chunker = ToolResultChunker()
        self._serializer = ToolCallSerializer()

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return chunked result."""
        # Delegate to MCP tool executor
        result = await self._mcp_executor.execute_tool(tool_name, arguments)

        # Chunk if needed
        if isinstance(result, dict):
            result = self._chunker.chunk_result(result)

        return result

    def to_anthropic_format(self, tool_calls: list[dict]) -> list[dict]:
        """Convert tool calls to Anthropic format."""
        return self._serializer.to_anthropic_format(tool_calls)

    def to_openai_format(self, tool_calls: list[dict]) -> list[dict]:
        """Convert tool calls to OpenAI format."""
        return self._serializer.to_openai_format(tool_calls)


# Singleton
_tool_executor: ToolExecutor | None = None


def get_tool_executor() -> ToolExecutor:
    """Get or create the tool executor singleton."""
    global _tool_executor
    if _tool_executor is None:
        _tool_executor = ToolExecutor()
    return _tool_executor
