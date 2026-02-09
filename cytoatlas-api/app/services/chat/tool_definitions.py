"""Tool definitions for CytoAtlas chat.

Centralized tool schemas in both Anthropic and OpenAI formats.
"""

# Import from mcp_tools to maintain compatibility
from app.services.mcp_tools import CYTOATLAS_TOOLS


def to_openai_tools(anthropic_tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool format to OpenAI function-calling format.

    The JSON Schema bodies (input_schema vs parameters) are identical;
    only the wrapping object differs between the two APIs.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for tool in anthropic_tools
    ]


OPENAI_TOOLS = to_openai_tools(CYTOATLAS_TOOLS)


__all__ = ["CYTOATLAS_TOOLS", "OPENAI_TOOLS", "to_openai_tools"]
