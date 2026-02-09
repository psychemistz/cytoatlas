"""Quick test: Compare Claude vs Mistral responses for the IFNG activity query.

Run: python test_claude_vs_mistral.py
"""

import asyncio
import json
import os
import sys

# Load env
from dotenv import load_dotenv
load_dotenv()

# Add app to path
sys.path.insert(0, os.path.dirname(__file__))

from app.services.mcp_tools import CYTOATLAS_TOOLS
from app.services.chat.chat_service import SYSTEM_PROMPT

QUERY = "What is the activity of IFNG across different immune cell types in CIMA?"


async def test_claude():
    """Call Claude API directly and show the raw response."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        return

    client = anthropic.AsyncAnthropic(api_key=api_key)

    messages = [{"role": "user", "content": QUERY}]

    print("=" * 60)
    print("  CLAUDE API RESPONSE")
    print("=" * 60)

    # First call
    response = await client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=CYTOATLAS_TOOLS,
        messages=messages,
    )

    print(f"\nStop reason: {response.stop_reason}")
    print(f"Input tokens: {response.usage.input_tokens}")
    print(f"Output tokens: {response.usage.output_tokens}")
    print(f"\nContent blocks ({len(response.content)}):")

    for i, block in enumerate(response.content):
        print(f"\n--- Block {i} (type={block.type}) ---")
        if block.type == "text":
            print(f"Text: {block.text[:500]}...")
        elif block.type == "tool_use":
            print(f"Tool: {block.name}")
            print(f"ID: {block.id}")
            print(f"Input: {json.dumps(block.input, indent=2)}")

    # If Claude made tool calls, simulate tool results and get final response
    if response.stop_reason == "tool_use":
        print("\n\n>>> Claude made tool calls. Simulating tool execution...")

        # Build continuation messages
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                # Execute tool for real
                from app.services.mcp_tools import ToolExecutor
                executor = ToolExecutor()
                result = await executor.execute_tool(block.name, block.input)
                print(f"\nTool result for {block.name}: {json.dumps(result, indent=2)[:1000]}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        messages.append({"role": "user", "content": tool_results})

        # Second call with tool results
        response2 = await client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=CYTOATLAS_TOOLS,
            messages=messages,
        )

        print(f"\n\n--- CLAUDE FINAL RESPONSE ---")
        print(f"Stop reason: {response2.stop_reason}")
        for block in response2.content:
            if block.type == "text":
                print(f"\nText:\n{block.text}")
            elif block.type == "tool_use":
                print(f"\nAdditional tool call: {block.name}")
                print(f"Input: {json.dumps(block.input, indent=2)}")

        # If there's another tool call (e.g., create_visualization), handle it
        if response2.stop_reason == "tool_use":
            print("\n\n>>> Claude made additional tool calls...")
            messages.append({"role": "assistant", "content": response2.content})

            tool_results2 = []
            for block in response2.content:
                if block.type == "tool_use":
                    from app.services.mcp_tools import ToolExecutor
                    executor = ToolExecutor()
                    result = await executor.execute_tool(block.name, block.input)
                    print(f"\nTool result for {block.name}: {json.dumps(result, indent=2)[:1000]}")
                    tool_results2.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

            messages.append({"role": "user", "content": tool_results2})

            response3 = await client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=CYTOATLAS_TOOLS,
                messages=messages,
            )

            print(f"\n\n--- CLAUDE FINAL RESPONSE (round 3) ---")
            for block in response3.content:
                if block.type == "text":
                    print(f"\nText:\n{block.text}")
                elif block.type == "tool_use":
                    print(f"\nTool call: {block.name}({json.dumps(block.input)})")


async def test_vllm():
    """Call vLLM/Mistral API and show the raw response."""
    from openai import AsyncOpenAI

    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8001/v1")
    client = AsyncOpenAI(base_url=base_url, api_key="not-needed")

    # Convert tools to OpenAI format
    from app.services.chat.tool_definitions import OPENAI_TOOLS

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUERY},
    ]

    print("\n" + "=" * 60)
    print("  vLLM/MISTRAL API RESPONSE")
    print("=" * 60)

    try:
        response = await client.chat.completions.create(
            model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            max_tokens=4096,
            tools=OPENAI_TOOLS,
            messages=messages,
        )

        message = response.choices[0].message
        print(f"\nFinish reason: {response.choices[0].finish_reason}")
        print(f"Content: {message.content[:500] if message.content else 'None'}...")

        if message.tool_calls:
            for tc in message.tool_calls:
                print(f"\nTool call: {tc.function.name}")
                print(f"Arguments: {tc.function.arguments}")
    except Exception as e:
        print(f"vLLM error: {e}")


if __name__ == "__main__":
    asyncio.run(test_claude())
    # Uncomment to also test vLLM:
    # asyncio.run(test_vllm())
