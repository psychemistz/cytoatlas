"""Test all 8 chat suggestion questions through Claude API.

Captures tool call patterns and visualization behavior for each question.
Run: ENVIRONMENT=development python test_all_questions.py
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from app.services.mcp_tools import CYTOATLAS_TOOLS, ToolExecutor
from app.services.chat.chat_service import SYSTEM_PROMPT

QUESTIONS = [
    "What is the activity of IFNG across different immune cell types in CIMA?",
    "Compare TNF activity between CIMA and Inflammation Atlas",
    "What cytokines are most differentially active in COVID-19 patients?",
    "How does IL-17A activity correlate with age in CD4 T cells?",
    "What are the top secreted proteins in tumor-associated macrophages?",
    "Explain what CytoSig activity scores mean",
    "Show me the validation metrics for the Inflammation Atlas",
    "Which organs show the highest IL-6 activity in scAtlas?",
]


async def run_claude_conversation(question: str, executor: ToolExecutor) -> dict:
    """Run a full multi-turn Claude conversation for one question."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages = [{"role": "user", "content": question}]

    all_tool_calls = []
    all_viz = []
    final_text = ""
    rounds = 0

    for _ in range(5):  # max 5 rounds
        rounds += 1
        response = await client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=CYTOATLAS_TOOLS,
            messages=messages,
        )

        # Extract text and tool calls
        text_parts = []
        tool_uses = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(block)
                all_tool_calls.append({
                    "name": block.name,
                    "input": block.input,
                })

        if response.stop_reason != "tool_use" or not tool_uses:
            final_text = "\n".join(text_parts)
            break

        # Execute tools and continue
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in tool_uses:
            result = await executor.execute_tool(block.name, block.input)

            # Check for visualization
            if isinstance(result, dict) and "visualization" in result:
                all_viz.append({
                    "type": result["visualization"]["type"],
                    "title": result["visualization"]["title"],
                })

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result)[:4000],  # Truncate large results
            })

        messages.append({"role": "user", "content": tool_results})

    # Check if final response has text
    if not final_text:
        for block in response.content:
            if block.type == "text":
                final_text += block.text

    return {
        "question": question,
        "rounds": rounds,
        "tool_calls": all_tool_calls,
        "visualizations_from_tools": all_viz,
        "response_preview": final_text[:500] if final_text else "(no text)",
        "has_viz_tool_call": any(tc["name"] == "create_visualization" for tc in all_tool_calls),
    }


async def main():
    executor = ToolExecutor()

    print("=" * 80)
    print("  Testing all 8 questions through Claude API")
    print("=" * 80)

    results = []
    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n{'='*80}")
        print(f"  Q{i}: {question}")
        print(f"{'='*80}")

        try:
            result = await run_claude_conversation(question, executor)
            results.append(result)

            print(f"  Rounds: {result['rounds']}")
            print(f"  Tool calls ({len(result['tool_calls'])}):")
            for tc in result["tool_calls"]:
                print(f"    - {tc['name']}({json.dumps(tc['input'], separators=(',', ':'))[:120]})")
            print(f"  Has create_visualization: {result['has_viz_tool_call']}")
            if result["visualizations_from_tools"]:
                for v in result["visualizations_from_tools"]:
                    print(f"    Viz: {v['type']} — {v['title']}")
            print(f"  Response preview: {result['response_preview'][:200]}...")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"question": question, "error": str(e)})

    # Summary
    print(f"\n\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    for i, r in enumerate(results, 1):
        if "error" in r:
            print(f"  Q{i}: ERROR - {r['error'][:80]}")
            continue
        tc_names = [tc["name"] for tc in r["tool_calls"]]
        print(f"  Q{i}: {r['rounds']} rounds | tools: {tc_names} | viz: {r['has_viz_tool_call']}")

    # Save full results
    output_path = "/vf/users/parks34/projects/2secactpy/logs/claude_test_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
