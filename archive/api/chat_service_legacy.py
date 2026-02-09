"""Chat service with dual LLM backend: vLLM (OpenAI-compatible) primary, Anthropic fallback.

Implements the LLM-powered chat interface with tool use for CytoAtlas queries.
"""

import json
import logging
from datetime import datetime
from typing import Any, AsyncGenerator

from app.config import get_settings
from app.schemas.chat import (
    ChatMessageResponse,
    ChatSuggestion,
    ChatSuggestionsResponse,
    DownloadableData,
    MessageRole,
    SuggestionCategory,
    StreamChunk,
    ToolCall,
    ToolResult,
    VisualizationConfig,
    VisualizationType,
)
from app.services.context_manager import ContextManager, ConversationContext, get_context_manager
from app.services.mcp_tools import CYTOATLAS_TOOLS, OPENAI_TOOLS, ToolExecutor, get_tool_executor

logger = logging.getLogger(__name__)
settings = get_settings()


SYSTEM_PROMPT = """You are CytoAtlas Assistant, an expert in single-cell cytokine activity analysis. You have access to CytoAtlas, a comprehensive resource containing pre-computed cytokine and secreted protein activity signatures across 17+ million human immune cells from three major atlases.

## Available Atlases

1. **CIMA (Chinese Immune Multi-omics Atlas)**
   - 6.5 million cells from 421 healthy adults
   - Age range: 25-85 years
   - Available data: cytokine/protein activity, age correlations, BMI correlations, blood biochemistry correlations

2. **Inflammation Atlas**
   - ~5 million cells across 20+ inflammatory diseases
   - Three cohorts: main, validation, external
   - Available data: disease vs healthy differential activity, treatment response predictions

3. **scAtlas (Human Tissue Atlas)**
   - 6.4 million cells across 35 organs/tissues
   - Includes both normal tissues and pan-cancer immune profiling
   - Available data: organ-specific activity, tumor vs adjacent comparisons

## Signature Types

- **CytoSig**: 43 cytokine activity signatures (IFNG, TNF, IL-6, IL-17A, etc.)
- **SecAct**: 1,170 secreted protein activity signatures

## How to Help Users

When users ask questions:
1. Use the available tools to retrieve relevant data from CytoAtlas
2. Always cite the specific atlas and signature type in your responses
3. For complex patterns, create visualizations to illustrate your findings
4. If users want to export data, use the export_data tool to prepare downloads
5. Provide biological context and interpretation when relevant

## Guidelines

- Be precise about which atlas and cell types you're referencing
- When comparing across atlases, note that cell type annotations may differ
- Activity values are z-scores; positive means upregulated, negative means downregulated
- For correlations, report both the correlation coefficient and statistical significance

## Important Disclaimer

Like other AI systems, CytoAtlas Assistant can make mistakes. Key findings should be validated with conventional bioinformatics approaches and experimental verification. The activity predictions are computational inferences based on gene expression patterns, not direct measurements."""


class ChatService:
    """Service for handling chat interactions with LLM API.

    Uses vLLM (OpenAI-compatible) as primary backend, with Anthropic Claude as fallback.
    """

    def __init__(self):
        self.settings = get_settings()
        self._llm_client = None
        self._anthropic_client = None

    @property
    def use_vllm(self) -> bool:
        """Check if vLLM backend is configured."""
        return bool(self.settings.llm_base_url)

    @property
    def llm_client(self):
        """Lazy-load OpenAI-compatible LLM client for vLLM."""
        if self._llm_client is None:
            from openai import AsyncOpenAI

            self._llm_client = AsyncOpenAI(
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
            )
        return self._llm_client

    @property
    def anthropic_client(self):
        """Lazy-load async Anthropic client (fallback)."""
        if self._anthropic_client is None:
            if not self.settings.anthropic_api_key:
                raise RuntimeError(
                    "No LLM backend configured: set LLM_BASE_URL for vLLM "
                    "or ANTHROPIC_API_KEY for Claude fallback"
                )
            import anthropic

            self._anthropic_client = anthropic.AsyncAnthropic(
                api_key=self.settings.anthropic_api_key
            )
        return self._anthropic_client

    # ── Tool / visualization helpers ──────────────────────────────────

    def _process_tool_result(
        self,
        tool_name: str,
        tool_id: str,
        result: dict,
        tool_calls: list[ToolCall],
        tool_results: list[ToolResult],
        visualizations: list[VisualizationConfig],
        conversation: ConversationContext,
    ) -> tuple[ToolCall, ToolResult, DownloadableData | None]:
        """Process a single tool execution result into schema objects."""
        tool_call = ToolCall(id=tool_id, name=tool_name, arguments=result)
        # tool_call is actually built by caller; we need arguments from the call
        # This helper processes the *result* after execution

        downloadable_data = None

        # Check for visualization in result
        if "visualization" in result:
            viz_data = result["visualization"]
            visualizations.append(VisualizationConfig(
                type=VisualizationType(viz_data["type"]),
                title=viz_data.get("title"),
                data=viz_data["data"],
                config=viz_data.get("config", {}),
                container_id=viz_data.get("container_id"),
            ))

        # Check for export in result
        if "export_id" in result:
            downloadable_data = DownloadableData(
                message_id=len(conversation.messages) + 1,
                format=result.get("format", "csv"),
                description=result.get("description", "Exported data"),
            )
            conversation.cache_data(result["export_id"], result)

        tool_result = ToolResult(
            tool_call_id=tool_id,
            content=json.dumps(result) if isinstance(result, dict) else str(result),
            is_error="error" in result if isinstance(result, dict) else False,
        )

        return tool_call, tool_result, downloadable_data

    # ── vLLM (OpenAI) backend ─────────────────────────────────────────

    def _build_openai_assistant_message(
        self, content: str | None, tool_calls_raw: list[dict]
    ) -> dict:
        """Build OpenAI-format assistant message for conversation history."""
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls_raw:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": self._ensure_json_string(tc["arguments"]),
                    },
                }
                for tc in tool_calls_raw
            ]
        return msg

    @staticmethod
    def _ensure_json_string(args: str | dict) -> str:
        """Ensure tool call arguments are a valid JSON string.

        Mistral may produce duplicated/garbled JSON during streaming.
        Strategy: try as-is, then try extracting valid JSON substrings.
        """
        if isinstance(args, dict):
            return json.dumps(args)
        try:
            json.loads(args)
            return args
        except (json.JSONDecodeError, TypeError):
            pass

        # Mistral often duplicates the JSON — try each '{' as a start
        for i in range(len(args) - 1, -1, -1):
            if args[i] == '{':
                candidate = args[i:]
                try:
                    json.loads(candidate)
                    logger.info("Repaired tool args from pos %d: %s (original: %s)", i, candidate, args)
                    return candidate
                except (json.JSONDecodeError, TypeError):
                    continue

        logger.warning("Could not repair tool args: %s", args)
        return "{}"

    async def _chat_vllm(
        self,
        messages: list[dict],
        system_prompt: str,
        conversation: ConversationContext,
    ) -> tuple[str, list[ToolCall], list[ToolResult], list[VisualizationConfig], DownloadableData | None, int, int]:
        """Non-streaming chat via vLLM OpenAI-compatible API."""
        tool_executor = get_tool_executor()
        tool_calls: list[ToolCall] = []
        tool_results: list[ToolResult] = []
        visualizations: list[VisualizationConfig] = []
        downloadable_data: DownloadableData | None = None
        response_text = ""
        total_prompt_tokens = 0
        total_completion_tokens = 0

        api_messages = [{"role": "system", "content": system_prompt}] + messages

        response = await self.llm_client.chat.completions.create(
            model=self.settings.chat_model,
            max_tokens=self.settings.chat_max_tokens,
            tools=OPENAI_TOOLS,
            messages=api_messages,
        )

        if response.usage:
            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens

        # Tool use loop
        while response.choices[0].finish_reason == "tool_calls":
            message = response.choices[0].message
            if message.content:
                response_text += message.content

            # Build raw tool call list for history
            raw_tc_list = []
            current_tool_results = []

            for tc in message.tool_calls:
                args_str = tc.function.arguments
                args = json.loads(args_str)

                tool_call = ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                )
                tool_calls.append(tool_call)
                raw_tc_list.append({"id": tc.id, "name": tc.function.name, "arguments": args_str})

                # Execute tool
                result = await tool_executor.execute_tool(tc.function.name, args)

                _, tool_result, dl_data = self._process_tool_result(
                    tc.function.name, tc.id, result,
                    tool_calls, tool_results, visualizations, conversation,
                )
                tool_results.append(tool_result)
                current_tool_results.append(tool_result)
                if dl_data:
                    downloadable_data = dl_data

            # Append assistant message + tool results to conversation
            api_messages.append(self._build_openai_assistant_message(message.content, raw_tc_list))
            for tr in current_tool_results:
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": tr.tool_call_id,
                    "content": tr.content,
                })

            # Get next response
            response = await self.llm_client.chat.completions.create(
                model=self.settings.chat_model,
                max_tokens=self.settings.chat_max_tokens,
                tools=OPENAI_TOOLS,
                messages=api_messages,
            )
            if response.usage:
                total_prompt_tokens += response.usage.prompt_tokens
                total_completion_tokens += response.usage.completion_tokens

        # Extract final text
        final_message = response.choices[0].message
        if final_message.content:
            response_text += final_message.content

        return (
            response_text, tool_calls, tool_results, visualizations,
            downloadable_data, total_prompt_tokens, total_completion_tokens,
        )

    async def _chat_stream_vllm(
        self,
        messages: list[dict],
        system_prompt: str,
        conversation: ConversationContext,
        message_id: int,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Streaming chat via vLLM OpenAI-compatible API."""
        tool_executor = get_tool_executor()
        tool_calls: list[ToolCall] = []
        tool_results: list[ToolResult] = []
        visualizations: list[VisualizationConfig] = []
        full_response = ""

        api_messages = [{"role": "system", "content": system_prompt}] + messages

        needs_continuation = True
        while needs_continuation:
            needs_continuation = False

            stream = await self.llm_client.chat.completions.create(
                model=self.settings.chat_model,
                max_tokens=self.settings.chat_max_tokens,
                tools=OPENAI_TOOLS,
                messages=api_messages,
                stream=True,
            )

            # Accumulate streamed tool call chunks
            collected_tool_calls: dict[int, dict] = {}
            finish_reason = None
            streamed_content = ""

            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    full_response += delta.content
                    streamed_content += delta.content
                    yield StreamChunk(
                        type="text",
                        content=delta.content,
                        message_id=message_id,
                    )
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
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            # If the model called tools, execute them and continue
            if finish_reason == "tool_calls" and collected_tool_calls:
                needs_continuation = True
                raw_tc_list = []
                current_tool_results = []

                for idx in sorted(collected_tool_calls.keys()):
                    tc_data = collected_tool_calls[idx]
                    raw_args = tc_data["arguments"]
                    repaired = self._ensure_json_string(raw_args)
                    args = json.loads(repaired)
                    tc_data["arguments"] = repaired

                    tool_call = ToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        arguments=args,
                    )
                    tool_calls.append(tool_call)
                    raw_tc_list.append(tc_data)

                    yield StreamChunk(
                        type="tool_call",
                        tool_call=tool_call,
                        message_id=message_id,
                    )

                    result = await tool_executor.execute_tool(tc_data["name"], args)

                    _, tool_result, _ = self._process_tool_result(
                        tc_data["name"], tc_data["id"], result,
                        tool_calls, tool_results, visualizations, conversation,
                    )
                    tool_results.append(tool_result)
                    current_tool_results.append(tool_result)

                    yield StreamChunk(
                        type="tool_result",
                        tool_result=tool_result,
                        message_id=message_id,
                    )

                    if "visualization" in result:
                        viz_data = result["visualization"]
                        viz = VisualizationConfig(
                            type=VisualizationType(viz_data["type"]),
                            title=viz_data.get("title"),
                            data=viz_data["data"],
                            config=viz_data.get("config", {}),
                            container_id=viz_data.get("container_id"),
                        )
                        yield StreamChunk(
                            type="visualization",
                            visualization=viz,
                            message_id=message_id,
                        )

                # Update api_messages for continuation
                api_messages.append(
                    self._build_openai_assistant_message(streamed_content or None, raw_tc_list)
                )
                for tr in current_tool_results:
                    api_messages.append({
                        "role": "tool",
                        "tool_call_id": tr.tool_call_id,
                        "content": tr.content,
                    })

        # Save to conversation
        conversation.add_assistant_message(
            content=full_response,
            tool_calls=tool_calls if tool_calls else None,
            tool_results=tool_results if tool_results else None,
            visualizations=visualizations if visualizations else None,
        )

        yield StreamChunk(type="done", message_id=message_id)

    # ── Anthropic (Claude) fallback backend ───────────────────────────

    async def _chat_anthropic(
        self,
        messages: list[dict],
        system_prompt: str,
        conversation: ConversationContext,
    ) -> tuple[str, list[ToolCall], list[ToolResult], list[VisualizationConfig], DownloadableData | None, int, int]:
        """Non-streaming chat via Anthropic Claude API (fallback)."""
        tool_executor = get_tool_executor()
        tool_calls: list[ToolCall] = []
        tool_results: list[ToolResult] = []
        visualizations: list[VisualizationConfig] = []
        downloadable_data: DownloadableData | None = None
        response_text = ""

        response = await self.anthropic_client.messages.create(
            model=self.settings.anthropic_chat_model,
            max_tokens=self.settings.chat_max_tokens,
            system=system_prompt,
            tools=CYTOATLAS_TOOLS,
            messages=messages,
        )

        # Handle tool use loop
        while response.stop_reason == "tool_use":
            for block in response.content:
                if block.type == "text":
                    response_text += block.text
                elif block.type == "tool_use":
                    tool_call = ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                    tool_calls.append(tool_call)

                    result = await tool_executor.execute_tool(block.name, block.input)

                    if "visualization" in result:
                        viz_data = result["visualization"]
                        visualizations.append(VisualizationConfig(
                            type=VisualizationType(viz_data["type"]),
                            title=viz_data.get("title"),
                            data=viz_data["data"],
                            config=viz_data.get("config", {}),
                            container_id=viz_data.get("container_id"),
                        ))

                    if "export_id" in result:
                        downloadable_data = DownloadableData(
                            message_id=len(conversation.messages) + 1,
                            format=result.get("format", "csv"),
                            description=result.get("description", "Exported data"),
                        )
                        conversation.cache_data(result["export_id"], result)

                    tool_result = ToolResult(
                        tool_call_id=block.id,
                        content=json.dumps(result) if isinstance(result, dict) else str(result),
                        is_error="error" in result if isinstance(result, dict) else False,
                    )
                    tool_results.append(tool_result)

            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tr.tool_call_id,
                        "content": tr.content,
                    }
                    for tr in tool_results[-len([b for b in response.content if b.type == "tool_use"]):]
                ],
            })

            response = await self.anthropic_client.messages.create(
                model=self.settings.anthropic_chat_model,
                max_tokens=self.settings.chat_max_tokens,
                system=system_prompt,
                tools=CYTOATLAS_TOOLS,
                messages=messages,
            )

        # Extract final text
        for block in response.content:
            if block.type == "text":
                response_text += block.text

        return (
            response_text, tool_calls, tool_results, visualizations,
            downloadable_data, response.usage.input_tokens, response.usage.output_tokens,
        )

    async def _chat_stream_anthropic(
        self,
        messages: list[dict],
        system_prompt: str,
        conversation: ConversationContext,
        message_id: int,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Streaming chat via Anthropic Claude API (fallback)."""
        tool_executor = get_tool_executor()
        tool_calls: list[ToolCall] = []
        tool_results: list[ToolResult] = []
        visualizations: list[VisualizationConfig] = []
        full_response = ""

        async with self.anthropic_client.messages.stream(
            model=self.settings.anthropic_chat_model,
            max_tokens=self.settings.chat_max_tokens,
            system=system_prompt,
            tools=CYTOATLAS_TOOLS,
            messages=messages,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        full_response += event.delta.text
                        yield StreamChunk(
                            type="text",
                            content=event.delta.text,
                            message_id=message_id,
                        )

            response = await stream.get_final_message()

            if response.stop_reason == "tool_use":
                for block in response.content:
                    if block.type == "tool_use":
                        tool_call = ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input,
                        )
                        tool_calls.append(tool_call)

                        yield StreamChunk(
                            type="tool_call",
                            tool_call=tool_call,
                            message_id=message_id,
                        )

                        result = await tool_executor.execute_tool(block.name, block.input)

                        tool_result = ToolResult(
                            tool_call_id=block.id,
                            content=json.dumps(result) if isinstance(result, dict) else str(result),
                            is_error="error" in result if isinstance(result, dict) else False,
                        )
                        tool_results.append(tool_result)

                        yield StreamChunk(
                            type="tool_result",
                            tool_result=tool_result,
                            message_id=message_id,
                        )

                        if "visualization" in result:
                            viz_data = result["visualization"]
                            viz = VisualizationConfig(
                                type=VisualizationType(viz_data["type"]),
                                title=viz_data.get("title"),
                                data=viz_data["data"],
                                config=viz_data.get("config", {}),
                                container_id=viz_data.get("container_id"),
                            )
                            visualizations.append(viz)
                            yield StreamChunk(
                                type="visualization",
                                visualization=viz,
                                message_id=message_id,
                            )

                # Continue with tool results
                current_tool_results = list(tool_results)

                while response.stop_reason == "tool_use":
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tr.tool_call_id,
                                "content": tr.content,
                            }
                            for tr in current_tool_results
                        ],
                    })
                    current_tool_results = []

                    response = await self.anthropic_client.messages.create(
                        model=self.settings.anthropic_chat_model,
                        max_tokens=self.settings.chat_max_tokens,
                        system=system_prompt,
                        tools=CYTOATLAS_TOOLS,
                        messages=messages,
                    )

                    for block in response.content:
                        if block.type == "text":
                            full_response += block.text
                            yield StreamChunk(
                                type="text",
                                content=block.text,
                                message_id=message_id,
                            )
                        elif block.type == "tool_use":
                            tool_call = ToolCall(
                                id=block.id,
                                name=block.name,
                                arguments=block.input,
                            )
                            tool_calls.append(tool_call)

                            yield StreamChunk(
                                type="tool_call",
                                tool_call=tool_call,
                                message_id=message_id,
                            )

                            result = await tool_executor.execute_tool(block.name, block.input)

                            tool_result = ToolResult(
                                tool_call_id=block.id,
                                content=json.dumps(result) if isinstance(result, dict) else str(result),
                                is_error="error" in result if isinstance(result, dict) else False,
                            )
                            tool_results.append(tool_result)
                            current_tool_results.append(tool_result)

                            yield StreamChunk(
                                type="tool_result",
                                tool_result=tool_result,
                                message_id=message_id,
                            )

                            if "visualization" in result:
                                viz_data = result["visualization"]
                                viz = VisualizationConfig(
                                    type=VisualizationType(viz_data["type"]),
                                    title=viz_data.get("title"),
                                    data=viz_data["data"],
                                    config=viz_data.get("config", {}),
                                    container_id=viz_data.get("container_id"),
                                )
                                visualizations.append(viz)
                                yield StreamChunk(
                                    type="visualization",
                                    visualization=viz,
                                    message_id=message_id,
                                )

        conversation.add_assistant_message(
            content=full_response,
            tool_calls=tool_calls if tool_calls else None,
            tool_results=tool_results if tool_results else None,
            visualizations=visualizations if visualizations else None,
        )

        yield StreamChunk(type="done", message_id=message_id)

    # ── Public API ────────────────────────────────────────────────────

    async def chat(
        self,
        content: str,
        conversation_id: int | None = None,
        session_id: str | None = None,
        user_id: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> ChatMessageResponse:
        """Process a chat message and return the response.

        Tries vLLM first; falls back to Anthropic on connection error.
        """
        context_manager = get_context_manager()

        conversation = context_manager.get_or_create_conversation(
            conversation_id, user_id, session_id
        )
        user_message = conversation.add_user_message(content)
        messages = conversation.to_messages_for_api()

        system_prompt = SYSTEM_PROMPT
        if context:
            system_prompt += f"\n\n## Current Context\n{json.dumps(context, indent=2)}"

        # Try vLLM primary, then Anthropic fallback
        try:
            if self.use_vllm:
                (
                    response_text, tool_calls, tool_results,
                    visualizations, downloadable_data,
                    input_tokens, output_tokens,
                ) = await self._chat_vllm(messages, system_prompt, conversation)
            else:
                raise ConnectionError("vLLM not configured")
        except (ConnectionError, OSError) as e:
            logger.warning("vLLM unavailable (%s), falling back to Anthropic", e)
            try:
                (
                    response_text, tool_calls, tool_results,
                    visualizations, downloadable_data,
                    input_tokens, output_tokens,
                ) = await self._chat_anthropic(messages, system_prompt, conversation)
            except Exception:
                logger.exception("Anthropic fallback also failed")
                raise RuntimeError("All LLM backends unavailable")
        except Exception:
            logger.exception("LLM API call failed")
            raise RuntimeError("Chat service encountered an error. Please try again.")

        assistant_message = conversation.add_assistant_message(
            content=response_text,
            tool_calls=tool_calls if tool_calls else None,
            tool_results=tool_results if tool_results else None,
            visualizations=visualizations if visualizations else None,
            downloadable_data=downloadable_data,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return ChatMessageResponse(
            message_id=assistant_message.id,
            conversation_id=conversation.conversation_id,
            role=MessageRole.ASSISTANT,
            content=response_text,
            tool_calls=tool_calls if tool_calls else None,
            tool_results=tool_results if tool_results else None,
            visualizations=visualizations if visualizations else None,
            downloadable_data=downloadable_data,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            created_at=assistant_message.created_at,
        )

    async def chat_stream(
        self,
        content: str,
        conversation_id: int | None = None,
        session_id: str | None = None,
        user_id: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a chat response.

        Tries vLLM first; falls back to Anthropic on connection error.
        """
        context_manager = get_context_manager()

        conversation = context_manager.get_or_create_conversation(
            conversation_id, user_id, session_id
        )
        user_message = conversation.add_user_message(content)
        messages = conversation.to_messages_for_api()

        system_prompt = SYSTEM_PROMPT
        if context:
            system_prompt += f"\n\n## Current Context\n{json.dumps(context, indent=2)}"

        message_id = len(conversation.messages) + 1

        try:
            if self.use_vllm:
                async for chunk in self._chat_stream_vllm(
                    messages, system_prompt, conversation, message_id
                ):
                    yield chunk
                return
            else:
                raise ConnectionError("vLLM not configured")
        except (ConnectionError, OSError) as e:
            logger.warning("vLLM streaming unavailable (%s), falling back to Anthropic", e)
        except Exception:
            logger.exception("Streaming chat failed")
            yield StreamChunk(type="error", content="An unexpected error occurred. Please try again.", message_id=message_id)
            return

        # Anthropic fallback for streaming
        try:
            async for chunk in self._chat_stream_anthropic(
                messages, system_prompt, conversation, message_id
            ):
                yield chunk
        except Exception:
            logger.exception("Anthropic streaming fallback failed")
            yield StreamChunk(type="error", content="Chat service is temporarily unavailable. Please try again later.", message_id=message_id)

    def get_suggestions(self) -> ChatSuggestionsResponse:
        """Get suggested queries for the chat."""
        suggestions = [
            ChatSuggestion(
                text="What is the activity of IFNG across different immune cell types in CIMA?",
                category=SuggestionCategory.EXPLORE,
                description="Explore IFNG activity patterns",
            ),
            ChatSuggestion(
                text="Compare TNF activity between CIMA and Inflammation Atlas",
                category=SuggestionCategory.COMPARE,
                description="Cross-atlas comparison",
            ),
            ChatSuggestion(
                text="What cytokines are most differentially active in COVID-19 patients?",
                category=SuggestionCategory.ANALYZE,
                description="Disease-specific analysis",
            ),
            ChatSuggestion(
                text="How does IL-17A activity correlate with age in CD4 T cells?",
                category=SuggestionCategory.ANALYZE,
                description="Correlation analysis",
            ),
            ChatSuggestion(
                text="What are the top secreted proteins in tumor-associated macrophages?",
                category=SuggestionCategory.EXPLORE,
                description="Pan-cancer immune analysis",
            ),
            ChatSuggestion(
                text="Explain what CytoSig activity scores mean",
                category=SuggestionCategory.EXPLAIN,
                description="Learn about the methodology",
            ),
            ChatSuggestion(
                text="Show me the validation metrics for the Inflammation Atlas",
                category=SuggestionCategory.EXPLAIN,
                description="Understand data quality",
            ),
            ChatSuggestion(
                text="Which organs show the highest IL-6 activity in scAtlas?",
                category=SuggestionCategory.EXPLORE,
                description="Tissue-specific patterns",
            ),
        ]

        return ChatSuggestionsResponse(suggestions=suggestions)


# Singleton instance
_chat_service: ChatService | None = None


def get_chat_service() -> ChatService:
    """Get or create the chat service singleton."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
