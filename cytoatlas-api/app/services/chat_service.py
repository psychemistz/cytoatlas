"""Chat service with Claude API integration.

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
from app.services.mcp_tools import CYTOATLAS_TOOLS, ToolExecutor, get_tool_executor

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
    """Service for handling chat interactions with Claude API."""

    def __init__(self):
        self.settings = get_settings()
        self._anthropic_client = None

    @property
    def anthropic_client(self):
        """Lazy-load Anthropic client."""
        if self._anthropic_client is None:
            if not self.settings.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not configured")

            import anthropic
            self._anthropic_client = anthropic.Anthropic(
                api_key=self.settings.anthropic_api_key
            )
        return self._anthropic_client

    async def chat(
        self,
        content: str,
        conversation_id: int | None = None,
        session_id: str | None = None,
        user_id: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> ChatMessageResponse:
        """Process a chat message and return the response.

        Args:
            content: User's message content
            conversation_id: Existing conversation ID (optional)
            session_id: Session ID for anonymous users
            user_id: User ID for authenticated users
            context: Additional context (current page, atlas, etc.)

        Returns:
            ChatMessageResponse with assistant's reply
        """
        context_manager = get_context_manager()
        tool_executor = get_tool_executor()

        # Get or create conversation
        conversation = context_manager.get_or_create_conversation(
            conversation_id, user_id, session_id
        )

        # Add user message
        user_message = conversation.add_user_message(content)

        # Build messages for API
        messages = conversation.to_messages_for_api()

        # Add context to system prompt if provided
        system_prompt = SYSTEM_PROMPT
        if context:
            system_prompt += f"\n\n## Current Context\n{json.dumps(context, indent=2)}"

        # Call Claude API
        try:
            response = self.anthropic_client.messages.create(
                model=self.settings.chat_model,
                max_tokens=self.settings.chat_max_tokens,
                system=system_prompt,
                tools=CYTOATLAS_TOOLS,
                messages=messages,
            )
        except Exception as e:
            logger.exception("Claude API call failed")
            raise RuntimeError(f"Chat service error: {str(e)}")

        # Process response
        tool_calls: list[ToolCall] = []
        tool_results: list[ToolResult] = []
        visualizations: list[VisualizationConfig] = []
        downloadable_data: DownloadableData | None = None
        response_text = ""

        # Handle tool use loop
        while response.stop_reason == "tool_use":
            # Extract tool uses from response
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

                    # Execute tool
                    result = await tool_executor.execute_tool(block.name, block.input)

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
                        # Cache the data for download
                        conversation.cache_data(result["export_id"], result)

                    tool_result = ToolResult(
                        tool_call_id=block.id,
                        content=json.dumps(result) if isinstance(result, dict) else str(result),
                        is_error="error" in result if isinstance(result, dict) else False,
                    )
                    tool_results.append(tool_result)

            # Continue conversation with tool results
            messages.append({
                "role": "assistant",
                "content": response.content,
            })
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

            # Get next response
            response = self.anthropic_client.messages.create(
                model=self.settings.chat_model,
                max_tokens=self.settings.chat_max_tokens,
                system=system_prompt,
                tools=CYTOATLAS_TOOLS,
                messages=messages,
            )

        # Extract final text response
        for block in response.content:
            if block.type == "text":
                response_text += block.text

        # Add assistant message to conversation
        assistant_message = conversation.add_assistant_message(
            content=response_text,
            tool_calls=tool_calls if tool_calls else None,
            tool_results=tool_results if tool_results else None,
            visualizations=visualizations if visualizations else None,
            downloadable_data=downloadable_data,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
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
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
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

        Yields StreamChunk objects as the response is generated.
        """
        context_manager = get_context_manager()
        tool_executor = get_tool_executor()

        # Get or create conversation
        conversation = context_manager.get_or_create_conversation(
            conversation_id, user_id, session_id
        )

        # Add user message
        user_message = conversation.add_user_message(content)

        # Build messages for API
        messages = conversation.to_messages_for_api()

        system_prompt = SYSTEM_PROMPT
        if context:
            system_prompt += f"\n\n## Current Context\n{json.dumps(context, indent=2)}"

        tool_calls: list[ToolCall] = []
        tool_results: list[ToolResult] = []
        visualizations: list[VisualizationConfig] = []
        full_response = ""
        message_id = len(conversation.messages) + 1

        try:
            # Stream response
            with self.anthropic_client.messages.stream(
                model=self.settings.chat_model,
                max_tokens=self.settings.chat_max_tokens,
                system=system_prompt,
                tools=CYTOATLAS_TOOLS,
                messages=messages,
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            full_response += event.delta.text
                            yield StreamChunk(
                                type="text",
                                content=event.delta.text,
                                message_id=message_id,
                            )

                # Get final message for tool handling
                response = stream.get_final_message()

                # Handle tool use
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

                            # Execute tool
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

                            # Check for visualization
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

                    # Continue with tool results - loop until no more tool use
                    current_tool_results = [tr for tr in tool_results]  # Copy current results

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
                        current_tool_results = []  # Reset for next round

                        # Get next response
                        response = self.anthropic_client.messages.create(
                            model=self.settings.chat_model,
                            max_tokens=self.settings.chat_max_tokens,
                            system=system_prompt,
                            tools=CYTOATLAS_TOOLS,
                            messages=messages,
                        )

                        # Process response - could be more text or more tool calls
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

                                # Execute tool
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

                                # Check for visualization
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

        except Exception as e:
            logger.exception("Streaming chat failed")
            yield StreamChunk(
                type="error",
                content=str(e),
                message_id=message_id,
            )
            return

        # Add assistant message to conversation
        conversation.add_assistant_message(
            content=full_response,
            tool_calls=tool_calls if tool_calls else None,
            tool_results=tool_results if tool_results else None,
            visualizations=visualizations if visualizations else None,
        )

        yield StreamChunk(
            type="done",
            message_id=message_id,
        )

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
