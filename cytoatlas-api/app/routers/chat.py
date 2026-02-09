"""Chat router for LLM-powered natural language interface."""

import json
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.core.cache import CacheService, get_cache
from app.core.security import get_current_user_optional
from app.models.user import User
from app.schemas.chat import (
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSuggestionsResponse,
    ConversationListResponse,
    ConversationResponse,
)
from app.services.chat import ChatService, get_chat_service
from app.services.chat.conversation_service import get_conversation_service

router = APIRouter(prefix="/chat", tags=["Chat"])
settings = get_settings()

# In-memory rate limit tracking (fallback when Redis unavailable)
_rate_limit_store: dict[str, tuple[int, float]] = {}  # key -> (count, window_start)


def get_session_id(request: Request) -> str:
    """Get or create a session ID from cookies."""
    session_id = request.cookies.get("cytoatlas_session")
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())
    return session_id


async def check_rate_limit(
    user: User | None,
    session_id: str,
    cache: CacheService | None = None,
) -> None:
    """Check if the user/session has exceeded rate limits.

    Uses Redis when available, falls back to in-memory tracking.
    Authenticated users get higher limits than anonymous users.
    """
    # Determine rate limit based on authentication status
    if user:
        limit = settings.auth_chat_limit_per_day
        key = f"chat_ratelimit:user:{user.id}"
    else:
        limit = settings.anon_chat_limit_per_day
        key = f"chat_ratelimit:session:{session_id}"

    window = 86400  # 24 hours in seconds
    current_time = time.time()

    # Try Redis first if available
    if cache and cache.redis:
        try:
            cache_key = key
            current = await cache.get(cache_key)

            if current is None:
                await cache.set(cache_key, "1", ttl=window)
                return

            count = int(current)
            if count >= limit:
                ttl = await cache.redis.ttl(cache_key)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Chat rate limit exceeded. Daily limit: {limit} messages. "
                           f"Try again in {ttl // 3600} hours.",
                    headers={
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(ttl),
                        "Retry-After": str(ttl),
                    },
                )

            await cache.incr(cache_key)
            return
        except HTTPException:
            raise
        except Exception:
            pass  # Fall through to in-memory

    # In-memory fallback
    global _rate_limit_store

    # Clean up expired entries periodically
    if len(_rate_limit_store) > 1000:
        expired_keys = [
            k for k, (_, start) in _rate_limit_store.items()
            if current_time - start > window
        ]
        for k in expired_keys:
            del _rate_limit_store[k]

    if key in _rate_limit_store:
        count, window_start = _rate_limit_store[key]

        # Check if window has expired
        if current_time - window_start > window:
            _rate_limit_store[key] = (1, current_time)
            return

        if count >= limit:
            remaining_time = int(window - (current_time - window_start))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Chat rate limit exceeded. Daily limit: {limit} messages. "
                       f"Try again in {remaining_time // 3600} hours.",
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(remaining_time),
                    "Retry-After": str(remaining_time),
                },
            )

        _rate_limit_store[key] = (count + 1, window_start)
    else:
        _rate_limit_store[key] = (1, current_time)


@router.post("/message", response_model=ChatMessageResponse)
async def send_message(
    request_data: ChatMessageRequest,
    request: Request,
    current_user: User | None = Depends(get_current_user_optional),
    service: ChatService = Depends(get_chat_service),
    cache: CacheService = Depends(get_cache),
) -> ChatMessageResponse:
    """Send a message to the CytoAtlas Assistant.

    The assistant can:
    - Search for cytokines, proteins, cell types, diseases, and organs
    - Retrieve activity data from all three atlases
    - Show correlations with age, BMI, and biochemistry (CIMA)
    - Compare signatures across atlases
    - Generate visualizations
    - Prepare data for download

    **Authentication Optional** - Anonymous users have limited daily requests.

    **Examples:**
    - "What is the activity of IFNG in CD8 T cells?"
    - "Compare TNF activity between CIMA and Inflammation Atlas"
    - "Show me the top differentially active cytokines in COVID-19"
    """
    session_id = request_data.session_id or get_session_id(request)
    user_id = current_user.id if current_user else None

    # Check rate limits
    await check_rate_limit(current_user, session_id, cache)

    try:
        response = await service.chat(
            content=request_data.content,
            conversation_id=request_data.conversation_id,
            session_id=session_id,
            user_id=user_id,
            context=request_data.context,
        )

        return response

    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@router.post("/message/stream")
async def send_message_stream(
    request_data: ChatMessageRequest,
    request: Request,
    current_user: User | None = Depends(get_current_user_optional),
    service: ChatService = Depends(get_chat_service),
    cache: CacheService = Depends(get_cache),
):
    """Stream a response from the CytoAtlas Assistant.

    Returns a Server-Sent Events (SSE) stream with chunks of the response.
    Each chunk is a JSON object with a `type` field:
    - `text`: Text content being generated
    - `tool_call`: Tool being invoked
    - `tool_result`: Result from a tool
    - `visualization`: Inline visualization to render
    - `done`: Stream complete
    - `error`: Error occurred

    **Example SSE events:**
    ```
    data: {"type": "text", "content": "I'll look up the IFNG activity..."}
    data: {"type": "tool_call", "tool_call": {"name": "get_activity_data", ...}}
    data: {"type": "tool_result", "tool_result": {...}}
    data: {"type": "text", "content": "Based on the data, IFNG shows..."}
    data: {"type": "done"}
    ```
    """
    session_id = request_data.session_id or get_session_id(request)
    user_id = current_user.id if current_user else None

    await check_rate_limit(current_user, session_id, cache)

    async def generate():
        try:
            async for chunk in service.chat_stream(
                content=request_data.content,
                conversation_id=request_data.conversation_id,
                session_id=session_id,
                user_id=user_id,
                context=request_data.context,
            ):
                yield f"data: {chunk.model_dump_json()}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    request: Request = None,
    current_user: User | None = Depends(get_current_user_optional),
    conversation_service = Depends(get_conversation_service),
) -> ConversationListResponse:
    """List your chat conversations.

    Returns conversations sorted by last activity (most recent first).
    """
    session_id = get_session_id(request) if request else None
    user_id = current_user.id if current_user else None

    conversations, total = await conversation_service.list_conversations(
        user_id=user_id,
        session_id=session_id,
        offset=offset,
        limit=limit,
    )

    return ConversationListResponse(
        conversations=conversations,
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: int,
    request: Request,
    current_user: User | None = Depends(get_current_user_optional),
    conversation_service = Depends(get_conversation_service),
) -> ConversationResponse:
    """Get a specific conversation with all messages.

    Returns the full conversation history including tool calls,
    visualizations, and downloadable data references.
    """
    session_id = get_session_id(request)
    user_id = current_user.id if current_user else None

    try:
        await conversation_service.get_or_create_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            session_id=session_id,
        )
        return await conversation_service.get_conversation_response(conversation_id)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Not authorized for this conversation")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    request: Request,
    current_user: User | None = Depends(get_current_user_optional),
    conversation_service = Depends(get_conversation_service),
) -> dict[str, Any]:
    """Delete a conversation.

    Permanently removes the conversation and all its messages.
    """
    session_id = get_session_id(request)
    user_id = current_user.id if current_user else None

    # Note: delete_conversation not yet implemented in ConversationService
    # For now, return not implemented
    raise HTTPException(
        status_code=501,
        detail="Conversation deletion not yet implemented in refactored service"
    )


@router.get("/suggestions", response_model=ChatSuggestionsResponse)
async def get_suggestions(
    service: ChatService = Depends(get_chat_service),
) -> ChatSuggestionsResponse:
    """Get suggested queries for the chat.

    Returns a list of example queries across different categories:
    - explore: Data exploration queries
    - compare: Cross-atlas comparisons
    - analyze: Analysis and correlation queries
    - explain: Questions about methodology
    """
    return service.get_suggestions()


@router.post("/download/{message_id}")
async def download_message_data(
    message_id: int,
    conversation_id: Annotated[int, Query(description="Conversation ID")],
    export_id: Annotated[str, Query(description="Export ID from tool result")],
    format: Annotated[str, Query(description="Export format")] = "csv",
    request: Request = None,
    current_user: User | None = Depends(get_current_user_optional),
    conversation_service = Depends(get_conversation_service),
):
    """Download data from a chat message.

    When the assistant prepares data for export using the export_data tool,
    this endpoint generates the actual downloadable file.

    **Parameters:**
    - `message_id`: ID of the message containing the export
    - `conversation_id`: ID of the conversation
    - `export_id`: Export ID from the tool result
    - `format`: Export format (csv or json)
    """
    session_id = get_session_id(request) if request else None
    user_id = current_user.id if current_user else None

    try:
        await conversation_service.get_or_create_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            session_id=session_id,
        )
    except PermissionError:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Get cached data
    cached = conversation_service.get_cached_data(conversation_id, export_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Export data not found or expired")

    # Generate file based on format
    if format == "csv":
        import csv
        import io

        output = io.StringIO()
        # Assume cached data has a standard structure
        data = cached.get("data", cached)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        else:
            # Flatten to CSV
            output.write(json.dumps(data))

        content = output.getvalue()
        media_type = "text/csv"
        filename = f"cytoatlas_export_{export_id}.csv"

    else:  # JSON
        content = json.dumps(cached, indent=2)
        media_type = "application/json"
        filename = f"cytoatlas_export_{export_id}.json"

    return StreamingResponse(
        iter([content]),
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
        },
    )


@router.get("/status")
async def chat_status() -> dict[str, Any]:
    """Get chat service status.

    Returns information about the chat service including:
    - Whether the LLM API is configured
    - Available tools
    - Rate limit information
    """
    api_configured = bool(settings.llm_base_url or settings.anthropic_api_key)

    return {
        "status": "operational" if api_configured else "limited",
        "llm_configured": api_configured,
        "model": settings.chat_model if api_configured else None,
        "tools_available": len([t["name"] for t in __import__("app.services.mcp_tools", fromlist=["CYTOATLAS_TOOLS"]).CYTOATLAS_TOOLS]),
        "rate_limits": {
            "anonymous": f"{settings.anon_chat_limit_per_day} messages/day",
            "authenticated": f"{settings.auth_chat_limit_per_day} messages/day",
        },
    }
