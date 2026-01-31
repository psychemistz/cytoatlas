"""WebSocket router for real-time updates."""

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query

from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["WebSocket"])

settings = get_settings()


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        # job_id -> list of websocket connections
        self.job_connections: dict[int, list[WebSocket]] = {}
        # chat session_id -> websocket connection
        self.chat_connections: dict[str, WebSocket] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def connect_job(self, websocket: WebSocket, job_id: int) -> None:
        """Connect a WebSocket to a job's progress updates."""
        await websocket.accept()
        async with self._lock:
            if job_id not in self.job_connections:
                self.job_connections[job_id] = []
            self.job_connections[job_id].append(websocket)
        logger.info(f"WebSocket connected to job {job_id}")

    async def disconnect_job(self, websocket: WebSocket, job_id: int) -> None:
        """Disconnect a WebSocket from a job's progress updates."""
        async with self._lock:
            if job_id in self.job_connections:
                if websocket in self.job_connections[job_id]:
                    self.job_connections[job_id].remove(websocket)
                if not self.job_connections[job_id]:
                    del self.job_connections[job_id]
        logger.info(f"WebSocket disconnected from job {job_id}")

    async def broadcast_job_progress(self, job_id: int, data: dict[str, Any]) -> None:
        """Broadcast progress update to all connections watching a job."""
        async with self._lock:
            connections = self.job_connections.get(job_id, [])

        dead_connections = []
        for connection in connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                dead_connections.append(connection)

        # Clean up dead connections
        if dead_connections:
            async with self._lock:
                for conn in dead_connections:
                    if job_id in self.job_connections and conn in self.job_connections[job_id]:
                        self.job_connections[job_id].remove(conn)

    async def connect_chat(self, websocket: WebSocket, session_id: str) -> None:
        """Connect a WebSocket for chat streaming."""
        await websocket.accept()
        async with self._lock:
            self.chat_connections[session_id] = websocket
        logger.info(f"WebSocket connected for chat session {session_id}")

    async def disconnect_chat(self, session_id: str) -> None:
        """Disconnect a chat WebSocket."""
        async with self._lock:
            if session_id in self.chat_connections:
                del self.chat_connections[session_id]
        logger.info(f"WebSocket disconnected for chat session {session_id}")

    async def send_chat_message(self, session_id: str, data: dict[str, Any]) -> bool:
        """Send a message to a chat session."""
        async with self._lock:
            websocket = self.chat_connections.get(session_id)

        if websocket:
            try:
                await websocket.send_json(data)
                return True
            except Exception as e:
                logger.warning(f"Failed to send chat message: {e}")
                await self.disconnect_chat(session_id)
        return False


# Global connection manager instance
manager = ConnectionManager()


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager."""
    return manager


@router.websocket("/ws/jobs/{job_id}")
async def job_progress_websocket(
    websocket: WebSocket,
    job_id: int,
) -> None:
    """WebSocket endpoint for job progress updates.

    Connect to receive real-time progress updates for a processing job.
    Messages are JSON objects with:
    - status: Current job status (processing, completed, failed)
    - progress: Progress percentage (0-100)
    - step: Current processing step description
    - error: Error message if failed
    """
    await manager.connect_job(websocket, job_id)

    # Also subscribe to Redis pub/sub if available
    redis_task = None
    if settings.use_redis:
        redis_task = asyncio.create_task(
            _subscribe_redis_job(websocket, job_id)
        )

    try:
        while True:
            # Keep connection alive, handle any client messages
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0  # 30 second timeout
                )
                # Client can send "ping" to keep alive
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send keepalive
                try:
                    await websocket.send_json({"type": "keepalive"})
                except Exception:
                    break
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        await manager.disconnect_job(websocket, job_id)
        if redis_task:
            redis_task.cancel()


async def _subscribe_redis_job(websocket: WebSocket, job_id: int) -> None:
    """Subscribe to Redis pub/sub for job progress."""
    try:
        import redis.asyncio as redis

        r = redis.from_url(settings.redis_url)
        pubsub = r.pubsub()
        await pubsub.subscribe(f"job:{job_id}:progress")

        async for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                await websocket.send_json(data)
    except Exception as e:
        logger.warning(f"Redis subscription error: {e}")


@router.websocket("/ws/chat/{session_id}")
async def chat_stream_websocket(
    websocket: WebSocket,
    session_id: str,
) -> None:
    """WebSocket endpoint for chat streaming.

    Connect to receive real-time streaming responses from the chat assistant.
    Messages are JSON objects with:
    - type: Message type (text, tool_call, tool_result, visualization, done)
    - content: Message content (varies by type)
    - message_id: ID of the message being streamed

    Client can send:
    - {"type": "message", "content": "user message"}
    - {"type": "cancel"} to stop generation
    - "ping" for keepalive
    """
    await manager.connect_chat(websocket, session_id)

    try:
        while True:
            try:
                data = await websocket.receive_text()

                # Handle keepalive
                if data == "ping":
                    await websocket.send_text("pong")
                    continue

                # Parse JSON message
                try:
                    message = json.loads(data)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "content": "Invalid JSON"
                    })
                    continue

                msg_type = message.get("type")

                if msg_type == "message":
                    # Handle user message - delegate to chat service
                    # This would trigger streaming response
                    await websocket.send_json({
                        "type": "ack",
                        "content": "Message received, processing..."
                    })
                    # The chat service will use send_chat_message to stream responses

                elif msg_type == "cancel":
                    await websocket.send_json({
                        "type": "cancelled",
                        "content": "Generation cancelled"
                    })

            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "keepalive"})

    except WebSocketDisconnect:
        logger.info(f"Chat WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Chat WebSocket error: {e}")
    finally:
        await manager.disconnect_chat(session_id)


@router.get("/ws/status")
async def websocket_status() -> dict[str, Any]:
    """Get WebSocket connection statistics."""
    return {
        "job_connections": {
            job_id: len(connections)
            for job_id, connections in manager.job_connections.items()
        },
        "chat_connections": len(manager.chat_connections),
        "total_connections": (
            sum(len(c) for c in manager.job_connections.values())
            + len(manager.chat_connections)
        ),
    }
