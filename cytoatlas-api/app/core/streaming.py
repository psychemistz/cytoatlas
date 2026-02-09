"""Streaming response for large datasets."""

import time
from collections.abc import AsyncIterator
from typing import Any

import orjson
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse


class StreamingJSONResponse(StreamingResponse):
    """
    Streaming JSON response for large datasets.

    Streams JSON arrays item by item to avoid loading entire dataset in memory.
    Includes ETag and Last-Modified headers for caching.
    """

    def __init__(
        self,
        content: AsyncIterator[dict] | list[dict],
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        media_type: str = "application/json",
        background: BackgroundTask | None = None,
        etag: str | None = None,
        last_modified: int | None = None,
    ):
        """
        Initialize streaming JSON response.

        Args:
            content: Async iterator or list of items to stream
            status_code: HTTP status code
            headers: Additional headers
            media_type: Media type
            background: Background task
            etag: ETag header value
            last_modified: Last-Modified timestamp (unix time)
        """
        # Convert list to async iterator if needed
        if isinstance(content, list):
            content = self._list_to_async_iterator(content)

        # Create streaming content
        streaming_content = stream_json_array(content)

        # Initialize headers
        init_headers = headers or {}

        # Add caching headers
        if etag:
            init_headers["ETag"] = etag

        if last_modified:
            # Convert unix timestamp to HTTP date format
            time_str = time.strftime(
                "%a, %d %b %Y %H:%M:%S GMT",
                time.gmtime(last_modified),
            )
            init_headers["Last-Modified"] = time_str

        super().__init__(
            content=streaming_content,
            status_code=status_code,
            headers=init_headers,
            media_type=media_type,
            background=background,
        )

    @staticmethod
    async def _list_to_async_iterator(items: list) -> AsyncIterator[Any]:
        """Convert list to async iterator."""
        for item in items:
            yield item


async def stream_json_array(items: AsyncIterator[dict]) -> AsyncIterator[bytes]:
    """
    Stream JSON array item by item.

    Yields JSON array opening bracket, items separated by commas,
    and closing bracket.

    Args:
        items: Async iterator of items

    Yields:
        JSON bytes
    """
    # Opening bracket
    yield b"["

    first = True
    async for item in items:
        # Add comma separator (except for first item)
        if not first:
            yield b","
        first = False

        # Serialize item
        json_bytes = orjson.dumps(item)
        yield json_bytes

    # Closing bracket
    yield b"]"


async def stream_json_objects(
    items: AsyncIterator[dict],
    separator: str = "\n",
) -> AsyncIterator[bytes]:
    """
    Stream JSON objects line by line (JSONL format).

    Each item is a complete JSON object on its own line.
    Useful for large datasets where JSON array parsing is memory-intensive.

    Args:
        items: Async iterator of items
        separator: Line separator (default: newline)

    Yields:
        JSON bytes
    """
    async for item in items:
        json_bytes = orjson.dumps(item)
        yield json_bytes + separator.encode()


def create_streaming_response(
    items: list[dict] | AsyncIterator[dict],
    format: str = "array",
    etag: str | None = None,
    last_modified: int | None = None,
) -> StreamingResponse:
    """
    Create streaming JSON response.

    Args:
        items: Items to stream
        format: "array" (JSON array) or "jsonl" (line-delimited JSON)
        etag: ETag header value
        last_modified: Last-Modified timestamp

    Returns:
        StreamingResponse
    """
    # Convert list to async iterator if needed
    if isinstance(items, list):

        async def _list_to_iter():
            for item in items:
                yield item

        items_iter = _list_to_iter()
    else:
        items_iter = items

    # Create streaming content
    if format == "array":
        content = stream_json_array(items_iter)
        media_type = "application/json"
    elif format == "jsonl":
        content = stream_json_objects(items_iter)
        media_type = "application/x-ndjson"
    else:
        raise ValueError(f"Unknown format: {format}")

    # Build headers
    headers = {}
    if etag:
        headers["ETag"] = etag
    if last_modified:
        time_str = time.strftime(
            "%a, %d %b %Y %H:%M:%S GMT",
            time.gmtime(last_modified),
        )
        headers["Last-Modified"] = time_str

    return StreamingResponse(
        content=content,
        media_type=media_type,
        headers=headers,
    )
