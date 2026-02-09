"""Base repository implementation with shared functionality."""

import hashlib
from abc import ABC
from collections.abc import AsyncIterator
from typing import Any

from app.repositories.protocols import CursorPage


class BaseRepository(ABC):
    """
    Abstract base repository with common pagination, filtering, and caching logic.

    Subclasses should implement specific data loading methods.
    """

    def apply_filters(self, data: list[dict], filters: dict[str, Any]) -> list[dict]:
        """
        Apply filters to data.

        Args:
            data: List of dictionaries to filter
            filters: Dictionary of field -> value filters

        Returns:
            Filtered data
        """
        if not filters:
            return data

        filtered = data
        for key, value in filters.items():
            if value is None:
                continue

            # Handle list values (OR logic)
            if isinstance(value, list):
                filtered = [d for d in filtered if d.get(key) in value]
            # Handle single value (exact match)
            else:
                filtered = [d for d in filtered if d.get(key) == value]

        return filtered

    def paginate_cursor(
        self,
        data: list[dict],
        cursor: str | None = None,
        limit: int = 100,
        include_total: bool = False,
    ) -> CursorPage:
        """
        Implement cursor-based pagination.

        The cursor is a base64-encoded offset for simplicity.
        For production, this could be improved with keyset pagination.

        Args:
            data: Full dataset
            cursor: Cursor string (base64 encoded offset)
            limit: Number of items per page
            include_total: Whether to include total count (expensive)

        Returns:
            CursorPage with items and next cursor
        """
        # Decode cursor to offset
        offset = 0
        if cursor:
            try:
                offset = int(self._decode_cursor(cursor))
            except (ValueError, TypeError):
                offset = 0

        # Slice data
        end = offset + limit
        items = data[offset:end]

        # Generate next cursor
        next_cursor = None
        if end < len(data):
            next_cursor = self._encode_cursor(end)

        # Optionally include total
        total = len(data) if include_total else None

        return CursorPage(
            items=items,
            next_cursor=next_cursor,
            total=total,
        )

    def _encode_cursor(self, offset: int) -> str:
        """Encode offset as cursor."""
        return hashlib.md5(str(offset).encode()).hexdigest()[:16] + f"-{offset}"

    def _decode_cursor(self, cursor: str) -> int:
        """Decode cursor to offset."""
        # Format: {hash}-{offset}
        if "-" in cursor:
            return int(cursor.split("-")[-1])
        return 0

    async def stream_items(self, data: list[dict]) -> AsyncIterator[dict]:
        """
        Stream items one by one.

        Args:
            data: List of items to stream

        Yields:
            Individual items
        """
        for item in data:
            yield item
