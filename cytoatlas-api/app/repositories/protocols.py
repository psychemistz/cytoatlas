"""Protocol definitions for repository pattern (PEP 544)."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class CursorPage:
    """Cursor-based pagination response."""

    items: list[dict]
    next_cursor: str | None
    total: int | None = None  # Optional, may be expensive to compute


class AtlasRepository(Protocol):
    """
    Protocol for atlas data repositories.

    This defines the interface all repositories must implement,
    allowing for different storage backends (JSON, Parquet, database).
    """

    async def get_activity(
        self,
        atlas: str,
        signature_type: str,
        **filters: Any,
    ) -> list[dict]:
        """
        Get activity data for an atlas.

        Args:
            atlas: Atlas name (e.g., "cima", "inflammation", "scatlas")
            signature_type: Type of signature ("cytosig" or "secact")
            **filters: Additional filters (cell_type, organ, etc.)

        Returns:
            List of activity records
        """
        ...

    async def get_correlations(
        self,
        atlas: str,
        variable: str,
        **filters: Any,
    ) -> list[dict]:
        """
        Get correlation data for an atlas.

        Args:
            atlas: Atlas name
            variable: Variable to correlate (e.g., "age", "bmi", "metabolite")
            **filters: Additional filters

        Returns:
            List of correlation records
        """
        ...

    async def get_differential(
        self,
        atlas: str,
        comparison: str,
        **filters: Any,
    ) -> list[dict]:
        """
        Get differential activity data.

        Args:
            atlas: Atlas name
            comparison: Type of comparison (e.g., "disease", "treatment", "cancer")
            **filters: Additional filters

        Returns:
            List of differential records
        """
        ...

    async def get_data(
        self,
        data_type: str,
        **filters: Any,
    ) -> list[dict]:
        """
        Generic method to get any data type.

        Args:
            data_type: Type of data to fetch
            **filters: Filters to apply

        Returns:
            List of records
        """
        ...

    async def stream_results(
        self,
        data_type: str,
        **filters: Any,
    ) -> AsyncIterator[dict]:
        """
        Stream results for large datasets.

        Args:
            data_type: Type of data to fetch
            **filters: Filters to apply

        Yields:
            Individual records
        """
        ...
