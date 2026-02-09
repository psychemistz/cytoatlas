"""Unit tests for repository pattern implementations."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from app.repositories.base import BaseRepository
from app.repositories.json_repository import JSONRepository
from app.repositories.protocols import CursorPage


class ConcreteRepository(BaseRepository):
    """Concrete implementation for testing abstract BaseRepository."""
    pass


class TestBaseRepository:
    """Tests for BaseRepository shared functionality."""

    def test_apply_filters_exact_match(self):
        """Filter by exact value match."""
        repo = ConcreteRepository()
        data = [
            {"cell_type": "CD4_T", "value": 1},
            {"cell_type": "CD8_T", "value": 2},
            {"cell_type": "CD4_T", "value": 3},
        ]

        result = repo.apply_filters(data, {"cell_type": "CD4_T"})
        assert len(result) == 2
        assert all(d["cell_type"] == "CD4_T" for d in result)

    def test_apply_filters_list_values(self):
        """Filter by list of values (OR logic)."""
        repo = ConcreteRepository()
        data = [
            {"cell_type": "CD4_T", "value": 1},
            {"cell_type": "CD8_T", "value": 2},
            {"cell_type": "Monocytes", "value": 3},
        ]

        result = repo.apply_filters(data, {"cell_type": ["CD4_T", "CD8_T"]})
        assert len(result) == 2

    def test_apply_filters_none_value_skipped(self):
        """None filter values are skipped."""
        repo = ConcreteRepository()
        data = [{"cell_type": "CD4_T", "value": 1}]

        result = repo.apply_filters(data, {"cell_type": None})
        assert len(result) == 1  # Not filtered

    def test_apply_filters_empty_dict(self):
        """Empty filter dict returns all data."""
        repo = ConcreteRepository()
        data = [{"a": 1}, {"a": 2}]

        result = repo.apply_filters(data, {})
        assert len(result) == 2

    def test_apply_filters_multiple_conditions(self):
        """Multiple filter conditions (AND logic)."""
        repo = ConcreteRepository()
        data = [
            {"cell_type": "CD4_T", "signature_type": "CytoSig"},
            {"cell_type": "CD4_T", "signature_type": "SecAct"},
            {"cell_type": "CD8_T", "signature_type": "CytoSig"},
        ]

        result = repo.apply_filters(data, {
            "cell_type": "CD4_T",
            "signature_type": "CytoSig",
        })
        assert len(result) == 1

    def test_cursor_pagination(self):
        """Cursor-based pagination returns correct pages."""
        repo = ConcreteRepository()
        data = [{"id": i} for i in range(50)]

        # First page
        page = repo.paginate_cursor(data, cursor=None, limit=10)
        assert len(page.items) == 10
        assert page.items[0]["id"] == 0
        assert page.next_cursor is not None

        # Second page
        page2 = repo.paginate_cursor(data, cursor=page.next_cursor, limit=10)
        assert len(page2.items) == 10
        assert page2.items[0]["id"] == 10

    def test_cursor_pagination_last_page(self):
        """Last page has no next_cursor."""
        repo = ConcreteRepository()
        data = [{"id": i} for i in range(15)]

        # Last page
        page = repo.paginate_cursor(data, cursor=None, limit=20)
        assert len(page.items) == 15
        assert page.next_cursor is None

    def test_cursor_pagination_with_total(self):
        """Total count is included when requested."""
        repo = ConcreteRepository()
        data = [{"id": i} for i in range(50)]

        page = repo.paginate_cursor(data, limit=10, include_total=True)
        assert page.total == 50

    def test_cursor_encode_decode(self):
        """Cursor encoding and decoding roundtrips."""
        repo = ConcreteRepository()
        cursor = repo._encode_cursor(42)
        offset = repo._decode_cursor(cursor)
        assert offset == 42

    async def test_stream_items(self):
        """stream_items yields all items."""
        repo = ConcreteRepository()
        data = [{"id": 1}, {"id": 2}, {"id": 3}]

        items = []
        async for item in repo.stream_items(data):
            items.append(item)

        assert len(items) == 3
        assert items[0]["id"] == 1


class TestJSONRepository:
    """Tests for JSONRepository with LRU cache."""

    @pytest.fixture(autouse=True)
    def setup_repo(self, tmp_path):
        """Set up a JSONRepository with test data."""
        self.repo = JSONRepository(max_cache_entries=3, max_cache_bytes=1024 * 1024)
        self.data_dir = tmp_path / "data"
        self.data_dir.mkdir()

        # Patch settings
        from app.config import get_settings
        settings = get_settings()
        self.original_path = settings.viz_data_path
        object.__setattr__(settings, "viz_data_path", self.data_dir)

        yield

        object.__setattr__(settings, "viz_data_path", self.original_path)

    def _write_json(self, filename, data):
        """Helper to write JSON files to test directory."""
        import orjson
        with open(self.data_dir / filename, "wb") as f:
            f.write(orjson.dumps(data))

    async def test_load_json_basic(self):
        """Loading a JSON file returns correct data."""
        self._write_json("test.json", [{"a": 1}, {"b": 2}])
        data = await self.repo._load_json("test.json")
        assert len(data) == 2
        assert data[0]["a"] == 1

    async def test_load_json_caches(self):
        """Second load uses cache (no disk read)."""
        self._write_json("cached.json", [{"val": 42}])

        # First load
        data1 = await self.repo._load_json("cached.json")
        assert len(self.repo._cache) == 1

        # Second load (should use cache)
        data2 = await self.repo._load_json("cached.json")
        assert data1 == data2
        assert len(self.repo._cache) == 1

    async def test_load_json_file_not_found(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            await self.repo._load_json("missing.json")

    async def test_lru_eviction(self):
        """Oldest entry is evicted when max_cache_entries is reached."""
        # max_cache_entries = 3
        self._write_json("f1.json", [1])
        self._write_json("f2.json", [2])
        self._write_json("f3.json", [3])
        self._write_json("f4.json", [4])

        await self.repo._load_json("f1.json")
        await self.repo._load_json("f2.json")
        await self.repo._load_json("f3.json")

        assert len(self.repo._cache) == 3

        # Loading f4 should evict f1
        await self.repo._load_json("f4.json")
        assert len(self.repo._cache) == 3

        # f1 should be evicted
        path_f1 = str(self.data_dir / "f1.json")
        assert path_f1 not in self.repo._cache

    async def test_clear_cache(self):
        """clear_cache removes all cached entries."""
        self._write_json("test.json", [1])
        await self.repo._load_json("test.json")

        self.repo.clear_cache()
        assert len(self.repo._cache) == 0
        assert self.repo._current_cache_bytes == 0

    async def test_get_cache_stats(self):
        """get_cache_stats returns correct metrics."""
        self._write_json("test.json", [1])
        await self.repo._load_json("test.json")

        stats = self.repo.get_cache_stats()
        assert stats["entries"] == 1
        assert stats["bytes"] > 0

    async def test_get_activity(self):
        """get_activity loads and filters data."""
        self._write_json("cima_activity.json", [
            {"signature_type": "CytoSig", "cell_type": "CD4_T", "value": 1},
            {"signature_type": "SecAct", "cell_type": "CD4_T", "value": 2},
        ])

        result = await self.repo.get_activity("cima", "CytoSig")
        assert len(result) == 1
        assert result[0]["signature_type"] == "CytoSig"

    async def test_get_activity_unknown_atlas(self):
        """get_activity raises for unknown atlas."""
        with pytest.raises(ValueError, match="Unknown atlas"):
            await self.repo.get_activity("unknown_atlas", "CytoSig")

    async def test_load_json_with_subdir(self):
        """Loading JSON from a subdirectory works."""
        subdir = self.data_dir / "validation"
        subdir.mkdir()
        import orjson
        with open(subdir / "sub_test.json", "wb") as f:
            f.write(orjson.dumps({"subdir": True}))

        data = await self.repo._load_json("sub_test.json", subdir="validation")
        assert data["subdir"] is True
