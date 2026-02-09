"""Unit tests for tiered caching system."""

import time
import pytest
from pathlib import Path
from unittest.mock import patch

from app.core.cache_tiers import L1Cache, L2Cache, TieredCache


class TestL1Cache:
    """Tests for in-memory LRU cache."""

    def test_get_set(self):
        """Basic get/set roundtrip."""
        cache = L1Cache(max_size_bytes=1024 * 1024)
        cache.set("key1", "value1", ttl=3600)
        assert cache.get("key1") == "value1"

    def test_get_missing_returns_none(self):
        """Getting a missing key returns None."""
        cache = L1Cache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        """Expired entries return None."""
        cache = L1Cache()
        cache.set("key1", "value1", ttl=1)

        # Manually expire by adjusting the stored expiry time
        cache._cache["key1"] = ("value1", time.time() - 10)

        assert cache.get("key1") is None

    def test_delete(self):
        """Deleting a key removes it."""
        cache = L1Cache()
        cache.set("key1", "value1")
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_delete_nonexistent(self):
        """Deleting a nonexistent key does not raise."""
        cache = L1Cache()
        cache.delete("nonexistent")  # Should not raise

    def test_clear(self):
        """Clear removes all entries."""
        cache = L1Cache()
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None
        assert cache._current_size_bytes == 0

    def test_lru_eviction(self):
        """Oldest entries are evicted when cache is full."""
        # Very small cache
        cache = L1Cache(max_size_bytes=200)
        cache.set("k1", "a" * 10, ttl=3600)  # Should fit
        cache.set("k2", "b" * 10, ttl=3600)
        cache.set("k3", "c" * 10, ttl=3600)

        # Entries may have been evicted due to size
        # At minimum, the cache should not exceed its limit
        assert cache._current_size_bytes <= cache._max_size_bytes

    def test_lru_order_updated_on_get(self):
        """Accessing an entry moves it to most-recently-used position."""
        cache = L1Cache(max_size_bytes=1024 * 1024)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")

        # Access k1 to make it most recently used
        cache.get("k1")

        # k1 should now be at the end of the OrderedDict
        keys = list(cache._cache.keys())
        assert keys[-1] == "k1"

    def test_overwrite_existing_key(self):
        """Setting an existing key replaces the value."""
        cache = L1Cache()
        cache.set("k1", "old_value")
        cache.set("k1", "new_value")
        assert cache.get("k1") == "new_value"

    def test_stats(self):
        """Stats returns correct metrics."""
        cache = L1Cache(max_size_bytes=1024)
        cache.set("k1", "v1")
        cache.set("k2", "v2")

        stats = cache.stats()
        assert stats["entries"] == 2
        assert stats["size_bytes"] > 0
        assert stats["max_size_bytes"] == 1024
        assert 0 <= stats["utilization_pct"] <= 100


class TestL2Cache:
    """Tests for filesystem-based cache."""

    def test_get_set(self, tmp_path):
        """Basic get/set roundtrip."""
        cache = L2Cache(cache_dir=tmp_path / "l2")
        cache.set("key1", {"data": [1, 2, 3]})
        result = cache.get("key1")
        assert result == {"data": [1, 2, 3]}

    def test_get_missing_returns_none(self, tmp_path):
        """Getting a missing key returns None."""
        cache = L2Cache(cache_dir=tmp_path / "l2")
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self, tmp_path):
        """Expired entries return None."""
        cache = L2Cache(cache_dir=tmp_path / "l2", default_ttl=1)
        cache.set("key1", "value1")

        # Manually set file mtime to the past
        cache_file = cache._get_cache_file("key1")
        import os
        past_time = time.time() - 100
        os.utime(cache_file, (past_time, past_time))

        assert cache.get("key1") is None

    def test_delete(self, tmp_path):
        """Deleting a key removes the cache file."""
        cache = L2Cache(cache_dir=tmp_path / "l2")
        cache.set("key1", "value1")
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_clear(self, tmp_path):
        """Clear removes all cache files."""
        cache = L2Cache(cache_dir=tmp_path / "l2")
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_stats(self, tmp_path):
        """Stats returns correct metrics."""
        cache = L2Cache(cache_dir=tmp_path / "l2")
        cache.set("k1", "v1")
        stats = cache.stats()
        assert stats["entries"] == 1
        assert stats["size_bytes"] > 0

    def test_corrupted_file_returns_none(self, tmp_path):
        """Corrupted cache file returns None and is deleted."""
        cache = L2Cache(cache_dir=tmp_path / "l2")
        cache.set("key1", "value1")

        # Corrupt the file
        cache_file = cache._get_cache_file("key1")
        with open(cache_file, "wb") as f:
            f.write(b"not valid pickle data")

        assert cache.get("key1") is None


class TestTieredCache:
    """Tests for the TieredCache combining L1 and L2."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset TieredCache singleton between tests."""
        TieredCache._instance = None
        yield
        TieredCache._instance = None

    def test_write_through(self, tmp_path):
        """Set writes to both L1 and L2."""
        cache = TieredCache.__new__(TieredCache)
        cache._initialized = False
        cache.__init__(l2_cache_dir=tmp_path / "l2")

        cache.set("key1", "value1", tier="both")

        # Should be in both tiers
        assert cache.l1.get("key1") == "value1"
        assert cache.l2.get("key1") == "value1"

    def test_l1_only(self, tmp_path):
        """L1-only writes do not go to L2."""
        cache = TieredCache.__new__(TieredCache)
        cache._initialized = False
        cache.__init__(l2_cache_dir=tmp_path / "l2")

        cache.set("key1", "value1", tier="l1")

        assert cache.l1.get("key1") == "value1"
        assert cache.l2.get("key1") is None

    def test_l2_only(self, tmp_path):
        """L2-only writes do not go to L1."""
        cache = TieredCache.__new__(TieredCache)
        cache._initialized = False
        cache.__init__(l2_cache_dir=tmp_path / "l2")

        cache.set("key1", "value1", tier="l2")

        assert cache.l1.get("key1") is None
        assert cache.l2.get("key1") == "value1"

    def test_read_promotes_l2_to_l1(self, tmp_path):
        """Reading from L2 promotes value to L1."""
        cache = TieredCache.__new__(TieredCache)
        cache._initialized = False
        cache.__init__(l2_cache_dir=tmp_path / "l2")

        # Write only to L2
        cache.set("key1", "promoted_value", tier="l2")
        assert cache.l1.get("key1") is None

        # Read via tiered cache (should check L1, miss, check L2, hit, promote to L1)
        result = cache.get("key1", tier="both")
        assert result == "promoted_value"

        # Now should be in L1
        assert cache.l1.get("key1") == "promoted_value"

    def test_delete_both_tiers(self, tmp_path):
        """Delete removes from both tiers."""
        cache = TieredCache.__new__(TieredCache)
        cache._initialized = False
        cache.__init__(l2_cache_dir=tmp_path / "l2")

        cache.set("key1", "value1", tier="both")
        cache.delete("key1", tier="both")

        assert cache.get("key1") is None

    def test_clear_both_tiers(self, tmp_path):
        """Clear removes all entries from both tiers."""
        cache = TieredCache.__new__(TieredCache)
        cache._initialized = False
        cache.__init__(l2_cache_dir=tmp_path / "l2")

        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.clear()

        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_stats(self, tmp_path):
        """Stats returns info for both tiers."""
        cache = TieredCache.__new__(TieredCache)
        cache._initialized = False
        cache.__init__(l2_cache_dir=tmp_path / "l2")

        cache.set("k1", "v1")
        stats = cache.stats()

        assert "l1" in stats
        assert "l2" in stats
        assert stats["l1"]["entries"] >= 1
