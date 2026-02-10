"""Unit tests for DuckDB repository implementation.

Uses an in-memory DuckDB database populated with small test datasets
to verify query building, safelist enforcement, filtering, streaming,
and error handling without requiring the full atlas_data.duckdb file.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import patch

import duckdb

from app.repositories.duckdb_repository import (
    DuckDBRepository,
    _KNOWN_TABLES,
    _validate_identifier,
)

# _build_select is a static method on DuckDBRepository
_build_select = DuckDBRepository._build_select


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_db(tmp_path: Path) -> Path:
    """Create a small DuckDB file with test tables for unit testing."""
    db_path = tmp_path / "test_atlas.duckdb"
    conn = duckdb.connect(str(db_path))

    # activity table (core protocol table)
    conn.execute("""
        CREATE TABLE activity AS SELECT * FROM (VALUES
            ('cima',    'CytoSig', 'CD4_T',     'IFNG',  0.85, 120),
            ('cima',    'CytoSig', 'CD8_T',     'IFNG',  1.52, 115),
            ('cima',    'CytoSig', 'Monocytes', 'TNF',   1.95, 130),
            ('cima',    'SecAct',  'CD4_T',     'CCL2',  1.65,  95),
            ('inflam',  'CytoSig', 'CD8_T',     'IFNG',  1.30, 200)
        ) AS t(atlas, signature_type, cell_type, signature, mean_activity, n_samples)
    """)

    # gene_list table (auto-discovered, small)
    conn.execute("""
        CREATE TABLE gene_list AS SELECT * FROM (VALUES
            ('IFNG', 'CytoSig'),
            ('TNF',  'CytoSig'),
            ('CCL2', 'SecAct')
        ) AS t(gene, signature_type)
    """)

    # singlecell_activity table (for query_cells tests)
    conn.execute("""
        CREATE TABLE singlecell_activity AS SELECT * FROM (VALUES
            ('cima', 'CD4_T', 'Blood', NULL,    'CytoSig', 'IFNG',  0.5),
            ('cima', 'CD4_T', 'Blood', NULL,    'CytoSig', 'IFNG',  1.2),
            ('cima', 'CD8_T', 'Blood', NULL,    'CytoSig', 'IFNG',  2.1),
            ('cima', 'CD8_T', 'Blood', NULL,    'CytoSig', 'TNF',  -0.3),
            ('cima', 'NK',    'Blood', NULL,    'CytoSig', 'IFNG',  1.8)
        ) AS t(atlas, cell_type, organ, disease, signature_type, target, activity)
    """)

    conn.close()
    return db_path


@pytest.fixture
def test_db(tmp_path):
    """Create a test DuckDB database and return (db_path, repo)."""
    db_path = _create_test_db(tmp_path)
    with patch("app.repositories.duckdb_repository.get_settings") as mock_settings:
        mock_settings.return_value.duckdb_atlas_path = db_path
        repo = DuckDBRepository(db_path)
    yield db_path, repo
    repo.close()


@pytest.fixture
def unavailable_repo(tmp_path):
    """Create a repo pointing to a non-existent DB file."""
    missing_path = tmp_path / "missing.duckdb"
    with patch("app.repositories.duckdb_repository.get_settings") as mock_settings:
        mock_settings.return_value.duckdb_atlas_path = missing_path
        repo = DuckDBRepository(missing_path)
    return repo


# ===========================================================================
# _validate_identifier
# ===========================================================================

class TestValidateIdentifier:
    """Tests for SQL identifier validation."""

    def test_valid_identifier(self):
        assert _validate_identifier("activity") == "activity"

    def test_valid_with_underscores(self):
        assert _validate_identifier("cima_celltype") == "cima_celltype"

    def test_invalid_with_spaces(self):
        with pytest.raises(ValueError, match="Invalid identifier"):
            _validate_identifier("bad table")

    def test_invalid_with_semicolon(self):
        with pytest.raises(ValueError, match="Invalid identifier"):
            _validate_identifier("activity; DROP TABLE")

    def test_invalid_with_quotes(self):
        with pytest.raises(ValueError, match="Invalid identifier"):
            _validate_identifier("activity'")

    def test_invalid_empty(self):
        with pytest.raises(ValueError, match="Invalid identifier"):
            _validate_identifier("")


# ===========================================================================
# _build_select
# ===========================================================================

class TestBuildSelect:
    """Tests for parameterised SQL query builder."""

    def test_no_filters(self):
        sql, params = _build_select("activity", {})
        assert sql == "SELECT * FROM activity"
        assert params == []

    def test_single_filter(self):
        sql, params = _build_select("activity", {"atlas": "cima"})
        assert sql == "SELECT * FROM activity WHERE atlas = $1"
        assert params == ["cima"]

    def test_multiple_filters(self):
        sql, params = _build_select("activity", {
            "atlas": "cima",
            "signature_type": "CytoSig",
        })
        assert "atlas = $1" in sql
        assert "signature_type = $2" in sql
        assert params == ["cima", "CytoSig"]

    def test_list_filter(self):
        sql, params = _build_select("activity", {
            "cell_type": ["CD4_T", "CD8_T"],
        })
        assert "cell_type IN ($1, $2)" in sql
        assert params == ["CD4_T", "CD8_T"]

    def test_none_filter_skipped(self):
        sql, params = _build_select("activity", {"atlas": None})
        assert sql == "SELECT * FROM activity"
        assert params == []

    def test_empty_list_filter_skipped(self):
        sql, params = _build_select("activity", {"cell_type": []})
        assert sql == "SELECT * FROM activity"
        assert params == []

    def test_mixed_filters(self):
        sql, params = _build_select("activity", {
            "atlas": "cima",
            "cell_type": ["CD4_T", "CD8_T"],
            "ignored": None,
        })
        assert "atlas = $1" in sql
        assert "cell_type IN ($2, $3)" in sql
        assert params == ["cima", "CD4_T", "CD8_T"]


# ===========================================================================
# DuckDBRepository — availability & lifecycle
# ===========================================================================

class TestDuckDBAvailability:
    """Tests for DB availability checks and connection lifecycle."""

    def test_available_with_valid_db(self, test_db):
        _, repo = test_db
        assert repo.available is True

    def test_unavailable_with_missing_db(self, unavailable_repo):
        assert unavailable_repo.available is False

    def test_close_and_reopen(self, test_db):
        _, repo = test_db
        assert repo.available is True
        repo.close()
        assert repo._conn is None
        # Should reconnect on next query
        repo._available = None
        assert repo.available is True

    def test_cache_stats(self, test_db):
        _, repo = test_db
        stats = repo.get_cache_stats()
        assert stats["backend"] == "duckdb"
        assert stats["available"] is True
        assert stats["query_count"] == 0

    def test_clear_cache_noop(self, test_db):
        """clear_cache is a no-op for DuckDB (it manages its own buffer pool)."""
        _, repo = test_db
        repo.clear_cache()  # Should not raise


# ===========================================================================
# DuckDBRepository — get_data (safelist-gated)
# ===========================================================================

class TestGetData:
    """Tests for generic get_data with safelist enforcement."""

    @pytest.mark.asyncio
    async def test_get_data_known_table(self, test_db):
        _, repo = test_db
        rows = await repo.get_data("activity")
        assert len(rows) == 5
        assert all(isinstance(r, dict) for r in rows)

    @pytest.mark.asyncio
    async def test_get_data_with_filter(self, test_db):
        _, repo = test_db
        rows = await repo.get_data("activity", atlas="cima")
        assert len(rows) == 4
        assert all(r["atlas"] == "cima" for r in rows)

    @pytest.mark.asyncio
    async def test_get_data_with_multiple_filters(self, test_db):
        _, repo = test_db
        rows = await repo.get_data(
            "activity",
            atlas="cima",
            signature_type="CytoSig",
        )
        assert len(rows) == 3
        assert all(r["atlas"] == "cima" and r["signature_type"] == "CytoSig" for r in rows)

    @pytest.mark.asyncio
    async def test_get_data_unknown_table_rejected(self, test_db):
        _, repo = test_db
        with pytest.raises(ValueError, match="Unknown data type"):
            await repo.get_data("sql_injection_attempt")

    @pytest.mark.asyncio
    async def test_get_data_invalid_identifier_rejected(self, test_db):
        _, repo = test_db
        with pytest.raises(ValueError, match="Invalid identifier"):
            await repo.get_data("bad table; DROP")

    @pytest.mark.asyncio
    async def test_get_data_unavailable_db(self, unavailable_repo):
        with pytest.raises(FileNotFoundError):
            await unavailable_repo.get_data("activity")

    @pytest.mark.asyncio
    async def test_get_data_gene_list(self, test_db):
        _, repo = test_db
        rows = await repo.get_data("gene_list")
        assert len(rows) == 3

    @pytest.mark.asyncio
    async def test_get_data_empty_result(self, test_db):
        _, repo = test_db
        rows = await repo.get_data("activity", atlas="nonexistent")
        assert rows == []

    @pytest.mark.asyncio
    async def test_query_stats_increment(self, test_db):
        _, repo = test_db
        await repo.get_data("activity")
        stats = repo.get_cache_stats()
        assert stats["query_count"] == 1
        assert stats["rows_returned"] == 5

        await repo.get_data("gene_list")
        stats = repo.get_cache_stats()
        assert stats["query_count"] == 2
        assert stats["rows_returned"] == 8  # 5 + 3


# ===========================================================================
# DuckDBRepository — protocol methods
# ===========================================================================

class TestProtocolMethods:
    """Tests for AtlasRepository protocol methods."""

    @pytest.mark.asyncio
    async def test_get_activity(self, test_db):
        _, repo = test_db
        rows = await repo.get_activity("cima", "CytoSig")
        assert len(rows) == 3
        assert all(r["atlas"] == "cima" for r in rows)
        assert all(r["signature_type"] == "CytoSig" for r in rows)

    @pytest.mark.asyncio
    async def test_get_activity_with_extra_filter(self, test_db):
        _, repo = test_db
        rows = await repo.get_activity("cima", "CytoSig", cell_type="CD4_T")
        assert len(rows) == 1
        assert rows[0]["signature"] == "IFNG"

    @pytest.mark.asyncio
    async def test_get_activity_empty_result(self, test_db):
        _, repo = test_db
        rows = await repo.get_activity("nonexistent", "CytoSig")
        assert rows == []


# ===========================================================================
# DuckDBRepository — stream_results
# ===========================================================================

class TestStreamResults:
    """Tests for streaming result iteration."""

    @pytest.mark.asyncio
    async def test_stream_all_rows(self, test_db):
        _, repo = test_db
        rows = []
        async for row in repo.stream_results("activity"):
            rows.append(row)
        assert len(rows) == 5

    @pytest.mark.asyncio
    async def test_stream_with_filter(self, test_db):
        _, repo = test_db
        rows = []
        async for row in repo.stream_results("activity", atlas="cima"):
            rows.append(row)
        assert len(rows) == 4

    @pytest.mark.asyncio
    async def test_stream_unknown_table_rejected(self, test_db):
        _, repo = test_db
        with pytest.raises(ValueError, match="Unknown data type"):
            async for _ in repo.stream_results("not_a_table"):
                pass

    @pytest.mark.asyncio
    async def test_stream_unavailable_db(self, unavailable_repo):
        with pytest.raises(FileNotFoundError):
            async for _ in unavailable_repo.stream_results("activity"):
                pass


# ===========================================================================
# DuckDBRepository — query_cells
# ===========================================================================

class TestQueryCells:
    """Tests for interactive single-cell queries."""

    @pytest.mark.asyncio
    async def test_query_cells_by_atlas(self, test_db):
        _, repo = test_db
        rows = await repo.query_cells("cima")
        assert len(rows) == 5

    @pytest.mark.asyncio
    async def test_query_cells_by_cell_type(self, test_db):
        _, repo = test_db
        rows = await repo.query_cells("cima", cell_type="CD8_T")
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_query_cells_by_target(self, test_db):
        _, repo = test_db
        rows = await repo.query_cells("cima", target="IFNG")
        assert len(rows) == 4

    @pytest.mark.asyncio
    async def test_query_cells_min_activity(self, test_db):
        _, repo = test_db
        rows = await repo.query_cells("cima", min_activity=1.0)
        assert all(r["activity"] >= 1.0 for r in rows)

    @pytest.mark.asyncio
    async def test_query_cells_max_activity(self, test_db):
        _, repo = test_db
        rows = await repo.query_cells("cima", max_activity=0.0)
        assert all(r["activity"] <= 0.0 for r in rows)
        assert len(rows) == 1  # TNF = -0.3

    @pytest.mark.asyncio
    async def test_query_cells_activity_range(self, test_db):
        _, repo = test_db
        rows = await repo.query_cells("cima", min_activity=0.5, max_activity=1.5)
        assert all(0.5 <= r["activity"] <= 1.5 for r in rows)

    @pytest.mark.asyncio
    async def test_query_cells_limit(self, test_db):
        _, repo = test_db
        rows = await repo.query_cells("cima", limit=2)
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_query_cells_limit_capped(self, test_db):
        """Limit is capped at 50,000 to protect memory."""
        _, repo = test_db
        rows = await repo.query_cells("cima", limit=100_000)
        # All 5 rows returned (below cap), verifying cap doesn't crash
        assert len(rows) == 5

    @pytest.mark.asyncio
    async def test_query_cells_unavailable_db(self, unavailable_repo):
        with pytest.raises(FileNotFoundError):
            await unavailable_repo.query_cells("cima")

    @pytest.mark.asyncio
    async def test_query_cells_combined_filters(self, test_db):
        _, repo = test_db
        rows = await repo.query_cells(
            "cima",
            cell_type="CD4_T",
            target="IFNG",
            signature_type="CytoSig",
        )
        assert len(rows) == 2
        assert all(r["cell_type"] == "CD4_T" for r in rows)


# ===========================================================================
# Safelist coverage
# ===========================================================================

class TestSafelist:
    """Verify the _KNOWN_TABLES safelist is internally consistent."""

    def test_known_tables_not_empty(self):
        assert len(_KNOWN_TABLES) > 0

    def test_known_tables_all_valid_identifiers(self):
        for table in _KNOWN_TABLES:
            # Should not raise
            _validate_identifier(table)

    def test_activity_in_safelist(self):
        assert "activity" in _KNOWN_TABLES

    def test_singlecell_in_safelist(self):
        assert "singlecell_activity" in _KNOWN_TABLES

    def test_known_tables_is_frozenset(self):
        assert isinstance(_KNOWN_TABLES, frozenset)

    def test_build_select_uses_quoted_table(self):
        """_build_select references table in FROM clause."""
        sql, _ = _build_select("activity", {})
        assert "FROM activity" in sql
