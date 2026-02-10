"""JSON ↔ DuckDB equivalence tests.

Verifies that DuckDB query results match JSON file loading for key tables,
ensuring the migration preserves data integrity.

These tests require both the real atlas_data.duckdb and visualization/data/
JSON files to be present. They are skipped automatically if either is missing.

Run:
    pytest tests/integration/test_duckdb_equivalence.py -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Paths — skip if data not available
# ---------------------------------------------------------------------------

_VIZ_DATA = Path("/data/parks34/projects/2cytoatlas/visualization/data")
_DUCKDB_PATH = Path("/data/parks34/projects/2cytoatlas/atlas_data.duckdb")

_HAS_DATA = _VIZ_DATA.exists() and _DUCKDB_PATH.exists()
pytestmark = pytest.mark.skipif(
    not _HAS_DATA,
    reason="Real viz data or atlas_data.duckdb not available",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(filename: str, subdir: str | None = None) -> Any:
    """Load a JSON file from visualization/data/."""
    if subdir:
        path = _VIZ_DATA / subdir / filename
    else:
        path = _VIZ_DATA / filename
    if not path.exists():
        pytest.skip(f"JSON file not found: {path}")
    with open(path) as f:
        return json.load(f)


def _query_duckdb(table: str, limit: int | None = None) -> list[dict]:
    """Query all rows from a DuckDB table."""
    import duckdb
    conn = duckdb.connect(str(_DUCKDB_PATH), read_only=True)
    sql = f"SELECT * FROM {table}"
    if limit:
        sql += f" LIMIT {limit}"
    result = conn.execute(sql)
    columns = [desc[0] for desc in result.description]
    rows = result.fetchall()
    conn.close()
    return [dict(zip(columns, row)) for row in rows]


def _values_close(a: Any, b: Any, rtol: float = 1e-6) -> bool:
    """Compare two values with tolerance for floats."""
    if a is None and b is None:
        return True
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        if a == 0 and b == 0:
            return True
        return abs(a - b) / max(abs(a), abs(b), 1e-15) < rtol
    return a == b


def _compare_rows(
    json_rows: list[dict],
    duckdb_rows: list[dict],
    key_cols: list[str],
    value_cols: list[str],
) -> list[str]:
    """Compare rows matched by key columns. Returns list of differences."""
    diffs = []

    # Build lookup from JSON rows
    def make_key(row: dict) -> tuple:
        return tuple(str(row.get(k, "")) for k in key_cols)

    json_lookup: dict[tuple, dict] = {}
    for row in json_rows:
        key = make_key(row)
        json_lookup[key] = row

    matched = 0
    for drow in duckdb_rows:
        key = make_key(drow)
        if key not in json_lookup:
            continue
        matched += 1
        jrow = json_lookup[key]
        for col in value_cols:
            jval = jrow.get(col)
            dval = drow.get(col)
            if not _values_close(jval, dval):
                diffs.append(f"Key {key}, col '{col}': JSON={jval} vs DuckDB={dval}")
                if len(diffs) > 20:
                    diffs.append("... (truncated)")
                    return diffs

    if matched == 0 and len(json_rows) > 0 and len(duckdb_rows) > 0:
        diffs.append(f"No rows matched on keys {key_cols} (json={len(json_rows)}, duckdb={len(duckdb_rows)})")

    return diffs


# ===========================================================================
# Row count equivalence
# ===========================================================================

class TestRowCounts:
    """Verify row counts match between JSON and DuckDB for flat tables."""

    @pytest.mark.parametrize("json_file,table,is_flat", [
        ("cima_metabolites_top.json", "cima_metabolites_top", True),
        ("scatlas_organs_top.json", "scatlas_organs_top", True),
        ("cohort_validation.json", "cohort_validation", False),
        ("cima_differential.json", "cima_differential", True),
    ])
    def test_row_count(self, json_file, table, is_flat):
        json_data = _load_json(json_file)
        duckdb_rows = _query_duckdb(table)

        if is_flat:
            assert isinstance(json_data, list), f"{json_file} should be a flat list"
            assert len(json_data) == len(duckdb_rows), (
                f"Row count mismatch for {table}: "
                f"JSON={len(json_data)}, DuckDB={len(duckdb_rows)}"
            )
        else:
            # Nested JSON — DuckDB has flattened rows, just check DuckDB has data
            assert len(duckdb_rows) > 0, f"DuckDB table {table} is empty"


# ===========================================================================
# Content equivalence — flat JSON tables
# ===========================================================================

class TestFlatTableEquivalence:
    """Verify content matches for flat JSON → DuckDB tables."""

    def test_cima_metabolites_top(self):
        """cima_metabolites_top.json should exactly match DuckDB."""
        json_rows = _load_json("cima_metabolites_top.json")
        duckdb_rows = _query_duckdb("cima_metabolites_top")

        diffs = _compare_rows(
            json_rows, duckdb_rows,
            key_cols=["protein", "feature", "signature"],
            value_cols=["rho", "pvalue", "n"],
        )
        assert not diffs, f"Differences found:\n" + "\n".join(diffs)

    def test_cima_differential(self):
        """cima_differential.json should match DuckDB on key columns."""
        json_rows = _load_json("cima_differential.json")
        duckdb_rows = _query_duckdb("cima_differential")

        assert len(json_rows) == len(duckdb_rows)

        # Spot-check first 100 rows by key
        diffs = _compare_rows(
            json_rows[:100], duckdb_rows,
            key_cols=["protein", "group1", "group2", "comparison", "signature"],
            value_cols=["activity_diff", "pvalue", "qvalue"],
        )
        assert not diffs, f"Differences found:\n" + "\n".join(diffs)

    def test_inflammation_severity(self):
        """inflammation_severity.json should match DuckDB."""
        json_rows = _load_json("inflammation_severity.json")
        duckdb_rows = _query_duckdb("inflammation_severity")

        assert len(json_rows) == len(duckdb_rows), (
            f"Row count: JSON={len(json_rows)}, DuckDB={len(duckdb_rows)}"
        )


# ===========================================================================
# Content equivalence — nested JSON tables
# ===========================================================================

class TestNestedTableEquivalence:
    """Verify nested JSON flattening produces expected DuckDB content."""

    def test_disease_sankey_has_expected_columns(self):
        """disease_sankey should have category/name/type/source/target/value."""
        duckdb_rows = _query_duckdb("disease_sankey")
        assert len(duckdb_rows) > 0
        expected_cols = {"category", "name", "type", "source", "target", "value"}
        actual_cols = set(duckdb_rows[0].keys())
        assert expected_cols.issubset(actual_cols), (
            f"Missing columns: {expected_cols - actual_cols}"
        )

    def test_cohort_validation_has_expected_columns(self):
        """cohort_validation should have correlation columns."""
        duckdb_rows = _query_duckdb("cohort_validation")
        assert len(duckdb_rows) > 0
        expected_cols = {"signature", "signature_type"}
        actual_cols = set(duckdb_rows[0].keys())
        assert expected_cols.issubset(actual_cols)

    def test_exhaustion_row_count(self):
        """exhaustion table should have substantial data."""
        duckdb_rows = _query_duckdb("exhaustion", limit=1)
        assert len(duckdb_rows) > 0


# ===========================================================================
# Schema equivalence — column names match between JSON and DuckDB
# ===========================================================================

class TestSchemaEquivalence:
    """Verify DuckDB column names match JSON record keys."""

    @pytest.mark.parametrize("json_file,table", [
        ("cima_metabolites_top.json", "cima_metabolites_top"),
        ("cima_differential.json", "cima_differential"),
        ("inflammation_celltype.json", "inflammation_celltype"),
        ("scatlas_organs.json", "scatlas_organs"),
    ])
    def test_column_names_match(self, json_file, table):
        json_data = _load_json(json_file)
        if not isinstance(json_data, list) or not json_data:
            pytest.skip(f"{json_file} is not a non-empty list")

        json_cols = set(json_data[0].keys())
        duckdb_rows = _query_duckdb(table, limit=1)
        duckdb_cols = set(duckdb_rows[0].keys())

        # DuckDB may add source_file or flatten-added columns, but JSON
        # columns should all be present in DuckDB
        missing = json_cols - duckdb_cols
        assert not missing, (
            f"Columns in JSON but not DuckDB for {table}: {missing}"
        )


# ===========================================================================
# DuckDB-only sanity checks
# ===========================================================================

class TestDuckDBSanity:
    """Sanity checks on the generated DuckDB database."""

    def test_all_known_tables_exist(self):
        """Every table in _KNOWN_TABLES should exist in atlas_data.duckdb."""
        import duckdb
        from app.repositories.duckdb_repository import _KNOWN_TABLES

        conn = duckdb.connect(str(_DUCKDB_PATH), read_only=True)
        result = conn.execute("SHOW TABLES").fetchall()
        actual_tables = {row[0] for row in result}
        conn.close()

        missing = _KNOWN_TABLES - actual_tables
        assert not missing, (
            f"Tables in _KNOWN_TABLES but not in DuckDB: {missing}"
        )

    def test_no_empty_tables(self):
        """All tables in _KNOWN_TABLES should have at least 1 row."""
        import duckdb
        from app.repositories.duckdb_repository import _KNOWN_TABLES

        conn = duckdb.connect(str(_DUCKDB_PATH), read_only=True)
        result = conn.execute("SHOW TABLES").fetchall()
        actual_tables = {row[0] for row in result}

        empty = []
        for table in _KNOWN_TABLES:
            if table not in actual_tables:
                continue
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if count == 0:
                empty.append(table)

        conn.close()
        assert not empty, f"Empty tables: {empty}"

    def test_total_row_count(self):
        """Total rows across all tables should be substantial."""
        import duckdb

        conn = duckdb.connect(str(_DUCKDB_PATH), read_only=True)
        result = conn.execute("SHOW TABLES").fetchall()
        total = 0
        for (table,) in result:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            total += count
        conn.close()

        # We know the DB has ~9.6M rows
        assert total > 5_000_000, f"Expected >5M rows, got {total}"
