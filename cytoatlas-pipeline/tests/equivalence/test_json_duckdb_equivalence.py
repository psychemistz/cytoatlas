"""Equivalence tests: JSON files vs DuckDB tables.

Verifies that the DuckDB database (atlas_data.duckdb) faithfully stores
the same data as the visualization JSON files. This validates the
convert_data_to_duckdb.py conversion script.

Auto-skipped when data is not available.

Run:
    pytest cytoatlas-pipeline/tests/equivalence/ -v
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

_DATA_ROOT = Path("/data/parks34/projects/2secactpy")
_VIZ_DATA = _DATA_ROOT / "visualization" / "data"
_DUCKDB_PATH = _DATA_ROOT / "atlas_data.duckdb"

_HAS_DATA = _VIZ_DATA.exists() and _DUCKDB_PATH.exists()
pytestmark = pytest.mark.skipif(
    not _HAS_DATA,
    reason="Visualization data or atlas_data.duckdb not available",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(filename: str) -> Any:
    path = _VIZ_DATA / filename
    if not path.exists():
        pytest.skip(f"JSON file not found: {path}")
    with open(path) as f:
        return json.load(f)


def _query_duckdb(sql: str, params: list | None = None) -> list[dict]:
    import duckdb
    conn = duckdb.connect(str(_DUCKDB_PATH), read_only=True)
    if params:
        result = conn.execute(sql, params)
    else:
        result = conn.execute(sql)
    columns = [desc[0] for desc in result.description]
    rows = result.fetchall()
    conn.close()
    return [dict(zip(columns, row)) for row in rows]


def _count_duckdb(table: str) -> int:
    rows = _query_duckdb(f"SELECT COUNT(*) AS cnt FROM {table}")
    return rows[0]["cnt"]


def _values_close(a: Any, b: Any, rtol: float = 1e-6) -> bool:
    if a is None and b is None:
        return True
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        if a == 0 and b == 0:
            return True
        return abs(a - b) / max(abs(a), abs(b), 1e-15) < rtol
    return a == b


# ===========================================================================
# Row count equivalence
# ===========================================================================

class TestRowCountEquivalence:
    """Verify row counts match between flat JSON lists and DuckDB tables."""

    @pytest.mark.parametrize("json_file,table", [
        ("cima_metabolites_top.json", "cima_metabolites_top"),
        ("cima_differential.json", "cima_differential"),
        ("inflammation_severity.json", "inflammation_severity"),
        ("inflammation_celltype.json", "inflammation_celltype"),
        ("scatlas_organs.json", "scatlas_organs"),
        ("scatlas_organs_top.json", "scatlas_organs_top"),
    ])
    def test_flat_table_row_count(self, json_file, table):
        json_data = _load_json(json_file)
        if not isinstance(json_data, list):
            pytest.skip(f"{json_file} is not a flat list")

        duckdb_count = _count_duckdb(table)
        assert len(json_data) == duckdb_count, (
            f"Row count mismatch for {table}: JSON={len(json_data)}, DuckDB={duckdb_count}"
        )


# ===========================================================================
# Column equivalence
# ===========================================================================

class TestColumnEquivalence:
    """Verify DuckDB tables have all JSON columns."""

    @pytest.mark.parametrize("json_file,table", [
        ("cima_metabolites_top.json", "cima_metabolites_top"),
        ("cima_differential.json", "cima_differential"),
        ("inflammation_celltype.json", "inflammation_celltype"),
        ("scatlas_organs.json", "scatlas_organs"),
    ])
    def test_json_columns_present_in_duckdb(self, json_file, table):
        json_data = _load_json(json_file)
        if not isinstance(json_data, list) or not json_data:
            pytest.skip(f"{json_file} is not a non-empty list")

        json_cols = set(json_data[0].keys())
        duckdb_rows = _query_duckdb(f"SELECT * FROM {table} LIMIT 1")
        duckdb_cols = set(duckdb_rows[0].keys())

        missing = json_cols - duckdb_cols
        assert not missing, (
            f"Columns in JSON but not DuckDB for {table}: {missing}"
        )


# ===========================================================================
# Content spot-checks
# ===========================================================================

class TestContentSpotChecks:
    """Spot-check specific values between JSON and DuckDB."""

    def test_cima_metabolites_top_values(self):
        """Spot-check rho values in cima_metabolites_top."""
        json_data = _load_json("cima_metabolites_top.json")
        if not json_data:
            pytest.skip("Empty JSON data")

        duckdb_rows = _query_duckdb("SELECT * FROM cima_metabolites_top LIMIT 500")

        # Build DuckDB lookup
        db_lookup = {}
        for row in duckdb_rows:
            key = (row.get("protein"), row.get("feature"), row.get("signature"))
            db_lookup[key] = row

        checked = 0
        diffs = []
        for jrow in json_data[:100]:
            key = (jrow.get("protein"), jrow.get("feature"), jrow.get("signature"))
            if key in db_lookup:
                drow = db_lookup[key]
                if not _values_close(jrow.get("rho"), drow.get("rho")):
                    diffs.append(
                        f"Key {key}: JSON rho={jrow.get('rho')}, DuckDB rho={drow.get('rho')}"
                    )
                checked += 1

        assert checked > 0, "No matching rows found for spot-check"
        assert not diffs, f"Value mismatches:\n" + "\n".join(diffs[:10])

    def test_cima_differential_values(self):
        """Spot-check activity_diff values in cima_differential."""
        json_data = _load_json("cima_differential.json")
        if not json_data:
            pytest.skip("Empty JSON data")

        duckdb_rows = _query_duckdb("SELECT * FROM cima_differential LIMIT 500")

        db_lookup = {}
        for row in duckdb_rows:
            key = (
                row.get("protein"),
                row.get("group1"),
                row.get("group2"),
                row.get("comparison"),
                row.get("signature"),
            )
            db_lookup[key] = row

        checked = 0
        diffs = []
        for jrow in json_data[:100]:
            key = (
                jrow.get("protein"),
                jrow.get("group1"),
                jrow.get("group2"),
                jrow.get("comparison"),
                jrow.get("signature"),
            )
            if key in db_lookup:
                drow = db_lookup[key]
                for col in ["activity_diff", "pvalue", "qvalue"]:
                    jv = jrow.get(col)
                    dv = drow.get(col)
                    if not _values_close(jv, dv):
                        diffs.append(f"Key {key}, {col}: JSON={jv}, DuckDB={dv}")
                checked += 1

        assert checked > 0, "No matching rows found for spot-check"
        assert not diffs, f"Value mismatches:\n" + "\n".join(diffs[:10])


# ===========================================================================
# Nested JSON flattening
# ===========================================================================

class TestNestedJSONFlattening:
    """Verify nested JSON structures were correctly flattened into DuckDB."""

    def test_disease_sankey_from_nested(self):
        """disease_sankey should have expected columns from nested JSON."""
        duckdb_count = _count_duckdb("disease_sankey")
        assert duckdb_count > 0

        rows = _query_duckdb("SELECT * FROM disease_sankey LIMIT 5")
        expected_cols = {"category", "name", "type", "source", "target", "value"}
        actual_cols = set(rows[0].keys())
        assert expected_cols.issubset(actual_cols), (
            f"Missing columns in disease_sankey: {expected_cols - actual_cols}"
        )

    def test_cohort_validation_from_nested(self):
        """cohort_validation should have signature columns."""
        duckdb_count = _count_duckdb("cohort_validation")
        assert duckdb_count > 0

        rows = _query_duckdb("SELECT * FROM cohort_validation LIMIT 5")
        expected_cols = {"signature", "signature_type"}
        actual_cols = set(rows[0].keys())
        assert expected_cols.issubset(actual_cols)


# ===========================================================================
# DuckDB table completeness
# ===========================================================================

class TestDuckDBTableCompleteness:
    """Verify all expected tables exist and are non-empty."""

    def test_all_core_tables_non_empty(self):
        """Core analysis tables should all have data."""
        import duckdb
        conn = duckdb.connect(str(_DUCKDB_PATH), read_only=True)
        result = conn.execute("SHOW TABLES").fetchall()
        all_tables = {row[0] for row in result}
        conn.close()

        empty = []
        for table in sorted(all_tables):
            count = _count_duckdb(table)
            if count == 0:
                empty.append(table)

        assert not empty, f"Empty DuckDB tables: {empty}"

    def test_total_rows_substantial(self):
        """Total rows across all tables should be >5M."""
        import duckdb
        conn = duckdb.connect(str(_DUCKDB_PATH), read_only=True)
        result = conn.execute("SHOW TABLES").fetchall()
        total = 0
        for (table,) in result:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            total += count
        conn.close()

        assert total > 5_000_000, f"Expected >5M total rows, got {total:,}"
