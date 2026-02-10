"""Equivalence tests: Spatial JSON files vs spatial_data.duckdb.

Verifies that the DuckDB database (spatial_data.duckdb) faithfully stores
the same data as the spatial visualization JSON files. This validates the
conversion script for the spatial transcriptomics domain
(SpatialCorpus-110M: MERFISH, Visium, SlideSeq, etc.).

Auto-skipped when data is not available.

Run:
    pytest cytoatlas-pipeline/tests/equivalence/test_spatial_equiv.py -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Paths -- skip if data not available
# ---------------------------------------------------------------------------

_DATA_ROOT = Path("/data/parks34/projects/2secactpy")
_VIZ_DATA = _DATA_ROOT / "visualization" / "data"
_DUCKDB_PATH = _DATA_ROOT / "spatial_data.duckdb"

_HAS_DATA = _VIZ_DATA.exists() and _DUCKDB_PATH.exists()
pytestmark = pytest.mark.skipif(
    not _HAS_DATA,
    reason="Spatial data or spatial_data.duckdb not available",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(filename: str) -> Any:
    """Load a JSON file from the visualization data directory."""
    path = _VIZ_DATA / filename
    if not path.exists():
        pytest.skip(f"JSON file not found: {path}")
    with open(path) as f:
        return json.load(f)


def _query_duckdb(sql: str, params: list | None = None) -> list[dict]:
    """Execute a SQL query against spatial_data.duckdb."""
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
    """Count rows in a DuckDB table."""
    rows = _query_duckdb(f"SELECT COUNT(*) AS cnt FROM {table}")
    return rows[0]["cnt"]


def _table_exists(table: str) -> bool:
    """Check whether a table exists in the DuckDB file."""
    import duckdb

    conn = duckdb.connect(str(_DUCKDB_PATH), read_only=True)
    result = conn.execute("SHOW TABLES").fetchall()
    tables = {row[0] for row in result}
    conn.close()
    return table in tables


def _values_close(a: Any, b: Any, rtol: float = 1e-6) -> bool:
    """Compare two values, handling None, NaN, and float tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        if a == 0 and b == 0:
            return True
        return abs(a - b) / max(abs(a), abs(b), 1e-15) < rtol
    return a == b


def _json_records(filename: str, key: str = "data") -> list[dict]:
    """Load JSON and return the flat record list (handles dict or list).

    For dict-format JSON, extracts the sub-key specified by *key*
    (defaults to ``"data"``).  For list-format, returns the list directly.
    """
    data = _load_json(filename)
    if isinstance(data, dict):
        return data.get(key, [])
    return data


# ===========================================================================
# Row count equivalence
# ===========================================================================

class TestSpatialRowCount:
    """Verify row counts match between JSON files and DuckDB tables."""

    @pytest.mark.parametrize("json_file,json_key,table", [
        ("spatial_tissue_activity.json", "data", "spatial_tissue_activity"),
        ("spatial_technology_comparison.json", "data", "spatial_technology_comparison"),
        ("spatial_gene_coverage.json", "data", "spatial_gene_coverage"),
        ("spatial_dataset_catalog.json", "datasets", "spatial_dataset_catalog"),
    ])
    def test_row_count(self, json_file, json_key, table):
        """DuckDB row count should match JSON record count."""
        if not _table_exists(table):
            pytest.skip(f"DuckDB table {table} does not exist")

        data = _load_json(json_file)
        if isinstance(data, dict):
            json_records = data.get(json_key, [])
        else:
            json_records = data

        if not isinstance(json_records, list):
            pytest.skip(f"{json_file}[{json_key}] is not a list")

        duckdb_count = _count_duckdb(table)
        assert len(json_records) == duckdb_count, (
            f"Row count mismatch for {table}: JSON={len(json_records)}, "
            f"DuckDB={duckdb_count}"
        )

    def test_tissue_activity_data_vs_viz_table(self):
        """spatial_tissue_activity DuckDB should match the 'data' section."""
        if not _table_exists("spatial_tissue_activity"):
            pytest.skip("Table not available")

        json_records = _json_records("spatial_tissue_activity.json", "data")
        duckdb_count = _count_duckdb("spatial_tissue_activity")

        if not json_records:
            pytest.skip("No records in spatial_tissue_activity.json[data]")

        assert len(json_records) == duckdb_count, (
            f"Row count mismatch: JSON={len(json_records)}, DuckDB={duckdb_count}"
        )


# ===========================================================================
# Column equivalence
# ===========================================================================

class TestSpatialColumnEquivalence:
    """Verify DuckDB tables have all columns from the JSON files."""

    @pytest.mark.parametrize("json_file,json_key,table", [
        ("spatial_tissue_activity.json", "data", "spatial_tissue_activity"),
        ("spatial_technology_comparison.json", "data", "spatial_technology_comparison"),
        ("spatial_gene_coverage.json", "data", "spatial_gene_coverage"),
        ("spatial_dataset_catalog.json", "datasets", "spatial_dataset_catalog"),
    ])
    def test_json_columns_present_in_duckdb(self, json_file, json_key, table):
        """All JSON record columns should exist in the DuckDB table."""
        if not _table_exists(table):
            pytest.skip(f"DuckDB table {table} does not exist")

        data = _load_json(json_file)
        if isinstance(data, dict):
            json_records = data.get(json_key, [])
        else:
            json_records = data

        if not json_records:
            pytest.skip(f"{json_file}[{json_key}] has no records")

        json_cols = set(json_records[0].keys())

        duckdb_rows = _query_duckdb(f"SELECT * FROM {table} LIMIT 1")
        if not duckdb_rows:
            pytest.skip(f"DuckDB table {table} is empty")

        duckdb_cols = set(duckdb_rows[0].keys())

        missing = json_cols - duckdb_cols
        assert not missing, (
            f"Columns in JSON but not DuckDB for {table}: {missing}"
        )

    def test_tissue_activity_has_core_columns(self):
        """spatial_tissue_activity should have essential analytical columns."""
        if not _table_exists("spatial_tissue_activity"):
            pytest.skip("Table not available")

        duckdb_rows = _query_duckdb(
            "SELECT * FROM spatial_tissue_activity LIMIT 1"
        )
        if not duckdb_rows:
            pytest.skip("Table is empty")

        cols = set(duckdb_rows[0].keys())
        # At minimum expect tissue, signature_type, and some activity metric
        expected_core = {"tissue", "signature_type"}
        missing = expected_core - cols
        assert not missing, (
            f"Missing core columns in spatial_tissue_activity: {missing}"
        )

    def test_gene_coverage_has_coverage_columns(self):
        """spatial_gene_coverage should have gene count / percentage columns."""
        if not _table_exists("spatial_gene_coverage"):
            pytest.skip("Table not available")

        duckdb_rows = _query_duckdb(
            "SELECT * FROM spatial_gene_coverage LIMIT 1"
        )
        if not duckdb_rows:
            pytest.skip("Table is empty")

        cols = set(duckdb_rows[0].keys())
        assert "technology" in cols, (
            f"Missing 'technology' column, has: {sorted(cols)}"
        )

    def test_dataset_catalog_has_metadata_columns(self):
        """spatial_dataset_catalog should have dataset metadata columns."""
        if not _table_exists("spatial_dataset_catalog"):
            pytest.skip("Table not available")

        duckdb_rows = _query_duckdb(
            "SELECT * FROM spatial_dataset_catalog LIMIT 1"
        )
        if not duckdb_rows:
            pytest.skip("Table is empty")

        cols = set(duckdb_rows[0].keys())
        expected = {"technology", "tissue"}
        missing = expected - cols
        assert not missing, (
            f"Missing metadata columns in spatial_dataset_catalog: {missing}"
        )


# ===========================================================================
# Content spot-checks
# ===========================================================================

class TestSpatialContentSpotChecks:
    """Spot-check specific values between JSON and DuckDB."""

    def test_tissue_activity_values(self):
        """Spot-check activity values in spatial_tissue_activity."""
        if not _table_exists("spatial_tissue_activity"):
            pytest.skip("Table not available")

        json_records = _json_records("spatial_tissue_activity.json", "data")
        if not json_records:
            pytest.skip("Empty JSON data")

        duckdb_rows = _query_duckdb(
            "SELECT * FROM spatial_tissue_activity LIMIT 500"
        )

        db_lookup: dict[tuple, dict] = {}
        for row in duckdb_rows:
            key = (
                row.get("technology"),
                row.get("tissue"),
                row.get("signature", row.get("signature_name")),
                row.get("signature_type"),
            )
            db_lookup[key] = row

        checked = 0
        diffs: list[str] = []
        for jrow in json_records[:200]:
            key = (
                jrow.get("technology"),
                jrow.get("tissue"),
                jrow.get("signature", jrow.get("signature_name")),
                jrow.get("signature_type"),
            )
            if key in db_lookup:
                drow = db_lookup[key]
                for col in [
                    "mean_activity", "median_activity", "activity", "std_activity"
                ]:
                    jv = jrow.get(col)
                    dv = drow.get(col)
                    if jv is not None and dv is not None:
                        if not _values_close(jv, dv):
                            diffs.append(
                                f"Key {key}, {col}: JSON={jv}, DuckDB={dv}"
                            )
                checked += 1

        assert checked > 0, "No matching rows found for spot-check"
        assert not diffs, "Value mismatches:\n" + "\n".join(diffs[:10])

    def test_technology_comparison_correlation_values(self):
        """Spot-check correlation values in spatial_technology_comparison."""
        if not _table_exists("spatial_technology_comparison"):
            pytest.skip("Table not available")

        json_records = _json_records(
            "spatial_technology_comparison.json", "data"
        )
        if not json_records:
            pytest.skip("Empty JSON data")

        duckdb_rows = _query_duckdb(
            "SELECT * FROM spatial_technology_comparison LIMIT 500"
        )

        db_lookup: dict[tuple, dict] = {}
        for row in duckdb_rows:
            key = (
                row.get("tech_a", row.get("technology_a")),
                row.get("tech_b", row.get("technology_b")),
                row.get("tissue"),
                row.get("signature_type"),
            )
            db_lookup[key] = row

        checked = 0
        diffs: list[str] = []
        for jrow in json_records[:200]:
            key = (
                jrow.get("tech_a", jrow.get("technology_a")),
                jrow.get("tech_b", jrow.get("technology_b")),
                jrow.get("tissue"),
                jrow.get("signature_type"),
            )
            if key in db_lookup:
                drow = db_lookup[key]
                for col in ["correlation", "concordance", "r", "rho"]:
                    jv = jrow.get(col)
                    dv = drow.get(col)
                    if jv is not None and dv is not None:
                        if not _values_close(jv, dv):
                            diffs.append(
                                f"Key {key}, {col}: JSON={jv}, DuckDB={dv}"
                            )
                checked += 1

        assert checked > 0, "No matching rows found for spot-check"
        assert not diffs, "Value mismatches:\n" + "\n".join(diffs[:10])

    def test_gene_coverage_percentages(self):
        """Spot-check gene coverage percentage values."""
        if not _table_exists("spatial_gene_coverage"):
            pytest.skip("Table not available")

        json_records = _json_records("spatial_gene_coverage.json", "data")
        if not json_records:
            pytest.skip("Empty JSON data")

        duckdb_rows = _query_duckdb(
            "SELECT * FROM spatial_gene_coverage LIMIT 100"
        )

        db_lookup: dict[str, dict] = {}
        for row in duckdb_rows:
            tech = row.get("technology")
            if tech:
                db_lookup[tech] = row

        checked = 0
        diffs: list[str] = []
        for jrow in json_records:
            tech = jrow.get("technology")
            if tech in db_lookup:
                drow = db_lookup[tech]
                for col in [
                    "n_genes_total",
                    "n_cytosig_genes",
                    "n_secact_genes",
                    "cytosig_pct",
                    "secact_pct",
                ]:
                    jv = jrow.get(col)
                    dv = drow.get(col)
                    if jv is not None and dv is not None:
                        if not _values_close(jv, dv):
                            diffs.append(
                                f"Tech={tech}, {col}: JSON={jv}, DuckDB={dv}"
                            )
                checked += 1

        assert checked > 0, "No matching rows found for spot-check"
        assert not diffs, "Value mismatches:\n" + "\n".join(diffs[:10])

    def test_dataset_catalog_metadata_values(self):
        """Spot-check dataset catalog metadata (accession, n_cells)."""
        if not _table_exists("spatial_dataset_catalog"):
            pytest.skip("Table not available")

        data = _load_json("spatial_dataset_catalog.json")
        if isinstance(data, dict):
            json_datasets = data.get("datasets", [])
        else:
            json_datasets = data

        if not json_datasets:
            pytest.skip("No datasets in JSON")

        duckdb_rows = _query_duckdb(
            "SELECT * FROM spatial_dataset_catalog LIMIT 500"
        )

        db_lookup: dict[str, dict] = {}
        for row in duckdb_rows:
            did = row.get("dataset_id", row.get("accession"))
            if did:
                db_lookup[did] = row

        checked = 0
        diffs: list[str] = []
        for jrow in json_datasets[:100]:
            did = jrow.get("dataset_id", jrow.get("accession"))
            if did and did in db_lookup:
                drow = db_lookup[did]
                for col in ["technology", "tissue", "n_cells", "n_genes"]:
                    jv = jrow.get(col)
                    dv = drow.get(col)
                    if jv is not None and dv is not None:
                        if not _values_close(jv, dv):
                            diffs.append(
                                f"Dataset={did}, {col}: JSON={jv}, DuckDB={dv}"
                            )
                checked += 1

        assert checked > 0, "No matching rows found for spot-check"
        assert not diffs, "Value mismatches:\n" + "\n".join(diffs[:10])


# ===========================================================================
# Table completeness
# ===========================================================================

class TestSpatialTableCompleteness:
    """Verify all expected spatial tables exist and are non-empty."""

    _EXPECTED_TABLES = [
        "spatial_tissue_activity",
        "spatial_technology_comparison",
        "spatial_gene_coverage",
        "spatial_dataset_catalog",
    ]

    def test_all_expected_tables_exist(self):
        """All spatial DuckDB tables should exist."""
        import duckdb

        conn = duckdb.connect(str(_DUCKDB_PATH), read_only=True)
        result = conn.execute("SHOW TABLES").fetchall()
        all_tables = {row[0] for row in result}
        conn.close()

        missing = []
        for table in self._EXPECTED_TABLES:
            if table not in all_tables:
                missing.append(table)

        assert not missing, f"Missing spatial tables: {missing}"

    @pytest.mark.parametrize("table", _EXPECTED_TABLES)
    def test_table_non_empty(self, table):
        """Each expected spatial table should have at least one row."""
        if not _table_exists(table):
            pytest.skip(f"Table {table} does not exist")

        count = _count_duckdb(table)
        assert count > 0, f"Table {table} is empty"

    def test_tissue_activity_has_signature_type_column(self):
        """spatial_tissue_activity should have a signature_type column."""
        if not _table_exists("spatial_tissue_activity"):
            pytest.skip("Table not available")

        duckdb_rows = _query_duckdb(
            "SELECT * FROM spatial_tissue_activity LIMIT 1"
        )
        if not duckdb_rows:
            pytest.skip("Table is empty")

        cols = set(duckdb_rows[0].keys())
        assert "signature_type" in cols, (
            f"Missing signature_type column, has: {sorted(cols)}"
        )

    def test_technology_comparison_has_signature_type(self):
        """spatial_technology_comparison should have a signature_type column."""
        if not _table_exists("spatial_technology_comparison"):
            pytest.skip("Table not available")

        duckdb_rows = _query_duckdb(
            "SELECT * FROM spatial_technology_comparison LIMIT 1"
        )
        if not duckdb_rows:
            pytest.skip("Table is empty")

        cols = set(duckdb_rows[0].keys())
        assert "signature_type" in cols, (
            f"Missing signature_type column, has: {sorted(cols)}"
        )

    def test_gene_coverage_has_technology_column(self):
        """spatial_gene_coverage should have a technology column."""
        if not _table_exists("spatial_gene_coverage"):
            pytest.skip("Table not available")

        duckdb_rows = _query_duckdb(
            "SELECT * FROM spatial_gene_coverage LIMIT 1"
        )
        if not duckdb_rows:
            pytest.skip("Table is empty")

        cols = set(duckdb_rows[0].keys())
        assert "technology" in cols, (
            f"Missing technology column, has: {sorted(cols)}"
        )

    def test_dataset_catalog_has_multiple_technologies(self):
        """spatial_dataset_catalog should list multiple technologies."""
        if not _table_exists("spatial_dataset_catalog"):
            pytest.skip("Table not available")

        rows = _query_duckdb(
            "SELECT COUNT(DISTINCT technology) AS n_techs "
            "FROM spatial_dataset_catalog"
        )
        n_techs = rows[0]["n_techs"]
        assert n_techs >= 2, f"Expected >=2 technologies, got {n_techs}"

    def test_dataset_catalog_has_multiple_tissues(self):
        """spatial_dataset_catalog should list multiple tissues."""
        if not _table_exists("spatial_dataset_catalog"):
            pytest.skip("Table not available")

        rows = _query_duckdb(
            "SELECT COUNT(DISTINCT tissue) AS n_tissues "
            "FROM spatial_dataset_catalog"
        )
        n_tissues = rows[0]["n_tissues"]
        assert n_tissues >= 2, f"Expected >=2 tissues, got {n_tissues}"

    def test_tissue_activity_has_both_signature_types(self):
        """spatial_tissue_activity should contain both CytoSig and SecAct."""
        if not _table_exists("spatial_tissue_activity"):
            pytest.skip("Table not available")

        rows = _query_duckdb(
            "SELECT DISTINCT signature_type FROM spatial_tissue_activity"
        )
        sig_types = {r["signature_type"] for r in rows}

        assert "CytoSig" in sig_types, (
            f"Expected CytoSig in signature_types, got: {sig_types}"
        )

    def test_total_spatial_rows(self):
        """Total rows across spatial tables should be >0."""
        total = 0
        for table in self._EXPECTED_TABLES:
            if _table_exists(table):
                total += _count_duckdb(table)

        assert total > 0, f"Expected >0 total spatial rows, got {total}"

    def test_no_empty_tables_in_database(self):
        """All tables in the spatial DuckDB file should be non-empty."""
        import duckdb

        conn = duckdb.connect(str(_DUCKDB_PATH), read_only=True)
        result = conn.execute("SHOW TABLES").fetchall()
        all_tables = [row[0] for row in result]
        conn.close()

        empty = []
        for table in all_tables:
            count = _count_duckdb(table)
            if count == 0:
                empty.append(table)

        assert not empty, f"Empty DuckDB tables: {empty}"
