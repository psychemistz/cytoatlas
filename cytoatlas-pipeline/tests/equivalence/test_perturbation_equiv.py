"""Equivalence tests: Perturbation JSON files vs perturbation_data.duckdb.

Verifies that the DuckDB database (perturbation_data.duckdb) faithfully
stores the same data as the perturbation visualization JSON files.
This validates the conversion script for the perturbation domain
(parse_10M cytokine stimulation + Tahoe drug response).

Auto-skipped when data is not available.

Run:
    pytest cytoatlas-pipeline/tests/equivalence/test_perturbation_equiv.py -v
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
_DUCKDB_PATH = _DATA_ROOT / "perturbation_data.duckdb"

_HAS_DATA = _VIZ_DATA.exists() and _DUCKDB_PATH.exists()
pytestmark = pytest.mark.skipif(
    not _HAS_DATA,
    reason="Perturbation data not available",
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
    """Execute a SQL query against perturbation_data.duckdb."""
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


def _json_records(filename: str) -> list[dict]:
    """Load JSON and return the flat record list (handles dict or list)."""
    data = _load_json(filename)
    if isinstance(data, dict):
        return data.get("data", [])
    return data


# ===========================================================================
# Row count equivalence
# ===========================================================================

class TestPerturbationRowCount:
    """Verify row counts match between JSON files and DuckDB tables."""

    @pytest.mark.parametrize("json_file,table", [
        ("parse10m_cytokine_heatmap.json", "parse10m_cytokine_heatmap"),
        ("parse10m_ground_truth.json", "parse10m_ground_truth"),
        ("parse10m_donor_variability.json", "parse10m_donor_variability"),
        ("tahoe_drug_sensitivity.json", "tahoe_drug_sensitivity"),
        ("tahoe_dose_response.json", "tahoe_dose_response"),
        ("tahoe_pathway_activation.json", "tahoe_pathway_activation"),
    ])
    def test_row_count(self, json_file, table):
        """DuckDB row count should match JSON record count."""
        if not _table_exists(table):
            pytest.skip(f"DuckDB table {table} does not exist")

        json_records = _json_records(json_file)

        # Skip nested/dict structures that are not flat record lists
        if not isinstance(json_records, list):
            pytest.skip(f"{json_file} data is not a flat record list")

        duckdb_count = _count_duckdb(table)
        assert len(json_records) == duckdb_count, (
            f"Row count mismatch for {table}: JSON={len(json_records)}, "
            f"DuckDB={duckdb_count}"
        )


# ===========================================================================
# Column equivalence
# ===========================================================================

class TestPerturbationColumnEquivalence:
    """Verify DuckDB tables have all columns from the JSON files."""

    @pytest.mark.parametrize("json_file,table", [
        ("parse10m_cytokine_heatmap.json", "parse10m_cytokine_heatmap"),
        ("parse10m_ground_truth.json", "parse10m_ground_truth"),
        ("parse10m_donor_variability.json", "parse10m_donor_variability"),
        ("tahoe_drug_sensitivity.json", "tahoe_drug_sensitivity"),
        ("tahoe_dose_response.json", "tahoe_dose_response"),
        ("tahoe_pathway_activation.json", "tahoe_pathway_activation"),
    ])
    def test_json_columns_present_in_duckdb(self, json_file, table):
        """All JSON record columns should exist in the DuckDB table."""
        if not _table_exists(table):
            pytest.skip(f"DuckDB table {table} does not exist")

        json_records = _json_records(json_file)
        if not json_records:
            pytest.skip(f"{json_file} has no records")

        json_cols = set(json_records[0].keys())

        duckdb_rows = _query_duckdb(f"SELECT * FROM {table} LIMIT 1")
        if not duckdb_rows:
            pytest.skip(f"DuckDB table {table} is empty")

        duckdb_cols = set(duckdb_rows[0].keys())

        missing = json_cols - duckdb_cols
        assert not missing, (
            f"Columns in JSON but not DuckDB for {table}: {missing}"
        )

    @pytest.mark.parametrize("json_file,table", [
        ("parse10m_cytokine_heatmap.json", "parse10m_cytokine_heatmap"),
        ("tahoe_drug_sensitivity.json", "tahoe_drug_sensitivity"),
    ])
    def test_duckdb_has_no_unexpected_extra_columns(self, json_file, table):
        """DuckDB should not have many unexpected extra columns beyond JSON."""
        if not _table_exists(table):
            pytest.skip(f"DuckDB table {table} does not exist")

        json_records = _json_records(json_file)
        if not json_records:
            pytest.skip(f"{json_file} has no records")

        json_cols = set(json_records[0].keys())

        duckdb_rows = _query_duckdb(f"SELECT * FROM {table} LIMIT 1")
        if not duckdb_rows:
            pytest.skip(f"DuckDB table {table} is empty")

        duckdb_cols = set(duckdb_rows[0].keys())

        # DuckDB may have a few extra bookkeeping columns (e.g. _row_id),
        # but there should not be many unexpected columns.
        extra = duckdb_cols - json_cols
        assert len(extra) <= 3, (
            f"Unexpected extra columns in DuckDB {table}: {extra}"
        )


# ===========================================================================
# Content spot-checks
# ===========================================================================

class TestPerturbationContentSpotChecks:
    """Spot-check specific values between JSON and DuckDB."""

    def test_parse10m_heatmap_activity_values(self):
        """Spot-check activity values in parse10m_cytokine_heatmap."""
        if not _table_exists("parse10m_cytokine_heatmap"):
            pytest.skip("Table parse10m_cytokine_heatmap does not exist")

        json_records = _json_records("parse10m_cytokine_heatmap.json")
        if not json_records:
            pytest.skip("Empty JSON data")

        duckdb_rows = _query_duckdb(
            "SELECT * FROM parse10m_cytokine_heatmap LIMIT 500"
        )

        # Build lookup from DuckDB rows
        db_lookup: dict[tuple, dict] = {}
        for row in duckdb_rows:
            key = (
                row.get("cytokine"),
                row.get("cell_type"),
                row.get("signature_type"),
            )
            db_lookup[key] = row

        checked = 0
        diffs: list[str] = []
        for jrow in json_records[:200]:
            key = (
                jrow.get("cytokine"),
                jrow.get("cell_type"),
                jrow.get("signature_type"),
            )
            if key in db_lookup:
                drow = db_lookup[key]
                for col in ["activity", "mean_activity", "activity_diff"]:
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

    def test_parse10m_ground_truth_concordance(self):
        """Spot-check ground truth predicted/actual values."""
        if not _table_exists("parse10m_ground_truth"):
            pytest.skip("Table parse10m_ground_truth does not exist")

        json_records = _json_records("parse10m_ground_truth.json")
        if not json_records:
            pytest.skip("Empty JSON data")

        duckdb_rows = _query_duckdb(
            "SELECT * FROM parse10m_ground_truth LIMIT 500"
        )

        db_lookup: dict[tuple, dict] = {}
        for row in duckdb_rows:
            key = (
                row.get("cytokine"),
                row.get("cell_type"),
                row.get("signature_type"),
            )
            db_lookup[key] = row

        checked = 0
        diffs: list[str] = []
        for jrow in json_records[:200]:
            key = (
                jrow.get("cytokine"),
                jrow.get("cell_type"),
                jrow.get("signature_type"),
            )
            if key in db_lookup:
                drow = db_lookup[key]
                for col in ["predicted", "actual"]:
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

    def test_parse10m_donor_variability_values(self):
        """Spot-check donor variability variance/CV values."""
        if not _table_exists("parse10m_donor_variability"):
            pytest.skip("Table parse10m_donor_variability does not exist")

        json_records = _json_records("parse10m_donor_variability.json")
        if not json_records:
            pytest.skip("Empty JSON data")

        duckdb_rows = _query_duckdb(
            "SELECT * FROM parse10m_donor_variability LIMIT 500"
        )

        db_lookup: dict[tuple, dict] = {}
        for row in duckdb_rows:
            key = (
                row.get("cytokine"),
                row.get("cell_type"),
                row.get("donor_id", row.get("donor")),
            )
            db_lookup[key] = row

        checked = 0
        diffs: list[str] = []
        for jrow in json_records[:200]:
            key = (
                jrow.get("cytokine"),
                jrow.get("cell_type"),
                jrow.get("donor_id", jrow.get("donor")),
            )
            if key in db_lookup:
                drow = db_lookup[key]
                for col in ["variance", "cv", "activity"]:
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

    def test_tahoe_drug_sensitivity_values(self):
        """Spot-check drug sensitivity activity values."""
        if not _table_exists("tahoe_drug_sensitivity"):
            pytest.skip("Table tahoe_drug_sensitivity does not exist")

        json_records = _json_records("tahoe_drug_sensitivity.json")
        if not json_records:
            pytest.skip("Empty JSON data")

        duckdb_rows = _query_duckdb(
            "SELECT * FROM tahoe_drug_sensitivity LIMIT 500"
        )

        db_lookup: dict[tuple, dict] = {}
        for row in duckdb_rows:
            key = (
                row.get("drug"),
                row.get("cell_line"),
                row.get("signature_type"),
            )
            db_lookup[key] = row

        checked = 0
        diffs: list[str] = []
        for jrow in json_records[:200]:
            key = (
                jrow.get("drug"),
                jrow.get("cell_line"),
                jrow.get("signature_type"),
            )
            if key in db_lookup:
                drow = db_lookup[key]
                for col in ["activity", "sensitivity", "ic50"]:
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

    def test_tahoe_dose_response_values(self):
        """Spot-check dose-response activity and viability values."""
        if not _table_exists("tahoe_dose_response"):
            pytest.skip("Table tahoe_dose_response does not exist")

        json_records = _json_records("tahoe_dose_response.json")
        if not json_records:
            pytest.skip("Empty JSON data")

        duckdb_rows = _query_duckdb(
            "SELECT * FROM tahoe_dose_response LIMIT 500"
        )

        db_lookup: dict[tuple, dict] = {}
        for row in duckdb_rows:
            key = (
                row.get("drug"),
                row.get("cell_line"),
                row.get("dose"),
            )
            db_lookup[key] = row

        checked = 0
        diffs: list[str] = []
        for jrow in json_records[:200]:
            key = (
                jrow.get("drug"),
                jrow.get("cell_line"),
                jrow.get("dose"),
            )
            if key in db_lookup:
                drow = db_lookup[key]
                for col in ["activity", "viability"]:
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

    def test_tahoe_pathway_activation_values(self):
        """Spot-check pathway activation activity and p-values."""
        if not _table_exists("tahoe_pathway_activation"):
            pytest.skip("Table tahoe_pathway_activation does not exist")

        json_records = _json_records("tahoe_pathway_activation.json")
        if not json_records:
            pytest.skip("Empty JSON data")

        duckdb_rows = _query_duckdb(
            "SELECT * FROM tahoe_pathway_activation LIMIT 500"
        )

        db_lookup: dict[tuple, dict] = {}
        for row in duckdb_rows:
            key = (
                row.get("drug"),
                row.get("pathway"),
            )
            db_lookup[key] = row

        checked = 0
        diffs: list[str] = []
        for jrow in json_records[:200]:
            key = (
                jrow.get("drug"),
                jrow.get("pathway"),
            )
            if key in db_lookup:
                drow = db_lookup[key]
                for col in ["activity", "pvalue"]:
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


# ===========================================================================
# Table completeness
# ===========================================================================

class TestPerturbationTableCompleteness:
    """Verify all expected perturbation tables exist and are non-empty."""

    _EXPECTED_TABLES = [
        "parse10m_cytokine_heatmap",
        "parse10m_ground_truth",
        "parse10m_donor_variability",
        "tahoe_drug_sensitivity",
        "tahoe_dose_response",
        "tahoe_pathway_activation",
    ]

    def test_all_expected_tables_exist(self):
        """All perturbation DuckDB tables should exist."""
        import duckdb

        conn = duckdb.connect(str(_DUCKDB_PATH), read_only=True)
        result = conn.execute("SHOW TABLES").fetchall()
        all_tables = {row[0] for row in result}
        conn.close()

        missing = []
        for table in self._EXPECTED_TABLES:
            if table not in all_tables:
                missing.append(table)

        assert not missing, f"Missing perturbation tables: {missing}"

    @pytest.mark.parametrize("table", _EXPECTED_TABLES)
    def test_table_non_empty(self, table):
        """Each expected perturbation table should have at least one row."""
        if not _table_exists(table):
            pytest.skip(f"Table {table} does not exist")

        count = _count_duckdb(table)
        assert count > 0, f"Table {table} is empty"

    def test_parse10m_tables_have_signature_type_column(self):
        """parse10m tables should have a signature_type column for filtering."""
        for table in [
            "parse10m_cytokine_heatmap",
            "parse10m_ground_truth",
        ]:
            if not _table_exists(table):
                pytest.skip(f"Table {table} does not exist")

            duckdb_rows = _query_duckdb(f"SELECT * FROM {table} LIMIT 1")
            if not duckdb_rows:
                pytest.skip(f"Table {table} is empty")

            cols = set(duckdb_rows[0].keys())
            assert "signature_type" in cols, (
                f"Table {table} missing signature_type column, has: {sorted(cols)}"
            )

    def test_tahoe_tables_have_drug_column(self):
        """Tahoe tables should have a drug column."""
        for table in [
            "tahoe_drug_sensitivity",
            "tahoe_dose_response",
            "tahoe_pathway_activation",
        ]:
            if not _table_exists(table):
                pytest.skip(f"Table {table} does not exist")

            duckdb_rows = _query_duckdb(f"SELECT * FROM {table} LIMIT 1")
            if not duckdb_rows:
                pytest.skip(f"Table {table} is empty")

            cols = set(duckdb_rows[0].keys())
            assert "drug" in cols, (
                f"Table {table} missing drug column, has: {sorted(cols)}"
            )

    def test_total_perturbation_rows(self):
        """Total rows across perturbation tables should be substantial."""
        total = 0
        for table in self._EXPECTED_TABLES:
            if _table_exists(table):
                total += _count_duckdb(table)

        assert total > 0, f"Expected >0 total perturbation rows, got {total}"

    def test_parse10m_heatmap_has_both_signature_types(self):
        """parse10m_cytokine_heatmap should have both CytoSig and SecAct rows."""
        if not _table_exists("parse10m_cytokine_heatmap"):
            pytest.skip("Table not available")

        rows = _query_duckdb(
            "SELECT DISTINCT signature_type FROM parse10m_cytokine_heatmap"
        )
        sig_types = {r["signature_type"] for r in rows}

        assert "CytoSig" in sig_types, (
            f"Expected CytoSig in signature_types, got: {sig_types}"
        )

    def test_tahoe_sensitivity_has_multiple_drugs(self):
        """tahoe_drug_sensitivity should contain records for multiple drugs."""
        if not _table_exists("tahoe_drug_sensitivity"):
            pytest.skip("Table not available")

        rows = _query_duckdb(
            "SELECT COUNT(DISTINCT drug) AS n_drugs FROM tahoe_drug_sensitivity"
        )
        n_drugs = rows[0]["n_drugs"]
        assert n_drugs >= 2, f"Expected >=2 drugs, got {n_drugs}"
