#!/usr/bin/env python3
"""Convert visualization JSON and CSV data to DuckDB for high-performance analytical queries.

Creates a single DuckDB database from:
  - Visualization JSON files (flat arrays, dict-wrapped, nested)
  - Cross-sample correlation CSVs
  - SQLite validation_scatter.db (if present)
  - Auto-discovered remaining JSON files

DuckDB advantages over JSON/SQLite:
  - Columnar storage with compression (2-10x smaller than JSON)
  - Vectorized query execution (10-100x faster analytical queries)
  - Predicate pushdown and parallel scanning
  - SQL interface for ad-hoc exploration

Usage:
  python scripts/convert_data_to_duckdb.py --all
  python scripts/convert_data_to_duckdb.py --table activity
  python scripts/convert_data_to_duckdb.py --table cross_sample_correlations
  python scripts/convert_data_to_duckdb.py --all --output /data/parks34/projects/2cytoatlas/atlas_data.duckdb
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
import zlib
from pathlib import Path

import duckdb
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIZ_DATA = PROJECT_ROOT / "visualization" / "data"
RESULTS = PROJECT_ROOT / "results"
CORR_DIR = RESULTS / "cross_sample_validation" / "correlations"
SQLITE_DB = VIZ_DATA / "validation_scatter.db"

# Maximum rows to accumulate per chunk when building DataFrames from nested JSON
CHUNK_SIZE = 500_000


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------
def load_json(path: Path):
    """Load JSON file using orjson if available, else stdlib json.

    Falls back to stdlib json when orjson fails (e.g. NaN/Infinity values
    which are valid in JavaScript but rejected by strict JSON parsers).
    """
    try:
        import orjson
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    except ImportError:
        with open(path) as f:
            return json.load(f)
    except Exception:
        # orjson rejects NaN/Infinity — fall back to stdlib json which accepts them
        logger.info("  orjson failed, falling back to stdlib json for %s", path.name)
        with open(path) as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Flatten helpers (matching convert_json_to_parquet.py patterns)
# ---------------------------------------------------------------------------
def flatten_age_bmi(data: dict) -> list[dict]:
    """Flatten nested age_bmi_boxplots structure.

    Input:  {atlas: {section: [records], ...}, ...}
    Output: flat list of records with 'atlas' and 'section' columns added.
    """
    rows = []
    for atlas, sections in data.items():
        if not isinstance(sections, dict):
            continue
        for section, records in sections.items():
            if not isinstance(records, list) or len(records) == 0:
                continue
            if not isinstance(records[0], dict):
                continue
            for rec in records:
                row = {"atlas": atlas, "section": section}
                row.update(rec)
                rows.append(row)
    return rows


def flatten_dict_of_lists(data: dict) -> list[dict]:
    """Flatten {key: [records]} -> flat list with 'category' column.

    Handles structures like cima_correlations.json ({age: [...], bmi: [...], ...})
    and inflammation_correlations.json.
    """
    rows = []
    for category, records in data.items():
        if not isinstance(records, list):
            continue
        if len(records) == 0:
            continue
        if not isinstance(records[0], dict):
            continue
        for rec in records:
            row = {"category": category}
            row.update(rec)
            rows.append(row)
    return rows


def flatten_bulk_rnaseq(data: dict) -> list[dict]:
    """Flatten bulk_rnaseq_validation.json.

    Input: {dataset: {n_samples, tissue_types/cancer_types, summary, donor_level, donor_scatter}}
    Where summary and donor_level are dicts keyed by sigtype (cytosig, secact, lincytosig),
    each containing a list of records or a summary dict.
    Output: flat records from donor_level arrays with dataset and sig_type columns.
    """
    rows = []
    for dataset, info in data.items():
        if not isinstance(info, dict):
            continue
        # Extract donor_level records: {sigtype: [records]}
        donor_level = info.get("donor_level")
        if isinstance(donor_level, dict):
            for sig_type, records in donor_level.items():
                if isinstance(records, list):
                    for rec in records:
                        if isinstance(rec, dict):
                            row = {"dataset": dataset, "data_section": "donor_level", "sig_type": sig_type}
                            row.update(rec)
                            rows.append(row)
        elif isinstance(donor_level, list):
            for rec in donor_level:
                if isinstance(rec, dict):
                    row = {"dataset": dataset, "data_section": "donor_level"}
                    row.update(rec)
                    rows.append(row)
    return rows


def flatten_cross_atlas(data: dict) -> list[dict]:
    """Flatten cross_atlas.json nested structure.

    Input: {section: {sub_section: data}} (deeply nested, heterogeneous)
    Output: flat records extracted from list-of-dict sub-sections.
    """
    rows = []
    _flatten_recursive(data, rows, depth=0, context={})
    return rows


def _flatten_recursive(obj, rows, depth, context, max_depth=5):
    """Recursively flatten nested dicts/lists into rows."""
    if depth > max_depth:
        return
    if isinstance(obj, list):
        if len(obj) > 0 and isinstance(obj[0], dict):
            for rec in obj:
                row = dict(context)
                row.update(rec)
                rows.append(row)
    elif isinstance(obj, dict):
        # Check if this level contains list-of-dicts values
        has_record_lists = any(
            isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict)
            for v in obj.values()
        )
        if has_record_lists:
            for k, v in obj.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    ctx = dict(context)
                    ctx[f"section_L{depth}"] = k
                    for rec in v:
                        row = dict(ctx)
                        row.update(rec)
                        rows.append(row)
        else:
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    ctx = dict(context)
                    ctx[f"section_L{depth}"] = k
                    _flatten_recursive(v, rows, depth + 1, ctx, max_depth)


FLATTEN_FUNCS = {
    "flatten_age_bmi": flatten_age_bmi,
    "flatten_dict_of_lists": flatten_dict_of_lists,
    "flatten_bulk_rnaseq": flatten_bulk_rnaseq,
    "flatten_cross_atlas": flatten_cross_atlas,
}


def _stabilize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent column types to prevent DuckDB inference mismatches.

    Converts object columns to explicit string type and replaces NaN with None
    so that DuckDB does not infer DOUBLE from a column that is actually VARCHAR.
    """
    import numpy as np

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).replace("nan", None).replace("None", None)
        elif df[col].dtype in (np.float64, np.float32):
            # Replace NaN with None so DuckDB uses NULL
            df[col] = df[col].where(df[col].notna(), None)
    return df


# ---------------------------------------------------------------------------
# Core conversion functions
# ---------------------------------------------------------------------------
def _size_str(path: Path) -> str:
    """Human-readable file size."""
    if not path.exists():
        return "N/A"
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    elif size < 1024 ** 2:
        return f"{size / 1024:.1f} KB"
    elif size < 1024 ** 3:
        return f"{size / (1024 ** 2):.1f} MB"
    else:
        return f"{size / (1024 ** 3):.2f} GB"


def _create_indexes(db: duckdb.DuckDBPyConnection, table_name: str, columns: list[str]):
    """Create indexes on specified columns for a DuckDB table."""
    for col in columns:
        idx_name = f"idx_{table_name}_{col}"
        try:
            db.execute(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table_name}" ("{col}")')
        except (duckdb.CatalogException, duckdb.BinderException):
            # Index may already exist or column may not exist in data
            logger.warning("Could not create index %s on %s.%s (column may not exist)", idx_name, table_name, col)


def _drop_table(db: duckdb.DuckDBPyConnection, table_name: str):
    """Drop a table if it exists."""
    db.execute(f'DROP TABLE IF EXISTS "{table_name}"')


def convert_flat_json(
    db: duckdb.DuckDBPyConnection,
    table_name: str,
    json_path: Path,
    indexes: list[str],
) -> int:
    """Load a flat JSON array into a DuckDB table with indexes.

    Args:
        db: DuckDB connection.
        table_name: Target table name.
        json_path: Path to JSON file containing a flat array of records.
        indexes: Column names to index.

    Returns:
        Number of rows inserted.
    """
    logger.info("Loading %s (%s) ...", json_path.name, _size_str(json_path))
    t0 = time.time()

    data = load_json(json_path)
    if not isinstance(data, list):
        raise ValueError(
            f"Expected flat JSON array in {json_path.name}, got {type(data).__name__}. "
            f"Top keys: {list(data.keys())[:5] if isinstance(data, dict) else 'N/A'}"
        )
    if len(data) == 0:
        logger.warning("Empty array in %s, skipping.", json_path.name)
        return 0

    load_time = time.time() - t0
    logger.info("  JSON loaded: %d records in %.1fs", len(data), load_time)

    # Convert in chunks to limit peak memory
    _drop_table(db, table_name)
    total_rows = 0
    for i in range(0, len(data), CHUNK_SIZE):
        chunk = data[i : i + CHUNK_SIZE]
        df = _stabilize_dtypes(pd.DataFrame(chunk))
        if i == 0:
            db.execute(
                f'CREATE TABLE "{table_name}" AS SELECT * FROM df'
            )
        else:
            db.execute(
                f'INSERT INTO "{table_name}" SELECT * FROM df'
            )
        total_rows += len(df)
        if len(data) > CHUNK_SIZE:
            logger.info("  Inserted chunk %d-%d (%d rows)", i, i + len(chunk), total_rows)

    del data
    _create_indexes(db, table_name, indexes)

    elapsed = time.time() - t0
    logger.info(
        "  Table '%s': %d rows, %d columns in %.1fs",
        table_name, total_rows, len(df.columns), elapsed,
    )
    return total_rows


def convert_nested_json(
    db: duckdb.DuckDBPyConnection,
    table_name: str,
    json_path: Path,
    extract_key: str | None,
    flatten_func: str | None,
    indexes: list[str],
) -> int:
    """Load nested JSON, flatten, and insert into DuckDB.

    Handles three patterns:
      1. extract_key: JSON is a dict, pull out a specific key that holds a list.
      2. flatten_func: JSON is nested, apply a named flatten function.
      3. Both None: treated as flat (will raise if not a list).

    Args:
        db: DuckDB connection.
        table_name: Target table name.
        json_path: Path to JSON file.
        extract_key: Dict key containing the record array (e.g., "data").
        flatten_func: Name of flatten function in FLATTEN_FUNCS.
        indexes: Column names to index.

    Returns:
        Number of rows inserted.
    """
    logger.info("Loading %s (%s) ...", json_path.name, _size_str(json_path))
    t0 = time.time()

    data = load_json(json_path)
    load_time = time.time() - t0
    logger.info("  JSON loaded in %.1fs", load_time)

    # Extract or flatten
    if extract_key:
        logger.info("  Extracting key '%s' ...", extract_key)
        if not isinstance(data, dict) or extract_key not in data:
            raise ValueError(
                f"Expected dict with key '{extract_key}' in {json_path.name}, "
                f"got {type(data).__name__}"
            )
        records = data[extract_key]
    elif flatten_func:
        logger.info("  Flattening with '%s' ...", flatten_func)
        func = FLATTEN_FUNCS[flatten_func]
        records = func(data)
    else:
        # Treat as flat
        records = data

    if not isinstance(records, list):
        raise ValueError(
            f"After extraction/flatten, expected list, got {type(records).__name__}"
        )

    if len(records) == 0:
        logger.warning("No records after extraction from %s, skipping.", json_path.name)
        return 0

    logger.info("  %d records extracted", len(records))
    del data

    # Build full DataFrame at once to avoid cross-chunk type inference mismatches.
    # Records are already in memory from extract/flatten, so no extra memory cost.
    df = _stabilize_dtypes(pd.DataFrame(records))
    del records
    total_rows = len(df)

    _drop_table(db, table_name)
    db.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM df')
    _create_indexes(db, table_name, indexes)

    elapsed = time.time() - t0
    logger.info(
        "  Table '%s': %d rows, %d columns in %.1fs",
        table_name, total_rows, len(df.columns), elapsed,
    )
    return total_rows


def convert_csv_files(
    db: duckdb.DuckDBPyConnection,
    table_name: str,
    csv_pattern: str,
    indexes: list[str],
) -> int:
    """Load multiple CSV files matching a glob pattern into a single DuckDB table.

    Adds a 'source_file' column indicating which CSV each row came from.

    Args:
        db: DuckDB connection.
        table_name: Target table name.
        csv_pattern: Glob pattern for CSV files (relative to project root).
        indexes: Column names to index.

    Returns:
        Total number of rows inserted.
    """
    csv_dir = Path(csv_pattern).parent
    csv_glob = Path(csv_pattern).name
    matching = sorted(csv_dir.glob(csv_glob))

    if not matching:
        logger.warning("No CSV files matched pattern '%s'", csv_pattern)
        return 0

    logger.info("Found %d CSV files matching '%s'", len(matching), csv_pattern)

    # Read all CSVs into one DataFrame to avoid cross-file type inference mismatches
    all_dfs = []
    for csv_path in matching:
        t0 = time.time()
        logger.info("  Reading %s (%s) ...", csv_path.name, _size_str(csv_path))
        df = pd.read_csv(csv_path, low_memory=False)
        df["source_file"] = csv_path.stem
        all_dfs.append(df)
        logger.info("    %d rows in %.1fs", len(df), time.time() - t0)

    if not all_dfs:
        logger.warning("No CSV data loaded")
        return 0

    combined = _stabilize_dtypes(pd.concat(all_dfs, ignore_index=True))
    del all_dfs
    total_rows = len(combined)

    _drop_table(db, table_name)
    db.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM combined')
    _create_indexes(db, table_name, indexes)
    logger.info("  Table '%s': %d total rows from %d files", table_name, total_rows, len(matching))
    return total_rows


def migrate_sqlite(
    db: duckdb.DuckDBPyConnection,
    sqlite_path: Path,
) -> int:
    """Migrate validation_scatter.db tables (scatter_targets, scatter_points) into DuckDB.

    scatter_targets is migrated as-is (metadata rows).
    scatter_points BLOBs (zlib-compressed JSON) are stored as BLOB in DuckDB.

    Args:
        db: DuckDB connection.
        sqlite_path: Path to the SQLite database.

    Returns:
        Total number of rows migrated.
    """
    if not sqlite_path.exists():
        logger.warning("SQLite DB not found at %s, skipping migration.", sqlite_path)
        return 0

    logger.info("Migrating SQLite DB: %s (%s)", sqlite_path.name, _size_str(sqlite_path))
    t0 = time.time()

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row

    total_rows = 0

    # --- scatter_targets ---
    logger.info("  Reading scatter_targets ...")
    rows = conn.execute("SELECT * FROM scatter_targets").fetchall()
    if rows:
        columns = rows[0].keys()
        data = [dict(r) for r in rows]
        df_targets = pd.DataFrame(data, columns=columns)

        _drop_table(db, "scatter_targets")
        db.execute('CREATE TABLE "scatter_targets" AS SELECT * FROM df_targets')
        _create_indexes(db, "scatter_targets", [
            "source", "atlas", "level", "sigtype", "target",
        ])

        n_targets = len(df_targets)
        total_rows += n_targets
        logger.info("    scatter_targets: %d rows", n_targets)
    else:
        logger.warning("    scatter_targets: empty")

    # --- scatter_points ---
    # Points contain zlib-compressed JSON BLOBs. We store them as-is in DuckDB
    # so the API layer can decompress on demand (same as SQLite path).
    logger.info("  Reading scatter_points ...")

    _drop_table(db, "scatter_points")
    db.execute("""
        CREATE TABLE "scatter_points" (
            target_id INTEGER PRIMARY KEY,
            points_blob BLOB NOT NULL
        )
    """)

    # Read and insert in batches to limit memory
    cursor = conn.execute("SELECT target_id, points_blob FROM scatter_points")
    batch = []
    batch_size = 1000
    n_points = 0

    while True:
        row = cursor.fetchone()
        if row is None:
            break
        batch.append((row["target_id"], row["points_blob"]))
        if len(batch) >= batch_size:
            df_batch = pd.DataFrame(batch, columns=["target_id", "points_blob"])
            db.execute('INSERT INTO "scatter_points" SELECT * FROM df_batch')
            n_points += len(batch)
            batch = []

    if batch:
        df_batch = pd.DataFrame(batch, columns=["target_id", "points_blob"])
        db.execute('INSERT INTO "scatter_points" SELECT * FROM df_batch')
        n_points += len(batch)

    total_rows += n_points
    logger.info("    scatter_points: %d rows", n_points)

    conn.close()

    elapsed = time.time() - t0
    logger.info("  SQLite migration complete: %d total rows in %.1fs", total_rows, elapsed)
    return total_rows


# ---------------------------------------------------------------------------
# Table configuration registry
# ---------------------------------------------------------------------------
# type: "flat_json"    -> convert_flat_json (JSON is a flat array of records)
# type: "nested_json"  -> convert_nested_json (uses extract_key or flatten_func)
# type: "csv"          -> convert_csv_files (glob pattern for CSVs)
TABLE_CONFIGS = {
    # ------- Large visualization JSON (explicit configs with indexes) -------
    "activity": {
        "source": VIZ_DATA / "activity_boxplot.json",
        "type": "flat_json",
        "indexes": ["atlas", "cell_type", "signature_type"],
    },
    "inflammation_disease": {
        "source": VIZ_DATA / "inflammation_disease.json",
        "type": "flat_json",
        "indexes": ["disease", "cell_type", "signature_type"],
    },
    "inflammation_disease_filtered": {
        "source": VIZ_DATA / "inflammation_disease_filtered.json",
        "type": "flat_json",
        "indexes": ["disease", "cell_type"],
    },
    "singlecell_activity": {
        "source": VIZ_DATA / "singlecell_activity.json",
        "type": "flat_json",
        "indexes": ["atlas", "cell_type", "signature_type"],
    },
    "scatlas_celltypes": {
        "source": VIZ_DATA / "scatlas_celltypes.json",
        "type": "nested_json",
        "extract_key": "data",
        "indexes": ["organ", "cell_type"],
    },
    "age_bmi_boxplots": {
        "source": VIZ_DATA / "age_bmi_boxplots.json",
        "type": "nested_json",
        "flatten_func": "flatten_age_bmi",
        "indexes": ["atlas", "section", "cell_type"],
    },
    "age_bmi_boxplots_filtered": {
        "source": VIZ_DATA / "age_bmi_boxplots_filtered.json",
        "type": "nested_json",
        "flatten_func": "flatten_age_bmi",
        "indexes": ["atlas", "section"],
    },
    "bulk_validation": {
        "source": VIZ_DATA / "bulk_rnaseq_validation.json",
        "type": "nested_json",
        "flatten_func": "flatten_bulk_rnaseq",
        "indexes": ["dataset", "data_section"],
    },
    # ------- Cross-sample correlation CSVs -------
    "cross_sample_correlations": {
        "source": str(CORR_DIR / "*.csv"),
        "type": "csv",
        "indexes": ["atlas", "signature", "source_file"],
    },
    # ------- Medium-size flat JSON files -------
    "inflammation_severity": {
        "source": VIZ_DATA / "inflammation_severity.json",
        "type": "flat_json",
        "indexes": ["disease", "severity", "signature_type"],
    },
    "inflammation_severity_filtered": {
        "source": VIZ_DATA / "inflammation_severity_filtered.json",
        "type": "flat_json",
        "indexes": ["disease"],
    },
    "scatlas_organs": {
        "source": VIZ_DATA / "scatlas_organs.json",
        "type": "flat_json",
        "indexes": ["organ", "signature_type"],
    },
    "gene_expression": {
        "source": VIZ_DATA / "gene_expression.json",
        "type": "flat_json",
        "indexes": ["gene", "cell_type", "atlas"],
    },
    "expression_boxplot": {
        "source": VIZ_DATA / "expression_boxplot.json",
        "type": "flat_json",
        "indexes": ["gene", "cell_type", "atlas"],
    },
    "cima_celltype": {
        "source": VIZ_DATA / "cima_celltype.json",
        "type": "flat_json",
        "indexes": ["cell_type", "signature_type"],
    },
    "inflammation_celltype": {
        "source": VIZ_DATA / "inflammation_celltype.json",
        "type": "flat_json",
        "indexes": ["cell_type", "signature_type"],
    },
    "cima_differential": {
        "source": VIZ_DATA / "cima_differential.json",
        "type": "flat_json",
        "indexes": ["protein", "group1"],
    },
    "inflammation_differential": {
        "source": VIZ_DATA / "inflammation_differential.json",
        "type": "flat_json",
        "indexes": ["protein", "disease"],
    },
    "cima_metabolites_top": {
        "source": VIZ_DATA / "cima_metabolites_top.json",
        "type": "flat_json",
        "indexes": ["protein"],
    },
    "scatlas_organs_top": {
        "source": VIZ_DATA / "scatlas_organs_top.json",
        "type": "flat_json",
        "indexes": ["organ"],
    },
    "cima_singlecell_cytosig": {
        "source": VIZ_DATA / "cima_singlecell_cytosig.json",
        "type": "flat_json",
        "indexes": ["cell_type", "signature"],
    },
    "scatlas_normal_singlecell_cytosig": {
        "source": VIZ_DATA / "scatlas_normal_singlecell_cytosig.json",
        "type": "flat_json",
        "indexes": ["cell_type", "signature"],
    },
    "scatlas_cancer_singlecell_cytosig": {
        "source": VIZ_DATA / "scatlas_cancer_singlecell_cytosig.json",
        "type": "flat_json",
        "indexes": ["cell_type", "signature"],
    },
    "scatlas_normal_singlecell_secact": {
        "source": VIZ_DATA / "scatlas_normal_singlecell_secact.json",
        "type": "flat_json",
        "indexes": ["cell_type", "signature"],
    },
    "scatlas_cancer_singlecell_secact": {
        "source": VIZ_DATA / "scatlas_cancer_singlecell_secact.json",
        "type": "flat_json",
        "indexes": ["cell_type", "signature"],
    },
    # ------- Nested JSON files -------
    "cima_correlations": {
        "source": VIZ_DATA / "cima_correlations.json",
        "type": "nested_json",
        "flatten_func": "flatten_dict_of_lists",
        "indexes": ["category", "protein"],
    },
    "inflammation_correlations": {
        "source": VIZ_DATA / "inflammation_correlations.json",
        "type": "nested_json",
        "flatten_func": "flatten_dict_of_lists",
        "indexes": ["category", "protein"],
    },
    "inflammation_celltype_correlations": {
        "source": VIZ_DATA / "inflammation_celltype_correlations.json",
        "type": "nested_json",
        "flatten_func": "flatten_dict_of_lists",
        "indexes": ["category", "cell_type"],
    },
    "cima_celltype_correlations": {
        "source": VIZ_DATA / "cima_celltype_correlations.json",
        "type": "nested_json",
        "flatten_func": "flatten_dict_of_lists",
        "indexes": ["category", "cell_type"],
    },
    "exhaustion": {
        "source": VIZ_DATA / "exhaustion.json",
        "type": "nested_json",
        "extract_key": "data",
        "indexes": ["cancer_type", "exhaustion_state"],
    },
    "immune_infiltration": {
        "source": VIZ_DATA / "immune_infiltration.json",
        "type": "nested_json",
        "extract_key": "data",
        "indexes": ["cancer_type", "signature_type"],
    },
    "caf_signatures": {
        "source": VIZ_DATA / "caf_signatures.json",
        "type": "nested_json",
        "extract_key": "data",
        "indexes": ["cancer_type", "caf_subtype"],
    },
    "adjacent_tissue": {
        "source": VIZ_DATA / "adjacent_tissue.json",
        "type": "nested_json",
        "flatten_func": "flatten_dict_of_lists",
        "indexes": ["cancer_type", "signature_type"],
    },
    "cancer_comparison": {
        "source": VIZ_DATA / "cancer_comparison.json",
        "type": "nested_json",
        "extract_key": "data",
        "indexes": ["cell_type", "signature_type"],
    },
    "cancer_types": {
        "source": VIZ_DATA / "cancer_types.json",
        "type": "nested_json",
        "extract_key": "data",
        "indexes": ["cancer_type", "signature"],
    },
    "organ_cancer_matrix": {
        "source": VIZ_DATA / "organ_cancer_matrix.json",
        "type": "nested_json",
        "extract_key": "comparisons",
        "indexes": ["cancer_type", "matched_organ"],
    },
    "cross_atlas": {
        "source": VIZ_DATA / "cross_atlas.json",
        "type": "nested_json",
        "flatten_func": "flatten_cross_atlas",
        "indexes": [],
    },
    "cohort_validation": {
        "source": VIZ_DATA / "cohort_validation.json",
        "type": "nested_json",
        "extract_key": "correlations",
        "indexes": ["signature", "signature_type"],
    },
    # ------- Additional service-referenced JSON files -------
    "cima_biochem_scatter": {
        "source": VIZ_DATA / "cima_biochem_scatter.json",
        "type": "nested_json",
        "extract_key": "samples",
        "indexes": [],
    },
    "cima_eqtl": {
        "source": VIZ_DATA / "cima_eqtl.json",
        "type": "nested_json",
        "extract_key": "eqtls",
        "indexes": ["gene"],
    },
    "cima_eqtl_top": {
        "source": VIZ_DATA / "cima_eqtl_top.json",
        "type": "nested_json",
        "extract_key": "eqtls",
        "indexes": ["gene"],
    },
    "cima_population_stratification": {
        "source": VIZ_DATA / "cima_population_stratification.json",
        "type": "nested_json",
        "flatten_func": "flatten_cross_atlas",
        "indexes": [],
    },
    "disease_sankey": {
        "source": VIZ_DATA / "disease_sankey.json",
        "type": "nested_json",
        "flatten_func": "flatten_dict_of_lists",
        "indexes": [],
    },
    "gene_list": {
        "source": VIZ_DATA / "gene_list.json",
        "type": "flat_json",
        "indexes": [],
    },
    "inflammation_cell_drivers": {
        "source": VIZ_DATA / "inflammation_cell_drivers.json",
        "type": "nested_json",
        "extract_key": "effects",
        "indexes": ["cell_type", "disease"],
    },
    "inflammation_longitudinal": {
        "source": VIZ_DATA / "inflammation_longitudinal.json",
        "type": "nested_json",
        "extract_key": "timepoint_activity",
        "indexes": ["disease", "signature_type"],
    },
    "summary_stats": {
        "source": VIZ_DATA / "summary_stats.json",
        "type": "nested_json",
        "flatten_func": "flatten_cross_atlas",
        "indexes": [],
    },
    "treatment_response": {
        "source": VIZ_DATA / "treatment_response.json",
        "type": "nested_json",
        "flatten_func": "flatten_dict_of_lists",
        "indexes": [],
    },
}

# Files to skip in auto-discovery (already handled above, too large, or not tabular)
SKIP_IN_AUTO_DISCOVER = {
    # Explicitly configured above
    "activity_boxplot.json",
    "inflammation_disease.json",
    "inflammation_disease_filtered.json",
    "singlecell_activity.json",
    "scatlas_celltypes.json",
    "age_bmi_boxplots.json",
    "age_bmi_boxplots_filtered.json",
    "bulk_rnaseq_validation.json",
    "inflammation_severity.json",
    "inflammation_severity_filtered.json",
    "scatlas_organs.json",
    "gene_expression.json",
    "expression_boxplot.json",
    "cima_celltype.json",
    "inflammation_celltype.json",
    "cima_differential.json",
    "inflammation_differential.json",
    "cima_metabolites_top.json",
    "scatlas_organs_top.json",
    "cima_singlecell_cytosig.json",
    "scatlas_normal_singlecell_cytosig.json",
    "scatlas_cancer_singlecell_cytosig.json",
    "scatlas_normal_singlecell_secact.json",
    "scatlas_cancer_singlecell_secact.json",
    "cima_correlations.json",
    "inflammation_correlations.json",
    "inflammation_celltype_correlations.json",
    "cima_celltype_correlations.json",
    "exhaustion.json",
    "immune_infiltration.json",
    "caf_signatures.json",
    "adjacent_tissue.json",
    "cancer_comparison.json",
    "cancer_types.json",
    "organ_cancer_matrix.json",
    "cross_atlas.json",
    "cohort_validation.json",
    # Additional service-referenced files (now explicitly configured)
    "cima_biochem_scatter.json",
    "cima_eqtl.json",
    "cima_eqtl_top.json",
    "cima_population_stratification.json",
    "disease_sankey.json",
    "gene_list.json",
    "inflammation_cell_drivers.json",
    "inflammation_longitudinal.json",
    "summary_stats.json",
    "treatment_response.json",
    # Too large / handled by SQLite scatter pipeline
    "bulk_donor_correlations.json",
    # Non-tabular metadata files
    "celltype_mapping.json",
    "search_index.json",
    "cima_signature_expression.json",
}


# ---------------------------------------------------------------------------
# Auto-discovery for remaining JSON files
# ---------------------------------------------------------------------------
def auto_discover_json(
    db: duckdb.DuckDBPyConnection,
    already_converted: set[str],
) -> int:
    """Scan visualization/data/ for JSON files not yet converted and auto-import them.

    For each file:
      - If it is a flat array of dicts, load directly.
      - If it is a dict with a 'data' key holding a list, extract that.
      - If it is a dict of lists-of-dicts, use flatten_dict_of_lists.
      - Otherwise, try flatten_cross_atlas as a general recursive flattener.
      - Skip files that produce no records or fail.

    Args:
        db: DuckDB connection.
        already_converted: Set of table names already converted.

    Returns:
        Total number of rows from auto-discovered tables.
    """
    total_rows = 0
    json_files = sorted(VIZ_DATA.glob("*.json"))

    for json_path in json_files:
        if json_path.name in SKIP_IN_AUTO_DISCOVER:
            continue

        table_name = json_path.stem.replace("-", "_").replace(".", "_")
        if table_name in already_converted:
            continue

        logger.info("Auto-discovering: %s (%s)", json_path.name, _size_str(json_path))

        try:
            data = load_json(json_path)
        except Exception as e:
            logger.warning("  Failed to load %s: %s", json_path.name, e)
            continue

        records = None

        # Strategy 1: flat array
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                records = data
            else:
                logger.info("  Skipping %s: list but not list-of-dicts", json_path.name)
                continue

        # Strategy 2: dict with 'data' key
        elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            if len(data["data"]) > 0 and isinstance(data["data"][0], dict):
                records = data["data"]
                logger.info("  Using 'data' key extraction")

        # Strategy 3: dict of lists-of-dicts
        elif isinstance(data, dict):
            # Check if top-level values are all lists of dicts
            list_values = {
                k: v for k, v in data.items()
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict)
            }
            if list_values:
                records = flatten_dict_of_lists(data)
                logger.info("  Using dict-of-lists flattening (%d categories)", len(list_values))
            else:
                # Strategy 4: recursive flatten
                records = flatten_cross_atlas(data)
                if records:
                    logger.info("  Using recursive flattening (%d records)", len(records))

        if not records or len(records) == 0:
            logger.info("  Skipping %s: no tabular records extracted", json_path.name)
            continue

        try:
            _drop_table(db, table_name)
            n = 0
            for i in range(0, len(records), CHUNK_SIZE):
                chunk = records[i : i + CHUNK_SIZE]
                df = _stabilize_dtypes(pd.DataFrame(chunk))
                if i == 0:
                    db.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM df')
                else:
                    db.execute(f'INSERT INTO "{table_name}" SELECT * FROM df')
                n += len(df)

            total_rows += n
            already_converted.add(table_name)
            logger.info("  Table '%s': %d rows, %d columns", table_name, n, len(df.columns))
        except Exception as e:
            logger.warning("  Failed to create table '%s': %s", table_name, e)
            continue

    return total_rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def convert_table(
    db: duckdb.DuckDBPyConnection,
    name: str,
    config: dict,
) -> int:
    """Dispatch a single table conversion based on its config.

    Args:
        db: DuckDB connection.
        name: Table name.
        config: Table configuration dict.

    Returns:
        Number of rows inserted, or 0 on failure/skip.
    """
    source = config["source"]
    table_type = config["type"]
    indexes = config.get("indexes", [])

    if table_type == "csv":
        return convert_csv_files(db, name, source, indexes)

    # JSON types
    source_path = Path(source) if isinstance(source, str) else source
    if not source_path.exists():
        logger.warning("Source file not found for table '%s': %s", name, source_path)
        return 0

    if table_type == "flat_json":
        return convert_flat_json(db, name, source_path, indexes)
    elif table_type == "nested_json":
        return convert_nested_json(
            db,
            name,
            source_path,
            extract_key=config.get("extract_key"),
            flatten_func=config.get("flatten_func"),
            indexes=indexes,
        )
    else:
        logger.error("Unknown table type '%s' for table '%s'", table_type, name)
        return 0


def print_summary(db: duckdb.DuckDBPyConnection, output_path: str, elapsed: float):
    """Print a summary of all tables in the database."""
    tables = db.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchall()

    print("\n" + "=" * 80)
    print("CONVERSION SUMMARY")
    print("=" * 80)

    total_rows = 0
    print(f"\n{'Table':<45} {'Rows':>12} {'Columns':>8}")
    print("-" * 67)

    for (table_name,) in tables:
        try:
            row_count = db.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]
            col_count = db.execute(
                f"SELECT COUNT(*) FROM information_schema.columns "
                f"WHERE table_name = '{table_name}'"
            ).fetchone()[0]
            total_rows += row_count
            print(f"  {table_name:<43} {row_count:>12,} {col_count:>8}")
        except Exception as e:
            print(f"  {table_name:<43} {'ERROR':>12} {str(e)[:20]}")

    print("-" * 67)
    print(f"  {'TOTAL':<43} {total_rows:>12,} {len(tables):>5} tables")

    # Database file size
    db_path = Path(output_path)
    if db_path.exists():
        print(f"\n  Database size: {_size_str(db_path)}")

    print(f"  Total time:   {elapsed:.1f}s")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Convert visualization JSON and CSV data to DuckDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available tables:
{chr(10).join(f'  - {name}' for name in sorted(TABLE_CONFIGS.keys()))}
  - (auto-discovered JSON files)
  - scatter_targets, scatter_points (from SQLite)

Examples:
  # Convert everything
  python scripts/convert_data_to_duckdb.py --all

  # Convert a single table
  python scripts/convert_data_to_duckdb.py --table activity

  # Convert multiple tables
  python scripts/convert_data_to_duckdb.py --table activity --table inflammation_disease

  # Custom output path
  python scripts/convert_data_to_duckdb.py --all --output /data/parks34/atlas.duckdb

  # List available tables
  python scripts/convert_data_to_duckdb.py --list
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all configured tables, migrate SQLite, and auto-discover remaining JSON",
    )
    parser.add_argument(
        "--table",
        type=str,
        action="append",
        dest="tables",
        help="Convert a specific table (can be repeated). Use --list to see available tables.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="atlas_data.duckdb",
        help="Output DuckDB file path (default: atlas_data.duckdb in current directory)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available table names and exit",
    )
    parser.add_argument(
        "--sqlite-only",
        action="store_true",
        help="Only migrate the SQLite scatter database",
    )
    parser.add_argument(
        "--auto-discover-only",
        action="store_true",
        help="Only auto-discover and convert remaining JSON files",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("Configured tables:")
        for name, config in sorted(TABLE_CONFIGS.items()):
            source = config["source"]
            src_str = str(source)
            if isinstance(source, Path):
                src_str = source.name
            print(f"  {name:<45} ({config['type']}: {src_str})")
        print("\nAdditional sources:")
        print(f"  {'scatter_targets, scatter_points':<45} (SQLite: {SQLITE_DB})")
        print(f"  {'(auto-discovered)':<45} (remaining JSON in {VIZ_DATA})")
        return

    # Validate arguments
    if not args.all and not args.tables and not args.sqlite_only and not args.auto_discover_only:
        parser.print_help()
        print("\nError: Must specify --all, --table TABLE, --sqlite-only, or --auto-discover-only",
              file=sys.stderr)
        sys.exit(1)

    # Validate table names
    if args.tables:
        for t in args.tables:
            if t not in TABLE_CONFIGS:
                print(f"Error: Unknown table '{t}'. Use --list to see available tables.",
                      file=sys.stderr)
                sys.exit(1)

    # Open DuckDB
    output_path = args.output
    logger.info("Opening DuckDB database: %s", output_path)
    db = duckdb.connect(output_path)

    # Enable progress bar for large operations
    db.execute("SET enable_progress_bar = true")

    t_start = time.time()
    converted = set()
    total_tables = 0
    total_rows = 0

    if args.all:
        # 1. Convert all configured tables
        logger.info("Converting %d configured tables ...", len(TABLE_CONFIGS))
        for name, config in TABLE_CONFIGS.items():
            try:
                n = convert_table(db, name, config)
                if n > 0:
                    converted.add(name)
                    total_tables += 1
                    total_rows += n
            except Exception as e:
                logger.error("Failed to convert table '%s': %s", name, e, exc_info=True)

        # 2. Migrate SQLite
        if SQLITE_DB.exists():
            logger.info("\nMigrating SQLite scatter database ...")
            try:
                n = migrate_sqlite(db, SQLITE_DB)
                if n > 0:
                    converted.add("scatter_targets")
                    converted.add("scatter_points")
                    total_tables += 2
                    total_rows += n
            except Exception as e:
                logger.error("Failed to migrate SQLite: %s", e, exc_info=True)
        else:
            logger.info("SQLite scatter DB not found, skipping. (%s)", SQLITE_DB)

        # 3. Auto-discover remaining JSON files
        logger.info("\nAuto-discovering remaining JSON files ...")
        try:
            n = auto_discover_json(db, converted)
            total_rows += n
        except Exception as e:
            logger.error("Auto-discovery failed: %s", e, exc_info=True)

    elif args.sqlite_only:
        n = migrate_sqlite(db, SQLITE_DB)
        total_rows += n
        total_tables += 2 if n > 0 else 0

    elif args.auto_discover_only:
        n = auto_discover_json(db, converted)
        total_rows += n

    else:
        # Convert specific tables
        for name in args.tables:
            try:
                n = convert_table(db, name, TABLE_CONFIGS[name])
                if n > 0:
                    converted.add(name)
                    total_tables += 1
                    total_rows += n
            except Exception as e:
                logger.error("Failed to convert table '%s': %s", name, e, exc_info=True)

    # Checkpoint to ensure all data is flushed
    db.execute("CHECKPOINT")

    elapsed = time.time() - t_start
    print_summary(db, output_path, elapsed)

    db.close()
    logger.info("Database closed: %s", output_path)


if __name__ == "__main__":
    main()
