#!/usr/bin/env python3
"""Convert spatial analysis results (SpatialCorpus-110M) to DuckDB.

Creates spatial_data.duckdb with tables for:
  - Technology-stratified activity results
  - Tissue-level summaries
  - Neighborhood activity patterns
  - Cross-technology reproducibility
  - Dataset metadata and gene coverage

Separate from atlas_data.duckdb per ADR-004 (multi-dataset storage).

Usage:
  python scripts/convert_spatial_to_duckdb.py --all
  python scripts/convert_spatial_to_duckdb.py --table spatial_activity
  python scripts/convert_spatial_to_duckdb.py --list
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
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
RESULTS = PROJECT_ROOT / "results"
VIZ_DATA = PROJECT_ROOT / "visualization" / "data"
DEFAULT_OUTPUT = PROJECT_ROOT / "spatial_data.duckdb"

SPATIAL_DIR = RESULTS / "spatial"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path: Path):
    """Load JSON file using orjson if available, else stdlib json."""
    try:
        import orjson
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    except ImportError:
        with open(path) as f:
            return json.load(f)
    except Exception:
        logger.info("  orjson failed, falling back to stdlib json for %s", path.name)
        with open(path) as f:
            return json.load(f)


def _stabilize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent column types, replace NaN with None for DuckDB."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).replace("nan", None).replace("None", None)
        elif df[col].dtype in (np.float64, np.float32):
            df[col] = df[col].where(df[col].notna(), None)
    return df


def _create_indexes(db, table: str, columns: list[str]) -> None:
    """Create indexes on specified columns if they exist in the table."""
    existing = {row[0] for row in db.execute(f"DESCRIBE {table}").fetchall()}
    for col in columns:
        if col in existing:
            idx_name = f"idx_{table}_{col}"
            try:
                db.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({col})")
            except Exception as e:
                logger.debug("Index %s skipped: %s", idx_name, e)


def _ingest_csv(db, table: str, csv_path: Path, indexes: list[str] | None = None) -> int:
    """Load a CSV file into a DuckDB table. Returns row count."""
    if not csv_path.exists():
        logger.warning("  CSV not found: %s", csv_path)
        return 0

    df = pd.read_csv(csv_path)
    df = _stabilize_dtypes(df)
    db.execute(f"DROP TABLE IF EXISTS {table}")
    db.execute(f"CREATE TABLE {table} AS SELECT * FROM df")

    count = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    logger.info("  %s: %s rows, %s columns", table, f"{count:,}", len(df.columns))

    if indexes:
        _create_indexes(db, table, indexes)

    return count


def _ingest_json(db, table: str, json_path: Path, indexes: list[str] | None = None) -> int:
    """Load a JSON file (flat array of records) into a DuckDB table. Returns row count."""
    if not json_path.exists():
        logger.warning("  JSON not found: %s", json_path)
        return 0

    data = load_json(json_path)
    if isinstance(data, dict) and len(data) == 1:
        data = list(data.values())[0]
    if not isinstance(data, list):
        logger.warning("  Expected list in %s, got %s", json_path.name, type(data).__name__)
        return 0

    df = pd.DataFrame(data)
    df = _stabilize_dtypes(df)
    db.execute(f"DROP TABLE IF EXISTS {table}")
    db.execute(f"CREATE TABLE {table} AS SELECT * FROM df")

    count = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    logger.info("  %s: %s rows, %s columns", table, f"{count:,}", len(df.columns))

    if indexes:
        _create_indexes(db, table, indexes)

    return count


# ---------------------------------------------------------------------------
# Table configurations
# ---------------------------------------------------------------------------
TABLE_CONFIGS = {
    "spatial_activity": {
        "source": SPATIAL_DIR / "spatial_activity_by_tissue.csv",
        "type": "csv",
        "indexes": ["technology", "tissue", "cell_type", "signature_type"],
        "description": "Technology x tissue x cell_type activity",
    },
    "spatial_tissue_summary": {
        "source": SPATIAL_DIR / "spatial_activity_by_tissue_summary.csv",
        "type": "csv",
        "indexes": ["tissue", "signature_type"],
        "description": "Tissue-level activity summary",
    },
    "spatial_neighborhood": {
        "source": SPATIAL_DIR / "spatial_neighborhood_activity.csv",
        "type": "csv",
        "indexes": ["tissue", "niche_cluster", "signature_type"],
        "description": "Niche-level activity patterns",
    },
    "spatial_technology_comparison": {
        "source": SPATIAL_DIR / "spatial_technology_comparison.csv",
        "type": "csv",
        "indexes": ["technology", "tissue", "signature_type"],
        "description": "Cross-technology reproducibility metrics",
    },
    "spatial_dataset_metadata": {
        "source": SPATIAL_DIR / "spatial_dataset_metadata.csv",
        "type": "csv",
        "indexes": ["technology", "tissue", "species"],
        "description": "251 dataset catalog with technology/tissue/species",
    },
    "spatial_coordinates": {
        "source": SPATIAL_DIR / "spatial_coordinates_sampled.csv",
        "type": "csv",
        "indexes": ["dataset_id"],
        "description": "Sampled spatial coordinates for visualization",
    },
    "spatial_gene_coverage": {
        "source": SPATIAL_DIR / "spatial_gene_coverage.csv",
        "type": "csv",
        "indexes": ["technology", "signature_type"],
        "description": "Gene panel coverage per technology vs signatures",
    },

    # Visualization JSON tables
    "spatial_tissue_activity_viz": {
        "source": VIZ_DATA / "spatial_tissue_activity.json",
        "type": "json",
        "indexes": ["tissue", "signature_type"],
        "description": "Tissue activity visualization data",
    },
    "spatial_technology_comparison_viz": {
        "source": VIZ_DATA / "spatial_technology_comparison.json",
        "type": "json",
        "indexes": ["technology", "tissue"],
        "description": "Technology comparison visualization data",
    },
    "spatial_gene_coverage_viz": {
        "source": VIZ_DATA / "spatial_gene_coverage.json",
        "type": "json",
        "indexes": ["technology"],
        "description": "Gene coverage visualization data",
    },
}


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
def convert_table(db, table_name: str) -> int:
    """Convert a single table. Returns row count."""
    config = TABLE_CONFIGS[table_name]
    source = config["source"]
    table_type = config["type"]
    indexes = config.get("indexes", [])

    logger.info("Converting %s (%s) ...", table_name, config["description"])

    if table_type == "csv":
        return _ingest_csv(db, table_name, source, indexes)
    elif table_type == "json":
        return _ingest_json(db, table_name, source, indexes)
    else:
        logger.warning("  Unknown type: %s", table_type)
        return 0


def convert_all(output_path: Path) -> None:
    """Convert all spatial data to DuckDB."""
    start = time.time()
    logger.info("Creating spatial DuckDB: %s", output_path)

    db = duckdb.connect(str(output_path))
    db.execute("SET threads TO 4")
    db.execute("SET memory_limit = '4GB'")

    total_rows = 0
    tables_created = 0

    for table_name in TABLE_CONFIGS:
        rows = convert_table(db, table_name)
        if rows > 0:
            total_rows += rows
            tables_created += 1

    # Summary
    elapsed = time.time() - start
    db_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Done: %d tables, %s total rows, %.1f MB, %.1f seconds",
        tables_created,
        f"{total_rows:,}",
        db_size,
        elapsed,
    )
    db.close()


def list_tables() -> None:
    """Print available tables."""
    print(f"\n{'Table':<40} {'Type':<6} {'Description'}")
    print("-" * 90)
    for name, config in TABLE_CONFIGS.items():
        print(f"{name:<40} {config['type']:<6} {config['description']}")
    print(f"\nTotal: {len(TABLE_CONFIGS)} tables")


def main():
    parser = argparse.ArgumentParser(description="Convert spatial data to DuckDB")
    parser.add_argument("--all", action="store_true", help="Convert all tables")
    parser.add_argument("--table", type=str, help="Convert a specific table")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output DuckDB path")
    parser.add_argument("--list", action="store_true", help="List available tables")
    args = parser.parse_args()

    if args.list:
        list_tables()
        return

    output_path = Path(args.output)

    if args.all:
        convert_all(output_path)
    elif args.table:
        if args.table not in TABLE_CONFIGS:
            logger.error("Unknown table: %s. Use --list to see available tables.", args.table)
            sys.exit(1)
        db = duckdb.connect(str(output_path))
        db.execute("SET threads TO 4")
        convert_table(db, args.table)
        db.close()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
