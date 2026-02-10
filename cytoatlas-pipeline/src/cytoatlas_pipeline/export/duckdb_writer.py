"""
DuckDB output writer for direct table insertion.

Writes pipeline results directly to DuckDB tables for serving via the API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

logger = logging.getLogger(__name__)


class DuckDBWriter:
    """Writes pipeline results directly to DuckDB tables.

    DuckDB is efficient for:
    - Analytical queries (columnar storage)
    - Compressed on-disk format (2-10x smaller than JSON)
    - Zero-copy integration with pandas/Arrow
    - Concurrent read access from API process
    """

    def __init__(
        self,
        db_path: str | Path,
        read_only: bool = False,
    ):
        if not HAS_DUCKDB:
            raise ImportError("duckdb is required for DuckDB export: pip install duckdb")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.db_path), read_only=read_only)

    def write_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        indexes: Optional[list[str]] = None,
        if_exists: str = "replace",
    ) -> int:
        """Write a DataFrame to a DuckDB table.

        Parameters
        ----------
        df : pd.DataFrame
            Data to write.
        table_name : str
            Target table name.
        indexes : list[str], optional
            Columns to create indexes on (for query performance).
        if_exists : str
            'replace' drops existing table, 'append' adds rows.

        Returns
        -------
        int
            Number of rows written.
        """
        if if_exists == "replace":
            self._conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        # Register the DataFrame and create table from it
        self._conn.register("_tmp_df", df)

        if if_exists == "replace" or not self._table_exists(table_name):
            self._conn.execute(
                f"CREATE TABLE {table_name} AS SELECT * FROM _tmp_df"
            )
        else:
            self._conn.execute(
                f"INSERT INTO {table_name} SELECT * FROM _tmp_df"
            )

        self._conn.unregister("_tmp_df")

        # Create indexes
        if indexes:
            for col in indexes:
                if col in df.columns:
                    idx_name = f"idx_{table_name}_{col}"
                    self._conn.execute(f"DROP INDEX IF EXISTS {idx_name}")
                    self._conn.execute(
                        f"CREATE INDEX {idx_name} ON {table_name} ({col})"
                    )

        n_rows = self._conn.execute(
            f"SELECT count(*) FROM {table_name}"
        ).fetchone()[0]

        logger.info("Wrote %d rows to DuckDB table %s", n_rows, table_name)
        return n_rows

    def write_activity(
        self,
        activity: pd.DataFrame,
        atlas: str,
        sig_type: str,
        table_name: str = "activity",
        metadata: Optional[pd.DataFrame] = None,
    ) -> int:
        """Write activity matrix in long format to DuckDB.

        Parameters
        ----------
        activity : pd.DataFrame
            Activity matrix (signatures x samples).
        atlas : str
            Atlas name.
        sig_type : str
            Signature type (cytosig, secact, lincytosig).
        table_name : str
            Target table name.
        metadata : pd.DataFrame, optional
            Sample metadata to join.

        Returns
        -------
        int
            Number of rows written.
        """
        # Melt to long format
        df = activity.T.reset_index()
        df = df.melt(
            id_vars=["index"],
            var_name="signature",
            value_name="activity",
        )
        df = df.rename(columns={"index": "sample"})
        df["atlas"] = atlas
        df["sig_type"] = sig_type

        # Merge metadata if provided
        if metadata is not None:
            df = df.merge(
                metadata.reset_index(),
                left_on="sample",
                right_on=metadata.index.name or "index",
                how="left",
            )

        return self.write_table(
            df,
            table_name,
            indexes=["atlas", "sig_type", "signature"],
            if_exists="append",
        )

    def write_correlations(
        self,
        correlations: pd.DataFrame,
        atlas: str,
        table_name: str = "correlations",
    ) -> int:
        """Write correlation results to DuckDB."""
        df = correlations.copy()
        df["atlas"] = atlas
        return self.write_table(
            df,
            table_name,
            indexes=["atlas", "signature", "variable"],
            if_exists="append",
        )

    def write_differential(
        self,
        results: pd.DataFrame,
        atlas: str,
        comparison: str,
        table_name: str = "differential",
    ) -> int:
        """Write differential analysis results to DuckDB."""
        df = results.copy()
        df["atlas"] = atlas
        df["comparison"] = comparison
        return self.write_table(
            df,
            table_name,
            indexes=["atlas", "comparison", "signature"],
            if_exists="append",
        )

    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        result = self._conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
        return result[0] > 0

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "DuckDBWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def write_activity_duckdb(
    activity: pd.DataFrame,
    db_path: str | Path,
    atlas: str,
    sig_type: str,
) -> int:
    """Convenience function to write activity to DuckDB."""
    with DuckDBWriter(db_path) as writer:
        return writer.write_activity(activity, atlas, sig_type)
