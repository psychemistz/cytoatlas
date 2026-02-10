"""DuckDB-backed repository implementation for atlas data.

Provides a single-file analytical backend that replaces the combination
of JSON, Parquet, and SQLite scatter repositories with a unified DuckDB
database.  All queries are read-only (the DB is opened with read_only=True)
and run through ``asyncio.run_in_executor`` so they never block the
FastAPI event loop.

Security: every user-supplied value reaches DuckDB via parameterised
queries only -- string interpolation is **never** used for values.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from functools import partial
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.repositories.base import BaseRepository

logger = logging.getLogger(__name__)

# Default batch size for fetchmany()-based streaming.
_STREAM_BATCH_SIZE = 2_000

# Tables that are known to exist in the DuckDB atlas file.
# Used by get_data() and stream_results() as a safelist to prevent
# arbitrary table access via user input.
_KNOWN_TABLES: frozenset[str] = frozenset(
    {
        # Core protocol tables
        "activity",
        "correlations",
        "differential",
        # Large visualization JSON tables
        "activity_boxplot",
        "inflammation_disease",
        "inflammation_disease_filtered",
        "singlecell_activity",
        "scatlas_celltypes",
        "age_bmi_boxplots",
        "age_bmi_boxplots_filtered",
        "bulk_validation",
        # Cross-sample correlation CSVs
        "cross_sample_correlations",
        # Medium-size flat JSON tables
        "inflammation_severity",
        "inflammation_severity_filtered",
        "scatlas_organs",
        "gene_expression",
        "expression_boxplot",
        "cima_celltype",
        "inflammation_celltype",
        "cima_differential",
        "inflammation_differential",
        "cima_metabolites_top",
        "scatlas_organs_top",
        "cima_singlecell_cytosig",
        "scatlas_normal_singlecell_cytosig",
        "scatlas_cancer_singlecell_cytosig",
        "scatlas_normal_singlecell_secact",
        "scatlas_cancer_singlecell_secact",
        # Nested JSON tables
        "cima_correlations",
        "inflammation_correlations",
        "inflammation_celltype_correlations",
        "cima_celltype_correlations",
        "exhaustion",
        "immune_infiltration",
        "caf_signatures",
        "adjacent_tissue",
        "cancer_comparison",
        "cancer_types",
        "organ_cancer_matrix",
        "cross_atlas",
        "cohort_validation",
        # Auto-discovered tables (loaded by services)
        "cima_biochem_scatter",
        "cima_eqtl",
        "cima_eqtl_top",
        "cima_population_stratification",
        "disease_sankey",
        "gene_list",
        "inflammation_cell_drivers",
        "inflammation_longitudinal",
        "summary_stats",
        "treatment_response",
        # SQLite migration tables
        "scatter_targets",
        "scatter_points",
        # Legacy aliases
        "cima_activity",
        "inflammation_activity",
        "scatlas_activity",
        "bulk_donor_correlations",
        "bulk_rnaseq_validation",
        "cross_sample_validation",
        "single_cell_validation",
    }
)


def _validate_identifier(name: str) -> str:
    """Validate that *name* is a safe SQL identifier (alphanumeric + underscores).

    Raises ``ValueError`` if the name contains anything unexpected.
    This is defence-in-depth on top of the ``_KNOWN_TABLES`` safelist.
    """
    if not name.isidentifier():
        raise ValueError(f"Invalid identifier: {name!r}")
    return name


class DuckDBRepository(BaseRepository):
    """Read-only DuckDB repository implementing the ``AtlasRepository`` protocol.

    Parameters
    ----------
    db_path : Path | None
        Explicit path to the ``.duckdb`` file.  Falls back to the
        ``duckdb_atlas_path`` setting from ``app.config``.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        settings = get_settings()
        self._db_path: Path = db_path or settings.duckdb_atlas_path
        self._conn: Any | None = None  # duckdb.DuckDBPyConnection
        self._available: bool | None = None

        # Simple query-level stats (not a full cache -- DuckDB manages its
        # own buffer pool internally).
        self._query_count: int = 0
        self._rows_returned: int = 0

    # ------------------------------------------------------------------
    #  Connection management
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Return ``True`` if the DuckDB file exists and can be opened."""
        if self._available is None:
            if not self._db_path.exists():
                logger.warning(
                    "DuckDB atlas file not found: %s  "
                    "(fall back to JSON/Parquet repositories)",
                    self._db_path,
                )
                self._available = False
            else:
                try:
                    self._get_conn()
                    self._available = True
                except Exception:
                    logger.warning(
                        "DuckDB atlas file exists but failed to open: %s",
                        self._db_path,
                        exc_info=True,
                    )
                    self._available = False
        return self._available

    def _get_conn(self) -> Any:
        """Return (or lazily create) a read-only DuckDB connection."""
        if self._conn is None:
            import duckdb

            self._conn = duckdb.connect(str(self._db_path), read_only=True)
            # Optimise for analytical reads.
            self._conn.execute("SET threads TO 4")
            self._conn.execute("SET memory_limit = '2GB'")
        return self._conn

    def _run_sync(self, fn: Any, *args: Any) -> Any:
        """Schedule *fn* on the default executor (non-blocking)."""
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, partial(fn, *args))

    # ------------------------------------------------------------------
    #  AtlasRepository protocol -- async public API
    # ------------------------------------------------------------------

    async def get_activity(
        self,
        atlas: str,
        signature_type: str,
        **filters: Any,
    ) -> list[dict]:
        """Return activity rows for *atlas* and *signature_type*.

        Queries the ``activity`` table with ``atlas`` and
        ``signature_type`` as mandatory predicates and any additional
        keyword filters pushed down to SQL WHERE clauses.
        """
        all_filters = {"atlas": atlas, "signature_type": signature_type, **filters}
        return await self._run_sync(
            self._query_table_sync, "activity", all_filters
        )

    async def get_correlations(
        self,
        atlas: str,
        variable: str,
        **filters: Any,
    ) -> list[dict]:
        """Return correlation rows for *atlas* and *variable*."""
        all_filters = {"atlas": atlas, "variable": variable, **filters}
        return await self._run_sync(
            self._query_table_sync, "correlations", all_filters
        )

    async def get_differential(
        self,
        atlas: str,
        comparison: str,
        **filters: Any,
    ) -> list[dict]:
        """Return differential-activity rows for *atlas* and *comparison*."""
        all_filters = {"atlas": atlas, "comparison": comparison, **filters}
        return await self._run_sync(
            self._query_table_sync, "differential", all_filters
        )

    async def get_data(
        self,
        data_type: str,
        **filters: Any,
    ) -> list[dict]:
        """Generic data access -- *data_type* maps to a DuckDB table name.

        Only tables listed in ``_KNOWN_TABLES`` may be queried.

        Raises
        ------
        ValueError
            If *data_type* is not in the safelist.
        FileNotFoundError
            If the DuckDB file is not available.
        """
        if not self.available:
            raise FileNotFoundError(
                f"DuckDB atlas file not found: {self._db_path}"
            )

        table = _validate_identifier(data_type)
        if table not in _KNOWN_TABLES:
            raise ValueError(
                f"Unknown data type: {data_type!r}. "
                f"Known tables: {sorted(_KNOWN_TABLES)}"
            )

        return await self._run_sync(self._query_table_sync, table, filters)

    async def stream_results(
        self,
        data_type: str,
        **filters: Any,
    ) -> AsyncIterator[dict]:
        """Stream rows from *data_type* in batches via ``fetchmany()``.

        This avoids materialising the full result set in memory for
        very large tables (e.g. ``singlecell_activity``).
        """
        if not self.available:
            raise FileNotFoundError(
                f"DuckDB atlas file not found: {self._db_path}"
            )

        table = _validate_identifier(data_type)
        if table not in _KNOWN_TABLES:
            raise ValueError(f"Unknown data type: {data_type!r}")

        sql, params = self._build_select(table, filters)

        # Fetch batches on the executor, yield rows one by one in the
        # async context.
        conn = self._get_conn()
        result = conn.execute(sql, params)
        columns = [desc[0] for desc in result.description]

        while True:
            batch = await self._run_sync(result.fetchmany, _STREAM_BATCH_SIZE)
            if not batch:
                break
            for row in batch:
                yield dict(zip(columns, row))

    # ------------------------------------------------------------------
    #  Extended API -- single-cell interactive queries
    # ------------------------------------------------------------------

    async def query_cells(
        self,
        atlas: str,
        *,
        cell_type: str | None = None,
        organ: str | None = None,
        disease: str | None = None,
        signature_type: str | None = None,
        target: str | None = None,
        min_activity: float | None = None,
        max_activity: float | None = None,
        limit: int = 1_000,
    ) -> list[dict]:
        """Run an interactive single-cell query with flexible predicates.

        This is a new capability enabled by DuckDB's analytical engine
        that was not feasible with JSON/Parquet backends.

        Parameters
        ----------
        atlas : str
            Atlas name (``cima``, ``inflammation``, ``scatlas``).
        cell_type, organ, disease, signature_type, target : str | None
            Optional equality filters.
        min_activity, max_activity : float | None
            Optional range filters on the ``activity`` column.
        limit : int
            Maximum number of rows to return (default 1 000, capped at
            50 000 to protect memory).

        Returns
        -------
        list[dict]
            Matching cell-level records.
        """
        if not self.available:
            raise FileNotFoundError(
                f"DuckDB atlas file not found: {self._db_path}"
            )

        safe_limit = min(max(limit, 1), 50_000)

        return await self._run_sync(
            self._query_cells_sync,
            atlas,
            cell_type,
            organ,
            disease,
            signature_type,
            target,
            min_activity,
            max_activity,
            safe_limit,
        )

    # ------------------------------------------------------------------
    #  Cache / stats interface (matches JSONRepository)
    # ------------------------------------------------------------------

    def get_cache_stats(self) -> dict[str, Any]:
        """Return query statistics.

        DuckDB manages its own internal buffer pool so there is no
        application-level LRU cache to report on.  Instead we expose
        query counts and total rows returned.
        """
        return {
            "backend": "duckdb",
            "db_path": str(self._db_path),
            "available": self.available,
            "query_count": self._query_count,
            "rows_returned": self._rows_returned,
        }

    def clear_cache(self) -> None:
        """No-op -- DuckDB manages its own buffer pool."""

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    #  Private synchronous helpers (run on executor)
    # ------------------------------------------------------------------

    def _query_table_sync(
        self,
        table: str,
        filters: dict[str, Any],
    ) -> list[dict]:
        """Execute a parameterised SELECT on *table* and return dicts."""
        if not self.available:
            raise FileNotFoundError(
                f"DuckDB atlas file not found: {self._db_path}"
            )

        sql, params = self._build_select(table, filters)
        conn = self._get_conn()
        result = conn.execute(sql, params)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()

        self._query_count += 1
        self._rows_returned += len(rows)

        return [dict(zip(columns, row)) for row in rows]

    def _query_cells_sync(
        self,
        atlas: str,
        cell_type: str | None,
        organ: str | None,
        disease: str | None,
        signature_type: str | None,
        target: str | None,
        min_activity: float | None,
        max_activity: float | None,
        limit: int,
    ) -> list[dict]:
        """Build and execute a single-cell query with range predicates."""
        clauses: list[str] = ["atlas = $1"]
        params: list[Any] = [atlas]
        idx = 2  # DuckDB positional parameters are 1-based

        if cell_type is not None:
            clauses.append(f"cell_type = ${idx}")
            params.append(cell_type)
            idx += 1
        if organ is not None:
            clauses.append(f"organ = ${idx}")
            params.append(organ)
            idx += 1
        if disease is not None:
            clauses.append(f"disease = ${idx}")
            params.append(disease)
            idx += 1
        if signature_type is not None:
            clauses.append(f"signature_type = ${idx}")
            params.append(signature_type)
            idx += 1
        if target is not None:
            clauses.append(f"target = ${idx}")
            params.append(target)
            idx += 1
        if min_activity is not None:
            clauses.append(f"activity >= ${idx}")
            params.append(min_activity)
            idx += 1
        if max_activity is not None:
            clauses.append(f"activity <= ${idx}")
            params.append(max_activity)
            idx += 1

        where = " AND ".join(clauses)
        sql = (
            f"SELECT * FROM singlecell_activity "
            f"WHERE {where} "
            f"LIMIT ${idx}"
        )
        params.append(limit)

        conn = self._get_conn()
        result = conn.execute(sql, params)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()

        self._query_count += 1
        self._rows_returned += len(rows)

        return [dict(zip(columns, row)) for row in rows]

    # ------------------------------------------------------------------
    #  SQL builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_select(
        table: str,
        filters: dict[str, Any],
    ) -> tuple[str, list[Any]]:
        """Build a parameterised ``SELECT * FROM <table> WHERE ...`` statement.

        Parameters are expressed as ``$1``, ``$2``, ... (DuckDB positional
        style).  List-valued filters are expanded to ``IN ($n, $n+1, ...)``.

        Returns
        -------
        tuple[str, list[Any]]
            The SQL string and the flat parameter list.
        """
        # table has already been validated via _validate_identifier + _KNOWN_TABLES
        clauses: list[str] = []
        params: list[Any] = []
        idx = 1

        for key, value in filters.items():
            if value is None:
                continue

            col = _validate_identifier(key)

            if isinstance(value, list):
                if not value:
                    continue
                placeholders = ", ".join(f"${idx + i}" for i in range(len(value)))
                clauses.append(f"{col} IN ({placeholders})")
                params.extend(value)
                idx += len(value)
            else:
                clauses.append(f"{col} = ${idx}")
                params.append(value)
                idx += 1

        if clauses:
            where = " WHERE " + " AND ".join(clauses)
        else:
            where = ""

        sql = f"SELECT * FROM {table}{where}"
        return sql, params
