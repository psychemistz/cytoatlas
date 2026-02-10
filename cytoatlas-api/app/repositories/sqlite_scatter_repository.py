"""SQLite-backed repository for validation scatter data.

Provides indexed lookups into a single DB file containing all scatter targets
and zlib-compressed point BLOBs. Uses stdlib sqlite3 with asyncio.run_in_executor
for non-blocking FastAPI integration.

DB schema:
    scatter_targets  — metadata (source, atlas, level, sigtype, target, rho, pval, …)
    scatter_points   — zlib-compressed JSON point arrays keyed by target_id

.. deprecated::
    The scatter_targets and scatter_points tables are migrated into the DuckDB
    atlas file by ``scripts/convert_data_to_duckdb.py``.  Once the DuckDB
    migration is complete, this module will be removed in favour of
    :class:`~app.repositories.duckdb_repository.DuckDBRepository`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import zlib
from functools import partial
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


class SQLiteScatterRepository:
    """Read-only repository over the validation_scatter.db SQLite file.

    .. deprecated::
        Prefer DuckDBRepository once atlas_data.duckdb is generated.
        The scatter tables are migrated into DuckDB by the conversion script.
    """

    def __init__(self, db_path: Path | None = None):
        settings = get_settings()
        self._db_path = db_path or settings.sqlite_scatter_db_path
        self._conn: sqlite3.Connection | None = None
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = Path(self._db_path).exists()
            if self._available:
                try:
                    self._get_conn()
                except Exception:
                    logger.warning("SQLite scatter DB exists but failed to open: %s", self._db_path)
                    self._available = False
        return self._available

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA mmap_size=268435456")  # 256 MB
            self._conn.execute("PRAGMA cache_size=-65536")     # 64 MB
            self._conn.execute("PRAGMA query_only=ON")
        return self._conn

    def _run(self, fn, *args):
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, partial(fn, *args))

    # ------------------------------------------------------------------ #
    #  Metadata queries                                                    #
    # ------------------------------------------------------------------ #

    def _list_atlases_sync(self, source: str) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT atlas FROM scatter_targets WHERE source=? ORDER BY atlas",
            (source,),
        ).fetchall()
        return [r["atlas"] for r in rows]

    async def list_atlases(self, source: str) -> list[str]:
        return await self._run(self._list_atlases_sync, source)

    def _list_levels_sync(self, source: str, atlas: str) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT level FROM scatter_targets "
            "WHERE source=? AND atlas=? AND level!='' ORDER BY level",
            (source, atlas),
        ).fetchall()
        return [r["level"] for r in rows]

    async def list_levels(self, source: str, atlas: str) -> list[str]:
        return await self._run(self._list_levels_sync, source, atlas)

    # ------------------------------------------------------------------ #
    #  Target listing (metadata only, no points)                           #
    # ------------------------------------------------------------------ #

    def _get_targets_sync(
        self, source: str, atlas: str, level: str, sigtype: str
    ) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT target, gene, rho, pval, n, rho_ci_lo, rho_ci_hi, "
            "       groups_json, extra_json "
            "FROM scatter_targets "
            "WHERE source=? AND atlas=? AND level=? AND sigtype=?",
            (source, atlas, level, sigtype),
        ).fetchall()
        result = []
        for r in rows:
            entry: dict[str, Any] = {
                "target": r["target"],
                "gene": r["gene"],
                "rho": r["rho"],
                "pval": r["pval"],
                "n": r["n"],
                "significant": (r["pval"] or 1.0) < 0.05,
            }
            if r["rho_ci_lo"] is not None:
                entry["rho_ci"] = [r["rho_ci_lo"], r["rho_ci_hi"]]
            if r["groups_json"]:
                entry["celltypes"] = json.loads(r["groups_json"])
            if r["extra_json"]:
                entry.update(json.loads(r["extra_json"]))
            result.append(entry)
        return result

    async def get_targets(
        self, source: str, atlas: str, level: str, sigtype: str
    ) -> list[dict]:
        return await self._run(self._get_targets_sync, source, atlas, level, sigtype)

    # ------------------------------------------------------------------ #
    #  Single target scatter (metadata + decompressed points)              #
    # ------------------------------------------------------------------ #

    def _get_scatter_sync(
        self, source: str, atlas: str, level: str, sigtype: str, target: str
    ) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT t.target, t.gene, t.rho, t.pval, t.n, "
            "       t.rho_ci_lo, t.rho_ci_hi, t.groups_json, t.extra_json, "
            "       p.points_blob "
            "FROM scatter_targets t "
            "JOIN scatter_points p ON p.target_id = t.id "
            "WHERE t.source=? AND t.atlas=? AND t.level=? AND t.sigtype=? AND t.target=?",
            (source, atlas, level, sigtype, target),
        ).fetchone()
        if row is None:
            return None

        entry: dict[str, Any] = {
            "target": row["target"],
            "gene": row["gene"],
            "rho": row["rho"],
            "pval": row["pval"],
            "n": row["n"],
        }
        if row["rho_ci_lo"] is not None:
            entry["rho_ci"] = [row["rho_ci_lo"], row["rho_ci_hi"]]
        if row["groups_json"]:
            groups = json.loads(row["groups_json"])
            entry["celltypes"] = groups
            entry["groups"] = groups
        if row["extra_json"]:
            entry.update(json.loads(row["extra_json"]))

        # Decompress points
        points = json.loads(zlib.decompress(row["points_blob"]))
        entry["points"] = points
        return entry

    async def get_scatter(
        self, source: str, atlas: str, level: str, sigtype: str, target: str
    ) -> dict | None:
        return await self._run(
            self._get_scatter_sync, source, atlas, level, sigtype, target
        )

    # ------------------------------------------------------------------ #
    #  Bulk RNA-seq specific queries                                       #
    # ------------------------------------------------------------------ #

    def _list_datasets_sync(self, source: str) -> list[str]:
        """List distinct atlas values for a given source (e.g. 'bulk_rnaseq')."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT atlas FROM scatter_targets WHERE source=? ORDER BY atlas",
            (source,),
        ).fetchall()
        return [r["atlas"] for r in rows]

    async def list_datasets(self, source: str) -> list[str]:
        return await self._run(self._list_datasets_sync, source)

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
