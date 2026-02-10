"""API v2 unified atlas endpoints.

Consolidates legacy per-atlas endpoints (CIMA, Inflammation, scAtlas) into
a single parameterized route pattern: /atlases/{atlas}/...

Also adds interactive single-cell query endpoints (new in v2).
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.base import BaseService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/atlases", tags=["Atlases v2"])


def _get_service() -> BaseService:
    return BaseService()


# ------------------------------------------------------------------ #
#  Activity                                                            #
# ------------------------------------------------------------------ #


@router.get("/{atlas}/activity")
async def get_activity(
    atlas: str,
    sig_type: str = Query("cytosig", description="Signature type"),
    cell_type: Optional[str] = Query(None, description="Filter by cell type"),
    limit: int = Query(1000, ge=1, le=50000),
    offset: int = Query(0, ge=0),
):
    """Get activity data for an atlas."""
    svc = _get_service()
    try:
        data = await svc.repository.get_activity(
            atlas, sig_type, cell_type=cell_type
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    total = len(data)
    return {
        "atlas": atlas,
        "sig_type": sig_type,
        "total": total,
        "data": data[offset : offset + limit],
    }


@router.get("/{atlas}/activity/signatures")
async def list_signatures(atlas: str, sig_type: str = Query("cytosig")):
    """List available signatures for an atlas."""
    svc = _get_service()
    try:
        data = await svc.repository.get_activity(atlas, sig_type)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    signatures = sorted({d.get("signature", d.get("protein", "")) for d in data})
    return {"atlas": atlas, "sig_type": sig_type, "signatures": signatures}


@router.get("/{atlas}/activity/cell-types")
async def list_cell_types(atlas: str, sig_type: str = Query("cytosig")):
    """List cell types with activity data."""
    svc = _get_service()
    try:
        data = await svc.repository.get_activity(atlas, sig_type)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    cell_types = sorted({d.get("cell_type", "") for d in data})
    return {"atlas": atlas, "cell_types": cell_types}


# ------------------------------------------------------------------ #
#  Correlations                                                        #
# ------------------------------------------------------------------ #


@router.get("/{atlas}/correlations")
async def get_correlations(
    atlas: str,
    variable: str = Query("age", description="Variable (age, bmi, metabolite)"),
    sig_type: Optional[str] = Query(None),
    cell_type: Optional[str] = Query(None),
    limit: int = Query(1000, ge=1, le=50000),
    offset: int = Query(0, ge=0),
):
    """Get correlation data for an atlas."""
    svc = _get_service()
    filters = {}
    if sig_type:
        filters["sig_type"] = sig_type
    if cell_type:
        filters["cell_type"] = cell_type
    try:
        data = await svc.repository.get_correlations(atlas, variable, **filters)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    total = len(data)
    return {
        "atlas": atlas,
        "variable": variable,
        "total": total,
        "data": data[offset : offset + limit],
    }


# ------------------------------------------------------------------ #
#  Differential                                                        #
# ------------------------------------------------------------------ #


@router.get("/{atlas}/differential")
async def get_differential(
    atlas: str,
    comparison: str = Query("disease", description="Comparison type"),
    sig_type: Optional[str] = Query(None),
    limit: int = Query(1000, ge=1, le=50000),
    offset: int = Query(0, ge=0),
):
    """Get differential activity data."""
    svc = _get_service()
    filters = {}
    if sig_type:
        filters["sig_type"] = sig_type
    try:
        data = await svc.repository.get_differential(atlas, comparison, **filters)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    total = len(data)
    return {
        "atlas": atlas,
        "comparison": comparison,
        "total": total,
        "data": data[offset : offset + limit],
    }


# ------------------------------------------------------------------ #
#  Generic data endpoint                                               #
# ------------------------------------------------------------------ #


@router.get("/{atlas}/data/{data_type}")
async def get_data(
    atlas: str,
    data_type: str,
    limit: int = Query(1000, ge=1, le=50000),
    offset: int = Query(0, ge=0),
):
    """Get any data type for an atlas (generic endpoint)."""
    svc = _get_service()
    try:
        data = await svc.repository.get_data(f"{atlas}_{data_type}")
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    total = len(data)
    return {
        "atlas": atlas,
        "data_type": data_type,
        "total": total,
        "data": data[offset : offset + limit],
    }


# ------------------------------------------------------------------ #
#  Interactive single-cell queries (NEW in v2)                         #
# ------------------------------------------------------------------ #


@router.get("/{atlas}/cells")
async def query_cells(
    atlas: str,
    cell_type: Optional[str] = Query(None),
    gene: Optional[str] = Query(None, description="Gene name to filter by expression"),
    min_expression: Optional[float] = Query(None, description="Minimum expression threshold"),
    limit: int = Query(10000, ge=1, le=100000),
):
    """Query individual cells from an atlas (requires DuckDB backend).

    This endpoint enables interactive single-cell exploration — filtering
    cells by type, gene expression, and metadata.
    """
    svc = _get_service()
    repo = svc.repository

    # Check if DuckDB backend supports cell queries
    if not hasattr(repo, "query_cells"):
        raise HTTPException(
            status_code=501,
            detail="Cell-level queries require DuckDB backend. "
            "Run: python scripts/convert_data_to_duckdb.py --all",
        )

    try:
        data = await repo.query_cells(
            atlas=atlas,
            cell_type=cell_type,
            gene=gene,
            min_expression=min_expression,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"atlas": atlas, "total": len(data), "cells": data}


@router.get("/{atlas}/cells/expression")
async def get_cell_expression(
    atlas: str,
    gene: str = Query(..., description="Gene name"),
    cell_type: Optional[str] = Query(None),
    limit: int = Query(10000, ge=1, le=100000),
):
    """Get gene expression values across individual cells."""
    svc = _get_service()
    repo = svc.repository

    if not hasattr(repo, "query_cells"):
        raise HTTPException(
            status_code=501,
            detail="Cell-level queries require DuckDB backend.",
        )

    try:
        data = await repo.query_cells(
            atlas=atlas,
            gene=gene,
            cell_type=cell_type,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"atlas": atlas, "gene": gene, "total": len(data), "cells": data}


@router.get("/{atlas}/cells/metadata")
async def get_cell_metadata(
    atlas: str,
    cell_type: Optional[str] = Query(None),
    limit: int = Query(10000, ge=1, le=100000),
):
    """Get cell metadata (cell type, sample, etc.) without expression data."""
    svc = _get_service()
    repo = svc.repository

    if not hasattr(repo, "query_cells"):
        raise HTTPException(
            status_code=501,
            detail="Cell-level queries require DuckDB backend.",
        )

    try:
        data = await repo.query_cells(
            atlas=atlas,
            cell_type=cell_type,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Strip expression values, keep only metadata
    metadata = [{k: v for k, v in d.items() if k != "expression"} for d in data]
    return {"atlas": atlas, "total": len(metadata), "cells": metadata}
